//! How an encoded batch reaches the device: whole or split into command buffers,
//! with the memory barrier a concurrent encoder needs between phases.

use objc2_metal::{
    MTLBarrierScope, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDispatchType,
};
use snarkrs_groth16::ProveError;

use super::MetalMsm;
use crate::mtl::{
    self, CommandBuffer, CommandQueueRef, ComputeCommandEncoder, ComputeCommandEncoderRef,
};

/// How a batch reaches the device: one command buffer with a memory barrier between
/// phases, or split, one command buffer per phase with the accumulation cut into
/// pieces of at most [`SPLIT_BUDGET`](super::window::SPLIT_BUDGET) mixed additions.
///
/// macOS kills a command buffer that holds the GPU too long while the display wants it
/// (`ImpactingInteractivity`; `fft.rs` measured 350 ms buffers losing five runs in six
/// and 200 ms ones none). A variable-time batch keeps one buffer, the shape its 0.16 ms
/// submission floor was measured against, although at 2^22 (js_384x384_d32) that buffer
/// is 1.2 s of GPU time: under two 2^22 proof loops beside two wgpu ones it was killed
/// about once in four proofs, and cut into pieces of at most 130 ms about once in two
/// (62 kills in 137 proofs against 40 in 159), since every piece can be killed and a
/// kill re-runs the whole batch; in neither did the MSM spend all eight attempts. A
/// constant-work batch goes out split: its four witness jobs are each the size of H's,
/// 1.5 s on anon-aadhaar in one buffer, and that buffer was killed the first time
/// another batch shared the device. Split, each piece is committed and waited on before
/// the next is encoded, the way the ceremony FFT submits: committing them back to back
/// and waiting at the end kept the queue deep enough that the transforms of a proof
/// sharing the device were killed instead. The wait stands in for the barrier, and the
/// cost is one round trip per piece, about 0.16 ms.
///
/// Every command buffer ends with its completion token behind a barrier (the encoder is
/// concurrent, so without one the token's thread could land before the reduce it vouches
/// for had finished), and every wait checks it.
pub(super) struct Submission<'q> {
    queue: &'q CommandQueueRef,
    seal: &'q crate::cb::Seal,
    split: bool,
    cbs: Vec<(CommandBuffer, crate::cb::Token)>,
    enc: Option<ComputeCommandEncoder>,
}

impl<'q> Submission<'q> {
    pub(super) fn new(msm: &'q MetalMsm, queue: &'q CommandQueueRef, split: bool) -> Self {
        Self {
            queue,
            seal: &msm.seal,
            split,
            cbs: Vec::new(),
            enc: None,
        }
    }

    /// The open encoder, opening a command buffer if none is.
    pub(super) fn enc(&mut self) -> &ComputeCommandEncoderRef {
        if self.enc.is_none() {
            let cb = crate::cb::command_buffer(self.queue);
            let enc = mtl::encoder_with(&cb, MTLDispatchType::Concurrent);
            mtl::set_label(
                &enc,
                if self.split {
                    "MSM batch piece"
                } else {
                    "MSM batch"
                },
            );
            self.enc = Some(enc);
            // The token is taken here rather than at `close` so the pair is pushed
            // together; the dispatch that writes it is encoded last, in `close`.
            self.cbs.push((cb, self.seal.next()));
        }
        self.enc.as_deref().expect("opened above")
    }

    /// Everything encoded after this reads what came before.
    pub(super) fn phase(&mut self) -> Result<(), ProveError> {
        if self.split {
            return self.close();
        }
        barrier(self.enc());
        Ok(())
    }

    /// Seals and commits the open command buffer, if any, and in split mode waits for
    /// it. The next `enc` opens a fresh one.
    pub(super) fn close(&mut self) -> Result<(), ProveError> {
        let Some(enc) = self.enc.take() else {
            return Ok(());
        };
        let (cb, token) = self
            .cbs
            .last()
            .expect("an open encoder has a command buffer");
        barrier(&enc);
        self.seal.encode_token(&enc, *token);
        enc.endEncoding();
        cb.commit();
        if self.split {
            self.seal.wait(cb, *token, "MSM batch piece")?;
        }
        Ok(())
    }

    /// Waits on every command buffer, in submission order, so the first failure
    /// reported is the first that happened.
    pub(super) fn wait(&mut self, context: &str) -> Result<(), ProveError> {
        self.close()?;
        for (cb, token) in &self.cbs {
            self.seal.wait(cb, *token, context)?;
        }
        Ok(())
    }
}

/// Everything encoded after this on a concurrent encoder reads what came before.
pub(super) fn barrier(enc: &ComputeCommandEncoderRef) {
    enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
}
