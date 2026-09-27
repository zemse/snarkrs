//! The one way results leave the GPU, written once so that it is correct on native and in a
//! browser.
//!
//! # Why there is exactly one of these and why it is async
//!
//! `Device::poll` is documented as a no-op on WebGPU ("Devices are automatically polled",
//! `wgpu-30.0.1/src/backend/webgpu.rs:2698`) and `PollType::Wait` has no effect there either.
//! The only thing that ever fires a `map_async` callback in a browser is a yield to the JS
//! event loop, so any code that blocks a thread waiting for the GPU deadlocks the tab. That
//! rules out a synchronous readback, and `pollster::block_on` compiles for wasm32 and then
//! hangs, which is worse than not compiling.
//!
//! # What the design said, what the API actually does, and the correction
//!
//! Design §8 U4 says to build this on `CommandEncoder::map_buffer_on_submit` "plus an async
//! channel", and adds that `device.poll` "cannot be the mechanism". The first half is right
//! and the API exists (wgpu 30.0.1, `src/api/command_buffer_actions.rs:103`, from
//! [wgpu#8125](https://github.com/gfx-rs/wgpu/pull/8125)). The second half is only right for
//! the web.
//!
//! Read what `map_buffer_on_submit` does: it pushes a `DeferredBufferMapping` onto the
//! encoder, and `Queue::submit` drains those and calls plain `Buffer::map_async` on each
//! (`src/api/command_buffer_actions.rs:31-42`). It is `map_async`, scheduled for you at the
//! right moment. Its own doc comment then says, in full: "For the callback to run, either
//! `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)` must be called elsewhere
//! in the runtime." On native, nothing after the submit drives the queue on its own, so the
//! `poll` is still required and this function would hang without it.
//!
//! So `map_buffer_on_submit` is worth using, but for the reason the wgpu PR gives rather than
//! the one the design gives: it removes the submit-then-map ordering hazard, where a
//! `map_async` issued before the submit is a validation error and one issued after has to be
//! sequenced by hand. The `poll` stays, unconditionally, with no `cfg`: it is what makes the
//! callback fire on native and it is a documented no-op on the web, which is exactly the
//! shape "one function correct on both targets" needs.
//!
//! This was tested rather than reasoned about. Deleting the `poll` below and running
//! `tests/device.rs::one_uniform_ring_feeds_three_dispatches_from_three_dynamic_offsets`
//! hangs the test past 60 s instead of finishing in 40 ms. Anyone who reads the design's
//! "device.poll cannot be the mechanism" and removes it will reproduce that.
//!
//! # Cost, and the size ceiling that keeps it irrelevant
//!
//! One `mapAsync` round trip is about 0.3 ms. Design §3 caps the whole per-proof readback at
//! 64 KiB; the bound the code actually gives is `n_windows + ones_groups` points per MSM, at
//! most 85 + 64 over four G1 jobs and one G2, which is 112 KiB. Either way it is paid once
//! per proof and nothing is chunked through it.
//!
//! That ceiling is load-bearing on wasm. `get_mapped_range` there copies the entire mapped
//! `ArrayBuffer` into linear memory, measured at
//! **33.8 ms for 64 MiB** against 2.6 ms for the GPU copy that produced it. At the hundred
//! KiB or so this reads, that is several hundred times clear. If a readback ever grows past
//! a megabyte, this function is the wrong tool and `as_uint8array()` is the right one.

use std::sync::atomic::{AtomicU32, Ordering};

use g16_core::ProveError;

use crate::device::{bad, WgpuBackend};
use crate::gather::storage_entry;

/// How [`Seal::verify`]'s error begins, so [`is_aborted`] can tell it from the other
/// [`ProveError::Device`] this backend raises, a lost device, which a retry in the same
/// process cannot bring back.
const ABORTED: &str = "the GPU did not run this submission to its end";

/// Whether `e` is a submission the GPU cut short, which is the one device fault worth
/// running again unchanged. See [`Seal`].
pub fn is_aborted(e: &ProveError) -> bool {
    matches!(e, ProveError::Device { backend: "wgpu", reason } if reason.starts_with(ABORTED))
}

/// Which of the next 32 submissions to refuse as if the GPU had cut them short: bit `i`
/// set means the `i`th [`Seal::close`] from now, on any seal, overwrites its token with 0
/// before the submit, which is exactly what a pass the GPU abandoned leaves behind. Every
/// close shifts it down one bit, so 0 is the normal state. A test knob for the retry paths
/// in `crate::stages` and `crate::batch`, which have to give the same answer on the attempt
/// after a refusal as they would have on the first; the GPU only produces a refusal under
/// another process's load, and never on demand.
pub static REFUSE_NEXT: AtomicU32 = AtomicU32::new(0);

/// A token the last dispatch of a submission writes and a blit copies out, so the host can
/// tell a submission the GPU ran to its end from one it cut short.
///
/// # Why a submission has to prove it finished
///
/// macOS aborts a Metal command buffer that keeps the GPU from the compositor for too long
/// (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`; `g16-metal`'s `cb.rs` and
/// `fft.rs` carry the measurements), and wgpu 30 does not say so: `wgpu-hal`'s Metal fence
/// advances on `MTLCommandBufferStatus::Error` exactly as on `Completed`
/// (`wgpu-hal-30.0.1/src/metal/mod.rs`, `Fence::get_latest`), nothing reads the command
/// buffer's `error`, and neither `on_uncaptured_error` nor the device-lost callback fires.
/// `poll` returns, the map callback fires, and the readback holds whatever the buffers held
/// before: the previous proof's window sums, or a test's sentinel. Measured by reading the
/// command buffers' own status through the Objective-C runtime while three other processes
/// proved on this M2 Max: 24 of 30 `js_16x16_d32` proofs had their stage 5 to 9 buffer
/// aborted, and `g16 prove --backend wgpu` failed its self-verify on 29 of 40 railgun-13x01
/// proofs. That is BUG-24, and every symptom in it: a G2 MSM at infinity, `pi_b` off, a
/// 430 ms MSM stage where 590 is typical.
///
/// # What an abort skips, measured, and where the token therefore has to be
///
/// The abort ends the compute pass that was running and nothing else in the command
/// buffer: over 40 G2 MSMs under that load, 27 aborted, and in every one of them a dispatch
/// at the end of the MSM's own pass had not run while a blit after the pass and a second
/// compute pass after that both had. So a token copied by a blit passes on an aborted
/// submission (the first draft of this did exactly that: 3 refusals in a run of 40 `g16
/// prove` calls that still produced 22 wrong proofs), and the token is instead written by
/// **the last dispatch of the last compute pass**, which is
/// present only if every dispatch before it ran, and then copied out to a mappable word.
/// It is a fresh epoch per submission rather than a constant, so a word still holding the
/// previous submission's token cannot pass for this one. One 4-byte `write_buffer`, one
/// thread and one 4-byte copy, with no Metal in it: the same check catches a lost WebGPU
/// device, and the retry it enables is in `crate::backend`.
pub struct Seal {
    /// The epoch, `write_buffer`'d before the submit.
    src: wgpu::Buffer,
    /// Where the dispatch copies it.
    dst: wgpu::Buffer,
    /// Where the blit after the pass copies that, for the host to map.
    map: wgpu::Buffer,
    bind: wgpu::BindGroup,
    epoch: AtomicU32,
    /// The epoch [`Self::dispatch`] wrote and [`Self::close`] has not yet taken, or 0.
    armed: AtomicU32,
}

/// One submission's token in flight: what [`Seal::dispatch`] wrote, for [`Seal::verify`] to
/// check once the submission has been waited on.
#[must_use = "a seal that is never verified checks nothing"]
pub struct Sealed {
    epoch: u32,
    rx: flume::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

/// The one kernel behind every [`Seal`] on a device: one thread copying one word. Built by
/// `WgpuBackend::with_profile` rather than per seal, because the tests make a [`Readback`]
/// per read and a pipeline per seal would be a compile per read.
pub struct SealKernel {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

const SEAL_WGSL: &str = "
@group(0) @binding(0) var<storage, read> SRC: array<u32>;
@group(0) @binding(1) var<storage, read_write> DST: array<u32>;

@compute @workgroup_size(1)
fn seal() {
    DST[0] = SRC[0];
}
";

impl SealKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("g16 seal"),
            source: wgpu::ShaderSource::Wgsl(SEAL_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g16 seal"),
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g16 seal"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("g16 seal"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("seal"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { bgl, pipeline }
    }
}

impl Seal {
    pub fn new(backend: &WgpuBackend, label: &str) -> Result<Self, ProveError> {
        let mk = |usage: wgpu::BufferUsages| {
            backend.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 4,
                usage,
                mapped_at_creation: false,
            })
        };
        let src = mk(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let dst = mk(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let map = mk(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let bind = backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &backend.seal_kernel().bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            });
        Ok(Self {
            src,
            dst,
            map,
            bind,
            epoch: AtomicU32::new(0),
            armed: AtomicU32::new(0),
        })
    }

    /// Writes this submission's token and dispatches the thread that lands it. **Must be
    /// the last dispatch of the submission's last compute pass**: a dispatch after it could
    /// be abandoned with the token already in place. [`Self::close`] follows, after the
    /// pass ends.
    pub fn dispatch(&self, backend: &WgpuBackend, pass: &mut wgpu::ComputePass<'_>) {
        // Never 0, which is what a fresh buffer holds.
        let mut epoch = self.epoch.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        if epoch == 0 {
            epoch = self.epoch.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        }
        backend
            .queue()
            .write_buffer(&self.src, 0, &epoch.to_le_bytes());
        pass.set_pipeline(&backend.seal_kernel().pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        self.armed.store(epoch, Ordering::Relaxed);
    }

    /// Encodes the copy out to the mappable word and registers its map. Last in `enc`,
    /// before `finish`. A submission whose pass did not call [`Self::dispatch`] gets a pass
    /// of its own here holding only the token, which is what a blit-only readback needs.
    pub fn close(&self, backend: &WgpuBackend, enc: &mut wgpu::CommandEncoder) -> Sealed {
        let mut epoch = self.armed.swap(0, Ordering::Relaxed);
        if epoch == 0 {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g16 seal"),
                timestamp_writes: None,
            });
            self.dispatch(backend, &mut pass);
            drop(pass);
            epoch = self.armed.swap(0, Ordering::Relaxed);
        }
        // The knob: a later `write_buffer` of the same word lands after the one `dispatch`
        // made, so the thread copies 0 and `verify` refuses it, as `tests/device.rs` does by
        // hand.
        let refuse = REFUSE_NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |m| Some(m >> 1))
            .unwrap_or(0);
        if refuse & 1 != 0 {
            backend
                .queue()
                .write_buffer(&self.src, 0, &0u32.to_le_bytes());
        }
        enc.copy_buffer_to_buffer(&self.dst, 0, &self.map, 0, 4);
        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&self.map, wgpu::MapMode::Read, 0..4, move |r| {
            let _ = tx.send(r);
        });
        Sealed { epoch, rx }
    }

    /// Checks the word, after the submission has been polled to completion the way
    /// [`Readback::submit_and_read`] and `WgpuBackend::wait_for_submitted_work` do.
    pub async fn verify(&self, sealed: Sealed) -> Result<(), ProveError> {
        sealed
            .rx
            .recv_async()
            .await
            .map_err(|_| bad("the seal's map callback was dropped before it fired"))?
            .map_err(|e| bad(format!("mapping the seal failed: {e}")))?;
        let got = {
            let view = self
                .map
                .slice(0..4)
                .get_mapped_range()
                .map_err(|e| bad(format!("the seal mapped but did not read: {e}")))?;
            u32::from_le_bytes([view[0], view[1], view[2], view[3]])
        };
        self.map.unmap();
        if got != sealed.epoch {
            // `Device` and not `Backend`, so `g16-cli`'s fallback treats it as the
            // transient it is once the retry in `crate::backend` has given up.
            return Err(ProveError::Device {
                backend: "wgpu",
                reason: format!(
                    "{ABORTED}: the completion token is {got:#x}, this submission's is \
                     {:#x}. On macOS that is a compute pass aborted for impacting \
                     interactivity while another process held the GPU, which wgpu does \
                     not report",
                    sealed.epoch
                ),
            });
        }
        Ok(())
    }

    /// The buffer the token is copied from. For `tests/device.rs`, which overwrites it
    /// between [`Self::dispatch`] and the submit to stand in for a pass the GPU cut short.
    pub fn source(&self) -> &wgpu::Buffer {
        &self.src
    }
}

/// A staging buffer plus the copy-and-map dance around it.
///
/// Allocated once and reused: the buffer is `MAP_READ | COPY_DST`, which is the one usage
/// combination a mappable buffer is allowed, and it can never also be a storage buffer. That
/// is a WebGPU rule and not a wgpu limitation, so the copy from the storage buffer into this
/// one is unavoidable rather than a missed optimisation.
pub struct Readback {
    staging: wgpu::Buffer,
    bytes: u64,
    /// Refuses the data of a submission the GPU abandoned. See [`Seal`].
    seal: Seal,
}

impl Readback {
    /// Allocates a staging buffer of exactly `bytes`.
    ///
    /// `bytes` must be a multiple of 4, which WebGPU requires of every buffer-to-buffer copy
    /// size and of every mapped range length.
    pub fn new(backend: &WgpuBackend, label: &str, bytes: u64) -> Result<Self, ProveError> {
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(bad(format!(
                "readback {label:?} of {bytes} bytes: a mapped range must be a nonzero \
                 multiple of 4"
            )));
        }
        let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            staging,
            bytes,
            seal: Seal::new(backend, label)?,
        })
    }

    /// The seal every [`Self::submit_and_read`] closes with. A caller whose submission has
    /// a compute pass calls [`Seal::dispatch`] on it as that pass's last dispatch; one
    /// without gets a pass of its own from [`Seal::close`].
    pub fn seal(&self) -> &Seal {
        &self.seal
    }

    /// Queues the device-to-staging copy into an encoder the caller is still building.
    ///
    /// Separate from [`Self::submit_and_read`] so the copy lands in the same command encoder
    /// as the dispatches that produced the data. A readback that opened its own encoder
    /// would be a submission of its own, uncounted and unsealed.
    pub fn copy_from(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_offset: u64,
        bytes: u64,
    ) -> Result<(), ProveError> {
        self.copy_from_at(enc, src, src_offset, 0, bytes)
    }

    /// Same, into a chosen offset of the staging buffer.
    ///
    /// [`crate::batch::MsmBatch`] is the reason this exists: a proof's five MSMs write five
    /// separate `results` buffers and the design budgets **one** readback for all of them, so
    /// each one is copied into its own window of a single staging allocation inside the same
    /// encoder as the dispatches. Concatenating them on the device costs five
    /// `copy_buffer_to_buffer` calls and saves a `mapAsync` round trip at about 0.3 ms for
    /// every MSM that finishes in the same submission as another.
    ///
    /// WebGPU requires both offsets and the size to be multiples of 4. That is checked here
    /// rather than left to the validation layer, whose message names a byte count and not the
    /// job whose window was misaligned.
    pub fn copy_from_at(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_offset: u64,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), ProveError> {
        if dst_offset.saturating_add(bytes) > self.bytes {
            return Err(bad(format!(
                "readback buffer holds {} bytes, asked to copy {bytes} to offset {dst_offset}",
                self.bytes
            )));
        }
        if !bytes.is_multiple_of(4)
            || !dst_offset.is_multiple_of(4)
            || !src_offset.is_multiple_of(4)
        {
            return Err(bad(format!(
                "a buffer to buffer copy needs every one of src {src_offset}, dst \
                 {dst_offset} and size {bytes} to be a multiple of 4"
            )));
        }
        enc.copy_buffer_to_buffer(src, src_offset, &self.staging, dst_offset, bytes);
        Ok(())
    }

    /// Finishes `enc`, submits it, waits for the GPU, and returns the first `bytes` of the
    /// staging buffer.
    ///
    /// Takes the encoder by value because `map_buffer_on_submit` has to be registered before
    /// `finish()`, which is the whole point of using it over a bare `map_async`.
    ///
    /// Returns an owned `Vec` rather than the mapped view. A `BufferView` borrows the buffer
    /// and would have to stay alive across the caller's parsing, which forbids reusing this
    /// `Readback` and, on wasm, holds a copy of the `ArrayBuffer` anyway. At the 64 KiB
    /// ceiling in the module docs the extra copy is not measurable.
    pub async fn submit_and_read(
        &self,
        backend: &WgpuBackend,
        enc: wgpu::CommandEncoder,
        bytes: u64,
    ) -> Result<Vec<u8>, ProveError> {
        if bytes > self.bytes {
            return Err(bad(format!(
                "readback buffer holds {} bytes, asked to read {bytes}",
                self.bytes
            )));
        }
        // `flume` and not `std::sync::mpsc`: the receiver has to be awaited, not blocked on,
        // and `mpsc::Receiver` has no async form. This is wgpu's own choice in
        // `examples/features/src/repeated_compute`.
        let mut enc = enc;
        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&self.staging, wgpu::MapMode::Read, 0..bytes, move |r| {
            let _ = tx.send(r);
        });
        // Last, after every copy into the staging buffer.
        let sealed = self.seal.close(backend, &mut enc);
        // Counted, so `tests/stages.rs` can hold `compute_h` to design §3's one submit.
        backend.submit([enc.finish()]);

        // Native: this is what runs the callback, and without it the await below never
        // resolves. Web: a documented no-op, and the browser event loop runs the callback
        // while the await is suspended. See the module docs.
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| backend.fault(format!("waiting for the GPU failed: {e}")))?;

        rx.recv_async()
            .await
            .map_err(|_| backend.fault("the map callback was dropped before it fired"))?
            .map_err(|e| backend.fault(format!("mapping the readback buffer failed: {e}")))?;

        let slice = self.staging.slice(0..bytes);
        let view = slice
            .get_mapped_range()
            .map_err(|e| bad(format!("the readback buffer mapped but did not read: {e}")))?;
        let out = view.to_vec();
        // Both in this order and both required: the view borrows the mapping, and a buffer
        // left mapped cannot be used by any later command.
        drop(view);
        self.staging.unmap();

        if let Some(e) = backend.take_error() {
            return Err(backend.fault(format!("device error during readback: {e}")));
        }
        // After the unmap, so a refused readback leaves the staging buffer usable.
        self.seal.verify(sealed).await?;
        Ok(out)
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.bytes
    }
}
