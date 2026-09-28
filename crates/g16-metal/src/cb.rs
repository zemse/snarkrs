//! Committing a command buffer and finding out whether it worked.
//!
//! `wait_until_completed` blocks until the GPU is done, and returns exactly the same way
//! whether the work succeeded or the command buffer faulted. Every commit site in this
//! crate used to call it and then read the output buffers regardless. Those buffers come
//! from a pool ([`crate::backend`]'s scratch reuse), so on a fault the next statement reads
//! whatever the *previous* proof left there: not garbage, not a crash, but a plausible set
//! of window sums belonging to a different witness. `msm_batch` then returned `Ok`.
//!
//! That is the failure icicle-snark shipped as issue #18, where large circuits produced
//! mathematically invalid proofs. It is invisible on this side of the boundary and only
//! surfaces when somebody else's verifier says no, which is why `snarkrs groth16 prove` now
//! self-verifies as well: the two guards are for the same fault, one at the source and one
//! at the exit.
//!
//! The status is one guard; [`Seal`] is the second. A proving command buffer ends with a
//! one-thread dispatch that writes the submission's epoch to a slot, as the last dispatch
//! of its last compute encoder, and the host refuses the buffer's output unless the epoch
//! is there. That catches a buffer whose work was cut short but whose status still says
//! `Completed`, which wgpu's Metal path met on this machine (`g16-wgpu`'s `readback::Seal`,
//! where an abort was measured to end the current compute encoder only, so a token written
//! by a later encoder or a blit passes). `examples/cb_seal_probe.rs` measures the same
//! question against this crate's own submissions; its numbers are in the commit that added
//! the seal.
//!
//! `metal-rs` 0.29 binds `status()` but not `error()`, so the NSError has to be read with a
//! raw `msg_send!`. That is safe here because the crate marks `CommandBufferRef` as
//! `objc::Message`, which is the same mechanism its own `status()` uses. The `objc` macros
//! come from `metal::objc`, which `metal` re-exports as `pub extern crate objc`: taking them
//! from there rather than from a separate dependency guarantees the two cannot resolve to
//! different versions of the runtime, which would make these selector calls unsound.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use g16_core::ProveError;
use metal::objc::runtime::Object;
use metal::objc::{class, msg_send, sel, sel_impl};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CommandQueueRef, ComputeCommandEncoderRef,
    ComputePipelineState, Device, Library, MTLCommandBufferStatus, MTLSize,
};

#[link(name = "Metal", kind = "framework")]
extern "C" {
    /// The NSError user-info key under which Metal 3 files one
    /// `MTLCommandBufferEncoderInfo` per encoder of a failed buffer, when the buffer was
    /// made with [`command_buffer`].
    static MTLCommandBufferEncoderInfoErrorKey: *mut Object;
}

/// A Rust `String` from an NSString, empty for nil.
///
/// # Safety
///
/// `s` is nil or a live NSString.
unsafe fn nsstring(s: *mut Object) -> String {
    if s.is_null() {
        return String::new();
    }
    let utf8: *const c_char = msg_send![s, UTF8String];
    if utf8.is_null() {
        return String::new();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}

/// A command buffer whose NSError, should it fail, names the execution state of every
/// encoder in it (`MTLCommandBufferErrorOptionEncoderExecutionStatus`). Every proving
/// submission is made here rather than with `new_command_buffer`, so that a fault says
/// which encoder faulted and which ones never ran. Priced at nothing measurable: see the
/// commit that added it.
pub(crate) fn command_buffer(queue: &CommandQueueRef) -> &CommandBufferRef {
    // SAFETY: `CommandQueueRef` is `objc::Message`. `MTLCommandBufferDescriptor` is
    // created at +1 and released once the queue has copied it, and
    // `commandBufferWithDescriptor:` returns the same autoreleased `+0` buffer as
    // `commandBuffer`, which is what metal-rs's `new_command_buffer` hands out as a
    // borrowed ref.
    unsafe {
        let desc: *mut Object = msg_send![class!(MTLCommandBufferDescriptor), new];
        // MTLCommandBufferErrorOptionEncoderExecutionStatus.
        let () = msg_send![desc, setErrorOptions: 1u64];
        let cb: &CommandBufferRef = msg_send![queue, commandBufferWithDescriptor: desc];
        let () = msg_send![desc, release];
        cb
    }
}

/// Every dispatch on the proving path goes through this pair rather than the encoder's
/// own, so a test can stop an attempt at any one of them the way a kill does (see
/// [`inject::arm_cut`]). Outside tests they forward and nothing else.
pub(crate) fn dispatch_threads(enc: &ComputeCommandEncoderRef, grid: MTLSize, tg: MTLSize) {
    #[cfg(test)]
    let grid = match inject::cut_dispatch(grid.width) {
        Some(width) => MTLSize::new(width, grid.height, grid.depth),
        None => return,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn dispatch_thread_groups(enc: &ComputeCommandEncoderRef, groups: MTLSize, tg: MTLSize) {
    #[cfg(test)]
    let groups = match inject::cut_dispatch(groups.width) {
        Some(width) => MTLSize::new(width, groups.height, groups.depth),
        None => return,
    };
    enc.dispatch_thread_groups(groups, tg);
}

/// The NSError text hanging off a failed command buffer, empty when there is none, with
/// the per-encoder execution states after it when the buffer was made with
/// [`command_buffer`] and Metal filed them.
fn error_text(cb: &CommandBufferRef) -> String {
    // SAFETY: `CommandBufferRef` is `objc::Message` (metal-rs asserts this for every
    // foreign object type it defines), `error` is a documented MTLCommandBuffer property
    // returning an autoreleased NSError or nil, `userInfo` an NSDictionary, the encoder
    // infos an NSArray of MTLCommandBufferEncoderInfo, and every result is null-checked
    // before it is followed. Nothing is retained or released, so there is no ownership
    // to get wrong.
    unsafe {
        let err: *mut Object = msg_send![cb, error];
        if err.is_null() {
            return String::new();
        }
        let desc: *mut Object = msg_send![err, localizedDescription];
        let mut text = format!(": {}", nsstring(desc));
        let info: *mut Object = msg_send![err, userInfo];
        if info.is_null() {
            return text;
        }
        let infos: *mut Object = msg_send![info, objectForKey: MTLCommandBufferEncoderInfoErrorKey];
        if infos.is_null() {
            return text;
        }
        let n: usize = msg_send![infos, count];
        for i in 0..n {
            let e: *mut Object = msg_send![infos, objectAtIndex: i];
            let label: *mut Object = msg_send![e, label];
            let state: i64 = msg_send![e, errorState];
            // MTLCommandEncoderErrorState.
            let state = match state {
                0 => "unknown",
                1 => "completed",
                2 => "affected",
                3 => "pending",
                4 => "faulted",
                _ => "?",
            };
            text.push_str(if i == 0 { " (encoders: " } else { ", " });
            text.push_str(&nsstring(label));
            text.push_str(": ");
            text.push_str(state);
        }
        if n > 0 {
            text.push(')');
        }
        text
    }
}

/// Commit already done by the caller: wait, then refuse to continue unless the GPU actually
/// completed the work. `context` names the dispatch, so a fault says which one died.
pub(crate) fn wait_ok(cb: &CommandBufferRef, context: &str) -> Result<(), ProveError> {
    let submitted = std::time::Instant::now();
    cb.wait_until_completed();
    // `G16_METAL_CB_TIMES=1` prints the driver's own GPU-busy window next to the wall
    // time the host waited. The gap between them is submission latency and host work,
    // which no amount of kernel tuning will remove, so the two numbers have to be read
    // together before anything is attributed to the kernels.
    if std::env::var_os("G16_METAL_CB_TIMES").is_some() {
        // SAFETY: same mechanism and same justification as `error_text` above. All four
        // are documented MTLCommandBuffer properties returning a CFTimeInterval, valid
        // once the buffer has completed, which `wait_until_completed` has just ensured.
        // metal-rs 0.29 binds none of them.
        let (gpu, kern) = unsafe {
            let gs: f64 = msg_send![cb, GPUStartTime];
            let ge: f64 = msg_send![cb, GPUEndTime];
            let ks: f64 = msg_send![cb, kernelStartTime];
            let ke: f64 = msg_send![cb, kernelEndTime];
            ((ge - gs) * 1e3, (ke - ks) * 1e3)
        };
        eprintln!(
            "[cb] {context}: wall {:.2} ms  gpu {:.2} ms  kernel {:.2} ms",
            submitted.elapsed().as_secs_f64() * 1e3,
            gpu,
            kern
        );
    }
    let status = cb.status();
    #[cfg(test)]
    if inject::fires() {
        return Err(ProveError::Device {
            backend: "metal",
            reason: format!("{context}: injected fault (status {status:?})"),
        });
    }
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(ProveError::Device {
        backend: "metal",
        reason: format!(
            "{context}: command buffer did not complete (status {status:?}){}",
            error_text(cb)
        ),
    })
}

/// Attempts at one submission before giving up, counting the first. The ceremony FFT's
/// contract: a macOS interactivity kill is a scheduling event, not an arithmetic one.
/// Eight, as `g16-wgpu`'s sealed submissions: under two 2^22 proof loops beside two wgpu
/// ones and a probe, about one compute_h attempt in two was killed, and four attempts gave
/// up 13 of 161 proofs; eight gave up none of 320 in two such runs.
pub(crate) const RETRIES: u32 = 8;

/// Backoff before attempt `attempt` (from 1): 400, 800, then 1600 ms, so eight attempts
/// wait 9.2 s in all.
fn backoff(attempt: u32) -> std::time::Duration {
    std::time::Duration::from_millis(200 << attempt.min(3))
}

/// What the driver said about a refused submission, read from the `kIOGPU...` name in
/// its NSError text. The NSError code does not tell them apart (1 for a kill and for a
/// victim alike), and the text is not an API, so anything unrecognised is [`Self::Other`]
/// and takes the blind retry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Refusal {
    /// `ImpactingInteractivity`: stopped so the display could have the GPU.
    Kill,
    /// `InnocentVictim`: discarded by the recovery from another buffer's hang.
    Victim,
    /// `Hang`: this buffer was running when the GPU hung.
    Hang,
    /// `SubmissionsIgnored`: never run, because its queue had caused GPU errors before.
    Ignored,
    /// Anything else: a missing token, H off the field, an unrecognised status.
    Other,
}

impl Refusal {
    pub(crate) fn of(e: &ProveError) -> Self {
        let ProveError::Device { reason, .. } = e else {
            return Self::Other;
        };
        if reason.contains("ErrorSubmissionsIgnored") {
            Self::Ignored
        } else if reason.contains("ErrorImpactingInteractivity") {
            Self::Kill
        } else if reason.contains("ErrorInnocentVictim") {
            Self::Victim
        } else if reason.contains("ErrorHang") {
            Self::Hang
        } else {
            Self::Other
        }
    }
}

/// A command queue that can be swapped for a fresh one.
///
/// After its buffers hang the GPU twice in a few seconds, macOS stops running a queue's
/// buffers at all: every one comes back `SubmissionsIgnored` at once, and waiting does
/// not end it (388 s and 16,412 refusals in the BUG-28 lane's h_probe, until it was
/// killed). The refusal sticks to the queue, not the process: a `snarkrs groth16 prove`
/// whose stages queue was ignored had its MSM queue's buffers run in the same second, and
/// an h_probe that swapped both refused queues had its next buffers run and every result
/// after it come back. A one-shot CLI gets fresh queues with its next process; a caller
/// holding a backend for its life did not, and every proof after the refusal failed. So
/// [`with_retry`] swaps the queue on that refusal and runs the attempt on the new one.
pub(crate) struct Queue {
    device: Device,
    current: Mutex<CommandQueue>,
}

impl Queue {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            current: Mutex::new(device.new_command_queue()),
        }
    }

    /// The queue to submit on now. A clone is a retain, so a swap does not pull it out
    /// from under an attempt still waiting on it.
    pub(crate) fn get(&self) -> CommandQueue {
        self.current
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Replaces `refused` with a fresh queue, unless another thread already has.
    fn renew(&self, refused: &CommandQueueRef) {
        let mut current = self.current.lock().unwrap_or_else(|p| p.into_inner());
        if std::ptr::eq::<CommandQueueRef>(&**current, refused) {
            *current = self.device.new_command_queue();
        }
    }
}

/// Runs `submit` on `queue` until it succeeds or [`RETRIES`] attempts have failed, and
/// returns the last error.
///
/// `submit` must be re-runnable from whatever a killed attempt left behind: it encodes
/// from inputs it does not write, and it has waited on every command buffer it committed
/// before it returns, so no dispatch of a failed attempt is still writing the scratch the
/// next one encodes against. It must commit only on the queue it is handed.
///
/// A kill, a victim, a hang and anything unrecognised are retried after [`backoff`]:
/// the failure means something else wants the GPU, and a genuine kernel fault fails
/// every attempt instead. A `SubmissionsIgnored` refusal is retried at once on a fresh
/// queue (see [`Queue`]); waiting would not help, since the old queue is never run
/// again.
///
/// Every failed attempt is reported on stderr with the thread that made it. A kill is rare
/// on a quiet machine (none in 100 runs of the concurrency test), so the line costs
/// nothing there, and on a busy one the retry rate is what an operator needs to see. It is
/// also what says whether a proof that later fails to verify went through a retry.
///
/// The proving path needs this as much as the ceremony. Four proofs sharing the device
/// had one of them killed in 10 of 20 runs of
/// `backend::tests::one_metal_circuit_proves_concurrently`, on circuits as small as
/// railgun-13x01, and on the gather, the transforms and the MSM batch alike.
pub(crate) fn with_retry<T>(
    queue: &Queue,
    mut submit: impl FnMut(&CommandQueueRef) -> Result<T, ProveError>,
) -> Result<T, ProveError> {
    let mut err = None;
    let mut wait = false;
    for attempt in 0..RETRIES {
        if wait {
            std::thread::sleep(backoff(attempt));
        }
        let q = queue.get();
        match submit(&q) {
            Ok(v) => return Ok(v),
            Err(e) => {
                #[cfg(test)]
                inject::attempt_failed();
                let refusal = Refusal::of(&e);
                let thread = std::thread::current();
                eprintln!(
                    "metal: attempt {} of {RETRIES} failed on thread {}: {e}{}",
                    attempt + 1,
                    thread.name().unwrap_or("unnamed"),
                    if refusal == Refusal::Ignored {
                        "; retrying on a new command queue"
                    } else {
                        ""
                    }
                );
                wait = refusal != Refusal::Ignored;
                if !wait {
                    queue.renew(&q);
                }
                err = Some(e);
            }
        }
    }
    Err(err.expect("RETRIES is at least 1"))
}

/// Slots in a [`Seal`]'s word buffer. A slot is reused after this many submissions on
/// the same seal, and every submission is waited on inside the call that made it, so
/// the previous user of a slot is long finished; the epoch, not the slot, is what the
/// check compares, so a stale value in a reused slot is still a mismatch.
const SEAL_SLOTS: u32 = 1024;

/// The completion token of a submission: the slot its epoch was written to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Token {
    slot: u32,
    epoch: u32,
}

/// One pipeline and one page of slots per kernel module, shared by every proof on it.
///
/// [`Self::encode`] takes the next epoch and encodes the one-thread dispatch that lands
/// it; [`Self::wait`] waits on the buffer and refuses it unless the slot holds the
/// epoch. The epoch is never 0, which is what a fresh slot holds, and it advances on
/// every submission from every thread, so two proofs in flight cannot share a token.
pub(crate) struct Seal {
    pipeline: ComputePipelineState,
    slots: Buffer,
    epoch: AtomicU32,
}

impl Seal {
    /// From a library that has `seal.metal` in it.
    pub(crate) fn new(device: &Device, library: &Library) -> Result<Self, ProveError> {
        let f = library
            .get_function("g16_seal", None)
            .map_err(|e| ProveError::Backend {
                backend: "metal",
                reason: format!("kernel g16_seal missing: {e}"),
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| ProveError::Backend {
                backend: "metal",
                reason: format!("pipeline g16_seal: {e}"),
            })?;
        let slots = crate::alloc::shared(device, SEAL_SLOTS as usize * 4)?;
        Ok(Self {
            pipeline,
            slots,
            epoch: AtomicU32::new(0),
        })
    }

    /// The next submission's token.
    pub(crate) fn next(&self) -> Token {
        let mut epoch = self.epoch.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        if epoch == 0 {
            epoch = self.epoch.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        }
        Token {
            slot: epoch % SEAL_SLOTS,
            epoch,
        }
    }

    /// Encodes the token write. **Must be the last dispatch of the buffer's last compute
    /// encoder**, behind a memory barrier if the encoder is concurrent: a dispatch after
    /// it could be abandoned with the token already in place, and a concurrent one could
    /// still be running.
    pub(crate) fn encode_token(&self, enc: &ComputeCommandEncoderRef, token: Token) {
        #[cfg(test)]
        if inject::cut_token() {
            return;
        }
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&self.slots), 0);
        let words = [token.slot, token.epoch];
        enc.set_bytes(1, 8, words.as_ptr().cast());
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    /// [`Self::next`] and [`Self::encode_token`] in one.
    pub(crate) fn encode(&self, enc: &ComputeCommandEncoderRef) -> Token {
        let token = self.next();
        self.encode_token(enc, token);
        token
    }

    /// [`wait_ok`], then the token. A buffer that says `Completed` without its token is
    /// the same [`ProveError::Device`] a killed one is, so `with_retry` re-runs it.
    pub(crate) fn wait(
        &self,
        cb: &CommandBufferRef,
        token: Token,
        context: &str,
    ) -> Result<(), ProveError> {
        wait_ok(cb, context)?;
        // SAFETY: `slots` is a shared buffer of `SEAL_SLOTS` u32s and `slot` is below
        // that; the buffer that wrote it has completed.
        let got = unsafe { *(self.slots.contents() as *const u32).add(token.slot as usize) };
        #[cfg(test)]
        let got = if inject::stale_fires() { !got } else { got };
        if got == token.epoch {
            return Ok(());
        }
        Err(ProveError::Device {
            backend: "metal",
            reason: format!(
                "{context}: command buffer completed (status {:?}) but its completion \
                 token is {got:#x}, this submission's is {:#x}: the GPU did not run it to \
                 its end{}",
                cb.status(),
                token.epoch,
                error_text(cb)
            ),
        })
    }
}

/// Fails chosen [`wait_ok`] calls on the current thread after the buffer has completed,
/// so a test can put a retry behind every submission of a proof and check the result is
/// unchanged. After a completed buffer every in-place dispatch has already run once, which
/// is the state a retry that skipped any re-initialisation would trip on.
///
/// A [`Fault::Stale`] leaves the status alone and corrupts the token [`Seal::wait`] reads
/// instead, at the same call index, so the check behind the status is exercised the same
/// way.
///
/// Neither of those runs a buffer partway. A cut ([`arm_cut`]) does: the chosen dispatch
/// is encoded at half its width, nothing after it in its command buffer is encoded, the
/// token included, so the wait refuses the buffer exactly as one the GPU stopped there,
/// and the retry then runs over whatever the half-run attempt left in the scratch. What
/// else is dropped is the [`Rest`]: the rest of that command buffer only, as an
/// interactivity kill of one buffer among several committed back to back, or the rest of
/// the attempt, as a GPU recovery that voids every buffer in flight.
#[cfg(test)]
pub(crate) mod inject {
    use std::cell::Cell;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Fault {
        /// `wait_ok` reports the buffer as not completed.
        Status,
        /// The token `Seal::wait` reads does not match.
        Stale,
    }

    /// What a cut drops after the dispatch it lands on.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Rest {
        /// The rest of that command buffer; later buffers of the attempt run whole.
        Buffer,
        /// Every dispatch and token until `with_retry` sees the failure.
        Attempt,
    }

    thread_local! {
        static CALLS: Cell<u32> = const { Cell::new(0) };
        static FAIL: Cell<Option<(u32, Fault)>> = const { Cell::new(None) };
        /// Every call from `FAIL`'s index on, not just the one.
        static STICKY: Cell<bool> = const { Cell::new(false) };
        static FIRED: Cell<bool> = const { Cell::new(false) };
        /// Dispatches through `cb::dispatch_*` on this thread since the last arm.
        static DISPATCHES: Cell<u32> = const { Cell::new(0) };
        static CUT: Cell<Option<(u32, Rest)>> = const { Cell::new(None) };
        /// A cut has landed and not yet run its course: dispatches and tokens are dropped.
        static CUTTING: Cell<Option<Rest>> = const { Cell::new(None) };
    }

    /// Counts calls from zero again, and fails call `at` with a status fault if given.
    pub(crate) fn arm(at: Option<u32>) {
        arm_with(at.map(|n| (n, Fault::Status)), false);
    }

    /// Counts calls from zero again, and fails call `at` with `fault` if given; with
    /// `sticky`, every call from `at` on, so no retry can succeed.
    pub(crate) fn arm_with(at: Option<(u32, Fault)>, sticky: bool) {
        CALLS.set(0);
        FAIL.set(at);
        STICKY.set(sticky);
        FIRED.set(false);
        DISPATCHES.set(0);
        CUT.set(None);
        CUTTING.set(None);
    }

    /// Counts from zero again and cuts the attempt at dispatch `at`, dropping `rest`.
    pub(crate) fn arm_cut(at: u32, rest: Rest) {
        arm(None);
        CUT.set(Some((at, rest)));
    }

    /// `wait_ok` calls on this thread since [`arm`].
    pub(crate) fn calls() -> u32 {
        CALLS.get()
    }

    /// Dispatches on this thread since [`arm`].
    pub(crate) fn dispatches() -> u32 {
        DISPATCHES.get()
    }

    /// Whether the armed call or cut was reached.
    pub(crate) fn fired() -> bool {
        FIRED.get()
    }

    /// The width a dispatch of `width` threads or groups is encoded at, or `None` for
    /// not at all.
    pub(super) fn cut_dispatch(width: u64) -> Option<u64> {
        let n = DISPATCHES.get();
        DISPATCHES.set(n + 1);
        if CUTTING.get().is_some() {
            return None;
        }
        match CUT.get() {
            Some((at, rest)) if n == at => {
                CUT.set(None);
                CUTTING.set(Some(rest));
                FIRED.set(true);
                Some(width / 2).filter(|w| *w > 0)
            }
            _ => Some(width),
        }
    }

    /// Whether the token of the buffer being sealed is dropped. A `Rest::Buffer` cut
    /// ends here; a `Rest::Attempt` one in [`attempt_failed`].
    pub(super) fn cut_token() -> bool {
        match CUTTING.get() {
            Some(Rest::Buffer) => {
                CUTTING.set(None);
                true
            }
            Some(Rest::Attempt) => true,
            None => false,
        }
    }

    /// `with_retry` saw an attempt fail: whatever a cut was still dropping, the retry
    /// runs whole.
    pub(super) fn attempt_failed() {
        CUTTING.set(None);
    }

    fn hits(n: u32, fault: Fault) -> bool {
        let hit = match FAIL.get() {
            Some((at, f)) if f == fault => n == at || (STICKY.get() && n > at),
            _ => false,
        };
        if hit {
            FIRED.set(true);
        }
        hit
    }

    pub(super) fn fires() -> bool {
        let n = CALLS.get();
        CALLS.set(n + 1);
        hits(n, Fault::Status)
    }

    /// For the `wait_ok` call just made, which `Seal::wait` follows at once.
    pub(super) fn stale_fires() -> bool {
        hits(CALLS.get().wrapping_sub(1), Fault::Stale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The check on the real device, without the injector: a sealed buffer passes, and a
    /// token whose dispatch never ran is refused as a device fault.
    #[test]
    fn a_token_that_was_never_written_is_refused() {
        let device = Device::system_default().expect("no Metal device");
        let library = device
            .new_library_with_source(crate::kernels::SEAL_MSL, &metal::CompileOptions::new())
            .unwrap_or_else(|e| panic!("MSL compile failed:\n{e}"));
        let seal = Seal::new(&device, &library).unwrap();
        let queue = device.new_command_queue();
        metal::objc::rc::autoreleasepool(|| {
            let cb = command_buffer(&queue);
            let enc = cb.new_compute_command_encoder();
            let written = seal.encode(enc);
            enc.end_encoding();
            cb.commit();
            seal.wait(cb, written, "sealed").unwrap();
            // The next epoch, which no dispatch has written.
            let skipped = Token {
                slot: (written.epoch + 1) % SEAL_SLOTS,
                epoch: written.epoch + 1,
            };
            let err = seal.wait(cb, skipped, "unsealed").unwrap_err();
            assert!(err.is_device_fault(), "{err}");
            assert!(err.to_string().contains("completion token"), "{err}");
        });
    }

    fn refused(status: &str) -> ProveError {
        ProveError::Device {
            backend: "metal",
            reason: format!(
                "stages 0-1 (gather): command buffer did not complete (status Error): {status}"
            ),
        }
    }

    /// The four statuses as the driver words them, from the BUG-28 and BUG-29 burst logs.
    #[test]
    fn the_driver_statuses_are_told_apart() {
        for (status, want) in [
            (
                "Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)",
                Refusal::Kill,
            ),
            (
                "Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)",
                Refusal::Victim,
            ),
            (
                "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)",
                Refusal::Hang,
            ),
            (
                "Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)",
                Refusal::Ignored,
            ),
            ("injected fault (status Completed)", Refusal::Other),
        ] {
            assert_eq!(Refusal::of(&refused(status)), want, "{status}");
        }
    }

    /// An ignored submission is retried at once on a new queue; a kill keeps the queue
    /// and backs off.
    #[test]
    fn an_ignored_submission_is_retried_on_a_new_queue() {
        let device = Device::system_default().expect("no Metal device");
        let queue = Queue::new(&device);
        for (status, renews) in [
            (
                "Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)",
                true,
            ),
            (
                "Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)",
                false,
            ),
        ] {
            let mut seen: Vec<*const CommandQueueRef> = Vec::new();
            let start = std::time::Instant::now();
            with_retry(&queue, |q| {
                seen.push(q);
                if seen.len() == 1 {
                    Err(refused(status))
                } else {
                    Ok(())
                }
            })
            .unwrap();
            assert_eq!(seen.len(), 2);
            assert_eq!(seen[0] != seen[1], renews, "{status}");
            assert_eq!(start.elapsed() < backoff(1), renews, "{status}");
        }
    }
}
