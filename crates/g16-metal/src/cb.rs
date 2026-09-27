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
//! surfaces when somebody else's verifier says no, which is why `g16 prove` now
//! self-verifies as well: the two guards are for the same fault, one at the source and one
//! at the exit.
//!
//! `metal-rs` 0.29 binds `status()` but not `error()`, so the NSError has to be read with a
//! raw `msg_send!`. That is safe here because the crate marks `CommandBufferRef` as
//! `objc::Message`, which is the same mechanism its own `status()` uses. The `objc` macros
//! come from `metal::objc`, which `metal` re-exports as `pub extern crate objc`: taking them
//! from there rather than from a separate dependency guarantees the two cannot resolve to
//! different versions of the runtime, which would make these selector calls unsound.

use std::ffi::CStr;
use std::os::raw::c_char;

use g16_core::ProveError;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{CommandBufferRef, MTLCommandBufferStatus};

/// The NSError text hanging off a failed command buffer, empty when there is none.
fn error_text(cb: &CommandBufferRef) -> String {
    // SAFETY: `CommandBufferRef` is `objc::Message` (metal-rs asserts this for every
    // foreign object type it defines), `error` is a documented MTLCommandBuffer property
    // returning an autoreleased NSError or nil, and every result is null-checked before
    // it is followed. Nothing is retained or released, so there is no ownership to get
    // wrong.
    unsafe {
        let err: *mut Object = msg_send![cb, error];
        if err.is_null() {
            return String::new();
        }
        let desc: *mut Object = msg_send![err, localizedDescription];
        if desc.is_null() {
            return String::new();
        }
        let utf8: *const c_char = msg_send![desc, UTF8String];
        if utf8.is_null() {
            return String::new();
        }
        format!(": {}", CStr::from_ptr(utf8).to_string_lossy())
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
        return Err(ProveError::Backend {
            backend: "metal",
            reason: format!("{context}: injected fault (status {status:?})"),
        });
    }
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(ProveError::Backend {
        backend: "metal",
        reason: format!(
            "{context}: command buffer did not complete (status {status:?}){}",
            error_text(cb)
        ),
    })
}

/// Attempts at one submission before giving up, counting the first. The ceremony FFT's
/// contract: a macOS interactivity kill is a scheduling event, not an arithmetic one.
pub(crate) const RETRIES: u32 = 4;

/// Runs `submit` until it succeeds or [`RETRIES`] attempts have failed, backing off
/// between them, and returns the last error.
///
/// `submit` must be re-runnable from whatever a killed attempt left behind: it encodes
/// from inputs it does not write, and it has waited on every command buffer it committed
/// before it returns, so no dispatch of a failed attempt is still writing the scratch the
/// next one encodes against. Blind rather than keyed on the interactivity error, because
/// the error text is not an API; a genuine kernel fault fails every attempt instead.
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
    mut submit: impl FnMut() -> Result<T, ProveError>,
) -> Result<T, ProveError> {
    let mut err = None;
    for attempt in 0..RETRIES {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(200 << attempt));
        }
        match submit() {
            Ok(v) => return Ok(v),
            Err(e) => {
                let thread = std::thread::current();
                eprintln!(
                    "metal: attempt {} of {RETRIES} failed on thread {}: {e}",
                    attempt + 1,
                    thread.name().unwrap_or("unnamed")
                );
                err = Some(e);
            }
        }
    }
    Err(err.expect("RETRIES is at least 1"))
}

/// Fails chosen [`wait_ok`] calls on the current thread after the buffer has completed,
/// so a test can put a retry behind every submission of a proof and check the result is
/// unchanged. After a completed buffer every in-place dispatch has already run once, which
/// is the state a retry that skipped any re-initialisation would trip on.
#[cfg(test)]
pub(crate) mod inject {
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<u32> = const { Cell::new(0) };
        static FAIL: Cell<Option<u32>> = const { Cell::new(None) };
        static FIRED: Cell<bool> = const { Cell::new(false) };
    }

    /// Counts calls from zero again, and fails call `at` if given.
    pub(crate) fn arm(at: Option<u32>) {
        CALLS.set(0);
        FAIL.set(at);
        FIRED.set(false);
    }

    /// `wait_ok` calls on this thread since [`arm`].
    pub(crate) fn calls() -> u32 {
        CALLS.get()
    }

    /// Whether the armed call was reached.
    pub(crate) fn fired() -> bool {
        FIRED.get()
    }

    pub(super) fn fires() -> bool {
        let n = CALLS.get();
        CALLS.set(n + 1);
        let hit = FAIL.get() == Some(n);
        if hit {
            FIRED.set(true);
        }
        hit
    }
}
