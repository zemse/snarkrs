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
    cb.wait_until_completed();
    let status = cb.status();
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
