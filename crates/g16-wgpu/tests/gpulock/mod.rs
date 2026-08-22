//! A GPU lock that actually spans processes.
//!
//! `cargo test` builds each `tests/*.rs` into its own binary and runs those binaries **in
//! parallel with each other**, so the `static Mutex` that `stages.rs` and `msm_g2.rs` use
//! serialises their own tests and nothing else. It cannot stop `gather.rs` timing a kernel in
//! one process while `msm_g2.rs` saturates the same GPU in another, which is exactly what
//! started happening the moment the G2 suite landed: `the_design_picked_the_workgroup_size_that_wins`
//! measured workgroup 128 as 65% off the best size and failed, then passed on its own at
//! `--test-threads=1`. The kernel was never the problem and neither was the shipped constant.
//!
//! A per-process mutex cannot fix a cross-process race, so this is `flock` on one file under
//! `target/`. Every timing test takes it, so at most one process measures the GPU at a time
//! while correctness tests, which do not care about contention, still run in parallel.
//!
//! This matters beyond tidiness. A suite that is only green at `--test-threads=1` is a suite
//! nobody runs, and this crate has already been red-by-default once for that exact reason.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::PathBuf;

pub struct GpuLock(File);

impl Drop for GpuLock {
    fn drop(&mut self) {
        // Best effort: the lock is released by closing the fd anyway, so a failure here can
        // only mean the file was already gone, which is not worth failing a test over.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// Blocks until this process owns the GPU for timing purposes.
///
/// Held for the life of the returned value. Take it in any test that asserts on a duration,
/// and do not take it in a test that only asserts on values: correctness tests are the bulk
/// of the suite and serialising them would turn a 2 minute run into a much longer one for no
/// benefit.
pub fn exclusive_gpu() -> GpuLock {
    // Under `target/` rather than a temp dir so it is scoped to this checkout, and so two
    // worktrees of this repo measuring at once do not block each other for no reason: they
    // are different GPUs' worth of work only if the machine has one GPU, which it does, but
    // a cross-checkout lock would be a surprise nobody asked for. Contention between
    // worktrees is rare; contention between this crate's own ten test binaries is constant.
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join("g16-wgpu-gpu-timing.lock");
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .expect("open the GPU timing lock");
    // LOCK_EX blocks, which is what we want: a timing test should wait its turn rather than
    // measure through someone else's dispatches or skip and report nothing.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    assert_eq!(rc, 0, "flock on the GPU timing lock failed");
    GpuLock(file)
}
