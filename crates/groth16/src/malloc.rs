//! macOS malloc's large cache, and the one way found to turn it off.
//!
//! A freed large block stays mapped and resident in the default zone's cache, and it is
//! still counted in the footprint and in max RSS, which is the peak `/usr/bin/time -l`
//! reports. When the next allocation asks for a different size the cached block sits
//! beside it rather than being reused: the CPU backend frees two 64 MB domain vectors at
//! the end of stage 4, and one of them is still resident through all five MSMs. On
//! macOS 27 `malloc_zone_pressure_relief` returns 0 without emptying the cache, for the
//! default zone and for every zone, and `setenv("MallocLargeCache", "0")` inside the
//! process changes nothing, because libmalloc reads its environment before `main`.
//!
//! Starting the process with `MallocLargeCache=0` does work: a freed large block is
//! unmapped at once. It also reaches allocations Rust never sees, which a global allocator
//! cannot. `snarkrs-csp-mem`'s witness generator is C++, and the blocks it frees were what
//! made an mmap-backed global allocator peak 100 MB higher there rather than lower.
//!
//! The price is that nothing large is reused, so a process that proves again pays the
//! page faults for its buffers again. A warm anon-aadhaar proof on the CPU backend measured
//! about 3.5% slower that way, which is why only the one-proof processes do this.

/// Re-executes the current binary with `MallocLargeCache=0`, unless it is already set.
///
/// For a process that proves once and exits. Call it before the work starts, while
/// nothing large is allocated and no thread is running. It returns only when it did not
/// exec: off macOS, when the variable is already set to any value (so
/// `MallocLargeCache=1` opts back into the cache), or when the exec failed, in which case
/// the process carries on with the cache as before. The pid does not change, so
/// `/usr/bin/time` and a parent waiting on the child still see one process.
pub fn reexec_without_large_cache() {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::process::CommandExt;

        if std::env::var_os("MallocLargeCache").is_some() {
            return;
        }
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let mut args = std::env::args_os();
        let mut cmd = std::process::Command::new(exe);
        if let Some(arg0) = args.next() {
            cmd.arg0(arg0);
        }
        // `exec` returns only on failure, which costs the saving and nothing else.
        let _ = cmd.args(args).env("MallocLargeCache", "0").exec();
    }
}
