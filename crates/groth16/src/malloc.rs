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

/// Frees `v` with its pages handed back to the OS now, not whenever malloc reuses them.
///
/// Dropping is not enough for a key section in a process that keeps the large cache. On
/// js_384x384_d32 the cache held 2.4 GB of dropped zkey sections for the rest of the
/// process. Mapping fresh anonymous pages over the block's whole pages releases the old
/// ones at once and leaves malloc owning an address range that costs nothing until it is
/// touched again. Off Apple targets this is a plain drop.
pub fn release<T: Copy>(mut v: Vec<T>) {
    let (start, bytes) = (v.as_mut_ptr() as usize, v.capacity() * size_of::<T>());
    #[cfg(target_vendor = "apple")]
    darwin::replace_pages(start, bytes);
    #[cfg(not(target_vendor = "apple"))]
    let _ = (start, bytes);
    drop(v);
}

#[cfg(target_vendor = "apple")]
mod darwin {
    use std::ffi::c_void;

    // libSystem, which every process links. Declared here rather than taken from `libc` so
    // this crate's dependency set does not change for two calls.
    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
    }
    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const MAP_PRIVATE: i32 = 0x2;
    const MAP_FIXED: i32 = 0x10;
    const MAP_ANON: i32 = 0x1000;
    /// Apple silicon's page. Also a multiple of a 4 KB page, so the rounding stays valid on
    /// a machine that has those.
    const PAGE: usize = 16 << 10;

    /// Maps fresh pages over the whole pages of the `bytes` at `start`, which must be a
    /// `Vec` allocation that [`super::release`] drops next.
    pub(super) fn replace_pages(start: usize, bytes: usize) {
        let lo = start.next_multiple_of(PAGE);
        let hi = (start + bytes) & !(PAGE - 1);
        if hi > lo {
            // SAFETY: `lo..hi` lies inside the `Vec`'s allocation, which nothing else
            // references, and its element type is `Copy`, so the drop that follows reads
            // none of it. The new pages are private, readable and writable, as malloc's were, so
            // it can reuse or unmap the block as it would have. XNU puts the old mapping
            // back if a fixed mapping fails, so a failure costs only the saving; nothing is
            // checked because nothing changes.
            unsafe {
                mmap(
                    lo as *mut c_void,
                    hi - lo,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_FIXED | MAP_ANON,
                    -1,
                    0,
                );
            }
        }
    }
}
