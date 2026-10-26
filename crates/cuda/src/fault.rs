//! The fault a test can put behind a readback of the group FFT.
//!
//! CUDA has no completion token to fail or to leave stale: a launch the driver accepted
//! and a stream that drained are all a readback has, and a store the device did not make
//! looks exactly like one it did. So the one fault here is that one, the Metal injector's
//! `WrongStore` (`crates/metal/src/cb.rs`): the first byte of a chosen readback is
//! flipped after the download, as a buffer the GPU reported written with a value it did
//! not store. Nothing here retries it; whether the checks in `fft.rs` notice is the
//! question it exists to ask.

/// Every host readback of the FFT follows its download with this, on the words about to
/// be checked, so a test can corrupt one store the drained stream did not see
/// ([`inject::Fault::WrongStore`]). Outside tests and the `fault-inject` feature it is
/// nothing.
pub(crate) fn stored(words: &mut [u32]) {
    #[cfg(any(test, feature = "fault-inject"))]
    if inject::wrong_store_fires() {
        if let Some(w) = words.first_mut() {
            *w ^= 1;
        }
    }
    #[cfg(not(any(test, feature = "fault-inject")))]
    let _ = words;
}

/// The per-thread injector. Readbacks on the arming thread are counted from [`arm`], and
/// the one at the armed index comes back with its first byte flipped; with `sticky`,
/// every one from there on, so no re-run can succeed.
///
/// The state is per thread: a fault armed here is reached by this thread's readbacks
/// only, and a caller whose work goes to a pool must arm on the thread that downloads.
/// `prepare` submits the device blocks of a section from the thread that called it, so a
/// test arms on its own thread. Exported as `snarkrs_cuda::inject` under the
/// `fault-inject` feature.
#[cfg(any(test, feature = "fault-inject"))]
pub mod inject {
    use std::cell::Cell;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Fault {
        /// The stream drained and the download returned; one byte of what it returned
        /// is not what the device holds.
        WrongStore,
    }

    thread_local! {
        static CALLS: Cell<u32> = const { Cell::new(0) };
        static FAIL: Cell<Option<(u32, Fault)>> = const { Cell::new(None) };
        /// Every readback from `FAIL`'s index on, not just the one.
        static STICKY: Cell<bool> = const { Cell::new(false) };
        static FIRED: Cell<bool> = const { Cell::new(false) };
    }

    /// Counts readbacks from zero again, and corrupts readback `at` if given.
    pub fn arm(at: Option<u32>) {
        arm_with(at.map(|n| (n, Fault::WrongStore)), false);
    }

    /// Counts readbacks from zero again, and corrupts readback `at` with `fault` if
    /// given; with `sticky`, every readback from `at` on.
    pub fn arm_with(at: Option<(u32, Fault)>, sticky: bool) {
        CALLS.set(0);
        FAIL.set(at);
        STICKY.set(sticky);
        FIRED.set(false);
    }

    /// Readbacks on this thread since [`arm`].
    pub fn calls() -> u32 {
        CALLS.get()
    }

    /// Whether the armed readback was reached.
    pub fn fired() -> bool {
        FIRED.get()
    }

    /// For the readback being made: counts it, and says whether it is the armed one.
    pub(super) fn wrong_store_fires() -> bool {
        let n = CALLS.get();
        CALLS.set(n + 1);
        let hit = match FAIL.get() {
            Some((at, Fault::WrongStore)) => n == at || (STICKY.get() && n > at),
            None => false,
        };
        if hit {
            FIRED.set(true);
        }
        hit
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        struct Reset;

        impl Drop for Reset {
            fn drop(&mut self) {
                arm(None);
            }
        }

        /// The armed readback alone is flipped, by one bit of its first word; sticky,
        /// every later one too; another thread sees nothing; and a reset clears it.
        #[test]
        fn one_readback_is_flipped_and_the_rest_come_back_whole() {
            let _reset = Reset;
            arm(None);
            let mut words = vec![0x1234_5678u32, 9];
            crate::fault::stored(&mut words);
            assert_eq!(words, [0x1234_5678, 9]);
            assert_eq!(calls(), 1);
            assert!(!fired());

            arm(Some(1));
            let mut a = vec![7u32];
            let mut b = vec![7u32];
            let mut c = vec![7u32];
            crate::fault::stored(&mut a);
            crate::fault::stored(&mut b);
            crate::fault::stored(&mut c);
            assert_eq!((a[0], b[0], c[0]), (7, 6, 7));
            assert!(fired());
            assert_eq!(calls(), 3);

            arm_with(Some((1, Fault::WrongStore)), true);
            let mut v = vec![0u32; 4];
            for w in v.iter_mut() {
                crate::fault::stored(std::slice::from_mut(w));
            }
            assert_eq!(v, [0, 1, 1, 1]);

            std::thread::spawn(|| {
                let mut w = vec![0u32];
                crate::fault::stored(&mut w);
                assert_eq!(w, [0]);
                assert!(!fired());
                assert_eq!(calls(), 1);
            })
            .join()
            .unwrap();
            assert_eq!(calls(), 4);

            let mut empty: Vec<u32> = Vec::new();
            crate::fault::stored(&mut empty);
            assert!(empty.is_empty());
        }
    }
}
