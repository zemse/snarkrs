//! `--backend auto`: the CPU for a proof too small to repay a CUDA device, CUDA above that.
//!
//! A CUDA proof has a floor the CPU does not. Warm, it is about 12 ms on a T4 for anything
//! past tiny_mul; one proof per process (the CLI) adds the context, the kernel load and the
//! key upload, about 0.5 s. Below that the CPU is faster, by 2x on tiny_mul warm and by 50x
//! on it cold. The crossover is measured, not derived:
//! `bench/results/cuda-auto-2026-11-11.md`.
//!
//! The choice needs the size of the CPU's work, and the CPU's work is not the constraint
//! count. Its MSMs skip the witness entries that are 0 or 1, which are 92 to 100% of the
//! wires of sha256, keccak256, rsa2048 and anon-aadhaar, so variable work is counted as the
//! H MSM's `domain_size` dense scalars plus the wires whose scalar is neither; constant work
//! skips nothing and counts every wire. That over the rayon pool is the CPU's share, and
//! CUDA is chosen when it is above the threshold for the residency.
//!
//! Only `auto` is decided here. An explicit `--backend` is always what runs.

use anyhow::Result;
use ark_ff::{One, Zero};
use snarkrs_field::Fr;
use snarkrs_formats::ProvingKey;
use snarkrs_groth16::Backend;

use crate::{log, BackendKind};

/// How many proofs one prepared circuit serves, which decides what CUDA's setup costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// One proof per prepare: the CLI, `bench --mode cold-process` and `cold-inprocess`.
    /// CUDA pays its context, kernel load and key upload on every proof.
    OneShot,
    /// Many proofs per prepare: `bench --mode warm`, a resident service. Setup is amortised.
    Resident,
}

/// What the CPU would have to do for one proof, as far as the choice needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workload {
    pub domain_size: usize,
    pub n_vars: usize,
    /// Witness entries other than 0 and 1, which the CPU MSMs cannot skip.
    pub live: usize,
    pub constant_work: bool,
}

impl Workload {
    pub fn of(pk: &ProvingKey, witness: &[Fr], constant_work: bool) -> Self {
        Self {
            domain_size: pk.domain_size,
            n_vars: pk.n_vars,
            live: live_scalars(witness),
            constant_work,
        }
    }

    /// The CPU's work in scalar-point products: H's dense `domain_size`, then every wire
    /// under constant work or the live ones otherwise.
    pub fn cpu_points(&self) -> usize {
        let wires = if self.constant_work {
            self.n_vars
        } else {
            self.live
        };
        self.domain_size + wires
    }
}

/// Witness entries other than 0 and 1.
pub fn live_scalars(witness: &[Fr]) -> usize {
    witness
        .iter()
        .filter(|w| !w.is_zero() && !w.is_one())
        .count()
}

/// Points per CPU thread above which CUDA is faster, by residency and work mode. Measured on
/// a T4 (g4dn.xlarge and 2xlarge); see the module docs.
pub const fn threshold(residency: Residency, constant_work: bool) -> usize {
    match (residency, constant_work) {
        (Residency::Resident, false) => RESIDENT_VARIABLE,
        (Residency::Resident, true) => RESIDENT_CONSTANT,
        (Residency::OneShot, false) => ONE_SHOT_VARIABLE,
        (Residency::OneShot, true) => ONE_SHOT_CONSTANT,
    }
}

// Warm, CUDA's floor is about 12 ms for any circuit past tiny_mul, which the CPU reaches
// between domain 2^9 and 2^11 on 8 threads (chain_500 and chain_2000), in either work mode.
const RESIDENT_VARIABLE: usize = 192;
const RESIDENT_CONSTANT: usize = 192;
// One shot, CUDA's floor is 0.46 s (0.55 s constant work), which the CPU reaches between
// rsa2048 and js_16x16_d32 on 8 threads (variable), js_8x8_d32 and js_16x16_d32 (constant).
const ONE_SHOT_VARIABLE: usize = 40_000;
const ONE_SHOT_CONSTANT: usize = 32_000;

/// The backend `--backend auto` runs: CUDA when this binary has it and the CPU's share of
/// the work is over the threshold, else the CPU. Never Metal or wgpu.
pub fn choose(work: &Workload, residency: Residency, threads: usize, cuda: bool) -> BackendKind {
    let per_thread = work.cpu_points() / threads.max(1);
    if cuda && per_thread > threshold(residency, work.constant_work) {
        BackendKind::Cuda
    } else {
        BackendKind::Cpu
    }
}

/// The size of the rayon pool proving runs on: `--threads`, else what rayon will give the
/// global pool. Read without building the pool, which costs tiny_mul's CPU proof 8% when
/// that proof never needed it.
pub fn pool_threads(threads: Option<usize>) -> usize {
    threads
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|n| n.parse().ok())
                .filter(|&n| n > 0)
        })
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Builds the backend [`choose`] picks on this binary and a pool of `threads`. A CUDA device
/// that fails to open goes to the CPU with a warning: `auto` promised a proof, not a device.
pub fn make(
    work: &Workload,
    residency: Residency,
    threads: usize,
) -> Result<(BackendKind, Box<dyn Backend>)> {
    let kind = choose(work, residency, threads, cfg!(feature = "cuda"));
    log::debug(format_args!(
        "backend auto: {} ({} cpu points over {threads} threads, {residency:?})",
        kind.as_str(),
        work.cpu_points()
    ));
    let build = |k| {
        if work.constant_work {
            crate::make_constant_work_backend(k)
        } else {
            crate::make_backend(k)
        }
    };
    match build(kind) {
        Ok(b) => Ok((kind, b)),
        Err(e) if kind != BackendKind::Cpu => {
            log::warn(format_args!("{e}; backend auto proves on the cpu"));
            Ok((BackendKind::Cpu, build(BackendKind::Cpu)?))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(domain_size: usize, n_vars: usize, live: usize, constant_work: bool) -> Workload {
        Workload {
            domain_size,
            n_vars,
            live,
            constant_work,
        }
    }

    #[test]
    fn zero_and_one_are_not_live() {
        let w = [
            Fr::from(0u64),
            Fr::from(1u64),
            Fr::from(2u64),
            -Fr::from(1u64),
        ];
        assert_eq!(live_scalars(&w), 2);
        assert_eq!(live_scalars(&[]), 0);
    }

    #[test]
    fn constant_work_counts_every_wire() {
        assert_eq!(work(1 << 16, 59_170, 0, false).cpu_points(), 1 << 16);
        assert_eq!(
            work(1 << 16, 59_170, 0, true).cpu_points(),
            (1 << 16) + 59_170
        );
        assert_eq!(
            work(1 << 15, 20_154, 18_406, false).cpu_points(),
            32_768 + 18_406
        );
    }

    #[test]
    fn without_cuda_auto_is_the_cpu() {
        let huge = work(1 << 24, 1 << 24, 1 << 24, false);
        for r in [Residency::OneShot, Residency::Resident] {
            assert_eq!(choose(&huge, r, 1, false), BackendKind::Cpu);
            assert_eq!(choose(&huge, r, 1, true), BackendKind::Cuda);
        }
    }

    #[test]
    fn a_tiny_circuit_stays_on_the_cpu_in_every_mode() {
        for cw in [false, true] {
            let tiny = work(4, 6, 5, cw);
            for r in [Residency::OneShot, Residency::Resident] {
                assert_eq!(choose(&tiny, r, 1, true), BackendKind::Cpu, "{r:?} {cw}");
            }
        }
    }

    /// One proof per process pays CUDA's setup, so it needs more work before CUDA pays.
    #[test]
    fn one_shot_needs_more_work_than_resident() {
        for cw in [false, true] {
            assert!(threshold(Residency::OneShot, cw) > threshold(Residency::Resident, cw));
        }
    }

    /// `(name, domain_size, n_vars, live)` of the measured ladder, and the faster backend on a
    /// T4 per box (4 and 8 threads) as `[warm var, warm cw, one-shot var, one-shot cw]`,
    /// `c` cpu, `g` cuda, `.` within 5% either way. bench/results/cuda-auto-2026-11-11.md.
    const LADDER: &[(&str, usize, usize, usize, &str, &str)] = &[
        ("tiny_mul", 8, 6, 5, "cccc", "cccc"),
        ("chain_120", 128, 122, 121, "cccc", "cccc"),
        ("chain_500", 512, 502, 501, "ggcc", "cccc"),
        ("chain_2000", 2048, 2002, 2001, "ggcc", "ggcc"),
        ("js_1x1_d8", 4096, 3373, 3234, "ggcc", "ggcc"),
        ("js_2x2_d16", 16384, 10194, 9906, "ggcc", "ggcc"),
        ("js_2x2_d32", 32768, 18002, 17682, "ggcc", "ggcc"),
        ("railgun-01x01", 32768, 20154, 18406, "ggcc", "ggcc"),
        ("tornado", 32768, 28300, 27595, "ggcc", "ggcc"),
        ("sha256", 65536, 59170, 0, "ggc.", "ggcc"),
        ("js_8x8_d32", 131072, 70640, 69372, "gggg", "ggcc"),
        ("js_16x16_d32", 262144, 140824, 138292, "gggg", "gggg"),
        ("railgun-13x01", 262144, 141499, 139399, "gggg", "ggg."),
        ("rsa2048", 262144, 190035, 5254, "gggg", "ggcg"),
        ("keccak256", 262144, 240257, 0, "gggg", "ggcg"),
        ("anon-aadhaar", 2097152, 1101048, 87385, "gggg", "gggg"),
    ];

    #[test]
    fn auto_picks_the_faster_backend_on_every_measured_circuit() {
        let cells = [
            (Residency::Resident, false),
            (Residency::Resident, true),
            (Residency::OneShot, false),
            (Residency::OneShot, true),
        ];
        for &(name, domain, n_vars, live, x4, x8) in LADDER {
            for (threads, want) in [(4, x4), (8, x8)] {
                for (&(r, cw), want) in cells.iter().zip(want.chars()) {
                    let got = choose(&work(domain, n_vars, live, cw), r, threads, true);
                    let ok = match want {
                        'c' => got == BackendKind::Cpu,
                        'g' => got == BackendKind::Cuda,
                        _ => true,
                    };
                    assert!(ok, "{name} {threads} threads {r:?} cw={cw}: {got:?}");
                }
            }
        }
    }

    #[test]
    fn threads_given_on_the_command_line_are_the_pool() {
        assert_eq!(pool_threads(Some(3)), 3);
        assert!(pool_threads(None) >= 1);
    }

    /// The threshold is per thread: the same circuit on a bigger CPU pool can stay there.
    #[test]
    fn more_cpu_threads_keep_more_on_the_cpu() {
        let t = threshold(Residency::OneShot, false);
        let w = work(t * 2, t * 2, 0, false);
        assert_eq!(choose(&w, Residency::OneShot, 1, true), BackendKind::Cuda);
        assert_eq!(choose(&w, Residency::OneShot, 4, true), BackendKind::Cpu);
    }
}
