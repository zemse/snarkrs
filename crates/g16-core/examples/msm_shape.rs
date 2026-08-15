//! What the five MSMs are actually made of, and what each one costs.
//!
//! `StageTimings` reports one `msm_us` for all five, which is enough to say MSM dominates
//! and not enough to say anything else. Two questions this answers that the aggregate
//! cannot:
//!
//!   * **The 0/1 claim.** `g16-msm`'s module docs assert "in bit-decomposition-heavy
//!     circuits over 99% of witness scalars are 0 or 1" and build the whole prescan fast
//!     path on it. That is measurable per circuit, and the answer decides whether the
//!     bucket loop is a sparse problem or a dense one.
//!   * **Where the MSM time sits.** Four MSMs take the witness (sparse); one takes H
//!     (dense, `domain_size` general scalars). If H is most of it, work on the dense
//!     bucket path; if the witness MSMs are, work on the prescan.
//!
//! Each MSM is timed on its own, outside the `rayon::join` nest the real prover uses, so
//! the numbers add up to more than `msm_us`: the prover overlaps them. That is the point.
//! The overlap efficiency (serial sum / observed parallel time) is reported too, because a
//! poor one is its own finding.
//!
//! `cargo run --release -p g16-core --example msm_shape -- bench/artifacts`

use std::time::Instant;

use g16_core::{cpu::CpuBackend, Backend, StageTimings};
use g16_field::{AffineRepr, Fr, One, Zero};
use g16_msm::{window_size, CpuMsm, MsmBackend};
use g16_zkey::{wtns::Witness, ProvingKey};

/// Largest |value| the review's proposed small-scalar fast path would catch.
const SMALL: u64 = 8;

/// What a scalar vector is made of, plus how many of its bases are the point at infinity.
///
/// Infinity bases are counted because `prescan` classifies scalars and never looks at the
/// base. A base at infinity contributes nothing to the result no matter what its scalar
/// is, so every Montgomery reduction and every window scan spent on it is wasted, and the
/// waste is invisible in the aggregate `msm_us`.
struct Shape {
    n: usize,
    zeros: usize,
    ones: usize,
    /// Scalars equal to +/-k for some 1 < k <= SMALL. Excludes 0 and 1, which already
    /// have fast paths, so this is exactly the extra ground the review's proposal covers.
    smalls: usize,
    inf_bases: usize,
    /// General scalars whose base is *not* infinity: the work that genuinely has to
    /// reach the bucket loop.
    useful: usize,
}

impl Shape {
    fn of<A: AffineRepr>(s: &[Fr], bases: &[A]) -> Self {
        assert_eq!(s.len(), bases.len(), "shape: length mismatch");
        // -k rather than recomputing a negation per scalar.
        let small: Vec<(Fr, Fr)> = (2..=SMALL)
            .map(|k| {
                let v = Fr::from(k);
                (v, -v)
            })
            .collect();
        let mut sh = Shape {
            n: s.len(),
            zeros: 0,
            ones: 0,
            smalls: 0,
            inf_bases: 0,
            useful: 0,
        };
        for (x, b) in s.iter().zip(bases) {
            let inf = b.is_zero();
            if inf {
                sh.inf_bases += 1;
            }
            if x.is_zero() {
                sh.zeros += 1;
            } else if x.is_one() {
                sh.ones += 1;
            } else {
                if small.iter().any(|(p, m)| x == p || x == m) {
                    sh.smalls += 1;
                }
                if !inf {
                    sh.useful += 1;
                }
            }
        }
        sh
    }
    fn general(&self) -> usize {
        self.n - self.zeros - self.ones
    }
    fn trivial_pct(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        100.0 * (self.zeros + self.ones) as f64 / self.n as f64
    }
    fn pct(&self, k: usize) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        100.0 * k as f64 / self.n as f64
    }
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/artifacts".to_string());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("reading {root}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .collect();
    dirs.sort();

    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let (n_vars, n_public, domain) = (pk.n_vars, pk.n_public, pk.domain_size);

        let circuit = CpuBackend::new().prepare(pk).unwrap();
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&witness, &mut t).unwrap();
        let h = h.to_host().unwrap().to_vec();
        let pk = circuit.key();

        // The MSM silently min()s a length mismatch in release, so every pairing is
        // checked here rather than trusted. A truncated probe would look fast and be wrong.
        let l_scalars = &witness[n_public + 1..];
        assert_eq!(pk.a_query.len(), witness.len(), "{name}: A query length");
        assert_eq!(pk.b_g2_query.len(), witness.len(), "{name}: B_G2 length");
        assert_eq!(pk.b_g1_query.len(), witness.len(), "{name}: B_G1 length");
        assert_eq!(pk.l_query.len(), l_scalars.len(), "{name}: L query length");
        assert_eq!(pk.h_query.len(), h.len(), "{name}: H query length");

        println!("\n=== {name} ===");
        println!("n_vars {n_vars}  n_public {n_public}  domain {domain}");

        let msm = CpuMsm::new();
        // Best of `reps`, not mean: on a laptop the mean of a contended run measures the
        // other tenants. Each MSM is run alone, so the pool is entirely its own.
        let best = |f: &mut dyn FnMut()| {
            (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };

        println!(
            "{:<8} {:>9} {:>8} {:>7} {:>9} {:>7} {:>8} {:>9} {:>10} {:>3} {:>9}",
            "msm", "len", "zeros", "ones", "general", "0/1 %", "|s|<=8", "inf bases", "inf %", "c", "ms"
        );
        let mut serial_total = 0.0;
        let mut rows: Vec<(&str, Shape, f64)> = Vec::new();
        for (label, shape, ms) in [
            ("A_g1", Shape::of(&witness, &pk.a_query), {
                best(&mut || {
                    let _ = std::hint::black_box(msm.msm_g1(&pk.a_query, &witness));
                })
            }),
            ("B_g2", Shape::of(&witness, &pk.b_g2_query), {
                best(&mut || {
                    let _ = std::hint::black_box(msm.msm_g2(&pk.b_g2_query, &witness));
                })
            }),
            ("B_g1", Shape::of(&witness, &pk.b_g1_query), {
                best(&mut || {
                    let _ = std::hint::black_box(msm.msm_g1(&pk.b_g1_query, &witness));
                })
            }),
            ("L_g1", Shape::of(l_scalars, &pk.l_query), {
                best(&mut || {
                    let _ = std::hint::black_box(msm.msm_g1(&pk.l_query, l_scalars));
                })
            }),
            ("H_g1", Shape::of(&h, &pk.h_query), {
                best(&mut || {
                    let _ = std::hint::black_box(msm.msm_g1(&pk.h_query, &h));
                })
            }),
        ] {
            serial_total += ms;
            rows.push((label, shape, ms));
        }

        for (label, s, ms) in &rows {
            println!(
                "{:<8} {:>9} {:>8} {:>7} {:>9} {:>6.2}% {:>8} {:>9} {:>9.2}% {:>3} {:>9.3}",
                label,
                s.n,
                s.zeros,
                s.ones,
                s.general(),
                s.trivial_pct(),
                s.smalls,
                s.inf_bases,
                s.pct(s.inf_bases),
                window_size(s.general()),
                ms
            );
        }
        // The rate the bucket loop would run at if it only touched pairs that can move
        // the answer. `general` is what it processes today; `useful` is what matters.
        for (label, s, ms) in &rows {
            if s.general() > 0 {
                println!(
                    "  {label:<6} {:>7.3} us per general scalar, {:>7.2}% of generals have an \
                     infinity base",
                    1e3 * ms / s.general() as f64,
                    100.0 * (s.general() - s.useful) as f64 / s.general() as f64,
                );
            }
        }

        // The parallel number the prover actually pays, same input, same instance.
        let mut t2 = StageTimings::default();
        let parallel_ms = (0..reps)
            .map(|_| {
                let mut t = StageTimings::default();
                let start = Instant::now();
                let _ = std::hint::black_box(
                    circuit
                        .msms(&witness, &g16_core::HPoly::Host(h.clone()), &mut t)
                        .unwrap(),
                );
                t2 = t;
                start.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::MAX, f64::min);
        let _ = t2;

        println!(
            "serial sum {serial_total:>8.3} ms   prover (5 overlapped) {parallel_ms:>8.3} ms   \
             overlap speedup {:.2}x",
            serial_total / parallel_ms
        );
        // Share of the serial sum, which is the right denominator for "which MSM should I
        // optimise": it is the work, independent of how well the pool happens to overlap.
        for (label, _, ms) in &rows {
            println!("  {label:<6} {:>5.1}% of serial MSM work", 100.0 * ms / serial_total);
        }
    }
}
