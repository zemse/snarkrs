//! Operation counts for the ceremony commands, no timing: what `groth16 setup`, `ptau
//! prepare phase2` and the key scaling of `ptau contribute` / `zkey contribute` ask of the
//! point arithmetic, under arkworks' affine double-and-add and under `glv::mul`.
//!
//! `cargo run --release -p snarkrs-ceremony --example ceremony_trace -- [--prepare K]...
//! [--contribute K]... [R1CS]...`
//!
//! Counts are point doublings (`dbl`) and additions (`add`, any formula) per group. Under
//! double-and-add a scalar of bit length `b` and weight `h` costs `b - 1` doublings and
//! `h - 1` additions. Under GLV it costs the longer wNAF of the two halves in doublings,
//! their nonzero digits less one in additions, and the odd-multiple table it builds.

use std::path::{Path, PathBuf};

use snarkrs_ceremony::r1cs::R1cs;
use snarkrs_ceremony::setup::{process_constraints, PointTerm};
use snarkrs_field::{BigInteger, FftField, Field, Fr, One, PrimeField};
use snarkrs_msm::glv;

#[derive(Default, Clone, Copy)]
struct Cost {
    muls: u64,
    dbl: u64,
    add: u64,
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, o: Self) {
        self.muls += o.muls;
        self.dbl += o.dbl;
        self.add += o.add;
    }
}

/// Double-and-add over the scalar's bits, which is what `Affine * Fr` does in ark-ec 0.5.
fn ark_cost(k: &Fr) -> Cost {
    let b = k.into_bigint();
    let bits = b.num_bits() as u64;
    let weight: u64 = b.0.iter().map(|l| l.count_ones() as u64).sum();
    Cost {
        muls: (bits > 0) as u64,
        dbl: bits.saturating_sub(1),
        add: weight.saturating_sub(1),
    }
}

/// The width-5 wNAF over all 254 bits that `point_times_fr` ran before GLV: table of 8
/// odd multiples (one doubling, seven additions), then one doubling a digit.
fn wnaf254_cost(k: &Fr) -> Cost {
    let d = k.into_bigint().find_wnaf(5).expect("width 5");
    let nz = d.iter().filter(|&&x| x != 0).count() as u64;
    Cost {
        muls: 1,
        dbl: d.len() as u64 - 1 + 1,
        add: nz - 1 + 7,
    }
}

/// `glv::mul`'s schedule, recomputed from its public decomposition.
fn glv_cost(k: &Fr) -> Cost {
    let (k1, k2) = glv::decompose(k);
    let (n1, z1, m1) = wnaf_shape(k1.unsigned_abs());
    let (n2, z2, m2) = wnaf_shape(k2.unsigned_abs());
    let n = n1.max(n2);
    if n == 0 {
        return Cost::default();
    }
    let len = (m1.max(m2) / 2 + 1) as u64;
    Cost {
        muls: 1,
        // The ladder's first doubling is of the identity, which costs nothing.
        dbl: n as u64 - 1 + (len > 1) as u64,
        add: z1 + z2 - 1 + (len - 1),
    }
}

/// (digits, nonzero digits, largest |digit|) of the width-5 NAF `glv::mul` uses.
fn wnaf_shape(mut k: u128) -> (usize, u64, u32) {
    let (mut n, mut nz, mut top) = (0, 0, 0);
    while k != 0 {
        if k & 1 == 1 {
            let m = (k & 31) as i32;
            let d = if m >= 16 { m - 32 } else { m };
            k = if d >= 0 {
                k - d as u128
            } else {
                k + (-d) as u128
            };
            nz += 1;
            top = top.max(d.unsigned_abs());
        }
        n += 1;
        k >>= 1;
    }
    (n, nz, top)
}

fn line(label: &str, ark: Cost, glv: Cost) {
    println!(
        "  {label:<34} muls {:>11}  ark dbl {:>13} add {:>12}  glv dbl {:>13} add {:>12}",
        ark.muls, ark.dbl, ark.add, glv.dbl, glv.add
    );
}

fn setup(path: &Path) {
    let r1cs = R1cs::open(path).expect("r1cs");
    let h = r1cs.header();
    let domain = h.domain_size();
    let acc = process_constraints(&r1cs, domain).expect("constraints");
    println!(
        "setup {}: n_vars {} n_constraints {} domain 2^{}",
        path.display(),
        h.n_vars,
        h.n_constraints,
        domain.trailing_zeros()
    );
    let mut bits_hist = [0u64; 5];
    for (label, slots) in [
        ("IC (G1)", &acc.ic),
        ("C (G1)", &acc.c),
        ("A (G1)", &acc.a),
        ("B1 (G1)", &acc.b1),
        ("B2 (G2)", &acc.b2),
    ] {
        let (mut ark, mut glv) = (Cost::default(), Cost::default());
        let (mut msm_slots, mut msm_terms, mut terms) = (0u64, 0u64, 0u64);
        for slot in slots.iter() {
            terms += slot.len() as u64;
            // `MULTIEXP_MIN_TERMS`: these go to the MSM, not to the scalar loop.
            if slot.len() >= 32 {
                msm_slots += 1;
                msm_terms += slot.len() as u64;
                continue;
            }
            for t in slot {
                let k = scalar(&r1cs, t);
                ark += ark_cost(&k);
                glv += glv_cost(&k);
                let b = k.into_bigint().num_bits();
                let nb = (-k).into_bigint().num_bits();
                bits_hist[match b.min(nb) {
                    0..=1 => 0,
                    2..=16 => 1,
                    17..=64 => 2,
                    65..=128 => 3,
                    _ => 4,
                }] += 1;
            }
        }
        println!(
            "  {label}: {} slots, {terms} terms, {msm_slots} slots / {msm_terms} terms to the MSM",
            slots.len()
        );
        line("  small-slot scalar multiplications", ark, glv);
    }
    println!(
        "  small-slot |k| or |r-k| bits: <=1 {}, 2-16 {}, 17-64 {}, 65-128 {}, >128 {}",
        bits_hist[0], bits_hist[1], bits_hist[2], bits_hist[3], bits_hist[4]
    );
}

fn scalar(r1cs: &R1cs, t: &PointTerm) -> Fr {
    match t.coef_ptr {
        Some(p) => r1cs.coef(p).expect("coef"),
        None => Fr::one(),
    }
}

/// `ptau prepare phase2` at `power`: per section, the blocks `2^0 .. 2^last` each get
/// `p 2^(p-1)` butterflies (two additions each, one twiddle unless it is 1) and `2^p`
/// scalings by `1/n`. The twiddles are walked exactly, so both columns are exact. Here
/// the "ark" column is the 254-bit wNAF the transform used before GLV, not ark-ec.
fn prepare(power: u32) {
    println!("prepare phase2 at power {power}");
    let mut g1 = (Cost::default(), Cost::default(), 0u64);
    let mut g2 = g1;
    for (label, last, is_g2) in [
        ("tauG1 (12)", power + 1, false),
        ("tauG2 (13)", power, true),
        ("alphaTauG1 (14)", power, false),
        ("betaTauG1 (15)", power, false),
    ] {
        let (mut ark, mut glv, mut adds) = (Cost::default(), Cost::default(), 0u64);
        for p in 0..=last {
            let (a, g, s) = block(p);
            ark += a;
            glv += g;
            adds += s;
        }
        line(label, ark, glv);
        println!("  {label:<34} butterfly additions {adds}");
        let t = if is_g2 { &mut g2 } else { &mut g1 };
        t.0 += ark;
        t.1 += glv;
        t.2 += adds;
    }
    line("total G1", g1.0, g1.1);
    line("total G2", g2.0, g2.1);
    println!("  butterfly additions G1 {} G2 {}", g1.2, g2.2);
}

/// One block of `2^p` points: (ark, glv) twiddle and scaling costs, butterfly additions.
fn block(p: u32) -> (Cost, Cost, u64) {
    let n = 1u64 << p;
    let (mut ark, mut glv) = (Cost::default(), Cost::default());
    if p == 0 {
        return (ark, glv, 0);
    }
    let mut root = Fr::TWO_ADIC_ROOT_OF_UNITY;
    for _ in p..Fr::TWO_ADICITY {
        root.square_in_place();
    }
    // Pass `exp` uses `w_exp^j`, `j < 2^(exp-1)`, each `n / 2^exp` times.
    let mut roots = vec![root];
    for _ in 1..p {
        let r = *roots.last().unwrap();
        roots.push(r * r);
    }
    for exp in 1..=p {
        let w = roots[(p - exp) as usize];
        let uses = n >> exp;
        let mut tw = Fr::one();
        for _ in 0..1u64 << (exp - 1) {
            if !tw.is_one() {
                for _ in 0..uses {
                    ark += wnaf254_cost(&tw);
                    glv += glv_cost(&tw);
                }
            }
            tw *= w;
        }
    }
    let inv = Fr::from(n).inverse().unwrap();
    for _ in 0..n {
        ark += wnaf254_cost(&inv);
        glv += glv_cost(&inv);
    }
    (ark, glv, p as u64 * n)
}

/// Key scaling at `power`: every point of the five sections times its own scalar, which
/// is a random-looking `Fr`. Costed from 4096 sampled scalars.
fn contribute(power: u32) {
    let n = 1u64 << power;
    let (mut ark, mut glv) = (Cost::default(), Cost::default());
    let mut t = Fr::from(0x5eed_u64).inverse().unwrap();
    let samples = 4096u64;
    for _ in 0..samples {
        ark += ark_cost(&t);
        glv += glv_cost(&t);
        t = t.square() + Fr::from(3u64);
    }
    let scale = |c: Cost, m: u64| Cost {
        muls: c.muls * m / samples,
        dbl: c.dbl * m / samples,
        add: c.add * m / samples,
    };
    println!("ptau contribute at power {power}");
    line(
        "G1 (2n-1 + n + n points)",
        scale(ark, 4 * n - 1),
        scale(glv, 4 * n - 1),
    );
    line("G2 (n + 1 points)", scale(ark, n + 1), scale(glv, n + 1));
}

fn main() {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prepare" => prepare(args.next().expect("power").parse().expect("power")),
            "--contribute" => contribute(args.next().expect("power").parse().expect("power")),
            path => setup(&PathBuf::from(path)),
        }
    }
}
