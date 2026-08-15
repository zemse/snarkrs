//! Cost of every primitive the prover is actually made of, single threaded.
//!
//! The CPU profile says 63% of all CPU in a proof is inside `ark-ff`'s `Fp` operations.
//! That is a direction, not a baseline: the next lane cannot tell whether a replacement
//! field is faster unless it knows what the current one costs. This is that table.
//!
//! Latency and throughput are both reported because they say different things and the
//! prover needs both. A bucket accumulation is a dependent chain into one bucket
//! (latency); a batch of independent butterflies is not (throughput). Latency is measured
//! with a serial dependency chain, throughput with four independent accumulators so the
//! out-of-order window has something to overlap.
//!
//! Read alongside `bench/results/profiling/asm-butterflies.txt`, which shows what the
//! compiler emits for these: 412 instructions for one butterfly, of which 64 are the
//! 64x64 multiplies of the Montgomery reduction and 27 are conditional branches, mostly
//! the compare-against-the-modulus that follows every single operation.
//!
//! `cargo run --release -p g16-core --example field_ops`

use std::time::Instant;

use g16_field::{
    CurveGroup, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective,
    PrimeField, UniformRand,
};

/// Nanoseconds per operation, best of `reps` passes of `iters` operations.
fn ns(reps: usize, iters: usize, f: &mut dyn FnMut(usize)) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f(iters);
            t.elapsed().as_secs_f64() * 1e9 / iters as f64
        })
        .fold(f64::MAX, f64::min)
}

/// M2 Max performance-core clock. Only used to turn nanoseconds into a cycle count, which
/// is the unit the emitted instruction stream can be compared against.
const GHZ: f64 = 3.68;

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let n = 1 << 20;
    let mut rng = ark_std::test_rng();

    let a = Fr::rand(&mut rng);
    let b = Fr::rand(&mut rng);
    let qa = Fq::rand(&mut rng);
    let qb = Fq::rand(&mut rng);
    let q2a = Fq2::rand(&mut rng);
    let q2b = Fq2::rand(&mut rng);
    let p1 = G1Projective::rand(&mut rng);
    let p1a: G1Affine = p1.into_affine();
    let p2 = G2Projective::rand(&mut rng);
    let p2a: G2Affine = p2.into_affine();

    println!("single threaded, best of {reps}, {n} operations per pass");
    println!("cycles assume a {GHZ} GHz performance core\n");
    println!(
        "{:<34}{:>11}{:>10}{:>13}{:>10}",
        "operation", "lat ns", "lat cyc", "thruput ns", "thr cyc"
    );

    let row = |name: &str, lat: f64, thr: f64| {
        println!(
            "{name:<34}{lat:>11.2}{:>10.0}{thr:>13.2}{:>10.0}",
            lat * GHZ,
            thr * GHZ
        );
    };

    // Fr multiply. Latency: one accumulator, every multiply waits for the last. Throughput:
    // four independent accumulators, which is enough to fill the multiplier pipeline.
    let lat = ns(reps, n, &mut |k| {
        let mut x = a;
        for _ in 0..k {
            x *= b;
        }
        let _ = std::hint::black_box(x);
    });
    let thr = ns(reps, n, &mut |k| {
        let (mut w, mut x, mut y, mut z) = (a, a + b, a + a, b);
        for _ in 0..k / 4 {
            w *= b;
            x *= b;
            y *= b;
            z *= b;
        }
        std::hint::black_box((w, x, y, z));
    });
    row("Fr mul (Montgomery, 4 limbs)", lat, thr);

    let lat = ns(reps, n, &mut |k| {
        let mut x = a;
        for _ in 0..k {
            x += b;
        }
        let _ = std::hint::black_box(x);
    });
    let thr = ns(reps, n, &mut |k| {
        let (mut w, mut x, mut y, mut z) = (a, a + b, a + a, b);
        for _ in 0..k / 4 {
            w += b;
            x += b;
            y += b;
            z += b;
        }
        std::hint::black_box((w, x, y, z));
    });
    row("Fr add (+ conditional subtract)", lat, thr);

    let lat = ns(reps, n, &mut |k| {
        let mut x = a;
        for _ in 0..k {
            x -= b;
        }
        let _ = std::hint::black_box(x);
    });
    row("Fr sub (+ conditional add)", lat, f64::NAN);

    let lat = ns(reps, n, &mut |k| {
        let mut x = qa;
        for _ in 0..k {
            x *= qb;
        }
        let _ = std::hint::black_box(x);
    });
    let thr = ns(reps, n, &mut |k| {
        let (mut w, mut x, mut y, mut z) = (qa, qa + qb, qa + qa, qb);
        for _ in 0..k / 4 {
            w *= qb;
            x *= qb;
            y *= qb;
            z *= qb;
        }
        std::hint::black_box((w, x, y, z));
    });
    row("Fq mul (base field, 4 limbs)", lat, thr);

    // Fq2 is where G2 gets expensive: arkworks routes its multiply through
    // `Fp::sum_of_products`, which the profile shows as 21% of all CPU in a proof.
    let lat = ns(reps, n / 2, &mut |k| {
        let mut x = q2a;
        for _ in 0..k {
            x *= q2b;
        }
        let _ = std::hint::black_box(x);
    });
    row("Fq2 mul (G2 tower)", lat, f64::NAN);

    // `into_bigint` is the Montgomery reduction the MSM's prescan pays once per general
    // scalar, and exactly once: the module docs are explicit that calling it per window
    // would be the classic way to halve an MSM's speed.
    let lat = ns(reps, n, &mut |k| {
        let mut acc = 0u64;
        let mut x = a;
        for _ in 0..k {
            acc ^= x.into_bigint().0[0];
            x += b;
        }
        std::hint::black_box(acc);
    });
    row("Fr into_bigint + Fr add", lat, f64::NAN);

    // The two curve operations the MSM is built from. The mixed add is the bucket
    // accumulation; the full add is the running-sum reduction over the buckets.
    //
    // The throughput column is the important one here and it is not a microbenchmarking
    // formality. `window_chunk` accumulates into 2^(c-1) different buckets, so consecutive
    // adds are usually independent and could in principle overlap. Whether they do is the
    // difference between the bucket loop being bound by the Fq multiply dependency chain
    // (fix: make the adds independent, e.g. batch affine accumulation) and being bound by
    // multiplier throughput (fix: a faster multiply, and nothing else).
    let lat = ns(reps, n / 4, &mut |k| {
        let mut x = p1;
        for _ in 0..k {
            x += &p1a;
        }
        let _ = std::hint::black_box(x);
    });
    let thr = ns(reps, n / 4, &mut |k| {
        let (mut w, mut x, mut y, mut z) = (p1, p1 + p1, p1 + p1 + p1, p1 + p1 + p1 + p1);
        for _ in 0..k / 4 {
            w += &p1a;
            x += &p1a;
            y += &p1a;
            z += &p1a;
        }
        let _ = std::hint::black_box((w, x, y, z));
    });
    row("G1 mixed add (proj += affine)", lat, thr);

    let lat = ns(reps, n / 4, &mut |k| {
        let mut x = p1;
        let y = p1 + p1;
        for _ in 0..k {
            x += &y;
        }
        let _ = std::hint::black_box(x);
    });
    row("G1 full add (proj += proj)", lat, f64::NAN);

    let lat = ns(reps, n / 8, &mut |k| {
        let mut x = p2;
        for _ in 0..k {
            x += &p2a;
        }
        let _ = std::hint::black_box(x);
    });
    let thr = ns(reps, n / 8, &mut |k| {
        let (mut w, mut x, mut y, mut z) = (p2, p2 + p2, p2 + p2 + p2, p2 + p2 + p2 + p2);
        for _ in 0..k / 4 {
            w += &p2a;
            x += &p2a;
            y += &p2a;
            z += &p2a;
        }
        let _ = std::hint::black_box((w, x, y, z));
    });
    row("G2 mixed add (proj += affine)", lat, thr);

    let lat = ns(reps, n / 8, &mut |k| {
        let mut x = p2;
        let y = p2 + p2;
        for _ in 0..k {
            x += &y;
        }
        let _ = std::hint::black_box(x);
    });
    row("G2 full add (proj += proj)", lat, f64::NAN);

    println!();
    println!(
        "note: `ark-ff` is declared with features = [\"asm\"] in the workspace Cargo.toml, \
         and that\n      assembly is gated on target_arch = \"x86_64\". On this machine \
         it is a no-op."
    );
}
