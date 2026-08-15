//! Measurement probe: is there headroom over ark-ff's generic Montgomery multiply on
//! aarch64? ark-ff's `asm` feature is x86_64-only, so on this M2 Max the prover runs
//! ark's generic Rust CIOS. This probe cross-checks two local reimplementations against
//! ark-ff on a couple hundred thousand operands and then times three regimes.
//!
//! Run with `cargo test --release -p g16-field --test aarch64_probe -- --nocapture`.
#![cfg(target_arch = "aarch64")]

use g16_field::{Fr, UniformRand};
use std::hint::black_box;
use std::time::Instant;

const N: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
const N0: u64 = 0xc2e1f593efffffff; // -r^{-1} mod 2^64

#[inline(always)]
fn limbs(x: &Fr) -> [u64; 4] {
    x.0 .0
}
#[inline(always)]
fn from_limbs(l: [u64; 4]) -> Fr {
    ark_ff::Fp(ark_ff::BigInt(l), core::marker::PhantomData)
}

#[inline(always)]
fn cond_sub(t: [u64; 4]) -> [u64; 4] {
    let (r0, br) = t[0].overflowing_sub(N[0]);
    let (r1, br) = borrowing_sub(t[1], N[1], br);
    let (r2, br) = borrowing_sub(t[2], N[2], br);
    let (r3, br) = borrowing_sub(t[3], N[3], br);
    if br {
        t
    } else {
        [r0, r1, r2, r3]
    }
}

#[inline(always)]
fn borrowing_sub(a: u64, b: u64, borrow: bool) -> (u64, bool) {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(borrow as u64);
    (d, b1 | b2)
}

/// Fused no-carry CIOS in plain Rust u128, gnark's shape (what ark-ff also implements).
#[inline(always)]
fn mont_mul_u128(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut t = [0u64; 4];
    for i in 0..4 {
        let bi = b[i] as u128;
        let mut aa = t[0] as u128 + a[0] as u128 * bi;
        let t0 = aa as u64;
        aa >>= 64;
        let m = t0.wrapping_mul(N0);
        let mut c = (t0 as u128 + (m as u128) * (N[0] as u128)) >> 64;
        for j in 1..4 {
            aa = t[j] as u128 + a[j] as u128 * bi + aa;
            let tj = aa as u64;
            aa >>= 64;
            c = tj as u128 + (m as u128) * (N[j] as u128) + c;
            t[j - 1] = c as u64;
            c >>= 64;
        }
        t[3] = (c as u64).wrapping_add(aa as u64);
    }
    cond_sub(t)
}

/// Same arithmetic as the GPU's two-pass CIOS, in aarch64 inline asm: mul/umulh with
/// adds/adcs carry chains, one asm block per round, the shift folded into the hi-part
/// addition of the reduction pass.
#[inline(always)]
fn mont_mul_asm(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut t0: u64 = 0;
    let mut t1: u64 = 0;
    let mut t2: u64 = 0;
    let mut t3: u64 = 0;
    for &bi in b.iter() {
        unsafe {
            core::arch::asm!(
                // multiply pass: t += a * bi, lo then hi chain; a4 is the carry word
                "mul   {l0}, {a0}, {bi}",
                "umulh {h0}, {a0}, {bi}",
                "mul   {l1}, {a1}, {bi}",
                "umulh {h1}, {a1}, {bi}",
                "mul   {l2}, {a2}, {bi}",
                "umulh {h2}, {a2}, {bi}",
                "mul   {l3}, {a3}, {bi}",
                "umulh {h3}, {a3}, {bi}",
                "adds  {t0}, {t0}, {l0}",
                "adcs  {t1}, {t1}, {l1}",
                "adcs  {t2}, {t2}, {l2}",
                "adcs  {t3}, {t3}, {l3}",
                "adc   {a4}, xzr, xzr",
                "adds  {t1}, {t1}, {h0}",
                "adcs  {t2}, {t2}, {h1}",
                "adcs  {t3}, {t3}, {h2}",
                "adc   {a4}, {a4}, {h3}",
                // m annihilates t0
                "mul   {m}, {t0}, {n0}",
                // reduction pass: t += m * N, lo chain; then hi chain fused with the
                // one-limb shift down
                "mul   {l0}, {m}, {q0}",
                "umulh {h0}, {m}, {q0}",
                "mul   {l1}, {m}, {q1}",
                "umulh {h1}, {m}, {q1}",
                "mul   {l2}, {m}, {q2}",
                "umulh {h2}, {m}, {q2}",
                "mul   {l3}, {m}, {q3}",
                "umulh {h3}, {m}, {q3}",
                "adds  {t0}, {t0}, {l0}",
                "adcs  {t1}, {t1}, {l1}",
                "adcs  {t2}, {t2}, {l2}",
                "adcs  {t3}, {t3}, {l3}",
                "adc   {a4}, {a4}, xzr",
                "adds  {t0}, {t1}, {h0}",
                "adcs  {t1}, {t2}, {h1}",
                "adcs  {t2}, {t3}, {h2}",
                "adc   {t3}, {a4}, {h3}",
                t0 = inout(reg) t0, t1 = inout(reg) t1,
                t2 = inout(reg) t2, t3 = inout(reg) t3,
                a0 = in(reg) a[0], a1 = in(reg) a[1],
                a2 = in(reg) a[2], a3 = in(reg) a[3],
                bi = in(reg) bi, n0 = in(reg) N0,
                q0 = in(reg) N[0], q1 = in(reg) N[1],
                q2 = in(reg) N[2], q3 = in(reg) N[3],
                l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
                h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
                m = out(reg) _, a4 = out(reg) _,
                options(pure, nomem, nostack)
            );
        }
    }
    cond_sub([t0, t1, t2, t3])
}

#[test]
fn probe() {
    let mut rng = ark_std::test_rng();
    for i in 0..200_000u64 {
        let a = if i < 4 {
            -Fr::from(i + 1)
        } else {
            Fr::rand(&mut rng)
        };
        let b = if i < 4 { -Fr::from(1u64) } else { Fr::rand(&mut rng) };
        let want = a * b;
        assert_eq!(from_limbs(mont_mul_u128(limbs(&a), limbs(&b))), want, "u128 wrong at {i}");
        assert_eq!(from_limbs(mont_mul_asm(limbs(&a), limbs(&b))), want, "asm wrong at {i}");
    }

    let x0 = Fr::rand(&mut rng);
    let m = Fr::rand(&mut rng);
    const ITERS: u64 = 4_000_000;
    let ml = limbs(&m);

    let t = Instant::now();
    let mut x = x0;
    for _ in 0..ITERS {
        x *= black_box(m);
    }
    let ark_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;

    let t = Instant::now();
    let mut y = limbs(&x0);
    for _ in 0..ITERS {
        y = mont_mul_u128(y, black_box(ml));
    }
    let u128_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;

    let t = Instant::now();
    let mut z = limbs(&x0);
    for _ in 0..ITERS {
        z = mont_mul_asm(z, black_box(ml));
    }
    let asm_ns = t.elapsed().as_nanos() as f64 / ITERS as f64;

    assert_eq!(from_limbs(y), x);
    assert_eq!(from_limbs(z), x);
    println!("latency chain:      ark {ark_ns:.2} ns   u128 {u128_ns:.2} ns   asm {asm_ns:.2} ns");

    // 8 independent chains: the ILP regime an NTT butterfly array pass lives in.
    let seeds: Vec<[u64; 4]> = (0..8).map(|_| limbs(&Fr::rand(&mut rng))).collect();
    let per = (ITERS / 4) as f64 * 8.0;

    let t = Instant::now();
    let mut xs: Vec<Fr> = seeds.iter().map(|s| from_limbs(*s)).collect();
    for _ in 0..ITERS / 4 {
        for x in xs.iter_mut() {
            *x *= black_box(m);
        }
    }
    let ark_tp = t.elapsed().as_nanos() as f64 / per;

    let t = Instant::now();
    let mut us: Vec<[u64; 4]> = seeds.clone();
    for _ in 0..ITERS / 4 {
        for u in us.iter_mut() {
            *u = mont_mul_u128(*u, black_box(ml));
        }
    }
    let u_tp = t.elapsed().as_nanos() as f64 / per;

    let t = Instant::now();
    let mut ys: Vec<[u64; 4]> = seeds.clone();
    for _ in 0..ITERS / 4 {
        for y in ys.iter_mut() {
            *y = mont_mul_asm(*y, black_box(ml));
        }
    }
    let asm_tp = t.elapsed().as_nanos() as f64 / per;

    for ((u, y), x) in us.iter().zip(&ys).zip(&xs) {
        assert_eq!(from_limbs(*u), *x);
        assert_eq!(from_limbs(*y), *x);
    }
    println!("8-chain throughput: ark {ark_tp:.2} ns   u128 {u_tp:.2} ns   asm {asm_tp:.2} ns");
}
