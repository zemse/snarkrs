//! Branch-free BN254 field arithmetic on raw Montgomery limbs.
//!
//! # Why this exists
//!
//! Profiling (bench/results/profiling/README.md, round 4) put 62.8% of CPU proof time
//! inside `ark-ff`'s `Fp`, and showed the multiply itself is near the machine limit; what
//! is not is everything wrapped around it. `ark-ff` finishes every add, sub and multiply
//! with a *compare-and-branch* against the modulus: a four-limb lexicographic compare
//! whose first branch is taken with probability about one half on Montgomery outputs,
//! which is a guaranteed-unpredictable branch in the innermost loop of the prover.
//! `cargo asm` on the NTT butterfly counted 27 conditional branches per iteration from
//! exactly this. The measured symptom at the point level: a G1 mixed add costs 247 ns
//! against a 148 ns floor set by its own multiplies.
//!
//! The functions here are the same arithmetic with the reduction done branch-free: the
//! conditional subtraction computes both candidates and blends them with a mask, which
//! LLVM lowers to `csel` on aarch64 and `cmov` on x86_64. No secret-independent branch
//! remains in add/sub/mul/sqr (`inverse` and comparisons are not on the hot path).
//!
//! # What lives here
//!
//! `RawFq`, `RawFq2` and `RawFr`: plain limb arrays in Montgomery form, bit-compatible
//! with `ark-ff`'s in-memory representation (arkworks stores Montgomery limbs
//! little-endian, which is exactly what these wrap), so conversion in either direction is
//! a copy, never an arithmetic operation. The intended consumer is a hot loop that
//! converts on entry, works on raw limbs throughout, and converts back on exit; the MSM
//! bucket accumulation in `snarkrs-msm` is the canonical one.
//!
//! Every operation keeps values fully reduced in `[0, p)`, matching `ark-ff` semantics
//! exactly. That makes cross-checking trivial (bit equality against ark on every sample)
//! and keeps this module drop-in safe; lazy `[0, 2p)` forms would buy a little more and
//! are the documented next step if measurement demands them.
//!
//! The multiply is the fused "no final carry" CIOS (gnark's shape), valid because BN254's
//! top word 0x30644e72e131a029 is below 2^63 - 1 on both fields. The same shape was
//! validated for the GPU ports against a bit-exact host simulation; here it is
//! additionally differential-tested against `ark-ff` in `tests/raw_matches_ark.rs`.
//!
//! On aarch64 the add, sub, conditional subtraction and multiply are inline assembly, the
//! same arithmetic in the instruction sequence LLVM will not produce from the portable
//! code: it turns a subtraction of the modulus into an addition of its complement and
//! carries every borrow through cset/csinc or cinc/sbc pairs. Single thread on an M2 Max
//! (`examples/raw_ops.rs`, ns, dependent chain / eight chains / random stream) that took
//! add from 6.43/3.30/3.40 to 2.72/1.84/1.81, sub from 5.39/2.49/2.64 to 2.71/1.84/1.83 and
//! mul from 25.7/19.4/19.6 to 20.4/13.9/13.9. Every other target keeps the portable code.

use crate::{Fq, Fq2, Fr};

/// BN254 base field modulus, little-endian u64 limbs.
pub const FQ_MOD: [u64; 4] = [
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
/// `-q^{-1} mod 2^64`.
pub const FQ_N0: u64 = 0x87d20782e4866389;

/// BN254 scalar field modulus, little-endian u64 limbs.
pub const FR_MOD: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
/// `-r^{-1} mod 2^64`.
pub const FR_N0: u64 = 0xc2e1f593efffffff;

// ---------------------------------------------------------------------------
// Limb kernels, shared by both fields. All #[inline(always)]: these exist to be
// swallowed whole by the caller's loop body.
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    // 128-bit subtract keeps the borrow as data rather than as a flag the compiler is
    // tempted to branch on.
    let t = (a as u128)
        .wrapping_sub(b as u128)
        .wrapping_sub(borrow as u128);
    (t as u64, (t >> 127) as u64)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `t < m`, the `[0, p)` invariant the whole module is written against. Only the
/// `debug_assert!`s on the conversion entry points call this; nothing on a hot path does.
fn is_reduced(t: &[u64; 4], m: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if t[i] != m[i] {
            return t[i] < m[i];
        }
    }
    false
}

/// `if t >= m { t - m } else { t }`, branch-free: both candidates are computed and
/// blended by a mask. LLVM emits csel/cmov for the blend.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn cond_sub(t: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (r0, b) = sbb(t[0], m[0], 0);
    let (r1, b) = sbb(t[1], m[1], b);
    let (r2, b) = sbb(t[2], m[2], b);
    let (r3, b) = sbb(t[3], m[3], b);
    // b == 1 means t < m: keep t. The mask is all-ones in that case.
    let keep = b.wrapping_neg();
    [
        (t[0] & keep) | (r0 & !keep),
        (t[1] & keep) | (r1 & !keep),
        (t[2] & keep) | (r2 & !keep),
        (t[3] & keep) | (r3 & !keep),
    ]
}

/// `(a + b) mod m` for fully reduced inputs. The carry out of limb 3 is provably zero
/// for both BN254 fields (moduli are 254-bit), so the sum fits four limbs and one
/// conditional subtraction finishes it.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn add_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (s0, c) = adc(a[0], b[0], 0);
    let (s1, c) = adc(a[1], b[1], c);
    let (s2, c) = adc(a[2], b[2], c);
    let (s3, _) = adc(a[3], b[3], c);
    cond_sub([s0, s1, s2, s3], m)
}

/// `(a - b) mod m`, branch-free: the modulus is masked in when the subtraction borrows.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn sub_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let (d0, bw) = sbb(a[0], b[0], 0);
    let (d1, bw) = sbb(a[1], b[1], bw);
    let (d2, bw) = sbb(a[2], b[2], bw);
    let (d3, bw) = sbb(a[3], b[3], bw);
    let mask = bw.wrapping_neg();
    let (r0, c) = adc(d0, m[0] & mask, 0);
    let (r1, c) = adc(d1, m[1] & mask, c);
    let (r2, c) = adc(d2, m[2] & mask, c);
    let (r3, _) = adc(d3, m[3] & mask, c);
    [r0, r1, r2, r3]
}

/// Montgomery product, fused no-final-carry CIOS. Montgomery in, Montgomery out, fully
/// reduced. Valid for moduli whose top word is at most (2^64 - 1)/2 - 1, which both
/// BN254 fields satisfy; see the module doc.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn mont_mul(a: [u64; 4], b: [u64; 4], m: &[u64; 4], n0: u64) -> [u64; 4] {
    let mut t = [0u64; 4];
    for i in 0..4 {
        let bi = b[i] as u128;
        let mut aa = t[0] as u128 + a[0] as u128 * bi;
        let t0 = aa as u64;
        aa >>= 64;
        let mm = t0.wrapping_mul(n0);
        let mut c = (t0 as u128 + (mm as u128) * (m[0] as u128)) >> 64;
        for j in 1..4 {
            aa = t[j] as u128 + a[j] as u128 * bi + aa;
            let tj = aa as u64;
            aa >>= 64;
            c = tj as u128 + (mm as u128) * (m[j] as u128) + c;
            t[j - 1] = c as u64;
            c >>= 64;
        }
        // Exact by the top-word bound: no carry out of limb 3.
        t[3] = (c as u64).wrapping_add(aa as u64);
    }
    cond_sub(t, m)
}

/// `(a + b) mod m` on aarch64: one adds/adcs chain for the sum, one subs/sbcs chain
/// against `m`, four csel. Written out because LLVM will not emit the subs/sbcs chain
/// against a constant: it rewrites `s - m` as `s + (2^64 - m)` and carries the borrow
/// through cinc/sbc or cset/csinc pairs, twice the instructions.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let [mut a0, mut a1, mut a2, mut a3] = a;
    // SAFETY: register arithmetic only: no memory, no stack; the flags are clobbered,
    // which `asm!` assumes unless told `preserves_flags`.
    unsafe {
        core::arch::asm!(
            "adds {a0}, {a0}, {b0}",
            "adcs {a1}, {a1}, {b1}",
            "adcs {a2}, {a2}, {b2}",
            "adc {a3}, {a3}, {b3}",
            "subs {d0}, {a0}, {m0}",
            "sbcs {d1}, {a1}, {m1}",
            "sbcs {d2}, {a2}, {m2}",
            "sbcs {d3}, {a3}, {m3}",
            "csel {a0}, {a0}, {d0}, lo",
            "csel {a1}, {a1}, {d1}, lo",
            "csel {a2}, {a2}, {d2}, lo",
            "csel {a3}, {a3}, {d3}, lo",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            m0 = in(reg) m[0],
            m1 = in(reg) m[1],
            m2 = in(reg) m[2],
            m3 = in(reg) m[3],
            d0 = out(reg) _,
            d1 = out(reg) _,
            d2 = out(reg) _,
            d3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [a0, a1, a2, a3]
}

/// `(a - b) mod m` on aarch64: the difference, then `m` masked in by csel when it
/// borrowed and added back.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sub_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let [mut a0, mut a1, mut a2, mut a3] = a;
    // SAFETY: register arithmetic only: no memory, no stack; the flags are clobbered,
    // which `asm!` assumes unless told `preserves_flags`.
    unsafe {
        core::arch::asm!(
            "subs {a0}, {a0}, {b0}",
            "sbcs {a1}, {a1}, {b1}",
            "sbcs {a2}, {a2}, {b2}",
            "sbcs {a3}, {a3}, {b3}",
            "csel {d0}, {m0}, xzr, lo",
            "csel {d1}, {m1}, xzr, lo",
            "csel {d2}, {m2}, xzr, lo",
            "csel {d3}, {m3}, xzr, lo",
            "adds {a0}, {a0}, {d0}",
            "adcs {a1}, {a1}, {d1}",
            "adcs {a2}, {a2}, {d2}",
            "adc {a3}, {a3}, {d3}",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            m0 = in(reg) m[0],
            m1 = in(reg) m[1],
            m2 = in(reg) m[2],
            m3 = in(reg) m[3],
            d0 = out(reg) _,
            d1 = out(reg) _,
            d2 = out(reg) _,
            d3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [a0, a1, a2, a3]
}

/// [`mont_mul`] on aarch64: gnark-crypto's 4-word arm64 `mul`
/// (`field/asm/element_4w/element_4w_arm64.s`), the same no-final-carry CIOS, a row at a
/// time with one adds/adcs chain per row, closed by the same branch-free subs/sbcs and
/// csel. LLVM's code for the portable version spends a cinc or cset per limb on carries
/// and rematerialises the modulus inside the loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mont_mul(a: [u64; 4], b: [u64; 4], m: &[u64; 4], n0: u64) -> [u64; 4] {
    let (t0, t1, t2, t3): (u64, u64, u64, u64);
    // SAFETY: register arithmetic only: no memory, no stack; flags clobbered.
    unsafe {
        core::arch::asm!(
            "mul {t0}, {a0}, {b0}",
            "mul {t1}, {a1}, {b0}",
            "mul {t2}, {a2}, {b0}",
            "mul {t3}, {a3}, {b0}",
            "umulh {tmp}, {a0}, {b0}",
            "adds {t1}, {t1}, {tmp}",
            "umulh {tmp}, {a1}, {b0}",
            "adcs {t2}, {t2}, {tmp}",
            "umulh {tmp}, {a2}, {b0}",
            "adcs {t3}, {t3}, {tmp}",
            "umulh {tmp}, {a3}, {b0}",
            "adc {t4}, {tmp}, xzr",
            "mul {m}, {t0}, {inv}",
            "mul {tmp}, {q0}, {m}",
            "adds {t0}, {t0}, {tmp}",
            "mul {tmp}, {q1}, {m}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {q2}, {m}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {q3}, {m}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, {t4}, xzr",
            "umulh {tmp}, {q0}, {m}",
            "adds {t0}, {t1}, {tmp}",
            "umulh {tmp}, {q1}, {m}",
            "adcs {t1}, {t2}, {tmp}",
            "umulh {tmp}, {q2}, {m}",
            "adcs {t2}, {t3}, {tmp}",
            "umulh {tmp}, {q3}, {m}",
            "adc {t3}, {t4}, {tmp}",
            "mul {tmp}, {a0}, {b1}",
            "adds {t0}, {t0}, {tmp}",
            "mul {m}, {t0}, {inv}",
            "mul {tmp}, {a1}, {b1}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {a2}, {b1}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {a3}, {b1}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, xzr, xzr",
            "umulh {tmp}, {a0}, {b1}",
            "adds {t1}, {t1}, {tmp}",
            "umulh {tmp}, {a1}, {b1}",
            "adcs {t2}, {t2}, {tmp}",
            "umulh {tmp}, {a2}, {b1}",
            "adcs {t3}, {t3}, {tmp}",
            "umulh {tmp}, {a3}, {b1}",
            "adc {t4}, {t4}, {tmp}",
            "mul {tmp}, {q0}, {m}",
            "adds {t0}, {t0}, {tmp}",
            "mul {tmp}, {q1}, {m}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {q2}, {m}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {q3}, {m}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, {t4}, xzr",
            "umulh {tmp}, {q0}, {m}",
            "adds {t0}, {t1}, {tmp}",
            "umulh {tmp}, {q1}, {m}",
            "adcs {t1}, {t2}, {tmp}",
            "umulh {tmp}, {q2}, {m}",
            "adcs {t2}, {t3}, {tmp}",
            "umulh {tmp}, {q3}, {m}",
            "adc {t3}, {t4}, {tmp}",
            "mul {tmp}, {a0}, {b2}",
            "adds {t0}, {t0}, {tmp}",
            "mul {m}, {t0}, {inv}",
            "mul {tmp}, {a1}, {b2}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {a2}, {b2}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {a3}, {b2}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, xzr, xzr",
            "umulh {tmp}, {a0}, {b2}",
            "adds {t1}, {t1}, {tmp}",
            "umulh {tmp}, {a1}, {b2}",
            "adcs {t2}, {t2}, {tmp}",
            "umulh {tmp}, {a2}, {b2}",
            "adcs {t3}, {t3}, {tmp}",
            "umulh {tmp}, {a3}, {b2}",
            "adc {t4}, {t4}, {tmp}",
            "mul {tmp}, {q0}, {m}",
            "adds {t0}, {t0}, {tmp}",
            "mul {tmp}, {q1}, {m}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {q2}, {m}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {q3}, {m}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, {t4}, xzr",
            "umulh {tmp}, {q0}, {m}",
            "adds {t0}, {t1}, {tmp}",
            "umulh {tmp}, {q1}, {m}",
            "adcs {t1}, {t2}, {tmp}",
            "umulh {tmp}, {q2}, {m}",
            "adcs {t2}, {t3}, {tmp}",
            "umulh {tmp}, {q3}, {m}",
            "adc {t3}, {t4}, {tmp}",
            "mul {tmp}, {a0}, {b3}",
            "adds {t0}, {t0}, {tmp}",
            "mul {m}, {t0}, {inv}",
            "mul {tmp}, {a1}, {b3}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {a2}, {b3}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {a3}, {b3}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, xzr, xzr",
            "umulh {tmp}, {a0}, {b3}",
            "adds {t1}, {t1}, {tmp}",
            "umulh {tmp}, {a1}, {b3}",
            "adcs {t2}, {t2}, {tmp}",
            "umulh {tmp}, {a2}, {b3}",
            "adcs {t3}, {t3}, {tmp}",
            "umulh {tmp}, {a3}, {b3}",
            "adc {t4}, {t4}, {tmp}",
            "mul {tmp}, {q0}, {m}",
            "adds {t0}, {t0}, {tmp}",
            "mul {tmp}, {q1}, {m}",
            "adcs {t1}, {t1}, {tmp}",
            "mul {tmp}, {q2}, {m}",
            "adcs {t2}, {t2}, {tmp}",
            "mul {tmp}, {q3}, {m}",
            "adcs {t3}, {t3}, {tmp}",
            "adc {t4}, {t4}, xzr",
            "umulh {tmp}, {q0}, {m}",
            "adds {t0}, {t1}, {tmp}",
            "umulh {tmp}, {q1}, {m}",
            "adcs {t1}, {t2}, {tmp}",
            "umulh {tmp}, {q2}, {m}",
            "adcs {t2}, {t3}, {tmp}",
            "umulh {tmp}, {q3}, {m}",
            "adc {t3}, {t4}, {tmp}",
            "subs {a0}, {t0}, {q0}",
            "sbcs {a1}, {t1}, {q1}",
            "sbcs {a2}, {t2}, {q2}",
            "sbcs {a3}, {t3}, {q3}",
            "csel {t0}, {a0}, {t0}, cs",
            "csel {t1}, {a1}, {t1}, cs",
            "csel {t2}, {a2}, {t2}, cs",
            "csel {t3}, {a3}, {t3}, cs",
            a0 = inout(reg) a[0] => _,
            a1 = inout(reg) a[1] => _,
            a2 = inout(reg) a[2] => _,
            a3 = inout(reg) a[3] => _,
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            q0 = in(reg) m[0],
            q1 = in(reg) m[1],
            q2 = in(reg) m[2],
            q3 = in(reg) m[3],
            inv = in(reg) n0,
            t0 = out(reg) t0,
            t1 = out(reg) t1,
            t2 = out(reg) t2,
            t3 = out(reg) t3,
            t4 = out(reg) _,
            m = out(reg) _,
            tmp = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [t0, t1, t2, t3]
}

// ---------------------------------------------------------------------------
// The operation surface, as a trait, so curve arithmetic can be written once over a
// generic base field and instantiated for G1 (RawFq) and G2 (RawFq2), exactly as the
// Metal and CUDA kernels do with their f_add/f_mul overload sets.
// ---------------------------------------------------------------------------

pub trait RawField: Copy + PartialEq + Send + Sync {
    const ZERO: Self;
    /// The Montgomery representative of 1.
    const ONE: Self;
    fn is_zero(&self) -> bool;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn sqr(self) -> Self;
    fn double(self) -> Self;
}

// ---------------------------------------------------------------------------
// RawFq
// ---------------------------------------------------------------------------

/// BN254 base field element as raw Montgomery limbs, always fully reduced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFq(pub [u64; 4]);

impl RawFq {
    pub const ZERO: RawFq = RawFq([0; 4]);
    /// R mod q, the Montgomery representative of 1. Pinned against ark in tests.
    pub const ONE: RawFq = RawFq([
        0xd35d438dc58f0d9d,
        0x0a78eb28f5c70b3d,
        0x666ea36f7879462c,
        0x0e0a77c19a07df2f,
    ]);

    #[inline(always)]
    pub fn from_fq(x: &Fq) -> Self {
        // arkworks stores exactly these limbs (Montgomery form, little-endian). Reading
        // the hidden field is the one place here that has no public equivalent: every
        // public accessor goes out through `into_bigint`, i.e. a Montgomery round-trip,
        // which is exactly the cost this type exists to avoid.
        //
        // This is also the boundary the `[0, p)` invariant is only ever crossed at. It is
        // not free-standing hygiene: `mont_mul`'s fused shape drops the carry out of limb
        // 3 because reduced inputs cannot produce one, so an unreduced element entering
        // here comes back out of the next multiply as a wrong field element. The GPU
        // unpack path is the live risk — `PackedFq::to_fq` is `new_unchecked` over raw
        // device words, range-checked by nothing but the kernels' closing `cond_sub`.
        debug_assert!(is_reduced(&x.0 .0, &FQ_MOD), "RawFq from an unreduced Fq");
        RawFq(x.0 .0)
    }

    #[inline(always)]
    pub fn to_fq(self) -> Fq {
        Fq::new_unchecked(ark_ff::BigInt::new(self.0))
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFq(add_mod(self.0, o.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFq(sub_mod(self.0, o.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        RawFq(sub_mod([0; 4], self.0, &FQ_MOD))
    }

    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        RawFq(mont_mul(self.0, o.0, &FQ_MOD, FQ_N0))
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        self.mul(self)
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        self.add(self)
    }
}

impl RawField for RawFq {
    const ZERO: Self = RawFq::ZERO;
    const ONE: Self = RawFq::ONE;
    #[inline(always)]
    fn is_zero(&self) -> bool {
        RawFq::is_zero(self)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        RawFq::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        RawFq::sub(self, o)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        RawFq::neg(self)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        RawFq::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        RawFq::sqr(self)
    }
    #[inline(always)]
    fn double(self) -> Self {
        RawFq::double(self)
    }
}

// ---------------------------------------------------------------------------
// RawFq2 = RawFq[u] / (u^2 + 1)
// ---------------------------------------------------------------------------

/// BN254 `Fq2` element as a pair of raw `Fq`. Multiplication is Karatsuba (three `Fq`
/// products), squaring the complex trick (two), the same shapes as `ark-ff` and the GPU
/// kernels use, so operation counts are comparable across all three.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFq2 {
    pub c0: RawFq,
    pub c1: RawFq,
}

impl RawFq2 {
    pub const ZERO: RawFq2 = RawFq2 {
        c0: RawFq::ZERO,
        c1: RawFq::ZERO,
    };
    pub const ONE: RawFq2 = RawFq2 {
        c0: RawFq::ONE,
        c1: RawFq::ZERO,
    };

    #[inline(always)]
    pub fn from_fq2(x: &Fq2) -> Self {
        RawFq2 {
            c0: RawFq::from_fq(&x.c0),
            c1: RawFq::from_fq(&x.c1),
        }
    }

    #[inline(always)]
    pub fn to_fq2(self) -> Fq2 {
        Fq2::new(self.c0.to_fq(), self.c1.to_fq())
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFq2 {
            c0: self.c0.add(o.c0),
            c1: self.c1.add(o.c1),
        }
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFq2 {
            c0: self.c0.sub(o.c0),
            c1: self.c1.sub(o.c1),
        }
    }

    #[inline(always)]
    pub fn neg(self) -> Self {
        RawFq2 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// `(a0 + a1 u)(b0 + b1 u)` with `u^2 = -1`: Karatsuba, three `Fq` products.
    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        let v0 = self.c0.mul(o.c0);
        let v1 = self.c1.mul(o.c1);
        let s = self.c0.add(self.c1);
        let t = o.c0.add(o.c1);
        RawFq2 {
            c0: v0.sub(v1),
            c1: s.mul(t).sub(v0).sub(v1),
        }
    }

    /// `(a0 + a1 u)^2 = (a0 + a1)(a0 - a1) + 2 a0 a1 u`.
    #[inline(always)]
    pub fn sqr(self) -> Self {
        let t0 = self.c0.add(self.c1);
        let t1 = self.c0.sub(self.c1);
        let t2 = self.c0.mul(self.c1);
        RawFq2 {
            c0: t0.mul(t1),
            c1: t2.add(t2),
        }
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        self.add(self)
    }
}

impl RawField for RawFq2 {
    const ZERO: Self = RawFq2::ZERO;
    const ONE: Self = RawFq2::ONE;
    #[inline(always)]
    fn is_zero(&self) -> bool {
        RawFq2::is_zero(self)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        RawFq2::add(self, o)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        RawFq2::sub(self, o)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        RawFq2::neg(self)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        RawFq2::mul(self, o)
    }
    #[inline(always)]
    fn sqr(self) -> Self {
        RawFq2::sqr(self)
    }
    #[inline(always)]
    fn double(self) -> Self {
        RawFq2::double(self)
    }
}

// ---------------------------------------------------------------------------
// RawFr
// ---------------------------------------------------------------------------

/// BN254 scalar field element as raw Montgomery limbs. The NTT butterflies in
/// `snarkrs-ntt` run on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RawFr(pub [u64; 4]);

impl RawFr {
    pub const ZERO: RawFr = RawFr([0; 4]);

    #[inline(always)]
    pub fn from_fr(x: &Fr) -> Self {
        debug_assert!(is_reduced(&x.0 .0, &FR_MOD), "RawFr from an unreduced Fr");
        RawFr(x.0 .0)
    }

    #[inline(always)]
    pub fn to_fr(self) -> Fr {
        Fr::new_unchecked(ark_ff::BigInt::new(self.0))
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        RawFr(add_mod(self.0, o.0, &FR_MOD))
    }

    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        RawFr(sub_mod(self.0, o.0, &FR_MOD))
    }

    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        RawFr(mont_mul(self.0, o.0, &FR_MOD, FR_N0))
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        self.mul(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m[0] * n0 == -1 mod 2^64` is the defining property of the Montgomery constant;
    /// pinned here so the hex literals above cannot rot silently.
    #[test]
    fn montgomery_constants_are_inverses() {
        assert_eq!(FQ_MOD[0].wrapping_mul(FQ_N0), u64::MAX);
        assert_eq!(FR_MOD[0].wrapping_mul(FR_N0), u64::MAX);
    }

    #[test]
    fn ones_match_ark() {
        use ark_ff::One;
        assert_eq!(RawFq::ONE.to_fq(), Fq::one());
        assert_eq!(RawFq2::ONE.to_fq2(), Fq2::one());
        assert_eq!(RawFq::ONE, <RawFq as RawField>::ONE);
    }

    #[test]
    fn moduli_match_ark() {
        use ark_ff::PrimeField;
        assert_eq!(Fq::MODULUS.0, FQ_MOD);
        assert_eq!(Fr::MODULUS.0, FR_MOD);
    }
}
