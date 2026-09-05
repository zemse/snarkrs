//! WGSL field arithmetic for BN254, emitted from Rust.
//!
//! # Why this is a generator and not a `.wgsl` file
//!
//! The limb-width sweep measured the same CIOS Montgomery multiply two ways on this M2
//! Max, at 65,536 threads and 256 chained multiplies each, every variant validated
//! against `num-bigint`:
//!
//! | form | G mul/s |
//! |---|---|
//! | `array<u32, 8>` indexed by a loop variable | 0.516 |
//! | straight-line code, every index a literal | **1.964** |
//!
//! 3.8x, for identical arithmetic. A WGSL function-scope `array<u32, N>` indexed by a loop
//! variable is lowered by naga to an MSL stack array, and a dynamically indexed stack array
//! does not stay in registers on Apple silicon, so every limb access becomes a load. With
//! literal indices it stays in registers. The unrolled 13x20 variants beat the looped 32x8
//! one too, so this is a codegen cliff and not an arithmetic one: **every limb loop in this
//! backend is emitted as straight-line code**, which is only bearable if a program writes it.
//!
//! # The second reason, which is the one that pays off daily
//!
//! [`FR`] and [`FQ`] read `FR_MODULUS`, `FR_N0`, `FQ_MODULUS` and `FQ_N0` out of
//! `g16-gpu-layout` at run time. There is no second copy of the modulus anywhere in this
//! crate. `g16-metal` cannot do that, and pays for it with two tests
//! (`layout.rs::msl_constants_match` and `msm.rs`) that grep MSL source for exact constant
//! lines to catch a copy drifting. `g16-wgpu` needs no such test because there is nothing to
//! drift. The Montgomery constants `R` and `R^2` that the shader also needs are *derived*
//! from the same modulus by [`pow2_mod`] rather than retyped, and
//! `tests::derived_montgomery_constants_match_ark` checks that derivation against `ark-ff`.
//!
//! # What is emitted
//!
//! Per field: `zero`, `one`, `is_zero`, `eq`, `add`, `sub`, `neg`, `mul`, `sqr`, `from_mont`,
//! `to_mont`, all prefixed (`fr_add`, `fq_add`). Then `Fq2 = Fq[u]/(u^2 + 1)` on top of the
//! `Fq` set, since BN254's G2 lives there. Nothing here allocates, branches on a uniform, or
//! touches a storage buffer: it is a prelude that kernel generators concatenate in front of
//! their own entry points.
//!
//! # Carry bounds, stated exactly rather than "comfortably"
//!
//! Every operand of every routine here must be a **canonical** representative, `< m`. Nothing
//! checks it, and the bounds below all rest on it.
//!
//! `mul64`'s high word fits in `u32` with **zero** headroom: `p11 <= 2^32 - 2^17 + 1`, plus two
//! terms of at most `2^16 - 2`, plus `mid >> 16 <= 2`, sums to exactly `2^32 - 1`. The largest
//! value actually attainable is `2^32 - 2`, at `mul64(0xffffffff, 0xffffffff)`, which
//! `tests/field_adversarial.rs::mul64_is_exact_including_at_its_true_maximum` runs on the
//! device. Both limbs really can be all ones: BN254's top modulus limb is `0x30644e72`, so
//! `0x30644e71_ffffffff...ffffffff` is a legal element of either field. No rearrangement of
//! `mul64` is safe without redoing that sum.
//!
//! **The CIOS carry word `c = p.y + cy + cy2` does not wrap, but not for the reason it is
//! tempting to give.** `p.y` reaches `2^32 - 2` and the carry out of `t[j] + p.x + c` reaches
//! **2**, both measured over 20,009 pairs by
//! `tests/field_adversarial.rs::the_cios_carry_word_never_wraps_and_has_exactly_zero_headroom`,
//! so "`p.y` is small, the two carry bits fit" is not an argument. What holds is the joint
//! bound: one inner step computes `a[j]*b_i + t[j] + c` with all of `a[j]`, `b_i`, `t[j]`, `c`
//! at most `2^32 - 1`, so the whole quantity is at most
//! `(2^32-1)^2 + 2*(2^32-1) = 2^64 - 1`. That is a 64-bit value whose low word is the new
//! `t[j]` and whose high word is the new `c`, so `c <= 2^32 - 1` by construction, inductively,
//! from `c = 0`. The same sum bounds the second inner loop with `m` in place of `a`. The
//! measured maximum of `c` is exactly `0xffffffff`, so the headroom is zero rather than merely
//! thin.
//!
//! CIOS needs `s + 2 = 10` accumulator words. At the start of outer round `i` the running
//! value satisfies `T < 2m`; the round forms `T + a*b_i + m*q_i < 2m*(1 + 2^32) < 2^288`,
//! which is nine words. `t9` is the tenth: it catches the carry out of the ninth word during
//! the *first* inner loop, and the second inner loop's own carry out of the ninth word is then
//! added to it to form the ninth word after the round's shift by `2^32`, which restores
//! `T < 2m`. That is the Koc-Acar-Kaliski bound and it is why `Field::mul_cios32` emits
//! `t0 .. t9`. Since `m < 2^255` is asserted before emission, `2m - 1 < 2^256` and the ninth
//! word is zero at the end, which is what makes one conditional subtraction the right
//! reduction and not just a cheap one.

use std::fmt::Write as _;

use g16_gpu_layout::{FQ_MODULUS, FQ_N0, FR_MODULUS, FR_N0, LIMBS};

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

/// Device limb layout, and with it the multiply algorithm.
///
/// Two members, and only one of them is wired into the backend. The other exists so that the
/// browser re-measure (Chrome uses Tint, not naga, and other platforms emit HLSL or SPIR-V)
/// is a config change rather than a kernel rewrite, which is the mitigation
/// this design commits to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Variant {
    /// 8 limbs of 32 bits, `R = 2^256`, CIOS with the 32x32 product emulated by four 16x16
    /// products. Byte-identical to `g16_gpu_layout::PackedFr` and `PackedFq`, so the base
    /// vectors this backend uploads are the same bytes `g16-metal` uploads.
    ///
    /// Measured at 1.964 G mul/s and 32 bytes per element, against 1.492 and 80 for the best
    /// small-limb form. Faster *and* smaller, which is why the literature's recommendation is
    /// rejected here.
    #[default]
    Cios32Unrolled,
    /// 20 limbs of 13 bits, `R = 2^260`, the mitschabaude / ICME "no carry in the inner loop"
    /// product as `msm-webgpu` ships it. Measured at 1.492 G mul/s and 80 bytes per element.
    ///
    /// **Its wire format is not `PackedFr`.** `R` is `2^260`, not `2^256`, so a flip to this
    /// variant needs a host-side repack of every scalar, every base coordinate and the
    /// witness, and that repack does not exist. Benchmark only until it does.
    NoCarry13x20,
}

impl Variant {
    /// Limbs per field element.
    pub const fn limbs(self) -> usize {
        match self {
            Self::Cios32Unrolled => 8,
            Self::NoCarry13x20 => 20,
        }
    }

    /// Value bits per limb. Each limb still occupies a whole `u32` on the device.
    pub const fn limb_bits(self) -> u32 {
        match self {
            Self::Cios32Unrolled => 32,
            Self::NoCarry13x20 => 13,
        }
    }

    /// Device bytes per field element: 32 for the default, 80 for the 13-bit form. The 2.5x
    /// is why the default wins even where the speeds are close, because a 2^18 domain vector
    /// is 8 MB against 20 MB and the Floor storage binding is 128 MiB.
    pub const fn bytes_per_elem(self) -> usize {
        self.limbs() * 4
    }

    /// Whether the emitted module needs [`MUL64`] in scope. Only the 32-bit form does; the
    /// 13-bit form's whole point is that a limb product fits in a `u32` unaided.
    pub const fn needs_mul64(self) -> bool {
        matches!(self, Self::Cios32Unrolled)
    }
}

// ---------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------

/// One prime field, exactly as much of it as the generator needs.
pub struct Field {
    /// Function-name prefix: `fr_add`, `fq_add`.
    pub lower: &'static str,
    /// Constant-name prefix: `FR_M0`, `FQ_M0`.
    pub upper: &'static str,
    /// WGSL type name for one element.
    pub ty: &'static str,
    /// Modulus, little-endian 32-bit limbs, straight from `g16-gpu-layout`.
    pub modulus: [u32; LIMBS],
    /// `-m^{-1} mod 2^32`, straight from `g16-gpu-layout`. Cross-checked against
    /// [`neg_inv`]'s own derivation every time a 32-bit module is emitted.
    pub n0: u32,
}

/// BN254's scalar field. Witness, NTT, `H = A*B - C`, MSM digits.
pub const FR: Field = Field {
    lower: "fr",
    upper: "FR",
    ty: "Fr",
    modulus: FR_MODULUS,
    n0: FR_N0,
};

/// BN254's base field. G1 coordinates, and the ground field of `Fq2` for G2.
pub const FQ: Field = Field {
    lower: "fq",
    upper: "FQ",
    ty: "Fq",
    modulus: FQ_MODULUS,
    n0: FQ_N0,
};

// ---------------------------------------------------------------------------
// The shared 64-bit product
// ---------------------------------------------------------------------------

/// `mul64`, emitted once per module and shared by every field.
///
/// WGSL has no 64-bit integer, no `mulExtended` and no `mulhi`, and gpuweb#1565 asking for
/// them has been open since March 2021 and was deferred "Post V1". Four 16x16 products plus a
/// recombination is the whole of the price of that, and it is most of why this backend runs at
/// 44% of the Metal one (1.964 against 4.51 G mul/s on the same GPU).
pub const MUL64: &str = r#"
// Exact 32x32 -> 64 product, low word in .x and high word in .y. No u64 anywhere.
//
// Zero headroom, so do not rearrange without redoing the sum: each partial product is at
// most (2^16-1)^2 = 2^32 - 2^17 + 1; mid <= 3*2^16 - 4; and
// hi <= (2^32 - 2^17 + 1) + 2*(2^16 - 2) + 2 = 2^32 - 1 exactly.
fn mul64(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((mid << 16u) | (p00 & 0xffffu),
                     p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}
"#;

/// `Fq2 = Fq[u]/(u^2 + 1)`, the field BN254's G2 lives over.
///
/// Static text, because it is written entirely in terms of `fq_*` and so has no limb loop to
/// unroll. Ported line for line from `g16-metal/src/shaders/msm.metal:236-313`, including
/// Karatsuba (3 `Fq` multiplies rather than 4) and the `(a0+a1)(a0-a1)` squaring (2 rather
/// than 3), so the two backends compute G2 the same way and a mismatch is a real bug rather
/// than a different formula.
pub const FQ2_OPS: &str = r#"
struct Fq2 { c0: Fq, c1: Fq }

fn fq2_zero() -> Fq2 { return Fq2(fq_zero(), fq_zero()); }
fn fq2_one() -> Fq2 { return Fq2(fq_one(), fq_zero()); }
fn fq2_is_zero(a: Fq2) -> bool { return fq_is_zero(a.c0) && fq_is_zero(a.c1); }
fn fq2_eq(a: Fq2, b: Fq2) -> bool { return fq_eq(a.c0, b.c0) && fq_eq(a.c1, b.c1); }
fn fq2_add(a: Fq2, b: Fq2) -> Fq2 { return Fq2(fq_add(a.c0, b.c0), fq_add(a.c1, b.c1)); }
fn fq2_sub(a: Fq2, b: Fq2) -> Fq2 { return Fq2(fq_sub(a.c0, b.c0), fq_sub(a.c1, b.c1)); }
fn fq2_neg(a: Fq2) -> Fq2 { return Fq2(fq_neg(a.c0), fq_neg(a.c1)); }

// (a0 + a1 u)(b0 + b1 u) = (a0 b0 - a1 b1) + (a0 b1 + a1 b0) u, with the cross term taken as
// (a0 + a1)(b0 + b1) - a0 b0 - a1 b1 so it costs one multiply instead of two.
fn fq2_mul(a: Fq2, b: Fq2) -> Fq2 {
    let v0 = fq_mul(a.c0, b.c0);
    let v1 = fq_mul(a.c1, b.c1);
    let cross = fq_sub(fq_sub(fq_mul(fq_add(a.c0, a.c1), fq_add(b.c0, b.c1)), v0), v1);
    return Fq2(fq_sub(v0, v1), cross);
}

// (a0 + a1 u)^2 = (a0 + a1)(a0 - a1) + 2 a0 a1 u.
fn fq2_sqr(a: Fq2) -> Fq2 {
    let t0 = fq_add(a.c0, a.c1);
    let t1 = fq_sub(a.c0, a.c1);
    let t2 = fq_mul(a.c0, a.c1);
    return Fq2(fq_mul(t0, t1), fq_add(t2, t2));
}
"#;

/// The whole field prelude: `mul64` if the variant needs it, then `Fr`, then `Fq`, then `Fq2`.
///
/// This is what a kernel generator puts in front of its own entry points. It is deliberately
/// one string and not a shader module: the sweep measured 72 ms to build a module plus
/// pipeline for 24 KiB of unrolled 32-bit multiply, and an earlier run recorded 129 s for one
/// monolithic shader that inlined a 20-limb unrolled multiply at a dozen sites, so shaders stay
/// small and separate and the pipelines get built once in `prepare`.
pub fn field_module(v: Variant) -> String {
    let mut s = format!(
        "// Generated by g16-wgpu::gen::field, variant {v:?}, {} limbs of {} bits, {} bytes\n\
         // per element. Constants come from g16-gpu-layout at run time; do not edit by hand.\n",
        v.limbs(),
        v.limb_bits(),
        v.bytes_per_elem()
    );
    if v.needs_mul64() {
        s.push_str(MUL64);
    }
    s.push_str(&FR.ops(v));
    s.push_str(&FQ.ops(v));
    s.push_str(FQ2_OPS);
    s
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

impl Field {
    /// Every operation for this field, in the given layout.
    ///
    /// Assumes [`MUL64`] is already in scope when `v.needs_mul64()`.
    pub fn ops(&self, v: Variant) -> String {
        let (n, b) = (v.limbs(), v.limb_bits());
        let ty = self.ty;
        let up = self.upper;

        // Derived, not retyped. `neg_inv` at 32 bits must reproduce the constant
        // g16-gpu-layout ships, which makes each a check on the other; the 13-bit n0 has no
        // published twin to check against, so it leans on this agreement at 32.
        let n0 = neg_inv(&self.modulus, b);
        if b == 32 {
            assert_eq!(
                n0, self.n0,
                "derived -m^-1 mod 2^32 disagrees with g16_gpu_layout::{up}_N0"
            );
        }
        // One conditional subtraction at the end of a Montgomery multiply is only enough
        // because the CIOS intermediate is below 2m, and that in turn needs m < 2^255 so the
        // ninth accumulator word is provably zero. Both BN254 moduli have top limb 0x3064e472.
        assert!(
            self.modulus[LIMBS - 1] < 0x8000_0000,
            "modulus is at least 2^255; the single conditional subtraction below is not enough"
        );

        let mods = to_limbs(&self.modulus, b, n);
        let r = to_limbs(&pow2_mod(&self.modulus, b * n as u32), b, n);
        let r2 = to_limbs(&pow2_mod(&self.modulus, 2 * b * n as u32), b, n);

        let mut s = String::new();
        writeln!(s, "\nalias {ty} = array<u32, {n}>;").unwrap();
        writeln!(s, "const {up}_NP: u32 = {n0}u;").unwrap();
        for (i, m) in mods.iter().enumerate() {
            writeln!(s, "const {up}_M{i}: u32 = {m}u;").unwrap();
        }
        writeln!(s, "const {up}_R = {};", ctor(ty, &r)).unwrap();
        writeln!(s, "const {up}_R2 = {};", ctor(ty, &r2)).unwrap();
        let mut one = vec![0u32; n];
        one[0] = 1;
        writeln!(s, "const {up}_STD_ONE = {};", ctor(ty, &one)).unwrap();
        writeln!(s, "const {up}_ZERO = {};", ctor(ty, &vec![0u32; n])).unwrap();

        let lo = self.lower;
        writeln!(s, "\nfn {lo}_zero() -> {ty} {{ return {up}_ZERO; }}").unwrap();
        writeln!(s, "fn {lo}_one() -> {ty} {{ return {up}_R; }}").unwrap();

        let ors = (0..n)
            .map(|i| format!("a[{i}]"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            s,
            "fn {lo}_is_zero(a: {ty}) -> bool {{ return ({ors}) == 0u; }}"
        )
        .unwrap();
        let xors = (0..n)
            .map(|i| format!("(a[{i}] ^ b[{i}])"))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            s,
            "fn {lo}_eq(a: {ty}, b: {ty}) -> bool {{ return ({xors}) == 0u; }}"
        )
        .unwrap();

        match v {
            Variant::Cios32Unrolled => {
                self.add_32(&mut s, n);
                self.sub_32(&mut s, n);
                self.mul_cios32(&mut s, n);
            }
            Variant::NoCarry13x20 => {
                self.add_small(&mut s, n, b);
                self.sub_small(&mut s, n, b);
                self.mul_nocarry(&mut s, n, b);
            }
        }

        writeln!(
            s,
            "\nfn {lo}_neg(a: {ty}) -> {ty} {{ return {lo}_sub({up}_ZERO, a); }}"
        )
        .unwrap();
        writeln!(
            s,
            "fn {lo}_sqr(a: {ty}) -> {ty} {{ return {lo}_mul(a, a); }}"
        )
        .unwrap();
        // A Montgomery multiply by the integer 1 is exactly a Montgomery reduction, and by
        // R^2 is exactly the lift. Neither needs its own routine.
        writeln!(
            s,
            "fn {lo}_from_mont(a: {ty}) -> {ty} {{ return {lo}_mul(a, {up}_STD_ONE); }}"
        )
        .unwrap();
        writeln!(
            s,
            "fn {lo}_to_mont(a: {ty}) -> {ty} {{ return {lo}_mul(a, {up}_R2); }}"
        )
        .unwrap();
        s
    }

    /// Returns a field element built limb by limb, never as an array value constructor.
    ///
    /// # Why this is not `return Fr(x0, x1, ..)`
    ///
    /// It was, and it made every proof this backend produced in Safari 26.6 garbage. WebKit
    /// translates WGSL to MSL itself, and it emits a WGSL array value constructor as a Metal
    /// `array<unsigned, 8>(a, b, ..)`. `metal::array` is an aggregate with no such
    /// constructor, so the Metal compiler answers "no matching constructor for
    /// initialization of 'array<unsigned int, 8>'" and every pipeline built from the module
    /// is invalid. WebKit does not have the same problem with a *constant*: `const FR_R =
    /// Fr(..)` comes out as `array<unsigned, 8>{..}`, with braces, which compiles. Only the
    /// runtime form is affected, and only these six functions ever used it.
    ///
    /// Nothing downstream notices. An invalid pipeline invalidates the command buffer that
    /// set it, an invalid command buffer makes `submit` a no-op, and the prover reads the
    /// buffers back unwritten: a 3 ms proof, in Chrome's 290, that snarkjs rejects.
    /// `getCompilationInfo()` is empty, because the WGSL is valid WGSL; the error is raised
    /// against `createComputePipeline` and is only visible inside an error scope. See
    /// `crate::device::WgpuBackend::scoped`, which is what now puts one there.
    ///
    /// The emitted form costs nothing: `var r: Fr;` is a Metal `array<unsigned, 8> r { };`
    /// and every store below has a literal index, so this is not the loop-variable indexing
    /// measured at 3.8x.
    ///
    /// `tests/wgsl_static.rs::no_generated_module_constructs_an_array_by_value` is the
    /// regression guard, and it reads the generated text rather than trusting this comment.
    fn ret_limbs(&self, s: &mut String, limbs: &[String]) {
        let ty = self.ty;
        writeln!(s, "    var r: {ty};").unwrap();
        for (j, e) in limbs.iter().enumerate() {
            writeln!(s, "    r[{j}] = {e};").unwrap();
        }
        writeln!(s, "    return r;").unwrap();
    }

    /// `(a + b) mod m` for 32-bit limbs. Carry emulated with `select`, since WGSL has no
    /// `uaddCarry`: the carry out of `s = x + y` is `select(0u, 1u, s < x)`.
    fn add_32(&self, s: &mut String, n: usize) {
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        writeln!(s, "\nfn {lo}_add(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
        writeln!(s, "    let s0 = a[0] + b[0];").unwrap();
        writeln!(s, "    var c: u32 = select(0u, 1u, s0 < a[0]);").unwrap();
        for j in 1..n {
            writeln!(s, "    let x{j} = a[{j}] + b[{j}]; let s{j} = x{j} + c;").unwrap();
            writeln!(
                s,
                "    c = select(0u, 1u, x{j} < a[{j}]) + select(0u, 1u, s{j} < x{j});"
            )
            .unwrap();
        }
        // For BN254 `c` is provably zero (a, b < m < 2^254 so a + b < 2^255), but it is fed
        // to the take test anyway. Unlike the multiply's ninth word, this one costs nothing
        // to honour and makes the routine correct for any modulus below 2^256: a + b < 2m
        // always, so one conditional subtraction is always sufficient.
        writeln!(s, "    let d0 = s0 - {up}_M0;").unwrap();
        writeln!(s, "    var borrow: u32 = select(0u, 1u, s0 < {up}_M0);").unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let e{j} = s{j} - {up}_M{j}; let d{j} = e{j} - borrow;"
            )
            .unwrap();
            writeln!(
                s,
                "    borrow = select(0u, 1u, s{j} < {up}_M{j}) + select(0u, 1u, e{j} < borrow);"
            )
            .unwrap();
        }
        writeln!(s, "    let take = (c != 0u) || (borrow == 0u);").unwrap();
        let limbs: Vec<String> = (0..n)
            .map(|j| format!("select(s{j}, d{j}, take)"))
            .collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }

    /// `(a - b) mod m` for 32-bit limbs. The modulus is masked in rather than added under an
    /// `if`, so there is no branch and no divergence.
    fn sub_32(&self, s: &mut String, n: usize) {
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        writeln!(s, "\nfn {lo}_sub(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
        writeln!(s, "    let d0 = a[0] - b[0];").unwrap();
        writeln!(s, "    var borrow: u32 = select(0u, 1u, a[0] < b[0]);").unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let e{j} = a[{j}] - b[{j}]; let d{j} = e{j} - borrow;"
            )
            .unwrap();
            writeln!(
                s,
                "    borrow = select(0u, 1u, a[{j}] < b[{j}]) + select(0u, 1u, e{j} < borrow);"
            )
            .unwrap();
        }
        writeln!(s, "    let mask = 0u - borrow;").unwrap();
        writeln!(s, "    let t0 = d0 + ({up}_M0 & mask);").unwrap();
        writeln!(s, "    var c: u32 = select(0u, 1u, t0 < d0);").unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let u{j} = d{j} + ({up}_M{j} & mask); let t{j} = u{j} + c;"
            )
            .unwrap();
            writeln!(
                s,
                "    c = select(0u, 1u, u{j} < d{j}) + select(0u, 1u, t{j} < u{j});"
            )
            .unwrap();
        }
        let limbs: Vec<String> = (0..n).map(|j| format!("t{j}")).collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }

    /// CIOS Montgomery product, 8 limbs of 32 bits, fully unrolled into ten named scalars.
    ///
    /// Lifted from the limb-width sweep's `cios32_unrolled`, which measured
    /// 1.964 G mul/s and was validated against `num-bigint` over 256 chained multiplies on
    /// 65,536 random inputs. The accumulator is `t0 .. t9`, never `array<u32, 10>`: see the
    /// module docs for what indexing an array by a loop variable costs.
    fn mul_cios32(&self, s: &mut String, n: usize) {
        assert_eq!(n, 8, "the CIOS emitter is written for 8 limbs");
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        writeln!(s, "\nfn {lo}_mul(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
        for j in 0..=9 {
            writeln!(s, "    var t{j}: u32 = 0u;").unwrap();
        }
        for i in 0..8 {
            writeln!(s, "    {{").unwrap();
            writeln!(s, "    let bi = b[{i}];").unwrap();
            writeln!(s, "    var c: u32 = 0u;").unwrap();
            // t += a * b[i]. The carry word `c = p.y + cy + cy2` cannot wrap, because
            // a[j]*b_i + t[j] + c <= (2^32-1)^2 + 2*(2^32-1) = 2^64 - 1 and (c, t[j]) is
            // exactly that 64-bit value. Not because p.y is small: it reaches 2^32 - 2, and
            // cy + cy2 reaches 2. See the module docs and
            // tests/field_adversarial.rs, which measures both maxima.
            for j in 0..8 {
                writeln!(
                    s,
                    "    {{ let p = mul64(a[{j}], bi); let s = t{j} + p.x; \
                     let cy = select(0u, 1u, s < t{j}); let s2 = s + c; \
                     let cy2 = select(0u, 1u, s2 < s); t{j} = s2; c = p.y + cy + cy2; }}"
                )
                .unwrap();
            }
            writeln!(
                s,
                "    {{ let s = t8 + c; t9 = select(0u, 1u, s < t8); t8 = s; }}"
            )
            .unwrap();
            // m = t0 * (-m^-1) mod 2^32, chosen so t + m*modulus has a zero low limb; the
            // second pass then shifts that zero limb off, which is the division by 2^32 that
            // makes this Montgomery rather than schoolbook.
            writeln!(s, "    let m = t0 * {up}_NP;").unwrap();
            writeln!(
                s,
                "    {{ let p = mul64(m, {up}_M0); let s = t0 + p.x; \
                 c = p.y + select(0u, 1u, s < t0); }}"
            )
            .unwrap();
            for j in 1..8 {
                writeln!(
                    s,
                    "    {{ let p = mul64(m, {up}_M{j}); let s = t{j} + p.x; \
                     let cy = select(0u, 1u, s < t{j}); let s2 = s + c; \
                     let cy2 = select(0u, 1u, s2 < s); t{} = s2; c = p.y + cy + cy2; }}",
                    j - 1
                )
                .unwrap();
            }
            writeln!(
                s,
                "    {{ let s = t8 + c; t7 = s; t8 = t9 + select(0u, 1u, s < t8); }}"
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
        // t8 is zero here and is therefore not tested: the round bound gives T < 2m (strictly,
        // T <= 2m - 1), and the `modulus < 2^255` assertion above makes 2m - 1 < 2^256.
        // Testing it would be worse than useless, because if it could ever be nonzero one
        // subtraction would not be enough and the honest fix would be a different reduction,
        // not an extra `||`. The host mirror in tests/field_adversarial.rs asserts t8 == 0 and
        // T < 2m on every pair it runs, including the all-ones operands.
        writeln!(s, "    var borrow: u32 = 0u;").unwrap();
        for j in 0..8 {
            writeln!(
                s,
                "    let e{j} = t{j} - {up}_M{j}; let f{j} = e{j} - borrow; \
                 borrow = select(0u, 1u, t{j} < {up}_M{j}) + select(0u, 1u, e{j} < borrow);"
            )
            .unwrap();
        }
        // `borrow == 0` means t >= m, so take the reduced form. Note `>=`, not `>`: research
        // file 02 flags ICME's `conditional_reduce` for using a strict `bigint_gt`. See
        // `tests/field.rs::the_conditional_subtraction_triggers_at_the_boundary` and its `Fq`
        // twin in `tests/field_adversarial.rs` for what that bug can and cannot reach. The
        // multiply can never produce t == m exactly, so the strict form is only wrong here at
        // t == m + 1 and above; `add` is where `>` really breaks, and it is tested there too.
        writeln!(s, "    let take = borrow == 0u;").unwrap();
        let limbs: Vec<String> = (0..8)
            .map(|j| format!("select(t{j}, f{j}, take)"))
            .collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }

    /// `(a + b) mod m` for sub-32-bit limbs, where a limb sum cannot overflow a `u32` at all
    /// and the carry is just a shift.
    fn add_small(&self, s: &mut String, n: usize, b: u32) {
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        let mask = (1u32 << b) - 1;
        writeln!(s, "\nfn {lo}_add(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
        writeln!(
            s,
            "    let x0 = a[0] + b[0]; let s0 = x0 & {mask}u; var c: u32 = x0 >> {b}u;"
        )
        .unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let x{j} = a[{j}] + b[{j}] + c; let s{j} = x{j} & {mask}u; c = x{j} >> {b}u;"
            )
            .unwrap();
        }
        let hi = 1u32 << b;
        writeln!(
            s,
            "    let y0 = s0 + {hi}u - {up}_M0; let d0 = y0 & {mask}u; var borrow: u32 = 1u - (y0 >> {b}u);"
        )
        .unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let y{j} = s{j} + {hi}u - {up}_M{j} - borrow; let d{j} = y{j} & {mask}u; \
                 borrow = 1u - (y{j} >> {b}u);"
            )
            .unwrap();
        }
        writeln!(s, "    let take = (c != 0u) || (borrow == 0u);").unwrap();
        let limbs: Vec<String> = (0..n)
            .map(|j| format!("select(s{j}, d{j}, take)"))
            .collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }

    /// `(a - b) mod m` for sub-32-bit limbs.
    fn sub_small(&self, s: &mut String, n: usize, b: u32) {
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        let (mask, hi) = ((1u32 << b) - 1, 1u32 << b);
        writeln!(s, "\nfn {lo}_sub(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
        writeln!(
            s,
            "    let x0 = a[0] + {hi}u - b[0]; let d0 = x0 & {mask}u; var borrow: u32 = 1u - (x0 >> {b}u);"
        )
        .unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let x{j} = a[{j}] + {hi}u - b[{j}] - borrow; let d{j} = x{j} & {mask}u; \
                 borrow = 1u - (x{j} >> {b}u);"
            )
            .unwrap();
        }
        writeln!(s, "    let addback = 0u - borrow;").unwrap();
        writeln!(
            s,
            "    let z0 = d0 + ({up}_M0 & addback); let t0 = z0 & {mask}u; var c: u32 = z0 >> {b}u;"
        )
        .unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    let z{j} = d{j} + ({up}_M{j} & addback) + c; let t{j} = z{j} & {mask}u; \
                 c = z{j} >> {b}u;"
            )
            .unwrap();
        }
        let limbs: Vec<String> = (0..n).map(|j| format!("t{j}")).collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }

    /// The mitschabaude / ICME "no carry in the inner loop" Montgomery product, unrolled.
    ///
    /// Lifted from the sweep's `mitschabaude_unrolled`, itself from
    /// `msm-webgpu/src/cuzk/wgsl/montgomery/mont_pro_product.template.wgsl`. The inner loop is
    /// `s[j-1] = s[j] + x_i*y_j + q_i*p_j` with no shift and no mask, and one carry pass at the
    /// very end. Sound only while `2B + log2(2N) < 32`, which is what caps B at 13 and is
    /// asserted here rather than assumed.
    fn mul_nocarry(&self, s: &mut String, n: usize, b: u32) {
        assert!(
            2.0 * b as f64 + ((2 * n) as f64).log2() < 32.0,
            "no-carry accumulator bound violated at B = {b}, N = {n}"
        );
        let (lo, up, ty) = (self.lower, self.upper, self.ty);
        let mask = (1u32 << b) - 1;
        writeln!(s, "\nfn {lo}_mul(x: {ty}, y: {ty}) -> {ty} {{").unwrap();
        for j in 0..n {
            writeln!(s, "    var s{j}: u32 = 0u;").unwrap();
        }
        for i in 0..n {
            writeln!(s, "    {{").unwrap();
            writeln!(s, "    let xi = x[{i}];").unwrap();
            writeln!(s, "    let t = s0 + xi * y[0];").unwrap();
            writeln!(s, "    let qi = ({up}_NP * (t & {mask}u)) & {mask}u;").unwrap();
            writeln!(s, "    let c = (t + qi * {up}_M0) >> {b}u;").unwrap();
            writeln!(s, "    s0 = s1 + xi * y[1] + qi * {up}_M1 + c;").unwrap();
            for j in 2..n {
                writeln!(s, "    s{} = s{j} + xi * y[{j}] + qi * {up}_M{j};", j - 1).unwrap();
            }
            writeln!(
                s,
                "    s{} = xi * y[{}] + qi * {up}_M{};",
                n - 2,
                n - 1,
                n - 1
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    var c: u32 = 0u;").unwrap();
        for j in 0..n {
            writeln!(
                s,
                "    {{ let v = s{j} + c; c = v >> {b}u; s{j} = v & {mask}u; }}"
            )
            .unwrap();
        }
        let hi = 1u32 << b;
        writeln!(s, "    var borrow: u32 = 0u;").unwrap();
        for j in 0..n {
            writeln!(
                s,
                "    let d{j} = s{j} + {hi}u - {up}_M{j} - borrow; borrow = 1u - (d{j} >> {b}u);"
            )
            .unwrap();
        }
        writeln!(s, "    let take = borrow == 0u;").unwrap();
        let limbs: Vec<String> = (0..n)
            .map(|j| format!("select(s{j}, d{j} & {mask}u, take)"))
            .collect();
        self.ret_limbs(s, &limbs);
        writeln!(s, "}}").unwrap();
    }
}

/// A constant of the field type, written through the **alias** rather than through a bare
/// `array<u32, N>(...)` literal.
///
/// That is not cosmetic. naga 30 emits a named MSL struct for a WGSL alias and a separate
/// anonymous one (`type_2`) for a bare array constructor, then refuses to convert between
/// them: `no viable conversion from returned value of type 'type_2' to function return type
/// 'Fr'`. Both spellings work on their own; mixing them does not compile. Probed on this M2
/// Max with wgpu 30.0.1.
fn ctor(ty: &str, v: &[u32]) -> String {
    let items = v
        .iter()
        .map(|x| format!("{x}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{ty}({items})")
}

// ---------------------------------------------------------------------------
// Host-side constant derivation
//
// Everything below exists so that the only field constants typed by a human in this project
// are the two moduli and the two n0 values in g16-gpu-layout, which are themselves checked
// against ark-ff there. R and R^2 are computed from the modulus, and `tests` checks that
// computation against ark-ff for both fields.
// ---------------------------------------------------------------------------

/// `a >= b` on little-endian 32-bit limbs.
fn ge(a: &[u32; LIMBS], b: &[u32; LIMBS]) -> bool {
    for i in (0..LIMBS).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// `a -= b`, wrapping. Callers only ever use it where `a >= b` modulo the carry bit.
fn sub_assign(a: &mut [u32; LIMBS], b: &[u32; LIMBS]) {
    let mut borrow = 0u64;
    for i in 0..LIMBS {
        let d = (a[i] as u64).wrapping_sub(b[i] as u64).wrapping_sub(borrow);
        a[i] = d as u32;
        borrow = (d >> 63) & 1;
    }
}

/// `x = 2x mod m`, given `x < m`.
///
/// The doubling can carry out of the top limb only when `m > 2^255`; either way `2x < 2m`, so
/// one subtraction lands back in `[0, m)` and the borrow cancels the carry.
fn dbl_mod(x: &mut [u32; LIMBS], m: &[u32; LIMBS]) {
    let mut carry = 0u32;
    for w in x.iter_mut() {
        let next = *w >> 31;
        *w = (*w << 1) | carry;
        carry = next;
    }
    if carry != 0 || ge(x, m) {
        sub_assign(x, m);
    }
}

/// `2^k mod m`, by `k` doublings from 1. `k` is at most 520 here, so this is microseconds.
pub fn pow2_mod(m: &[u32; LIMBS], k: u32) -> [u32; LIMBS] {
    let mut x = [0u32; LIMBS];
    x[0] = 1;
    for _ in 0..k {
        dbl_mod(&mut x, m);
    }
    x
}

/// Re-limb a 256-bit little-endian value into `n` limbs of `bits` bits each.
///
/// Bit at a time, which is slow and obviously correct. It runs once per shader build.
fn to_limbs(x: &[u32; LIMBS], bits: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let mut out = 0u32;
            for k in 0..bits {
                let bit = i as u32 * bits + k;
                let w = (bit / 32) as usize;
                if w < LIMBS && (x[w] >> (bit % 32)) & 1 == 1 {
                    out |= 1 << k;
                }
            }
            out
        })
        .collect()
}

/// `-m^{-1} mod 2^bits`, the CIOS per-limb reduction multiplier.
///
/// Newton on the odd modulus: `inv <- inv * (2 - m*inv)` doubles the number of correct bits
/// each step, so `bits` steps is far more than enough and costs nothing.
pub fn neg_inv(m: &[u32; LIMBS], bits: u32) -> u32 {
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let m_low = m[0] & mask;
    let mut inv: u32 = 1;
    for _ in 0..bits {
        let t = m_low.wrapping_mul(inv) & mask;
        inv = inv.wrapping_mul(2u32.wrapping_sub(t)) & mask;
    }
    0u32.wrapping_sub(inv) & mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{BigInt, Field as _, PrimeField};
    use g16_field::{Fq, Fr};

    fn split(limbs: [u64; 4]) -> [u32; LIMBS] {
        let mut out = [0u32; LIMBS];
        for (i, l) in limbs.iter().enumerate() {
            out[2 * i] = *l as u32;
            out[2 * i + 1] = (*l >> 32) as u32;
        }
        out
    }

    fn join(limbs: [u32; LIMBS]) -> BigInt<4> {
        let mut out = [0u64; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
        }
        BigInt::new(out)
    }

    /// The point of the whole derivation: no human types R or R^2 into this crate, and the
    /// arithmetic that produces them is checked against arkworks rather than against a blog
    /// post.
    ///
    /// arkworks stores an element in Montgomery form, so the limbs of `ONE` *are* `R mod m`,
    /// and the limbs of the element whose standard value is `R mod m` are `R^2 mod m`. As a
    /// third opinion the Fr values also match the `FR_R` and `FR_R2` arrays hand-written in
    /// `g16-metal/src/shaders/bn254_fr.metal:73-76`, which were derived independently.
    #[test]
    fn derived_montgomery_constants_match_ark() {
        let r_fr = pow2_mod(&FR_MODULUS, 256);
        assert_eq!(r_fr, split(Fr::ONE.0 .0), "Fr: R mod r");
        let r2_fr = pow2_mod(&FR_MODULUS, 512);
        let lift_fr = Fr::from_bigint(join(r_fr)).unwrap();
        assert_eq!(r2_fr, split(lift_fr.0 .0), "Fr: R^2 mod r");

        let r_fq = pow2_mod(&FQ_MODULUS, 256);
        assert_eq!(r_fq, split(Fq::ONE.0 .0), "Fq: R mod q");
        let r2_fq = pow2_mod(&FQ_MODULUS, 512);
        let lift_fq = Fq::from_bigint(join(r_fq)).unwrap();
        assert_eq!(r2_fq, split(lift_fq.0 .0), "Fq: R^2 mod q");

        // The MSL copies, so a drift in either direction shows up here too.
        assert_eq!(
            r_fr,
            [
                0x4fff_fffb,
                0xac96_341c,
                0x9f60_cd29,
                0x36fc_7695,
                0x7879_462e,
                0x666e_a36f,
                0x9a07_df2f,
                0x0e0a_77c1
            ]
        );
        assert_eq!(
            r2_fr,
            [
                0xae21_6da7,
                0x1bb8_e645,
                0xe35c_59e3,
                0x53fe_3ab1,
                0x53bb_8085,
                0x8c49_833d,
                0x7f4e_44a5,
                0x0216_d0b1
            ]
        );
    }

    /// `neg_inv` at 32 bits has to reproduce what `g16-gpu-layout` ships, which is itself
    /// checked against ark-ff there. `Field::ops` asserts this on every emission; this test
    /// makes the failure legible when it happens.
    #[test]
    fn derived_n0_matches_the_layout_crate() {
        assert_eq!(neg_inv(&FR_MODULUS, 32), FR_N0);
        assert_eq!(neg_inv(&FQ_MODULUS, 32), FQ_N0);
        // 13-bit n0 has no published twin. Check the defining property instead:
        // m * n0 == -1 mod 2^13.
        for m in [FR_MODULUS, FQ_MODULUS] {
            let n0 = neg_inv(&m, 13);
            assert_eq!((m[0] & 8191).wrapping_mul(n0) & 8191, 8191);
        }
    }

    /// Both variants must emit for both fields without tripping a bound assertion, and the
    /// sizes are recorded because shader build cost tracks source size: file 08 measured 18 ms
    /// for 2.4 KiB of looped source against 72 ms for 24 KiB unrolled, and research file 03
    /// records 129 s for a monolithic shader.
    #[test]
    fn the_generated_module_is_the_size_we_think_it_is() {
        for v in [Variant::Cios32Unrolled, Variant::NoCarry13x20] {
            let one = FR.ops(v);
            let all = field_module(v);
            println!(
                "{v:?}: one field {} lines / {} bytes, full module (Fr + Fq + Fq2) {} lines / {} bytes",
                one.lines().count(),
                one.len(),
                all.lines().count(),
                all.len()
            );
            assert!(all.contains("fn fr_mul("));
            assert!(all.contains("fn fq_mul("));
            assert!(all.contains("fn fq2_mul("));
            // A regression here means the emitter changed shape. Update the numbers with the
            // reason, do not widen the window.
            let lines = all.lines().count();
            assert!(
                (300..2600).contains(&lines),
                "{v:?} emitted {lines} lines, which is outside the range this test was written against"
            );
        }
    }

    /// The modulus limbs the shader sees are the ones `g16-gpu-layout` defines, in both
    /// layouts. This is the test that `g16-metal` has to write as a grep over MSL text.
    #[test]
    fn emitted_modulus_limbs_are_the_layout_crates() {
        let src = FR.ops(Variant::Cios32Unrolled);
        for (i, m) in FR_MODULUS.iter().enumerate() {
            assert!(src.contains(&format!("const FR_M{i}: u32 = {m}u;")));
        }
        // And the 13-bit relimbing really is the same integer, checked by reassembling it.
        let small = to_limbs(&FR_MODULUS, 13, 20);
        let mut back = [0u32; LIMBS];
        for (i, l) in small.iter().enumerate() {
            for k in 0..13u32 {
                if (l >> k) & 1 == 1 {
                    let bit = i as u32 * 13 + k;
                    if (bit as usize) < 32 * LIMBS {
                        back[(bit / 32) as usize] |= 1 << (bit % 32);
                    }
                }
            }
        }
        assert_eq!(back, FR_MODULUS);
    }
}
