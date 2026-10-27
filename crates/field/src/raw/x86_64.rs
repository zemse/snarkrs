//! x86_64 assembly for `raw.rs`'s add, sub and multiply. The add and sub are baseline x86_64.
//! The multiply needs bmi2 and adx: a build with both enabled calls it directly, and any
//! other build goes through [`mulx`], so a CPU without them never reaches it.

use super::{FQ_MOD, FQ_N0, FR_MOD, FR_N0};

/// The CPU probe behind [`mulx`]: 0 not yet run, 1 portable multiply, 2 `mulx` multiply.
#[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
static MULX: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Whether [`mont_mul`] runs the assembly. A constant in a `+adx,+bmi2` build. Otherwise one
/// byte compare per multiply, after a probe on first use; `is_x86_feature_detected!` caches
/// too, but costs an atomic load and a bit test per feature.
#[inline(always)]
pub(super) fn mulx() -> bool {
    #[cfg(all(target_feature = "bmi2", target_feature = "adx"))]
    return true;
    #[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
    {
        // `2` first, so the assembly is one compare away.
        let v = MULX.load(core::sync::atomic::Ordering::Relaxed);
        v == 2 || (v == 0 && probe())
    }
}

#[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
#[cold]
#[inline(never)]
fn probe() -> bool {
    let yes = std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("adx");
    // A racing `disable_mulx` must win, so only an unprobed flag is written.
    let _ = MULX.compare_exchange(
        0,
        if yes { 2 } else { 1 },
        core::sync::atomic::Ordering::Relaxed,
        core::sync::atomic::Ordering::Relaxed,
    );
    MULX.load(core::sync::atomic::Ordering::Relaxed) == 2
}

pub(super) fn disable_mulx() {
    #[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
    MULX.store(1, core::sync::atomic::Ordering::Relaxed);
}

/// A modulus and its `n0`, for the multiply to read RIP-relative rather than through a
/// register.
static FQ_PARAMS: [u64; 5] = [FQ_MOD[0], FQ_MOD[1], FQ_MOD[2], FQ_MOD[3], FQ_N0];
static FR_PARAMS: [u64; 5] = [FR_MOD[0], FR_MOD[1], FR_MOD[2], FR_MOD[3], FR_N0];

/// `(a + b) mod m`: one add/adc chain, one sub/sbb chain against `m` on copies, four cmov.
/// About half the latency of LLVM's code for the portable version, which carries each limb
/// through its own sbb/add/adc/sar group and materialises the modulus with movabs.
#[inline(always)]
pub(super) fn add_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let [mut a0, mut a1, mut a2, mut a3] = a;
    // SAFETY: reads the 32 bytes behind `m`; writes registers only; no stack; flags
    // clobbered.
    unsafe {
        core::arch::asm!(
            "add {a0}, {b0}",
            "adc {a1}, {b1}",
            "adc {a2}, {b2}",
            "adc {a3}, {b3}",
            "mov {b0}, {a0}",
            "sub {b0}, qword ptr [{pm}]",
            "mov {b1}, {a1}",
            "sbb {b1}, qword ptr [{pm} + 8]",
            "mov {b2}, {a2}",
            "sbb {b2}, qword ptr [{pm} + 16]",
            "mov {b3}, {a3}",
            "sbb {b3}, qword ptr [{pm} + 24]",
            "cmovnc {a0}, {b0}",
            "cmovnc {a1}, {b1}",
            "cmovnc {a2}, {b2}",
            "cmovnc {a3}, {b3}",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            b0 = inout(reg) b[0] => _,
            b1 = inout(reg) b[1] => _,
            b2 = inout(reg) b[2] => _,
            b3 = inout(reg) b[3] => _,
            pm = in(reg) m.as_ptr(),
            options(pure, readonly, nostack),
        );
    }
    [a0, a1, a2, a3]
}

/// `(a - b) mod m`: the difference, then `m` or zero by cmov on the borrow, added back.
/// The portable version's u128 borrows come out of LLVM as sbb/add/adc/sar per limb.
#[inline(always)]
pub(super) fn sub_mod(a: [u64; 4], b: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    let [mut a0, mut a1, mut a2, mut a3] = a;
    // SAFETY: reads the 32 bytes behind `m`; writes registers only; no stack; flags
    // clobbered.
    unsafe {
        core::arch::asm!(
            "sub {a0}, {b0}",
            "sbb {a1}, {b1}",
            "sbb {a2}, {b2}",
            "sbb {a3}, {b3}",
            "mov {b0:e}, 0",
            "mov {b1:e}, 0",
            "mov {b2:e}, 0",
            "mov {b3:e}, 0",
            "cmovc {b0}, qword ptr [{pm}]",
            "cmovc {b1}, qword ptr [{pm} + 8]",
            "cmovc {b2}, qword ptr [{pm} + 16]",
            "cmovc {b3}, qword ptr [{pm} + 24]",
            "add {a0}, {b0}",
            "adc {a1}, {b1}",
            "adc {a2}, {b2}",
            "adc {a3}, {b3}",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            b0 = inout(reg) b[0] => _,
            b1 = inout(reg) b[1] => _,
            b2 = inout(reg) b[2] => _,
            b3 = inout(reg) b[3] => _,
            pm = in(reg) m.as_ptr(),
            options(pure, readonly, nostack),
        );
    }
    [a0, a1, a2, a3]
}

/// The multiply's body, instantiated once per field so each reads its own static.
macro_rules! mont_mul_asm {
    ($a:expr, $b:expr, $params:path) => {{
        let (r0, r1, r2, r3): (u64, u64, u64, u64);
        // SAFETY: reads the 32 bytes behind `b` and the 40 of the static; writes registers
        // only; no stack; flags clobbered. `mulx` needs bmi2 and `adcx`/`adox` adx, which
        // the build enables or `mulx()` has found on this CPU.
        unsafe {
            core::arch::asm!(
                // Row 0: t = a * b0.
                "mov rdx, qword ptr [{pb}]",
                "xor {lo:e}, {lo:e}",
                "mulx {x1}, {x0}, {a0}",
                "mulx {x2}, {lo}, {a1}",
                "adcx {x1}, {lo}",
                "mulx {x3}, {lo}, {a2}",
                "adcx {x2}, {lo}",
                "mulx {x4}, {lo}, {a3}",
                "adcx {x3}, {lo}",
                "mov edx, 0",
                "adcx {x4}, rdx",
                // Reduce: t = (t + q m) / 2^64, q = x0 * n0. Zeroes x0.
                "mov rdx, {x0}",
                "imul rdx, qword ptr [rip + {p} + 32]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, qword ptr [rip + {p}]",
                "adox {x0}, {lo}",
                "adcx {x1}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 8]",
                "adox {x1}, {lo}",
                "adcx {x2}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 16]",
                "adox {x2}, {lo}",
                "adcx {x3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 24]",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "adox {x4}, {x0}",
                // Row 1: t += a * b1, limbs (x1, x2, x3, x4), x0 the new top.
                "mov rdx, qword ptr [{pb} + 8]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, {a0}",
                "adox {x1}, {lo}",
                "adcx {x2}, {hi}",
                "mulx {hi}, {lo}, {a1}",
                "adox {x2}, {lo}",
                "adcx {x3}, {hi}",
                "mulx {hi}, {lo}, {a2}",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, {a3}",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "mov edx, 0",
                "adox {x0}, rdx",
                "mov rdx, {x1}",
                "imul rdx, qword ptr [rip + {p} + 32]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, qword ptr [rip + {p}]",
                "adox {x1}, {lo}",
                "adcx {x2}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 8]",
                "adox {x2}, {lo}",
                "adcx {x3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 16]",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 24]",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "adox {x0}, {x1}",
                // Row 2: limbs (x2, x3, x4, x0), x1 the new top.
                "mov rdx, qword ptr [{pb} + 16]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, {a0}",
                "adox {x2}, {lo}",
                "adcx {x3}, {hi}",
                "mulx {hi}, {lo}, {a1}",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, {a2}",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "mulx {hi}, {lo}, {a3}",
                "adox {x0}, {lo}",
                "adcx {x1}, {hi}",
                "mov edx, 0",
                "adox {x1}, rdx",
                "mov rdx, {x2}",
                "imul rdx, qword ptr [rip + {p} + 32]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, qword ptr [rip + {p}]",
                "adox {x2}, {lo}",
                "adcx {x3}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 8]",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 16]",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 24]",
                "adox {x0}, {lo}",
                "adcx {x1}, {hi}",
                "adox {x1}, {x2}",
                // Row 3: limbs (x3, x4, x0, x1), x2 the new top.
                "mov rdx, qword ptr [{pb} + 24]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, {a0}",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, {a1}",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "mulx {hi}, {lo}, {a2}",
                "adox {x0}, {lo}",
                "adcx {x1}, {hi}",
                "mulx {hi}, {lo}, {a3}",
                "adox {x1}, {lo}",
                "adcx {x2}, {hi}",
                "mov edx, 0",
                "adox {x2}, rdx",
                "mov rdx, {x3}",
                "imul rdx, qword ptr [rip + {p} + 32]",
                "xor {lo:e}, {lo:e}",
                "mulx {hi}, {lo}, qword ptr [rip + {p}]",
                "adox {x3}, {lo}",
                "adcx {x4}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 8]",
                "adox {x4}, {lo}",
                "adcx {x0}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 16]",
                "adox {x0}, {lo}",
                "adcx {x1}, {hi}",
                "mulx {hi}, {lo}, qword ptr [rip + {p} + 24]",
                "adox {x1}, {lo}",
                "adcx {x2}, {hi}",
                "adox {x2}, {x3}",
                // t is (x4, x0, x1, x2) < 2m: subtract m, keep t if that borrowed.
                "mov {lo}, {x4}",
                "sub {lo}, qword ptr [rip + {p}]",
                "mov {hi}, {x0}",
                "sbb {hi}, qword ptr [rip + {p} + 8]",
                "mov rdx, {x1}",
                "sbb rdx, qword ptr [rip + {p} + 16]",
                "mov {pb}, {x2}",
                "sbb {pb}, qword ptr [rip + {p} + 24]",
                "cmovnc {x4}, {lo}",
                "cmovnc {x0}, {hi}",
                "cmovnc {x1}, rdx",
                "cmovnc {x2}, {pb}",
                a0 = in(reg) $a[0],
                a1 = in(reg) $a[1],
                a2 = in(reg) $a[2],
                a3 = in(reg) $a[3],
                pb = inout(reg) $b.as_ptr() => _,
                p = sym $params,
                x0 = out(reg) r1,
                x1 = out(reg) r2,
                x2 = out(reg) r3,
                x3 = out(reg) _,
                x4 = out(reg) r0,
                lo = out(reg) _,
                hi = out(reg) _,
                out("rdx") _,
                options(pure, readonly, nostack),
            );
        }
        [r0, r1, r2, r3]
    }};
}

/// `mont_mul` on x86_64 with bmi2 and adx: the portable version's CIOS, one row per limb of
/// `b`, each row a product pass and a reduction pass. Every pass is one `mulx` per limb with
/// the low halves summed on the `adox` (OF) chain and the high halves on the `adcx` (CF)
/// chain, so the two carry chains run side by side instead of serialising on CF. LLVM's code
/// for the portable version uses `mulx` but carries everything through one `adc` chain plus
/// `setb`/`movzx` pairs.
///
/// The running value is five limbs in `x0..x4`, renamed rather than moved: a reduction zeroes
/// the bottom limb, which then becomes the next row's top limb, so the result comes out in
/// `(x4, x0, x1, x2)`. Each row keeps `t < 2m` (Montgomery's bound for reduced inputs), so a
/// row's sum stays below `2m * 2^64 < 2^320` and the top limb never carries out; both chains
/// end on it with their last flag, through a zero from `mov edx, 0` (`mov` leaves the flags).
///
/// `a` stays in registers and `b` is read a limb at a time into `rdx`; the modulus and `n0`
/// are read RIP-relative from a static, which leaves the thirteen allocatable registers
/// exactly enough. Passing `a` through memory too was measured slower: the caller copies the
/// limbs with 16-byte moves, and a 16-byte load of two 8-byte stores cannot be forwarded.
///
/// Without `+adx,+bmi2` at compile time this checks [`mulx`] first and falls back to the
/// portable multiply, inlined beside the assembly.
#[inline(always)]
pub(super) fn mont_mul(a: [u64; 4], b: [u64; 4], m: &[u64; 4], n0: u64) -> [u64; 4] {
    #[cfg(not(all(target_feature = "bmi2", target_feature = "adx")))]
    if !mulx() {
        return super::mont_mul_portable(a, b, m, n0);
    }
    // Both arms are constant after inlining, so this compiles to one of the two.
    if *m == FQ_MOD && n0 == FQ_N0 {
        mont_mul_asm!(a, b, FQ_PARAMS)
    } else {
        debug_assert!(*m == FR_MOD && n0 == FR_N0);
        mont_mul_asm!(a, b, FR_PARAMS)
    }
}
