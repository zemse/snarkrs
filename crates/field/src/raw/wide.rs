//! Double-width products and a separate Montgomery reduction on aarch64, for
//! `RawFq2::mul`'s lazy Karatsuba. Each is one asm block: LLVM carries the u128 chains of
//! the portable shape through cset/cinc pairs, which measured slower than the three fused
//! multiplies it replaces (`examples/raw_ops.rs`, RawFq2 mul 49.6 ns against 39.7).

/// `a * b` as eight limbs, a row of `a * b[i]` at a time. Each row is its low halves
/// added in one adds/adcs chain and its high halves, one limb up, in a second.
#[inline(always)]
pub(super) fn mul_wide(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    // SAFETY: register arithmetic only: no memory, no stack; flags clobbered.
    unsafe {
        core::arch::asm!(
            // a * b0 into r0..r4.
            "mul {r0}, {a0}, {b0}",
            "umulh {r1}, {a0}, {b0}",
            "mul {t}, {a1}, {b0}",
            "umulh {r2}, {a1}, {b0}",
            "adds {r1}, {r1}, {t}",
            "mul {t}, {a2}, {b0}",
            "umulh {r3}, {a2}, {b0}",
            "adcs {r2}, {r2}, {t}",
            "mul {t}, {a3}, {b0}",
            "umulh {r4}, {a3}, {b0}",
            "adcs {r3}, {r3}, {t}",
            "adc {r4}, {r4}, xzr",
            // + a * b1 into r1..r5.
            "mul {t}, {a0}, {b1}",
            "adds {r1}, {r1}, {t}",
            "mul {t}, {a1}, {b1}",
            "adcs {r2}, {r2}, {t}",
            "mul {t}, {a2}, {b1}",
            "adcs {r3}, {r3}, {t}",
            "mul {t}, {a3}, {b1}",
            "adcs {r4}, {r4}, {t}",
            "cset {r5}, cs",
            "umulh {t}, {a0}, {b1}",
            "adds {r2}, {r2}, {t}",
            "umulh {t}, {a1}, {b1}",
            "adcs {r3}, {r3}, {t}",
            "umulh {t}, {a2}, {b1}",
            "adcs {r4}, {r4}, {t}",
            "umulh {t}, {a3}, {b1}",
            "adc {r5}, {r5}, {t}",
            // + a * b2 into r2..r6.
            "mul {t}, {a0}, {b2}",
            "adds {r2}, {r2}, {t}",
            "mul {t}, {a1}, {b2}",
            "adcs {r3}, {r3}, {t}",
            "mul {t}, {a2}, {b2}",
            "adcs {r4}, {r4}, {t}",
            "mul {t}, {a3}, {b2}",
            "adcs {r5}, {r5}, {t}",
            "cset {r6}, cs",
            "umulh {t}, {a0}, {b2}",
            "adds {r3}, {r3}, {t}",
            "umulh {t}, {a1}, {b2}",
            "adcs {r4}, {r4}, {t}",
            "umulh {t}, {a2}, {b2}",
            "adcs {r5}, {r5}, {t}",
            "umulh {t}, {a3}, {b2}",
            "adc {r6}, {r6}, {t}",
            // + a * b3 into r3..r7.
            "mul {t}, {a0}, {b3}",
            "adds {r3}, {r3}, {t}",
            "mul {t}, {a1}, {b3}",
            "adcs {r4}, {r4}, {t}",
            "mul {t}, {a2}, {b3}",
            "adcs {r5}, {r5}, {t}",
            "mul {t}, {a3}, {b3}",
            "adcs {r6}, {r6}, {t}",
            "cset {r7}, cs",
            "umulh {t}, {a0}, {b3}",
            "adds {r4}, {r4}, {t}",
            "umulh {t}, {a1}, {b3}",
            "adcs {r5}, {r5}, {t}",
            "umulh {t}, {a2}, {b3}",
            "adcs {r6}, {r6}, {t}",
            "umulh {t}, {a3}, {b3}",
            "adc {r7}, {r7}, {t}",
            a0 = in(reg) a[0],
            a1 = in(reg) a[1],
            a2 = in(reg) a[2],
            a3 = in(reg) a[3],
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            r0 = out(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            r4 = out(reg) r4,
            r5 = out(reg) r5,
            r6 = out(reg) r6,
            r7 = out(reg) r7,
            t = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// One reduction round on `u`: `u = (u + k m) / 2^64` with `k = u0 n0`, which clears the
/// bottom limb. That limb's sum is `0 mod 2^64`, so its product is never formed: it
/// carries exactly when `u0 != 0`, which `cmp u0, #1` puts in the carry flag.
macro_rules! redc_round {
    () => {
        concat!(
            "mul {k}, {u0}, {n0}\n",
            "cmp {u0}, #1\n",
            "mul {t}, {k}, {m1}\n",
            "adcs {u1}, {u1}, {t}\n",
            "mul {t}, {k}, {m2}\n",
            "adcs {u2}, {u2}, {t}\n",
            "mul {t}, {k}, {m3}\n",
            "adcs {u3}, {u3}, {t}\n",
            "cset {c}, cs\n",
            "umulh {t}, {k}, {m0}\n",
            "adds {u0}, {u1}, {t}\n",
            "umulh {t}, {k}, {m1}\n",
            "adcs {u1}, {u2}, {t}\n",
            "umulh {t}, {k}, {m2}\n",
            "adcs {u2}, {u3}, {t}\n",
            "umulh {t}, {k}, {m3}\n",
            "adc {u3}, {c}, {t}\n",
        )
    };
}

/// `(u + k m) / 2^256` for the `k < 2^256` that makes it exact: four rounds. For any
/// four-limb `u` the result is at most `m`.
#[inline(always)]
pub(super) fn redc_low(u: [u64; 4], m: &[u64; 4], n0: u64) -> [u64; 4] {
    let [mut u0, mut u1, mut u2, mut u3] = u;
    // SAFETY: register arithmetic only: no memory, no stack; flags clobbered.
    unsafe {
        core::arch::asm!(
            redc_round!(),
            redc_round!(),
            redc_round!(),
            redc_round!(),
            u0 = inout(reg) u0,
            u1 = inout(reg) u1,
            u2 = inout(reg) u2,
            u3 = inout(reg) u3,
            m0 = in(reg) m[0],
            m1 = in(reg) m[1],
            m2 = in(reg) m[2],
            m3 = in(reg) m[3],
            n0 = in(reg) n0,
            k = out(reg) _,
            t = out(reg) _,
            c = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [u0, u1, u2, u3]
}

/// `a - b` on eight limbs, wrapping, and the borrow out as 0 or 1.
#[inline(always)]
pub(super) fn sub_wide(a: [u64; 8], b: [u64; 8]) -> ([u64; 8], u64) {
    let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7] = a;
    let bw: u64;
    // SAFETY: register arithmetic only: no memory, no stack; flags clobbered.
    unsafe {
        core::arch::asm!(
            "subs {a0}, {a0}, {b0}",
            "sbcs {a1}, {a1}, {b1}",
            "sbcs {a2}, {a2}, {b2}",
            "sbcs {a3}, {a3}, {b3}",
            "sbcs {a4}, {a4}, {b4}",
            "sbcs {a5}, {a5}, {b5}",
            "sbcs {a6}, {a6}, {b6}",
            "sbcs {a7}, {a7}, {b7}",
            "cset {bw}, lo",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            a4 = inout(reg) a4,
            a5 = inout(reg) a5,
            a6 = inout(reg) a6,
            a7 = inout(reg) a7,
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            b4 = in(reg) b[4],
            b5 = in(reg) b[5],
            b6 = in(reg) b[6],
            b7 = in(reg) b[7],
            bw = lateout(reg) bw,
            options(pure, nomem, nostack),
        );
    }
    ([a0, a1, a2, a3, a4, a5, a6, a7], bw)
}

/// `a + b` on four limbs, wrapping: no reduction, the caller bounds the sum.
#[inline(always)]
pub(super) fn add_wrap(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let [mut a0, mut a1, mut a2, mut a3] = a;
    // SAFETY: register arithmetic only: no memory, no stack; flags clobbered.
    unsafe {
        core::arch::asm!(
            "adds {a0}, {a0}, {b0}",
            "adcs {a1}, {a1}, {b1}",
            "adcs {a2}, {a2}, {b2}",
            "adc {a3}, {a3}, {b3}",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            b0 = in(reg) b[0],
            b1 = in(reg) b[1],
            b2 = in(reg) b[2],
            b3 = in(reg) b[3],
            options(pure, nomem, nostack),
        );
    }
    [a0, a1, a2, a3]
}
