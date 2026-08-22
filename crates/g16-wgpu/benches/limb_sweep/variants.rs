//! The limb layouts the sweep measures that the backend does **not** ship.
//!
//! The two variants `g16-wgpu` carries live in `src/gen/field.rs` and the sweep pulls them
//! from there, so the numbers this sweep reports are numbers for the
//! code that actually runs in the prover. Everything in this file is a loser kept so the
//! comparison stays honest: three looped forms, which exist only to measure what unrolling is
//! worth, and the lazy-carry unrolled family at every limb width from 11 to 15, which exists
//! so "13-bit limbs win" can be tested rather than repeated.
//!
//! These emit a function called `montmul` over a bare `array<u32, N>`, not the crate's
//! `fr_mul` over the `Fr` alias, which is why [`driver`] takes both names.

/// Lazy-carry CIOS for limb widths where `(2^B - 1)^2 + 2^B + carry` provably fits in u32,
/// i.e. B <= 15. One u32 multiply per limb product and no product splitting at all.
pub fn lazy_cios(b: u32, n: usize, modulus: &[u32], n0: u32) -> String {
    let mask = (1u32 << b) - 1;
    let mods = modulus
        .iter()
        .map(|x| format!("{x}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
const N: u32 = {n}u;
const B: u32 = {b}u;
const MASK: u32 = {mask}u;
const NP: u32 = {n0}u;
const MODU = array<u32, {n}>({mods});

fn montmul(a: array<u32, {n}>, bb: array<u32, {n}>) -> array<u32, {n}> {{
    var t: array<u32, {np2}>;
    for (var k = 0u; k < N + 2u; k = k + 1u) {{ t[k] = 0u; }}

    for (var i = 0u; i < N; i = i + 1u) {{
        var c: u32 = 0u;
        let bi = bb[i];
        for (var j = 0u; j < N; j = j + 1u) {{
            // a[j]*bi <= (2^B-1)^2, plus t[j] < 2^B, plus c < 2^(32-B). Fits in u32 for B <= 15.
            let x = t[j] + a[j] * bi + c;
            t[j] = x & MASK;
            c = x >> B;
        }}
        let x0 = t[N] + c;
        t[N] = x0 & MASK;
        t[N + 1u] = x0 >> B;

        let m = (t[0] * NP) & MASK;
        c = (t[0] + m * MODU[0]) >> B;
        for (var j = 1u; j < N; j = j + 1u) {{
            let x = t[j] + m * MODU[j] + c;
            t[j - 1u] = x & MASK;
            c = x >> B;
        }}
        let x1 = t[N] + c;
        t[N - 1u] = x1 & MASK;
        t[N] = t[N + 1u] + (x1 >> B);
    }}

    // One conditional subtraction, branch-free: the CIOS intermediate is below 2r.
    var red: array<u32, {n}>;
    var borrow: u32 = 0u;
    for (var k = 0u; k < N; k = k + 1u) {{
        let d = t[k] + (1u << B) - MODU[k] - borrow;
        red[k] = d & MASK;
        borrow = 1u - (d >> B);
    }}
    var out: array<u32, {n}>;
    let take = 1u - borrow;                 // borrow==0 means t >= r, so take the reduced form
    for (var k = 0u; k < N; k = k + 1u) {{
        out[k] = select(t[k], red[k], take == 1u);
    }}
    return out;
}}
"#,
        np2 = n + 2
    )
}

/// B = 16. Every limb product is up to (2^16-1)^2 which alone fills a u32, so the product
/// must be split before anything is added to it. Two extra ops per limb product.
pub fn cios16(n: usize, modulus: &[u32], n0: u32) -> String {
    let mods = modulus
        .iter()
        .map(|x| format!("{x}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
const N: u32 = {n}u;
const B: u32 = 16u;
const MASK: u32 = 65535u;
const NP: u32 = {n0}u;
const MODU = array<u32, {n}>({mods});

fn montmul(a: array<u32, {n}>, bb: array<u32, {n}>) -> array<u32, {n}> {{
    var t: array<u32, {np2}>;
    for (var k = 0u; k < N + 2u; k = k + 1u) {{ t[k] = 0u; }}

    for (var i = 0u; i < N; i = i + 1u) {{
        var c: u32 = 0u;
        let bi = bb[i];
        for (var j = 0u; j < N; j = j + 1u) {{
            let p = a[j] * bi;
            let x = t[j] + (p & MASK) + c;
            t[j] = x & MASK;
            c = (x >> 16u) + (p >> 16u);
        }}
        let x0 = t[N] + c;
        t[N] = x0 & MASK;
        t[N + 1u] = x0 >> 16u;

        let m = (t[0] * NP) & MASK;
        let p0 = m * MODU[0];
        c = ((t[0] + (p0 & MASK)) >> 16u) + (p0 >> 16u);
        for (var j = 1u; j < N; j = j + 1u) {{
            let p = m * MODU[j];
            let x = t[j] + (p & MASK) + c;
            t[j - 1u] = x & MASK;
            c = (x >> 16u) + (p >> 16u);
        }}
        let x1 = t[N] + c;
        t[N - 1u] = x1 & MASK;
        t[N] = t[N + 1u] + (x1 >> 16u);
    }}

    var red: array<u32, {n}>;
    var borrow: u32 = 0u;
    for (var k = 0u; k < N; k = k + 1u) {{
        let d = t[k] + 65536u - MODU[k] - borrow;
        red[k] = d & MASK;
        borrow = 1u - (d >> 16u);
    }}
    var out: array<u32, {n}>;
    let take = 1u - borrow;
    for (var k = 0u; k < N; k = k + 1u) {{
        out[k] = select(t[k], red[k], take == 1u);
    }}
    return out;
}}
"#,
        np2 = n + 2
    )
}

/// B = 32, 8 limbs, our wire format, but with the limb loops left as loops. This is the
/// control for the unrolling result: identical arithmetic to `Variant::Cios32Unrolled`, and
/// it measured 0.516 G mul/s against 1.964.
pub fn cios32_looped(modulus: &[u32], n0: u32) -> String {
    let mods = modulus
        .iter()
        .map(|x| format!("{x}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
const N: u32 = 8u;
const NP: u32 = {n0}u;
const MODU = array<u32, 8>({mods});

fn mul64(a: u32, b: u32) -> vec2<u32> {{
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (mid << 16u) | (p00 & 0xffffu);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}}

// x + y + carry_in, returning sum and carry_out.
fn addc(x: u32, y: u32, cin: u32) -> vec2<u32> {{
    let s1 = x + y;
    let c1 = select(0u, 1u, s1 < x);
    let s2 = s1 + cin;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}}

fn montmul(a: array<u32, 8>, bb: array<u32, 8>) -> array<u32, 8> {{
    var t: array<u32, 10>;
    for (var k = 0u; k < 10u; k = k + 1u) {{ t[k] = 0u; }}

    for (var i = 0u; i < 8u; i = i + 1u) {{
        var c: u32 = 0u;
        let bi = bb[i];
        for (var j = 0u; j < 8u; j = j + 1u) {{
            let p = mul64(a[j], bi);
            let s = addc(t[j], p.x, c);
            t[j] = s.x;
            c = p.y + s.y;              // p.y <= 2^32 - 2^17 so this cannot wrap
        }}
        let s0 = addc(t[8], c, 0u);
        t[8] = s0.x;
        t[9] = s0.y;

        let m = t[0] * NP;              // low 32 bits are all that matter
        let p0 = mul64(m, MODU[0]);
        let s1 = addc(t[0], p0.x, 0u);
        c = p0.y + s1.y;
        for (var j = 1u; j < 8u; j = j + 1u) {{
            let p = mul64(m, MODU[j]);
            let s = addc(t[j], p.x, c);
            t[j - 1u] = s.x;
            c = p.y + s.y;
        }}
        let s2 = addc(t[8], c, 0u);
        t[7] = s2.x;
        t[8] = t[9] + s2.y;
    }}

    var red: array<u32, 8>;
    var borrow: u32 = 0u;
    for (var k = 0u; k < 8u; k = k + 1u) {{
        let d1 = t[k] - MODU[k];
        let b1 = select(0u, 1u, t[k] < MODU[k]);
        let d2 = d1 - borrow;
        let b2 = select(0u, 1u, d1 < borrow);
        red[k] = d2;
        borrow = b1 + b2;
    }}
    var out: array<u32, 8>;
    let take = 1u - borrow;
    for (var k = 0u; k < 8u; k = k + 1u) {{
        out[k] = select(t[k], red[k], take == 1u);
    }}
    return out;
}}
"#
    )
}

/// The lazy-carry form, unrolled. Present at B = 11 to 15 so the unrolled family can be
/// swept, which is what shows that unrolling beats every choice of limb width.
pub fn lazy_cios_unrolled(b: u32, n: usize, modulus: &[u32], n0: u32) -> String {
    use std::fmt::Write as _;
    let mask = (1u32 << b) - 1;
    let mut s = String::new();
    writeln!(s, "const MASK: u32 = {mask}u;").unwrap();
    writeln!(s, "const NP: u32 = {n0}u;").unwrap();
    for (i, m) in modulus.iter().enumerate() {
        writeln!(s, "const M{i}: u32 = {m}u;").unwrap();
    }
    writeln!(
        s,
        "\nfn montmul(a: array<u32, {n}>, bb: array<u32, {n}>) -> array<u32, {n}> {{"
    )
    .unwrap();
    for j in 0..=n + 1 {
        writeln!(s, "    var t{j}: u32 = 0u;").unwrap();
    }
    for i in 0..n {
        writeln!(s, "    {{").unwrap();
        writeln!(s, "    let bi = bb[{i}];").unwrap();
        writeln!(s, "    var c: u32 = 0u;").unwrap();
        for j in 0..n {
            writeln!(
                s,
                "    {{ let x = t{j} + a[{j}] * bi + c; t{j} = x & MASK; c = x >> {b}u; }}"
            )
            .unwrap();
        }
        writeln!(
            s,
            "    {{ let x = t{n} + c; t{n} = x & MASK; t{} = x >> {b}u; }}",
            n + 1
        )
        .unwrap();
        writeln!(s, "    let m = (t0 * NP) & MASK;").unwrap();
        writeln!(s, "    c = (t0 + m * M0) >> {b}u;").unwrap();
        for j in 1..n {
            writeln!(
                s,
                "    {{ let x = t{j} + m * M{j} + c; t{} = x & MASK; c = x >> {b}u; }}",
                j - 1
            )
            .unwrap();
        }
        writeln!(
            s,
            "    {{ let x = t{n} + c; t{} = x & MASK; t{n} = t{} + (x >> {b}u); }}",
            n - 1,
            n + 1
        )
        .unwrap();
        writeln!(s, "    }}").unwrap();
    }
    writeln!(s, "    var borrow: u32 = 0u;").unwrap();
    for j in 0..n {
        writeln!(
            s,
            "    let d{j} = t{j} + {}u - M{j} - borrow; borrow = 1u - (d{j} >> {b}u);",
            1u32 << b
        )
        .unwrap();
    }
    writeln!(s, "    let take = borrow == 0u;").unwrap();
    writeln!(s, "    var out: array<u32, {n}>;").unwrap();
    for j in 0..n {
        writeln!(s, "    out[{j}] = select(t{j}, d{j} & MASK, take);").unwrap();
    }
    writeln!(s, "    return out;\n}}").unwrap();
    s
}

/// The driver kernel, shared by every variant. The multiply is chained into itself so the
/// compiler cannot hoist or drop it, which makes this a latency chain per thread and a
/// throughput measurement once there are enough threads to fill the machine.
///
/// `ty` and `f` differ between the crate's generator (`Fr`, `fr_mul`) and this file
/// (`array<u32, N>`, `montmul`), and they must not be mixed: naga emits a distinct MSL struct
/// for a WGSL alias and refuses to convert it to the one it emits for a bare array literal.
pub fn driver(n: usize, ty: &str, f: &str) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> IN_A: array<u32>;
@group(0) @binding(1) var<storage, read> IN_B: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;   // x = iterations, y = element count

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.y) {{ return; }}
    let base = i * {n}u;
    var a: {ty};
    var b: {ty};
    for (var k = 0u; k < {n}u; k = k + 1u) {{
        a[k] = IN_A[base + k];
        b[k] = IN_B[base + k];
    }}
    for (var it = 0u; it < params.x; it = it + 1u) {{
        a = {f}(a, b);
    }}
    for (var k = 0u; k < {n}u; k = k + 1u) {{
        OUT[base + k] = a[k];
    }}
}}
"#
    )
}
