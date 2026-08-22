//! Dump the two field variants' WGSL, plus a reference vector, for a standalone browser page.
//!
//! Such a page talks to WebGPU directly from JavaScript, with no `wgpu`, no `wasm-bindgen`
//! and no build step, so it is an independent second path to the same number the wasm sweep
//! reaches through `wgpu` on wasm32. Two implementations agreeing is what a measurement that
//! can stop a project deserves; one is a claim.
//!
//! Independent does not mean it may run different arithmetic. The shaders here are written by
//! `g16_wgpu::gen`, byte for byte the source the native sweep and the prover use, and the
//! driver kernel appended to them is a verbatim copy of `benches/limb_sweep/variants.rs`.
//! Nothing about the page is hand-written WGSL.
//!
//! `vectors.json` carries the first 8 elements' inputs and the expected outputs, computed
//! here with `num-bigint`. The page regenerates all 65,536 inputs itself from the same
//! SplitMix64 seed with JavaScript `BigInt` and refuses to run if its element 0 disagrees
//! with the one in this file. That pins the browser run to the same numbers the native run
//! multiplied, so the two G mul/s figures are comparable, and it makes the JS reference an
//! independent check on the Rust one rather than a copy of it.
//!
//! Run with `cargo run -p g16-wgpu --example dump-field-wgsl --release`. Output lands in
//! `target/field-wgsl/`, so a static server is the only other thing needed to re-take the
//! measurement.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use g16_gpu_layout::FR_MODULUS;
use g16_wgpu::gen::{Variant, FR, MUL64};
use num_bigint::BigUint;
use num_traits::One;

/// Threads, and elements. Same as `benches/limb_sweep` so the numbers line up.
const ELEMS: u32 = 1 << 16;
/// Chained dependent multiplies per thread. Chained so nothing can be hoisted or dropped.
const ITERS: u32 = 256;
/// How many elements get a reference value written out.
///
/// 256 and not 8, and the difference is not caution. A kernel that drops the final
/// conditional subtraction of `m` returns the right residue class in the wrong
/// representative, which is the `>` versus `>=` trap ICME shipped. Whether that shows up in
/// element `i` is a coin flip:
/// CIOS leaves `T < m^2/R + m`, and `m/R = 0.1888` for BN254, so a raw CIOS output is over
/// `m` and needs the subtraction only 0.1888/1.1888 = 15.9% of the time. Checking 8 elements
/// therefore misses such a kernel with probability 0.841^8 = 24%, and it *did*: patching
/// `take = false` into the generated `fr_mul` was measured passing an 8-element check in
/// Chrome while running 1.6% faster. At 256 elements the miss probability is 4e-20.
///
/// The native sweep in `benches/limb_sweep/main.rs` and the wasm one both still check 8,
/// so both carry that 24% hole.
const CHECK: usize = 256;
/// SplitMix64 seed, the same constant `benches/limb_sweep/main.rs` uses.
const SEED: u64 = 0xC0FF_EE11;

/// The driver kernel, copied verbatim from `benches/limb_sweep/variants.rs::driver`. It is
/// duplicated rather than imported because a bench target is not importable from an example,
/// and it is small enough that the duplication is visible in review.
fn driver(n: usize, ty: &str, f: &str) -> String {
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

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_fr(&mut self, r: &BigUint) -> BigUint {
        let mut bytes = [0u8; 32];
        for c in bytes.chunks_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes) % r
    }
}

fn to_limbs(x: &BigUint, b: u32, n: usize) -> Vec<u32> {
    let mask = (BigUint::one() << b) - BigUint::one();
    (0..n)
        .map(|i| {
            let d = (x >> (b as usize * i)) & &mask;
            d.iter_u32_digits().next().unwrap_or(0)
        })
        .collect()
}

fn json_limbs(v: &[u32]) -> String {
    let body = v
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn main() {
    // Two levels up from crates/g16-wgpu, which is the repo root whether or not the caller
    // ran cargo from there. Build output, not a committed artifact: the page that consumes
    // it is served from wherever the caller puts it.
    let out: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/field-wgsl");
    fs::create_dir_all(&out).expect("target/field-wgsl must be creatable");

    let r = BigUint::from_slice(&FR_MODULUS);

    // Only the first CHECK elements are needed, but the RNG has to be walked in the same
    // order as the page walks it, and the page generates all ELEMS. Drawing all of a_std
    // before any of b_std is what benches/limb_sweep does, so it is what the page must do.
    let mut rng = Rng(SEED);
    let a_std: Vec<BigUint> = (0..ELEMS).map(|_| rng.next_fr(&r)).collect();
    let b_std: Vec<BigUint> = (0..ELEMS).map(|_| rng.next_fr(&r)).collect();

    let mut json = String::new();
    writeln!(json, "{{").unwrap();
    writeln!(json, "  \"seed\": \"{SEED:#018x}\",").unwrap();
    writeln!(json, "  \"elems\": {ELEMS},").unwrap();
    writeln!(json, "  \"iters\": {ITERS},").unwrap();
    writeln!(json, "  \"check\": {CHECK},").unwrap();
    writeln!(json, "  \"modulus\": \"{r:#x}\",").unwrap();
    writeln!(json, "  \"variants\": [").unwrap();

    for (vi, v) in [Variant::Cios32Unrolled, Variant::NoCarry13x20]
        .into_iter()
        .enumerate()
    {
        let (b, n) = (v.limb_bits(), v.limbs());

        let mut src = String::new();
        if v.needs_mul64() {
            src.push_str(MUL64);
        }
        src.push_str(&FR.ops(v));
        src.push_str(&driver(n, "Fr", "fr_mul"));

        let file = format!("fr_{}.wgsl", format!("{v:?}").to_lowercase());
        fs::write(out.join(&file), &src).expect("write shader");

        // R is 2^256 for the 32-bit form and 2^260 for the 13-bit one, which is exactly why
        // the 13-bit variant is benchmark-only: its wire format is not PackedFr and a flip
        // to it needs a host-side repack that does not exist. See gen::Variant.
        let big_r = BigUint::one() << (b as usize * n);
        let mont = |x: &BigUint| (x * &big_r) % &r;
        let rinv = big_r.modpow(&(&r - BigUint::from(2u32)), &r);

        let mut a_json = Vec::new();
        let mut b_json = Vec::new();
        let mut e_json = Vec::new();
        for i in 0..CHECK {
            let am = mont(&a_std[i]);
            let bm = mont(&b_std[i]);
            // The closed form after ITERS chained Montgomery multiplies by b is
            // a * b^ITERS * R^-ITERS, still in Montgomery form. Computed the slow honest way
            // rather than by modpow, so a mistake in the closed form cannot hide.
            let mut want = am.clone();
            for _ in 0..ITERS {
                want = (&want * &bm % &r) * &rinv % &r;
            }
            a_json.push(json_limbs(&to_limbs(&am, b, n)));
            b_json.push(json_limbs(&to_limbs(&bm, b, n)));
            e_json.push(json_limbs(&to_limbs(&want, b, n)));
        }

        let comma = if vi == 0 { "," } else { "" };
        writeln!(json, "    {{").unwrap();
        writeln!(json, "      \"variant\": \"{v:?}\",").unwrap();
        writeln!(json, "      \"bits\": {b},").unwrap();
        writeln!(json, "      \"limbs\": {n},").unwrap();
        writeln!(json, "      \"bytes_per_elem\": {},", v.bytes_per_elem()).unwrap();
        writeln!(json, "      \"shader\": \"{file}\",").unwrap();
        writeln!(json, "      \"src_bytes\": {},", src.len()).unwrap();
        writeln!(json, "      \"a_mont\": [{}],", a_json.join(",")).unwrap();
        writeln!(json, "      \"b_mont\": [{}],", b_json.join(",")).unwrap();
        writeln!(json, "      \"expected\": [{}]", e_json.join(",")).unwrap();
        writeln!(json, "    }}{comma}").unwrap();

        println!(
            "{v:?}: {file}, {} bytes of WGSL, {n} limbs of {b} bits",
            src.len()
        );
    }
    writeln!(json, "  ]").unwrap();
    writeln!(json, "}}").unwrap();

    fs::write(out.join("vectors.json"), &json).expect("write vectors");
    println!("vectors.json: {} bytes", json.len());
}
