//! Compile and link the circom witness generators.
//!
//! Which generator each circuit is built from is decided by `bench/scripts/csp-fetch.sh`
//! and recorded there as `<name>.staged.cpp`/`.staged.dat`: a local `--sanity_check 0`
//! build where that provably reproduces upstream's, and upstream's own where it does
//! not. That script explains both, and which circuits fall on which side. This build
//! reads the staged pair and nothing else.
//!
//! The witnesscalc checkout is patched before it is built; see [`patch_fr`].

use std::path::{Path, PathBuf};
use std::process::Command;

/// Circuits whose generator is linked in. Keep in step with `witness!` in `circuits.rs`.
const CIRCUITS: &[&str] = &[
    "sha256_128",
    "sha256_256",
    "sha256_512",
    "sha256_1024",
    "sha256_2048",
    "keccak_128",
    "keccak_256",
    "keccak_512",
    "keccak_1024",
    "keccak_2048",
    "poseidon_2",
    "poseidon_4",
    "poseidon_8",
    "poseidon_12",
    "poseidon_16",
    "ecdsa_32",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("../vendor/csp-benchmarks/circom/circuits");
    let staged = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("circuits");

    // witnesscalc-adapter compiles every `.cpp` in one flat directory, and upstream keeps
    // each circuit in a subdirectory of its own. Hard link rather than copy: poseidon_16
    // alone is 20 MB of generated C++.
    std::fs::create_dir_all(&staged).unwrap();
    for name in CIRCUITS {
        let family = name.rsplit_once('_').expect("name is family_size").0;
        let dir = src.join(family).join(name);
        // `csp-fetch.sh` decides, per circuit, whether the staged generator is a local
        // `--sanity_check 0` build or upstream's own, and writes the answer here. Read
        // only that pair: a `.cpp` and a `.dat` from different builds do not describe the
        // same circuit, and requiring the staged names rather than falling back to
        // upstream's means a stale checkout fails here instead of silently mixing them.
        for (from, to) in [
            (
                dir.join(format!("{name}.staged.cpp")),
                format!("{name}.cpp"),
            ),
            (
                dir.join(format!("{name}.staged.dat")),
                format!("{name}.dat"),
            ),
        ] {
            if !from.is_file() {
                panic!(
                    "missing {}\nrun bench/scripts/csp-fetch.sh first",
                    from.display()
                );
            }
            link(&from, &staged.join(to));
        }
    }
    println!("cargo:rerun-if-changed={}", src.display());

    // Clone before `build_and_link` would, so there is a checkout to patch. It only
    // clones when the directory is absent, so it finds this one and goes straight to the
    // build.
    let witnesscalc = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("witnesscalc");
    if !witnesscalc.exists() {
        clone_witnesscalc(&witnesscalc);
    }
    println!("cargo:rerun-if-env-changed=G16_CSP_STOCK_FR");
    if std::env::var_os("G16_CSP_STOCK_FR").is_none() {
        patch_fr(&witnesscalc);
    }
    witnesscalc_adapter::build_and_link(staged.to_str().unwrap());

    // `build_and_link` emits `-l dylib=witnesscalc_<circuit>` but no rpath, so without
    // this the binary links and then dies at start-up with "no LC_RPATH's found".
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for sub in [
        "witnesscalc/package/lib",
        "witnesscalc/build_witnesscalc/src",
    ] {
        let dir = out.join(sub);
        if dir.is_dir() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
}

fn link(from: &Path, to: &Path) {
    if to.exists() {
        std::fs::remove_file(to).ok();
    }
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to).expect("staging witness generator source");
    }
}

fn clone_witnesscalc(dir: &Path) {
    let run = |args: &[&str], cwd: Option<&Path>| {
        let mut c = Command::new("git");
        c.args(args);
        if let Some(d) = cwd {
            c.current_dir(d);
        }
        let ok = c.status().expect("git is not on PATH").success();
        assert!(ok, "git {} failed", args.join(" "));
    };
    run(
        &[
            "clone",
            "https://github.com/zkmopro/witnesscalc.git",
            dir.to_str().unwrap(),
        ],
        None,
    );
    run(&["submodule", "update", "--init", "--recursive"], Some(dir));
}

/// Give `Fr_mod` and `Fr_idiv` a path that does not go through GMP.
///
/// circom routes every witness time `%` and `\\` through these two functions, and the
/// shipped bodies answer each call with three `mpz_init`/`mpz_clear` pairs and a multi
/// precision division: 53.7 ns for a remainder that often fits in a register, 4,300,800
/// times for one `keccak_2048` witness and 2.0 million times for one `ecdsa_32` witness.
///
/// Three shapes cover almost every call these circuits make. Two small non-negative
/// shorts is keccak and sha256 index arithmetic, `(x + 1) % 5` and `r % 64`, and measures
/// 1.18 ns against 53.7. A power of two divisor is what the bigint gadgets divide by all
/// through `ecdsa_32`, and the answer is a shift and a mask. A single limb divisor with a
/// single limb dividend is one `udiv`. Anything wider still goes to GMP.
///
/// `Fr_toMpz` normalises an element into [0, q) before dividing and `Fr_fromMpz` tags the
/// result `Fr_SHORT` when it fits in a signed int, leaving `longVal` alone, so these
/// paths reproduce the shipped representation and not merely the shipped value. All
/// sixteen reference witnesses are byte identical across the change.
///
/// Set `G16_CSP_STOCK_FR` to build the checkout unpatched, which is how the before
/// numbers in `bench/results/csp` were measured.
fn patch_fr(witnesscalc: &Path) {
    const HELPERS: &str = r##"
// g16: divide without GMP whenever the operands make that possible.
//
// circom routes every witness time `%` and `\` through Fr_mod and Fr_idiv, and the
// shipped bodies answer each one with three mpz_init/mpz_clear pairs and a multi
// precision division. Three shapes cover almost every call these circuits make:
// two small non-negative shorts (keccak and sha256 index arithmetic), a divisor that
// is a power of two (the bigint gadgets divide by 2^64 throughout), and a single limb
// divisor with a single limb dividend.
//
// Fr_toMpz normalises an element into [0, q) before dividing and Fr_fromMpz tags the
// result Fr_SHORT when it fits in a signed int, leaving longVal alone, so these paths
// reproduce the shipped representation exactly and not merely the shipped value.

static inline bool g16_limbs(uint64_t out[4], PFrElement e) {
    FrElement t;
    Fr_toNormal(&t, e);
    if (t.type & Fr_LONG) {
        out[0] = t.longVal[0]; out[1] = t.longVal[1];
        out[2] = t.longVal[2]; out[3] = t.longVal[3];
        return true;
    }
    // A negative short stands for shortVal + q, a 254 bit number. None of the circuits
    // benchmarked here produce one as a dividend, so it goes to the general path.
    if (t.shortVal < 0) return false;
    out[0] = (uint64_t)t.shortVal; out[1] = out[2] = out[3] = 0;
    return true;
}

static inline void g16_store(PFrElement r, const uint64_t v[4]) {
    if (!v[1] && !v[2] && !v[3] && v[0] <= 0x7fffffffull) {
        r->type = Fr_SHORT;
        r->shortVal = (int32_t)v[0];
        return;                      // Fr_fromMpz leaves longVal untouched here
    }
    r->type = Fr_LONG;
    r->longVal[0] = v[0]; r->longVal[1] = v[1];
    r->longVal[2] = v[2]; r->longVal[3] = v[3];
}

// The one bit set in b, or -1 when b is not a power of two.
static inline int g16_log2(const uint64_t b[4]) {
    int bit = -1;
    for (int i = 0; i < 4; i++) {
        if (!b[i]) continue;
        if (b[i] & (b[i] - 1)) return -1;       // more than one bit in this limb
        if (bit >= 0) return -1;                // and a bit in an earlier limb
        bit = i * 64 + __builtin_ctzll(b[i]);
    }
    return bit;
}

static inline bool g16_divmod(PFrElement r, PFrElement a, PFrElement b, bool quotient) {
    uint64_t av[4], bv[4], out[4] = {0, 0, 0, 0};
    if (!g16_limbs(av, a) || !g16_limbs(bv, b)) return false;
    if (!(bv[0] | bv[1] | bv[2] | bv[3])) return false;   // let the general path decide

    int k = g16_log2(bv);
    if (k >= 0) {
        if (quotient) {                                    // a >> k
            int w = k / 64, s = k % 64;
            for (int i = 0; i + w < 4; i++) {
                out[i] = av[i + w] >> s;
                if (s && i + w + 1 < 4) out[i] |= av[i + w + 1] << (64 - s);
            }
        } else {                                           // a & ((1 << k) - 1)
            for (int i = 0; i < 4; i++) {
                if (k >= (i + 1) * 64) out[i] = av[i];
                else if (k > i * 64) out[i] = av[i] & ((1ull << (k - i * 64)) - 1);
            }
        }
        g16_store(r, out);
        return true;
    }
    // One limb over one limb is a single udiv; anything wider is left to GMP, which
    // measurement says is not worth special casing for these circuits.
    if (!(bv[1] | bv[2] | bv[3]) && !(av[1] | av[2] | av[3])) {
        out[0] = quotient ? av[0] / bv[0] : av[0] % bv[0];
        g16_store(r, out);
        return true;
    }
    return false;
}
"##;
    let fr = witnesscalc.join("build/fr.cpp");
    // Restore before patching rather than checking for a marker. A marker can only say
    // that some version of this patch is present, not that it is this one, and an
    // OUT_DIR outlives edits to this file. The checkout is a git clone and the adapter
    // rebuilds it from scratch on every run, so this costs nothing.
    let ok = Command::new("git")
        .args(["checkout", "--", "build/fr.cpp"])
        .current_dir(witnesscalc)
        .status()
        .expect("git is not on PATH")
        .success();
    assert!(
        ok,
        "could not restore build/fr.cpp in the witnesscalc checkout"
    );
    let src = std::fs::read_to_string(&fr).expect("witnesscalc checkout has no build/fr.cpp");
    assert!(
        !src.contains("g16"),
        "git restored a build/fr.cpp that is still patched"
    );
    let mut out = src;
    for (name, op, quotient) in [("Fr_mod", "%", "false"), ("Fr_idiv", "/", "true")] {
        let sig = format!("void {name}(PFrElement r, PFrElement a, PFrElement b) {{");
        let at = out
            .find(&sig)
            .unwrap_or_else(|| panic!("{name} is not declared the way this patch expects"));
        let fast = format!(
            "\n    // g16: short operands, then the limb paths, then GMP\n\
             \x20   if (!((a->type | b->type) & Fr_LONG) && a->shortVal >= 0 && b->shortVal > 0) {{\n\
             \x20       int32_t v = a->shortVal {op} b->shortVal;\n\
             \x20       r->type = Fr_SHORT;\n\
             \x20       r->shortVal = v;\n\
             \x20       return;\n\
             \x20   }}\n\
             \x20   if (g16_divmod(r, a, b, {quotient})) return;\n"
        );
        out.insert_str(at + sig.len(), &fast);
    }
    // Ahead of every caller. Fr_idiv is defined before Fr_mod in this file, so anchoring
    // on either function name puts the helpers after one of them.
    const ANCHOR: &str = "#include \"fr.hpp\"\n";
    let at = out
        .find(ANCHOR)
        .expect("build/fr.cpp does not open by including fr.hpp");
    out.insert_str(at + ANCHOR.len(), HELPERS);
    std::fs::write(&fr, out).expect("writing the patched build/fr.cpp");
}
