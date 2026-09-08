//! `--backend cpu` against `--backend metal` for the four apply-key commands.
//!
//! The bar is byte identity of the whole output file, not "the proof verifies" and not
//! "snarkjs accepts it". The CPU path is already byte-identical to snarkjs 0.7.6, so
//! cpu-against-metal equality carries the snarkjs claim across without running snarkjs at
//! all; the two `snarkjs verify` tests at the bottom are a second opinion on top of it,
//! not the thing being relied on.
//!
//! Both phases are made comparable the same way. A beacon is a pure function of its
//! inputs and needs nothing. A contribution mixes 64 OS-random bytes into its seed by
//! design, so both runs are handed the same pinned RNG:
//! [`phase1::contribute_with`] for phase 1 and [`contribute::contribute_with`] for phase
//! 2, the latter made public by this milestone for exactly this comparison.
//!
//! Every test runs the whole matrix twice, once at the shipped crossover and once with
//! [`MetalKeyScale::with_min_points`] at 1. The second pass is not redundant: at the
//! shipped crossover a power-8 `.ptau` and a `tiny_mul` zkey never reach the device at
//! all, so without it the small-input cases would be testing the host fallback against
//! itself.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use g16_ceremony::transcript::rng_from_entropy_with;
use g16_ceremony::{contribute, phase1};
use g16_ceremony::{ContributionParams, CpuKeyScale};
use g16_metal::MetalKeyScale;
use g16_msm::KeyScale;

/// `01 02 .. 20`, the beacon the phase 1 and phase 2 suites already use.
const BEACON_HEX: &str = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";

fn beacon_bytes() -> Vec<u8> {
    (0..BEACON_HEX.len() / 2)
        .map(|i| u8::from_str_radix(&BEACON_HEX[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

/// The 64 bytes a contribution would have taken from the OS, pinned so both backends draw
/// the same delta. Same convention as `tests/phase1.rs`.
fn os_bytes() -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, b) in out.iter_mut().enumerate() {
        *b = i as u8;
    }
    out
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-metal-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bench_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .join(rel)
}

/// The Metal backend at a given crossover, or `None` on a machine with no device.
///
/// A skip rather than a failure: this file is compiled by a feature, and the feature says
/// "test the Metal path", not "this machine has a GPU". Every caller prints why.
///
/// `min_points` of `None` leaves the shipped crossover alone.
fn metal(min_points: Option<usize>) -> Option<MetalKeyScale> {
    match MetalKeyScale::new() {
        Ok(k) => Some(match min_points {
            Some(n) => k.with_min_points(n),
            None => k,
        }),
        Err(e) => {
            eprintln!("SKIPPED: no Metal backend: {e}");
            None
        }
    }
}

/// The two crossovers every case is run at: the shipped one, which routes a short section
/// home, and one that forces even a 19-point section onto the device.
const CROSSOVERS: [(Option<usize>, &str); 2] = [(None, "shipped"), (Some(1), "device")];

fn assert_same_bytes(what: &str, cpu: &Path, gpu: &Path) {
    let a = std::fs::read(cpu).unwrap();
    let b = std::fs::read(gpu).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: cpu wrote {} bytes, metal wrote {}",
        a.len(),
        b.len()
    );
    if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
        panic!(
            "{what}: first difference at byte {i}, cpu 0x{:02x} against metal 0x{:02x}",
            a[i], b[i]
        );
    }
}

/// Every zkey small enough to rescale twice per crossover without the suite becoming a
/// benchmark. `tiny_mul` is 19 points a side and is the case the host fallback exists for.
fn zkeys() -> Vec<(String, PathBuf)> {
    [
        ("tiny_mul", "artifacts/tiny_mul/circuit.zkey"),
        ("js_1x1_d8", "artifacts/js_1x1_d8/circuit.zkey"),
        ("js_2x2_d32", "artifacts/js_2x2_d32/circuit.zkey"),
    ]
    .iter()
    .map(|(name, rel)| ((*name).to_owned(), bench_path(rel)))
    .filter(|(_, p)| p.is_file())
    .collect()
}

#[test]
fn zkey_beacon_is_byte_identical_across_backends() {
    let inputs = zkeys();
    assert!(!inputs.is_empty(), "no zkeys under bench/artifacts");
    let dir = tmp_dir("zkey-beacon");
    for (min_points, label) in CROSSOVERS {
        let Some(key) = metal(min_points) else { return };
        for (name, zkey) in &inputs {
            let cpu = dir.join(format!("{name}.{label}.cpu.zkey"));
            let gpu = dir.join(format!("{name}.{label}.metal.zkey"));
            let a =
                contribute::beacon(zkey, &cpu, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
            let b = contribute::beacon(zkey, &gpu, None, &beacon_bytes(), 10, &key).unwrap();
            assert_eq!(a.hash, b.hash, "{name} ({label}): contribution hash");
            assert_same_bytes(&format!("zkey beacon {name} ({label})"), &cpu, &gpu);
        }
    }
}

#[test]
fn zkey_contribute_is_byte_identical_across_backends() {
    let inputs = zkeys();
    assert!(!inputs.is_empty(), "no zkeys under bench/artifacts");
    let dir = tmp_dir("zkey-contribute");
    for (min_points, label) in CROSSOVERS {
        let Some(key) = metal(min_points) else { return };
        for (name, zkey) in &inputs {
            let cpu = dir.join(format!("{name}.{label}.cpu.zkey"));
            let gpu = dir.join(format!("{name}.{label}.metal.zkey"));
            // The same pinned stream to both, so the two runs draw the same delta and any
            // difference left in the file is the arithmetic.
            let rng = || rng_from_entropy_with(&os_bytes(), "metal comparison");
            let a = contribute::contribute_with(zkey, &cpu, None, rng(), &CpuKeyScale).unwrap();
            let b = contribute::contribute_with(zkey, &gpu, None, rng(), &key).unwrap();
            assert_eq!(a.hash, b.hash, "{name} ({label}): contribution hash");
            assert_same_bytes(&format!("zkey contribute {name} ({label})"), &cpu, &gpu);
        }
    }
}

/// The two phase 1 inputs. A fresh power-10 file is the cheap one; `local_13.ptau` is
/// snarkjs' own and carries real prior contributions, and at power 13 section 2 fills a
/// whole 16,384-point `response_chunk` while section 3 fills a G2 one, so it is the
/// smallest input where the device sees a full chunk in both groups.
fn ptau_inputs(dir: &Path) -> Vec<(String, PathBuf)> {
    let fresh = dir.join("new_10.ptau");
    phase1::ptau_new(10, &fresh).unwrap();
    let mut out = vec![("new_10".to_owned(), fresh)];
    let local = bench_path("ptau/local_13.ptau");
    if local.is_file() {
        out.push(("local_13".to_owned(), local));
    }
    out
}

#[test]
fn ptau_beacon_is_byte_identical_across_backends() {
    let dir = tmp_dir("ptau-beacon");
    let inputs = ptau_inputs(&dir);
    for (min_points, label) in CROSSOVERS {
        let Some(key) = metal(min_points) else { return };
        for (name, ptau) in &inputs {
            let cpu = dir.join(format!("{name}.{label}.cpu.ptau"));
            let gpu = dir.join(format!("{name}.{label}.metal.ptau"));
            let a = phase1::beacon(ptau, &cpu, None, &beacon_bytes(), 10, &CpuKeyScale).unwrap();
            let b = phase1::beacon(ptau, &gpu, None, &beacon_bytes(), 10, &key).unwrap();
            assert_eq!(
                a.response_hash, b.response_hash,
                "{name} ({label}): response hash"
            );
            assert_eq!(
                a.next_challenge, b.next_challenge,
                "{name} ({label}): next challenge"
            );
            assert_same_bytes(&format!("ptau beacon {name} ({label})"), &cpu, &gpu);
        }
    }
}

#[test]
fn ptau_contribute_is_byte_identical_across_backends() {
    let dir = tmp_dir("ptau-contribute");
    let inputs = ptau_inputs(&dir);
    for (min_points, label) in CROSSOVERS {
        let Some(key) = metal(min_points) else { return };
        for (name, ptau) in &inputs {
            let cpu = dir.join(format!("{name}.{label}.cpu.ptau"));
            let gpu = dir.join(format!("{name}.{label}.metal.ptau"));
            let rng = || rng_from_entropy_with(&os_bytes(), "metal comparison");
            let params = || ContributionParams {
                name: Some("metal comparison".into()),
                ..Default::default()
            };
            let a = phase1::contribute_with(ptau, &cpu, params(), rng(), &CpuKeyScale).unwrap();
            let b = phase1::contribute_with(ptau, &gpu, params(), rng(), &key).unwrap();
            assert_eq!(
                a.response_hash, b.response_hash,
                "{name} ({label}): response hash"
            );
            assert_same_bytes(&format!("ptau contribute {name} ({label})"), &cpu, &gpu);
        }
    }
}

/// The point at infinity through both of `MetalKeyScale`'s paths, because no file this
/// repo ships puts one in front of `apply_key`.
///
/// That is worth stating plainly, since the brief calls infinity a live path. It is, but
/// not for this primitive: every checked-in `.zkey` has zero identity points across
/// sections 8 and 9 (surveyed over all thirteen artifacts), and the five `.ptau` point
/// sections hold powers of tau, which are never the identity. Where infinity really is
/// live is ptau section 12's Lagrange padding, which `prepare` owns, and the A/B/C query
/// vectors, which `setup` owns.
///
/// So the coverage has to be planted, and it is planted here rather than left to
/// `g16-metal`'s own kernel test because that test cannot reach the host fallback: below
/// the crossover the points never leave this process.
#[test]
fn the_identity_survives_both_of_the_metal_paths() {
    use g16_field::{AffineRepr, CurveGroup, Fr, G1Affine, G2Affine};

    let Some(device) = metal(Some(1)) else { return };
    let host = MetalKeyScale::new().unwrap().with_min_points(usize::MAX);

    let mut seed = Fr::from(3u64);
    let mut g1: Vec<G1Affine> = Vec::new();
    let mut g2: Vec<G2Affine> = Vec::new();
    for i in 0..1500 {
        if i % 7 == 0 {
            g1.push(G1Affine::identity());
            g2.push(G2Affine::identity());
        } else {
            seed *= Fr::from(5u64);
            g1.push((G1Affine::generator() * seed).into_affine());
            g2.push((G2Affine::generator() * seed).into_affine());
        }
    }

    for (first, inc, what) in [
        (Fr::from(7u64), Fr::from(11u64), "geometric"),
        (Fr::from(7u64), Fr::from(1u64), "constant, the zkey case"),
    ] {
        let mut want1 = g1.clone();
        CpuKeyScale.apply_key_g1(&mut want1, first, inc).unwrap();
        let mut want2 = g2.clone();
        CpuKeyScale.apply_key_g2(&mut want2, first, inc).unwrap();
        for (key, path) in [(&device, "device"), (&host, "host fallback")] {
            let mut got1 = g1.clone();
            key.apply_key_g1(&mut got1, first, inc).unwrap();
            assert_eq!(got1, want1, "G1 {what}, {path}");
            let mut got2 = g2.clone();
            key.apply_key_g2(&mut got2, first, inc).unwrap();
            assert_eq!(got2, want2, "G2 {what}, {path}");
        }
    }
}

// --- snarkjs, the second opinion ---

fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    Command::new(&bin).arg("--help").output().ok().map(|_| bin)
}

fn expect_snarkjs(bin: &str, args: &[&str]) {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running snarkjs {args:?}: {e}"));
    let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "snarkjs {args:?} failed:\n{log}");
}

#[test]
fn snarkjs_verifies_a_metal_ptau_chain() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED snarkjs_verifies_a_metal_ptau_chain: no snarkjs on PATH");
        return;
    };
    let Some(key) = metal(Some(1)) else { return };
    let dir = tmp_dir("snarkjs-ptau");
    let fresh = dir.join("new_10.ptau");
    let contributed = dir.join("c_10.ptau");
    let final_ = dir.join("cb_10.ptau");
    phase1::ptau_new(10, &fresh).unwrap();
    phase1::contribute_with(
        &fresh,
        &contributed,
        ContributionParams::default(),
        rng_from_entropy_with(&os_bytes(), "metal comparison"),
        &key,
    )
    .unwrap();
    phase1::beacon(&contributed, &final_, None, &beacon_bytes(), 10, &key).unwrap();
    expect_snarkjs(&bin, &["powersoftau", "verify", final_.to_str().unwrap()]);
}

/// `snarkjs zkvi` over a chain both of whose links the device produced.
///
/// The init key is built by snarkjs itself, from a circuit and a `.ptau` this repo ships,
/// so nothing in the chain comes from our `setup`. `zkvi`, not `zkey verify`: the short
/// alias is claimed by the r1cs form (`cli.js:220`), which re-runs a whole setup.
#[test]
fn snarkjs_verifies_a_metal_zkey_contribution() {
    let Some(bin) = snarkjs_bin() else {
        eprintln!("SKIPPED snarkjs_verifies_a_metal_zkey_contribution: no snarkjs on PATH");
        return;
    };
    let r1cs = bench_path("artifacts/js_1x1_d8/circuit.r1cs");
    let ptau = bench_path("ptau/local_19.ptau");
    if !r1cs.is_file() || !ptau.is_file() {
        eprintln!("SKIPPED snarkjs_verifies_a_metal_zkey_contribution: no js_1x1_d8 or local_19");
        return;
    }
    let Some(key) = metal(Some(1)) else { return };
    let dir = tmp_dir("snarkjs-zkey");
    let init = dir.join("init.zkey");
    expect_snarkjs(
        &bin,
        &[
            "groth16",
            "setup",
            r1cs.to_str().unwrap(),
            ptau.to_str().unwrap(),
            init.to_str().unwrap(),
        ],
    );

    let contributed = dir.join("c1.zkey");
    let final_ = dir.join("c2.zkey");
    contribute::contribute_with(
        &init,
        &contributed,
        Some("metal comparison"),
        rng_from_entropy_with(&os_bytes(), "metal comparison"),
        &key,
    )
    .unwrap();
    contribute::beacon(
        &contributed,
        &final_,
        Some("final"),
        &beacon_bytes(),
        10,
        &key,
    )
    .unwrap();
    expect_snarkjs(
        &bin,
        &[
            "zkvi",
            init.to_str().unwrap(),
            ptau.to_str().unwrap(),
            final_.to_str().unwrap(),
        ],
    );
}

// --- measurement ---

/// The crossover sweep behind [`MetalKeyScale`]'s `KEY_MIN_POINTS`, and the per-command
/// timings in `contribute-metal-timings.md`.
///
/// Ignored because it is a benchmark: it says nothing about correctness and it takes
/// minutes. `cargo test --release -p g16-ceremony --features metal -- --ignored
/// --nocapture crossover`.
#[test]
#[ignore = "benchmark"]
fn crossover_sweep() {
    use g16_field::{AffineRepr, Fr, G1Affine, G2Affine};

    let Some(key) = metal(Some(1)) else { return };
    let mut seed = Fr::from(3u64);
    let first = Fr::from(7u64);
    let inc = Fr::from(11u64);

    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "n", "g1 cpu ms", "g1 gpu ms", "g2 cpu ms", "g2 gpu ms"
    );
    for bits in 4..=16u32 {
        let n = 1usize << bits;
        let mut g1: Vec<G1Affine> = Vec::with_capacity(n);
        let mut g2: Vec<G2Affine> = Vec::with_capacity(n);
        for _ in 0..n {
            seed *= Fr::from(5u64);
            g1.push((G1Affine::generator() * seed).into());
            g2.push((G2Affine::generator() * seed).into());
        }
        let ms = |f: &dyn Fn()| {
            f();
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        };
        let a = ms(&|| {
            let mut v = g1.clone();
            CpuKeyScale.apply_key_g1(&mut v, first, inc).unwrap();
        });
        let b = ms(&|| {
            let mut v = g1.clone();
            key.apply_key_g1(&mut v, first, inc).unwrap();
        });
        let c = ms(&|| {
            let mut v = g2.clone();
            CpuKeyScale.apply_key_g2(&mut v, first, inc).unwrap();
        });
        let d = ms(&|| {
            let mut v = g2.clone();
            key.apply_key_g2(&mut v, first, inc).unwrap();
        });
        println!("{n:>10}  {a:>12.3}  {b:>12.3}  {c:>12.3}  {d:>12.3}");
    }
}
