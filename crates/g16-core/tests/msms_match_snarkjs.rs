//! Cross-check stages 5-9 against snarkjs' own five multiexps, point by point.
//!
//! `msm_expected.json` is produced by replaying snarkjs' prover in ffjavascript up to but
//! not including the blinding, and dumping the five raw MSM results. This is the test
//! that isolates an l_query offset error or an h_query length error: both still produce
//! a proof, and both are invisible to any check that only looks at the final pairing when
//! the pairing itself is wrong for a second reason.

use g16_core::{cpu::CpuBackend, Backend, StageTimings};
use g16_field::*;
use g16_zkey::{wtns::Witness, ProvingKey};
use std::path::{Path, PathBuf};

fn artifacts() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize();
    let Ok(root) = root else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("msm_expected.json").is_file() && d.join("circuit.zkey").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

fn fq(s: &serde_json::Value) -> Fq {
    let n: num_bigint::BigUint = s.as_str().unwrap().parse().unwrap();
    Fq::from_le_bytes_mod_order(&n.to_bytes_le())
}

/// snarkjs writes affine points as `[x, y, 1]` (or `[0, 1, 0]` for infinity).
fn g1(v: &serde_json::Value) -> G1Affine {
    let a = v.as_array().unwrap();
    let z = fq(&a[2]);
    if z.is_zero() {
        G1Affine::identity()
    } else {
        assert_eq!(z, Fq::ONE, "expected an affine point");
        G1Affine::new(fq(&a[0]), fq(&a[1]))
    }
}

fn g2(v: &serde_json::Value) -> G2Affine {
    let a = v.as_array().unwrap();
    let c = |i: usize| {
        let p = a[i].as_array().unwrap();
        Fq2::new(fq(&p[0]), fq(&p[1]))
    };
    let z = c(2);
    if z.is_zero() {
        G2Affine::identity()
    } else {
        assert_eq!(z, Fq2::ONE, "expected an affine point");
        G2Affine::new(c(0), c(1))
    }
}

#[test]
fn five_msms_match_snarkjs() {
    let found = artifacts();
    if found.is_empty() {
        eprintln!("SKIPPED: no artifact carries msm_expected.json");
        return;
    }
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let circuit = CpuBackend::new().prepare(pk).unwrap();
        let mut t = StageTimings::default();
        let h = circuit.compute_h(&w, &mut t).unwrap();
        let m = circuit.msms(&w, &h, &mut t).unwrap();

        let text = std::fs::read_to_string(dir.join("msm_expected.json")).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();

        assert_eq!(m.a_g1.into_affine(), g1(&v["a_g1"]), "{name}: A MSM");
        assert_eq!(m.b_g1.into_affine(), g1(&v["b_g1"]), "{name}: B1 MSM");
        assert_eq!(m.b_g2.into_affine(), g2(&v["b_g2"]), "{name}: B2 MSM");
        assert_eq!(m.l_g1.into_affine(), g1(&v["l_g1"]), "{name}: L MSM");
        assert_eq!(m.h_g1.into_affine(), g1(&v["h_g1"]), "{name}: H MSM");
        eprintln!("{name}: all five MSMs match snarkjs");
    }
}
