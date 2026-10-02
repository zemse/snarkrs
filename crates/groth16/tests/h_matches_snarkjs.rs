//! Cross-check stages 0-4 against snarkjs' own pipeline, element by element.
//!
//! `h_expected.json` is produced by replaying snarkjs' `buildABC1` + `ifft` +
//! `batchApplyKey(inc)` + `fft` + `joinABC` in ffjavascript and dumping the resulting
//! scalars as decimal strings. If our H agrees with that vector at every index then the
//! coset shift, the transform ordering, the iNTT normalisation and the "no Z division"
//! convention are all right; a pairing check alone cannot separate those.
//!
//! `bench/artifacts/` is gitignored, so a fresh clone has nothing to replay and this test
//! skips. Set `G16_REQUIRE_VECTORS` to make that a failure instead: a test that reports
//! `ok` having asserted nothing is worse than one that is absent.

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey};
use snarkrs_groth16::{cpu::CpuBackend, Backend, StageTimings};
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
        .filter(|d| d.is_dir())
        .filter(|d| {
            resources::complete_files(
                d,
                &["h_expected.json", "circuit.zkey", "circuit.wtns"],
                "h_matches_snarkjs_element_by_element",
            )
        })
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

#[test]
fn h_matches_snarkjs_element_by_element() {
    let found = artifacts();
    if found.is_empty() {
        resources::skip_vector("h_matches_snarkjs_element_by_element: no complete artifact with h_expected.json under bench/artifacts");
        return;
    }
    for (name, dir) in found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let circuit = CpuBackend::new().prepare(pk).unwrap();
        let mut t = StageTimings::default();
        let got = circuit.compute_h(&w, &mut t).unwrap();
        let got = got.to_host().unwrap();

        let text = std::fs::read_to_string(dir.join("h_expected.json")).unwrap();
        let want: Vec<String> = serde_json::from_str(&text).unwrap();
        assert_eq!(got.len(), want.len(), "{name}: H length");
        for (i, (g, wexp)) in got.iter().zip(&want).enumerate() {
            let n: num_bigint::BigUint = wexp.parse().unwrap();
            let expected = Fr::from(n);
            assert_eq!(*g, expected, "{name}: H[{i}]");
        }
        eprintln!("{name}: H matches snarkjs at all {} indices", got.len());
    }
}
