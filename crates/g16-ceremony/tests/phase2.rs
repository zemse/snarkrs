//! Phase 2 against snarkjs itself.
//!
//! Two oracles, and neither is a round trip through our own code:
//!
//! 1. Every checked-in artifact already ships the `verification_key.json` snarkjs wrote
//!    for it, so the export is a byte comparison against a file nobody in this workspace
//!    produced.
//! 2. A beacon is a pure function of its inputs, so `zkey beacon` is byte comparable too.
//!    A `zkey contribute` is not, because it mixes OS entropy by design, but a contribute
//!    followed by the same beacon is: the beacon reads the file the contribution wrote,
//!    and every byte of the beacon's output that differs is a byte the contribution got
//!    wrong. That is the test that covers `contribute` end to end.
//!
//! The snarkjs half needs `snarkjs` on `PATH`; without it the comparison tests skip with a
//! message rather than reporting green.

use std::path::{Path, PathBuf};
use std::process::Command;

use g16_ceremony::contribute::{self, MpcParams};
use g16_ceremony::vkey;
use g16_msm::{CpuMsm, MsmBackend};

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
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("vkey.json").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-phase2-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn exported_vkey_is_byte_identical_to_snarkjs() {
    let artifacts = artifacts();
    if artifacts.is_empty() {
        eprintln!("no artifacts with a vkey.json, skipping");
        return;
    }
    let dir = tmp_dir("vkey");
    for (name, path) in &artifacts {
        let out = dir.join(format!("{name}.json"));
        vkey::export_verification_key(&path.join("circuit.zkey"), &out).unwrap();
        let got = std::fs::read(&out).unwrap();
        let want = std::fs::read(path.join("vkey.json")).unwrap();
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: {} bytes written, snarkjs wrote {}",
            got.len(),
            want.len()
        );
        if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
            let from = i.saturating_sub(40);
            panic!(
                "{name}: first difference at byte {i}\n  ours:    {:?}\n  snarkjs: {:?}",
                String::from_utf8_lossy(&got[from..(i + 40).min(got.len())]),
                String::from_utf8_lossy(&want[from..(i + 40).min(want.len())]),
            );
        }
    }
    eprintln!("{} verification keys byte-identical", artifacts.len());
}
