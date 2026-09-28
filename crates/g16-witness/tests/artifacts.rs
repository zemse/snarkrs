//! Every circuit in `bench/artifacts` that ships its wasm: the witness we compute from its
//! `input.json` is byte for byte the checked-in `circuit.wtns`, and byte for byte what
//! `snarkjs wtns calculate` writes today. Skips loudly without the artifacts or snarkjs.
#![cfg(feature = "wasm")]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use g16_witness::{wtns_bytes, Input, WitnessCalculator};

fn artifacts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

/// `(name, wasm, dir)` for every artifact with an input, a witness and a wasm, whichever
/// of circom's `<name>_js/<name>.wasm` layouts it has.
fn circuits() -> Vec<(String, PathBuf, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(artifacts()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let dir = e.path();
        if !dir.join("input.json").is_file() || !dir.join("circuit.wtns").is_file() {
            continue;
        }
        let Ok(sub) = std::fs::read_dir(&dir) else {
            continue;
        };
        for s in sub.flatten() {
            let js = s.path();
            let name = js.file_name().unwrap().to_string_lossy().into_owned();
            let Some(stem) = name.strip_suffix("_js") else {
                continue;
            };
            let wasm = js.join(format!("{stem}.wasm"));
            if wasm.is_file() {
                let n = dir.file_name().unwrap().to_string_lossy().into_owned();
                out.push((n, wasm, dir.clone()));
            }
        }
    }
    out.sort();
    out
}

fn snarkjs_bin() -> Option<String> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    let out = Command::new(&bin).arg("--help").output().ok()?;
    String::from_utf8_lossy(&out.stdout)
        .contains("snarkjs@0.7.6")
        .then_some(bin)
}

#[test]
fn every_artifact_matches_its_wtns_and_snarkjs() {
    let all = circuits();
    if all.is_empty() {
        eprintln!("SKIPPED every_artifact_matches_its_wtns_and_snarkjs: no bench/artifacts");
        return;
    }
    let js = snarkjs_bin();
    if js.is_none() {
        eprintln!("SKIPPED the snarkjs half: set SNARKJS to snarkjs 0.7.6");
    }
    let scratch =
        std::env::temp_dir().join(format!("g16-witness-artifacts-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for (name, wasm, dir) in &all {
        let started = std::time::Instant::now();
        let calc = WitnessCalculator::from_file(wasm).unwrap();
        let compiled = started.elapsed();
        let text = std::fs::read_to_string(dir.join("input.json")).unwrap();
        let input = Input::from_json_str(&text).unwrap();
        let w = calc
            .calculate(&input)
            .unwrap_or_else(|e| panic!("{name}: {}", e.snarkjs_line()));
        let ours = wtns_bytes(&w);
        eprintln!(
            "{name}: {} wires, compiled in {compiled:.2?}, witness in {:.2?}",
            w.len(),
            started.elapsed() - compiled
        );
        let checked_in = std::fs::read(dir.join("circuit.wtns")).unwrap();
        assert!(ours == checked_in, "{name}: differs from circuit.wtns");

        if let Some(js) = &js {
            let out = scratch.join(format!("{name}.wtns"));
            let status = Command::new(js)
                .arg("wtns")
                .arg("calculate")
                .arg(wasm)
                .arg(dir.join("input.json"))
                .arg(&out)
                .stdout(Stdio::null())
                .status()
                .unwrap();
            assert!(status.success(), "{name}: snarkjs failed");
            assert!(
                ours == std::fs::read(&out).unwrap(),
                "{name}: differs from snarkjs"
            );
        }
    }
    std::fs::remove_dir_all(&scratch).ok();
}

/// tornado's wasm is circom 1. snarkjs runs it; this refuses it by name.
#[test]
fn circom_1_is_refused_clearly() {
    let wasm = artifacts().join("tornado/circuit.wasm");
    if !wasm.is_file() {
        eprintln!("SKIPPED circom_1_is_refused_clearly: no tornado artifact");
        return;
    }
    let e = WitnessCalculator::from_file(&wasm)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("circom 1"), "{e}");
}
