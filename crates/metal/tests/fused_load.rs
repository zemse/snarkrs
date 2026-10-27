//! `MetalBackend::load`, the zkey decoded straight into the device's buffers, against the
//! two-step `ProvingKey::load` then `prepare` it replaces: the same proofs, bit for bit, and
//! the same refusals.

#![cfg(target_os = "macos")]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use std::path::{Path, PathBuf};

use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::{
    cpu::CpuBackend, prove::prove_with_blinders, verify::verify, Backend, PreparedCircuit,
    StageTimings,
};
use snarkrs_metal::{LoadError, MetalBackend};

/// Every artifact under `bench/artifacts`, one level into containers such as `csp/`, as
/// `audit_metal_vs_cpu.rs` walks them.
fn artifacts(test: &str) -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<(String, PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for d in entries.flatten().map(|e| e.path()).filter(|d| d.is_dir()) {
            let name = format!("{prefix}{}", d.file_name().unwrap().to_string_lossy());
            if name == "large" {
                continue;
            }
            if d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file() {
                out.push((name, d));
            } else if depth > 0 {
                walk(&d, &format!("{name}/"), depth - 1, out);
            }
        }
    }
    let mut out = Vec::new();
    if let Ok(root) = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
    {
        walk(&root, "", 1, &mut out);
    }
    out.sort();
    if out.is_empty() {
        resources::skip(format_args!("{test}: no artifacts under bench/artifacts"));
    }
    out
}

fn zkey_len(dir: &Path) -> u64 {
    std::fs::metadata(dir.join("circuit.zkey")).unwrap().len()
}

fn proof_of(c: &dyn PreparedCircuit, witness: &[Fr]) -> (String, String, String) {
    let p = prove_with_blinders(
        c,
        witness,
        Fr::from(7u64),
        Fr::from(9u64),
        &mut StageTimings::default(),
    )
    .unwrap();
    (p.a.to_string(), p.b.to_string(), p.c.to_string())
}

/// The fused load proves what the CPU proves at the same blinders, on every artifact, and
/// the proof verifies against snarkjs' own vkey. On the keys small enough to afford it the
/// constant-work backend is checked too, which can put B in G2 on the device.
#[test]
fn the_fused_load_proves_what_the_cpu_proves() {
    let metal = MetalBackend::new().expect("Metal device");
    let constant = MetalBackend::constant_work().expect("Metal device");
    for (name, dir) in artifacts("the_fused_load_proves_what_the_cpu_proves") {
        let zkey = dir.join("circuit.zkey");
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let cpu = CpuBackend::new()
            .prepare(ProvingKey::load(&zkey).unwrap())
            .unwrap();
        let want = proof_of(cpu.as_ref(), &witness);
        drop(cpu);

        let fused = metal.load(&zkey).unwrap();
        assert_eq!(fused.backend_name(), "metal");
        assert_eq!(
            proof_of(&fused, &witness),
            want,
            "{name}: fused proof differs"
        );

        let p = prove_with_blinders(
            &fused,
            &witness,
            Fr::from(3u64),
            Fr::from(5u64),
            &mut StageTimings::default(),
        )
        .unwrap();
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        verify(&vk, &witness[1..=fused.n_public()], &p)
            .unwrap_or_else(|e| panic!("{name}: fused proof rejected by vkey.json: {e}"));
        drop(fused);

        if zkey_len(&dir) <= 100 << 20 {
            let fused = constant.load(&zkey).unwrap();
            assert_eq!(
                proof_of(&fused, &witness),
                want,
                "{name}: constant-work fused proof differs"
            );
        }
        println!("FUSED {name:<20} proof identical to the CPU's, and verifies");
    }
}

fn mutated(dir: &Path, what: &str, edit: impl Fn(&mut Vec<u8>)) -> PathBuf {
    let mut bytes = std::fs::read(dir.join("circuit.zkey")).unwrap();
    edit(&mut bytes);
    let path = std::env::temp_dir().join(format!(
        "snarkrs-metal-fused-{what}-{}.zkey",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    path
}

/// Offset of the first section `id`'s payload.
fn section_at(bytes: &[u8], id: u32) -> usize {
    let nsec = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut off = 12usize;
    for _ in 0..nsec {
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let slen = u64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap()) as usize;
        off += 12;
        if sid == id {
            return off;
        }
        off += slen;
    }
    panic!("no section {id}");
}

/// `delta_g2` overwritten with `gamma_g2`, the shape of a key with no phase-2 contribution.
fn unblind(b: &mut [u8]) {
    let s2 = section_at(b, 2);
    let n8q = u32::from_le_bytes(b[s2..s2 + 4].try_into().unwrap()) as usize;
    let p = s2 + 4 + n8q;
    let n8r = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
    let alpha1 = p + 4 + n8r + 12;
    let gamma2 = alpha1 + 64 + 64 + 128;
    let delta2 = gamma2 + 128 + 64;
    b.copy_within(gamma2..gamma2 + 128, delta2);
}

/// A bad key is refused by the fused load with the error `ProvingKey::load` gives, and the
/// unblinded opt-in takes an unblinded key while `load` still refuses it.
#[test]
fn the_fused_load_refuses_what_the_loader_refuses() {
    let metal = MetalBackend::new().expect("Metal device");
    let Some((name, dir)) = artifacts("the_fused_load_refuses_what_the_loader_refuses")
        .into_iter()
        .min_by_key(|(_, d)| zkey_len(d))
    else {
        return;
    };
    type Edit = fn(&mut Vec<u8>);
    let edits: [(&str, Edit); 3] = [
        ("a-off-curve", |b| {
            let at = section_at(b, 5) + 32;
            b[at] ^= 1;
        }),
        ("h-off-curve", |b| {
            let at = section_at(b, 9) + 64 + 32;
            b[at] ^= 1;
        }),
        ("coef-signal", |b| {
            let at = section_at(b, 4) + 4 + 8;
            b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }),
    ];
    for (what, edit) in edits {
        let path = mutated(&dir, what, edit);
        let want = ProvingKey::load(&path)
            .err()
            .expect("load accepted it")
            .to_string();
        let got = metal.load(&path);
        let _ = std::fs::remove_file(&path);
        match got {
            Err(LoadError::Key(e)) => assert_eq!(e.to_string(), want, "{name}: {what}"),
            Err(e) => panic!("{name}: {what}: not a key error: {e}"),
            Ok(_) => panic!("{name}: {what}: the fused load accepted it"),
        }
    }

    let path = mutated(&dir, "unblinded", |b| unblind(b));
    let refused = metal.load(&path);
    let taken = metal.load_allowing_unblinded(&path);
    let _ = std::fs::remove_file(&path);
    match refused {
        Err(LoadError::Key(e)) => assert!(
            e.to_string().contains("gamma_g2 equals delta_g2"),
            "{name}: {e}"
        ),
        Err(e) => panic!("{name}: unblinded: not a key error: {e}"),
        Ok(_) => panic!("{name}: the fused load accepted an unblinded key"),
    }
    let c = taken.unwrap_or_else(|e| panic!("{name}: the opt-in refused it: {e}"));
    assert!(c.key().vk.gamma_g2 == c.key().vk.delta_g2, "{name}");
}
