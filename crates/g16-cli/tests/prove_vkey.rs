//! `g16 prove --vkey`, and the `n_public` the zkey declares.
//!
//! `n_public` decides how much of the witness `prove` writes to public.json, and it comes
//! from the zkey. A key that overstates it publishes private wires, and its own verifying key
//! can be made to agree, so self-verify against the zkey cannot catch it. These tests drive
//! the real binary, because the exit status, the message and the absence of a proof.json are
//! all part of the contract.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn artifact(name: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .join(name);
    dir.join("circuit.zkey").is_file().then_some(dir)
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-vkey-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn prove(zkey: &Path, wtns: &Path, out: &Path, vkey: Option<&Path>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_g16"));
    cmd.args(["prove", "--backend", "cpu", "--zkey"])
        .arg(zkey)
        .arg("--witness")
        .arg(wtns)
        .arg("--proof")
        .arg(out.join("proof.json"))
        .arg("--public")
        .arg(out.join("public.json"));
    if let Some(v) = vkey {
        cmd.arg("--vkey").arg(v);
    }
    cmd.output().unwrap()
}

fn said(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn a_matching_vkey_proves_and_a_foreign_one_is_refused_before_proving() {
    let (Some(tiny), Some(other)) = (artifact("tiny_mul"), artifact("js_1x1_d8")) else {
        eprintln!("SKIPPED prove_vkey: needs tiny_mul and js_1x1_d8");
        return;
    };
    let out = scratch("match");
    let zkey = tiny.join("circuit.zkey");
    let wtns = tiny.join("circuit.wtns");

    let o = prove(&zkey, &wtns, &out, Some(&tiny.join("vkey.json")));
    assert!(o.status.success(), "matching vkey: {}", said(&o));
    assert!(out.join("proof.json").is_file());
    std::fs::remove_file(out.join("proof.json")).unwrap();

    let o = prove(&zkey, &wtns, &out, Some(&other.join("vkey.json")));
    assert_eq!(o.status.code(), Some(1), "foreign vkey: {}", said(&o));
    assert!(
        said(&o).contains("they are not the same circuit"),
        "the error did not name the problem: {}",
        said(&o)
    );
    assert!(
        !out.join("proof.json").exists(),
        "a refused run wrote a proof"
    );
    std::fs::remove_dir_all(&out).ok();
}

/// A zkey rebuilt so it calls every wire public: `n_public = n_vars - 1`, section 3 grown to
/// match and section 8 emptied. It parses, and proving it would write the whole witness to
/// public.json. Without `--vkey` the CLI must refuse it before it proves.
#[test]
fn a_key_that_calls_every_wire_public_is_refused_without_a_vkey() {
    let Some(tiny) = artifact("tiny_mul") else {
        eprintln!("SKIPPED all_public: no tiny_mul");
        return;
    };
    let out = scratch("all-public");
    let bytes = std::fs::read(tiny.join("circuit.zkey")).unwrap();
    let mut sections = split(&bytes);

    // Section 2: n8q, q, n8r, r, then n_vars and n_public.
    let hdr = &mut sections.iter_mut().find(|(id, _)| *id == 2).unwrap().1;
    let n_vars = u32::from_le_bytes(hdr[72..76].try_into().unwrap()) as usize;
    let n_public = u32::from_le_bytes(hdr[76..80].try_into().unwrap()) as usize;
    let lie = n_vars - 1;
    hdr[76..80].copy_from_slice(&(lie as u32).to_le_bytes());

    // IC gets one point per extra public input (a copy of IC[0], which is on the curve), and
    // the L query loses one point per wire that stopped being private.
    const G1: usize = 64;
    let extra = lie - n_public;
    let ic = &mut sections.iter_mut().find(|(id, _)| *id == 3).unwrap().1;
    let first = ic[..G1].to_vec();
    for _ in 0..extra {
        ic.extend_from_slice(&first);
    }
    let l = &mut sections.iter_mut().find(|(id, _)| *id == 8).unwrap().1;
    l.truncate(l.len() - extra * G1);

    let zkey = out.join("all-public.zkey");
    std::fs::write(&zkey, join(&bytes[..8], &sections)).unwrap();

    let o = prove(&zkey, &tiny.join("circuit.wtns"), &out, None);
    assert_eq!(o.status.code(), Some(1), "all-public key: {}", said(&o));
    assert!(
        said(&o).contains("declares all"),
        "the error did not name the problem: {}",
        said(&o)
    );
    assert!(
        !out.join("public.json").exists(),
        "the witness reached public.json"
    );
    std::fs::remove_dir_all(&out).ok();
}

/// An iden3 binfile: magic and version (8 bytes), a u32 section count, then per section a
/// u32 id, a u64 length and the payload.
fn split(bytes: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let mut pos = 12;
    let mut out = Vec::new();
    for _ in 0..n {
        let id = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        let len = u64::from_le_bytes(bytes[pos + 4..pos + 12].try_into().unwrap()) as usize;
        pos += 12;
        out.push((id, bytes[pos..pos + len].to_vec()));
        pos += len;
    }
    out
}

fn join(head: &[u8], sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = head.to_vec();
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (id, payload) in sections {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}
