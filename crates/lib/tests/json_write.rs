//! `write_proof` and `write_public` publish through a stage: a failure leaves the old file
//! as it was with no stage behind, success replaces it, and a new file gets the umask's
//! mode. The one test that sets the umask, which is process wide, is in this binary.

use std::path::{Path, PathBuf};

use snarkrs_field::{AffineRepr, G1Affine, G2Affine};
use snarkrs_lib::json::{proof_to_string, public_to_string};
use snarkrs_lib::{write_proof, write_public, Fr, Proof};

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "snarkrs-lib-json-write-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

fn proof() -> Proof {
    Proof {
        a: G1Affine::generator(),
        b: G2Affine::generator(),
        c: G1Affine::generator(),
    }
}

const PUBLIC: [u64; 2] = [7, 11];

fn public() -> Vec<Fr> {
    PUBLIC.iter().map(|&x| Fr::from(x)).collect()
}

/// Both writers, so each test runs over the pair: what they write and how they fail.
type Writer = fn(&Path) -> Result<(), snarkrs_lib::json::FileError>;

fn writers() -> [(&'static str, Writer, String, &'static str); 2] {
    [
        (
            "proof.json",
            |p| write_proof(p, &proof()),
            proof_to_string(&proof()),
            "writing proof to",
        ),
        (
            "public.json",
            |p| write_public(p, &public()),
            public_to_string(&public()),
            "writing public signals to",
        ),
    ]
}

#[test]
fn success_replaces_the_file() {
    let dir = dir("replace");
    for (name, write, text, _) in writers() {
        let out = dir.join(name);
        std::fs::write(&out, b"old").unwrap();
        write(&out).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), text);
    }
    assert_eq!(entries(&dir), ["proof.json", "public.json"], "orphan stage");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_directory_destination_is_left_alone_with_no_stage() {
    let dir = dir("directory");
    std::fs::write(dir.join("keep"), b"old").unwrap();
    for (_, write, _, what) in writers() {
        let taken = dir.join("taken");
        std::fs::create_dir_all(&taken).unwrap();
        std::fs::write(taken.join("sentinel"), b"old").unwrap();
        let e = write(&taken).unwrap_err();
        assert!(
            e.to_string()
                .starts_with(&format!("{what} {}: ", taken.display())),
            "{e}"
        );
        assert_eq!(std::fs::read(taken.join("sentinel")).unwrap(), b"old");
        assert_eq!(entries(&dir), ["keep", "taken"], "orphan stage");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// `umask(2)`, declared here because this crate has no libc dependency. `mode_t` is 16
/// bits on Apple platforms and 32 elsewhere.
#[cfg(unix)]
unsafe fn umask(mask: u32) -> u32 {
    #[cfg(target_vendor = "apple")]
    type ModeT = u16;
    #[cfg(not(target_vendor = "apple"))]
    type ModeT = u32;
    extern "C" {
        fn umask(mask: ModeT) -> ModeT;
    }
    u32::from(umask(mask as ModeT))
}

#[cfg(unix)]
#[test]
fn a_new_file_follows_the_umask_and_an_existing_one_keeps_its_mode() {
    use std::os::unix::fs::PermissionsExt;
    let dir = dir("mode");
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    for (name, write, text, _) in writers() {
        let out = dir.join(name);
        // SAFETY: umask has no preconditions.
        let old = unsafe { umask(0o022) };
        let wrote = write(&out);
        unsafe { umask(old) };
        wrote.unwrap();
        assert_eq!(mode(&out), 0o644, "{name}");
        std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o640)).unwrap();
        write(&out).unwrap();
        assert_eq!(mode(&out), 0o640, "{name}");
        assert_eq!(std::fs::read_to_string(&out).unwrap(), text);
    }
    assert_eq!(entries(&dir), ["proof.json", "public.json"], "orphan stage");
    std::fs::remove_dir_all(dir).unwrap();
}
