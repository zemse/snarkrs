//! The witness file writers publish through a stage: a failure leaves the old file as it
//! was with no stage behind, success replaces it, and a new file gets the umask's mode.
//! One test per binary sets the umask, which is process wide, so this file holds it.

use std::path::{Path, PathBuf};

use snarkrs_field::Fr;
use snarkrs_witness::{write_wtns, wtns_bytes};

fn dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "snarkrs-witness-write-{name}-{}",
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

fn witness(n: u64) -> Vec<Fr> {
    (0..n).map(|i| Fr::from(i * i + 1)).collect()
}

#[test]
fn write_wtns_replaces_the_file_and_a_directory_destination_is_left_alone() {
    let dir = dir("wtns");
    let out = dir.join("witness.wtns");
    std::fs::write(&out, b"old witness").unwrap();
    write_wtns(&out, &witness(4)).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), wtns_bytes(&witness(4)));
    assert_eq!(entries(&dir), ["witness.wtns"]);

    let taken = dir.join("taken");
    std::fs::create_dir(&taken).unwrap();
    std::fs::write(taken.join("sentinel"), b"old").unwrap();
    let e = write_wtns(&taken, &witness(4)).unwrap_err();
    assert!(
        e.to_string()
            .starts_with(&format!("writing {}: ", taken.display())),
        "{e}"
    );
    assert_eq!(std::fs::read(taken.join("sentinel")).unwrap(), b"old");
    assert_eq!(entries(&dir), ["taken", "witness.wtns"], "orphan stage");
    assert_eq!(std::fs::read(&out).unwrap(), wtns_bytes(&witness(4)));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn write_wtns_into_a_missing_directory_fails_naming_the_destination() {
    let dir = dir("missing");
    let out = dir.join("no").join("witness.wtns");
    let e = write_wtns(&out, &witness(2)).unwrap_err();
    assert!(
        e.to_string()
            .starts_with(&format!("writing {}: ", out.display())),
        "{e}"
    );
    assert!(entries(&dir).is_empty());
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
fn a_new_witness_follows_the_umask_and_an_existing_one_keeps_its_mode() {
    use std::os::unix::fs::PermissionsExt;
    let dir = dir("mode");
    let out = dir.join("witness.wtns");
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    // SAFETY: umask has no preconditions.
    let old = unsafe { umask(0o022) };
    let wrote = write_wtns(&out, &witness(3));
    unsafe { umask(old) };
    wrote.unwrap();
    assert_eq!(mode(&out), 0o644);
    std::fs::set_permissions(&out, std::fs::Permissions::from_mode(0o640)).unwrap();
    write_wtns(&out, &witness(5)).unwrap();
    assert_eq!(mode(&out), 0o640);
    assert_eq!(std::fs::read(&out).unwrap(), wtns_bytes(&witness(5)));
    assert_eq!(entries(&dir), ["witness.wtns"]);
    std::fs::remove_dir_all(dir).unwrap();
}
