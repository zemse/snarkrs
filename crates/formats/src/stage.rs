//! Publishing a file through a synced stage: the output is written to an exclusively
//! reserved sibling and renamed over the destination only once it is complete, so a failed
//! write leaves the destination as it was and no stage behind.
//!
//! This is the one copy every crate that writes a user-named path uses: the ceremony's
//! keys, the witness writers and the proof writers. It lives here because snarkrs-formats
//! is the lowest crate all of them already depend on.
//!
//! The operation must flush its writers and drop all file handles before returning. All
//! rereads of the output must use the supplied stage path. An error leaves the destination
//! unchanged; cleanup is best effort and never replaces the primary error.
//!
//! The stage is fsynced before the rename and, on Unix, the directory after it. That
//! gives atomic visibility to other readers, not crash durability: macOS `fsync` does not
//! flush the drive cache (`F_FULLFSYNC` would), a failed directory sync is ignored because
//! the destination is already replaced by then, and nothing here checks what the
//! operation wrote. It assumes a trusted output directory, not adversarial path
//! replacement.
//!
//! The operation writes only to the stage, so an output that is also one of its inputs
//! (the same path, a path through `..`, or a symlink to it) reads the old input to the
//! end and is then replaced by the result. A symlinked destination is followed, as
//! `File::create` and snarkjs follow it: the stage goes beside the final target and
//! replaces that, and the link stays a link. A hard-linked destination gets a new inode,
//! so its other names keep the old bytes.
//!
//! Name separation covers case folding, canonical Unicode normalization and trailing
//! dot/space trimming, which preserve ASCII decimal digits and their order. DOS 8.3,
//! device aliases and other alternate-name namespaces are outside this guarantee.
//! Existing regular-file permissions are copied at commit; ownership, ACLs and hard-link
//! identity are not preserved. Inherited ACLs remain filesystem-dependent.

use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Run `operation` on one stage per destination, in order, then rename every stage over
/// its destination. New files and in-progress stages are 0600 on Unix and platform
/// defaults elsewhere.
///
/// Nothing is renamed until `operation` has returned and every stage is synced, so an
/// error up to that point leaves every destination as it was and removes every stage. The
/// renames themselves are not one atomic step: a crash, or a rename that fails, between
/// two of them leaves the earlier destinations replaced and the later ones not. A
/// destination named twice is renamed over twice, and the later stage wins. A destination
/// that is a directory is refused before anything is staged, since its rename would fail
/// only after the others had run.
pub fn staged_all<T, E: From<std::io::Error>>(
    destinations: &[&Path],
    operation: impl FnOnce(&[PathBuf]) -> Result<T, E>,
) -> Result<T, E> {
    staged_all_with_counter(destinations, &NEXT_STAGE, true, operation)
}

/// [`staged_all`] for outputs meant to be read by others, such as a proof, a witness or a
/// verifier contract. A new destination gets the mode a plain `File::create` would give
/// it, `0666` less the umask, because its stage is created with the default open mode
/// rather than 0600; the stage is that readable while it is being written. An existing
/// destination keeps its mode, as in [`staged_all`].
pub fn staged_all_default_mode<T, E: From<std::io::Error>>(
    destinations: &[&Path],
    operation: impl FnOnce(&[PathBuf]) -> Result<T, E>,
) -> Result<T, E> {
    staged_all_with_counter(destinations, &NEXT_STAGE, false, operation)
}

static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

/// One destination of [`staged_all`], from reservation to rename.
struct Pending {
    destination: PathBuf,
    parent: PathBuf,
    permissions: Option<std::fs::Permissions>,
    stage: Stage,
}

fn staged_all_with_counter<T, E: From<std::io::Error>>(
    destinations: &[&Path],
    next: &AtomicU64,
    private: bool,
    operation: impl FnOnce(&[PathBuf]) -> Result<T, E>,
) -> Result<T, E> {
    let mut pending = Vec::with_capacity(destinations.len());
    for destination in destinations {
        let destination = follow_links(destination)?;
        let Some(file_name) = destination.file_name() else {
            return Err(
                std::io::Error::new(ErrorKind::InvalidInput, "output has no file name").into(),
            );
        };
        let permissions = match std::fs::symlink_metadata(&destination) {
            Ok(meta) if meta.is_dir() => {
                return Err(std::io::Error::new(
                    ErrorKind::IsADirectory,
                    "the output is a directory",
                )
                .into())
            }
            Ok(meta) => meta.is_file().then(|| meta.permissions()),
            Err(e) if e.kind() == ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let stage = Stage::create_with(&parent, file_name, next, private)?;
        pending.push(Pending {
            destination,
            parent,
            permissions,
            stage,
        });
    }
    // Each stage avoids its own destination's name only. One that took another member's
    // name would be renamed over before its own turn, so the set is refused instead.
    for p in &pending {
        let stage = p.stage.path.as_ref().expect("uncommitted stage has a path");
        let clash = pending.iter().any(|q| {
            q.parent == p.parent
                && q.destination
                    .file_name()
                    .zip(stage.file_name())
                    .is_some_and(|(a, b)| a.eq_ignore_ascii_case(b))
        });
        if clash {
            return Err(std::io::Error::new(
                ErrorKind::AlreadyExists,
                "an output is named like a snarkrs stage",
            )
            .into());
        }
    }
    let paths: Vec<PathBuf> = pending
        .iter()
        .map(|p| p.stage.path.clone().expect("uncommitted stage has a path"))
        .collect();
    let result = operation(&paths)?;
    for (p, path) in pending.iter().zip(&paths) {
        if let Some(permissions) = &p.permissions {
            std::fs::set_permissions(path, permissions.clone())?;
        }
        File::open(path)?.sync_all()?;
    }
    for (p, path) in pending.iter_mut().zip(&paths) {
        std::fs::rename(path, &p.destination)?;
        p.stage.path = None;
    }
    #[cfg(unix)]
    for (i, p) in pending.iter().enumerate() {
        if pending[..i].iter().any(|q| q.parent == p.parent) {
            continue;
        }
        if let Ok(dir) = File::open(&p.parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(result)
}

/// `path` with every symlink in its last component followed, dangling ones included, so
/// that a link is written through rather than replaced.
fn follow_links(path: &Path) -> std::io::Result<PathBuf> {
    let mut path = path.to_path_buf();
    // Linux's MAXSYMLINKS.
    for _ in 0..40 {
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let target = std::fs::read_link(&path)?;
                path = match path.parent() {
                    Some(parent) => parent.join(target),
                    None => target,
                };
            }
            Ok(_) => return Ok(path),
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(path),
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        ErrorKind::InvalidInput,
        "too many levels of symbolic links",
    ))
}

/// A reserved, uniquely named file that is removed when it drops, unless committed.
pub struct Stage {
    path: Option<PathBuf>,
}

impl Stage {
    /// A private scratch file in `beside`'s directory, under a name no user file has.
    pub fn beside(beside: &Path) -> std::io::Result<Self> {
        let parent = beside
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let excluded = beside.file_name().unwrap_or_default();
        Self::create(parent, excluded, &NEXT_STAGE)
    }

    pub fn path(&self) -> &Path {
        self.path.as_deref().expect("uncommitted stage has a path")
    }

    fn create(
        parent: &Path,
        excluded: &std::ffi::OsStr,
        next: &AtomicU64,
    ) -> std::io::Result<Self> {
        Self::create_with(parent, excluded, next, true)
    }

    /// `private` makes the stage 0600 on Unix; otherwise it gets the default open mode.
    fn create_with(
        parent: &Path,
        excluded: &std::ffi::OsStr,
        next: &AtomicU64,
        private: bool,
    ) -> std::io::Result<Self> {
        let first_digit = excluded
            .as_encoded_bytes()
            .iter()
            .find(|b| b.is_ascii_digit());
        let tag = u8::from(first_digit == Some(&b'0'));
        for _ in 0..128 {
            let id = next.fetch_add(1, Ordering::Relaxed);
            // The first ASCII digit differs even when the filesystem folds the prefix.
            let name = format!(".snarkrs-stage-{tag}-{}-{id}", std::process::id());
            if std::ffi::OsStr::new(&name) == excluded {
                continue;
            }
            let path = parent.join(name);
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            if private {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            #[cfg(not(unix))]
            let _ = private;
            match options.open(&path) {
                Ok(file) => {
                    drop(file);
                    return Ok(Self { path: Some(path) });
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            ErrorKind::AlreadyExists,
            "could not reserve a snarkrs stage after 128 attempts",
        ))
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("snarkrs-stage-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn staged_one<T>(
        destination: &Path,
        next: &AtomicU64,
        operation: impl FnOnce(&Path) -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        staged_all_with_counter(&[destination], next, true, |stages| operation(&stages[0]))
    }

    fn check_requested_destination(fail: bool) {
        let dir = dir(if fail {
            "requested-error"
        } else {
            "requested-success"
        });
        let name = format!(".snarkrs-stage-{}-0", std::process::id());
        for parent in [dir.clone(), dir.join(".")] {
            let out = parent.join(&name);
            assert!(!out.exists());
            let result = staged_one(&out, &AtomicU64::new(0), |stage| {
                assert!(
                    !out.exists(),
                    "reserved stage is visible at the requested destination"
                );
                std::fs::write(stage, b"complete output")?;
                assert!(!out.exists(), "output published before operation returned");
                if fail {
                    return Err(std::io::Error::other("injected failure"));
                }
                Ok(7)
            });
            if fail {
                assert_eq!(result.unwrap_err().to_string(), "injected failure");
                assert!(!out.exists());
            } else {
                assert_eq!(result.unwrap(), 7);
                assert_eq!(std::fs::read(&out).unwrap(), b"complete output");
                assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
                std::fs::remove_file(&out).unwrap();
            }
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "orphan stage");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn requested_destination_is_never_used_as_stage_on_success() {
        check_requested_destination(false);
    }

    #[test]
    fn requested_destination_is_never_used_as_stage_on_failure() {
        check_requested_destination(true);
    }

    fn check_case_alias_destination(fail: bool) {
        let dir = dir(if fail {
            "case-alias-error"
        } else {
            "case-alias-success"
        });
        let old_name = format!(".snarkrs-stage-{}-0", std::process::id());
        for alias in [
            old_name.to_ascii_uppercase(),
            old_name.replacen('k', "K", 1),
            old_name.replacen('s', "ſ", 1),
        ] {
            let probe = dir.join(&old_name);
            std::fs::write(&probe, b"probe").unwrap();
            let aliases = dir.join(&alias).exists();
            std::fs::remove_file(&probe).unwrap();
            if !aliases {
                eprintln!(
                    "filesystem does not alias {alias:?} to {old_name:?}; skipping this alias"
                );
                continue;
            }
            eprintln!("verified filesystem alias: {alias:?}");
            for parent in [dir.clone(), dir.join(".")] {
                let out = parent.join(&alias);
                assert!(!out.exists());
                let result = staged_one(&out, &AtomicU64::new(0), |stage| {
                    assert!(
                        !out.exists(),
                        "case-equivalent stage is visible at destination"
                    );
                    std::fs::write(stage, b"complete output")?;
                    assert!(!out.exists(), "output published before operation returned");
                    if fail {
                        return Err(std::io::Error::other("injected failure"));
                    }
                    Ok(7)
                });
                if fail {
                    assert!(result.is_err());
                    assert!(!out.exists());
                } else {
                    assert_eq!(result.unwrap(), 7);
                    assert_eq!(std::fs::read(&out).unwrap(), b"complete output");
                    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
                    std::fs::remove_file(&out).unwrap();
                }
                assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "orphan stage");
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn case_alias_destination_stays_absent_until_success() {
        check_case_alias_destination(false);
    }

    #[test]
    fn case_alias_destination_stays_absent_on_failure() {
        check_case_alias_destination(true);
    }

    #[test]
    fn stage_discriminator_precedes_pid_and_counter() {
        let dir = dir("discriminator");
        for (excluded, tag) in [
            ("output", 0),
            ("file0", 1),
            ("file1", 0),
            ("file9", 0),
            (".SNARKRS-STAGE-0-123-0", 1),
            (".snarKrs-stage-1-123-0", 0),
            (".ſnarkrs-stage-0-123-0. ", 1),
            ("non-ASCII-０", 0),
        ] {
            let stage =
                Stage::create(&dir, std::ffi::OsStr::new(excluded), &AtomicU64::new(0)).unwrap();
            let expected = format!(".snarkrs-stage-{tag}-{}-0", std::process::id());
            assert_eq!(
                stage.path.as_ref().unwrap().file_name().unwrap(),
                std::ffi::OsStr::new(&expected)
            );
            drop(stage);
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn stage_discriminator_accepts_non_utf8_exclusion() {
        use std::os::unix::ffi::OsStrExt;
        let dir = dir("discriminator-bytes");
        let excluded = std::ffi::OsStr::from_bytes(&[0xff, b'0']);
        let stage = Stage::create(&dir, excluded, &AtomicU64::new(0)).unwrap();
        let expected = format!(".snarkrs-stage-1-{}-0", std::process::id());
        assert_eq!(
            stage.path.as_ref().unwrap().file_name().unwrap(),
            std::ffi::OsStr::new(&expected)
        );
        drop(stage);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn collision_retry_never_removes_unowned_files_and_is_bounded() {
        let dir = dir("collision");
        let collision = |id| dir.join(format!(".snarkrs-stage-0-{}-{id}", std::process::id()));
        std::fs::write(collision(0), b"unowned").unwrap();
        let next = AtomicU64::new(0);
        let excluded = std::ffi::OsStr::new("output");
        let stage = Stage::create(&dir, excluded, &next).unwrap();
        assert_eq!(stage.path.as_ref().unwrap(), &collision(1));
        drop(stage);
        assert_eq!(std::fs::read(collision(0)).unwrap(), b"unowned");
        assert!(!collision(1).exists());
        for id in 1..128 {
            std::fs::write(collision(id), b"unowned").unwrap();
        }
        assert!(
            matches!(Stage::create(&dir, excluded, &AtomicU64::new(0)), Err(e) if e.kind() == ErrorKind::AlreadyExists)
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 128);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_set_publishes_all_or_nothing() {
        let dir = dir("set");
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::write(&a, b"old a").unwrap();
        let result: Result<(), std::io::Error> = staged_all(&[&a, &b], |stages| {
            std::fs::write(&stages[0], b"new a")?;
            std::fs::write(&stages[1], b"new b")?;
            Err(std::io::Error::other("late failure"))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&a).unwrap(), b"old a");
        assert!(!b.exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "orphan stage");
        let result: Result<(), std::io::Error> = staged_all(&[&a, &b], |stages| {
            std::fs::write(&stages[0], b"new a")?;
            std::fs::write(&stages[1], b"new b")
        });
        result.unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), b"new a");
        assert_eq!(std::fs::read(&b).unwrap(), b"new b");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_set_with_a_directory_member_stages_nothing() {
        let dir = dir("set-directory");
        let (a, b) = (dir.join("a"), dir.join("b"));
        std::fs::write(&a, b"old a").unwrap();
        std::fs::create_dir(&b).unwrap();
        let result: Result<(), std::io::Error> =
            staged_all(&[&a, &b], |_| panic!("set with a directory ran"));
        assert_eq!(result.unwrap_err().kind(), ErrorKind::IsADirectory);
        assert_eq!(std::fs::read(&a).unwrap(), b"old a");
        assert!(b.is_dir());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "orphan stage");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_set_member_named_like_a_stage_is_refused() {
        let dir = dir("set-clash");
        let a = dir.join("a");
        let clash = dir.join(format!(".snarkrs-stage-0-{}-0", std::process::id()));
        let result: Result<(), std::io::Error> =
            staged_all_with_counter(&[&a, &clash], &AtomicU64::new(0), true, |_| {
                panic!("clashing set ran")
            });
        assert_eq!(result.unwrap_err().kind(), ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "orphan stage");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_new_output_gets_the_umask_mode_or_0600_and_an_old_one_keeps_its_own() {
        use std::os::unix::fs::PermissionsExt;
        let dir = dir("mode");
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        // The umask is process wide, so this is the only test in the crate that reads it.
        // SAFETY: umask has no preconditions.
        let umask = unsafe { libc_umask(0o022) };
        let (shared, private) = (dir.join("shared"), dir.join("private"));
        let wrote: std::io::Result<()> = staged_all_default_mode(&[&shared], |s| {
            assert_eq!(mode(&s[0]), 0o644);
            std::fs::write(&s[0], b"new")
        });
        wrote.unwrap();
        let wrote: std::io::Result<()> = staged_all(&[&private], |s| std::fs::write(&s[0], b"new"));
        wrote.unwrap();
        unsafe { libc_umask(umask) };
        assert_eq!((mode(&shared), mode(&private)), (0o644, 0o600));
        for p in [&shared, &private] {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o640)).unwrap();
        }
        let wrote: std::io::Result<()> =
            staged_all_default_mode(&[&shared], |s| std::fs::write(&s[0], b"again"));
        wrote.unwrap();
        let wrote: std::io::Result<()> =
            staged_all(&[&private], |s| std::fs::write(&s[0], b"again"));
        wrote.unwrap();
        assert_eq!((mode(&shared), mode(&private)), (0o640, 0o640));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "orphan stage");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// `umask(2)`, declared here because this crate has no libc dependency. `mode_t` is
    /// 16 bits on Apple platforms and 32 elsewhere.
    #[cfg(unix)]
    unsafe fn libc_umask(mask: u32) -> u32 {
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
    fn symlinks_are_written_through_and_loops_refused() {
        use std::os::unix::fs::symlink;
        let dir = dir("symlinks");
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        // A chain of two links, the second dangling into another directory.
        symlink("sub/middle", dir.join("link")).unwrap();
        symlink("target", sub.join("middle")).unwrap();
        for content in [&b"first"[..], b"second"] {
            staged_one(&dir.join("link"), &NEXT_STAGE, |stage| {
                assert_eq!(stage.parent().unwrap(), sub);
                std::fs::write(stage, content)
            })
            .unwrap();
            assert_eq!(std::fs::read(sub.join("target")).unwrap(), content);
            assert!(std::fs::symlink_metadata(dir.join("link"))
                .unwrap()
                .file_type()
                .is_symlink());
            assert_eq!(std::fs::read_dir(&sub).unwrap().count(), 2);
        }
        symlink("loop-b", dir.join("loop-a")).unwrap();
        symlink("loop-a", dir.join("loop-b")).unwrap();
        let result: std::io::Result<()> = staged_one(&dir.join("loop-a"), &NEXT_STAGE, |_| {
            panic!("loop destination")
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 4);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
