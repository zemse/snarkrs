//! The CLI's own output files, published the way the ceremony's are: written to a synced
//! stage beside the destination and renamed over it, so a failed write leaves the old file
//! as it was and no stage behind. [`snarkrs_ceremony::write::staged_all`] states what that
//! does and does not promise.
//!
//! Every error names the destination, never the stage, so the messages are the ones a
//! direct write gave.

use std::path::{Path, PathBuf};

use anyhow::Context;
use snarkrs_field::Fr;
use snarkrs_groth16::Proof;
use snarkrs_witness::WitnessError;

use crate::json::{self, FileError};

/// Which side of [`publish_all`] failed: the caller's write, or the staging around it.
enum Failure<E> {
    Write(E),
    Stage(std::io::Error),
}

impl<E> From<std::io::Error> for Failure<E> {
    fn from(e: std::io::Error) -> Self {
        Failure::Stage(e)
    }
}

/// `write` given one stage per path, in order, then every stage renamed over its path.
/// `stage` turns a failure of the staging itself into the caller's error.
pub fn publish_all<E>(
    paths: &[&Path],
    stage: impl FnOnce(std::io::Error) -> E,
    write: impl FnOnce(&[PathBuf]) -> Result<(), E>,
) -> Result<(), E> {
    snarkrs_ceremony::write::staged_all(paths, |stages| write(stages).map_err(Failure::Write))
        .map_err(|f| match f {
            Failure::Write(e) => e,
            Failure::Stage(e) => stage(e),
        })
}

/// `bytes` at `path`, with `writing <path>` on the error as a direct write had it.
pub fn publish(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    publish_all(&[path], anyhow::Error::from, |stages| {
        std::fs::write(&stages[0], bytes).map_err(Into::into)
    })
    .with_context(|| format!("writing {}", path.display()))
}

/// `proof.json` and `public.json` as one set: neither is replaced unless both are written.
pub fn publish_proof(
    proof_out: &Path,
    proof: &Proof,
    public_out: &Path,
    public: &[Fr],
) -> anyhow::Result<()> {
    let (proof_text, public_text) = (json::proof_to_string(proof), json::public_to_string(public));
    let file = |what, path: &Path| {
        let path = path.to_path_buf();
        move |error| FileError::Io { what, path, error }
    };
    publish_all(
        &[proof_out, public_out],
        |e| {
            anyhow::Error::from(e).context(format!(
                "writing {} and {}",
                proof_out.display(),
                public_out.display()
            ))
        },
        |stages| {
            std::fs::write(&stages[0], proof_text).map_err(file("writing proof to", proof_out))?;
            std::fs::write(&stages[1], public_text)
                .map_err(file("writing public signals to", public_out))?;
            Ok(())
        },
    )
}

/// A `.wtns` at `path`, from `write` given the stage. An I/O error that names the stage is
/// reworded to name `path`.
pub fn publish_wtns(
    path: &Path,
    write: impl FnOnce(&Path) -> Result<(), WitnessError>,
) -> Result<(), WitnessError> {
    let io = |source| WitnessError::Io {
        context: format!("writing {}", path.display()),
        source,
    };
    publish_all(&[path], io, |stages| {
        write(&stages[0]).map_err(|e| match e {
            WitnessError::Io { context, source }
                if context == format!("writing {}", stages[0].display()) =>
            {
                io(source)
            }
            e => e,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("snarkrs-cli-publish-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn entries(dir: &Path) -> Vec<std::ffi::OsString> {
        let mut names: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn publish_replaces_and_a_directory_destination_is_left_alone() {
        let dir = dir("bytes");
        let out = dir.join("out.sol");
        std::fs::write(&out, b"old").unwrap();
        publish(&out, b"new").unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"new");
        assert_eq!(entries(&dir), ["out.sol"]);
        let taken = dir.join("taken");
        std::fs::create_dir(&taken).unwrap();
        let e = publish(&taken, b"new").unwrap_err();
        assert!(format!("{e:#}").starts_with(&format!("writing {}: ", taken.display())));
        assert!(taken.is_dir());
        assert_eq!(entries(&dir), ["out.sol", "taken"], "orphan stage");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_proof_and_its_public_signals_are_replaced_together_or_not_at_all() {
        let dir = dir("proof");
        let (proof_out, public_out) = (dir.join("proof.json"), dir.join("public.json"));
        use snarkrs_field::{AffineRepr, G1Affine, G2Affine};
        let proof = Proof {
            a: G1Affine::generator(),
            b: G2Affine::generator(),
            c: G1Affine::generator(),
        };
        let public = [Fr::from(7u64)];
        std::fs::write(&proof_out, b"old proof").unwrap();
        // Both stages written, then a failure before either is renamed.
        let e = publish_all(&[&proof_out, &public_out], anyhow::Error::from, |stages| {
            std::fs::write(&stages[0], json::proof_to_string(&proof))?;
            std::fs::write(&stages[1], json::public_to_string(&public))?;
            Err(std::io::Error::other("injected").into())
        })
        .unwrap_err();
        assert!(e.is::<std::io::Error>());
        assert_eq!(std::fs::read(&proof_out).unwrap(), b"old proof");
        assert!(!public_out.exists());
        assert_eq!(entries(&dir), ["proof.json"], "orphan stage");

        publish_proof(&proof_out, &proof, &public_out, &public).unwrap();
        assert_eq!(
            std::fs::read_to_string(&proof_out).unwrap(),
            json::proof_to_string(&proof)
        );
        assert_eq!(
            std::fs::read_to_string(&public_out).unwrap(),
            json::public_to_string(&public)
        );
        assert_eq!(entries(&dir), ["proof.json", "public.json"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_witness_error_names_the_destination_and_keeps_it() {
        let dir = dir("wtns");
        let out = dir.join("witness.wtns");
        std::fs::write(&out, b"old witness").unwrap();
        let e = publish_wtns(&out, |stage| {
            std::fs::write(stage, b"partial").unwrap();
            Err(WitnessError::Io {
                context: format!("writing {}", stage.display()),
                source: std::io::Error::other("injected"),
            })
        })
        .unwrap_err();
        assert_eq!(
            e.to_string(),
            format!("writing {}: injected", out.display())
        );
        assert_eq!(std::fs::read(&out).unwrap(), b"old witness");
        assert_eq!(entries(&dir), ["witness.wtns"], "orphan stage");
        let w = [Fr::from(1u64), Fr::from(5u64)];
        publish_wtns(&out, |stage| snarkrs_witness::write_wtns(stage, &w)).unwrap();
        assert_eq!(
            std::fs::read(&out).unwrap(),
            snarkrs_witness::wtns_bytes(&w)
        );
        assert_eq!(entries(&dir), ["witness.wtns"]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
