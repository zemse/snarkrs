//! snarkjs' `proof.json` and `public.json`: the encoding from [`snarkrs_groth16::json`], and
//! the path-taking readers and writers on top of it.
//!
//! Everything about `c0`/`c1` ordering, normal-form decimals and the projective third
//! component is documented in [`snarkrs_groth16::json`], in one copy, because the browser
//! prover has no filesystem and needs the encoding without these.

use snarkrs_field::Fr;
use snarkrs_groth16::Proof;
use std::path::{Path, PathBuf};

pub use snarkrs_groth16::json::{
    dec, parse_field, proof_from_str, proof_from_value, proof_to_string, public_from_str,
    public_from_value, public_to_string, read_g1, read_g2, JsonError,
};

/// A `proof.json` or `public.json` that could not be read or written. The message names the
/// file and carries the cause, so printing it is enough.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FileError {
    #[error("{what} {}: {error}", path.display())]
    Io {
        what: &'static str,
        path: PathBuf,
        error: std::io::Error,
    },
    #[error("in {}: {error}", path.display())]
    Parse { path: PathBuf, error: JsonError },
}

fn io(what: &'static str, path: &Path) -> impl FnOnce(std::io::Error) -> FileError {
    let path = path.to_path_buf();
    move |error| FileError::Io { what, path, error }
}

fn parse(path: &Path) -> impl FnOnce(JsonError) -> FileError {
    let path = path.to_path_buf();
    move |error| FileError::Parse { path, error }
}

/// `text` at `path`, through a stage beside it that is renamed over `path` once written, so
/// an error leaves `path` as it was. [`snarkrs_formats::stage`] states what that does and
/// does not promise. A new file gets the umask's mode and an existing one keeps its own.
fn publish(path: &Path, text: String) -> std::io::Result<()> {
    snarkrs_formats::stage::staged_all_default_mode(&[path], |stages| {
        std::fs::write(&stages[0], text)
    })
}

/// `proof.json` at `path`. A failed write leaves the old file as it was.
pub fn write_proof(path: &Path, p: &Proof) -> Result<(), FileError> {
    publish(path, proof_to_string(p)).map_err(io("writing proof to", path))
}

/// `public.json` at `path`. A failed write leaves the old file as it was.
pub fn write_public(path: &Path, public: &[Fr]) -> Result<(), FileError> {
    publish(path, public_to_string(public)).map_err(io("writing public signals to", path))
}

pub fn read_proof(path: &Path) -> Result<Proof, FileError> {
    let text = std::fs::read_to_string(path).map_err(io("reading proof from", path))?;
    proof_from_str(&text).map_err(parse(path))
}

pub fn read_public(path: &Path) -> Result<Vec<Fr>, FileError> {
    let text = std::fs::read_to_string(path).map_err(io("reading public signals from", path))?;
    public_from_str(&text).map_err(parse(path))
}
