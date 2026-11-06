//! A pure-Rust solver for gnark v0.14 BN254 R1CS files: given the public and secret inputs,
//! it computes what gnark's `r1cs.Solve` returns as `R1CSSolution{W, A, B}`, the full wire
//! vector and every constraint's `A·w` and `B·w`, bit for bit, with no Go anywhere.
//!
//! - [`GnarkR1cs::read`] parses a file `cs.WriteTo` wrote and checks it once, so that
//!   solving it is straight-line field arithmetic. See [`GnarkR1cs`] for what it proves.
//! - [`Solver`] holds the hints, by gnark's hint ID ([`hint_id`]), with gnark's standard
//!   ones built in, and [`Solver::solve`] runs the levels, each across the rayon pool.
//! - [`sp1`] has SP1's four KoalaBear hints and reads SP1's witness JSON, for SP1's
//!   Groth16 wrap circuit: [`Solver::sp1`] then [`sp1::Witness::assignment`].
//!
//! Only `BlueprintGenericR1C` and `BlueprintGenericHint` instructions are supported, which
//! is everything gnark's R1CS frontend emits; commitments and GKR are refused at read.

mod cbor;
pub mod hints;
mod intcomp;
mod r1cs;
mod solve;
pub mod sp1;

pub use hints::{hint_id, Hint};
pub use r1cs::GnarkR1cs;
pub use solve::Solver;

use snarkrs_field::Fr;

/// What gnark's `Solve` returns for an R1CS, less `C`, which the Groth16 prover never reads.
pub struct Solution {
    /// Every wire: the one wire, public, secret, internal.
    pub w: Vec<Fr>,
    /// Per constraint, `L·w`.
    pub a: Vec<Fr>,
    /// Per constraint, `R·w`.
    pub b: Vec<Fr>,
}

/// The marker gnark-crypto's `unsafe.WriteMarker` writes.
const DUMP_MARKER: u64 = 0xdeadbeef;

impl Solution {
    /// The solution as gnark-crypto's `unsafe.WriteMarker` then `unsafe.WriteSlice` of W,
    /// A and B write it: a u64 marker, then each vector as a u64 length and its elements'
    /// Montgomery limbs, little-endian.
    pub fn write_dump(&self, mut out: impl std::io::Write) -> std::io::Result<()> {
        out.write_all(&DUMP_MARKER.to_le_bytes())?;
        let mut buf = Vec::with_capacity(1 << 20);
        for v in [&self.w, &self.a, &self.b] {
            out.write_all(&(v.len() as u64).to_le_bytes())?;
            for chunk in v.chunks(1 << 15) {
                buf.clear();
                for x in chunk {
                    for limb in x.0 .0 {
                        buf.extend_from_slice(&limb.to_le_bytes());
                    }
                }
                out.write_all(&buf)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReadError {
    #[error("{context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },
    #[error("gnark {0} is not supported; this reader reads gnark 0.14 files")]
    UnsupportedVersion(String),
    #[error("malformed gnark R1CS {section}: {reason}")]
    Malformed {
        section: &'static str,
        reason: String,
    },
}

impl ReadError {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        ReadError::Io {
            context: format!("reading {}", path.display()),
            source,
        }
    }

    pub(crate) fn malformed(section: &'static str, reason: impl Into<String>) -> Self {
        ReadError::Malformed {
            section,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SolveError {
    #[error(
        "invalid witness size, got {public} public and {secret} secret, expected {expected_public} and {expected_secret}"
    )]
    WitnessSize {
        public: usize,
        secret: usize,
        expected_public: usize,
        expected_secret: usize,
    },
    #[error("{what} holds {len} elements, {want} expected")]
    BufferSize {
        what: &'static str,
        len: usize,
        want: usize,
    },
    /// gnark's `solver missing hint(s)`, with the names the circuit gives them.
    #[error("solver missing hint(s): {0:?}")]
    MissingHints(Vec<String>),
    /// gnark's `UnsatisfiedConstraintError`. `message` is gnark's `a ⋅ b != c`; gnark
    /// prints a constraint's debug information instead when the circuit carries some.
    #[error("constraint #{constraint} is not satisfied: {message}")]
    Unsatisfied { constraint: usize, message: String },
    #[error("hint {name} ({id}): {message}")]
    Hint {
        id: u32,
        name: String,
        message: String,
    },
}
