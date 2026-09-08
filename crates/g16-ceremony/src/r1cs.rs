//! `.r1cs`, the circuit as circom compiles it, and the only input to `groth16 setup` that
//! is not a ptau.
//!
//! Three sections, magic `r1cs`, version 1 (`r1csfile.js:359`). Setup reads section 1 and
//! the whole of section 2 (`zkey_new.js:47`, `:136`) and never opens section 3, the
//! wire-to-label map. Sections 4 and 5 carry circom's PLONK custom gates; `zkey_new.js`
//! does not look at `useCustomGates` at all, so `groth16 setup` silently ignores them.
//!
//! Two traps in the header. `nOutputs` comes **before** `nPubInputs`, and swapping them is
//! undetectable because both are u32 and only their sum reaches `nPublic`. And `prime` is
//! the **scalar** field `r`, not the base field `q` that the ptau header carries.
//!
//! The constraint coefficients are **plain little-endian**, not Montgomery
//! (`r1csfile.js:320` writes with `toRprLE`, which strips Montgomery on the way out).
//! That matters twice over: setup feeds these bytes to the MSM as scalars untouched
//! (`zkey_new.js:437-443`), and separately re-encodes the same numbers as *double*
//! Montgomery for zkey section 4 (`zkey_new.js:326-331`). One value, two encodings, one
//! command.

use std::path::Path;

use g16_field::Fr;
use g16_zkey::binfile::BinFile;

use crate::CeremonyError;

/// `.r1cs` magic (`r1csfile.js:359`).
pub const R1CS_MAGIC: &[u8; 4] = b"r1cs";
/// Highest container version this reader accepts. r1csfile writes 1.
pub const R1CS_MAX_VERSION: u32 = 1;

/// Section 1, in field order. `n8` and `prime` are checked against BN254 at open, so they
/// are not kept.
#[derive(Clone, Copy, Debug)]
pub struct R1csHeader {
    /// Includes signal 0, the constant ONE.
    pub n_vars: u32,
    pub n_outputs: u32,
    pub n_pub_inputs: u32,
    pub n_prv_inputs: u32,
    /// The only u64 in the header.
    pub n_labels: u64,
    pub n_constraints: u32,
}

impl R1csHeader {
    /// Bytes of section 1 on BN254.
    pub const BYTES: usize = 32 + crate::N8;

    /// `nOutputs + nPubInputs` (`zkey_new.js:71`), the count the zkey header stores and
    /// the number of IC points minus one. Signals `0 ..= n_public()` are exactly the IC
    /// signals (`zkey_new.js:234`).
    pub fn n_public(&self) -> usize {
        self.n_outputs as usize + self.n_pub_inputs as usize
    }
}

/// One term of a linear combination: which signal, and where its coefficient sits.
///
/// `coef_ptr` is a byte offset into [`R1cs::constraint_bytes`], not a decoded value. That
/// is snarkjs' own representation (`zkey_new.js` carries a `coefPtr` through every
/// accumulator list) and it is the reason setup can hand raw bytes to the MSM without a
/// conversion: the r1cs encoding is already the plain little-endian one a scalar wants.
#[derive(Clone, Copy, Debug)]
pub struct Term {
    pub signal: u32,
    pub coef_ptr: u32,
}

/// One constraint, as three linear combinations in file order.
#[derive(Clone, Debug, Default)]
pub struct Constraint {
    pub a: Vec<Term>,
    pub b: Vec<Term>,
    pub c: Vec<Term>,
}

/// A mapped `.r1cs`, header parsed, section 2 left as bytes.
///
/// Section 2 is deliberately not decoded into `Fr` up front. It is a bare concatenation of
/// `nConstraints` records with no count prefix and no index, so the parse is one forward
/// walk either way, and setup wants the raw coefficient bytes rather than field elements.
pub struct R1cs {
    file: BinFile,
    header: R1csHeader,
}

impl R1cs {
    /// Map the file, check the magic, the version and the prime, and parse section 1.
    pub fn open(path: &Path) -> Result<Self, CeremonyError> {
        let _ = path;
        todo!("R1cs::open")
    }

    /// [`R1cs::open`] for a file already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, CeremonyError> {
        let _ = bytes;
        todo!("R1cs::from_bytes")
    }

    pub fn header(&self) -> &R1csHeader {
        &self.header
    }

    /// The container, for a caller that needs a section this type does not expose.
    pub fn file(&self) -> &BinFile {
        &self.file
    }

    /// The whole of section 2, unparsed. Every [`Term::coef_ptr`] is an offset into this
    /// slice.
    pub fn constraint_bytes(&self) -> Result<&[u8], CeremonyError> {
        todo!("R1cs::constraint_bytes")
    }

    /// Walk section 2 once. The result is `n_constraints` entries in file order; setup
    /// depends on that order because zkey section 4 is written in it and must not be
    /// sorted (`zkey_new.js:241`, `:269`).
    pub fn constraints(&self) -> Result<Vec<Constraint>, CeremonyError> {
        todo!("R1cs::constraints")
    }

    /// The 32 raw plain-little-endian bytes at `coef_ptr`, for a caller feeding the MSM.
    pub fn coef_bytes(&self, coef_ptr: u32) -> Result<&[u8], CeremonyError> {
        let _ = coef_ptr;
        todo!("R1cs::coef_bytes")
    }

    /// The coefficient at `coef_ptr` as a field element. Plain little-endian, so this is
    /// [`g16_zkey::binfile::fr_normal`] and NOT one of the Montgomery decoders.
    pub fn coef(&self, coef_ptr: u32) -> Result<Fr, CeremonyError> {
        let _ = coef_ptr;
        todo!("R1cs::coef")
    }

    /// Section 3, the `nVars` u64 label ids. Setup never reads it; `g16 r1cs info` would.
    pub fn label_map(&self) -> Result<Vec<u64>, CeremonyError> {
        todo!("R1cs::label_map")
    }
}
