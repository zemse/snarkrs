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
//!
//! Section 2 is read twice by setup, once per direction. `zkey_new.js` walks it forward to
//! emit section 4 in file order, and it also needs, per signal, every constraint that
//! signal appears in, because `composeAndWritePoints` produces one output point per signal
//! out of that signal's terms (`zkey_new.js:222-300`). [`R1cs::constraints`] is the first
//! view and [`SignalIndex`] is the second, built by the same counting sort that
//! [`g16_zkey::Coefficients`] uses on zkey section 4.

use std::path::Path;

use g16_field::Fr;
use g16_zkey::binfile::{expect_records, fr_normal, BinFile, Cursor};

use crate::CeremonyError;

/// `.r1cs` magic (`r1csfile.js:359`).
pub const R1CS_MAGIC: &[u8; 4] = b"r1cs";
/// Highest container version this reader accepts. r1csfile writes 1.
pub const R1CS_MAX_VERSION: u32 = 1;

/// Section ids, `r1csfile.js:5-9`.
pub const S_HEADER: u32 = 1;
pub const S_CONSTRAINTS: u32 = 2;
pub const S_WIRE_TO_LABEL: u32 = 3;
pub const S_CUSTOM_GATES_LIST: u32 = 4;
pub const S_CUSTOM_GATES_USES: u32 = 5;

/// One term of a linear combination on disk: `u32` signal index then one `n8` coefficient.
pub const TERM_BYTES: usize = 4 + crate::N8;

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

    /// `floor(log2(nConstraints + nPublic)) + 1`, the domain exponent setup will use
    /// (`zkey_new.js:59`). One copy of the formula lives in [`crate::setup`]; this is the
    /// spelling every other command reaches for, since it is a property of the circuit.
    pub fn cir_power(&self) -> u32 {
        crate::setup::circuit_power(self.n_constraints as usize, self.n_public())
    }

    /// `2^cir_power`, and always strictly greater than `nConstraints + nPublic`, which is
    /// what makes the synthetic public-signal rows at index `nConstraints + s` land inside
    /// the Lagrange window.
    pub fn domain_size(&self) -> usize {
        1usize << self.cir_power()
    }
}

/// Which linear combination a term came from. `A` and `B` are the only two that reach
/// zkey section 4, and the numbering here is that field's (`zkey_new.js:242`, `:270`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Matrix {
    A,
    B,
    C,
}

impl Matrix {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
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

/// Terms per matrix. Section 4's record count is `a + b + n_public + 1`, since only the A
/// and B passes push coefficients and the public-signal tail adds one row each
/// (`zkey_new.js:242`, `:270`, `:298`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatrixNnz {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

impl MatrixNnz {
    pub fn total(&self) -> usize {
        self.a + self.b + self.c
    }
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
        Self::parse(BinFile::open(path, R1CS_MAGIC, R1CS_MAX_VERSION)?)
    }

    /// [`R1cs::open`] for a file already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, CeremonyError> {
        Self::parse(BinFile::from_bytes(bytes, R1CS_MAGIC, R1CS_MAX_VERSION)?)
    }

    /// Everything after the container is open, shared by both constructors.
    fn parse(file: BinFile) -> Result<Self, CeremonyError> {
        let mut cur = Cursor::new(file.unique_section(S_HEADER)?, S_HEADER);
        // `u32 n8` then the prime, the same shape both other headers use, except that here
        // the prime is `r` rather than `q`.
        crate::check_modulus(&mut cur, &crate::r_le())?;
        let header = R1csHeader {
            n_vars: cur.u32()?,
            n_outputs: cur.u32()?,
            n_pub_inputs: cur.u32()?,
            n_prv_inputs: cur.u32()?,
            n_labels: cur.u64()?,
            n_constraints: cur.u32()?,
        };
        if cur.remaining() != 0 {
            return Err(CeremonyError::malformed(
                S_HEADER,
                format!("{} bytes left after the header", cur.remaining()),
            ));
        }
        // Every coefficient is addressed by a u32 offset into section 2, both here and in
        // snarkjs' accumulator lists, so a section past 4 GiB would silently alias rather
        // than fail. JS numbers do not have that ceiling, which is why the check is ours.
        let body = file.unique_section(S_CONSTRAINTS)?;
        if body.len() > u32::MAX as usize {
            return Err(CeremonyError::malformed(
                S_CONSTRAINTS,
                format!(
                    "{} bytes does not fit a u32 coefficient pointer",
                    body.len()
                ),
            ));
        }
        Ok(Self { file, header })
    }

    pub fn header(&self) -> &R1csHeader {
        &self.header
    }

    /// The container, for a caller that needs a section this type does not expose.
    pub fn file(&self) -> &BinFile {
        &self.file
    }

    /// Whether sections 4 and 5 are both present, which is r1csfile's whole definition of
    /// `useCustomGates` (`r1csfile.js:54-55`). Groth16 setup ignores it.
    pub fn has_custom_gates(&self) -> bool {
        let has = |id: u32| self.file.sections().iter().any(|s| s.id == id);
        has(S_CUSTOM_GATES_LIST) && has(S_CUSTOM_GATES_USES)
    }

    /// The whole of section 2, unparsed. Every [`Term::coef_ptr`] is an offset into this
    /// slice.
    pub fn constraint_bytes(&self) -> Result<&[u8], CeremonyError> {
        Ok(self.file.unique_section(S_CONSTRAINTS)?)
    }

    /// Walk section 2 once, calling `f` with `(matrix, constraint index, term)` in exactly
    /// the order the bytes are laid out: A, B then C within each constraint, constraints
    /// ascending. Every view below is built on this, so there is one bounds check to get
    /// right rather than four.
    fn walk<F>(&self, mut f: F) -> Result<(), CeremonyError>
    where
        F: FnMut(Matrix, u32, Term),
    {
        let body = self.constraint_bytes()?;
        let n_vars = self.header.n_vars;
        let mut pos = 0usize;
        for c in 0..self.header.n_constraints {
            for matrix in [Matrix::A, Matrix::B, Matrix::C] {
                let n_terms = read_u32(body, pos, c)? as usize;
                pos += 4;
                // One check for the whole linear combination: `n_terms` is attacker
                // controlled and `n_terms * TERM_BYTES` is what would wrap.
                let end = n_terms
                    .checked_mul(TERM_BYTES)
                    .and_then(|span| pos.checked_add(span))
                    .filter(|end| *end <= body.len())
                    .ok_or_else(|| {
                        CeremonyError::malformed(
                            S_CONSTRAINTS,
                            format!(
                                "constraint {c} {matrix:?} declares {n_terms} terms, past the end \
                                 of the section"
                            ),
                        )
                    })?;
                while pos < end {
                    let signal = read_u32(body, pos, c)?;
                    if signal >= n_vars {
                        return Err(CeremonyError::malformed(
                            S_CONSTRAINTS,
                            format!("constraint {c} references signal {signal} of {n_vars}"),
                        ));
                    }
                    // The pointer is to the value, not to the term: snarkjs takes it after
                    // reading the signal index (`zkey_new.js:222-225`).
                    let coef_ptr = (pos + 4) as u32;
                    pos += TERM_BYTES;
                    f(matrix, c, Term { signal, coef_ptr });
                }
            }
        }
        if pos != body.len() {
            return Err(CeremonyError::malformed(
                S_CONSTRAINTS,
                format!(
                    "{} constraints end at byte {pos} of {}",
                    self.header.n_constraints,
                    body.len()
                ),
            ));
        }
        Ok(())
    }

    /// Walk section 2 once. The result is `n_constraints` entries in file order; setup
    /// depends on that order because zkey section 4 is written in it and must not be
    /// sorted (`zkey_new.js:241`, `:269`).
    pub fn constraints(&self) -> Result<Vec<Constraint>, CeremonyError> {
        let mut out = vec![Constraint::default(); self.header.n_constraints as usize];
        self.walk(|matrix, c, term| {
            let slot = &mut out[c as usize];
            match matrix {
                Matrix::A => slot.a.push(term),
                Matrix::B => slot.b.push(term),
                Matrix::C => slot.c.push(term),
            }
        })?;
        Ok(out)
    }

    /// Terms per matrix, without materialising them. Cheap enough to call for a bench
    /// column, and it is the check that predicts zkey section 4's record count.
    pub fn nonzeros(&self) -> Result<MatrixNnz, CeremonyError> {
        let mut nnz = MatrixNnz::default();
        self.walk(|matrix, _, _| match matrix {
            Matrix::A => nnz.a += 1,
            Matrix::B => nnz.b += 1,
            Matrix::C => nnz.c += 1,
        })?;
        Ok(nnz)
    }

    /// The 32 raw plain-little-endian bytes at `coef_ptr`, for a caller feeding the MSM.
    pub fn coef_bytes(&self, coef_ptr: u32) -> Result<&[u8], CeremonyError> {
        let body = self.constraint_bytes()?;
        let start = coef_ptr as usize;
        body.get(start..start + crate::N8).ok_or_else(|| {
            CeremonyError::malformed(
                S_CONSTRAINTS,
                format!("coefficient pointer {coef_ptr} is past the end of the section"),
            )
        })
    }

    /// The coefficient at `coef_ptr` as a field element.
    ///
    /// Plain little-endian, so this is [`g16_zkey::binfile::fr_normal`] and NOT one of the
    /// Montgomery decoders. Settled from both sides: the writer is `F.toRprLE`
    /// (`r1csfile.js:320`), which is `buff.set(this.fromMontgomery(a))`
    /// (`ffjavascript/src/wasm_field1.js:224-226`) and therefore strips Montgomery, and
    /// the reader is `r1cs.F.fromRprLE` (`r1csfile.js:112`), which is `toMontgomery`
    /// (`wasm_field1.js:238-241`) and puts it back. Decoding these bytes as single
    /// Montgomery parses cleanly and produces a zkey that fails verification with no other
    /// symptom.
    pub fn coef(&self, coef_ptr: u32) -> Result<Fr, CeremonyError> {
        Ok(fr_normal(self.coef_bytes(coef_ptr)?, S_CONSTRAINTS)?)
    }

    /// Section 3, the `nVars` u64 label ids. Setup never reads it; `g16 r1cs info` would.
    pub fn label_map(&self) -> Result<Vec<u64>, CeremonyError> {
        let data = self.file.unique_section(S_WIRE_TO_LABEL)?;
        let n_vars = self.header.n_vars as usize;
        expect_records(data, n_vars, 8, S_WIRE_TO_LABEL)?;
        Ok((0..n_vars)
            .map(|i| {
                let b = &data[i * 8..i * 8 + 8];
                u64::from_le_bytes(b.try_into().expect("slice is 8 bytes"))
            })
            .collect())
    }
}

/// One term of section 2 seen from the signal's side: the matrix it sits in, the
/// constraint index that becomes a Lagrange element index, and the coefficient pointer.
#[derive(Clone, Copy, Debug)]
pub struct SignalTerm {
    pub matrix: Matrix,
    pub constraint: u32,
    pub coef_ptr: u32,
}

/// Section 2 inverted: for each signal, every term that mentions it, in file order.
///
/// This is the shape `composeAndWritePoints` consumes. Setup emits one point per signal
/// out of that signal's terms, so the forward walk in [`R1cs::constraints`] would have it
/// scattering into `nVars` growing vectors; a counting sort builds the same thing in two
/// passes with one allocation, which is exactly why [`g16_zkey::Coefficients`] sorts zkey
/// section 4 at load rather than scattering per proof.
///
/// Offsets are `u32` for the same reason the coefficient pointers are: one term occupies
/// [`TERM_BYTES`] of a section that [`R1cs::parse`] has already capped at 4 GiB, so there
/// can be no more than about 119 million of them.
pub struct SignalIndex {
    /// `row_ptr[s] .. row_ptr[s + 1]` indexes `terms`. Length `n_vars + 1`.
    row_ptr: Vec<u32>,
    terms: Vec<SignalTerm>,
}

impl SignalIndex {
    pub fn build(r1cs: &R1cs) -> Result<Self, CeremonyError> {
        let n_vars = r1cs.header.n_vars as usize;
        let mut row_ptr = vec![0u32; n_vars + 1];
        r1cs.walk(|_, _, term| row_ptr[term.signal as usize + 1] += 1)?;
        // Prefix sum in place turns the histogram into the row offsets.
        for i in 0..n_vars {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Every slot is overwritten below, because the counts came from the same walk.
        let mut terms = vec![
            SignalTerm {
                matrix: Matrix::A,
                constraint: 0,
                coef_ptr: 0,
            };
            row_ptr[n_vars] as usize
        ];
        let mut cursor = row_ptr.clone();
        r1cs.walk(|matrix, constraint, term| {
            let slot = &mut cursor[term.signal as usize];
            terms[*slot as usize] = SignalTerm {
                matrix,
                constraint,
                coef_ptr: term.coef_ptr,
            };
            *slot += 1;
        })?;

        Ok(Self { row_ptr, terms })
    }

    /// `n_vars`, so signal indices `0 .. n_signals()` are all addressable.
    pub fn n_signals(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// Every term mentioning `signal`, in file order. Empty for a signal that appears in
    /// no constraint, which is not an error: snarkjs leaves that slot `undefined` and
    /// writes the group identity for it (`zkey_new.js:461-466`).
    pub fn terms(&self, signal: u32) -> &[SignalTerm] {
        let s = signal as usize;
        match (self.row_ptr.get(s), self.row_ptr.get(s + 1)) {
            (Some(&start), Some(&end)) => &self.terms[start as usize..end as usize],
            _ => &[],
        }
    }

    /// Total terms across every signal, which equals [`MatrixNnz::total`].
    pub fn total_terms(&self) -> usize {
        self.terms.len()
    }
}

fn read_u32(body: &[u8], pos: usize, constraint: u32) -> Result<u32, CeremonyError> {
    let b = body.get(pos..pos + 4).ok_or_else(|| {
        CeremonyError::malformed(
            S_CONSTRAINTS,
            format!("constraint {constraint} runs past the end of the section"),
        )
    })?;
    Ok(u32::from_le_bytes(b.try_into().expect("slice is 4 bytes")))
}
