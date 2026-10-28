//! snarkjs `.zkey` (Groth16, BN254) and `.wtns` parsing.
//!
//! Section layout, from the snarkjs binfile format:
//!   1 header (protocol id)          2 groth16 header (curve, nVars, nPublic, domainSize, alpha/beta/delta/...)
//!   3 IC (verifier)                 4 coefficients  (m, c, s, coef) records
//!   5 A  bases G1  [u_j(tau)]_1     6 B1 bases G1   [v_j(tau)]_1
//!   7 B2 bases G2  [v_j(tau)]_2     8 C  bases G1   (private wires only)
//!   9 H  bases G1                   10 contributions
//!
//! Points are stored in **Montgomery form, little-endian**, which is NOT what
//! `ark-serialize` expects; converting out of Montgomery is required and is the single
//! most common source of "my prover produces garbage" bugs. Section 4 field values are
//! worse: they are in *double* Montgomery form (`v * R^2`), while `.wtns` values are in
//! plain form. All three conventions are decoded in [`binfile`], which documents where
//! each one was read out of the snarkjs source.

pub mod binfile;
pub mod calldata;
#[cfg(not(target_family = "wasm"))]
pub mod export_json;
#[cfg(not(target_family = "wasm"))]
pub mod file_info;
pub mod json_out;
#[cfg(not(target_family = "wasm"))]
pub mod stage;
pub mod wtns;

use binfile::FirstError;
use binfile::*;
use rayon::prelude::*;
use snarkrs_field::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Everything needed to prove, as parsed from a `.zkey`.
pub struct ProvingKey {
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,
    /// Section 5: bases for the A MSM. Length `n_vars`.
    pub a_query: Vec<G1Affine>,
    /// Section 6: bases for the B-in-G1 MSM. Length `n_vars`.
    pub b_g1_query: Vec<G1Affine>,
    /// Section 7: bases for the B-in-G2 MSM. Length `n_vars`.
    pub b_g2_query: Vec<G2Affine>,
    /// Section 8: bases for the L MSM. Length `n_vars - n_public - 1`.
    pub l_query: Vec<G1Affine>,
    /// Section 9: bases for the H MSM. Length `domain_size`, and all of them are used.
    /// ffjavascript's `multiExp` rejects a scalar vector shorter than the base vector, so
    /// the H MSM runs over the full section rather than `domain_size - 1` of it.
    pub h_query: Vec<G1Affine>,
    /// Section 4, already sorted into CSR by constraint index. See [`Coefficients`].
    pub coeffs: Coefficients,
    pub vk: VerifyingKey,
}

/// One of the two constraint matrices a zkey holds. `index` is the position in
/// [`Coefficients`]' per-matrix arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    A,
    B,
}

impl Matrix {
    pub const BOTH: [Matrix; 2] = [Matrix::A, Matrix::B];

    pub const fn index(self) -> usize {
        match self {
            Matrix::A => 0,
            Matrix::B => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Matrix::A => "A",
            Matrix::B => "B",
        }
    }
}

/// Section 4 in CSR form: `A[c] = sum over row c of coef * w[s]`.
///
/// snarkjs stores an unordered `(matrix, constraint, signal, coef)` list, which on CPU
/// is consumed as a *scatter* under striped mutexes and on GPU cannot be consumed that
/// way at all (there is no 32-byte atomic). Sorting once at key load turns it into a
/// race-free gather that both backends share. This is a one-time cost paid in `prepare`,
/// never per proof.
///
/// The values are interned. A key holds few distinct coefficients (31 in keccak_2048's
/// 11.9 million records, 878 in anon-aadhaar's 3.3 million), so each nonzero stores a
/// `u32` index into `values` rather than a 32-byte `Fr` of its own: on keccak_2048, 48 MB
/// of indices in place of 381 MB of values, which the CPU backend holds for the life of
/// the circuit.
///
/// The fields are `pub`, so a hand-built `Coefficients` can hold anything. What
/// `read_coefficients` upholds, and what a consumer of a parsed key may assume, is:
/// `row_ptr[m].len() == domain_size + 1`; `row_ptr[m][0] == 0`;
/// `row_ptr[m]` is non-decreasing;
/// `row_ptr[m][domain_size] == signal[m].len() == value_id[m].len()`; every
/// `signal[m][k] < n_vars`; and every `value_id[m][k] < values.len()`. The backends
/// re-check these at prepare time with [`Coefficients::check_structure`] because a key
/// built by hand is still reachable.
pub struct Coefficients {
    /// `row_ptr[m][c]..row_ptr[m][c+1]` indexes into `signal`/`value_id`, for matrix `m`,
    /// indexed by [`Matrix::index`]. Transposing them is silent and produces a proof
    /// that does not verify.
    pub row_ptr: [Vec<u32>; 2],
    pub signal: [Vec<u32>; 2],
    /// Nonzero `k` of matrix `m` is `values[value_id[m][k]]`.
    pub value_id: [Vec<u32>; 2],
    /// The distinct values of both matrices.
    pub values: Vec<Fr>,
}

impl Coefficients {
    /// A key built by hand, one value per nonzero, interned.
    pub fn from_values(row_ptr: [Vec<u32>; 2], signal: [Vec<u32>; 2], value: [Vec<Fr>; 2]) -> Self {
        let mut ids = std::collections::HashMap::new();
        let mut values = Vec::new();
        let value_id = value.map(|v| {
            v.iter()
                .map(|x| {
                    *ids.entry(*x).or_insert_with(|| {
                        values.push(*x);
                        (values.len() - 1) as u32
                    })
                })
                .collect()
        });
        Self {
            row_ptr,
            signal,
            value_id,
            values,
        }
    }

    /// Nonzero `k` of matrix `m`.
    pub fn value(&self, m: usize, k: usize) -> Fr {
        self.values[self.value_id[m][k] as usize]
    }

    /// Matrix `m`'s values in nonzero order, for a backend that uploads them one per
    /// nonzero.
    pub fn dense(&self, m: usize) -> Vec<Fr> {
        self.value_id[m]
            .par_iter()
            .map(|&id| self.values[id as usize])
            .collect()
    }

    /// Check both matrices' CSR bounds against the domain and witness length. Paid once
    /// per key, before a backend uploads or gathers a hand-built key's coefficients.
    pub fn check_structure(&self, domain_size: usize, n_vars: usize) -> Result<(), ZkeyError> {
        for matrix in Matrix::BOTH {
            let m = matrix.index();
            Self::check_matrix(
                matrix,
                &self.row_ptr[m],
                &self.signal[m],
                self.value_id[m].len(),
                domain_size,
                n_vars,
            )?;
            let n = self.values.len();
            let past = self.value_id[m]
                .par_iter()
                .position_first(|&id| id as usize >= n);
            if let Some(k) = past {
                return Err(ZkeyError::Malformed {
                    section: 4,
                    reason: format!(
                        "matrix {} nonzero {k} references value {} beyond the {n} values",
                        matrix.name(),
                        self.value_id[m][k]
                    ),
                });
            }
        }
        Ok(())
    }

    /// [`Coefficients::check_structure`] for one matrix held somewhere other than a
    /// `Coefficients`, such as the device memory a [`KeySink`] handed out: its row
    /// pointers, its signals and how many values it has.
    pub fn check_matrix(
        matrix: Matrix,
        rp: &[u32],
        signal: &[u32],
        values: usize,
        domain_size: usize,
        n_vars: usize,
    ) -> Result<(), ZkeyError> {
        let bad = |reason| ZkeyError::Malformed { section: 4, reason };
        let rows_plus_one = domain_size
            .checked_add(1)
            .ok_or_else(|| bad(format!("domain size {domain_size} overflows CSR row count")))?;
        let name = matrix.name();
        if rp.len() != rows_plus_one {
            return Err(bad(format!(
                "matrix {name} has {} row_ptr entries, domain size {domain_size} needs {rows_plus_one}",
                rp.len()
            )));
        }
        for c in 1..rp.len() {
            if rp[c] < rp[c - 1] {
                return Err(bad(format!(
                    "matrix {name} row_ptr is not monotone at row {}: {} then {}",
                    c - 1,
                    rp[c - 1],
                    rp[c]
                )));
            }
        }
        if rp[0] != 0 {
            return Err(bad(format!(
                "matrix {name} row_ptr starts at {}, expected 0",
                rp[0]
            )));
        }
        let total = rp[domain_size] as usize;
        if total != signal.len() || total != values {
            return Err(bad(format!(
                "matrix {name} row_ptr ends at {total} but has {} signals and {values} values",
                signal.len(),
            )));
        }
        for (k, &s) in signal.iter().enumerate() {
            if s as usize >= n_vars {
                return Err(bad(format!(
                    "matrix {name} nonzero {k} references signal {s} beyond n_vars {n_vars}"
                )));
            }
        }
        Ok(())
    }
}

/// Memory of a backend's own that [`ProvingKey::load_into`] decodes the bulk of a key
/// straight into, in place of the key's vectors: the four G1 query sections 5, 6, 8 and
/// 9, and both CSR matrices of section 4. Every check the loader makes runs exactly as it
/// does for [`ProvingKey::load`]; only where a validated value is written changes. It
/// exists for unified memory, where the device reads host memory directly and a parsed
/// vector would only be copied into a device buffer and dropped.
///
/// Each method is asked once, after the section's length has been checked against the
/// header and before its first value is decoded, and may answer `None` to keep that
/// section in the key as usual. What it hands out must be exactly as long as asked for.
pub trait KeySink {
    /// A G1 point as the sink stores it.
    type G1: Send;
    /// A section 4 value as the sink stores it.
    type Fr: Send;

    fn g1(p: &G1Affine) -> Self::G1;
    fn fr(v: &Fr) -> Self::Fr;

    /// Room for the `n` points of query section `section`, and one infinity flag each.
    fn g1_section(
        &mut self,
        section: u32,
        n: usize,
    ) -> Result<Option<PointsInto<'_, Self::G1>>, ZkeyError>;

    /// Room for both matrices, indexed by [`Matrix::index`]: `rows + 1` row pointers each,
    /// and `nnz[m]` signals and values for matrix `m`.
    fn coefficients(
        &mut self,
        rows: usize,
        nnz: [usize; 2],
    ) -> Result<Option<[CsrInto<'_, Self::Fr>; 2]>, ZkeyError>;
}

/// Where one query section goes. `infinity[i]` is whether point `i` is the identity.
pub struct PointsInto<'a, T> {
    pub points: &'a mut [T],
    pub infinity: &'a mut [bool],
}

/// Where one CSR matrix goes, with the invariants [`Coefficients`] documents.
pub struct CsrInto<'a, T> {
    pub row_ptr: &'a mut [u32],
    pub signal: &'a mut [u32],
    pub value: &'a mut [T],
}

/// The sink every plain load uses: it takes nothing, so every section stays in the key.
struct Keep;

impl KeySink for Keep {
    type G1 = G1Affine;
    type Fr = Fr;

    fn g1(p: &G1Affine) -> G1Affine {
        *p
    }
    fn fr(v: &Fr) -> Fr {
        *v
    }
    fn g1_section(
        &mut self,
        _: u32,
        _: usize,
    ) -> Result<Option<PointsInto<'_, G1Affine>>, ZkeyError> {
        Ok(None)
    }
    fn coefficients(
        &mut self,
        _: usize,
        _: [usize; 2],
    ) -> Result<Option<[CsrInto<'_, Fr>; 2]>, ZkeyError> {
        Ok(None)
    }
}

pub struct VerifyingKey {
    pub alpha_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    pub delta_g2: G2Affine,
    /// `gamma^-1 * L_i(tau) * g1` for the public inputs, length `n_public + 1`.
    pub ic: Vec<G1Affine>,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ZkeyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a zkey file (bad magic {0:?})")]
    BadMagic([u8; 4]),
    #[error("file is {0} bytes, too short for a binfile header")]
    TooShort(usize),
    #[error("unsupported protocol id {0} (only groth16 = 1)")]
    UnsupportedProtocol(u32),
    #[error("unsupported curve: expected BN254")]
    UnsupportedCurve,
    #[error("missing section {0}")]
    MissingSection(u32),
    #[error("malformed section {section}: {reason}")]
    Malformed { section: u32, reason: String },
    /// A stored field element whose limbs are at or above the modulus. Separate from
    /// [`ZkeyError::Malformed`] because the decoders that raise it are handed a bare slice
    /// and do not know which section it came from.
    #[error("{0} element is not below the modulus")]
    NonCanonical(&'static str),
    #[error("verification key json: {0}")]
    BadJson(String),
    /// A verifying key that is well formed but lets anyone forge, or accepts everything.
    #[error("unsafe verifying key: {0}")]
    UnsafeVerifyingKey(&'static str),
}

/// snarkjs' groth16 protocol id in section 1.
const PROTOCOL_GROTH16: u32 = 1;

/// Section 4 record: three u32 indices then one 32-byte scalar.
const COEF_RECORD: usize = 12 + FR_BYTES;

/// How much of a key the parser validates. See [`ProvingKey::load`] for what the checked
/// mode adds, [`ProvingKey::load_unchecked`] for what skipping it gives up and
/// [`ProvingKey::load_allowing_unblinded`] for the one refusal the third mode drops.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    Full,
    /// `Full` without the `gamma_g2 != delta_g2` refusal. Only the native loader takes it.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    AllowUnblinded,
    Unchecked,
}

impl ProvingKey {
    /// Parse a `.zkey` by mmap and validate it. Must not copy the point sections more than
    /// once.
    ///
    /// On top of the container, header and point-at-infinity gates that every load runs,
    /// this checks that every point in sections 5 to 9 is on the curve, and refuses a key
    /// whose `gamma_g2` equals its `delta_g2`. The curve check costs 7 ms on js_16x16_d32
    /// and 35 ms on anon-aadhaar, against loads of 80 and 440 ms. The G2 query is not
    /// subgroup checked here: that is 70 to 90 us a point, 1.2 s and 6.6 s on the same
    /// two keys, and [`snarkrs_groth16`'s `prove`] checks the one G2 point that leaves the
    /// prover instead.
    ///
    /// `gamma_g2 == delta_g2` is what a key with no phase-2 contribution looks like: both
    /// are left at the G2 generator, and anyone can then forge a proof of any statement
    /// with `A = alpha`, `B = beta`, `C = -L`. It is the key the Foom and Veil verifiers
    /// were drained through. Development keys look like this, so
    /// [`ProvingKey::load_unchecked`] still accepts it.
    ///
    /// Native only. There is no filesystem on `wasm32-unknown-unknown`, so the browser
    /// uses [`ProvingKey::from_bytes`] instead.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::parse(BinFile::open(path, b"zkey", 2)?, Check::Full, &mut Keep)
    }

    /// [`ProvingKey::load`] without the per-point curve checks on sections 5 to 9 and
    /// without the `gamma_g2 != delta_g2` refusal. For a key this process generated or
    /// otherwise already trusts. The container, header and infinity gates still run: they
    /// are O(1) and they are what keeps a bad file from allocating or panicking.
    ///
    /// An off-curve query point is not caught at load this way. It yields an off-curve
    /// proof element, which the checked `prove` in `snarkrs-groth16` still refuses to return.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_unchecked(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::parse(
            BinFile::open(path, b"zkey", 2)?,
            Check::Unchecked,
            &mut Keep,
        )
    }

    /// [`ProvingKey::load`] that accepts a key whose `gamma_g2` equals its `delta_g2`, and
    /// runs every other check `load` does, the per-point curve checks included. Benchmark
    /// only: it exists so published benchmark keys with no phase-2 contribution can be
    /// timed. Anyone can forge a proof against such a key, so a key loaded this way must
    /// never back a proof anyone relies on. Pair it with `snarkrs-groth16`'s
    /// `prove_allowing_unblinded`, since the checked `prove` refuses the key too.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_allowing_unblinded(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::parse(
            BinFile::open(path, b"zkey", 2)?,
            Check::AllowUnblinded,
            &mut Keep,
        )
    }

    /// [`ProvingKey::load`] that decodes the G1 query sections 5, 6, 8 and 9 and both
    /// section 4 matrices into `sink` instead of the key, with every check `load` runs.
    /// Each section the sink takes is left empty in the returned key, so that key is the
    /// header, the verifying key, the G2 query and whatever the sink declined.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_into<S: KeySink>(path: &std::path::Path, sink: &mut S) -> Result<Self, ZkeyError> {
        Self::parse(BinFile::open(path, b"zkey", 2)?, Check::Full, sink)
    }

    /// [`ProvingKey::load_allowing_unblinded`] into a [`KeySink`], as
    /// [`ProvingKey::load_into`] is to `load`. Benchmark only, for the same reason.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_allowing_unblinded_into<S: KeySink>(
        path: &std::path::Path,
        sink: &mut S,
    ) -> Result<Self, ZkeyError> {
        Self::parse(
            BinFile::open(path, b"zkey", 2)?,
            Check::AllowUnblinded,
            sink,
        )
    }

    /// Parse and validate a `.zkey` that is already in memory, for the browser, where the
    /// key arrives over `fetch` rather than from a path. Validates exactly what
    /// [`ProvingKey::load`] does.
    ///
    /// By value, not `&[u8]`: at `js_16x16_d32` the key is 94.4 MB, and a borrow would
    /// force the caller to hold a second live owner for as long as the `ProvingKey`
    /// exists. In a 4 GiB wasm32 address space that doubling is worth avoiding on its own,
    /// and the caller has no use for the bytes afterwards anyway.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ZkeyError> {
        Self::parse(
            BinFile::from_bytes(bytes, b"zkey", 2)?,
            Check::Full,
            &mut Keep,
        )
    }

    /// [`ProvingKey::from_bytes`] with the checks [`ProvingKey::load_unchecked`] skips.
    pub fn from_bytes_unchecked(bytes: Vec<u8>) -> Result<Self, ZkeyError> {
        Self::parse(
            BinFile::from_bytes(bytes, b"zkey", 2)?,
            Check::Unchecked,
            &mut Keep,
        )
    }

    /// Everything after the container is opened, shared by all seven constructors so the
    /// paths cannot diverge on a single validation.
    fn parse<S: KeySink>(file: BinFile, check: Check, sink: &mut S) -> Result<Self, ZkeyError> {
        let mut s1 = Cursor::new(file.unique_section(1)?, 1);
        let protocol = s1.u32()?;
        if protocol != PROTOCOL_GROTH16 {
            return Err(ZkeyError::UnsupportedProtocol(protocol));
        }

        let mut s2 = Cursor::new(file.unique_section(2)?, 2);
        check_modulus(&mut s2, &Fq::MODULUS.to_bytes_le())?;
        check_modulus(&mut s2, &Fr::MODULUS.to_bytes_le())?;
        let n_vars = s2.u32()? as usize;
        let n_public = s2.u32()? as usize;
        let domain_size = s2.u32()? as usize;
        // Order here is alpha1, beta1, beta2, gamma2, delta1, delta2. Section 2 is one
        // packed blob, so a single swapped pair shifts every later point.
        let alpha_g1 = g1(s2.take(G1_BYTES)?)?;
        let beta_g1 = g1(s2.take(G1_BYTES)?)?;
        let beta_g2 = g2(s2.take(G2_BYTES)?)?;
        let gamma_g2 = g2(s2.take(G2_BYTES)?)?;
        let delta_g1 = g1(s2.take(G1_BYTES)?)?;
        let delta_g2 = g2(s2.take(G2_BYTES)?)?;
        if s2.remaining() != 0 {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!("{} trailing bytes after the groth16 header", s2.remaining()),
            });
        }

        // Bound `domain_size` here, before a single byte is allocated from it.
        //
        // Two independent gates, because the header is attacker-controlled and everything
        // below sizes allocations from it. Ordering is the whole point: `read_coefficients`
        // allocates four vectors of `domain_size + 1` and section 9 is read as
        // `domain_size` points, and the length check that would refute a lie used to run
        // afterwards. A 4 KB file claiming 2^31 therefore cost 34 GB and 11.5 seconds
        // before erroring. Same class as CVE-2024-50354 in gnark.
        //
        // 1. Shape and two-adicity, see `check_domain_size`.
        check_domain_size(domain_size)?;
        // 2. Cross-check against the file that is actually present. Section 9 holds exactly
        //    `domain_size` G1 points, so its length refutes a lying header for the cost of
        //    a slice lookup, with nothing allocated yet.
        expect_records(file.unique_section(9)?, domain_size, G1_BYTES, 9)?;
        // `n_public` is attacker-supplied and `+ 1` is not free. On a 32-bit `usize` - and
        // this crate is built for wasm32 - `u32::MAX + 1` wraps to 0 in release, where
        // overflow checks are off. Section 3 is then read as an empty IC, the guard below
        // becomes `n_vars < 0` and never fires, and `n_vars - n_ic` wraps back to `n_vars`.
        // That is the same empty-`ic` key `want_ic` on the JSON path exists to refuse,
        // reached through the other door.
        let n_ic = n_public
            .checked_add(1)
            .ok_or_else(|| ZkeyError::Malformed {
                section: 2,
                reason: format!("n_public {n_public} is impossibly large"),
            })?;
        // The L query covers the private wires only, so an n_vars that does not leave
        // room for `1 + n_public` would underflow the section 8 length below.
        if n_vars < n_ic {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!("n_vars {n_vars} does not cover 1 + n_public {n_public}"),
            });
        }

        // Unchecked here because every IC point gets the full check further down, in both
        // modes: there are `n_public + 1` of them and the verifier depends on them.
        let ic = read_g1_section(&file, 3, n_ic, Check::Unchecked, &mut Keep)?.points;
        let coeffs = read_coefficients(&file.map_section(4)?, n_vars, domain_size, sink)?;
        let a_query = read_g1_section(&file, 5, n_vars, check, sink)?;
        let b_g1_query = read_g1_section(&file, 6, n_vars, check, sink)?;
        let b_g2_query = read_g2_section(&file, 7, n_vars, check)?;
        let l_query = read_g1_section(&file, 8, n_vars - n_ic, check, sink)?;
        let h_query = read_g1_section(&file, 9, domain_size, check, sink)?;

        // Only the O(1) points are validated on load. The query sections are millions of
        // points on a real circuit and a subgroup check each would dominate key load, so
        // they are checked in tests instead. Validating these six is still worth it: a
        // wrong offset anywhere in the header shows up here rather than as an unverifiable
        // proof twenty seconds later.
        check_g1(&alpha_g1, 2, "alpha_g1")?;
        check_g1(&beta_g1, 2, "beta_g1")?;
        check_g2(&beta_g2, 2, "beta_g2")?;
        check_g2(&gamma_g2, 2, "gamma_g2")?;
        check_g1(&delta_g1, 2, "delta_g1")?;
        check_g2(&delta_g2, 2, "delta_g2")?;
        // And none of the six may be the point at infinity. This is the gate that stops a
        // malicious key from silently switching zero knowledge off.
        //
        // `check_g1` and `check_g2` do not catch it: arkworks' `is_on_curve` and
        // `is_in_correct_subgroup_assuming_on_curve` both return TRUE for the identity, and
        // ffjavascript encodes infinity as affine (0, 0), which `binfile::g1` faithfully
        // maps to the identity. So an all-zero point passes every check above.
        //
        // The attack, confirmed by running it: zero `beta_g1`, `delta_g1` and the whole of
        // the section 6 B-in-G1 query. Then `pi_a = alpha + sum(w_j A_j) + r * delta_g1`
        // loses its `r` term and becomes a deterministic function of the entire witness,
        // and `pib1` collapses to the identity. Proofs from that key still verify against
        // the untouched genuine `vkey.json`, with `pi_a` bit-identical across runs, because
        // none of the three edited quantities appears in the verification key. No verifier
        // anywhere can detect it: not ours, not snarkjs, not rapidsnark. The result is a
        // witness confirmation oracle, and outright recovery for a low-entropy witness.
        //
        // Rejecting is safe: every one of these six is [x]_1 or [x]_2 for a nonzero element
        // of the toxic waste, so a real setup can never produce the identity here. Only a
        // key that was tampered with or generated with zero toxic waste can trip this.
        for (is_inf, what) in [
            (alpha_g1.infinity, "alpha_g1"),
            (beta_g1.infinity, "beta_g1"),
            (delta_g1.infinity, "delta_g1"),
        ] {
            if is_inf {
                return Err(ZkeyError::Malformed {
                    section: 2,
                    reason: format!(
                        "{what} is the point at infinity, which no honest setup produces; \
                         this key cannot provide zero knowledge"
                    ),
                });
            }
        }
        for (is_inf, what) in [
            (beta_g2.infinity, "beta_g2"),
            (gamma_g2.infinity, "gamma_g2"),
            (delta_g2.infinity, "delta_g2"),
        ] {
            if is_inf {
                return Err(ZkeyError::Malformed {
                    section: 2,
                    reason: format!(
                        "{what} is the point at infinity, which no honest setup produces; \
                         this key cannot provide zero knowledge"
                    ),
                });
            }
        }
        // A whole query section of identities is the other half of the same attack. Note
        // this is deliberately an ALL check and not an ANY check: real snarkjs keys do
        // contain points at infinity in these sections, 33.9% of the B query in
        // js_16x16_d32, so rejecting them individually would reject every real key.
        for (section, all_inf, n) in [
            (5u32, a_query.all_infinity, a_query.n),
            (6, b_g1_query.all_infinity, b_g1_query.n),
            (7, b_g2_query.iter().all(|p| p.infinity), b_g2_query.len()),
            (9, h_query.all_infinity, h_query.n),
        ] {
            if n > 0 && all_inf {
                return Err(ZkeyError::Malformed {
                    section,
                    reason: format!(
                        "every one of the {n} points in this query section is the point at \
                         infinity; the section has been zeroed"
                    ),
                });
            }
        }
        for (i, p) in ic.iter().enumerate() {
            check_g1(p, 3, &format!("ic[{i}]"))?;
        }
        if check == Check::Full && gamma_g2 == delta_g2 {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: "gamma_g2 equals delta_g2, which is what a key with no phase-2 \
                         contribution looks like; anyone can forge a proof against it. \
                         Contribute to phase 2, or load it with `load_unchecked` for \
                         development"
                    .into(),
            });
        }

        Ok(Self {
            n_vars,
            n_public,
            domain_size,
            alpha_g1,
            beta_g1,
            beta_g2,
            delta_g1,
            delta_g2,
            a_query: a_query.points,
            b_g1_query: b_g1_query.points,
            b_g2_query,
            l_query: l_query.points,
            h_query: h_query.points,
            coeffs,
            vk: VerifyingKey {
                alpha_g1,
                beta_g2,
                gamma_g2,
                delta_g2,
                ic,
            },
        })
    }
}

/// The largest `log2(domainSize)` a key can have and still be proved.
///
/// Every prover builds its coset shift from `Domain::new(2 * domain_size)`, because
/// snarkjs evaluates on the odd 2n-th roots of unity. So the binding constraint is a
/// 2n-th root, not the n-th root the NTT needs, and Fr has one only up to
/// 2^TWO_ADICITY (28 for BN254). That leaves 2^27.
const MAX_LOG_DOMAIN_SIZE: u32 = Fr::TWO_ADICITY - 1;

/// Refuses a header `domainSize` that is not a power of two, or that no prover can handle.
/// A larger domain has no 2n-th root of unity, so accepting it buys nothing and costs an
/// allocation, and would only fail later at prepare with a less useful message.
fn check_domain_size(domain_size: usize) -> Result<(), ZkeyError> {
    if domain_size == 0 || !domain_size.is_power_of_two() {
        return Err(ZkeyError::Malformed {
            section: 2,
            reason: format!("domain size {domain_size} is not a power of two"),
        });
    }
    let log_size = domain_size.trailing_zeros();
    if log_size > MAX_LOG_DOMAIN_SIZE {
        return Err(ZkeyError::Malformed {
            section: 2,
            reason: format!(
                "domain size 2^{log_size} exceeds 2^{MAX_LOG_DOMAIN_SIZE}: the prover needs a \
                 2n-th root of unity and the two-adicity of Fr is {}",
                Fr::TWO_ADICITY
            ),
        });
    }
    Ok(())
}

/// Reads `u32 n8, modulus[n8]` and rejects anything but the expected BN254 prime.
fn check_modulus(c: &mut Cursor<'_>, want: &[u8]) -> Result<(), ZkeyError> {
    let n8 = c.u32()? as usize;
    if n8 != want.len() {
        return Err(ZkeyError::UnsupportedCurve);
    }
    if c.take(n8)? != want {
        return Err(ZkeyError::UnsupportedCurve);
    }
    Ok(())
}

/// One G1 section: its points, unless a [`KeySink`] took them, and what the infinity gate
/// in [`ProvingKey::parse`] needs to know about them either way.
struct G1Section {
    points: Vec<G1Affine>,
    n: usize,
    all_infinity: bool,
}

fn read_g1_section<S: KeySink>(
    file: &BinFile,
    id: u32,
    n: usize,
    check: Check,
    sink: &mut S,
) -> Result<G1Section, ZkeyError> {
    let section = file.map_section(id)?;
    let data = &*section;
    expect_records(data, n, G1_BYTES, id)?;
    // One pass over the mapped bytes straight into the output: the section is never
    // materialised as an intermediate buffer. The curve check rides in the same pass,
    // while the point is still in cache, which is what makes it close to free. BN254 G1
    // has cofactor 1, so on the curve is in the subgroup.
    let decode = |i: usize, b: &[u8]| {
        let p = g1(b)?;
        if check != Check::Unchecked && !valid_g1(&p) {
            return Err(off_curve(id, i));
        }
        Ok(p)
    };
    if let Some(PointsInto { points, infinity }) = sink.g1_section(id, n)? {
        assert_eq!(points.len(), n, "sink room for section {id} points");
        assert_eq!(infinity.len(), n, "sink room for section {id} flags");
        // What is stored is the decoded point, not the bytes again: on the mmap path a
        // second read could see a file someone changed after the check.
        let first_err = FirstError::default();
        let finite = data
            .par_chunks_exact(G1_BYTES)
            .zip(points.par_iter_mut())
            .zip(infinity.par_iter_mut())
            .enumerate()
            .map(|(i, ((b, out), inf))| {
                let p = decode(i, b).unwrap_or_else(|e| {
                    first_err.record(i, e);
                    G1Affine::identity()
                });
                *out = S::g1(&p);
                *inf = p.infinity;
                !p.infinity
            })
            .reduce(|| false, |a, b| a || b);
        first_err.into_result()?;
        return Ok(G1Section {
            points: Vec::new(),
            n,
            all_infinity: !finite,
        });
    }
    let points = decode_records(data, G1_BYTES, G1Affine::identity(), decode)?;
    Ok(G1Section {
        all_infinity: points.iter().all(|p| p.infinity),
        n,
        points,
    })
}

fn read_g2_section(
    file: &BinFile,
    id: u32,
    n: usize,
    check: Check,
) -> Result<Vec<G2Affine>, ZkeyError> {
    let section = file.map_section(id)?;
    let data = &*section;
    expect_records(data, n, G2_BYTES, id)?;
    // On the curve only. See `ProvingKey::load` for why the subgroup is checked on the
    // proof rather than on each of these points.
    decode_records(data, G2_BYTES, G2Affine::identity(), |i, b| {
        let p = g2(b)?;
        if check != Check::Unchecked && !p.is_on_curve() {
            return Err(off_curve(id, i));
        }
        Ok(p)
    })
}

fn off_curve(section: u32, i: usize) -> ZkeyError {
    ZkeyError::Malformed {
        section,
        reason: format!("point {i} is not on the curve"),
    }
}

fn valid_g1(p: &G1Affine) -> bool {
    p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()
}

fn valid_g2(p: &G2Affine) -> bool {
    p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()
}

fn check_g1(p: &G1Affine, section: u32, what: &str) -> Result<(), ZkeyError> {
    if !valid_g1(p) {
        return Err(ZkeyError::Malformed {
            section,
            reason: format!("{what} is not a valid G1 point"),
        });
    }
    Ok(())
}

fn check_g2(p: &G2Affine, section: u32, what: &str) -> Result<(), ZkeyError> {
    if !valid_g2(p) {
        return Err(ZkeyError::Malformed {
            section,
            reason: format!("{what} is not a valid G2 point"),
        });
    }
    Ok(())
}

/// Section 4 into CSR, by counting sort on `(matrix, constraint)`.
///
/// Counting sort rather than `sort_by_key`: the histogram *is* the `row_ptr` we need, so
/// one counting pass plus one placement pass produces both the ordering and the index in
/// O(n + domain_size), with no comparison sort and no hash map. It is also stable, which
/// keeps the placement deterministic across runs and across backends.
///
/// Both passes run over the pool. The count is atomic increments; the placement claims
/// each record's slot with an atomic bump of its row's cursor, which leaves a row's
/// records in whatever order the threads got there, and then sorts each row by record
/// index, which is the order the serial placement left them in. Rows hold a handful of
/// records, so that sort is an insertion sort. On keccak_2048's 11.9 million records the
/// two serial passes were 38 and 45 ms.
fn read_coefficients<S: KeySink>(
    data: &[u8],
    n_vars: usize,
    domain_size: usize,
    sink: &mut S,
) -> Result<Coefficients, ZkeyError> {
    let mut c = Cursor::new(data, 4);
    let n_coefs = c.u32()? as usize;
    expect_records(&data[4..], n_coefs, COEF_RECORD, 4)?;

    let check_record = |i: usize| -> Result<(usize, usize), ZkeyError> {
        let (m, constraint, _, _) = coef_indices(data, i)?;
        if constraint >= domain_size {
            return Err(ZkeyError::Malformed {
                section: 4,
                reason: format!("constraint {constraint} is outside domain size {domain_size}"),
            });
        }
        Ok((m, constraint))
    };
    // Every record is checked before any is counted, and the error is the first record's
    // in file order, as the serial pass reported it.
    let atomic = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
    let counts = [atomic(domain_size + 1), atomic(domain_size + 1)];
    let first_err = FirstError::default();
    (0..n_coefs)
        .into_par_iter()
        .for_each(|i| match check_record(i) {
            Ok((m, constraint)) => {
                counts[m][constraint + 1].fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => first_err.record(i, e),
        });
    first_err.into_result()?;

    // Prefix sum in place turns the histogram into the row offsets.
    let mut row_ptr = counts.map(|c| c.into_iter().map(AtomicU32::into_inner).collect::<Vec<_>>());
    for ptr in &mut row_ptr {
        for i in 0..domain_size {
            ptr[i + 1] += ptr[i];
        }
    }

    let totals = [
        row_ptr[0][domain_size] as usize,
        row_ptr[1][domain_size] as usize,
    ];
    let mut into = sink.coefficients(domain_size, totals)?;
    if let Some(into) = &into {
        for (m, d) in into.iter().enumerate() {
            assert_eq!(
                d.row_ptr.len(),
                domain_size + 1,
                "sink room for row_ptr {m}"
            );
            assert_eq!(d.signal.len(), totals[m], "sink room for signal {m}");
            assert_eq!(d.value.len(), totals[m], "sink room for value {m}");
        }
    }
    let mut kept = match into {
        Some(_) => [Vec::new(), Vec::new()],
        None => [vec![0u32; totals[0]], vec![0u32; totals[1]]],
    };
    let signal: [&mut [u32]; 2] = match &mut into {
        Some([a, b]) => [&mut *a.signal, &mut *b.signal],
        None => {
            let [a, b] = &mut kept;
            [a, b]
        }
    };
    // The record each slot came from. The placement only moves indices; the Montgomery
    // decode of each value then runs in parallel per slot.
    let mut source = [vec![0u32; totals[0]], vec![0u32; totals[1]]];

    let r_inv = r_inv();
    let value_of = |i: usize| {
        let off = 4 + i * COEF_RECORD + 12;
        fr_double_montgomery(&data[off..off + FR_BYTES], &r_inv)
    };
    // The serial loop decoded values in record order, so a bad value before a bad index
    // was the error reported. Keep that order.
    let index_error = |i: usize, err: ZkeyError| match (0..i).find_map(|j| value_of(j).err()) {
        Some(value_err) => value_err,
        None => err,
    };
    let cursor = row_ptr
        .each_ref()
        .map(|rp| rp.iter().map(|&p| AtomicU32::new(p)).collect::<Vec<_>>());
    let [sig_a, sig_b] = signal;
    let signal_at = [Rows::new(sig_a), Rows::new(sig_b)];
    let source_at = source.each_mut().map(|s| Rows::new(s));
    // `Some` for an index error, which the serial pass reported after any bad value in an
    // earlier record; `None` for a matrix id, which it reported as it was.
    let first_err = FirstError::default();
    (0..n_coefs).into_par_iter().for_each(|i| {
        let (m, constraint, sig, _) = match coef_indices(data, i) {
            Ok(r) => r,
            Err(e) => return first_err.record(i, (e, false)),
        };
        if sig >= n_vars {
            let e = ZkeyError::Malformed {
                section: 4,
                reason: format!("signal {sig} is outside n_vars {n_vars}"),
            };
            return first_err.record(i, (e, true));
        }
        // Pass 1 checked `constraint`, but on the mmap path this is a second read of a file
        // someone else may be writing, and a record that changed in between indexed past
        // the end: SIGABRT on `js_16x16_d32` in 4 of 6 live races. Re-check both indices
        // so a changed record is an error rather than an abort or a silently wrong CSR.
        let slot = (constraint < domain_size)
            .then(|| cursor[m][constraint].fetch_add(1, Ordering::Relaxed))
            .filter(|&slot| slot < row_ptr[m][constraint + 1]);
        let Some(slot) = slot else {
            let e = ZkeyError::Malformed {
                section: 4,
                reason: format!("record {i} changed between the two passes over section 4"),
            };
            return first_err.record(i, (e, true));
        };
        // SAFETY: the bump hands every slot below the row's end to exactly one record.
        unsafe {
            signal_at[m].write(slot as usize, sig as u32);
            source_at[m].write(slot as usize, i as u32);
        }
    });
    if let Some((i, (e, after_values))) = first_err.into_first() {
        return Err(if after_values { index_error(i, e) } else { e });
    }
    for m in 0..2 {
        let rp = &row_ptr[m];
        (0..domain_size)
            .into_par_iter()
            .with_min_len(1 << 12)
            .for_each(|c| {
                let (lo, hi) = (rp[c] as usize, rp[c + 1] as usize);
                // SAFETY: rows are disjoint ranges of both vectors, and the placement that
                // wrote them has finished.
                unsafe { sort_row(&source_at[m], &signal_at[m], lo, hi) };
            });
    }

    let Some(into) = into else {
        let raw = |i: u32| -> &[u8; FR_BYTES] {
            let off = 4 + i as usize * COEF_RECORD + 12;
            data[off..off + FR_BYTES]
                .try_into()
                .expect("slice is FR_BYTES long")
        };
        let [a, b] = &source;
        let (value_id, values) = match intern(&source, raw) {
            // Every record's value is one of `distinct`, so decoding those checks them
            // all. On a failure, the error is the first slot's in A then B, as it is when
            // every slot is decoded.
            Some((value_id, distinct)) => {
                let values = collect_records(
                    distinct
                        .par_iter()
                        .map(|b| fr_double_montgomery(&b[..], &r_inv)),
                    Fr::zero(),
                );
                let values = match values {
                    Ok(v) => v,
                    Err(_) => {
                        return Err(a
                            .iter()
                            .chain(b)
                            .find_map(|&i| value_of(i as usize).err())
                            .expect("a distinct value failed, so some record's did"))
                    }
                };
                (value_id, values)
            }
            None => {
                let values = collect_records(
                    a.par_iter().chain(b).map(|&i| value_of(i as usize)),
                    Fr::zero(),
                )?;
                let n_a = a.len() as u32;
                let value_id = [
                    (0..n_a).into_par_iter().collect(),
                    (n_a..n_a + b.len() as u32).into_par_iter().collect(),
                ];
                (value_id, values)
            }
        };
        return Ok(Coefficients {
            row_ptr,
            signal: kept,
            value_id,
            values,
        });
    };
    for (m, d) in into.into_iter().enumerate() {
        let first_err = FirstError::default();
        d.value
            .par_iter_mut()
            .zip(source[m].par_iter())
            .enumerate()
            .for_each(|(slot, (out, &i))| match value_of(i as usize) {
                Ok(v) => *out = S::fr(&v),
                Err(e) => first_err.record(slot, e),
            });
        first_err.into_result()?;
        d.row_ptr.copy_from_slice(&row_ptr[m]);
    }
    Ok(Coefficients {
        row_ptr: Default::default(),
        signal: Default::default(),
        value_id: Default::default(),
        values: Vec::new(),
    })
}

/// Slots per chunk of [`intern`]'s parallel pass.
const INTERN_CHUNK: usize = 1 << 16;
/// Lines in each chunk's cache of recently seen values.
const INTERN_LINES: usize = 1 << 12;
/// Fewer nonzeros than this keep a value each. Interning saves such a key at most 7 MB,
/// and with a chunk or two to spread over it loaded tornado's 88 thousand 0.3 ms slower
/// than decoding every value.
const INTERN_MIN: usize = 1 << 18;

/// Both matrices' slot indices, and the distinct raw values they index.
type Interned<'d> = ([Vec<u32>; 2], Vec<&'d [u8; FR_BYTES]>);

/// Each slot's index into the distinct raw values of both matrices, and those values in
/// first-seen order, A before B. `raw(i)` is record `i`'s value bytes, and `source[m]`
/// names the record behind each slot of matrix `m`.
///
/// Each chunk interns its slots into a map of its own, then a serial merge numbers the
/// chunks' values and a second parallel pass renumbers each chunk's slots. `None` for a
/// key below [`INTERN_MIN`] nonzeros, or whose chunks hold more than one distinct value
/// per eight slots: it is not worth a map, and its slots keep a value each, which is 4
/// bytes per nonzero more than storing the values alone and costs no merge.
fn intern<'d>(
    source: &[Vec<u32>; 2],
    raw: impl Fn(u32) -> &'d [u8; FR_BYTES] + Sync,
) -> Option<Interned<'d>> {
    use std::collections::HashMap;

    let total = source[0].len() + source[1].len();
    if total < INTERN_MIN {
        return None;
    }
    let mut value_id = source.each_ref().map(|s| vec![0u32; s.len()]);
    let [id_a, id_b] = &mut value_id;
    let local: Vec<Vec<&[u8; FR_BYTES]>> = id_a
        .par_chunks_mut(INTERN_CHUNK)
        .zip(source[0].par_chunks(INTERN_CHUNK))
        .chain(
            id_b.par_chunks_mut(INTERN_CHUNK)
                .zip(source[1].par_chunks(INTERN_CHUNK)),
        )
        .map(|(ids, src)| {
            let mut seen = HashMap::new();
            let mut distinct = Vec::new();
            // A direct-mapped cache in front of the map, on the value's low 12 bits, which
            // are as good as random for a Montgomery form: most slots hit it and skip the
            // hash.
            let mut recent: Vec<Option<(&[u8; FR_BYTES], u32)>> = vec![None; INTERN_LINES];
            for (id, &i) in ids.iter_mut().zip(src) {
                let b = raw(i);
                let line = u16::from_le_bytes([b[0], b[1]]) as usize % INTERN_LINES;
                if let Some((k, v)) = recent[line] {
                    if k == b {
                        *id = v;
                        continue;
                    }
                }
                *id = *seen.entry(b).or_insert_with(|| {
                    distinct.push(b);
                    (distinct.len() - 1) as u32
                });
                recent[line] = Some((b, *id));
            }
            distinct
        })
        .collect();
    if local.iter().map(Vec::len).sum::<usize>() > total / 8 {
        return None;
    }

    let mut seen = HashMap::new();
    let mut distinct = Vec::new();
    let renumber: Vec<Vec<u32>> = local
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .map(|&b| {
                    *seen.entry(b).or_insert_with(|| {
                        distinct.push(b);
                        (distinct.len() - 1) as u32
                    })
                })
                .collect()
        })
        .collect();
    let [id_a, id_b] = &mut value_id;
    id_a.par_chunks_mut(INTERN_CHUNK)
        .chain(id_b.par_chunks_mut(INTERN_CHUNK))
        .zip(renumber.par_iter())
        .for_each(|(ids, to)| ids.iter_mut().for_each(|id| *id = to[*id as usize]));
    Some((value_id, distinct))
}

/// A vector the placement pass writes from many threads, one slot each.
struct Rows<'a> {
    ptr: *mut u32,
    len: usize,
    _borrow: std::marker::PhantomData<&'a mut [u32]>,
}

// SAFETY: only written through `write` and `sort_row`, whose callers keep threads on
// disjoint slots.
unsafe impl Send for Rows<'_> {}
unsafe impl Sync for Rows<'_> {}

impl<'a> Rows<'a> {
    fn new(v: &'a mut [u32]) -> Self {
        Self {
            ptr: v.as_mut_ptr(),
            len: v.len(),
            _borrow: std::marker::PhantomData,
        }
    }

    /// # Safety
    ///
    /// No other thread touches slot `i` meanwhile.
    unsafe fn write(&self, i: usize, v: u32) {
        assert!(i < self.len);
        *self.ptr.add(i) = v;
    }
}

/// Sorts `source[lo..hi]` ascending, moving `signal[lo..hi]` along with it.
///
/// # Safety
///
/// No other thread touches `lo..hi` of either vector meanwhile.
unsafe fn sort_row(source: &Rows<'_>, signal: &Rows<'_>, lo: usize, hi: usize) {
    assert!(lo <= hi && hi <= source.len && hi <= signal.len);
    let src = std::slice::from_raw_parts_mut(source.ptr.add(lo), hi - lo);
    let sig = std::slice::from_raw_parts_mut(signal.ptr.add(lo), hi - lo);
    // Insertion sort is quadratic, and a key may put any number of records in one row.
    if src.len() > 32 {
        let mut pairs: Vec<(u32, u32)> = src.iter().copied().zip(sig.iter().copied()).collect();
        pairs.sort_unstable_by_key(|&(i, _)| i);
        for (k, (i, s)) in pairs.into_iter().enumerate() {
            src[k] = i;
            sig[k] = s;
        }
        return;
    }
    for k in 1..src.len() {
        let mut j = k;
        while j > 0 && src[j - 1] > src[j] {
            src.swap(j - 1, j);
            sig.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// `(matrix, constraint, signal, offset of the value)` for record `i`.
fn coef_indices(data: &[u8], i: usize) -> Result<(usize, usize, usize, usize), ZkeyError> {
    let base = 4 + i * COEF_RECORD;
    let rd = |off: usize| -> u32 {
        u32::from_le_bytes(data[off..off + 4].try_into().expect("slice is 4 bytes"))
    };
    let m = rd(base) as usize;
    // snarkjs filters out matrix 2 (the C matrix) before writing, so anything but A or B
    // means we are reading at the wrong offset.
    if m > 1 {
        return Err(ZkeyError::Malformed {
            section: 4,
            reason: format!("record {i} has matrix id {m}, expected 0 or 1"),
        });
    }
    Ok((m, rd(base + 4) as usize, rd(base + 8) as usize, base + 12))
}

impl VerifyingKey {
    /// Refuse a key that no honest setup produces and that breaks soundness. O(1), so the
    /// checked `verify` in `snarkrs-groth16` runs it on every call: the fields are `pub`, and a
    /// key built by hand or over FFI never went through a loader.
    ///
    /// * `alpha_g1`, `beta_g2`, `gamma_g2` or `delta_g2` at infinity. With every pair
    ///   degenerate the pairing product is the empty product, 1, and every proof verifies.
    /// * `gamma_g2 == delta_g2`: no phase-2 contribution. `A = alpha`, `B = beta`,
    ///   `C = -L` then verifies for any public input. See [`ProvingKey::load`].
    /// * An empty `ic`, which has no constant-wire point.
    pub fn check_structure(&self) -> Result<(), ZkeyError> {
        self.structure(false)
    }

    /// [`VerifyingKey::check_structure`] without the `gamma_g2 != delta_g2` refusal, for a
    /// key [`ProvingKey::load_allowing_unblinded`] loaded. Benchmark only, for the same
    /// reason: anyone can forge a proof that passes against such a key.
    pub fn check_structure_allowing_unblinded(&self) -> Result<(), ZkeyError> {
        self.structure(true)
    }

    fn structure(&self, allow_unblinded: bool) -> Result<(), ZkeyError> {
        let bad = |what| Err(ZkeyError::UnsafeVerifyingKey(what));
        if self.alpha_g1.infinity {
            return bad("alpha_g1 is the point at infinity");
        }
        if self.beta_g2.infinity {
            return bad("beta_g2 is the point at infinity");
        }
        if self.gamma_g2.infinity {
            return bad("gamma_g2 is the point at infinity");
        }
        if self.delta_g2.infinity {
            return bad("delta_g2 is the point at infinity");
        }
        if !allow_unblinded && self.gamma_g2 == self.delta_g2 {
            return bad(
                "gamma_g2 equals delta_g2, so the key had no phase-2 contribution and anyone \
                 can forge a proof against it",
            );
        }
        if self.ic.is_empty() {
            return bad("IC is empty; it must carry at least the constant-wire point");
        }
        Ok(())
    }

    /// Parse snarkjs' `verification_key.json`, so we can verify against the same key
    /// snarkjs uses without trusting our own zkey reader. Every point is checked on the
    /// curve and in the subgroup, and [`VerifyingKey::check_structure`] runs.
    ///
    /// Native only, because there is no filesystem in a browser. The browser path is
    /// [`VerifyingKey::from_json_str`], which this is a thin wrapper over.
    #[cfg(not(target_family = "wasm"))]
    pub fn from_json(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    /// [`VerifyingKey::from_json`] without [`VerifyingKey::check_structure`], for a
    /// development key that never had a phase-2 contribution. The points are still
    /// checked: that costs microseconds and an off-subgroup point is never wanted.
    #[cfg(not(target_family = "wasm"))]
    pub fn from_json_unchecked(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::from_json_str_unchecked(&std::fs::read_to_string(path)?)
    }

    /// Same as [`VerifyingKey::from_json`], on text the caller already has.
    pub fn from_json_str(text: &str) -> Result<Self, ZkeyError> {
        let vk = Self::from_json_str_unchecked(text)?;
        vk.check_structure()?;
        Ok(vk)
    }

    /// Same as [`VerifyingKey::from_json_unchecked`], on text the caller already has.
    ///
    /// The page in `../webgpu-trial` cross-checks snarkjs' proofs against our own verifier,
    /// and it has `vkey.json` as a string from `fetch`, not as a path.
    pub fn from_json_str_unchecked(text: &str) -> Result<Self, ZkeyError> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| ZkeyError::BadJson(e.to_string()))?;

        match v.get("protocol").and_then(|p| p.as_str()) {
            Some("groth16") => {}
            other => {
                return Err(ZkeyError::BadJson(format!(
                    "protocol is {other:?}, expected groth16"
                )))
            }
        }
        match v.get("curve").and_then(|c| c.as_str()) {
            Some("bn128") => {}
            other => {
                return Err(ZkeyError::BadJson(format!(
                    "curve is {other:?}, expected bn128"
                )))
            }
        }

        let ic_json = v
            .get("IC")
            .and_then(|i| i.as_array())
            .ok_or_else(|| ZkeyError::BadJson("IC is missing or not an array".into()))?;
        let n_public = v
            .get("nPublic")
            .and_then(|n| n.as_u64())
            .ok_or_else(|| ZkeyError::BadJson("nPublic is missing".into()))?
            as usize;
        // `n_public` is attacker-supplied and `+ 1` is not free: in release, where overflow
        // checks are off, `usize::MAX + 1` wraps to 0 and an empty IC then satisfies this
        // guard. That produced a key with no points at all, which `aggregate_public` indexed
        // and died on.
        let want_ic = n_public
            .checked_add(1)
            .ok_or_else(|| ZkeyError::BadJson(format!("nPublic {n_public} is impossibly large")))?;
        if ic_json.len() != want_ic {
            return Err(ZkeyError::BadJson(format!(
                "IC has {} entries, nPublic {n_public} implies {want_ic}",
                ic_json.len(),
            )));
        }

        let mut ic = Vec::with_capacity(ic_json.len());
        for (i, p) in ic_json.iter().enumerate() {
            ic.push(json_g1(p, &format!("IC[{i}]"))?);
        }

        Ok(Self {
            alpha_g1: json_g1(field(&v, "vk_alpha_1")?, "vk_alpha_1")?,
            beta_g2: json_g2(field(&v, "vk_beta_2")?, "vk_beta_2")?,
            gamma_g2: json_g2(field(&v, "vk_gamma_2")?, "vk_gamma_2")?,
            delta_g2: json_g2(field(&v, "vk_delta_2")?, "vk_delta_2")?,
            ic,
        })
    }
}

fn field<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, ZkeyError> {
    v.get(key)
        .ok_or_else(|| ZkeyError::BadJson(format!("{key} is missing")))
}

/// A decimal string in *ordinary* form, the opposite convention to the binary sections.
fn json_fq(v: &serde_json::Value, what: &str) -> Result<Fq, ZkeyError> {
    let s = v
        .as_str()
        .ok_or_else(|| ZkeyError::BadJson(format!("{what} is not a string")))?;
    let n: num_bigint::BigUint = s
        .parse()
        .map_err(|_| ZkeyError::BadJson(format!("{what} is not a decimal integer: {s:?}")))?;
    let bi = ark_ff::BigInt::<4>::try_from(n)
        .map_err(|_| ZkeyError::BadJson(format!("{what} does not fit in 256 bits")))?;
    Fq::from_bigint(bi).ok_or_else(|| {
        ZkeyError::BadJson(format!("{what} is not below the base field modulus: {s}"))
    })
}

/// snarkjs emits G1 as projective `[x, y, z]`. The z is not decoration: `[0, 1, 0]` is
/// how it writes the point at infinity, so ignoring z would turn infinity into a bogus
/// affine `(0, 1)`.
fn json_g1(v: &serde_json::Value, what: &str) -> Result<G1Affine, ZkeyError> {
    let a = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| ZkeyError::BadJson(format!("{what} is not a 3-element array")))?;
    let x = json_fq(&a[0], what)?;
    let y = json_fq(&a[1], what)?;
    let z = json_fq(&a[2], what)?;
    if z.is_zero() {
        return Ok(G1Affine::identity());
    }
    if z != Fq::ONE {
        return Err(ZkeyError::BadJson(format!(
            "{what} has z = {z}, expected 1 or 0"
        )));
    }
    let p = G1Affine::new_unchecked(x, y);
    if !valid_g1(&p) {
        return Err(ZkeyError::BadJson(format!(
            "{what} is not a valid G1 point"
        )));
    }
    Ok(p)
}

/// `[[x0, x1], [y0, y1], [z0, z1]]`, with the `Fq2` components in `c0, c1` order.
fn json_g2(v: &serde_json::Value, what: &str) -> Result<G2Affine, ZkeyError> {
    let a = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| ZkeyError::BadJson(format!("{what} is not a 3-element array")))?;
    let comp = |i: usize| -> Result<Fq2, ZkeyError> {
        let c = a[i]
            .as_array()
            .filter(|c| c.len() == 2)
            .ok_or_else(|| ZkeyError::BadJson(format!("{what}[{i}] is not a 2-element array")))?;
        Ok(Fq2::new(json_fq(&c[0], what)?, json_fq(&c[1], what)?))
    };
    let x = comp(0)?;
    let y = comp(1)?;
    let z = comp(2)?;
    if z.is_zero() {
        return Ok(G2Affine::identity());
    }
    if z != Fq2::ONE {
        return Err(ZkeyError::BadJson(format!(
            "{what} has z = {z}, expected [1, 0] or [0, 0]"
        )));
    }
    let p = G2Affine::new_unchecked(x, y);
    if !valid_g2(&p) {
        return Err(ZkeyError::BadJson(format!(
            "{what} is not a valid G2 point"
        )));
    }
    Ok(p)
}

#[cfg(test)]
mod csr_tests {
    use super::*;

    fn coefficients() -> Coefficients {
        Coefficients {
            row_ptr: [vec![0, 1, 1, 2, 2], vec![0, 1, 1, 2, 2]],
            signal: [vec![0, 1], vec![0, 1]],
            value_id: [vec![0; 2], vec![0; 2]],
            values: vec![Fr::one()],
        }
    }

    fn rejects(c: &Coefficients, m: usize, message: &str) {
        let ZkeyError::Malformed { section, reason } = c.check_structure(4, 2).unwrap_err() else {
            panic!("expected a malformed CSR error");
        };
        assert_eq!(section, 4);
        assert!(
            reason.contains(&format!("matrix {}", ["A", "B"][m])),
            "{reason}"
        );
        assert!(reason.contains(message), "{reason}");
    }

    #[test]
    fn rejects_nonzero_origins() {
        for m in 0..2 {
            let mut c = coefficients();
            c.row_ptr[m][0] = 1;
            rejects(&c, m, "starts at 1, expected 0");
        }
    }

    #[test]
    fn rejects_wrong_row_lengths() {
        for m in 0..2 {
            for rows in [vec![], vec![0, 1, 1, 2], vec![0, 1, 1, 2, 2, 2]] {
                let mut c = coefficients();
                c.row_ptr[m] = rows;
                rejects(&c, m, "row_ptr entries");
            }
        }
    }

    #[test]
    fn rejects_wrong_totals_and_array_lengths() {
        for m in 0..2 {
            for fault in 0..4 {
                let mut c = coefficients();
                match fault {
                    0 => c.row_ptr[m] = vec![0, 1, 1, 1, 1],
                    1 => c.row_ptr[m][4] = 3,
                    2 => {
                        c.signal[m].pop();
                    }
                    _ => {
                        c.value_id[m].pop();
                    }
                }
                rejects(&c, m, "row_ptr ends at");
            }
        }
    }

    #[test]
    fn rejects_decreasing_rows() {
        for m in 0..2 {
            for rows in [vec![0, 2, 1, 2, 2], vec![0, 3, 3, 3, 2]] {
                let mut c = coefficients();
                c.row_ptr[m] = rows;
                rejects(&c, m, "not monotone");
            }
        }
    }

    #[test]
    fn rejects_out_of_range_signals() {
        for m in 0..2 {
            for signal in [2, u32::MAX] {
                let mut c = coefficients();
                c.signal[m][1] = signal;
                rejects(&c, m, "beyond n_vars 2");
            }
        }
    }

    #[test]
    fn rejects_out_of_range_value_ids() {
        for m in 0..2 {
            for id in [1, u32::MAX] {
                let mut c = coefficients();
                c.value_id[m][1] = id;
                rejects(&c, m, "beyond the 1 values");
            }
        }
    }

    #[test]
    fn accepts_empty_rows_and_matrices() {
        assert!(coefficients().check_structure(4, 2).is_ok());
        for m in 0..2 {
            let mut c = coefficients();
            c.row_ptr[m].fill(0);
            c.signal[m].clear();
            c.value_id[m].clear();
            assert!(c.check_structure(4, 2).is_ok());
        }
        let c = Coefficients {
            row_ptr: [vec![0; 5], vec![0; 5]],
            signal: [vec![], vec![]],
            value_id: [vec![], vec![]],
            values: vec![],
        };
        assert!(c.check_structure(4, 0).is_ok());
    }

    #[test]
    fn rejects_overflowing_row_count() {
        assert!(matches!(
            coefficients().check_structure(usize::MAX, 2),
            Err(ZkeyError::Malformed { section: 4, .. })
        ));
    }
}

#[cfg(test)]
mod tests;
