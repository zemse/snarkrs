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
pub mod wtns;

use binfile::*;
use g16_field::*;
use rayon::prelude::*;

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

/// Section 4 in CSR form: `A[c] = sum over row c of coef * w[s]`.
///
/// snarkjs stores an unordered `(matrix, constraint, signal, coef)` list, which on CPU
/// is consumed as a *scatter* under striped mutexes and on GPU cannot be consumed that
/// way at all (there is no 32-byte atomic). Sorting once at key load turns it into a
/// race-free gather that both backends share. This is a one-time cost paid in `prepare`,
/// never per proof.
pub struct Coefficients {
    /// `row_ptr[m][c]..row_ptr[m][c+1]` indexes into `signal`/`value`, for matrix m in {A, B}.
    pub row_ptr: [Vec<u32>; 2],
    pub signal: [Vec<u32>; 2],
    pub value: [Vec<Fr>; 2],
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
pub enum ZkeyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a zkey file (bad magic {0:?})")]
    BadMagic([u8; 4]),
    #[error("unsupported protocol id {0} (only groth16 = 1)")]
    UnsupportedProtocol(u32),
    #[error("unsupported curve: expected BN254")]
    UnsupportedCurve,
    #[error("missing section {0}")]
    MissingSection(u32),
    #[error("malformed section {section}: {reason}")]
    Malformed { section: u32, reason: String },
    #[error("verification key json: {0}")]
    BadJson(String),
}

/// snarkjs' groth16 protocol id in section 1.
const PROTOCOL_GROTH16: u32 = 1;

/// Section 4 record: three u32 indices then one 32-byte scalar.
const COEF_RECORD: usize = 12 + FR_BYTES;

impl ProvingKey {
    /// Parse a `.zkey` by mmap. Must not copy the point sections more than once.
    ///
    /// Native only. There is no filesystem on `wasm32-unknown-unknown`, so the browser
    /// uses [`ProvingKey::from_bytes`] instead.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::parse(BinFile::open(path, b"zkey", 2)?)
    }

    /// Parse a `.zkey` that is already in memory, for the browser, where the key arrives
    /// over `fetch` rather than from a path.
    ///
    /// By value, not `&[u8]`: at `js_16x16_d32` the key is 94.4 MB, and a borrow would
    /// force the caller to hold a second live owner for as long as the `ProvingKey`
    /// exists. In a 4 GiB wasm32 address space that doubling is worth avoiding on its own,
    /// and the caller has no use for the bytes afterwards anyway.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ZkeyError> {
        Self::parse(BinFile::from_bytes(bytes, b"zkey", 2)?)
    }

    /// Everything after the container is opened, shared by both constructors so the two
    /// paths cannot diverge on a single validation.
    fn parse(file: BinFile) -> Result<Self, ZkeyError> {
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
        let alpha_g1 = g1(s2.take(G1_BYTES)?);
        let beta_g1 = g1(s2.take(G1_BYTES)?);
        let beta_g2 = g2(s2.take(G2_BYTES)?);
        let gamma_g2 = g2(s2.take(G2_BYTES)?);
        let delta_g1 = g1(s2.take(G1_BYTES)?);
        let delta_g2 = g2(s2.take(G2_BYTES)?);
        if s2.remaining() != 0 {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!("{} trailing bytes after the groth16 header", s2.remaining()),
            });
        }

        if domain_size == 0 || !domain_size.is_power_of_two() {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!("domain size {domain_size} is not a power of two"),
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
        // 1. Two-adicity. The NTT needs a 2^log_size-th root of unity, and Fr has one only
        //    up to 2^TWO_ADICITY (28 for BN254). A larger domain cannot be transformed at
        //    all, so accepting it buys nothing and costs an allocation.
        let log_size = domain_size.trailing_zeros();
        if log_size > Fr::TWO_ADICITY {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!(
                    "domain size 2^{log_size} exceeds the two-adicity of Fr (2^{}), so no \
                     root of unity of that order exists",
                    Fr::TWO_ADICITY
                ),
            });
        }
        // 2. Cross-check against the file that is actually present. Section 9 holds exactly
        //    `domain_size` G1 points, so its length refutes a lying header for the cost of
        //    a slice lookup, with nothing allocated yet.
        expect_records(file.unique_section(9)?, domain_size, G1_BYTES, 9)?;
        // The L query covers the private wires only, so an n_vars that does not leave
        // room for `1 + n_public` would underflow the section 8 length below.
        if n_vars < n_public + 1 {
            return Err(ZkeyError::Malformed {
                section: 2,
                reason: format!("n_vars {n_vars} does not cover 1 + n_public {n_public}"),
            });
        }

        let ic = read_g1_section(&file, 3, n_public + 1)?;
        let coeffs = read_coefficients(file.unique_section(4)?, n_vars, domain_size)?;
        let a_query = read_g1_section(&file, 5, n_vars)?;
        let b_g1_query = read_g1_section(&file, 6, n_vars)?;
        let b_g2_query = read_g2_section(&file, 7, n_vars)?;
        let l_query = read_g1_section(&file, 8, n_vars - n_public - 1)?;
        let h_query = read_g1_section(&file, 9, domain_size)?;

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
            (5u32, a_query.iter().all(|p| p.infinity), a_query.len()),
            (6, b_g1_query.iter().all(|p| p.infinity), b_g1_query.len()),
            (7, b_g2_query.iter().all(|p| p.infinity), b_g2_query.len()),
            (9, h_query.iter().all(|p| p.infinity), h_query.len()),
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

        Ok(Self {
            n_vars,
            n_public,
            domain_size,
            alpha_g1,
            beta_g1,
            beta_g2,
            delta_g1,
            delta_g2,
            a_query,
            b_g1_query,
            b_g2_query,
            l_query,
            h_query,
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

fn read_g1_section(file: &BinFile, id: u32, n: usize) -> Result<Vec<G1Affine>, ZkeyError> {
    let data = file.unique_section(id)?;
    expect_records(data, n, G1_BYTES, id)?;
    // One pass over the mapped bytes straight into the output vector: the section is
    // never materialised as an intermediate buffer.
    Ok(data.par_chunks_exact(G1_BYTES).map(g1).collect())
}

fn read_g2_section(file: &BinFile, id: u32, n: usize) -> Result<Vec<G2Affine>, ZkeyError> {
    let data = file.unique_section(id)?;
    expect_records(data, n, G2_BYTES, id)?;
    Ok(data.par_chunks_exact(G2_BYTES).map(g2).collect())
}

fn check_g1(p: &G1Affine, section: u32, what: &str) -> Result<(), ZkeyError> {
    if !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve() {
        return Err(ZkeyError::Malformed {
            section,
            reason: format!("{what} is not a valid G1 point"),
        });
    }
    Ok(())
}

fn check_g2(p: &G2Affine, section: u32, what: &str) -> Result<(), ZkeyError> {
    if !p.is_on_curve() || !p.is_in_correct_subgroup_assuming_on_curve() {
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
fn read_coefficients(
    data: &[u8],
    n_vars: usize,
    domain_size: usize,
) -> Result<Coefficients, ZkeyError> {
    let mut c = Cursor::new(data, 4);
    let n_coefs = c.u32()? as usize;
    expect_records(&data[4..], n_coefs, COEF_RECORD, 4)?;

    let mut counts = [vec![0u32; domain_size + 1], vec![0u32; domain_size + 1]];
    for i in 0..n_coefs {
        let (m, constraint, _, _) = coef_indices(data, i)?;
        if constraint >= domain_size {
            return Err(ZkeyError::Malformed {
                section: 4,
                reason: format!("constraint {constraint} is outside domain size {domain_size}"),
            });
        }
        counts[m][constraint + 1] += 1;
    }

    // Prefix sum in place turns the histogram into the row offsets.
    let mut row_ptr = counts;
    for m in 0..2 {
        for i in 0..domain_size {
            row_ptr[m][i + 1] += row_ptr[m][i];
        }
    }

    let totals = [
        row_ptr[0][domain_size] as usize,
        row_ptr[1][domain_size] as usize,
    ];
    let mut signal = [vec![0u32; totals[0]], vec![0u32; totals[1]]];
    let mut value = [vec![Fr::zero(); totals[0]], vec![Fr::zero(); totals[1]]];

    let r_inv = r_inv();
    let mut cursor = [row_ptr[0].clone(), row_ptr[1].clone()];
    for i in 0..n_coefs {
        let (m, constraint, sig, off) = coef_indices(data, i)?;
        if sig >= n_vars {
            return Err(ZkeyError::Malformed {
                section: 4,
                reason: format!("signal {sig} is outside n_vars {n_vars}"),
            });
        }
        let slot = cursor[m][constraint] as usize;
        cursor[m][constraint] += 1;
        signal[m][slot] = sig as u32;
        value[m][slot] = fr_double_montgomery(&data[off..off + FR_BYTES], &r_inv);
    }

    Ok(Coefficients {
        row_ptr,
        signal,
        value,
    })
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
    /// Parse snarkjs' `verification_key.json`, so we can verify against the same key
    /// snarkjs uses without trusting our own zkey reader.
    ///
    /// Native only, because there is no filesystem in a browser. The browser path is
    /// [`VerifyingKey::from_json_str`], which this is a thin wrapper over.
    #[cfg(not(target_family = "wasm"))]
    pub fn from_json(path: &std::path::Path) -> Result<Self, ZkeyError> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    /// Same, on text the caller already has.
    ///
    /// The page in `../webgpu-trial` cross-checks snarkjs' proofs against our own verifier,
    /// and it has `vkey.json` as a string from `fetch`, not as a path.
    pub fn from_json_str(text: &str) -> Result<Self, ZkeyError> {
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
    let bytes = n.to_bytes_le();
    let bi = ark_ff::BigInt::<4>::try_from(n.clone())
        .map_err(|_| ZkeyError::BadJson(format!("{what} does not fit in 256 bits")))?;
    Fq::from_bigint(bi).ok_or_else(|| {
        ZkeyError::BadJson(format!(
            "{what} is not below the base field modulus ({} bytes)",
            bytes.len()
        ))
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
    check_g1(&p, 0, what).map_err(|e| ZkeyError::BadJson(e.to_string()))?;
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
    check_g2(&p, 0, what).map_err(|e| ZkeyError::BadJson(e.to_string()))?;
    Ok(p)
}

#[cfg(test)]
mod tests;
