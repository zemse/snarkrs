//! The tests that matter here are the ones that can tell a Montgomery mix-up apart from
//! a correct parse. Length and shape assertions cannot: every wrong-encoding bug produces
//! a perfectly well-shaped key. So the artifact tests cross the binary reader against a
//! second, independent source of the same numbers (snarkjs' decimal JSON) and, where a
//! proof is available, against the pairing equation itself.

use super::*;
use ark_ec::pairing::{Pairing, PairingOutput};
use std::path::{Path, PathBuf};

/// Montgomery constants for BN254, as snarkjs writes them. Independently derived: these
/// are `2^256 mod q` and `(2^256)^2 mod r`, and the second one is byte for byte what a
/// real `circuit.zkey` stores for a section-4 coefficient of 1.
const R_MOD_Q: &str =
    "6350874878119819312338956282401532409788428879151445726012394534686998597021";
const R2_MOD_R: &str =
    "944936681149208446651664254269745548490766851729442924617792859073125903783";

fn le32(decimal: &str) -> [u8; 32] {
    let n: num_bigint::BigUint = decimal.parse().unwrap();
    let mut out = [0u8; 32];
    let b = n.to_bytes_le();
    out[..b.len()].copy_from_slice(&b);
    out
}

#[test]
fn r_constants_are_the_arkworks_montgomery_radix() {
    // `r_inv()` claims to be the element whose value is R^-1, so its inverse is R and
    // R^2 is the integer a real zkey stores for a section-4 coefficient of 1.
    let r = r_inv().inverse().unwrap();
    assert_eq!((r * r).into_bigint().to_bytes_le(), le32(R2_MOD_R).to_vec());
    // Same identity in the base field, where the radix is taken mod q instead of r.
    let rq = Fq::new_unchecked(ark_ff::BigInt::new([1, 0, 0, 0]))
        .inverse()
        .unwrap();
    assert_eq!(rq.into_bigint().to_bytes_le(), le32(R_MOD_Q).to_vec());
}

#[test]
fn fq_reads_snarkjs_montgomery_limbs() {
    // Stored limbs of `R mod q` are the Montgomery representation of exactly 1.
    assert_eq!(binfile::fq(&le32(R_MOD_Q)).unwrap(), Fq::ONE);
    // And the naive reading, the one that silently breaks provers, is not 1.
    assert_ne!(Fq::from_le_bytes_mod_order(&le32(R_MOD_Q)), Fq::ONE);
}

#[test]
fn section4_values_are_double_montgomery() {
    let inv = r_inv();
    assert_eq!(
        binfile::fr_double_montgomery(&le32(R2_MOD_R), &inv).unwrap(),
        Fr::ONE
    );
    // Single-Montgomery reading, the mistake this decoder exists to avoid, gives R.
    assert_eq!(
        Fr::new_unchecked(binfile::bigint(&le32(R2_MOD_R))),
        inv.inverse().unwrap()
    );
}

#[test]
fn zero_coordinates_decode_to_the_identity() {
    assert_eq!(
        binfile::g1(&[0u8; binfile::G1_BYTES]).unwrap(),
        G1Affine::identity()
    );
    assert_eq!(
        binfile::g2(&[0u8; binfile::G2_BYTES]).unwrap(),
        G2Affine::identity()
    );
    // The literal affine pair (0, 0) is not on the curve, so anything that forwarded it
    // unchanged would be caught here.
    assert!(!G1Affine::new_unchecked(Fq::zero(), Fq::zero()).is_on_curve());
}

/// arkworks keeps the stored limbs of a field element below the modulus, and
/// `new_unchecked` is the one door into an element that does not honour it. One
/// `add_assign` on limbs in `[q, 2^256)` drops the carry out of 256 bits and lands on a
/// value that is not even congruent mod `q`, so `is_on_curve` - the check meant to refuse
/// the point - is computed with broken arithmetic. The range test therefore has to happen
/// at the byte boundary, before anything is constructed.
#[test]
fn limbs_at_or_above_the_modulus_are_refused() {
    let q = Fq::MODULUS.to_bytes_le();
    let r = Fr::MODULUS.to_bytes_le();
    assert!(matches!(
        binfile::fq(&q),
        Err(ZkeyError::NonCanonical("base field"))
    ));
    assert!(binfile::fq(&[0xff; FQ_BYTES]).is_err());
    assert!(matches!(
        binfile::fr_double_montgomery(&r, &r_inv()),
        Err(ZkeyError::NonCanonical("scalar field"))
    ));

    // And through the point decoders, in a coordinate other than the first, so a check
    // applied to only one of them would still show up here.
    let mut p1 = [0u8; G1_BYTES];
    p1[FQ_BYTES..].copy_from_slice(&q);
    assert!(binfile::g1(&p1).is_err());
    let mut p2 = [0u8; G2_BYTES];
    p2[3 * FQ_BYTES..].copy_from_slice(&q);
    assert!(binfile::g2(&p2).is_err());

    // `q - 1` is the largest coordinate a real file can carry, and it must still decode.
    let mut just_below = q.clone();
    just_below[0] -= 1;
    assert!(binfile::fq(&just_below).is_ok());
}

/// `n * stride` used to be a plain multiply. `n` comes straight out of the header, so on a
/// 32-bit `usize` - and this crate is built for wasm32 - a record count of 2^30 against the
/// 44-byte coefficient stride wraps to zero, and the one length gate every section goes
/// through then accepts an empty payload. The check is target-independent, so this is.
#[test]
fn a_record_count_that_overflows_usize_is_refused() {
    let n = usize::MAX / COEF_RECORD + 1;
    assert!(matches!(
        binfile::expect_records(&[], n, COEF_RECORD, 4),
        Err(ZkeyError::Malformed { section: 4, .. })
    ));
}

/// The parallel decode writes a placeholder for a bad record rather than stopping, so
/// what it has to get right is which error comes back: the lowest bad record's, however
/// the work was split.
#[test]
fn decode_records_reports_the_lowest_bad_record() {
    let n = 1 << 16;
    let mut data = vec![0u8; n * FR_BYTES];
    for i in 0..n {
        data[i * FR_BYTES..][..8].copy_from_slice(&(i as u64).to_le_bytes());
    }
    let decode = |_: usize, b: &[u8]| binfile::fr_normal(b, 2);
    let got = decode_records(&data, FR_BYTES, Fr::zero(), decode).unwrap();
    assert_eq!(got, (0..n as u64).map(Fr::from).collect::<Vec<_>>());

    for bad in [n - 1, 40_000, 7] {
        data[bad * FR_BYTES..][..FR_BYTES].fill(0xff);
    }
    let err = decode_records(&data, FR_BYTES, Fr::zero(), |i, b| {
        binfile::fr_normal(b, 2).map_err(|_| ZkeyError::Malformed {
            section: 2,
            reason: format!("record {i}"),
        })
    })
    .unwrap_err();
    assert!(err.to_string().ends_with("record 7"), "{err}");
}

/// Encodes `v` the way snarkjs writes a section-4 coefficient: the limbs of `v * R^2`.
fn encode_coef(v: u64) -> [u8; 32] {
    let r = r_inv().inverse().unwrap();
    let stored = (Fr::from(v) * r).0.to_bytes_le();
    let mut out = [0u8; 32];
    out[..stored.len().min(32)].copy_from_slice(&stored[..stored.len().min(32)]);
    out
}

fn synthetic_section4(records: &[(u32, u32, u32, u64)]) -> Vec<u8> {
    let mut buf = (records.len() as u32).to_le_bytes().to_vec();
    for &(m, c, s, v) in records {
        buf.extend_from_slice(&m.to_le_bytes());
        buf.extend_from_slice(&c.to_le_bytes());
        buf.extend_from_slice(&s.to_le_bytes());
        buf.extend_from_slice(&encode_coef(v));
    }
    buf
}

#[test]
fn csr_is_stable_and_reproduces_the_records() {
    // Deliberately out of order in both matrix and constraint, with two records sharing
    // a row so the stability of the counting sort is observable.
    let records = [
        (1u32, 3u32, 2u32, 7u64),
        (0, 3, 5, 11),
        (0, 0, 1, 13),
        (0, 3, 4, 17),
        (1, 0, 0, 19),
        (0, 7, 6, 23),
    ];
    let coeffs = read_coefficients(&synthetic_section4(&records), 8, 8).unwrap();

    assert_row_ptr(&coeffs, 8, &[4, 2]);

    // Row 3 of matrix A keeps the file order of its two records: signal 5 before 4.
    let a = &coeffs.row_ptr[0];
    let row3 = a[3] as usize..a[4] as usize;
    assert_eq!(&coeffs.signal[0][row3.clone()], &[5, 4]);
    assert_eq!(&coeffs.value[0][row3], &[Fr::from(11u64), Fr::from(17u64)]);

    let mut want: Vec<_> = records
        .iter()
        .map(|&(m, c, s, v)| (m, c, s, Fr::from(v)))
        .collect();
    want.sort_by_key(|&(m, c, s, _)| (m, c, s));
    let mut got = flatten(&coeffs);
    got.sort_by_key(|&(m, c, s, _)| (m, c, s));
    assert_eq!(got, want);
}

#[test]
fn csr_rejects_records_outside_the_domain_or_witness() {
    assert!(read_coefficients(&synthetic_section4(&[(0, 9, 1, 3)]), 8, 8).is_err());
    assert!(read_coefficients(&synthetic_section4(&[(0, 1, 9, 3)]), 8, 8).is_err());
    assert!(read_coefficients(&synthetic_section4(&[(2, 1, 1, 3)]), 8, 8).is_err());
}

/// Rebuilds the flat `(matrix, constraint, signal, value)` list from CSR.
fn flatten(c: &Coefficients) -> Vec<(u32, u32, u32, Fr)> {
    let mut out = Vec::new();
    for m in 0..2usize {
        for row in 0..c.row_ptr[m].len() - 1 {
            for k in c.row_ptr[m][row] as usize..c.row_ptr[m][row + 1] as usize {
                out.push((m as u32, row as u32, c.signal[m][k], c.value[m][k]));
            }
        }
    }
    out
}

fn assert_row_ptr(c: &Coefficients, domain_size: usize, totals: &[usize; 2]) {
    for m in 0..2usize {
        assert_eq!(c.row_ptr[m].len(), domain_size + 1, "matrix {m}");
        assert_eq!(c.row_ptr[m][0], 0, "matrix {m}");
        for w in c.row_ptr[m].windows(2) {
            assert!(w[0] <= w[1], "matrix {m} row_ptr is not monotonic");
        }
        assert_eq!(
            c.row_ptr[m][domain_size] as usize, totals[m],
            "matrix {m} row_ptr tail"
        );
        assert_eq!(c.signal[m].len(), totals[m], "matrix {m}");
        assert_eq!(c.value[m].len(), totals[m], "matrix {m}");
    }
}

// Artifact-backed tests. `bench/scripts/gen-artifacts.sh` produces these; when the
// directory is empty every test below reports a skip instead of passing silently.

struct Artifact {
    name: String,
    dir: PathBuf,
}

fn artifacts() -> Vec<Artifact> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize();
    let Ok(root) = root else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<Artifact> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .map(|dir| Artifact {
            name: dir.file_name().unwrap().to_string_lossy().into_owned(),
            dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Runs `f` over every artifact, and makes an empty artifact directory loud rather than
/// a silent pass. A test that quietly succeeds on zero inputs is worse than no test.
fn for_each_artifact(test: &str, f: impl Fn(&Artifact)) {
    let found = artifacts();
    if found.is_empty() {
        eprintln!(
            "SKIPPED {test}: no artifacts under bench/artifacts (run bench/scripts/gen-artifacts.sh)"
        );
        return;
    }
    for a in &found {
        eprintln!("{test}: {}", a.name);
        f(a);
    }
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn decimals(v: &serde_json::Value) -> Vec<Fr> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let n: num_bigint::BigUint = s.as_str().unwrap().parse().unwrap();
            Fr::from_le_bytes_mod_order(&n.to_bytes_le())
        })
        .collect()
}

/// THE Montgomery test. The zkey path reads binary Montgomery limbs; the JSON path reads
/// decimal ordinary form. Nothing but a correct Montgomery conversion makes them agree.
#[test]
fn zkey_montgomery_agrees_with_the_decimal_vkey_json() {
    for_each_artifact("zkey_vs_vkey_json", |a| {
        let pk = ProvingKey::load(&a.dir.join("circuit.zkey")).unwrap();
        let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();

        assert_eq!(pk.vk.alpha_g1, vk.alpha_g1, "{}: alpha_g1", a.name);
        assert_eq!(pk.vk.beta_g2, vk.beta_g2, "{}: beta_g2", a.name);
        assert_eq!(pk.vk.gamma_g2, vk.gamma_g2, "{}: gamma_g2", a.name);
        assert_eq!(pk.vk.delta_g2, vk.delta_g2, "{}: delta_g2", a.name);
        assert_eq!(pk.vk.ic.len(), pk.n_public + 1, "{}: ic length", a.name);
        assert_eq!(vk.ic.len(), pk.vk.ic.len(), "{}: ic length", a.name);
        for (i, (from_zkey, from_json)) in pk.vk.ic.iter().zip(&vk.ic).enumerate() {
            assert_eq!(from_zkey, from_json, "{}: ic[{i}]", a.name);
        }
    });
}

#[test]
fn every_parsed_point_is_on_curve_and_in_the_prime_subgroup() {
    for_each_artifact("point_validity", |a| {
        let pk = ProvingKey::load(&a.dir.join("circuit.zkey")).unwrap();

        let g1_sections: [(&str, &Vec<G1Affine>); 5] = [
            ("ic", &pk.vk.ic),
            ("a_query", &pk.a_query),
            ("b_g1_query", &pk.b_g1_query),
            ("l_query", &pk.l_query),
            ("h_query", &pk.h_query),
        ];
        for (what, points) in g1_sections {
            for (i, p) in points.iter().enumerate() {
                assert!(p.is_on_curve(), "{}: {what}[{i}] off curve", a.name);
                assert!(
                    p.is_in_correct_subgroup_assuming_on_curve(),
                    "{}: {what}[{i}] outside the prime subgroup",
                    a.name
                );
            }
        }
        for (i, p) in pk.b_g2_query.iter().enumerate() {
            assert!(p.is_on_curve(), "{}: b_g2_query[{i}] off curve", a.name);
            assert!(
                p.is_in_correct_subgroup_assuming_on_curve(),
                "{}: b_g2_query[{i}] outside the prime subgroup",
                a.name
            );
        }
        for (what, p) in [
            ("alpha_g1", pk.alpha_g1),
            ("beta_g1", pk.beta_g1),
            ("delta_g1", pk.delta_g1),
        ] {
            assert!(
                p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve(),
                "{}: {what}",
                a.name
            );
        }
        for (what, p) in [
            ("beta_g2", pk.beta_g2),
            ("delta_g2", pk.delta_g2),
            ("gamma_g2", pk.vk.gamma_g2),
        ] {
            assert!(
                p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve(),
                "{}: {what}",
                a.name
            );
        }
    });
}

#[test]
fn section_lengths_follow_the_header() {
    for_each_artifact("section_lengths", |a| {
        let pk = ProvingKey::load(&a.dir.join("circuit.zkey")).unwrap();
        assert_eq!(pk.a_query.len(), pk.n_vars, "{}", a.name);
        assert_eq!(pk.b_g1_query.len(), pk.n_vars, "{}", a.name);
        assert_eq!(pk.b_g2_query.len(), pk.n_vars, "{}", a.name);
        assert_eq!(pk.l_query.len(), pk.n_vars - pk.n_public - 1, "{}", a.name);
        assert_eq!(pk.h_query.len(), pk.domain_size, "{}", a.name);
        assert!(pk.domain_size.is_power_of_two(), "{}", a.name);
    });
}

/// The witness path has its own encoding question, and `public.json` answers it: those
/// decimals are what snarkjs itself pulled out of the same bytes and what the reference
/// proof was verified against.
#[test]
fn witness_starts_with_one_and_matches_public_json() {
    for_each_artifact("witness_vs_public_json", |a| {
        let pk = ProvingKey::load(&a.dir.join("circuit.zkey")).unwrap();
        let w = wtns::Witness::load(&a.dir.join("circuit.wtns")).unwrap();

        assert_eq!(w.0.len(), pk.n_vars, "{}: witness length", a.name);
        assert_eq!(w.0[0], Fr::ONE, "{}: witness[0]", a.name);

        let public = decimals(&json(&a.dir.join("public.json")));
        assert_eq!(public.len(), pk.n_public, "{}: public length", a.name);
        for (i, p) in public.iter().enumerate() {
            assert_eq!(&w.0[i + 1], p, "{}: public signal {i}", a.name);
        }
    });
}

#[test]
fn coefficients_csr_reproduces_the_raw_section() {
    for_each_artifact("coefficients_csr", |a| {
        let pk = ProvingKey::load(&a.dir.join("circuit.zkey")).unwrap();

        // Re-read section 4 independently of the CSR builder, so the comparison is
        // against the file rather than against the same code path twice.
        let file = BinFile::open(&a.dir.join("circuit.zkey"), b"zkey", 2).unwrap();
        let raw = file.unique_section(4).unwrap();
        let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        let inv = r_inv();
        let mut want: Vec<(u32, u32, u32, Fr)> = (0..n)
            .map(|i| {
                let b = 4 + i * COEF_RECORD;
                let rd = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
                (
                    rd(b),
                    rd(b + 4),
                    rd(b + 8),
                    binfile::fr_double_montgomery(&raw[b + 12..b + 44], &inv).unwrap(),
                )
            })
            .collect();

        let totals = [
            pk.coeffs.row_ptr[0][pk.domain_size] as usize,
            pk.coeffs.row_ptr[1][pk.domain_size] as usize,
        ];
        assert_eq!(totals[0] + totals[1], n, "{}: record count", a.name);
        assert_row_ptr(&pk.coeffs, pk.domain_size, &totals);

        let mut got = flatten(&pk.coeffs);
        // Sorting both sides compares the multisets, which is the invariant a permuting
        // sort must preserve. Row order within a row is asserted separately, on the
        // synthetic case where the input order is known.
        let key = |t: &(u32, u32, u32, Fr)| (t.0, t.1, t.2, t.3.into_bigint().0);
        want.sort_by_key(key);
        got.sort_by_key(key);
        assert_eq!(got, want, "{}: CSR does not reproduce section 4", a.name);

        // Every signal index must address the witness, or the gather reads out of bounds.
        for m in 0..2 {
            for &s in &pk.coeffs.signal[m] {
                assert!((s as usize) < pk.n_vars, "{}: signal {s}", a.name);
            }
        }
    });
}

/// End-to-end proof that the verifying key was parsed correctly: snarkjs' own reference
/// proof must satisfy the pairing equation under our parsed key and our parsed witness.
/// A single wrong coordinate anywhere in `vk` breaks this.
#[test]
fn snarkjs_reference_proof_verifies_under_the_parsed_key() {
    for_each_artifact("reference_proof", |a| {
        let proof_path = a.dir.join("proof.json");
        if !proof_path.is_file() {
            eprintln!("  no proof.json for {}, skipping", a.name);
            return;
        }
        let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
        let p = json(&proof_path);
        let pi_a = json_g1(&p["pi_a"], "pi_a").unwrap();
        let pi_b = json_g2(&p["pi_b"], "pi_b").unwrap();
        let pi_c = json_g1(&p["pi_c"], "pi_c").unwrap();
        let public = decimals(&json(&a.dir.join("public.json")));

        let mut vk_x = vk.ic[0].into_group();
        for (i, s) in public.iter().enumerate() {
            vk_x += vk.ic[i + 1] * s;
        }

        let out = Bn254::multi_pairing(
            [(-pi_a).into(), vk.alpha_g1, vk_x.into_affine(), pi_c],
            [pi_b, vk.beta_g2, vk.gamma_g2, vk.delta_g2],
        );
        assert_eq!(out, PairingOutput::zero(), "{}: pairing check", a.name);
    });
}

// Regression tests for the two key-handling findings from the audit. Both were
// confirmed by running the attack before the fix landed, so both are pinned here.

/// Byte offsets inside section 2, derived from the file rather than hardcoded.
///
/// Layout, in order: `n8q`, `q`, `n8r`, `r`, `nVars`, `nPublic`, `domainSize`, then
/// `alpha1` (G1), `beta1` (G1), `beta2` (G2), `gamma2` (G2), `delta1` (G1), `delta2` (G2).
struct Header2 {
    n_public_at: usize,
    domain_size_at: usize,
    beta_g1_at: usize,
    delta_g1_at: usize,
}

fn locate(bytes: &[u8]) -> Header2 {
    assert_eq!(&bytes[..4], b"zkey");
    let nsec = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut off = 12usize;
    let mut s2 = None;
    for _ in 0..nsec {
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let slen = u64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap()) as usize;
        off += 12;
        if sid == 2 && s2.is_none() {
            s2 = Some(off);
        }
        off += slen;
    }
    let s2 = s2.expect("section 2");
    let n8q = u32::from_le_bytes(bytes[s2..s2 + 4].try_into().unwrap()) as usize;
    let p = s2 + 4 + n8q;
    let n8r = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    let p = p + 4 + n8r + 8; // skip nVars, nPublic
    let alpha1 = p + 4;
    Header2 {
        n_public_at: p - 4,
        domain_size_at: p,
        beta_g1_at: alpha1 + 64,
        delta_g1_at: alpha1 + 64 + 64 + 128 + 128,
    }
}

fn load_mutated(
    name: &str,
    base: &std::path::Path,
    edit: impl Fn(&mut Vec<u8>),
) -> Result<(), ZkeyError> {
    let mut bytes = std::fs::read(base.join("circuit.zkey")).unwrap();
    edit(&mut bytes);
    let path = std::env::temp_dir().join(format!(
        "g16-zkey-regression-{name}-{}.zkey",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let r = ProvingKey::load(&path).map(|_| ());
    let _ = std::fs::remove_file(&path);
    r
}

/// The critical finding: zeroing `beta_g1` or `delta_g1` switches zero knowledge off while
/// proofs still verify against the genuine vkey, because neither quantity appears in it.
///
/// The identity passes `is_on_curve` and `is_in_correct_subgroup_assuming_on_curve` in
/// arkworks, so only an explicit infinity test catches this.
#[test]
fn a_zeroed_toxic_waste_point_is_rejected() {
    for_each_artifact("a_zeroed_toxic_waste_point_is_rejected", |a| {
        for (what, pick) in [
            (
                "beta_g1",
                (|h: &Header2| h.beta_g1_at) as fn(&Header2) -> usize,
            ),
            ("delta_g1", |h: &Header2| h.delta_g1_at),
        ] {
            let err = load_mutated(what, &a.dir, |b| {
                let at = pick(&locate(b));
                b[at..at + 64].fill(0);
            })
            .expect_err(&format!(
                "{}: a zkey with {what} zeroed must be rejected; accepting it means this \
                 key silently provides no zero knowledge",
                a.name
            ));
            let msg = err.to_string();
            assert!(msg.contains(what), "{}: unhelpful error: {msg}", a.name);
            assert!(
                msg.contains("infinity"),
                "{}: unhelpful error: {msg}",
                a.name
            );
        }
    });
}

/// The same `nPublic + 1` the JSON path above guards, out of a file that is just as
/// attacker-controlled. On a 32-bit `usize` the add wraps: section 3 is read as an empty
/// IC, the `n_vars < n_ic` guard becomes `n_vars < 0` and never fires, and `n_vars - n_ic`
/// wraps back to `n_vars`, which is the empty-`ic` key the JSON regression exists to
/// prevent. On 64-bit the u32 cannot overflow the add and the guard is what refuses it.
/// Either way the file must not load, so the test does not care which one fired.
#[test]
fn a_zkey_claiming_an_impossible_npublic_is_refused() {
    for_each_artifact("a_zkey_claiming_an_impossible_npublic_is_refused", |a| {
        let err = load_mutated("npublic", &a.dir, |b| {
            let at = locate(b).n_public_at;
            b[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        })
        .expect_err(&format!(
            "{}: nPublic = u32::MAX must be refused, not parsed into an empty ic",
            a.name
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("n_public"),
            "{}: unhelpful error: {msg}",
            a.name
        );
    });
}

/// The range check at the byte boundary, reached through the full parse rather than the
/// decoder alone. `beta_g1`'s x set to `q` is still 256 bits, so nothing but an explicit
/// compare refuses it, and `check_g1` cannot: it would run `is_on_curve` on the very
/// element whose limbs break the arithmetic it is computed with.
#[test]
fn a_zkey_coordinate_at_the_modulus_is_refused() {
    for_each_artifact("a_zkey_coordinate_at_the_modulus_is_refused", |a| {
        let err = load_mutated("noncanonical", &a.dir, |b| {
            let at = locate(b).beta_g1_at;
            b[at..at + FQ_BYTES].copy_from_slice(&Fq::MODULUS.to_bytes_le());
        })
        .expect_err(&format!(
            "{}: a coordinate at q must be refused, not installed as Montgomery limbs",
            a.name
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("not below the modulus"),
            "{}: unhelpful error: {msg}",
            a.name
        );
    });
}

/// The unmutated key must still load. A gate that rejects real keys is worse than no gate,
/// and real snarkjs keys genuinely do contain points at infinity inside the query sections,
/// which is why the query check is ALL and not ANY.
#[test]
fn the_genuine_key_still_loads_after_the_infinity_gates() {
    for_each_artifact(
        "the_genuine_key_still_loads_after_the_infinity_gates",
        |a| {
            ProvingKey::load(&a.dir.join("circuit.zkey"))
                .unwrap_or_else(|e| panic!("{}: genuine key rejected: {e}", a.name));
        },
    );
}

/// The allocation bomb. A four kilobyte file claiming `domainSize = 2^31` used to cost
/// 34 GB and 11.5 seconds before the section 9 length check refuted it, because that check
/// ran after the allocations sized from the header. Both gates now run first, so this is a
/// parse error with nothing allocated.
#[test]
fn a_lying_domain_size_is_refused_before_anything_is_allocated() {
    for_each_artifact(
        "a_lying_domain_size_is_refused_before_anything_is_allocated",
        |a| {
            for claim in [1u32 << 31, 1 << 29, 1 << 24] {
                let t = std::time::Instant::now();
                let err = load_mutated("bomb", &a.dir, |b| {
                    let at = locate(b).domain_size_at;
                    b[at..at + 4].copy_from_slice(&claim.to_le_bytes());
                })
                .expect_err(&format!("{}: domainSize {claim} must be rejected", a.name));
                // The point is not only that it errors but that it errors cheaply. Ten seconds
                // is a very loose bound; the real figure is milliseconds. A regression that
                // reintroduces the allocation shows up as tens of seconds and tens of GB.
                assert!(
                    t.elapsed().as_secs() < 10,
                    "{}: rejecting domainSize {claim} took {:?}, which means something was \
                 allocated from the header before it was validated",
                    a.name,
                    t.elapsed()
                );
                let msg = err.to_string();
                assert!(
                    msg.contains("two-adicity")
                        || msg.contains("section 9")
                        || msg.contains("records"),
                    "{}: expected a two-adicity or section-length error, got: {msg}",
                    a.name
                );
            }
        },
    );
}

/// `nPublic + 1` used to be an unchecked add on a `usize` that came straight from an
/// attacker-supplied `u64`. In release, where overflow checks are off, `usize::MAX + 1`
/// wraps to 0, so `"nPublic": 18446744073709551615` paired with `"IC": []` satisfied the
/// length guard and produced a key with an empty `ic`. `aggregate_public` then indexed
/// `vk.ic[0]` and the process died on a bounds check, which the release profile turns into
/// a SIGABRT because it sets `panic = "abort"`.
///
/// Two hundred bytes of JSON, one dead verifier. Same family as gnark-crypto#355 and #730.
///
/// Built by mutating a genuine vkey so every other field parses and the test reaches the
/// guard it is actually about.
#[test]
fn a_vkey_claiming_an_impossible_npublic_is_refused() {
    for_each_artifact("a_vkey_claiming_an_impossible_npublic_is_refused", |a| {
        let genuine = a.dir.join("vkey.json");
        if !genuine.exists() {
            eprintln!("  {}: no vkey.json, skipped", a.name);
            return;
        }
        // nPublic = u64::MAX is the wrap; nPublic = 0 against an empty IC is the same guard
        // approached from the other side, and must be refused because IC always carries the
        // constant-wire point even when there are no public inputs.
        for (label, n_public) in [("u64::MAX", u64::MAX), ("zero", 0u64)] {
            let mut v = json(&genuine);
            v["nPublic"] = serde_json::json!(n_public);
            v["IC"] = serde_json::json!([]);

            let dir = std::env::temp_dir().join(format!(
                "g16-vkey-{}-{label}-{}",
                a.name,
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("vkey.json");
            std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();

            let got = VerifyingKey::from_json(&path);
            let _ = std::fs::remove_dir_all(&dir);

            let err = match got {
                Err(e) => e,
                Ok(_) => panic!(
                    "{}: nPublic = {label} with an empty IC must be refused, not accepted",
                    a.name
                ),
            };
            let msg = err.to_string();
            assert!(
                msg.contains("nPublic") || msg.contains("IC"),
                "{}: expected the error to name nPublic or IC, got: {msg}",
                a.name
            );
        }
    });
}

/// The mmap path and the byte path must produce the same key.
///
/// This is not a formality. The two constructors used to be one function that read
/// `self.map`; they are now two entry points into a shared `index`/`parse` pair over an
/// `enum Backing`, and the failure mode of getting that wrong is a browser build that
/// parses a zkey slightly differently from the native build and produces a proof nobody
/// can verify. Comparing field for field, rather than proving with both, is what makes a
/// single wrong section offset visible as itself.
///
/// `l_query` is compared too even though it is derived from `n_vars - n_public - 1`,
/// because that arithmetic is exactly the kind that a divergent header parse breaks.
#[test]
fn the_byte_path_and_the_mmap_path_parse_identically() {
    for_each_artifact("byte_path_vs_mmap_path", |a| {
        let path = a.dir.join("circuit.zkey");
        let mapped = ProvingKey::load(&path).unwrap();
        let owned = ProvingKey::from_bytes(std::fs::read(&path).unwrap()).unwrap();

        let n = &a.name;
        assert_eq!(mapped.n_vars, owned.n_vars, "{n}: n_vars");
        assert_eq!(mapped.n_public, owned.n_public, "{n}: n_public");
        assert_eq!(mapped.domain_size, owned.domain_size, "{n}: domain_size");
        assert_eq!(mapped.alpha_g1, owned.alpha_g1, "{n}: alpha_g1");
        assert_eq!(mapped.beta_g1, owned.beta_g1, "{n}: beta_g1");
        assert_eq!(mapped.beta_g2, owned.beta_g2, "{n}: beta_g2");
        assert_eq!(mapped.delta_g1, owned.delta_g1, "{n}: delta_g1");
        assert_eq!(mapped.delta_g2, owned.delta_g2, "{n}: delta_g2");
        assert_eq!(mapped.a_query, owned.a_query, "{n}: a_query");
        assert_eq!(mapped.b_g1_query, owned.b_g1_query, "{n}: b_g1_query");
        assert_eq!(mapped.b_g2_query, owned.b_g2_query, "{n}: b_g2_query");
        assert_eq!(mapped.l_query, owned.l_query, "{n}: l_query");
        assert_eq!(mapped.h_query, owned.h_query, "{n}: h_query");
        for m in 0..2 {
            assert_eq!(
                mapped.coeffs.row_ptr[m], owned.coeffs.row_ptr[m],
                "{n}: coeffs.row_ptr[{m}]"
            );
            assert_eq!(
                mapped.coeffs.signal[m], owned.coeffs.signal[m],
                "{n}: coeffs.signal[{m}]"
            );
            assert_eq!(
                mapped.coeffs.value[m], owned.coeffs.value[m],
                "{n}: coeffs.value[{m}]"
            );
        }
        assert_eq!(mapped.vk.alpha_g1, owned.vk.alpha_g1, "{n}: vk.alpha_g1");
        assert_eq!(mapped.vk.beta_g2, owned.vk.beta_g2, "{n}: vk.beta_g2");
        assert_eq!(mapped.vk.gamma_g2, owned.vk.gamma_g2, "{n}: vk.gamma_g2");
        assert_eq!(mapped.vk.delta_g2, owned.vk.delta_g2, "{n}: vk.delta_g2");
        assert_eq!(mapped.vk.ic, owned.vk.ic, "{n}: vk.ic");

        // The witness reader shares the same container, so it shares the same risk.
        let wtns = a.dir.join("circuit.wtns");
        let w_mapped = wtns::Witness::load(&wtns).unwrap();
        let w_owned = wtns::Witness::from_bytes(std::fs::read(&wtns).unwrap()).unwrap();
        assert_eq!(w_mapped.0, w_owned.0, "{n}: witness");
    });
}

/// Both constructors must reject the same malformed containers. They share one `index`
/// today; this is what notices if someone later inlines the checks back into `open` and
/// leaves the byte path, which is the one the browser uses, without them.
#[test]
fn the_byte_path_rejects_what_the_mmap_path_rejects() {
    // Shorter than the 12-byte magic + version + nSections header. These four bytes are
    // the right magic, so the error must not claim otherwise.
    assert!(matches!(
        ProvingKey::from_bytes(b"zkey".to_vec()),
        Err(ZkeyError::TooShort(4))
    ));
    // Right length, wrong magic.
    assert!(matches!(
        ProvingKey::from_bytes(b"wtns\x02\0\0\0\0\0\0\0".to_vec()),
        Err(ZkeyError::BadMagic(m)) if &m == b"wtns"
    ));
    // Version 3 is past what this reader claims to understand.
    assert!(matches!(
        ProvingKey::from_bytes(b"zkey\x03\0\0\0\0\0\0\0".to_vec()),
        Err(ZkeyError::Malformed { section: 0, .. })
    ));
    // One section header promised, zero bytes of it present.
    assert!(matches!(
        ProvingKey::from_bytes(b"zkey\x02\0\0\0\x01\0\0\0".to_vec()),
        Err(ZkeyError::Malformed { section: 0, .. })
    ));
    // A section claiming 2^40 bytes inside a 24-byte file, which is the check that keeps
    // `unique_section`'s unchecked slicing sound.
    let mut lying = b"zkey\x02\0\0\0\x01\0\0\0\x01\0\0\0".to_vec();
    lying.extend_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(matches!(
        ProvingKey::from_bytes(lying),
        Err(ZkeyError::Malformed { section: 1, .. })
    ));
}
