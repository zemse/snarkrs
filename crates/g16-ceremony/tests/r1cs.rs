//! Read every `.r1cs` in `bench/artifacts` and check the parse against two independent
//! oracles: snarkjs' own `r1cs info` output, saved next to each circuit, and the `.zkey`
//! snarkjs produced from that same circuit.
//!
//! The zkey is the sharper of the two. Its section 4 is the A and B terms of section 2 in
//! push order (`zkey_new.js:242`, `:270`), re-encoded as double Montgomery, so replaying
//! our walk against it checks the term order, the constraint indices, the signal indices
//! and the coefficient encoding at once. A single Montgomery factor out of place shows up
//! on the first record, which is the whole point: the wrong decoder parses cleanly and
//! only fails much later, at proof verification.

use g16_ceremony::r1cs::{Matrix, R1cs, SignalIndex, TERM_BYTES};
use g16_ceremony::Groth16Header;
use g16_field::Fr;
use g16_zkey::binfile::{bigint, fr_double_montgomery, r_inv, BinFile, FR_BYTES};
use std::path::{Path, PathBuf};

/// Section 4 record: three u32 indices then one 32-byte scalar.
const COEF_RECORD: usize = 12 + FR_BYTES;

/// Every artifact carrying both a circuit and snarkjs' own count of it. The r1cs is not
/// always named `circuit.r1cs`; some artifacts name it after the template.
fn artifacts() -> Vec<(String, PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for dir in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        if !dir.join("r1cs-info.txt").is_file() {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut r1cs: Vec<PathBuf> = files
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "r1cs"))
            .collect();
        r1cs.sort();
        if let Some(path) = r1cs.into_iter().next() {
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            out.push((name, path, dir));
        }
    }
    out.sort();
    out
}

/// `r1cs info` prints through a logger that wraps every line in ANSI colour codes, so the
/// value is the digits that follow the label rather than the rest of the line.
fn info_field(text: &str, label: &str) -> u64 {
    let at = text
        .find(label)
        .unwrap_or_else(|| panic!("r1cs-info.txt has no {label:?}"));
    text[at + label.len()..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or_else(|_| panic!("no number after {label:?}"))
}

#[test]
fn header_matches_snarkjs_r1cs_info() {
    let found = artifacts();
    assert!(!found.is_empty(), "no artifact carries an r1cs");
    for (name, path, dir) in found {
        let r1cs = R1cs::open(&path).unwrap();
        let h = r1cs.header();
        let info = std::fs::read_to_string(dir.join("r1cs-info.txt")).unwrap();
        assert_eq!(h.n_vars as u64, info_field(&info, "# of Wires:"), "{name}");
        assert_eq!(
            h.n_constraints as u64,
            info_field(&info, "# of Constraints:"),
            "{name}"
        );
        assert_eq!(
            h.n_prv_inputs as u64,
            info_field(&info, "# of Private Inputs:"),
            "{name}"
        );
        assert_eq!(
            h.n_pub_inputs as u64,
            info_field(&info, "# of Public Inputs:"),
            "{name}"
        );
        assert_eq!(h.n_labels, info_field(&info, "# of Labels:"), "{name}");
        assert_eq!(
            h.n_outputs as u64,
            info_field(&info, "# of Outputs:"),
            "{name}"
        );
        eprintln!(
            "{name}: {} constraints, {} wires, {} public, domain 2^{}",
            h.n_constraints,
            h.n_vars,
            h.n_public(),
            h.cir_power()
        );
    }
}

#[test]
fn constraint_walk_and_signal_index_agree() {
    for (name, path, _) in artifacts() {
        let r1cs = R1cs::open(&path).unwrap();
        let h = *r1cs.header();
        let constraints = r1cs.constraints().unwrap();
        assert_eq!(constraints.len(), h.n_constraints as usize, "{name}");

        let nnz = r1cs.nonzeros().unwrap();
        let counted: [usize; 3] = constraints.iter().fold([0; 3], |mut acc, c| {
            acc[0] += c.a.len();
            acc[1] += c.b.len();
            acc[2] += c.c.len();
            acc
        });
        assert_eq!([nnz.a, nnz.b, nnz.c], counted, "{name}");

        // Section 2 is `nConstraints` records of three counted linear combinations, so the
        // byte length is fully determined by the term counts. If it is not, the walk found
        // its terms somewhere other than where they are.
        let expect = 12 * h.n_constraints as usize + nnz.total() * TERM_BYTES;
        assert_eq!(r1cs.constraint_bytes().unwrap().len(), expect, "{name}");

        let index = SignalIndex::build(&r1cs).unwrap();
        assert_eq!(index.n_signals(), h.n_vars as usize, "{name}");
        assert_eq!(index.total_terms(), nnz.total(), "{name}");
        // Invert the inversion: every term the index reports for a signal must be findable
        // in the constraint it names, under the same coefficient pointer.
        for s in 0..h.n_vars {
            for t in index.terms(s) {
                let c = &constraints[t.constraint as usize];
                let lc = match t.matrix {
                    Matrix::A => &c.a,
                    Matrix::B => &c.b,
                    Matrix::C => &c.c,
                };
                assert!(
                    lc.iter()
                        .any(|term| term.signal == s && term.coef_ptr == t.coef_ptr),
                    "{name}: signal {s} term at {} is not in constraint {}",
                    t.coef_ptr,
                    t.constraint
                );
            }
        }
        eprintln!(
            "{name}: nnz A {} B {} C {}, {} terms indexed by signal",
            nnz.a,
            nnz.b,
            nnz.c,
            index.total_terms()
        );
    }
}

#[test]
fn derived_sizes_match_the_zkey_snarkjs_built() {
    let mut checked = 0;
    for (name, path, dir) in artifacts() {
        let zkey = dir.join("circuit.zkey");
        if !zkey.is_file() {
            continue;
        }
        let r1cs = R1cs::open(&path).unwrap();
        let h = *r1cs.header();
        let file = BinFile::open(&zkey, b"zkey", 2).unwrap();
        let zh = Groth16Header::read(file.unique_section(2).unwrap()).unwrap();
        assert_eq!(zh.n_vars, h.n_vars, "{name}: nVars");
        assert_eq!(zh.n_public as usize, h.n_public(), "{name}: nPublic");
        assert_eq!(
            zh.domain_size as usize,
            h.domain_size(),
            "{name}: domainSize"
        );
        checked += 1;
    }
    assert!(checked > 0, "no artifact carries a circuit.zkey");
}

#[test]
fn coefficients_replay_zkey_section_4() {
    let r_inv = r_inv();
    let mut checked = 0;
    for (name, path, dir) in artifacts() {
        let zkey = dir.join("circuit.zkey");
        if !zkey.is_file() {
            continue;
        }
        let r1cs = R1cs::open(&path).unwrap();
        let h = *r1cs.header();
        let file = BinFile::open(&zkey, b"zkey", 2).unwrap();
        let s4 = file.unique_section(4).unwrap();
        let n_coefs = u32::from_le_bytes(s4[..4].try_into().unwrap()) as usize;

        // The A and B terms, plus one synthetic row per public signal and the ONE signal
        // (`zkey_new.js:290-300`). The C pass pushes nothing.
        let nnz = r1cs.nonzeros().unwrap();
        assert_eq!(
            n_coefs,
            nnz.a + nnz.b + h.n_public() + 1,
            "{name}: section 4 record count"
        );

        let record = |i: usize| -> (u32, u32, u32, Fr) {
            let base = 4 + i * COEF_RECORD;
            let rd = |off: usize| u32::from_le_bytes(s4[off..off + 4].try_into().unwrap());
            (
                rd(base),
                rd(base + 4),
                rd(base + 8),
                fr_double_montgomery(&s4[base + 12..base + 12 + FR_BYTES], &r_inv),
            )
        };

        let constraints = r1cs.constraints().unwrap();
        let mut i = 0;
        for (c, constraint) in constraints.iter().enumerate() {
            for (matrix, lc) in [(Matrix::A, &constraint.a), (Matrix::B, &constraint.b)] {
                for term in lc {
                    let (m, rc, s, v) = record(i);
                    assert_eq!(m, matrix.as_u32(), "{name}: record {i} matrix");
                    assert_eq!(rc as usize, c, "{name}: record {i} constraint");
                    assert_eq!(s, term.signal, "{name}: record {i} signal");
                    // The two files disagree on encoding and must still agree on value:
                    // plain little-endian here, `v * R^2` there.
                    assert_eq!(r1cs.coef(term.coef_ptr).unwrap(), v, "{name}: record {i}");
                    i += 1;
                }
            }
        }

        // The tail: `a = 1` on a synthetic row per public signal, so that `A_s(x)` is
        // nonzero for signals that appear in no A term of their own.
        for s in 0..=h.n_public() {
            let (m, rc, sig, v) = record(i);
            assert_eq!(m, 0, "{name}: tail {s} matrix");
            assert_eq!(
                rc as usize,
                h.n_constraints as usize + s,
                "{name}: tail {s}"
            );
            assert_eq!(sig as usize, s, "{name}: tail {s} signal");
            assert_eq!(v, Fr::from(1u64), "{name}: tail {s} coefficient");
            i += 1;
        }
        assert_eq!(i, n_coefs, "{name}: records left over");
        checked += 1;
        eprintln!("{name}: {n_coefs} section 4 records replayed from the r1cs");
    }
    assert!(checked > 0, "no artifact carries a circuit.zkey");
}

#[test]
fn label_map_has_one_entry_per_signal() {
    for (name, path, _) in artifacts() {
        let r1cs = R1cs::open(&path).unwrap();
        let map = r1cs.label_map().unwrap();
        assert_eq!(map.len(), r1cs.header().n_vars as usize, "{name}");
        // Signal 0 is the constant ONE and circom always labels it 0.
        assert_eq!(map[0], 0, "{name}: label of the ONE signal");
    }
}

#[test]
fn a_zkey_is_not_an_r1cs() {
    for (_, _, dir) in artifacts() {
        let zkey = dir.join("circuit.zkey");
        if zkey.is_file() {
            assert!(R1cs::open(&zkey).is_err());
            return;
        }
    }
}

#[test]
fn coefficients_are_plain_little_endian() {
    // A hand-built one-constraint file, so the assertion is on a value rather than on a
    // round trip: `2 * s1 * s1 = s2`, coefficient 2 written as the integer 2.
    let mut body = Vec::new();
    let mut lc = |terms: &[(u32, u64)]| {
        body.extend_from_slice(&(terms.len() as u32).to_le_bytes());
        for (signal, coef) in terms {
            body.extend_from_slice(&signal.to_le_bytes());
            let mut v = [0u8; 32];
            v[..8].copy_from_slice(&coef.to_le_bytes());
            body.extend_from_slice(&v);
        }
    };
    lc(&[(1, 2)]);
    lc(&[(1, 1)]);
    lc(&[(2, 1)]);

    let mut header = Vec::new();
    header.extend_from_slice(&32u32.to_le_bytes());
    header.extend_from_slice(&g16_ceremony::r_le());
    for v in [3u32, 1, 0, 1] {
        header.extend_from_slice(&v.to_le_bytes());
    }
    header.extend_from_slice(&3u64.to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes());
    assert_eq!(header.len(), g16_ceremony::r1cs::R1csHeader::BYTES);

    let mut bytes = Vec::from(*b"r1cs");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    for (id, body) in [(1u32, &header), (2u32, &body)] {
        bytes.extend_from_slice(&id.to_le_bytes());
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(body);
    }

    let r1cs = R1cs::from_bytes(bytes).unwrap();
    let c = &r1cs.constraints().unwrap()[0];
    assert_eq!(r1cs.coef(c.a[0].coef_ptr).unwrap(), Fr::from(2u64));
    // The same bytes read as single Montgomery would be `2 * R^-1`, which is a perfectly
    // valid field element and would go undetected until a proof failed to verify.
    assert_ne!(
        Fr::new_unchecked(bigint(r1cs.coef_bytes(c.a[0].coef_ptr).unwrap())),
        Fr::from(2u64)
    );

    // One constraint, one output and no public inputs, so nPublic is 1 and the domain is
    // floor(log2(1 + 1)) + 1 = 2.
    let h = *r1cs.header();
    assert_eq!(h.n_public(), 1);
    assert_eq!(h.cir_power(), 2);
    assert_eq!(h.domain_size(), 4);
}
