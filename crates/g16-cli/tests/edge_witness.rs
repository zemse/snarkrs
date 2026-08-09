//! Adversarial edge cases around the witness vector and the `Fr` boundary values.
//!
//! Every test here is written against the outcome that would be *dangerous*, not the one
//! that is merely surprising. For a prover the dangerous outcome is a proof that verifies
//! when it should not, so the malformed-witness tests assert "reject OR fail to verify"
//! and record which of the two actually happened, rather than pinning one of them and
//! calling the other a regression.
//!
//! The three groups:
//!
//! * **Shape.** Wrong-length witnesses and a constant-one wire that is not one. These are
//!   caller mistakes, and they must surface as errors or as unverifiable proofs, never as
//!   a panic and never as a silently accepted proof.
//! * **Blinders.** `r` and `s` at 0 and at `r_modulus - 1`. These are catastrophic for
//!   zero-knowledge and completely irrelevant to soundness: stage 11's algebra has to
//!   close for every pair in `Fr x Fr`, so the degenerate pairs are the sharpest test of
//!   the assembly, because nothing masks a transcription error.
//! * **Field boundaries in the witness.** 0, 1 and `r_modulus - 1` written into wires the
//!   circuit constrains. These break the R1CS, so the proofs must *fail*. That direction
//!   matters more than it looks: it is the only test here that would catch a prover which
//!   accidentally produced universally-verifying proofs, which every positive test in the
//!   suite would happily pass.
//!
//! Everything runs against the checked-in artifacts and skips loudly when they are absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use g16_core::cpu::CpuBackend;
use g16_core::prove::{prove, prove_with_blinders};
use g16_core::verify::{verify, VerifyError};
use g16_core::{Backend, PreparedCircuit, ProveError, StageTimings};
use g16_field::{Fr, One, PrimeField, Zero};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

// ---------------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------------

fn artifacts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

struct Variant {
    name: String,
    dir: PathBuf,
}

fn variants() -> Vec<Variant> {
    let Ok(entries) = std::fs::read_dir(artifacts_root()) else {
        return Vec::new();
    };
    let mut out: Vec<Variant> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .map(|dir| Variant {
            name: dir.file_name().unwrap().to_string_lossy().into_owned(),
            dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The two smallest circuits, for the tests that need many proofs per variant. Ordering
/// is by constraint count, which for these artifacts is the same as sorting by the
/// on-disk zkey size, so this stays right if new variants are added.
fn small_variants() -> Vec<Variant> {
    let mut v = variants();
    v.sort_by_key(|x| {
        std::fs::metadata(x.dir.join("circuit.zkey"))
            .map(|m| m.len())
            .unwrap_or(u64::MAX)
    });
    v.truncate(2);
    v
}

struct Case {
    circuit: Box<dyn PreparedCircuit>,
    witness: Vec<Fr>,
    vk: VerifyingKey,
    public: Vec<Fr>,
}

fn load(dir: &Path) -> Case {
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
    let circuit = CpuBackend::new().prepare(pk).unwrap();
    let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
    let public: Vec<Fr> = serde_json::from_str::<Vec<String>>(
        &std::fs::read_to_string(dir.join("public.json")).unwrap(),
    )
    .unwrap()
    .iter()
    .map(|s| {
        let n: num_bigint::BigUint = s.parse().unwrap();
        Fr::from_le_bytes_mod_order(&n.to_bytes_le())
    })
    .collect();
    Case {
        circuit,
        witness,
        vk,
        public,
    }
}

fn each(test: &str, list: Vec<Variant>, f: impl Fn(&str, &Case)) {
    if list.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
        return;
    }
    for v in &list {
        eprintln!("{test}: {}", v.name);
        f(&v.name, &load(&v.dir));
    }
}

/// What happened when we proved with a witness that should not have been provable.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// The prover refused. Best case.
    Rejected,
    /// The prover produced a proof and the verifier rejected it. Acceptable.
    Unverifiable,
    /// The prover produced a proof that verifies. The one outcome that must never happen.
    Accepted,
}

/// Prove with fixed blinders (cheap and deterministic; the blinders are irrelevant to
/// whether a *wrong* witness yields a verifying proof) and classify the result.
fn outcome(case: &Case, witness: &[Fr]) -> Outcome {
    let mut t = StageTimings::default();
    let (r, s) = (Fr::from(3u64), Fr::from(5u64));
    match prove_with_blinders(case.circuit.as_ref(), witness, r, s, &mut t) {
        Err(_) => Outcome::Rejected,
        Ok(p) => match verify(&case.vk, &case.public, &p) {
            Ok(()) => Outcome::Accepted,
            Err(_) => Outcome::Unverifiable,
        },
    }
}

// ---------------------------------------------------------------------------------
// 1. the constant-one wire
// ---------------------------------------------------------------------------------

/// `w[0]` is the constant-one wire. Every QAP row is written against `w[0] = 1` and the
/// verifier folds it in as the bare `IC[0]` term with an implicit coefficient of one, so a
/// witness with any other value there describes a different statement than the one the
/// verifier is checking.
///
/// `Witness::load` refuses such a file (covered separately below), but the in-memory API
/// takes a plain `&[Fr]` and has no such gate, so this is the path a caller building a
/// witness itself would take. The test does not insist that the prover reject it; it
/// insists that the resulting proof cannot verify.
#[test]
fn a_constant_one_wire_that_is_not_one_never_yields_a_verifying_proof() {
    each("constant_one_wire", small_variants(), |name, case| {
        for (label, v) in [
            ("zero", Fr::zero()),
            ("two", Fr::from(2u64)),
            ("r-1", -Fr::one()),
        ] {
            let mut w = case.witness.clone();
            assert_eq!(w[0], Fr::one(), "{name}: fixture w[0] was not 1");
            w[0] = v;
            let got = outcome(case, &w);
            eprintln!("  w[0] = {label}: {got:?}");
            assert_ne!(
                got,
                Outcome::Accepted,
                "{name}: w[0] = {label} produced a VERIFYING proof - the prover \
                     accepts a witness that does not satisfy the constant-one convention"
            );
        }
    });
}

/// The file-level gate, driven through the real binary so the exit status and the message
/// are both part of the contract. A `.wtns` whose first scalar is not 1 must be refused at
/// parse time, before any proving work, and must not leave a `proof.json` behind.
#[test]
fn the_wtns_loader_rejects_a_non_one_constant_wire() {
    let found = small_variants();
    let Some(v) = found.first() else {
        eprintln!("SKIPPED the_wtns_loader_rejects_a_non_one_constant_wire: no artifacts");
        return;
    };
    let out = scratch("wtns-w0");

    let bytes = std::fs::read(v.dir.join("circuit.wtns")).unwrap();
    let start = section_payload_start(&bytes, 2).expect("wtns has a section 2");
    let mut patched = bytes.clone();
    // Little-endian 2, replacing the little-endian 1 that has to be there.
    assert_eq!(patched[start], 1, "fixture w[0] was not the integer 1");
    patched[start] = 2;
    let wtns = out.join("w0-is-two.wtns");
    std::fs::write(&wtns, &patched).unwrap();

    let proof = out.join("proof.json");
    let o = Command::new(env!("CARGO_BIN_EXE_g16"))
        .args([
            "prove",
            "--zkey",
            v.dir.join("circuit.zkey").to_str().unwrap(),
            "--witness",
            wtns.to_str().unwrap(),
            "--proof",
            proof.to_str().unwrap(),
            "--public",
            out.join("public.json").to_str().unwrap(),
            "--backend",
            "cpu",
        ])
        .output()
        .unwrap();
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        !o.status.success(),
        "prove accepted a w[0] != 1 witness: {said}"
    );
    assert!(
        said.contains("witness[0] is not 1"),
        "the error did not name the problem: {said}"
    );
    // A clean error, not a signal: a panic would abort and an unwind would still be a
    // process failure, so the status alone does not distinguish them.
    assert_eq!(o.status.code(), Some(1), "expected a clean exit(1): {said}");
    assert!(!said.contains("panicked at"), "the loader panicked: {said}");
    assert!(!proof.exists(), "a rejected witness still wrote a proof");
    std::fs::remove_dir_all(&out).ok();
}

/// Byte offset of the payload of the first section with `id`, in an iden3 binfile:
/// 4 magic bytes, u32 version, u32 nSections, then `u32 id, u64 len, payload` repeated.
fn section_payload_start(bytes: &[u8], id: u32) -> Option<usize> {
    let u32_at = |p: usize| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
    let u64_at = |p: usize| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    let n = u32_at(8) as usize;
    let mut pos = 12usize;
    for _ in 0..n {
        let sec = u32_at(pos);
        let len = u64_at(pos + 4) as usize;
        pos += 12;
        if sec == id {
            return Some(pos);
        }
        pos += len;
    }
    None
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g16-edge-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------------
// 2. wrong-length witnesses
// ---------------------------------------------------------------------------------

/// A witness of the wrong length is a caller bug and must arrive as
/// [`ProveError::WitnessLength`] carrying both numbers, not as an out-of-bounds panic
/// inside the parallel gather and not as a proof built from whatever happened to be
/// adjacent in memory.
///
/// Checked on all three entry points, because they are three separate guards: `compute_h`,
/// `msms`, and the `prove` that drives both.
#[test]
fn a_witness_of_the_wrong_length_is_a_clean_error() {
    each("wrong_length", small_variants(), |name, case| {
        let n = case.witness.len();
        assert_eq!(n, case.circuit.n_vars());

        // A valid H, so the `msms` length check is reached rather than short-circuited by
        // a mismatched H buffer.
        let mut t = StageTimings::default();
        let good_h = case.circuit.compute_h(&case.witness, &mut t).unwrap();

        let mut too_long = case.witness.clone();
        too_long.push(Fr::one());
        let mut way_too_long = case.witness.clone();
        way_too_long.extend(std::iter::repeat(Fr::zero()).take(n));

        let bad: Vec<(&str, Vec<Fr>)> = vec![
            ("empty", Vec::new()),
            ("one short", case.witness[..n - 1].to_vec()),
            ("only the constant wire", vec![Fr::one()]),
            ("one long", too_long),
            ("double length", way_too_long),
        ];

        for (label, w) in bad {
            let want = case.circuit.n_vars();
            let got_len = w.len();

            match case.circuit.compute_h(&w, &mut t) {
                Err(ProveError::WitnessLength { got, want: expect }) => {
                    assert_eq!(
                        (got, expect),
                        (got_len, want),
                        "{name}/{label}: wrong numbers"
                    );
                }
                Err(e) => panic!("{name}/{label}: compute_h gave the wrong error: {e}"),
                Ok(_) => panic!("{name}/{label}: compute_h accepted a {got_len}-entry witness"),
            }

            match case.circuit.msms(&w, &good_h, &mut t) {
                Err(ProveError::WitnessLength { got, want: expect }) => {
                    assert_eq!(
                        (got, expect),
                        (got_len, want),
                        "{name}/{label}: wrong numbers"
                    );
                }
                Err(e) => panic!("{name}/{label}: msms gave the wrong error: {e}"),
                Ok(_) => panic!("{name}/{label}: msms accepted a {got_len}-entry witness"),
            }

            let mut t2 = StageTimings::default();
            match prove_with_blinders(case.circuit.as_ref(), &w, Fr::zero(), Fr::zero(), &mut t2) {
                Err(ProveError::WitnessLength { .. }) => {}
                Err(e) => panic!("{name}/{label}: prove gave the wrong error: {e}"),
                Ok(_) => panic!("{name}/{label}: prove accepted a {got_len}-entry witness"),
            }
            eprintln!("  {label} ({got_len} vs {want}): WitnessLength");
        }
    });
}

/// Same failure through the binary, so the CLI reports it rather than unwinding. The
/// witness is truncated by one field element, which keeps the binfile header self
/// consistent for everything except the record count.
#[test]
fn the_cli_rejects_a_truncated_witness_without_panicking() {
    let found = small_variants();
    let Some(v) = found.first() else {
        eprintln!("SKIPPED the_cli_rejects_a_truncated_witness: no artifacts");
        return;
    };
    let out = scratch("short-wtns");

    let bytes = std::fs::read(v.dir.join("circuit.wtns")).unwrap();
    let mut patched = bytes.clone();
    patched.truncate(bytes.len() - 32);
    let wtns = out.join("short.wtns");
    std::fs::write(&wtns, &patched).unwrap();

    let proof = out.join("proof.json");
    let o = Command::new(env!("CARGO_BIN_EXE_g16"))
        .args([
            "prove",
            "--zkey",
            v.dir.join("circuit.zkey").to_str().unwrap(),
            "--witness",
            wtns.to_str().unwrap(),
            "--proof",
            proof.to_str().unwrap(),
            "--public",
            out.join("public.json").to_str().unwrap(),
            "--backend",
            "cpu",
        ])
        .output()
        .unwrap();
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        !o.status.success(),
        "prove accepted a truncated witness: {said}"
    );
    assert_eq!(o.status.code(), Some(1), "expected a clean exit(1): {said}");
    assert!(!said.contains("panicked at"), "the loader panicked: {said}");
    assert!(!proof.exists(), "a rejected witness still wrote a proof");
    std::fs::remove_dir_all(&out).ok();
}

// ---------------------------------------------------------------------------------
// 3. blinders at the boundaries
// ---------------------------------------------------------------------------------

/// `r` and `s` are zero-knowledge, not soundness. Stage 11 is
///
/// ```text
///   A  = [A] + alpha + r*delta
///   B  = [B] + beta  + s*delta        (G2)
///   B1 = [B] + beta  + s*delta        (G1)
///   C  = [L] + [H] + s*A + r*B1 - r*s*delta
/// ```
///
/// and the pairing equation closes for *every* `(r, s)` in `Fr x Fr` because the `r*s*delta`
/// subtraction is exactly the double-counted cross term. So the degenerate pairs are not a
/// curiosity, they are the cleanest probe of the assembly: with `r = s = 0` nothing masks a
/// wrong sign or a raw-instead-of-blinded `A` in the `s*A` term, and with `r = s = -1` the
/// cross term is `+delta` rather than `-delta`, which catches the opposite mistake.
///
/// These proofs leak the witness and must never be produced outside a test. They must,
/// however, verify.
#[test]
fn boundary_blinders_still_produce_verifying_proofs() {
    let zero = Fr::zero();
    let one = Fr::one();
    let minus_one = -Fr::one();
    // r_modulus - 1, spelled out from the modulus rather than as `-1`, to prove the two
    // are the same element and that nothing in the pipeline treats the top of the range
    // specially.
    let r_minus_1 = {
        let m: num_bigint::BigUint = Fr::MODULUS.into();
        Fr::from_le_bytes_mod_order(&(m - 1u32).to_bytes_le())
    };
    assert_eq!(r_minus_1, minus_one, "r-1 and -1 are the same element");

    let pairs: [(&str, Fr, Fr); 8] = [
        ("r=0, s=0", zero, zero),
        ("r=0, s=1", zero, one),
        ("r=1, s=0", one, zero),
        ("r=-1, s=0", minus_one, zero),
        ("r=0, s=-1", zero, minus_one),
        ("r=-1, s=-1", minus_one, minus_one),
        ("r=-1, s=1", minus_one, one),
        ("r=1, s=-1", one, minus_one),
    ];

    each("boundary_blinders", variants(), |name, case| {
        for (label, r, s) in pairs {
            let mut t = StageTimings::default();
            let p = prove_with_blinders(case.circuit.as_ref(), &case.witness, r, s, &mut t)
                .unwrap_or_else(|e| panic!("{name}/{label}: prove failed: {e}"));
            verify(&case.vk, &case.public, &p)
                .unwrap_or_else(|e| panic!("{name}/{label}: proof did not verify: {e}"));
            eprintln!("  {label}: verified");
        }

        // `r = s = 0` removes every blinder, so the proof becomes a pure function of the
        // witness. Two runs must then be bit-identical, which is what makes it usable as
        // a differential fixture against another prover.
        let mut t = StageTimings::default();
        let a =
            prove_with_blinders(case.circuit.as_ref(), &case.witness, zero, zero, &mut t).unwrap();
        let b =
            prove_with_blinders(case.circuit.as_ref(), &case.witness, zero, zero, &mut t).unwrap();
        assert_eq!(
            (a.a, a.b, a.c),
            (b.a, b.b, b.c),
            "{name}: unblinded proof is not deterministic"
        );

        // And distinct blinders must actually move the proof, otherwise stage 11 is
        // ignoring them and every zero-knowledge claim in the crate is false.
        let c =
            prove_with_blinders(case.circuit.as_ref(), &case.witness, one, zero, &mut t).unwrap();
        assert_ne!(a.a, c.a, "{name}: r did not change A");
        let d =
            prove_with_blinders(case.circuit.as_ref(), &case.witness, zero, one, &mut t).unwrap();
        assert_ne!(a.b, d.b, "{name}: s did not change B");
    });
}

// ---------------------------------------------------------------------------------
// 4. field boundary values in the witness
// ---------------------------------------------------------------------------------

/// Writing 0, 1 or `r_modulus - 1` into a wire the circuit constrains breaks the R1CS, so
/// the proof must fail. This is the negative control for the whole suite: a prover that
/// had, say, a degenerate `H` or an `L` MSM that collapsed to the identity would pass every
/// positive test above while producing proofs that verify for any witness at all.
///
/// Only *private* wires are touched, so the public inputs the verifier is given stay the
/// ones in `public.json`. Mutating a public wire would change the statement as well as the
/// witness, and the rejection would then prove nothing about the prover.
///
/// A mutation that happens to be a no-op (the wire already held that value) is skipped
/// rather than asserted on, and a wire the circuit does not constrain would legitimately
/// still verify - the test reports any such position by name instead of failing blind.
#[test]
fn field_boundary_values_in_a_constrained_wire_do_not_verify() {
    each("field_boundaries", small_variants(), |name, case| {
        let n = case.witness.len();
        let first_private = case.circuit.n_public() + 1;
        assert!(first_private < n, "{name}: circuit has no private wires");

        // The first private wire and the last wire. In a circom witness the first is an
        // input signal and the last is the deepest intermediate, so between them they
        // cover both ends of the constraint graph.
        let positions = [first_private, n - 1];
        let values: [(&str, Fr); 3] = [("0", Fr::zero()), ("1", Fr::one()), ("r-1", -Fr::one())];

        let mut checked = 0usize;
        for pos in positions {
            for (label, v) in values {
                if case.witness[pos] == v {
                    eprintln!("  w[{pos}] is already {label}, skipping");
                    continue;
                }
                let mut w = case.witness.clone();
                w[pos] = v;
                let got = outcome(case, &w);
                eprintln!("  w[{pos}] = {label}: {got:?}");
                assert_eq!(
                    got,
                    Outcome::Unverifiable,
                    "{name}: w[{pos}] = {label} gave {got:?}; a witness that violates the \
                     constraint system produced a proof the verifier accepted"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "{name}: every mutation was a no-op, nothing was tested"
        );

        // The unmodified witness still verifies, so the assertions above are about the
        // mutation and not about a broken fixture.
        assert_eq!(
            outcome(case, &case.witness),
            Outcome::Accepted,
            "{name}: fixture is broken"
        );
    });
}

/// A public wire moved without moving the published signal is the same story from the
/// other side: prover and verifier now disagree about the statement, and the pairing must
/// notice. `w[1]` is the first public signal for every artifact here.
#[test]
fn a_public_wire_that_disagrees_with_the_published_signal_does_not_verify() {
    each("public_disagrees", small_variants(), |name, case| {
        assert!(case.circuit.n_public() >= 1);
        let mut w = case.witness.clone();
        w[1] += Fr::one();
        let got = outcome(case, &w);
        eprintln!("  w[1] += 1: {got:?}");
        assert_ne!(
            got,
            Outcome::Accepted,
            "{name}: prover and verifier disagreed and it verified"
        );
    });
}

// ---------------------------------------------------------------------------------
// 5. blinding is live on the real proving path
// ---------------------------------------------------------------------------------

/// `prove` draws `r` and `s` from the caller's CSPRNG. Two proofs of the *same* witness
/// must therefore differ in all three points, and both must verify. A prover that hard
/// coded the blinders, or drew them from a seeded RNG constructed inside `prove`, would
/// pass every other test in this file and leak the witness in production.
#[test]
fn two_proofs_of_the_same_witness_differ_and_both_verify() {
    each("live_blinding", small_variants(), |name, case| {
        let mut rng = ark_std::rand::thread_rng();
        let mut t = StageTimings::default();
        let p1 = prove(case.circuit.as_ref(), &case.witness, &mut rng, &mut t).unwrap();
        let p2 = prove(case.circuit.as_ref(), &case.witness, &mut rng, &mut t).unwrap();

        verify(&case.vk, &case.public, &p1).unwrap();
        verify(&case.vk, &case.public, &p2).unwrap();

        assert_ne!(p1.a, p2.a, "{name}: A repeated across two proofs");
        assert_ne!(p1.b, p2.b, "{name}: B repeated across two proofs");
        assert_ne!(p1.c, p2.c, "{name}: C repeated across two proofs");

        // And a third, from a *fresh* `thread_rng`, so the difference is not merely the
        // stream advancing within one generator instance.
        let mut rng2 = ark_std::rand::thread_rng();
        let p3 = prove(case.circuit.as_ref(), &case.witness, &mut rng2, &mut t).unwrap();
        verify(&case.vk, &case.public, &p3).unwrap();
        assert_ne!(p1.a, p3.a, "{name}: a fresh RNG reproduced the first proof");
    });
}

/// The shape of the rejection matters as much as the fact of it: a wrong-length *public*
/// vector is a caller error and must not be reported as a failed pairing, or an integrator
/// will spend the afternoon looking for a soundness bug in a mis-sized array.
#[test]
fn the_verifier_separates_a_shape_error_from_a_failed_pairing() {
    each("verify_shapes", small_variants(), |name, case| {
        let mut t = StageTimings::default();
        let p = prove_with_blinders(
            case.circuit.as_ref(),
            &case.witness,
            Fr::zero(),
            Fr::zero(),
            &mut t,
        )
        .unwrap();
        verify(&case.vk, &case.public, &p).unwrap();

        let n = case.public.len();
        assert!(
            matches!(
                verify(&case.vk, &case.public[..n - 1], &p),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{name}: a short public vector was not a shape error"
        );

        let mut long = case.public.clone();
        long.push(Fr::zero());
        assert!(
            matches!(
                verify(&case.vk, &long, &p),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{name}: a long public vector was not a shape error"
        );

        assert!(
            matches!(
                verify(&case.vk, &[], &p),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{name}: an empty public vector was not a shape error"
        );

        // Right shape, wrong values, including the two field boundaries.
        for (label, v) in [("0", Fr::zero()), ("1", Fr::one()), ("r-1", -Fr::one())] {
            let mut bad = case.public.clone();
            if bad[0] == v {
                continue;
            }
            bad[0] = v;
            assert!(
                matches!(verify(&case.vk, &bad, &p), Err(VerifyError::PairingFailed)),
                "{name}: public[0] = {label} was accepted"
            );
        }
    });
}
