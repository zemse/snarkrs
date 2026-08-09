//! The verifier's negative matrix: every proof shape that MUST be rejected, and the one
//! that must not be.
//!
//! The positive tests elsewhere in this workspace answer "does a correct proof verify".
//! That question is cheap to pass by accident: a verifier that returned `Ok(())`
//! unconditionally passes every one of them. This file asks the complementary question,
//! which is the one soundness actually rests on: for each way a proof can be wrong, does
//! the verifier say no?
//!
//! The matrix is built by mutating a known-good proof and asserting rejection:
//!
//! | family                | cases                                                      |
//! |-----------------------|------------------------------------------------------------|
//! | single bit flips      | 5 bit positions x 8 coordinates (a.x, a.y, b.x.c0, b.x.c1, |
//! |                       | b.y.c0, b.y.c1, c.x, c.y)                                  |
//! | intra-proof splice    | pi_a := pi_c, pi_c := pi_a, pi_a and pi_c swapped          |
//! | cross-proof splice    | each element, and each pair, taken from a second valid     |
//! |                       | proof of the SAME statement                                |
//! | point at infinity     | in each of the three positions                             |
//! | negation              | of each of the three elements individually                 |
//! | wrong public inputs   | one value changed, one too few, one too many               |
//! | wrong statement       | a proof for circuit X checked against circuit Y            |
//!
//! and the cases that must be ACCEPTED, because Groth16 is malleable by design and a
//! verifier that "fixed" that would be rejecting valid proofs:
//!
//! * `malleated_proof_still_verifies` - `(A z^-1, B z, C)` for random nonzero `z`;
//! * the `(-A, -B, C)` tail of `negating_any_element_is_rejected`, which is `z = -1` and
//!   is the instance a reviewer's intuition most often gets backwards.
//!
//! Read the comment on `malleated_proof_still_verifies` before adding any uniqueness,
//! canonicalisation, or replay check anywhere near a proof.
//!
//! Everything runs against the checked-in artifacts and skips loudly when they are
//! absent, so a fresh clone without `gen-artifacts.sh` cannot report false green.

use std::path::{Path, PathBuf};

use g16_cli::json::{proof_from_value, proof_to_string, read_public};
use g16_core::cpu::CpuBackend;
use g16_core::prove::prove_with_blinders;
use g16_core::verify::{verify, VerifyError};
use g16_core::{Backend, PreparedCircuit, Proof, StageTimings};
use g16_field::*;
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};
use num_bigint::BigUint;

/// The small circuit (3 public signals) and one mid-size circuit (6 public signals).
/// Two sizes rather than one because the small one has a domain small enough that a
/// degenerate H would go unnoticed, and because a bug that only shows up once the
/// windowed MSM has more than one window would otherwise never be mutated at all.
const VARIANTS: [&str; 2] = ["tiny_mul", "js_2x2_d16"];

/// A third variant, used only for the wrong-statement test. It has the same number of
/// public signals as `js_2x2_d16`, so swapping the two exercises the pairing check
/// rather than the much easier arity check.
const SAME_ARITY_SIBLING: (&str, &str) = ("js_2x2_d16", "js_2x2_d32");

fn artifacts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

struct Fixture {
    name: String,
    vk: VerifyingKey,
    public: Vec<Fr>,
    circuit: Box<dyn PreparedCircuit>,
    witness: Vec<Fr>,
}

impl Fixture {
    /// Deterministic on purpose. Every case below has to be reproducible from the test
    /// name alone, and `prove_with_blinders` is the only way to get that; the blinders
    /// are still varied between the two proofs so the cross-proof splices are splicing
    /// genuinely different group elements.
    fn prove(&self, r: u64, s: u64) -> Proof {
        let mut t = StageTimings::default();
        prove_with_blinders(
            self.circuit.as_ref(),
            &self.witness,
            Fr::from(r),
            Fr::from(s),
            &mut t,
        )
        .expect("proving a checked-in witness must succeed")
    }
}

fn load(name: &str) -> Option<Fixture> {
    let dir = artifacts_root().join(name);
    let complete = ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"]
        .iter()
        .all(|f| dir.join(f).is_file());
    if !complete {
        eprintln!("SKIPPED {name}: no artifacts under {}", dir.display());
        return None;
    }
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
    let circuit = CpuBackend::new().prepare(pk).unwrap();
    Some(Fixture {
        name: name.to_string(),
        vk: VerifyingKey::from_json(&dir.join("vkey.json")).unwrap(),
        public: read_public(&dir.join("public.json")).unwrap(),
        circuit,
        witness,
    })
}

/// Runs `f` over each variant that is present. Reports a skip rather than passing
/// silently when none are.
fn each(test: &str, f: impl Fn(&Fixture)) {
    let found: Vec<Fixture> = VARIANTS.iter().filter_map(|n| load(n)).collect();
    if found.is_empty() {
        eprintln!("SKIPPED {test}: no artifacts at all");
        return;
    }
    for fx in &found {
        eprintln!("{test}: {}", fx.name);
        f(fx);
    }
}

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------

fn to_biguint<F: PrimeField>(x: F) -> BigUint {
    BigUint::from_bytes_le(&x.into_bigint().to_bytes_le())
}

/// Flip one bit of a base-field coordinate, in its NORMAL (non-Montgomery) form, which
/// is the form a proof is transmitted in. The result is reduced mod p, so this models a
/// corrupted wire value that is still a syntactically well-formed field element rather
/// than an out-of-range integer (that case is `json.rs`'s job and is tested there).
///
/// Returns `None` when the flip is a no-op after reduction, so a vacuous case cannot
/// masquerade as a passing one.
fn flip_bit(x: Fq, bit: u32) -> Option<Fq> {
    let flipped = to_biguint(x) ^ (BigUint::from(1u32) << bit);
    let y = Fq::from_le_bytes_mod_order(&flipped.to_bytes_le());
    (y != x).then_some(y)
}

/// Which coordinate of the proof a bit flip lands in.
#[derive(Clone, Copy, Debug)]
enum Coord {
    AX,
    AY,
    BXC0,
    BXC1,
    BYC0,
    BYC1,
    CX,
    CY,
}

impl Coord {
    const ALL: [Coord; 8] = [
        Coord::AX,
        Coord::AY,
        Coord::BXC0,
        Coord::BXC1,
        Coord::BYC0,
        Coord::BYC1,
        Coord::CX,
        Coord::CY,
    ];

    /// The mutated proof, or `None` if the flip reduced away.
    ///
    /// Built with `new_unchecked` deliberately. A bit flip almost always leaves the
    /// curve, and the point of this test is that the pairing check rejects such a proof
    /// on its own, without leaning on the deserializer's on-curve screen. Both layers
    /// have to say no; `off_curve_mutations_are_also_rejected_at_the_json_layer` below
    /// covers the other one.
    fn apply(self, p: &Proof, bit: u32) -> Option<Proof> {
        let mut out = p.clone();
        match self {
            Coord::AX => out.a = G1Affine::new_unchecked(flip_bit(p.a.x, bit)?, p.a.y),
            Coord::AY => out.a = G1Affine::new_unchecked(p.a.x, flip_bit(p.a.y, bit)?),
            Coord::BXC0 => {
                out.b = G2Affine::new_unchecked(Fq2::new(flip_bit(p.b.x.c0, bit)?, p.b.x.c1), p.b.y)
            }
            Coord::BXC1 => {
                out.b = G2Affine::new_unchecked(Fq2::new(p.b.x.c0, flip_bit(p.b.x.c1, bit)?), p.b.y)
            }
            Coord::BYC0 => {
                out.b = G2Affine::new_unchecked(p.b.x, Fq2::new(flip_bit(p.b.y.c0, bit)?, p.b.y.c1))
            }
            Coord::BYC1 => {
                out.b = G2Affine::new_unchecked(p.b.x, Fq2::new(p.b.y.c0, flip_bit(p.b.y.c1, bit)?))
            }
            Coord::CX => out.c = G1Affine::new_unchecked(flip_bit(p.c.x, bit)?, p.c.y),
            Coord::CY => out.c = G1Affine::new_unchecked(p.c.x, flip_bit(p.c.y, bit)?),
        }
        Some(out)
    }
}

/// Bit positions spread across a 254-bit coordinate: the two lowest, a limb boundary,
/// and two high positions. A prover-side bug that only touches a top limb would survive
/// a test that flipped bit 0 alone.
const BITS: [u32; 5] = [0, 1, 63, 127, 200];

fn eq_proof(a: &Proof, b: &Proof) -> bool {
    a.a == b.a && a.b == b.b && a.c == b.c
}

/// Assert the verifier rejects, and say which case failed when it does not. A mutated
/// proof that still verifies is the single worst outcome this file can produce, so the
/// message names the case rather than relying on the line number.
#[track_caller]
fn must_reject_with_pairing_failure(fx: &Fixture, public: &[Fr], proof: &Proof, case: &str) {
    match verify(&fx.vk, public, proof) {
        Err(VerifyError::PairingFailed) => {}
        Err(e) => panic!(
            "{}: {case} was rejected as {e}, expected a pairing failure",
            fx.name
        ),
        Ok(()) => panic!("{}: verifier ACCEPTED {case}", fx.name),
    }
}

// ---------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------

/// Sanity floor for everything below. If the unmutated proof did not verify, every
/// rejection in this file would be trivially true and the whole file would be worthless.
#[test]
fn the_baseline_proof_verifies() {
    each("the_baseline_proof_verifies", |fx| {
        verify(&fx.vk, &fx.public, &fx.prove(3, 5)).expect("baseline proof must verify");
    });
}

#[test]
fn single_bit_flips_in_every_coordinate_are_rejected() {
    each("single_bit_flips", |fx| {
        let good = fx.prove(3, 5);
        verify(&fx.vk, &fx.public, &good).unwrap();

        let mut checked = 0;
        for coord in Coord::ALL {
            for bit in BITS {
                let Some(bad) = coord.apply(&good, bit) else {
                    continue;
                };
                assert!(
                    !eq_proof(&bad, &good),
                    "{}: flipping {coord:?} bit {bit} was a no-op",
                    fx.name
                );
                must_reject_with_pairing_failure(
                    fx,
                    &fx.public,
                    &bad,
                    &format!("a single bit flip in {coord:?} at bit {bit}"),
                );
                checked += 1;
            }
        }
        // Guards against the loop silently doing nothing, e.g. if `flip_bit` started
        // returning `None` for everything.
        assert_eq!(
            checked,
            Coord::ALL.len() * BITS.len(),
            "{}: some bit flips reduced away and were skipped",
            fx.name
        );
    });
}

/// A bit flip almost certainly leaves the curve, and the JSON reader must reject it
/// before the pairing ever runs. This is the defence that matters for a verifier fed
/// untrusted input: `verify` itself documents that it does no curve or subgroup check.
#[test]
fn off_curve_mutations_are_also_rejected_at_the_json_layer() {
    each("off_curve_at_json_layer", |fx| {
        let good = fx.prove(3, 5);
        for coord in Coord::ALL {
            let Some(bad) = coord.apply(&good, 1) else {
                continue;
            };
            // `proof_to_string` writes whatever affine coordinates it is given, so a
            // mutated point survives the write and has to be caught on the way back in.
            let text = proof_to_string(&bad);
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            let on_curve = match coord {
                Coord::BXC0 | Coord::BXC1 | Coord::BYC0 | Coord::BYC1 => bad.b.is_on_curve(),
                _ => bad.a.is_on_curve() && bad.c.is_on_curve(),
            };
            if on_curve {
                // Astronomically unlikely, but do not silently pass if it happens: the
                // pairing check is then the only line of defence and it was already
                // asserted above.
                eprintln!("  {coord:?} bit 1 landed back on the curve; pairing check covers it");
                continue;
            }
            assert!(
                proof_from_value(&value).is_err(),
                "{}: the JSON reader accepted an off-curve {coord:?}",
                fx.name
            );
        }
    });
}

#[test]
fn substituting_one_element_of_the_proof_for_another_is_rejected() {
    each("intra_proof_splice", |fx| {
        let good = fx.prove(3, 5);
        // A and C are both G1, so this is the one substitution that type-checks in a
        // real deserializer too: an attacker can post it and it is well-formed.
        for (case, bad) in [
            (
                "pi_a := pi_c",
                Proof {
                    a: good.c,
                    ..good.clone()
                },
            ),
            (
                "pi_c := pi_a",
                Proof {
                    c: good.a,
                    ..good.clone()
                },
            ),
            (
                "pi_a and pi_c swapped",
                Proof {
                    a: good.c,
                    b: good.b,
                    c: good.a,
                },
            ),
        ] {
            must_reject_with_pairing_failure(fx, &fx.public, &bad, case);
        }
    });
}

/// The interesting adversary: someone who has TWO valid proofs of the same statement and
/// mixes them. Both are individually accepted, so nothing about the elements themselves
/// is malformed. Only the relation between them is broken.
#[test]
fn cross_proof_splices_of_the_same_statement_are_rejected() {
    each("cross_proof_splice", |fx| {
        let p1 = fx.prove(3, 5);
        let p2 = fx.prove(7, 11);
        verify(&fx.vk, &fx.public, &p1).unwrap();
        verify(&fx.vk, &fx.public, &p2).unwrap();
        // Different blinders have to give genuinely different elements, otherwise the
        // splices below are splicing a proof into itself and prove nothing.
        assert_ne!(p1.a, p2.a, "{}: blinders did not change A", fx.name);
        assert_ne!(p1.b, p2.b, "{}: blinders did not change B", fx.name);
        assert_ne!(p1.c, p2.c, "{}: blinders did not change C", fx.name);

        let splices: [(&str, Proof); 6] = [
            (
                "A from proof 2",
                Proof {
                    a: p2.a,
                    b: p1.b,
                    c: p1.c,
                },
            ),
            (
                "B from proof 2",
                Proof {
                    a: p1.a,
                    b: p2.b,
                    c: p1.c,
                },
            ),
            (
                "C from proof 2",
                Proof {
                    a: p1.a,
                    b: p1.b,
                    c: p2.c,
                },
            ),
            (
                "A and B from proof 2",
                Proof {
                    a: p2.a,
                    b: p2.b,
                    c: p1.c,
                },
            ),
            (
                "A and C from proof 2",
                Proof {
                    a: p2.a,
                    b: p1.b,
                    c: p2.c,
                },
            ),
            (
                "B and C from proof 2",
                Proof {
                    a: p1.a,
                    b: p2.b,
                    c: p2.c,
                },
            ),
        ];
        for (case, bad) in splices {
            must_reject_with_pairing_failure(fx, &fx.public, &bad, case);
        }

        // All three together is just proof 2, which must still verify. Without this the
        // test above could pass because the verifier rejects everything.
        verify(
            &fx.vk,
            &fx.public,
            &Proof {
                a: p2.a,
                b: p2.b,
                c: p2.c,
            },
        )
        .unwrap();
    });
}

/// The identity is the one point every pairing implementation special-cases, so it is
/// the one most likely to short-circuit a check into vacuous truth. `e(0, B) = 1`, which
/// removes a factor from the product entirely.
#[test]
fn the_point_at_infinity_is_rejected_in_every_position() {
    each("point_at_infinity", |fx| {
        let good = fx.prove(3, 5);
        let cases: [(&str, Proof); 3] = [
            (
                "pi_a = O",
                Proof {
                    a: G1Affine::identity(),
                    ..good.clone()
                },
            ),
            (
                "pi_b = O",
                Proof {
                    b: G2Affine::identity(),
                    ..good.clone()
                },
            ),
            (
                "pi_c = O",
                Proof {
                    c: G1Affine::identity(),
                    ..good.clone()
                },
            ),
        ];
        for (case, bad) in cases {
            must_reject_with_pairing_failure(fx, &fx.public, &bad, case);
        }
        // The all-zero proof is the degenerate one a stub or a zeroed buffer produces,
        // and it must not be mistaken for a proof of anything.
        must_reject_with_pairing_failure(
            fx,
            &fx.public,
            &Proof {
                a: G1Affine::identity(),
                b: G2Affine::identity(),
                c: G1Affine::identity(),
            },
            "an all-identity proof",
        );
    });
}

/// Negation is the cheapest algebraic mutation an attacker can apply and it stays on the
/// curve and in the subgroup, so no deserializer screen will catch it. Only the pairing
/// check can.
#[test]
fn negating_any_element_is_rejected() {
    each("negation", |fx| {
        let good = fx.prove(3, 5);
        let cases: [(&str, Proof); 3] = [
            (
                "-pi_a",
                Proof {
                    a: -good.a,
                    ..good.clone()
                },
            ),
            (
                "-pi_b",
                Proof {
                    b: -good.b,
                    ..good.clone()
                },
            ),
            (
                "-pi_c",
                Proof {
                    c: -good.c,
                    ..good.clone()
                },
            ),
        ];
        for (case, bad) in cases {
            must_reject_with_pairing_failure(fx, &fx.public, &bad, case);
        }

        // Negating BOTH A and B must be ACCEPTED, and this is the trap in the family.
        // `e(-A, -B) = e(A, B)^((-1)(-1)) = e(A, B)`, so `(-A, -B, C)` is precisely the
        // `z = -1` instance of the rescaling in `malleated_proof_still_verifies` below.
        // It is the cheapest malleability there is (a field negation each, no inversion,
        // no scalar multiplication) and it is the one a reviewer's intuition gets wrong:
        // "both elements changed, so reject" is not sound reasoning, it is a bug.
        // Asserted here rather than only in the malleability test because this is where
        // someone reading the negation cases will look.
        verify(
            &fx.vk,
            &fx.public,
            &Proof {
                a: -good.a,
                b: -good.b,
                c: good.c,
            },
        )
        .unwrap_or_else(|e| {
            panic!(
                "{}: the verifier rejected (-A, -B, C), which is the z = -1 \
                 re-randomisation and must verify: {e}",
                fx.name
            )
        });
    });
}

#[test]
fn wrong_public_inputs_are_rejected() {
    each("wrong_public_inputs", |fx| {
        let good = fx.prove(3, 5);
        verify(&fx.vk, &fx.public, &good).unwrap();

        // One value changed, at every position. A verifier that aggregated only IC[0..1]
        // would still pass a test that only ever touched signal 0.
        for i in 0..fx.public.len() {
            let mut public = fx.public.clone();
            public[i] += Fr::one();
            must_reject_with_pairing_failure(
                fx,
                &public,
                &good,
                &format!("public signal {i} incremented"),
            );
        }

        // Zeroing every signal: the statement is now "nothing", and the aggregate is
        // just IC[0].
        let zeros = vec![Fr::zero(); fx.public.len()];
        if zeros != fx.public {
            must_reject_with_pairing_failure(fx, &zeros, &good, "all public signals zeroed");
        }

        // Arity errors are a different class: a caller bug, not a rejected proof, and
        // the verifier must distinguish them rather than folding both into "invalid".
        let short = &fx.public[..fx.public.len() - 1];
        assert!(
            matches!(
                verify(&fx.vk, short, &good),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{}: too few public inputs was not reported as an arity error",
            fx.name
        );
        let mut long = fx.public.clone();
        long.push(Fr::from(42u64));
        assert!(
            matches!(
                verify(&fx.vk, &long, &good),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{}: too many public inputs was not reported as an arity error",
            fx.name
        );
        // And empty, which is the shape a caller gets from a mis-parsed public.json.
        assert!(
            matches!(
                verify(&fx.vk, &[], &good),
                Err(VerifyError::PublicInputCount { .. })
            ),
            "{}: an empty public vector was not reported as an arity error",
            fx.name
        );
    });
}

/// A proof for circuit X, checked against circuit Y's verifying key and public signals.
/// This is the cross-circuit replay an aggregator gets wrong when it looks up the wrong
/// verifying key, and it must fail even when the two circuits happen to agree on how
/// many public signals they take.
#[test]
fn a_proof_for_one_statement_does_not_verify_against_another() {
    let (a_name, b_name) = SAME_ARITY_SIBLING;
    let (Some(x), Some(y)) = (load(a_name), load(b_name)) else {
        eprintln!("SKIPPED a_proof_for_one_statement: need both {a_name} and {b_name}");
        return;
    };
    assert_eq!(
        x.public.len(),
        y.public.len(),
        "{a_name} and {b_name} were chosen because they have the same public arity"
    );
    let px = x.prove(3, 5);
    let py = y.prove(3, 5);
    verify(&x.vk, &x.public, &px).unwrap();
    verify(&y.vk, &y.public, &py).unwrap();

    // Same arity, so this reaches the pairing rather than stopping at the arity check.
    must_reject_with_pairing_failure(
        &y,
        &y.public,
        &px,
        &format!("a proof for {a_name} checked under {b_name}"),
    );
    must_reject_with_pairing_failure(
        &x,
        &x.public,
        &py,
        &format!("a proof for {b_name} checked under {a_name}"),
    );

    // Also the mixed forms: right key, wrong signals, and wrong key, right signals.
    must_reject_with_pairing_failure(
        &x,
        &y.public,
        &px,
        &format!("{a_name}'s proof and key with {b_name}'s signals"),
    );
    must_reject_with_pairing_failure(
        &y,
        &x.public,
        &py,
        &format!("{b_name}'s proof and key with {a_name}'s signals"),
    );

    // And a differing-arity pair, which must be caught as an arity error rather than as
    // a pairing failure, because the two mean different things to the caller.
    if let Some(tiny) = load("tiny_mul") {
        if tiny.public.len() != x.public.len() {
            let pt = tiny.prove(3, 5);
            assert!(
                matches!(
                    verify(&x.vk, &tiny.public, &pt),
                    Err(VerifyError::PublicInputCount { .. })
                ),
                "a proof from a circuit with a different public arity was not caught as one"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The one case that must be ACCEPTED
// ---------------------------------------------------------------------------

/// **This test asserts an ACCEPT, and that is not a bug.**
///
/// Groth16 is malleable. Given any valid `(A, B, C)` and any nonzero `z` in `Fr`, the
/// triple `(A * z^-1, B * z, C)` is a different, equally valid proof of the same
/// statement, and producing it needs no witness and no trapdoor. The pairing check is
///
/// ```text
/// e(A, B) = e(alpha, beta) * e(L_bar, gamma) * e(C, delta)
/// ```
///
/// and `e(A z^-1, B z) = e(A, B)^(z^-1 * z) = e(A, B)` by bilinearity. The left side is
/// unchanged; the right side never mentioned `A` or `B` at all. So there are on the
/// order of `r` distinct encodings of every proof, roughly 2^254 of them, and anyone who
/// merely *observes* a proof can mint as many as they like.
///
/// The consequence is for the caller, not for the verifier:
///
/// **Never use proof bytes as a nullifier, a replay key, a deduplication key, a cache
/// key with security meaning, or any kind of identity.** A contract that stores
/// `keccak(proof)` and rejects repeats does not prevent replay: an attacker re-randomises
/// the observed proof and walks straight through. Key on `(verifying key, public
/// inputs)`, which is what the proof actually attests to, and put any anti-replay value
/// (a nonce, a nullifier) *inside* the public inputs where the proof commits to it.
///
/// This is pinned as a test rather than left as a comment because it reads like a bug.
/// The failure mode to guard against is a well-meaning change that adds a uniqueness or
/// canonicalisation check to the verifier: that would not stop the attack (the attacker
/// controls `z` and can pick a canonical-looking one) and it WOULD start rejecting valid
/// proofs. If this test ever fails, the verifier has been made unsound-adjacent in the
/// other direction and the change should be reverted, not accommodated.
#[test]
fn malleated_proof_still_verifies() {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    each("malleated_proof_still_verifies", |fx| {
        let good = fx.prove(3, 5);
        verify(&fx.vk, &fx.public, &good).unwrap();

        let mut rng = StdRng::from_seed([31u8; 32]);
        for i in 0..8 {
            let z = loop {
                let z = Fr::rand(&mut rng);
                if !z.is_zero() && z != Fr::one() {
                    break z;
                }
            };
            let zi = z.inverse().expect("z is nonzero");
            let mal = Proof {
                a: (good.a * zi).into_affine(),
                b: (good.b * z).into_affine(),
                c: good.c,
            };

            // It really is a different encoding, otherwise the accept below is vacuous.
            assert_ne!(mal.a, good.a, "{}: rescaling {i} did not move A", fx.name);
            assert_ne!(mal.b, good.b, "{}: rescaling {i} did not move B", fx.name);
            assert_eq!(mal.c, good.c, "{}: C must be untouched", fx.name);

            verify(&fx.vk, &fx.public, &mal).unwrap_or_else(|e| {
                panic!(
                    "{}: the verifier REJECTED a legitimately re-randomised proof: {e}. \
                     Groth16 is malleable; see this test's comment.",
                    fx.name
                )
            });

            // And it survives the serialization boundary, which is the form an attacker
            // would actually publish: distinct bytes, same statement, still accepted.
            let text = proof_to_string(&mal);
            assert_ne!(
                text,
                proof_to_string(&good),
                "{}: the re-randomised proof serialised identically",
                fx.name
            );
            let round_tripped = proof_from_value(&serde_json::from_str(&text).unwrap())
                .unwrap_or_else(|e| {
                    panic!(
                        "{}: the JSON reader rejected a valid re-randomised proof: {e:?}",
                        fx.name
                    )
                });
            verify(&fx.vk, &fx.public, &round_tripped).unwrap();
        }
    });
}
