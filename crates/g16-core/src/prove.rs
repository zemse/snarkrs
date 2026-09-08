//! Stages 10-11: blinding and assembly, plus the top-level `prove` that drives a backend.

// Not `std::time`: `Instant::now()` panics at run time on wasm32-unknown-unknown, and
// `assemble` below is on the browser prover's path. web-time is a re-export of `std::time`
// on every other target, so nothing native changes.
use web_time::Instant;

use crate::{MsmOutputs, PreparedCircuit, Proof, ProveError, StageTimings};
use g16_field::*;
use g16_zkey::ProvingKey;

/// Full proof: stages 0-11. `rng` supplies the zero-knowledge blinders `r` and `s`.
pub fn prove<R: ark_std::rand::RngCore + ark_std::rand::CryptoRng>(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    rng: &mut R,
    timings: &mut StageTimings,
) -> Result<Proof, ProveError> {
    // Stage 10. The whole zero-knowledge property rests on these two, which is why the
    // bound is `CryptoRng` and why the deterministic path below is a separate function
    // rather than a default argument someone could reach for by accident.
    let r = Fr::rand(rng);
    let s = Fr::rand(rng);
    prove_with_blinders(circuit, witness, r, s, timings)
}

/// Refuse to prove in a build that would take tens of minutes to do it.
///
/// The guard sits here rather than in the backends because this is the one choke point every
/// proving test reaches, and putting it in nine test files leaves the tenth unprotected.
/// [`prove_trace`] is deliberately exempt: a trace is one proof produced on purpose to debug a
/// device, which is the single case where a slow build is the point.
///
/// `cfg(unoptimized)` comes from build.rs, so at opt-level 2 or above this compiles to nothing
/// and there is no branch on the hot path to inline away.
#[inline]
fn refuse_if_unoptimized() -> Result<(), ProveError> {
    #[cfg(unoptimized)]
    if std::env::var_os("G16_ALLOW_UNOPTIMIZED_PROVING").is_none() {
        return Err(ProveError::Unoptimized);
    }
    Ok(())
}

/// Deterministic variant with caller-supplied `r`, `s`. Used only by tests, so a proof
/// can be compared against a reference implementation bit for bit. Never use in
/// production: reusing `r`/`s` across proofs of different witnesses leaks the witness.
pub fn prove_with_blinders(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    r: Fr,
    s: Fr,
    timings: &mut StageTimings,
) -> Result<Proof, ProveError> {
    refuse_if_unoptimized()?;
    let h = stage!("stages 0-4 compute_h", circuit.compute_h(witness, timings))?;
    let m = stage!("stages 5-9 msms", circuit.msms(witness, &h, timings))?;
    Ok(assemble(circuit.key(), &m, r, s, timings))
}

/// A full proof plus every intermediate value that two machines must agree on.
///
/// **Debugging only**, for the reasons in [`crate::trace`]: the blinders are the fixed
/// constants there, so this is not a proof anybody may publish. It exists because a wrong
/// 256-bit number at the end of a proof says nothing about which stage wrote it, and the
/// backend that produces one only does so on a device that is not on this desk.
///
/// Synchronous, so the GPU backends reach it only through their blocking `PreparedCircuit`
/// impl. The browser prover cannot use it at all and assembles the same `Trace` itself, for
/// the same reason [`assemble`] is public: every WebGPU readback is asynchronous.
pub fn prove_trace(
    circuit: &dyn PreparedCircuit,
    witness: &[Fr],
    timings: &mut StageTimings,
) -> Result<crate::trace::Trace, ProveError> {
    let (r, s) = crate::trace::blinders();
    let h = circuit.compute_h(witness, timings)?;
    let m = circuit.msms(witness, &h, timings)?;
    let proof = assemble(circuit.key(), &m, r, s, timings);
    // After the MSMs, not before: a readback inserted ahead of them is an extra submit in
    // the middle of the sequence being investigated, and one hypothesis for the iPhone is
    // that the sequence itself is what goes wrong. The MSMs only read that buffer, so
    // copying it down afterwards sees the same values.
    let host_h = circuit.h_to_host(&h);
    Ok(crate::trace::Trace::build(
        circuit.backend_name(),
        circuit.n_vars(),
        circuit.n_public(),
        circuit.domain_size(),
        witness,
        host_h.as_deref(),
        &m,
        &proof,
    ))
}

/// Stage 11 alone: blind the five MSM outputs into `(A, B, C)`.
///
/// Split out of [`prove_with_blinders`] because the browser prover cannot go through
/// [`PreparedCircuit`] at all. That trait is synchronous and every WebGPU readback is
/// asynchronous, so `g16-wgpu`'s wasm entry point awaits stages 0 to 4 and 5 to 9 itself and
/// then needs exactly this. Copying these six lines into that crate instead is how a
/// transcription error gets into one backend and not the other, and the one term that invites
/// it is `s * pi_a`: that is the *blinded* A, not the raw MSM.
pub fn assemble(
    pk: &ProvingKey,
    m: &MsmOutputs,
    r: Fr,
    s: Fr,
    timings: &mut StageTimings,
) -> Proof {
    let start = Instant::now();

    // Stage 11, in snarkjs' order (groth16_prove.js). Deriving it from the article's
    // equation gives the same thing: A = alpha + sum(a_i w_i) + r*delta lives in G1,
    // B = beta + sum(b_i w_i) + s*delta in G2, and C absorbs the cross terms s*A and
    // r*B1 with the double-counted r*s*delta subtracted back off. The one place a
    // transcription can go wrong is `s * pi_a`: that is the *blinded* A, not the raw
    // MSM, because the s*A term has to carry the alpha and r*delta pieces too.
    let (pi_a, pi_b, pi_c) = stage!("s11 blind (6 scalar multiplications)", {
        let pi_a = m.a_g1 + pk.alpha_g1 + pk.delta_g1 * r;
        let pi_b = m.b_g2 + pk.beta_g2 + pk.delta_g2 * s;
        let pib1 = m.b_g1 + pk.beta_g1 + pk.delta_g1 * s;
        let pi_c = m.l_g1 + m.h_g1 + pi_a * s + pib1 * r - pk.delta_g1 * (r * s);
        (pi_a, pi_b, pi_c)
    });

    // Two batch normalisations rather than three separate inversions.
    let (g1, g2) = stage!("s11 normalise (2 batch inversions)", {
        (
            G1Projective::normalize_batch(&[pi_a, pi_c]),
            G2Projective::normalize_batch(&[pi_b]),
        )
    });
    timings.assemble_us += start.elapsed().as_micros() as u64;

    Proof {
        a: g1[0],
        b: g2[0],
        c: g1[1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::verify::{aggregate_public, verify, VerifyError};
    use crate::Backend;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};
    use std::path::{Path, PathBuf};

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

    /// Runs `f` over every artifact, printing which one. An empty artifact directory
    /// reports a skip on stderr instead of passing silently, so a missing fixture cannot
    /// be mistaken for a green end-to-end test.
    fn for_each_artifact(test: &str, f: impl Fn(&Artifact)) {
        let found = artifacts();
        if found.is_empty() {
            eprintln!("SKIPPED {test}: no artifacts under bench/artifacts");
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

    fn dec_fq(v: &serde_json::Value) -> Fq {
        let n: num_bigint::BigUint = v.as_str().unwrap().parse().unwrap();
        Fq::from_le_bytes_mod_order(&n.to_bytes_le())
    }

    fn public_inputs(dir: &Path) -> Vec<Fr> {
        json(&dir.join("public.json"))
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let n: num_bigint::BigUint = s.as_str().unwrap().parse().unwrap();
                Fr::from_le_bytes_mod_order(&n.to_bytes_le())
            })
            .collect()
    }

    /// snarkjs writes proof points as projective triples of decimal strings, the same
    /// encoding `vkey.json` uses. Parsed here rather than through `g16-zkey`'s private
    /// helpers so this crate's tests do not depend on that crate widening its API.
    fn reference_proof(dir: &Path) -> Proof {
        let v = json(&dir.join("proof.json"));
        let g1 = |k: &str| {
            let p = &v[k];
            let z = dec_fq(&p[2]);
            if z.is_zero() {
                G1Affine::identity()
            } else {
                let zi = z.inverse().unwrap();
                G1Affine::new(dec_fq(&p[0]) * zi, dec_fq(&p[1]) * zi)
            }
        };
        let fq2 = |p: &serde_json::Value| Fq2::new(dec_fq(&p[0]), dec_fq(&p[1]));
        let p = &v["pi_b"];
        let z = fq2(&p[2]);
        let b = if z.is_zero() {
            G2Affine::identity()
        } else {
            let zi = z.inverse().unwrap();
            G2Affine::new(fq2(&p[0]) * zi, fq2(&p[1]) * zi)
        };
        Proof {
            a: g1("pi_a"),
            b,
            c: g1("pi_c"),
        }
    }

    fn prepared(dir: &Path) -> (Box<dyn PreparedCircuit>, Vec<Fr>) {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let circuit = CpuBackend::new().prepare(pk).unwrap();
        (circuit, witness)
    }

    /// The domain root has to be the *same element* ffjavascript uses, not merely some
    /// primitive root of the right order: the QAP polynomials were interpolated on
    /// snarkjs' ordering of the domain, so a different root permutes every evaluation.
    /// The two values below were printed from ffjavascript's bn128 `Fr.w`.
    #[test]
    fn domain_roots_match_ffjavascript() {
        let w = |d: &str| {
            let n: num_bigint::BigUint = d.parse().unwrap();
            Fr::from_le_bytes_mod_order(&n.to_bytes_le())
        };
        assert_eq!(
            Domain::new(1 << 28).unwrap().group_gen,
            w("19103219067921713944291392827692070036145651957329286315305642004821462161904")
        );
        assert_eq!(
            Domain::new(16).unwrap().group_gen,
            w("14940766826517323942636479241147756311199852622225275649687664389641784935947")
        );
    }

    /// Separates "our prover is broken" from "our verifier is broken". If this fails and
    /// the end-to-end test also fails, the verifier is the suspect.
    #[test]
    fn verifier_accepts_snarkjs_own_proof() {
        for_each_artifact("verifier_accepts_snarkjs_own_proof", |a| {
            if !a.dir.join("proof.json").is_file() {
                eprintln!("  no proof.json, skipping");
                return;
            }
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);
            verify(&vk, &public, &reference_proof(&a.dir)).unwrap();
        });
    }

    /// The test the whole project exists for: our prover, our verifier, snarkjs' key.
    #[test]
    fn end_to_end_our_proof_verifies() {
        for_each_artifact("end_to_end_our_proof_verifies", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);

            let mut t = StageTimings::default();
            let mut rng = StdRng::from_seed([7u8; 32]);
            let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).unwrap();
            verify(&vk, &public, &proof).unwrap();
        });
    }

    /// The trait declares `Send + Sync` and says one instance must be safe to call from
    /// several threads, which is the whole reason `prepare` is separate: a server holds
    /// one resident key and proves many witnesses. Every other test drives one instance
    /// from one thread, so a racing cache (today the NTT twiddle map, tomorrow a device
    /// scratch buffer in a GPU backend) would only ever show up in production.
    #[test]
    fn one_prepared_circuit_proves_concurrently() {
        for_each_artifact("one_prepared_circuit_proves_concurrently", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);
            let circuit = circuit.as_ref();
            let (witness, vk, public) = (&witness, &vk, &public);

            std::thread::scope(|scope| {
                let threads: Vec<_> = (0..4u8)
                    .map(|seed| {
                        scope.spawn(move || {
                            let mut t = StageTimings::default();
                            let mut rng = StdRng::from_seed([seed; 32]);
                            let proof = prove(circuit, witness, &mut rng, &mut t).unwrap();
                            verify(vk, public, &proof).unwrap();
                        })
                    })
                    .collect();
                for t in threads {
                    t.join().unwrap();
                }
            });
        });
    }

    /// The public signals in `public.json` must be exactly the witness entries snarkjs
    /// publishes, otherwise `end_to_end` could pass against a self-consistent but wrong
    /// public vector.
    #[test]
    fn public_json_matches_the_witness_prefix() {
        for_each_artifact("public_json_matches_the_witness_prefix", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let public = public_inputs(&a.dir);
            assert_eq!(public.len(), circuit.n_public());
            assert_eq!(&witness[1..=circuit.n_public()], &public[..]);
        });
    }

    #[test]
    fn verifier_rejects_a_flipped_public_input() {
        for_each_artifact("verifier_rejects_a_flipped_public_input", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let mut t = StageTimings::default();
            let mut rng = StdRng::from_seed([9u8; 32]);
            let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).unwrap();

            let mut public = public_inputs(&a.dir);
            public[0] += Fr::one();
            assert!(matches!(
                verify(&vk, &public, &proof),
                Err(VerifyError::PairingFailed)
            ));

            // A short vector is a shape error, not a pairing failure, and the two must
            // not be conflated: one is a caller bug, the other is a rejected proof.
            assert!(matches!(
                verify(&vk, &public[..public.len() - 1], &proof),
                Err(VerifyError::PublicInputCount { .. })
            ));
        });
    }

    #[test]
    fn verifier_rejects_a_negated_a() {
        for_each_artifact("verifier_rejects_a_negated_a", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);
            let mut t = StageTimings::default();
            let mut rng = StdRng::from_seed([11u8; 32]);
            let mut proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).unwrap();
            verify(&vk, &public, &proof).unwrap();

            proof.a = -proof.a;
            assert!(matches!(
                verify(&vk, &public, &proof),
                Err(VerifyError::PairingFailed)
            ));
        });
    }

    #[test]
    fn blinders_are_deterministic() {
        for_each_artifact("blinders_are_deterministic", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let (r, s) = (Fr::from(12345u64), Fr::from(67890u64));
            let mut t = StageTimings::default();
            let p1 = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let p2 = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            assert_eq!((p1.a, p1.b, p1.c), (p2.a, p2.b, p2.c));

            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            verify(&vk, &public_inputs(&a.dir), &p1).unwrap();
        });
    }

    /// The whole comparison rests on this: two runs of the same code over the same witness
    /// must produce byte-identical traces. If they do not, a difference between a laptop and
    /// a phone means nothing, because the laptop already disagrees with itself.
    ///
    /// It also verifies the proof the trace carries. A trace whose proof does not verify is
    /// a trace of something that is not a proof, and every row in it would be a comparison
    /// between two wrong answers.
    #[test]
    fn a_trace_is_reproducible_and_its_proof_verifies() {
        for_each_artifact("a_trace_is_reproducible", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let mut t = StageTimings::default();
            let t1 = prove_trace(circuit.as_ref(), &witness, &mut t).unwrap();
            let t2 = prove_trace(circuit.as_ref(), &witness, &mut t).unwrap();
            assert!(crate::trace::diff(&t1, &t2).is_empty());
            assert_eq!(t1.to_text(), t2.to_text());
            assert_eq!(t1.to_json(), t2.to_json());

            // Not an empty trace dressed as a matching one: the H chunks are the part that
            // would silently vanish if `compute_h` ever stopped handing back host memory.
            let text = t1.to_text();
            assert!(text.contains("h_chunk[00]"), "{text}");
            assert!(!text.contains("msm_a_g1        infinity"), "{text}");

            let (r, s) = crate::trace::blinders();
            let proof = prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).unwrap();
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            verify(&vk, &public_inputs(&a.dir), &proof).unwrap();
        });
    }

    /// A trace that could not tell two different executions apart would report "no
    /// difference" for the bug it exists to find.
    #[test]
    fn a_trace_reports_the_stage_that_differs() {
        for_each_artifact("a_trace_reports_the_stage_that_differs", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let mut t = StageTimings::default();
            let good = prove_trace(circuit.as_ref(), &witness, &mut t).unwrap();

            // One wire changed, which is the smallest thing a miscomputed stage can look
            // like. Not the public prefix: that would move the shape block too and the
            // point here is that a matching shape does not imply a matching execution.
            let mut bent = witness.clone();
            let last = bent.len() - 1;
            bent[last] += Fr::one();
            let bad = prove_trace(circuit.as_ref(), &bent, &mut t).unwrap();

            // Section 8 covers every private wire, so the L MSM is the one row that has to
            // move whichever wire this is, and C carries L into the proof. The rest depend
            // on whether that wire appears in the A, B or C matrices, and on a real circuit
            // most of them do not: the last wire of `railgun-01x01` leaves `h_digest`
            // untouched. Asserting on the first differing row would be asserting on the
            // fixture rather than on the trace.
            let names: Vec<&str> = crate::trace::diff(&good, &bad)
                .iter()
                .map(|(n, _, _)| *n)
                .collect();
            assert!(names.contains(&"msm_l_g1"), "{names:?}");
            assert!(names.contains(&"proof_c"), "{names:?}");
        });
    }

    /// r and s must actually randomise the proof. A prover that ignored them would still
    /// pass every positive test above, and would leak the witness.
    #[test]
    fn distinct_blinders_give_distinct_proofs_that_both_verify() {
        for_each_artifact("distinct_blinders_give_distinct_proofs", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);

            let mut t = StageTimings::default();
            let mut rng1 = StdRng::from_seed([1u8; 32]);
            let mut rng2 = StdRng::from_seed([2u8; 32]);
            let p1 = prove(circuit.as_ref(), &witness, &mut rng1, &mut t).unwrap();
            let p2 = prove(circuit.as_ref(), &witness, &mut rng2, &mut t).unwrap();

            assert_ne!(p1.a, p2.a);
            assert_ne!(p1.b, p2.b);
            assert_ne!(p1.c, p2.c);
            verify(&vk, &public, &p1).unwrap();
            verify(&vk, &public, &p2).unwrap();
        });
    }

    /// Zero blinders make our proof a deterministic function of the witness, and the H
    /// evaluations are then the only thing standing between the MSMs and the pairing
    /// check. This is the test that fails loudly if the coset or the Z convention is
    /// wrong, because with r = s = 0 nothing masks the error.
    #[test]
    fn zero_blinders_still_verify() {
        for_each_artifact("zero_blinders_still_verify", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let mut t = StageTimings::default();
            let proof =
                prove_with_blinders(circuit.as_ref(), &witness, Fr::zero(), Fr::zero(), &mut t)
                    .unwrap();
            verify(&vk, &public_inputs(&a.dir), &proof).unwrap();
        });
    }

    /// `H` is one scalar per section 9 base, and it must not be all zeros: a degenerate
    /// H would make the H MSM the identity, at which point every later test would pass
    /// while proving nothing about stages 1-4.
    #[test]
    fn h_has_one_nonzero_scalar_per_h_query_base() {
        for_each_artifact("h_has_one_nonzero_scalar_per_h_query_base", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let mut t = StageTimings::default();
            let h = circuit.compute_h(&witness, &mut t).unwrap();
            assert_eq!(h.len(), circuit.domain_size());
            assert_eq!(h.len(), circuit.key().h_query.len());
            assert!(h.to_host().unwrap().iter().any(|x| !x.is_zero()));
            // Stage timings have to be filled in, otherwise `bench` reports a free prover.
            assert!(t.gather_us > 0 || t.ntt_us > 0 || t.pointwise_us > 0);
        });
    }

    #[test]
    fn witness_length_is_checked() {
        for_each_artifact("witness_length_is_checked", |a| {
            let (circuit, witness) = prepared(&a.dir);
            let mut t = StageTimings::default();
            let short = &witness[..witness.len() - 1];
            assert!(matches!(
                circuit.compute_h(short, &mut t),
                Err(ProveError::WitnessLength { .. })
            ));
        });
    }

    #[test]
    fn aggregate_public_is_the_ic_combination() {
        for_each_artifact("aggregate_public_is_the_ic_combination", |a| {
            let vk = VerifyingKey::from_json(&a.dir.join("vkey.json")).unwrap();
            let public = public_inputs(&a.dir);
            let got = aggregate_public(&vk, &public).unwrap();

            let mut want = vk.ic[0].into_group();
            for i in 0..public.len() {
                want += vk.ic[i + 1] * public[i];
            }
            assert_eq!(got, want);
            assert!(aggregate_public(&vk, &public[..public.len() - 1]).is_err());
        });
    }
}
