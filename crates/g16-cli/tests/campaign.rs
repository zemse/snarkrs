//! A proving campaign: many proofs, many statements, one question — does the verifier
//! ever reject something it should have accepted?
//!
//! `roundtrip.rs` proves each artifact once through the binary. Once is enough to catch a
//! wrong constant, and useless against anything that depends on the blinders, on which
//! witness went in, or on a circuit having been proved before. Three classes of bug hide
//! from a single proof and show up here:
//!
//!   * blinder-dependent breakage. `r` and `s` are fresh per proof, so a stage-11
//!     transcription error that happens to cancel for one lucky pair (`r = 0`, `s = 0`,
//!     `r = s`) survives a one-shot test and dies against a few dozen random pairs.
//!   * statement-dependent breakage. Every artifact ships exactly one witness, so a
//!     prover could be wrong for every input except the one on disk and still pass.
//!     `tiny_mul`'s constraint system is `out = a*b*c`, which means a valid witness can
//!     be written down in Rust for any `(a, b, c)` and hundreds of genuinely different
//!     statements can be proved without a witness generator.
//!   * state-dependent breakage. `PreparedCircuit` documents itself as safe to prove with
//!     concurrently from `&self`; the campaign reuses one instance for every proof of a
//!     variant, and one test hammers it from several threads at once.
//!
//! Everything here asserts *verification*, never proof equality, except where the
//! blinders are pinned on purpose. Two proofs of the same statement are supposed to
//! differ — that is the zero knowledge — so a separate assertion demands that they do.
//! A prover that dropped `r` and `s` on the floor would still verify.

use std::path::{Path, PathBuf};

use ark_serialize::CanonicalSerialize;
use ark_std::rand::{rngs::StdRng, thread_rng, SeedableRng};
use g16_cli::artifacts::{variants, Variant};
use g16_cli::{make_backend, BackendKind};
use g16_core::prove::{prove, prove_with_blinders};
use g16_core::verify::verify;
use g16_core::{PreparedCircuit, Proof, StageTimings};
use g16_field::{Fr, PrimeField, UniformRand};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

fn artifacts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts")
}

/// Every variant that is fully generated, or an empty list. Missing artifacts are a skip
/// with a loud message rather than a pass, so a fresh clone cannot report false green.
fn found(test: &str) -> Vec<Variant> {
    let v = variants(artifacts_root());
    if v.is_empty() {
        eprintln!("SKIPPED {test}: no complete artifacts under bench/artifacts");
    }
    v
}

/// How many proofs to spend on a circuit of this domain size.
///
/// Proving cost is roughly linear in the domain, so the budget per variant is held
/// roughly flat: the small circuits get enough repetitions to actually sample the blinder
/// space, and the two big ones get enough to be more than a smoke test while keeping the
/// default run inside a couple of minutes in release mode. The heavy counts live in the
/// `#[ignore]`d test at the bottom.
fn reps_for(domain_size: usize) -> usize {
    match domain_size {
        0..=2048 => 40,
        2049..=16_384 => 24,
        16_385..=65_536 => 16,
        65_537..=262_144 => 8,
        _ => 4,
    }
}

struct Loaded {
    circuit: Box<dyn PreparedCircuit>,
    vk: VerifyingKey,
    witness: Vec<Fr>,
    public: Vec<Fr>,
}

/// Load a variant onto a backend and settle, once, where the public inputs come from.
///
/// The proof is checked against `witness[1..=n_public]` rather than against the shipped
/// `public.json`, because that slice is what the prover's own L-query split treats as
/// public — if the two disagreed, every proof in the campaign would verify against a
/// statement nobody asserted. So they are compared here, once per variant, and the rest
/// of the file uses the witness slice.
fn load(v: &Variant, backend: BackendKind) -> Loaded {
    let pk = ProvingKey::load(&v.zkey()).unwrap_or_else(|e| panic!("{}: zkey: {e}", v.name));
    let witness = Witness::load(&v.wtns())
        .unwrap_or_else(|e| panic!("{}: wtns: {e}", v.name))
        .0;
    let vk = VerifyingKey::from_json(&v.vkey()).unwrap_or_else(|e| panic!("{}: vkey: {e}", v.name));
    let circuit = make_backend(backend)
        .unwrap_or_else(|e| panic!("{}: backend: {e}", v.name))
        .prepare(pk)
        .unwrap_or_else(|e| panic!("{}: prepare: {e}", v.name));

    let public = witness[1..=circuit.n_public()].to_vec();
    let from_json = public_json(&v.dir.join("public.json"));
    assert_eq!(
        public, from_json,
        "{}: witness[1..=n_public] is not the statement snarkjs published",
        v.name
    );
    Loaded {
        circuit,
        vk,
        witness,
        public,
    }
}

fn public_json(path: &Path) -> Vec<Fr> {
    let text = std::fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let n: num_bigint::BigUint = s.as_str().unwrap().parse().unwrap();
            Fr::from_le_bytes_mod_order(&n.to_bytes_le())
        })
        .collect()
}

fn bytes(p: &Proof) -> Vec<u8> {
    let mut out = Vec::new();
    p.a.serialize_compressed(&mut out).unwrap();
    p.b.serialize_compressed(&mut out).unwrap();
    p.c.serialize_compressed(&mut out).unwrap();
    out
}

/// One proof with fresh OS-seeded blinders, verified. Panics with the proof index so a
/// failure at repetition 37 of 40 says which one.
fn prove_and_verify(l: &Loaded, label: &str, i: usize) -> Proof {
    let mut t = StageTimings::default();
    let mut rng = thread_rng();
    let proof = prove(l.circuit.as_ref(), &l.witness, &mut rng, &mut t)
        .unwrap_or_else(|e| panic!("{label}: proof {i} failed to build: {e}"));
    verify(&l.vk, &l.public, &proof)
        .unwrap_or_else(|e| panic!("{label}: proof {i} did not verify: {e}"));
    proof
}

/// The main campaign: every variant, many proofs each, every one verified.
///
/// Also the zero-knowledge sanity check. Each proof is compared against the previous
/// one's bytes, and every pair must differ in all three group elements. A prover that
/// ignored `r` and `s` — or sampled them from a broken RNG that returned a constant —
/// would produce identical proofs that all verify perfectly, which is the failure mode a
/// verification-only campaign cannot see.
#[test]
fn every_variant_survives_a_blinder_sweep() {
    let found = found("every_variant_survives_a_blinder_sweep");
    for v in &found {
        let l = load(v, BackendKind::Cpu);
        let n = reps_for(l.circuit.domain_size());
        eprintln!(
            "{}: domain {}, {} vars, {} public -> {n} proofs on cpu",
            v.name,
            l.circuit.domain_size(),
            l.circuit.n_vars(),
            l.circuit.n_public()
        );

        let mut seen: Vec<Vec<u8>> = Vec::with_capacity(n);
        for i in 0..n {
            let proof = prove_and_verify(&l, &v.name, i);

            // Against every earlier proof, not just the previous one: a prover that
            // cycled through a short list of blinders would beat a neighbours-only check.
            let b = bytes(&proof);
            assert!(
                !seen.contains(&b),
                "{}: proof {i} is byte-identical to an earlier proof of the same \
                 statement — the blinders are not being applied, so this prover is not \
                 zero knowledge",
                v.name
            );
            if let Some(prev) = seen.last() {
                assert_ne!(prev, &b, "{}: proof {i} repeats proof {}", v.name, i - 1);
            }
            seen.push(b);
        }

        // The three elements must each move, not just the concatenation. `pi_c` absorbs
        // both blinders, `pi_a` only `r` and `pi_b` only `s`, so a dropped `s` would still
        // change the bytes through `pi_c` while leaving `pi_b` pinned.
        let a: Vec<_> = (0..n).map(|i| &seen[i][..32]).collect();
        assert_eq!(
            a.iter().collect::<std::collections::BTreeSet<_>>().len(),
            n,
            "{}: pi_a repeated across proofs — r is not random",
            v.name
        );
    }
}

/// Distinct *statements*, not just distinct blinders.
///
/// Every artifact ships one witness, so the whole suite could be passing on a prover that
/// is correct for exactly one input. `tiny_mul` is `t = a*b; out = t*c` with `[a, b]`
/// public alongside the output, so its satisfying assignment is a closed form:
/// `w = [1, a*b*c, a, b, c, a*b]`. That makes hundreds of genuinely different statements
/// available without circom, snarkjs, or a wasm witness calculator in the test path.
///
/// The layout is asserted against the shipped witness before any of it is trusted, so if
/// `gen-artifacts.sh` ever changes the circuit this test says the layout moved rather
/// than quietly proving nonsense.
#[test]
fn tiny_mul_proves_hundreds_of_different_statements() {
    let found = found("tiny_mul_proves_hundreds_of_different_statements");
    let Some(v) = found.iter().find(|v| v.name == "tiny_mul") else {
        eprintln!("SKIPPED tiny_mul_proves_hundreds_of_different_statements: no tiny_mul");
        return;
    };
    let l = load(v, BackendKind::Cpu);

    // Confirm the assignment really is [1, out, a, b, c, t] before generating any.
    assert_eq!(l.circuit.n_vars(), 6, "tiny_mul layout changed");
    assert_eq!(l.circuit.n_public(), 3, "tiny_mul layout changed");
    let (w, s) = (&l.witness, |i: usize| l.witness[i]);
    assert_eq!(w.len(), 6);
    assert_eq!(s(5), s(2) * s(3), "w[5] is not a*b");
    assert_eq!(s(1), s(5) * s(4), "w[1] is not t*c");

    // Seeded, so a failure names a reproducible triple rather than a lost one.
    let mut rng = StdRng::from_seed([0x5a; 32]);
    let mut publics: Vec<Vec<Fr>> = Vec::new();
    let n = 200;
    for i in 0..n {
        let (a, b, c) = (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng));
        let t = a * b;
        let out = t * c;
        let witness = vec![Fr::from(1u64), out, a, b, c, t];
        let public = vec![out, a, b];

        let mut timings = StageTimings::default();
        let mut blinders = thread_rng();
        let proof = prove(l.circuit.as_ref(), &witness, &mut blinders, &mut timings)
            .unwrap_or_else(|e| panic!("tiny_mul statement {i}: prove failed: {e}"));
        verify(&l.vk, &public, &proof).unwrap_or_else(|e| {
            panic!("tiny_mul statement {i} (a={a}, b={b}, c={c}) did not verify: {e}")
        });
        publics.push(public);
    }
    // Guard against the generator accidentally producing the same statement twice, which
    // would silently reduce this to a blinder sweep.
    let mut sorted: Vec<_> = publics.iter().map(|p| format!("{p:?}")).collect();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), n, "the statement generator repeated itself");
}

/// The other half of "the verifier never fails": it also has to fail when it should.
///
/// A campaign that only ever proves true statements cannot distinguish a correct verifier
/// from `fn verify(..) -> Ok(())`. Two ways to be wrong are checked against `tiny_mul`:
/// an unsatisfying assignment (the intermediate wire `t` is corrupted, so the public
/// inputs stay perfectly valid and only the constraint system is violated), and a
/// perfectly good proof checked against a different statement.
#[test]
fn tiny_mul_rejects_what_it_should() {
    let found = found("tiny_mul_rejects_what_it_should");
    let Some(v) = found.iter().find(|v| v.name == "tiny_mul") else {
        eprintln!("SKIPPED tiny_mul_rejects_what_it_should: no tiny_mul");
        return;
    };
    let l = load(v, BackendKind::Cpu);
    let mut rng = StdRng::from_seed([0xa5; 32]);

    for i in 0..8 {
        let (a, b, c) = (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng));
        let (t, out) = (a * b, a * b * c);
        let public = vec![out, a, b];

        // Bad witness, honest public inputs. The R1CS is unsatisfied, so H is not
        // divisible by Z and the pairing check must fail.
        let bogus = vec![Fr::from(1u64), out, a, b, c, t + Fr::from(1u64)];
        let mut timings = StageTimings::default();
        let proof = prove(l.circuit.as_ref(), &bogus, &mut thread_rng(), &mut timings)
            .expect("proving an unsatisfying witness should still produce a Proof value");
        assert!(
            verify(&l.vk, &public, &proof).is_err(),
            "round {i}: verifier accepted a proof built from an unsatisfying witness"
        );

        // Honest witness, wrong statement.
        let honest = vec![Fr::from(1u64), out, a, b, c, t];
        let mut timings = StageTimings::default();
        let good = prove(l.circuit.as_ref(), &honest, &mut thread_rng(), &mut timings).unwrap();
        verify(&l.vk, &public, &good).expect("the honest control did not verify");
        let other = vec![out + Fr::from(1u64), a, b];
        assert!(
            verify(&l.vk, &other, &good).is_err(),
            "round {i}: verifier accepted a proof against the wrong public inputs"
        );
    }
}

/// `PreparedCircuit` claims to be safe to prove with concurrently from `&self`. If it is
/// not, the symptom is a proof that simply fails to verify with nothing else to go on —
/// exactly the bug a serial campaign would never reproduce and a user would hit under
/// load. Every thread shares one prepared circuit and every proof must verify.
#[test]
fn one_prepared_circuit_serves_concurrent_provers() {
    let found = found("one_prepared_circuit_serves_concurrent_provers");
    for v in found.iter().filter(|v| {
        // The small circuits only; this is a race hunt, not a throughput measurement.
        matches!(v.name.as_str(), "tiny_mul" | "js_1x1_d8" | "js_2x2_d16")
    }) {
        let l = load(v, BackendKind::Cpu);
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
        eprintln!("{}: {threads} threads x 6 proofs", v.name);

        let all: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|tid| {
                    let l = &l;
                    scope.spawn(move || {
                        (0..6)
                            .map(|i| bytes(&prove_and_verify(l, &format!("{tid}"), i)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        let mut uniq = all.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            all.len(),
            "{}: two concurrent proofs came out identical",
            v.name
        );
    }
}

/// Same witness, two backends, both proofs verify — and with the blinders pinned, the two
/// backends must agree bit for bit.
///
/// The verification-only half is the real test: an accelerator that is subtly wrong in
/// stage 4 produces a proof that fails the pairing check, and nothing else reports it.
/// The bit-for-bit half is the sharper one where it applies, because it localises a
/// disagreement to the backend rather than to the assembly, but it only means anything
/// with `r` and `s` fixed: `prove` samples them per call, so the two proofs are supposed
/// to differ.
#[cfg(feature = "metal")]
#[test]
fn metal_and_cpu_agree_on_every_variant() {
    let found = found("metal_and_cpu_agree_on_every_variant");
    for v in &found {
        let cpu = load(v, BackendKind::Cpu);
        let gpu = load(v, BackendKind::Metal);
        assert_eq!(gpu.circuit.backend_name(), "metal");
        let n = (reps_for(cpu.circuit.domain_size()) / 4).max(2);
        eprintln!("{}: {n} proofs on each of cpu and metal", v.name);

        for i in 0..n {
            let a = prove_and_verify(&cpu, &format!("{} cpu", v.name), i);
            let b = prove_and_verify(&gpu, &format!("{} metal", v.name), i);
            // Different blinders on each side, so they must NOT match. Equality here
            // would mean the blinders are not reaching the proof.
            assert_ne!(
                bytes(&a),
                bytes(&b),
                "{}: cpu and metal produced the same proof under independent blinders",
                v.name
            );
        }

        // Now pin the blinders and demand exact agreement.
        let mut rng = StdRng::from_seed([0x11; 32]);
        for i in 0..n {
            let (r, s) = (Fr::rand(&mut rng), Fr::rand(&mut rng));
            let mut t = StageTimings::default();
            let a = prove_with_blinders(cpu.circuit.as_ref(), &cpu.witness, r, s, &mut t).unwrap();
            let mut t = StageTimings::default();
            let b = prove_with_blinders(gpu.circuit.as_ref(), &gpu.witness, r, s, &mut t).unwrap();
            verify(&cpu.vk, &cpu.public, &a).unwrap_or_else(|e| panic!("{} cpu {i}: {e}", v.name));
            verify(&gpu.vk, &gpu.public, &b)
                .unwrap_or_else(|e| panic!("{} metal {i}: {e}", v.name));
            assert_eq!(
                bytes(&a),
                bytes(&b),
                "{}: cpu and metal disagree at fixed blinders (round {i})",
                v.name
            );
        }
    }
}

/// Pinned blinders reproduce, on one backend. Cheap, and it is the assumption the
/// cross-backend test above rests on: if `prove_with_blinders` were not actually
/// deterministic, that test would be comparing noise to noise.
#[test]
fn fixed_blinders_reproduce_exactly() {
    let found = found("fixed_blinders_reproduce_exactly");
    for v in found.iter().take(3) {
        let l = load(v, BackendKind::Cpu);
        let mut rng = StdRng::from_seed([0x33; 32]);
        for i in 0..4 {
            let (r, s) = (Fr::rand(&mut rng), Fr::rand(&mut rng));
            let mut t = StageTimings::default();
            let a = prove_with_blinders(l.circuit.as_ref(), &l.witness, r, s, &mut t).unwrap();
            let mut t = StageTimings::default();
            let b = prove_with_blinders(l.circuit.as_ref(), &l.witness, r, s, &mut t).unwrap();
            assert_eq!(
                bytes(&a),
                bytes(&b),
                "{}: round {i} did not reproduce",
                v.name
            );
            verify(&l.vk, &l.public, &a).unwrap_or_else(|e| panic!("{}: {e}", v.name));
        }
    }
}

/// Degenerate blinders. `r = 0`, `s = 0` and `r = s` are the pairs where a stage-11
/// transcription error cancels, and a random sweep will never draw them: the chance of
/// hitting `r = 0` in `Fr` is about 2^-254. They are the corners most likely to be wrong
/// and the ones least likely to be sampled, so they are enumerated rather than hoped for.
///
/// `r = 0` also strips the only thing hiding `A`, and `s = 0` the only thing hiding `B`.
/// Both still have to verify — they are valid proofs, merely not zero-knowledge ones.
#[test]
fn degenerate_blinders_still_verify() {
    let found = found("degenerate_blinders_still_verify");
    for v in &found {
        let l = load(v, BackendKind::Cpu);
        let one = Fr::from(1u64);
        let k = Fr::from(0x9e3779b97f4a7c15u64);
        let corners: [(Fr, Fr); 7] = [
            (Fr::from(0u64), Fr::from(0u64)),
            (Fr::from(0u64), one),
            (one, Fr::from(0u64)),
            (one, one),
            (k, k),
            (k, -k),
            (-one, -one),
        ];
        for (i, (r, s)) in corners.iter().enumerate() {
            let mut t = StageTimings::default();
            let p = prove_with_blinders(l.circuit.as_ref(), &l.witness, *r, *s, &mut t)
                .unwrap_or_else(|e| panic!("{}: corner {i}: prove: {e}", v.name));
            verify(&l.vk, &l.public, &p).unwrap_or_else(|e| {
                panic!("{}: corner {i} (r={r}, s={s}) did not verify: {e}", v.name)
            });
        }
        eprintln!("{}: 7 degenerate blinder pairs verified", v.name);
    }
}

/// The deep campaign: the same sweep with every variant flattened to the same count, so
/// the two large circuits get as many blinder pairs as the small ones instead of the
/// reduced share the default budget gives them.
///
/// `#[ignore]`d on wall clock, not on principle: measured at 223 seconds for 1200 proofs
/// on an M2 Max, dominated by the 140k-constraint variant. The default sweep above covers
/// the same ground in under a minute and is the one that has to stay in everyone's
/// `cargo test`.
///
///     cargo test --release -p g16-cli --test campaign -- --ignored --nocapture
#[test]
#[ignore = "long: 200 proofs per variant, roughly four minutes in release mode"]
fn deep_campaign_two_hundred_proofs_per_variant() {
    const N: usize = 200;
    let found = found("deep_campaign_two_hundred_proofs_per_variant");
    for v in &found {
        let l = load(v, BackendKind::Cpu);
        eprintln!("{}: {N} proofs", v.name);
        let mut seen: Vec<Vec<u8>> = Vec::with_capacity(N);
        for i in 0..N {
            let b = bytes(&prove_and_verify(&l, &v.name, i));
            assert!(!seen.contains(&b), "{}: proof {i} repeated", v.name);
            seen.push(b);
        }
    }
}
