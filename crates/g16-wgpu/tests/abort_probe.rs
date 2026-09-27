//! BUG-33's real half: stages 0 to 4 looped under another process's GPU load until macOS
//! aborts one, then the retried `H` against the CPU's and the proof it finishes through the
//! verifier, with the pools as fresh as `g16 prove`'s are.
//!
//! Ignored, because on a quiet GPU nothing aborts and it proves nothing. Run it beside two
//! proof loops (the 2a2babe lane's `load.sh`), under the cross-process lock so no timing
//! test lands inside the load:
//!
//! ```text
//! ITERS=40 lockf -k "$TMPDIR/g16-gpu-timing.lock" \
//!   cargo test --release -p g16-wgpu --test abort_probe -- --ignored --nocapture
//! ```
//!
//! Every iteration prepares the circuit again, as a `g16 prove` process does, so the
//! scratch stages 0 to 4 allocate after an abort and the MSM batch's are as new as the
//! CLI's; then it runs `compute_h` until one attempt is refused (or `TRIES` have run), reads
//! `H` back and compares it with the CPU's, runs the MSMs over it, assembles the proof at
//! the trace blinders, compares it with the CPU proof at the same blinders and verifies it.
//! One line per iteration says which of those held. `ARTIFACT` picks the key, default
//! `railgun-13x01`, the one the CLI runs used.

use std::path::Path;
use std::sync::Arc;

use g16_core::cpu::CpuCircuit;
use g16_core::prove::{assemble_trace, prove_with_blinders};
use g16_core::verify::verify;
use g16_core::{Backend, PreparedCircuit, StageTimings};
use g16_wgpu::backend::WgpuProver;
use g16_wgpu::{LimitsProfile, WgpuBackend};
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "needs another process loading the GPU; see the module docs"]
fn stages_0_to_4_retried_after_a_real_abort_are_exact() {
    let iters = env_or("ITERS", 20);
    let tries = env_or("TRIES", 200);
    let name = std::env::var("ARTIFACT").unwrap_or_else(|_| "railgun-13x01".into());
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .join(&name);
    if !dir.join("circuit.zkey").is_file() {
        eprintln!("SKIPPED stages_0_to_4_retried_after_a_real_abort_are_exact: no {name}");
        return;
    }
    let witness = Witness::load(&dir.join("circuit.wtns"))
        .expect("circuit.wtns")
        .0;
    let vkey = VerifyingKey::from_json(&dir.join("vkey.json")).expect("vkey.json");
    let (r, s) = g16_core::trace::blinders();

    // The oracle: the CPU's H and the CPU's proof at the trace blinders.
    let cpu = CpuCircuit::new(ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey"))
        .expect("cpu circuit");
    let mut t = StageTimings::default();
    let cpu_h = cpu
        .compute_h(&witness, &mut t)
        .expect("cpu compute_h")
        .to_host()
        .expect("the cpu H is a host vector")
        .to_vec();
    let cpu_h_poly = cpu.compute_h(&witness, &mut t).expect("cpu compute_h");
    let cpu_m = cpu.msms(&witness, &cpu_h_poly, &mut t).expect("cpu msms");
    let cpu_proof = prove_with_blinders(&cpu, &witness, r, s, &mut t).expect("cpu proof");
    let public = witness[1..=cpu.n_public()].to_vec();
    verify(&vkey, &public, &cpu_proof).expect("the cpu proof verifies");

    let device = Arc::new(
        pollster::block_on(WgpuBackend::with_profile(LimitsProfile::Floor))
            .expect("no wgpu device at the Floor profile"),
    );
    let prover = WgpuProver::with_device(Arc::clone(&device)).expect("wgpu prover");

    let (mut aborted, mut h_wrong, mut proof_wrong, mut proof_differs) = (0, 0, 0, 0);
    for i in 0..iters {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("zkey");
        let circuit = prover.prepare(pk).expect("prepare");
        let mut t = StageTimings::default();
        let mut attempts = 0u64;
        let mut ran = 0usize;
        let mut h = None;
        while ran < tries {
            ran += 1;
            let before = device.submits();
            match circuit.compute_h(&witness, &mut t) {
                Ok(v) => {
                    attempts = device.submits() - before;
                    h = Some(v);
                }
                Err(e) => {
                    eprintln!("iter {i}: compute_h {ran} failed: {e}");
                    continue;
                }
            }
            if attempts > 1 {
                break;
            }
        }
        let Some(h) = h else {
            eprintln!("iter {i}: no H after {ran} tries");
            continue;
        };
        // Which entries a wrong H gets wrong says what left them: a half-run transform
        // wrongs a structured subset, a kernel that never ran leaves zeros.
        let got_h = circuit.h_to_host(&h).expect("a wgpu H reads back");
        let bad: Vec<usize> = (0..cpu_h.len().min(got_h.len()))
            .filter(|&i| got_h[i] != cpu_h[i])
            .collect();
        let h_ok = bad.is_empty() && got_h.len() == cpu_h.len();
        let h_report = if h_ok {
            "ok".to_string()
        } else {
            format!(
                "WRONG: {} of {} entries, first {:?}, zeros among them {}",
                bad.len(),
                cpu_h.len(),
                bad.first(),
                bad.iter()
                    .filter(|&&i| got_h[i] == g16_field::Fr::from(0u64))
                    .count()
            )
        };
        let plain = prover.msm().last_submits();
        let m = match circuit.msms(&witness, &h, &mut t) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("iter {i}: msms failed: {e}");
                continue;
            }
        };
        let msm_submits = prover.msm().last_submits();
        let proof = assemble_trace(circuit.key(), &m, &mut t);
        let same = (proof.a, proof.b, proof.c) == (cpu_proof.a, cpu_proof.b, cpu_proof.c);
        let wrong_msms: Vec<&str> = [
            ("A", m.a_g1 == cpu_m.a_g1),
            ("B-G2", m.b_g2 == cpu_m.b_g2),
            ("B-G1", m.b_g1 == cpu_m.b_g1),
            ("L", m.l_g1 == cpu_m.l_g1),
            ("H", m.h_g1 == cpu_m.h_g1),
        ]
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(n, _)| *n)
        .collect();
        let verified = verify(&vkey, &public, &proof).is_ok();
        if attempts > 1 {
            aborted += 1;
            h_wrong += usize::from(!h_ok);
            proof_wrong += usize::from(!verified);
            proof_differs += usize::from(!same);
        }
        eprintln!(
            "iter {i}: compute_h ran {ran} times, the last took {attempts} attempt(s); \
             H {h_report}; msm submits {msm_submits} (plain {plain}), wrong MSMs \
             {wrong_msms:?}; proof {}, {}",
            if same {
                "same as cpu"
            } else {
                "DIFFERS from cpu"
            },
            if verified {
                "verifies"
            } else {
                "does NOT verify"
            },
        );
    }
    eprintln!(
        "stage 0-4 aborts retried: {aborted}; H wrong after one: {h_wrong}; proof differing \
         from the cpu's after one: {proof_differs}; proof not verifying after one: {proof_wrong}"
    );
    assert_eq!(h_wrong, 0, "H wrong after a retried stage 0-4 abort");
    assert_eq!(
        proof_wrong, 0,
        "proof wrong after a retried stage 0-4 abort"
    );
}
