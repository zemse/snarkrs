//! Independent verification pass. Written by a reviewer, not by the backend author.
//!
//! Three things the crate's own tests do not establish:
//!
//! 1. `distinct_witnesses_concurrently_do_not_cross_talk` - the existing concurrency
//!    test runs the SAME witness with the SAME blinders in all eight threads, so two
//!    threads handed the same scratch buffer would still agree. This one gives every
//!    thread a different witness, so a shared buffer produces a wrong answer.
//! 2. `dump_h_and_points` - writes both backends' H and MSM points to disk so the
//!    comparison can be done outside this crate's assertions.
//! 3. `gpu_output_is_load_bearing` - shows the proof depends on what the GPU wrote, by
//!    scribbling on the device H buffer between compute_h and msms.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use g16_core::{
    cpu::CpuBackend, prove::prove_with_blinders, verify::verify, Backend, PreparedCircuit,
    StageTimings,
};
use g16_field::{CurveGroup, Fr};
use g16_metal::stages::{HHandle, TAG};
use g16_metal::MetalBackend;
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};

fn artifacts() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize()
        .expect("bench/artifacts");
    let mut out: Vec<(String, PathBuf)> = std::fs::read_dir(&root)
        .expect("read_dir")
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

/// Eight threads, eight DIFFERENT witnesses, one circuit. Each thread's answer is
/// compared against that witness's own serially computed answer.
///
/// The perturbed witnesses are not valid R1CS assignments, so the proofs they produce
/// do not verify and are not asked to. What is being tested is that the backend is a
/// function of its input under concurrency, which a shared scratch buffer would break.
#[test]
fn distinct_witnesses_concurrently_do_not_cross_talk() {
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let base = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let c = metal.prepare(pk).unwrap();

        // Eight distinct inputs: the real witness, then seven perturbations at
        // different positions so no two share an H vector.
        let mut inputs: Vec<Vec<Fr>> = vec![base.clone()];
        for k in 1..8u64 {
            let mut w = base.clone();
            let i = (k as usize * 7 + 1) % w.len();
            w[i] += Fr::from(k * 1_000_003);
            inputs.push(w);
        }

        // Serial reference for each.
        let mut want = Vec::new();
        for w in &inputs {
            let mut t = StageTimings::default();
            let h = c.compute_h(w, &mut t).unwrap();
            let hv = h.device_handle::<HHandle>(TAG).unwrap().to_host();
            let m = c.msms(w, &h, &mut t).unwrap();
            drop(h);
            want.push((hv, m.a_g1, m.b_g2, m.b_g1, m.l_g1, m.h_g1));
        }

        let c: &dyn PreparedCircuit = c.as_ref();
        let (inputs, want) = (&inputs, &want);
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..8usize)
                .map(|i| {
                    s.spawn(move || {
                        // A few repetitions so the threads genuinely overlap.
                        for _ in 0..3 {
                            let mut t = StageTimings::default();
                            let w = &inputs[i];
                            let h = c.compute_h(w, &mut t).unwrap();
                            let hv = h.device_handle::<HHandle>(TAG).unwrap().to_host();
                            let m = c.msms(w, &h, &mut t).unwrap();
                            drop(h);
                            let e = &want[i];
                            assert_eq!(hv, e.0, "thread {i}: H differs under concurrency");
                            assert_eq!(m.a_g1, e.1, "thread {i}: A differs");
                            assert_eq!(m.b_g2, e.2, "thread {i}: B G2 differs");
                            assert_eq!(m.b_g1, e.3, "thread {i}: B G1 differs");
                            assert_eq!(m.l_g1, e.4, "thread {i}: L differs");
                            assert_eq!(m.h_g1, e.5, "thread {i}: H MSM differs");
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
        });
        println!("CROSSTALK {name}: 8 distinct witnesses x 3 reps, every result matched its own serial reference");
    }
}

/// Writes both backends' stage outputs to `$G16_DUMP_DIR` so they can be diffed by
/// something that is not this file's `assert_eq!`.
#[test]
fn dump_h_and_points() {
    let Some(out) = std::env::var_os("G16_DUMP_DIR") else {
        eprintln!("SKIPPED dump_h_and_points: set G16_DUMP_DIR");
        return;
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let cpu = CpuBackend::new()
            .prepare(ProvingKey::load(&dir.join("circuit.zkey")).unwrap())
            .unwrap();
        let gpu = metal
            .prepare(ProvingKey::load(&dir.join("circuit.zkey")).unwrap())
            .unwrap();

        for (tag, c) in [("cpu", &cpu), ("metal", &gpu)] {
            let mut t = StageTimings::default();
            let h = c.compute_h(&witness, &mut t).unwrap();
            let hv = match h.device_handle::<HHandle>(TAG) {
                Some(d) => d.to_host(),
                None => h.to_host().unwrap().to_vec(),
            };
            let mut s = String::new();
            for x in &hv {
                s.push_str(&format!("{x}\n"));
            }
            std::fs::write(out.join(format!("{name}.{tag}.h.txt")), s).unwrap();

            let m = c.msms(&witness, &h, &mut t).unwrap();
            let pts = format!(
                "A {:?}\nBG2 {:?}\nBG1 {:?}\nL {:?}\nH {:?}\n",
                m.a_g1.into_affine(),
                m.b_g2.into_affine(),
                m.b_g1.into_affine(),
                m.l_g1.into_affine(),
                m.h_g1.into_affine(),
            );
            std::fs::write(out.join(format!("{name}.{tag}.pts.txt")), pts).unwrap();
        }
        println!("DUMPED {name}");
    }
}

/// If the H the GPU wrote were not what stage 9 reads, corrupting it would change
/// nothing. It must break the proof.
#[test]
fn gpu_output_is_load_bearing() {
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let c = metal.prepare(pk).unwrap();
        let public = witness[1..=c.n_public()].to_vec();
        let mut t = StageTimings::default();

        // Clean run first, to be sure the artifact is good, and to record the honest
        // H MSM before anything is touched.
        let good =
            prove_with_blinders(c.as_ref(), &witness, Fr::from(3u64), Fr::from(4u64), &mut t)
                .unwrap();
        verify(&vk, &public, &good).unwrap();
        let want_h_msm = {
            let h = c.compute_h(&witness, &mut t).unwrap();
            let m = c.msms(&witness, &h, &mut t).unwrap();
            m.h_g1
        };

        // Now scribble one bit on the H buffer the MSM reads. Note this is `h_mont`,
        // not `h_std`: `MetalCircuit::msms` re-derives standard form from the Montgomery
        // copy with its own `fr_mont_to_std` dispatch, so stage 4's `h_std` output is
        // never read on the proving path. Corrupting `h_std` changes nothing, which was
        // measured before this line was written the way it is.
        let h = c.compute_h(&witness, &mut t).unwrap();
        {
            let handle = h.device_handle::<HHandle>(TAG).unwrap();
            let buf = handle.h_mont();
            // SAFETY: shared storage, no GPU work in flight against this scratch, and a
            // u32 has no invalid bit patterns.
            unsafe {
                let p = buf.contents() as *mut u32;
                *p ^= 1;
            }
        }
        let m = c.msms(&witness, &h, &mut t).unwrap();
        drop(h);
        assert!(
            m.h_g1 != want_h_msm,
            "{name}: corrupting the device H changed nothing, so stage 9 is not reading it"
        );
        println!("LOAD-BEARING {name}: flipping one bit of the device H buffer changed the H MSM");
    }
}
