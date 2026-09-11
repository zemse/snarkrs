//! Independent audit: the Metal backend against the CPU backend, stage by stage, on the
//! real proving path (device-resident H feeding stage 9), for every artifact.
//!
//! Added by a review pass. Not part of the crate's own suite; delete freely.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use g16_core::{
    cpu::CpuBackend, prove::prove_with_blinders, verify::verify, Backend, HPoly, PreparedCircuit,
    ProveError, StageTimings,
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

/// H element by element, then the five MSM points after normalisation, on the real path.
#[test]
fn every_stage_of_the_metal_backend_matches_the_cpu() {
    let found = artifacts();
    assert!(!found.is_empty(), "no artifacts");
    let metal = MetalBackend::new().expect("Metal device");

    let mut h_total = 0usize;
    let mut pt_total = 0usize;
    for (name, dir) in &found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let cpu = CpuBackend::new().prepare(pk).unwrap();

        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let gpu = metal.prepare(pk).unwrap();
        assert_eq!(gpu.backend_name(), "metal");

        let mut tc = StageTimings::default();
        let mut tg = StageTimings::default();
        let hc = cpu.compute_h(&witness, &mut tc).unwrap();
        let hg = gpu.compute_h(&witness, &mut tg).unwrap();

        // The GPU value must genuinely be device resident, not a host vector some
        // fallback produced.
        assert!(
            matches!(hg, HPoly::Device { .. }),
            "{name}: metal compute_h returned a host vector"
        );
        let handle = hg
            .device_handle::<HHandle>(TAG)
            .expect("{name}: not a metal handle");

        let want = hc.to_host().unwrap();
        let got_mont = handle.to_host();
        let got_std = handle
            .to_host_std()
            .expect("h_std holds a non-canonical residue");
        assert_eq!(got_mont.len(), want.len(), "{name}: H length");

        let mut mont_ok = 0usize;
        let mut std_ok = 0usize;
        for i in 0..want.len() {
            if got_mont[i] == want[i] {
                mont_ok += 1;
            }
            if got_std[i] == want[i] {
                std_ok += 1;
            }
        }
        println!(
            "H  {name:<14} montgomery {mont_ok}/{}  standard {std_ok}/{}",
            want.len(),
            want.len()
        );
        assert_eq!(mont_ok, want.len(), "{name}: H montgomery mismatch");
        assert_eq!(std_ok, want.len(), "{name}: H standard mismatch");
        h_total += mont_ok + std_ok;

        // Five MSMs. CPU takes the CPU H; the GPU takes its own device-resident H, so
        // this is the production path and not a rewired one.
        let mc = cpu.msms(&witness, &hc, &mut tc).unwrap();
        let mg = gpu.msms(&witness, &hg, &mut tg).unwrap();
        let pairs: [(&str, bool); 5] = [
            ("A  -> G1", mg.a_g1.into_affine() == mc.a_g1.into_affine()),
            ("B  -> G2", mg.b_g2.into_affine() == mc.b_g2.into_affine()),
            ("B  -> G1", mg.b_g1.into_affine() == mc.b_g1.into_affine()),
            ("L  -> G1", mg.l_g1.into_affine() == mc.l_g1.into_affine()),
            ("H  -> G1", mg.h_g1.into_affine() == mc.h_g1.into_affine()),
        ];
        let ok = pairs.iter().filter(|(_, b)| *b).count();
        println!("MSM {name:<14} {ok}/5 points equal after normalisation");
        for (label, b) in pairs {
            assert!(b, "{name}: MSM {label} differs");
        }
        pt_total += ok;

        // And the assembled proofs must be identical, blinders fixed.
        let pc = prove_with_blinders(
            cpu.as_ref(),
            &witness,
            Fr::from(7u64),
            Fr::from(9u64),
            &mut tc,
        )
        .unwrap();
        let pg = prove_with_blinders(
            gpu.as_ref(),
            &witness,
            Fr::from(7u64),
            Fr::from(9u64),
            &mut tg,
        )
        .unwrap();
        assert_eq!(
            (pc.a, pc.b, pc.c),
            (pg.a, pg.b, pg.c),
            "{name}: proof differs"
        );
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let public = witness[1..=gpu.n_public()].to_vec();
        verify(&vk, &public, &pg).unwrap();
        println!("PROOF {name:<14} byte-identical to the CPU proof, and verifies");
    }
    println!("TOTAL  H elements matched {h_total}, MSM points matched {pt_total}");
}

/// Same witness twice on one circuit: the H buffer, the five points and the proof must
/// all be identical. Pooled scratch that leaked state between proofs would show here.
#[test]
fn proving_the_same_witness_twice_is_stable() {
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let c = metal.prepare(pk).unwrap();
        let mut t = StageTimings::default();

        let h1 = c.compute_h(&witness, &mut t).unwrap();
        let v1 = h1.device_handle::<HHandle>(TAG).unwrap().to_host();
        let m1 = c.msms(&witness, &h1, &mut t).unwrap();
        drop(h1);
        let h2 = c.compute_h(&witness, &mut t).unwrap();
        let v2 = h2.device_handle::<HHandle>(TAG).unwrap().to_host();
        let m2 = c.msms(&witness, &h2, &mut t).unwrap();
        drop(h2);

        assert_eq!(v1, v2, "{name}: H differs between two runs");
        assert_eq!(m1.a_g1, m2.a_g1, "{name}: A differs");
        assert_eq!(m1.b_g2, m2.b_g2, "{name}: B G2 differs");
        assert_eq!(m1.b_g1, m2.b_g1, "{name}: B G1 differs");
        assert_eq!(m1.l_g1, m2.l_g1, "{name}: L differs");
        assert_eq!(m1.h_g1, m2.h_g1, "{name}: H differs");
        println!("STABLE {name}");
    }
}

/// Every wrong witness length, not just one short. Long, short by many, empty.
#[test]
fn wrong_witness_lengths_are_all_rejected() {
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let c = metal.prepare(pk).unwrap();
        let n = witness.len();
        let mut t = StageTimings::default();

        let mut longer = witness.clone();
        longer.push(Fr::from(1u64));
        let cases: Vec<(&str, Vec<Fr>)> = vec![
            ("empty", Vec::new()),
            ("short by 1", witness[..n - 1].to_vec()),
            ("half", witness[..n / 2].to_vec()),
            ("long by 1", longer),
        ];
        for (label, w) in &cases {
            let r = c.compute_h(w, &mut t);
            assert!(
                matches!(r, Err(ProveError::WitnessLength { .. })),
                "{name}/{label}: compute_h did not reject"
            );
        }
        // Good H, then a bad witness into msms.
        let h = c.compute_h(&witness, &mut t).unwrap();
        for (label, w) in &cases {
            let r = c.msms(w, &h, &mut t);
            assert!(
                matches!(r, Err(ProveError::WitnessLength { .. })),
                "{name}/{label}: msms did not reject"
            );
        }
        // Still usable.
        c.msms(&witness, &h, &mut t).unwrap();
        println!("REJECTED {name}: all four bad lengths, circuit still usable");
    }
}

/// Eight threads on one circuit, distinct blinders, every proof verified. The contract
/// promises Send + Sync; a shared scratch buffer would show up as a proof that does not
/// verify rather than as a crash.
#[test]
fn eight_concurrent_proofs_on_one_circuit() {
    let metal = MetalBackend::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let c = metal.prepare(pk).unwrap();
        let public = witness[1..=c.n_public()].to_vec();
        let c: &dyn PreparedCircuit = c.as_ref();
        let (witness, vk, public) = (&witness, &vk, &public);

        // A single-threaded reference with fixed blinders, so a concurrent run can be
        // compared against a known-good answer and not only against "it verified".
        let mut t = StageTimings::default();
        let want = prove_with_blinders(c, witness, Fr::from(5u64), Fr::from(6u64), &mut t).unwrap();

        std::thread::scope(|s| {
            let hs: Vec<_> = (0..8u64)
                .map(|_| {
                    s.spawn(move || {
                        let mut t = StageTimings::default();
                        let p =
                            prove_with_blinders(c, witness, Fr::from(5u64), Fr::from(6u64), &mut t)
                                .unwrap();
                        verify(vk, public, &p).unwrap();
                        (p.a, p.b, p.c)
                    })
                })
                .collect();
            for h in hs {
                let got = h.join().unwrap();
                assert_eq!(
                    got,
                    (want.a, want.b, want.c),
                    "{name}: concurrent proof differs"
                );
            }
        });
        println!("CONCURRENT {name}: 8 threads, all identical to the serial proof");
    }
}

/// How much of the reported `msm_us` is host work rather than GPU work.
///
/// `msms` charges four things to `msm_us`: packing the witness into standard form on the
/// host (`upload_scalars`), one extra command buffer converting the device H out of
/// Montgomery form, the batched MSM command buffer, and the host-side Horner combine
/// over the window sums. Only the third and fourth are what a "GPU MSM" number usually
/// means.
#[test]
fn where_the_msm_time_actually_goes() {
    use g16_metal::msm::{Job, JobG1, JobG2, MetalMsm};
    use std::time::Instant;

    let msm = MetalMsm::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let a = msm.upload_g1_bases(&pk.a_query);
        let b1 = msm.upload_g1_bases(&pk.b_g1_query);
        let b2 = msm.upload_g2_bases(&pk.b_g2_query);
        let l = msm.upload_g1_bases(&pk.l_query);
        let hq = msm.upload_g1_bases(&pk.h_query);

        let cpu = CpuBackend::new().prepare(pk).unwrap();
        let mut t = StageTimings::default();
        let h = cpu.compute_h(&witness, &mut t).unwrap();
        let h_host = h.to_host().unwrap().to_vec();
        let n_public = cpu.n_public();
        let n = witness.len();
        let l_len = cpu.key().l_query.len();

        // warm up the pools
        let mut pack_ms = Vec::new();
        let mut batch_ms = Vec::new();
        for rep in 0..9 {
            let s = Instant::now();
            let w = msm.upload_scalars(&witness);
            let hs = msm.upload_scalars(&h_host);
            let p = s.elapsed().as_secs_f64() * 1e3;

            let s = Instant::now();
            let jobs = [
                Job::G1(JobG1 {
                    bases: &a,
                    base_off: 0,
                    scalars: &w,
                    scalar_off: 0,
                    n,
                }),
                Job::G2(JobG2 {
                    bases: &b2,
                    base_off: 0,
                    scalars: &w,
                    scalar_off: 0,
                    n,
                }),
                Job::G1(JobG1 {
                    bases: &b1,
                    base_off: 0,
                    scalars: &w,
                    scalar_off: 0,
                    n,
                }),
                Job::G1(JobG1 {
                    bases: &l,
                    base_off: 0,
                    scalars: &w,
                    scalar_off: n_public + 1,
                    n: l_len,
                }),
                Job::G1(JobG1 {
                    bases: &hq,
                    base_off: 0,
                    scalars: &hs,
                    scalar_off: 0,
                    n: h_host.len(),
                }),
            ];
            let _ = msm.msm_batch(&jobs).unwrap();
            let b = s.elapsed().as_secs_f64() * 1e3;
            if rep >= 3 {
                pack_ms.push(p);
                batch_ms.push(b);
            }
        }
        pack_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        batch_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let p = pack_ms[pack_ms.len() / 2];
        let b = batch_ms[batch_ms.len() / 2];
        println!(
            "SPLIT {name:<14} host scalar pack {p:7.2} ms   msm_batch {b:7.2} ms   host share {:.1}%",
            100.0 * p / (p + b)
        );
    }
}

/// Two different circuits, different domain sizes, proving concurrently through one
/// `MetalBackend`. They share the compiled pipelines, the command queues and, crucially,
/// the MSM scratch pool, which is the object a cross-circuit race would corrupt.
#[test]
fn two_circuits_share_one_backend_under_load() {
    let metal = MetalBackend::new().unwrap();
    let found = artifacts();
    let mut loaded = Vec::new();
    for (name, dir) in &found {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let vk = VerifyingKey::from_json(&dir.join("vkey.json")).unwrap();
        let c = metal.prepare(pk).unwrap();
        let public = witness[1..=c.n_public()].to_vec();
        let mut t = StageTimings::default();
        let want =
            prove_with_blinders(c.as_ref(), &witness, Fr::from(3u64), Fr::from(4u64), &mut t)
                .unwrap();
        loaded.push((name.clone(), c, witness, vk, public, want));
    }

    std::thread::scope(|s| {
        let hs: Vec<_> = loaded
            .iter()
            .flat_map(|item| {
                (0..3).map(move |_| {
                    s.spawn(move || {
                        let (name, c, witness, vk, public, want) = item;
                        for _ in 0..3 {
                            let mut t = StageTimings::default();
                            let p = prove_with_blinders(
                                c.as_ref(),
                                witness,
                                Fr::from(3u64),
                                Fr::from(4u64),
                                &mut t,
                            )
                            .unwrap();
                            verify(vk, public, &p).unwrap();
                            assert_eq!(
                                (p.a, p.b, p.c),
                                (want.a, want.b, want.c),
                                "{name}: proof drifted under cross-circuit load"
                            );
                        }
                    })
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    });
    println!(
        "CROSS-CIRCUIT {} circuits x 3 threads x 3 proofs, all identical and all verify",
        loaded.len()
    );
}

/// What the duplicated H conversion costs.
///
/// Stage 4 already writes H in standard form (`h_std`), but `MetalCircuit::msms` cannot
/// wrap a foreign buffer in a `ScalarBuf`, so it recomputes the same values from
/// `h_mont` with `fr_mont_to_std` in a command buffer of its own. This measures that
/// second conversion; the first one's cost is the extra `n * 32` bytes stage 4 writes
/// and is not separable without editing the shader.
#[test]
fn the_second_h_conversion_costs_this_much() {
    use g16_metal::msm::MetalMsm;
    use g16_metal::stages::HStages;
    use std::time::Instant;

    let stages = HStages::new().unwrap();
    let msm = MetalMsm::new().unwrap();
    for (name, dir) in artifacts() {
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let res = stages.prepare(&pk).unwrap();
        let mut t = StageTimings::default();

        let mut conv = Vec::new();
        let mut whole = Vec::new();
        for rep in 0..15 {
            let s = Instant::now();
            let h = res.compute_h(&stages, &witness, &mut t).unwrap();
            let w = s.elapsed().as_secs_f64() * 1e3;
            let handle = h.device_handle::<HHandle>(TAG).unwrap();
            let s = Instant::now();
            let _ = msm
                .scalars_from_device_mont(handle.h_mont(), handle.len())
                .unwrap();
            let c = s.elapsed().as_secs_f64() * 1e3;
            drop(h);
            if rep >= 5 {
                conv.push(c);
                whole.push(w);
            }
        }
        conv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        whole.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "CONV {name:<14} stages 0-4 {:6.3} ms   redundant mont->std {:6.3} ms",
            whole[whole.len() / 2],
            conv[conv.len() / 2]
        );
    }
}

/// Adversarial scalar vectors through the GPU MSM, against arkworks' own
/// `VariableBaseMSM` as an independent oracle (not our Pippenger, so a shared bug in the
/// CPU backend cannot hide here).
///
/// The shapes are chosen to hit the places where the host and the kernels have to agree:
/// the zero/one classification that sizes `cap`, the `ones` kernel, the signed-digit
/// borrow at the top window, and a bucket fat enough to span many slices.
#[test]
fn adversarial_scalars_agree_with_arkworks() {
    use ark_ec::VariableBaseMSM;
    use ark_ff::{BigInteger, PrimeField};
    use g16_metal::msm::MetalMsm;

    let (_, dir) = artifacts()
        .into_iter()
        .find(|(n, _)| n == "js_8x8_d32")
        .expect("js_8x8_d32");
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
    let msm = MetalMsm::new().unwrap();

    let n = 4096usize;
    let bases: Vec<g16_field::G1Affine> = pk.a_query[..n].to_vec();
    let gb = msm.upload_g1_bases(&bases);

    // A tiny deterministic xorshift; no rand dependency in this crate.
    let mut st = 0x243F6A8885A308D3u64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let rand_fr = |x: u64, y: u64| {
        let mut b = [0u8; 32];
        b[..8].copy_from_slice(&x.to_le_bytes());
        b[8..16].copy_from_slice(&y.to_le_bytes());
        Fr::from_le_bytes_mod_order(&b)
    };

    let minus_one = -Fr::from(1u64);
    let r_minus_two = minus_one - Fr::from(1u64);
    // r - 1 has its top bits set, which is what forces the borrow in the top window.
    let top = Fr::from_le_bytes_mod_order(&minus_one.into_bigint().to_bytes_le());

    let mut cases: Vec<(&str, Vec<Fr>)> = vec![
        ("all zero", vec![Fr::from(0u64); n]),
        ("all one", vec![Fr::from(1u64); n]),
        ("all -1", vec![minus_one; n]),
        ("all r-2", vec![r_minus_two; n]),
        ("top-window borrow", vec![top; n]),
        ("single general in ones", {
            let mut v = vec![Fr::from(1u64); n];
            v[n / 2] = minus_one;
            v
        }),
        ("single general in zeros", {
            let mut v = vec![Fr::from(0u64); n];
            v[7] = rand_fr(0xDEAD, 0xBEEF);
            v
        }),
        ("one element", vec![minus_one]),
        ("two elements", vec![Fr::from(1u64), minus_one]),
        ("random", (0..n).map(|_| rand_fr(next(), next())).collect()),
        ("mostly zero, some random", {
            (0..n)
                .map(|i| {
                    if i % 97 == 0 {
                        rand_fr(next(), next())
                    } else {
                        Fr::from(0u64)
                    }
                })
                .collect()
        }),
        ("one fat bucket", {
            // Every scalar the same general value, so one bucket per window holds all n.
            vec![rand_fr(0x1234, 0x5678); n]
        }),
        (
            "small values",
            (0..n).map(|i| Fr::from((i % 5) as u64)).collect(),
        ),
        (
            // All bits set in a short scalar: every bounded window borrows, so the
            // recoding's carry out of the per-upload top window (see `Plan::new`'s
            // bit bound) must be provably zero, not merely usually zero.
            "bounded-window borrow",
            vec![Fr::from((1u64 << 35) - 1); n],
        ),
        ("word-sized values", {
            (0..n).map(|_| Fr::from(next() & 0xFFFF_FFFF)).collect()
        }),
    ];
    // Five more random draws, so the fixed shapes are not the whole sample.
    for _ in 0..5 {
        cases.push((
            "random draw",
            (0..n).map(|_| rand_fr(next(), next())).collect::<Vec<_>>(),
        ));
    }

    let mut ok = 0usize;
    for (label, scalars) in &cases {
        let m = scalars.len();
        let want = g16_field::G1Projective::msm(&bases[..m], scalars).unwrap();
        let sb = msm.upload_scalars(scalars);
        let got = msm
            .msm_g1(&gb, &sb)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        // msm_g1 uses min(bases, scalars) so the short cases work.
        assert_eq!(
            got.into_affine(),
            want.into_affine(),
            "{label}: GPU MSM disagrees with arkworks"
        );
        ok += 1;
        println!("ADVERSARIAL ok  {label} (n={m})");
    }
    println!(
        "ADVERSARIAL {ok}/{} shapes matched arkworks exactly",
        cases.len()
    );
}
