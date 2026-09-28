//! Which words of H come back wrong under load, and what do they hold?
//!
//! BUG-28's reproducer (`independent_audit::distinct_witnesses_concurrently_do_not_cross_talk`
//! beside a 2^22 proof loop) stops at the first thread whose `to_host` refuses a
//! non-canonical H. This runs the same eight-thread, eight-witness loop and, for every
//! result that differs from the CPU backend's, prints which entries differ, what they
//! hold (zero, another witness's H at the same index, this witness's H in Montgomery
//! form, canonical but unknown, non-canonical), how they cluster (index ranges, byte
//! offset of the first, its phase within a 16 KB page), which of the five MSM points
//! differ, whether those are on the curve and whether a second `msms` over the same H
//! agrees, and whether the words changed again between `compute_h` and the end of
//! `msms`. Every line carries a unix timestamp so it can be set beside `log show`.
//!
//! `cargo run --release -p g16-metal --example h_probe -- [--threads 8] [--reps 3]
//! [--rounds 1] [--only csp/keccak] [--max-log 20] [--no-msm]`

#[cfg(target_os = "macos")]
mod probe {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use g16_core::{cpu::CpuBackend, Backend, MsmOutputs, PreparedCircuit, StageTimings};
    use g16_field::Fr;
    use g16_metal::layout::{PackedFr, PackedScalar, FR_MODULUS};
    use g16_metal::stages::{HHandle, TAG};
    use g16_metal::MetalBackend;
    use g16_zkey::{wtns::Witness, ProvingKey};

    const PAGE: usize = 16 << 10;

    fn artifacts() -> Vec<(String, PathBuf)> {
        fn walk(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<(String, PathBuf)>) {
            for d in std::fs::read_dir(dir).expect("read_dir").flatten() {
                let d = d.path();
                if !d.is_dir() {
                    continue;
                }
                let name = format!("{prefix}{}", d.file_name().unwrap().to_string_lossy());
                if name == "large" {
                    continue;
                }
                if d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file() {
                    out.push((name, d));
                } else if depth > 0 {
                    walk(&d, &format!("{name}/"), depth - 1, out);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/artifacts")
            .canonicalize()
            .expect("bench/artifacts");
        let mut out = Vec::new();
        walk(&root, "", 1, &mut out);
        out.sort();
        out
    }

    fn stamp() -> String {
        let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        format!("{}.{:03}", d.as_secs(), d.subsec_millis())
    }

    fn canonical(v: &[u32; 8]) -> bool {
        for i in (0..8).rev() {
            if v[i] != FR_MODULUS[i] {
                return v[i] < FR_MODULUS[i];
            }
        }
        false
    }

    fn hex(v: &[u32; 8]) -> String {
        v.iter()
            .rev()
            .map(|w| format!("{w:08x}"))
            .collect::<Vec<_>>()
            .join("")
    }

    /// The words of `buf`, copied. Nothing is in flight against it: `compute_h` waited.
    fn words(buf: &metal::Buffer, n: usize) -> Vec<[u32; 8]> {
        unsafe { core::slice::from_raw_parts(buf.contents() as *const [u32; 8], n) }.to_vec()
    }

    struct Reference {
        std: Vec<[u32; 8]>,
        mont: Vec<[u32; 8]>,
        msm: MsmOutputs,
    }

    fn msm_diff(got: &MsmOutputs, want: &MsmOutputs) -> Vec<&'static str> {
        let mut d = Vec::new();
        if got.a_g1 != want.a_g1 {
            d.push("A");
        }
        if got.b_g2 != want.b_g2 {
            d.push("B_G2");
        }
        if got.b_g1 != want.b_g1 {
            d.push("B_G1");
        }
        if got.l_g1 != want.l_g1 {
            d.push("L");
        }
        if got.h_g1 != want.h_g1 {
            d.push("H");
        }
        d
    }

    /// Which of the five points satisfy their curve equation.
    fn on_curve(m: &MsmOutputs) -> [bool; 5] {
        use g16_field::CurveGroup;
        [
            m.a_g1.into_affine().is_on_curve(),
            m.b_g2.into_affine().is_on_curve(),
            m.b_g1.into_affine().is_on_curve(),
            m.l_g1.into_affine().is_on_curve(),
            m.h_g1.into_affine().is_on_curve(),
        ]
    }

    /// Everything about one wrong H, as text.
    fn describe(got: &[[u32; 8]], i: usize, refs: &[Reference], base: usize) -> String {
        let want = &refs[i].std;
        let mut wrong = Vec::new();
        let (mut zero, mut noncanon, mut mont, mut other) = (0, 0, 0, 0);
        let mut xtalk = vec![0usize; refs.len()];
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            if g == w {
                continue;
            }
            wrong.push(k);
            if *g == [0u32; 8] {
                zero += 1;
            } else if !canonical(g) {
                noncanon += 1;
            } else if *g == refs[i].mont[k] {
                mont += 1;
            } else {
                let mut hit = false;
                for (j, r) in refs.iter().enumerate() {
                    if j != i && r.std[k] == *g {
                        xtalk[j] += 1;
                        hit = true;
                    }
                }
                if !hit {
                    other += 1;
                }
            }
        }
        if wrong.is_empty() {
            return String::new();
        }
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &k in &wrong {
            match ranges.last_mut() {
                Some((_, hi)) if *hi + 1 == k => *hi = k,
                _ => ranges.push((k, k)),
            }
        }
        let shown: Vec<String> = ranges
            .iter()
            .take(16)
            .map(|(lo, hi)| {
                if lo == hi {
                    format!("{lo}")
                } else {
                    format!("{lo}..={hi}")
                }
            })
            .collect();
        let first = wrong[0];
        let off = first * 32;
        let mut s = format!(
            "    wrong {} of {} entries: zero {zero}, non-canonical {noncanon}, montgomery-of-own {mont}, \
             other-canonical {other}, cross-talk {:?}\n    {} ranges: {}{}\n    first wrong at index {first}, \
             byte offset {off} (page phase {}), buffer base page phase {}, span {}..={}\n",
            wrong.len(),
            got.len(),
            xtalk.iter().enumerate().filter(|(_, c)| **c > 0).collect::<Vec<_>>(),
            ranges.len(),
            shown.join(" "),
            if ranges.len() > 16 { " ..." } else { "" },
            off % PAGE,
            base % PAGE,
            wrong[0],
            wrong[wrong.len() - 1]
        );
        for &k in wrong.iter().take(4) {
            s.push_str(&format!(
                "    [{k}] got {} want {}\n",
                hex(&got[k]),
                hex(&want[k])
            ));
        }
        s
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut threads = 8usize;
        let mut reps = 3usize;
        let mut rounds = 1usize;
        let mut only: Option<String> = None;
        let mut max_log = 30u32;
        let mut do_msm = true;
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--threads" => threads = it.next().unwrap().parse().unwrap(),
                "--reps" => reps = it.next().unwrap().parse().unwrap(),
                "--rounds" => rounds = it.next().unwrap().parse().unwrap(),
                "--only" => only = Some(it.next().unwrap().clone()),
                "--max-log" => max_log = it.next().unwrap().parse().unwrap(),
                "--no-msm" => do_msm = false,
                other => panic!("unknown argument {other}"),
            }
        }

        let metal = MetalBackend::new().unwrap();
        let cpu = CpuBackend::new();
        let out = Mutex::new(());
        let wrong_total = AtomicUsize::new(0);
        let total = AtomicUsize::new(0);
        eprintln!(
            "[{}] h_probe: threads {threads} reps {reps} rounds {rounds} msm {do_msm}",
            stamp()
        );

        for (name, dir) in artifacts() {
            if let Some(f) = &only {
                if !name.contains(f.as_str()) {
                    continue;
                }
            }
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
            if pk.domain_size.trailing_zeros() > max_log {
                continue;
            }
            let base = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
            let n = pk.domain_size;
            let c = metal.prepare(pk).unwrap();
            let ref_c = cpu
                .prepare(ProvingKey::load(&dir.join("circuit.zkey")).unwrap())
                .unwrap();

            let mut inputs: Vec<Vec<Fr>> = vec![base.clone()];
            for k in 1..threads.max(2) as u64 {
                let mut w = base.clone();
                let i = (k as usize * 7 + 1) % w.len();
                w[i] += Fr::from(k * 1_000_003);
                inputs.push(w);
            }

            // Ground truth from the CPU, which no GPU event can touch.
            let t0 = Instant::now();
            let refs: Vec<Reference> = inputs
                .iter()
                .map(|w| {
                    let mut t = StageTimings::default();
                    let h = ref_c.compute_h(w, &mut t).unwrap();
                    let hv = h.to_host().unwrap();
                    let msm = ref_c.msms(w, &h, &mut t).unwrap();
                    Reference {
                        std: hv.iter().map(|x| PackedScalar::from_fr(x).v).collect(),
                        mont: hv.iter().map(|x| PackedFr::from_fr(x).v).collect(),
                        msm,
                    }
                })
                .collect();
            eprintln!(
                "[{}] {name}: domain 2^{}, {} cpu references in {:.1} s",
                stamp(),
                n.trailing_zeros(),
                refs.len(),
                t0.elapsed().as_secs_f64()
            );

            let c: &dyn PreparedCircuit = c.as_ref();
            let (name, inputs, refs, out) = (&name, &inputs, &refs, &out);
            let (wrong_total, total) = (&wrong_total, &total);
            for round in 0..rounds {
                let round_wrong = AtomicUsize::new(0);
                let round_wrong = &round_wrong;
                std::thread::scope(|s| {
                    for i in 0..threads {
                        let thread = std::thread::Builder::new().name(format!("{name} t{i}"));
                        thread
                            .spawn_scoped(s, move || {
                                for rep in 0..reps {
                                    let w = &inputs[i];
                                    let mut t = StageTimings::default();
                                    let t0 = Instant::now();
                                    let h = match c.compute_h(w, &mut t) {
                                        Ok(h) => h,
                                        Err(e) => {
                                            let _g = out.lock().unwrap();
                                            eprintln!("[{}] {name} t{i} rep {rep}: compute_h ERR {e}", stamp());
                                            continue;
                                        }
                                    };
                                    let h_ms = t0.elapsed().as_secs_f64() * 1e3;
                                    let handle = h.device_handle::<HHandle>(TAG).unwrap();
                                    let base = handle.h_std().contents() as usize;
                                    let raw1 = words(handle.h_std(), n);
                                    let to_host_ok = handle.to_host().is_some();
                                    let mut msm_text = String::new();
                                    let mut msm_wrong = false;
                                    let mut raw2_changed = 0usize;
                                    if do_msm {
                                        let t1 = Instant::now();
                                        match c.msms(w, &h, &mut t) {
                                            Ok(m) => {
                                                let d = msm_diff(&m, &refs[i].msm);
                                                msm_wrong = !d.is_empty();
                                                msm_text = format!(
                                                    "msms {:.1} ms, differs {:?}",
                                                    t1.elapsed().as_secs_f64() * 1e3,
                                                    d
                                                );
                                                if msm_wrong {
                                                    // Is the wrong point even a curve point, and
                                                    // does the same H give the right answer a
                                                    // moment later?
                                                    msm_text.push_str(&format!(
                                                        ", on-curve {:?}",
                                                        on_curve(&m)
                                                    ));
                                                    match c.msms(w, &h, &mut t) {
                                                        Ok(m2) => msm_text.push_str(&format!(
                                                            ", rerun differs {:?}",
                                                            msm_diff(&m2, &refs[i].msm)
                                                        )),
                                                        Err(e) => msm_text
                                                            .push_str(&format!(", rerun ERR {e}")),
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                msm_wrong = true;
                                                msm_text = format!("msms ERR {e}");
                                            }
                                        }
                                        let raw2 = words(handle.h_std(), n);
                                        raw2_changed = raw1.iter().zip(&raw2).filter(|(a, b)| a != b).count();
                                    }
                                    drop(h);
                                    total.fetch_add(1, Ordering::Relaxed);
                                    let text = describe(&raw1, i, refs, base);
                                    let h_wrong = !text.is_empty();
                                    if h_wrong || msm_wrong || !to_host_ok {
                                        wrong_total.fetch_add(1, Ordering::Relaxed);
                                        round_wrong.fetch_add(1, Ordering::Relaxed);
                                        let _g = out.lock().unwrap();
                                        eprintln!(
                                            "[{}] WRONG {name} t{i} rep {rep} round {round}: compute_h {h_ms:.1} ms, to_host {}, {msm_text}, words changed after msms {raw2_changed}\n{text}",
                                            stamp(),
                                            if to_host_ok { "ok" } else { "NON-CANONICAL" },
                                        );
                                    }
                                }
                            })
                            .unwrap();
                    }
                });
                eprintln!(
                    "[{}] ROUND {name} {round}: {} of {} results wrong",
                    stamp(),
                    round_wrong.load(Ordering::Relaxed),
                    threads * reps
                );
            }
        }
        eprintln!(
            "[{}] TOTAL: {} of {} results wrong",
            stamp(),
            wrong_total.load(Ordering::Relaxed),
            total.load(Ordering::Relaxed)
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    probe::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("h_probe needs macOS");
}
