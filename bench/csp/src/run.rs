//! The measurement loop.
//!
//! Upstream drives this with criterion at `sample_size(10)` and publishes the mean point
//! estimate. Criterion reaches that mean through a ramp of 1, 2, ... 10 iterations per
//! sample plus a three second warm-up, which is 55 proofs and change for one number. At
//! keccak_2048's ten seconds a proof that is an hour and a half for a single cell of the
//! table, and there are fifteen cells. So the statistic is reproduced and the ramp is
//! not: one warm-up iteration, then `reps` timed ones, and the mean of those is what goes
//! in the row. Median, min and max go in the breakdown next to it, because a mean of ten
//! on a laptop is one background process away from being wrong and the spread is the only
//! way to see that happen.

use crate::metrics::{Breakdown, Metrics, Properties};
use crate::{circuits, inputs, metrics, KeyPolicy, Prover, Variant};
use anyhow::{Context, Result};
use snarkrs_field::Fr;
use snarkrs_formats::{wtns::Witness, ProvingKey};
use snarkrs_groth16::StageTimings;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct Config<'a> {
    pub artifacts: &'a Path,
    pub witness_src: &'a Path,
    pub prover: Prover,
    pub reps: usize,
    pub out_dir: &'a Path,
    /// Peak-RSS samples per variant, each its own process. Zero leaves `peak_memory` at
    /// the collector's 0 marker, which is what a timing-only rerun wants.
    pub mem_reps: usize,
    /// The single-shot prover `mem_reps` runs. Beside this binary unless overridden.
    pub mem_bin: &'a Path,
    /// A `snarkjs` to check each warm-up proof with, outside the timed region. None skips it.
    pub snarkjs: Option<&'a Path>,
    /// Load a key with `gamma_g2 == delta_g2` under [`KeyPolicy::Unblinded`] rather than
    /// refuse it. Off by default.
    pub allow_unblinded: bool,
}

/// One timed iteration, split by phase.
#[derive(Default, Clone, Copy)]
struct Split {
    witness: Duration,
    zkey_load: Duration,
    prepare: Duration,
    prove: Duration,
    stages: StageTimings,
}

impl Split {
    fn total(&self) -> Duration {
        self.witness + self.zkey_load + self.prepare + self.prove
    }
}

/// The constraint count the upstream row carries, derived the way `ark-circom` derives
/// it: the highest constraint index section 4 mentions, less the public signals that
/// snarkjs appended as constraints of their own. It is one below the count `snarkjs r1cs
/// info` prints, and matching the collector matters more than matching snarkjs.
pub fn num_constraints(pk: &ProvingKey) -> usize {
    let mut max_index = 0usize;
    for m in 0..2 {
        let row_ptr = &pk.coeffs.row_ptr[m];
        for c in (0..pk.domain_size).rev() {
            if row_ptr[c + 1] > row_ptr[c] {
                max_index = max_index.max(c);
                break;
            }
        }
    }
    max_index.saturating_sub(pk.n_public)
}

/// The public signals snarkjs would publish for this witness.
fn public_of(witness: &[Fr], n_public: usize) -> Result<Vec<Fr>> {
    anyhow::ensure!(
        witness.len() > n_public,
        "witness has {} entries, too short for {n_public} public signals",
        witness.len()
    );
    Ok(witness[1..=n_public].to_vec())
}

/// One full proof, timed by phase. Everything here is inside upstream's timed region.
fn proof_iteration(
    variant: Variant,
    input_json: String,
    zkey: &Path,
    prover: &Prover,
    key: KeyPolicy,
) -> Result<(Split, snarkrs_groth16::Proof, Vec<Fr>)> {
    let mut split = Split::default();

    let t = Instant::now();
    let wtns = circuits::witness(variant, input_json)?;
    split.witness = t.elapsed();

    match prover {
        Prover::G16(backend) => {
            let t = Instant::now();
            let (pk, _) = crate::load_key(zkey, key == KeyPolicy::Unblinded)?;
            let n_public = pk.n_public;
            split.zkey_load = t.elapsed();

            // The witness parse is charged to the prepare phase rather than to witness
            // generation: it is our deserialiser, not the generator's, and upstream pays
            // the same cost in `parse_bigints_to_witness` after the thread join.
            let t = Instant::now();
            let witness = Witness::from_bytes(wtns)
                .with_context(|| format!("{}: parsing generated witness", variant.name()))?
                .0;
            let public = public_of(&witness, n_public)?;
            let circuit = crate::make_backend(*backend)?.prepare(pk)?;
            split.prepare = t.elapsed();

            let t = Instant::now();
            let proof = key.prove(circuit.as_ref(), &witness, &mut split.stages)?;
            split.prove = t.elapsed();
            Ok((split, proof, public))
        }
        Prover::Rapidsnark(bin) => {
            // The CLI takes paths, so the witness has to reach the disk. That write is not
            // a cost the in-process binding pays, so it is charged to `prepare` and kept
            // out of `prove` rather than folded in silently. `zkey_load` stays zero: the
            // read happens inside the subprocess and there is no way to see it from here.
            let t = Instant::now();
            let dir = tempdir(variant)?;
            let wtns_path = dir.join("circuit.wtns");
            std::fs::write(&wtns_path, &wtns)?;
            split.prepare = t.elapsed();

            let proof_path = dir.join("proof.json");
            let public_path = dir.join("public.json");
            let t = Instant::now();
            let out = std::process::Command::new(bin)
                .arg(zkey)
                .arg(&wtns_path)
                .arg(&proof_path)
                .arg(&public_path)
                .output()
                .with_context(|| format!("running {}", bin.display()))?;
            split.prove = t.elapsed();
            anyhow::ensure!(
                out.status.success(),
                "{} on {}: {}",
                bin.display(),
                variant.name(),
                String::from_utf8_lossy(&out.stderr)
            );

            let proof =
                snarkrs_groth16::json::proof_from_str(&std::fs::read_to_string(&proof_path)?)?;
            let public =
                snarkrs_groth16::json::public_from_str(&std::fs::read_to_string(&public_path)?)?;
            std::fs::remove_dir_all(&dir).ok();
            Ok((split, proof, public))
        }
    }
}

/// A private directory under the system temp dir, named for the variant so two runs of
/// different variants cannot collide and a leftover is obvious.
fn tempdir(variant: Variant) -> Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join(format!(
        "snarkrs-csp-{}-{}",
        variant.name(),
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// `snarkjs groth16 verify` on `proof`, against a vkey snarkjs exports from `zkey` itself
/// and the `public.json` that `snarkrs-csp artifacts` wrote. Our verifier shares our
/// encoders and our zkey decoding, so it cannot see an error made the same way on both
/// sides; snarkjs reads the key and checks the proof with none of our code. A rejected
/// proof is left on disk beside the exported vkey and its path is in the error.
fn snarkjs_verify(
    bin: &Path,
    variant: Variant,
    zkey: &Path,
    artifacts: &Path,
    proof: &snarkrs_groth16::Proof,
) -> Result<()> {
    let public = artifacts.join(variant.name()).join("public.json");
    anyhow::ensure!(
        public.is_file(),
        "{}: {} missing; run snarkrs-csp artifacts",
        variant.name(),
        public.display()
    );
    let tmp = tempdir(variant)?;
    let vkey = tmp.join("vkey.json");
    std::fs::remove_file(&vkey).ok();
    let out = std::process::Command::new(bin)
        .args(["zkey", "export", "verificationkey"])
        .arg(zkey)
        .arg(&vkey)
        .output()
        .with_context(|| format!("{}: running {}", variant.name(), bin.display()))?;
    check_snarkjs_output(&out, None)
        .and_then(|()| check_vkey(&vkey))
        .with_context(|| {
            format!(
                "{}: snarkjs did not export a verification key from {}",
                variant.name(),
                zkey.display()
            )
        })?;

    let proof_path = tmp.join("proof.json");
    std::fs::write(&proof_path, snarkrs_groth16::json::proof_to_string(proof))?;
    let out = std::process::Command::new(bin)
        .args(["groth16", "verify"])
        .arg(&vkey)
        .arg(&public)
        .arg(&proof_path)
        .output()
        .with_context(|| format!("{}: running {}", variant.name(), bin.display()))?;
    // snarkjs 0.7 exits 1 on a reject, but only the `OK!` line says it actually checked.
    check_snarkjs_output(&out, Some("OK!")).with_context(|| {
        format!(
            "{}: snarkjs did not accept {}",
            variant.name(),
            proof_path.display()
        )
    })?;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

/// Success is exit 0, plus `marker` on stdout when the command prints one.
fn check_snarkjs_output(out: &std::process::Output, marker: Option<&str>) -> Result<()> {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    anyhow::ensure!(
        out.status.success() && marker.is_none_or(|m| stdout.contains(m)),
        "snarkjs status: {}; stdout: {}; stderr: {}",
        out.status,
        stdout.trim(),
        stderr.trim()
    );
    Ok(())
}

/// `zkey export verificationkey` prints nothing on success, so the file is the evidence.
fn check_vkey(vkey: &Path) -> Result<()> {
    let text =
        std::fs::read_to_string(vkey).with_context(|| format!("reading {}", vkey.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", vkey.display()))?;
    anyhow::ensure!(
        json["protocol"] == "groth16",
        "{} is not a groth16 verification key",
        vkey.display()
    );
    Ok(())
}

fn mean(xs: &[Duration]) -> Duration {
    Duration::from_nanos((xs.iter().map(|d| d.as_nanos()).sum::<u128>() / xs.len() as u128) as u64)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Benchmark one variant and write its row and breakdown into `cfg.out_dir`.
pub fn variant(variant: Variant, cfg: &Config) -> Result<(Metrics, Breakdown)> {
    let zkey = variant.zkey(cfg.artifacts);
    anyhow::ensure!(
        zkey.is_file(),
        "{} missing; run bench/scripts/csp-fetch.sh",
        zkey.display()
    );
    let input_json = inputs::json(variant.target, variant.input_size);
    let (pk, key) = crate::load_key(&zkey, cfg.allow_unblinded)?;
    if key == KeyPolicy::Unblinded {
        eprintln!(
            "{}: gamma_g2 equals delta_g2, so anyone can forge a proof against this key; \
             measured under --allow-unblinded-zkey and labelled unblinded",
            variant.name()
        );
    }
    let n_constraints = num_constraints(&pk);
    let prover = match &cfg.prover {
        Prover::G16(b) => Prover::G16(b.resolve(n_constraints)),
        p => p.clone(),
    };
    let reload = || crate::load_key(&zkey, key == KeyPolicy::Unblinded).map(|(pk, _)| pk);

    // Warm-up. Its proof is the one that gets verified and measured for size, so a
    // variant whose proof does not verify fails before any timing is reported rather
    // than after.
    let (_, proof, public) = proof_iteration(variant, input_json.clone(), &zkey, &prover, key)?;
    key.verify(&pk.vk, &public, &proof)
        .with_context(|| format!("{}: our own verifier rejected the proof", variant.name()))?;
    if let Some(snarkjs) = cfg.snarkjs {
        snarkjs_verify(snarkjs, variant, &zkey, cfg.artifacts, &proof)?;
    }
    let proof_size = snarkrs_groth16::json::proof_to_string(&proof).len()
        + snarkrs_groth16::json::public_to_string(&public).len();
    drop(pk);

    let mut splits = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let (split, _, _) = proof_iteration(variant, input_json.clone(), &zkey, &prover, key)?;
        splits.push(split);
    }

    // Verification, upstream's way: re-read the key, derive the verifier from it, check.
    // Then again from an already-parsed key, which is the honest cost of a verifier.
    let mut verify_full = Vec::with_capacity(cfg.reps);
    let mut verify_vkey = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let t = Instant::now();
        let pk = reload()?;
        key.verify(&pk.vk, &public, &proof)?;
        verify_full.push(t.elapsed());

        let t = Instant::now();
        key.verify(&pk.vk, &public, &proof)?;
        verify_vkey.push(t.elapsed());
    }

    let totals: Vec<Duration> = splits.iter().map(Split::total).collect();
    let mut sorted = totals.clone();
    sorted.sort();

    let avg = |f: fn(&Split) -> Duration| mean(&splits.iter().map(f).collect::<Vec<_>>());
    let avg_us = |f: fn(&StageTimings) -> u64| {
        splits.iter().map(|s| f(&s.stages)).sum::<u64>() / splits.len() as u64
    };

    let row = Metrics {
        name: cfg.prover.system().to_string(),
        feat: cfg.prover.feat(n_constraints, key),
        target: variant.target.as_str().to_string(),
        input_size: variant.input_size,
        proof_duration: mean(&totals),
        verify_duration: mean(&verify_full),
        cycles: None,
        proof_size,
        preprocessing_size: variant.preprocessing_size(cfg.artifacts, cfg.witness_src)? as usize,
        num_constraints: n_constraints,
        // rapidsnark's peak is not ours to measure through `snarkrs-csp-mem`, which links
        // our prover. Left at the collector's 0 marker rather than reported as a number
        // that came from the wrong process.
        peak_memory: match (&prover, cfg.mem_reps) {
            (_, 0) | (Prover::Rapidsnark(_), _) => 0,
            (Prover::G16(backend), reps) => {
                crate::mem::sample(cfg.mem_bin, variant, *backend, key, cfg.artifacts, reps)?
            }
        },
        acceleration: None,
        properties: Properties::default(),
    };

    let breakdown = Breakdown {
        variant: variant.name(),
        backend: cfg.prover.label(n_constraints, key),
        auto_metal_from: matches!(cfg.prover, Prover::G16(crate::Backend::Auto))
            .then_some(crate::AUTO_METAL_FROM),
        unblinded_zkey: key == KeyPolicy::Unblinded,
        reps: cfg.reps,
        witness_ms: ms(avg(|s| s.witness)),
        zkey_load_ms: ms(avg(|s| s.zkey_load)),
        prepare_ms: ms(avg(|s| s.prepare)),
        prove_ms: ms(avg(|s| s.prove)),
        total_ms: ms(mean(&totals)),
        total_median_ms: ms(sorted[sorted.len() / 2]),
        total_min_ms: ms(sorted[0]),
        total_max_ms: ms(sorted[sorted.len() - 1]),
        verify_vkey_ms: ms(mean(&verify_vkey)),
        gather_us: avg_us(|t| t.gather_us),
        ntt_us: avg_us(|t| t.ntt_us),
        pointwise_us: avg_us(|t| t.pointwise_us),
        msm_us: avg_us(|t| t.msm_us),
        assemble_us: avg_us(|t| t.assemble_us),
    };

    std::fs::create_dir_all(cfg.out_dir)?;
    let row_path = cfg.out_dir.join(metrics::filename(
        variant.target.as_str(),
        variant.input_size,
        cfg.prover.system(),
        row.feat.as_deref(),
    ));
    std::fs::write(&row_path, serde_json::to_string_pretty(&row)?)?;
    let breakdown_path = cfg.out_dir.join(format!(
        "{}_{}_breakdown.json",
        variant.name(),
        breakdown.backend.replace('/', "_")
    ));
    std::fs::write(&breakdown_path, serde_json::to_string_pretty(&breakdown)?)?;

    Ok((row, breakdown))
}

/// The oracle against a shell script standing in for snarkjs. Each test uses its own
/// variant, because `tempdir` is keyed by variant and process and the tests run in parallel.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::Target;
    use snarkrs_field::{AffineRepr, G1Affine, G2Affine};
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    const EXPORT_OK: &str = r#"printf '{"protocol":"groth16"}' > "$5""#;
    /// Accepts only a vkey that the export step wrote, never the artifact directory's.
    const VERIFY_OK: &str = r#"case "$3" in */artifacts/*) exit 9 ;; esac; grep -q groth16 "$3" && echo '[INFO]  snarkJS: OK!'"#;

    fn variant(input_size: usize) -> Variant {
        Variant {
            target: Target::Sha256,
            input_size,
        }
    }

    fn proof() -> snarkrs_groth16::Proof {
        snarkrs_groth16::Proof {
            a: G1Affine::generator(),
            b: G2Affine::generator(),
            c: G1Affine::generator(),
        }
    }

    /// A directory holding an executable `snarkjs` that runs `export` for `zkey ...` and
    /// `verify` for `groth16 verify ...`, and an artifact directory with only public.json.
    fn setup(name: &str, v: Variant, export: &str, verify: &str) -> (PathBuf, PathBuf) {
        let root =
            std::env::temp_dir().join(format!("snarkrs-csp-test-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let artifacts = root.join("artifacts");
        std::fs::create_dir_all(artifacts.join(v.name())).unwrap();
        std::fs::write(
            artifacts.join(v.name()).join("public.json"),
            "[\n \"1\"\n]\n",
        )
        .unwrap();
        let bin = root.join("snarkjs");
        std::fs::write(
            &bin,
            format!("#!/bin/sh\ncase \"$1\" in\nzkey) {export} ;;\ngroth16) {verify} ;;\nesac\n"),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        (bin, artifacts)
    }

    fn run(name: &str, v: Variant, export: &str, verify: &str) -> Result<()> {
        let (bin, artifacts) = setup(name, v, export, verify);
        let result = snarkjs_verify(&bin, v, &v.zkey(&artifacts), &artifacts, &proof());
        std::fs::remove_dir_all(bin.parent().unwrap()).ok();
        std::fs::remove_dir_all(tempdir(v).unwrap()).ok();
        result
    }

    fn error(name: &str, v: Variant, export: &str, verify: &str) -> String {
        format!("{:#}", run(name, v, export, verify).unwrap_err())
    }

    #[test]
    fn accepts_with_the_vkey_snarkjs_exported() {
        run("accept", variant(128), EXPORT_OK, VERIFY_OK).unwrap();
    }

    #[test]
    fn export_nonzero_exit_names_the_variant() {
        let e = error(
            "export-exit",
            variant(256),
            "echo 'bad zkey' >&2; exit 3",
            VERIFY_OK,
        );
        assert!(e.starts_with("sha256_256: snarkjs did not export"), "{e}");
        assert!(e.contains("exit status: 3"), "{e}");
        assert!(e.contains("stderr: bad zkey"), "{e}");
    }

    #[test]
    fn export_without_a_file_fails() {
        let e = error("export-nofile", variant(512), "true", VERIFY_OK);
        assert!(e.starts_with("sha256_512: snarkjs did not export"), "{e}");
        assert!(e.contains("reading"), "{e}");
    }

    #[test]
    fn verify_without_ok_fails() {
        let e = error(
            "verify-nook",
            variant(1024),
            EXPORT_OK,
            "echo 'Invalid proof'",
        );
        assert!(e.starts_with("sha256_1024: snarkjs did not accept"), "{e}");
        assert!(e.contains("stdout: Invalid proof"), "{e}");
    }

    #[test]
    fn verify_nonzero_exit_fails_even_with_ok() {
        let e = error(
            "verify-exit",
            variant(2048),
            EXPORT_OK,
            "echo 'snarkJS: OK!'; exit 1",
        );
        assert!(e.starts_with("sha256_2048: snarkjs did not accept"), "{e}");
        assert!(e.contains("exit status: 1"), "{e}");
    }

    #[test]
    fn missing_snarkjs_names_the_variant() {
        let v = Variant {
            target: Target::Keccak,
            input_size: 128,
        };
        let (bin, artifacts) = setup("missing", v, EXPORT_OK, VERIFY_OK);
        std::fs::remove_file(&bin).unwrap();
        let e = format!(
            "{:#}",
            snarkjs_verify(&bin, v, &v.zkey(&artifacts), &artifacts, &proof()).unwrap_err()
        );
        std::fs::remove_dir_all(bin.parent().unwrap()).ok();
        std::fs::remove_dir_all(tempdir(v).unwrap()).ok();
        assert!(e.starts_with("keccak_128: running"), "{e}");
    }
}
