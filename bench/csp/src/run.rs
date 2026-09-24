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
use crate::{circuits, inputs, metrics, Prover, Variant};
use anyhow::{Context, Result};
use g16_core::{prove::prove, verify::verify, StageTimings};
use g16_field::Fr;
use g16_zkey::{wtns::Witness, ProvingKey};
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
) -> Result<(Split, g16_core::Proof, Vec<Fr>)> {
    let mut split = Split::default();

    let t = Instant::now();
    let wtns = circuits::witness(variant, input_json)?;
    split.witness = t.elapsed();

    match prover {
        Prover::G16(backend) => {
            let t = Instant::now();
            let pk =
                ProvingKey::load(zkey).with_context(|| format!("loading {}", zkey.display()))?;
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
            let proof = prove(
                circuit.as_ref(),
                &witness,
                &mut ark_std::rand::thread_rng(),
                &mut split.stages,
            )?;
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

            let proof = g16_core::json::proof_from_str(&std::fs::read_to_string(&proof_path)?)?;
            let public = g16_core::json::public_from_str(&std::fs::read_to_string(&public_path)?)?;
            std::fs::remove_dir_all(&dir).ok();
            Ok((split, proof, public))
        }
    }
}

/// A private directory under the system temp dir, named for the variant so two runs of
/// different variants cannot collide and a leftover is obvious.
fn tempdir(variant: Variant) -> Result<std::path::PathBuf> {
    let dir =
        std::env::temp_dir().join(format!("g16-csp-{}-{}", variant.name(), std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
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

    // Warm-up. Its proof is the one that gets verified and measured for size, so a
    // variant whose proof does not verify fails before any timing is reported rather
    // than after.
    let (_, proof, public) = proof_iteration(variant, input_json.clone(), &zkey, &cfg.prover)?;
    let pk = ProvingKey::load(&zkey)?;
    verify(&pk.vk, &public, &proof)
        .with_context(|| format!("{}: our own verifier rejected the proof", variant.name()))?;
    let n_constraints = num_constraints(&pk);
    let proof_size = g16_core::json::proof_to_string(&proof).len()
        + g16_core::json::public_to_string(&public).len();
    drop(pk);

    let mut splits = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let (split, _, _) = proof_iteration(variant, input_json.clone(), &zkey, &cfg.prover)?;
        splits.push(split);
    }

    // Verification, upstream's way: re-read the key, derive the verifier from it, check.
    // Then again from an already-parsed key, which is the honest cost of a verifier.
    let mut verify_full = Vec::with_capacity(cfg.reps);
    let mut verify_vkey = Vec::with_capacity(cfg.reps);
    for _ in 0..cfg.reps {
        let t = Instant::now();
        let pk = ProvingKey::load(&zkey)?;
        verify(&pk.vk, &public, &proof)?;
        verify_full.push(t.elapsed());

        let t = Instant::now();
        verify(&pk.vk, &public, &proof)?;
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
        feat: cfg.prover.feat(),
        target: variant.target.as_str().to_string(),
        input_size: variant.input_size,
        proof_duration: mean(&totals),
        verify_duration: mean(&verify_full),
        cycles: None,
        proof_size,
        preprocessing_size: variant.preprocessing_size(cfg.artifacts, cfg.witness_src)? as usize,
        num_constraints: n_constraints,
        // rapidsnark's peak is not ours to measure through `g16-csp-mem`, which links
        // our prover. Left at the collector's 0 marker rather than reported as a number
        // that came from the wrong process.
        peak_memory: match (&cfg.prover, cfg.mem_reps) {
            (_, 0) | (Prover::Rapidsnark(_), _) => 0,
            (Prover::G16(backend), reps) => {
                crate::mem::sample(cfg.mem_bin, variant, *backend, cfg.artifacts, reps)?
            }
        },
        acceleration: None,
        properties: Properties::default(),
    };

    let breakdown = Breakdown {
        variant: variant.name(),
        backend: cfg.prover.label(),
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
