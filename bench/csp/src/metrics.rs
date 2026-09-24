//! `{target}_{input}_{system}_{feat}_metrics.json`, the row the ethproofs collector reads.
//!
//! Field for field what `utils::bench::Metrics` serialises upstream, including the two
//! conventions that are not obvious from the field names: durations are **nanoseconds as
//! a bare integer**, and absent optionals are **omitted** rather than written as `null`
//! (`#[serde_with::skip_serializing_none]` over there, `skip_serializing_if` here). A row
//! that writes `"feat": null` is rejected by the collector's `deny_unknown_fields` sibling
//! on the properties block, so this is not cosmetic.

use serde::Serialize;
use std::time::Duration;

/// The classification block, flattened into the row. Upstream calls this
/// `BenchProperties` and its own `circom` entry is the reference for everything except
/// `is_audited`: theirs is `partially_audited` on the strength of a circomlib/bigint
/// audit, ours is `not_audited` because this prover has not been audited at all.
#[derive(Serialize, Clone, Debug)]
pub struct Properties {
    pub proving_system: &'static str,
    pub field_curve: &'static str,
    pub iop: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pcs: Option<&'static str>,
    pub arithm: &'static str,
    pub is_zk: bool,
    pub is_zkvm: bool,
    pub security_bits: u64,
    pub is_pq: bool,
    pub is_maintained: bool,
    pub is_audited: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isa: Option<&'static str>,
}

impl Default for Properties {
    fn default() -> Self {
        Self {
            proving_system: "Groth16",
            field_curve: "Bn254",
            iop: "Groth16",
            pcs: None,
            arithm: "R1CS",
            is_zk: true,
            is_zkvm: false,
            // BN254's pairing security after the exTNFS estimates, which is the number
            // upstream's circom row carries and the reason it is 100 and not 128.
            security_bits: 100,
            is_pq: false,
            is_maintained: true,
            is_audited: "not_audited",
            isa: None,
        }
    }
}

fn nanos<S: serde::Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.as_nanos() as u64)
}

#[derive(Serialize, Clone, Debug)]
pub struct Metrics {
    /// The proving system's name in the published table.
    pub name: String,
    /// The variant tag. We put the backend here, so `g16` on Metal and `g16` on the CPU
    /// are two rows rather than one row that depends on the build flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feat: Option<String>,
    pub target: String,
    pub input_size: usize,
    #[serde(serialize_with = "nanos")]
    pub proof_duration: Duration,
    #[serde(serialize_with = "nanos")]
    pub verify_duration: Duration,
    /// zkVMs only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles: Option<u64>,
    pub proof_size: usize,
    pub preprocessing_size: usize,
    pub num_constraints: usize,
    pub peak_memory: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceleration: Option<&'static str>,
    #[serde(flatten)]
    pub properties: Properties,
}

/// `{target}_{input}_{system}[_{feat}]_metrics.json`, upstream's naming.
pub fn filename(target: &str, input_size: usize, system: &str, feat: Option<&str>) -> String {
    match feat {
        Some(f) if !f.is_empty() => format!("{target}_{input_size}_{system}_{f}_metrics.json"),
        _ => format!("{target}_{input_size}_{system}_metrics.json"),
    }
}

/// Our own numbers, written next to the row above rather than into it.
///
/// The upstream schema has one slot for proving time, so the row has to report the whole
/// pipeline. That single number cannot answer the question this repo actually cares
/// about, which is where the time went, so the split is kept here: a reader who wants to
/// know how much of a keccak_2048 proof was spent reading 1.1 GB of key off the disk can
/// find out without rerunning anything.
#[derive(Serialize, Clone, Debug, Default)]
pub struct Breakdown {
    pub variant: String,
    pub backend: String,
    pub reps: usize,
    /// Per-phase means, milliseconds.
    pub witness_ms: f64,
    pub zkey_load_ms: f64,
    pub prepare_ms: f64,
    pub prove_ms: f64,
    /// `witness + zkey_load + prepare + prove`, the number the row reports.
    pub total_ms: f64,
    pub total_median_ms: f64,
    pub total_min_ms: f64,
    pub total_max_ms: f64,
    /// Verification against a `verification_key.json`, with no zkey read. This is what a
    /// verifier actually costs; the row's `verify_duration` is dominated by the key read
    /// upstream's API forces.
    pub verify_vkey_ms: f64,
    /// `g16-core`'s own stage split of the proving call, microseconds, means.
    pub gather_us: u64,
    pub ntt_us: u64,
    pub pointwise_us: u64,
    pub msm_us: u64,
    pub assemble_us: u64,
}
