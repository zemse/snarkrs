//! Host-only inventory prerequisite for a serial pair, not a complete workload trace.

use std::io::Write;
use std::path::{Path, PathBuf};

use snarkrs_formats::{wtns::Witness, HBasis, ProvingKey, VerifyingKey};

use super::{resident_bytes, CudaMsm, CudaStages};
use crate::stages::probe_coefficient_schedule;

struct Config {
    root: PathBuf,
    output: PathBuf,
    max_fused: u32,
}

impl Config {
    fn parse(root: Option<&str>, output: Option<&str>, cap: Option<&str>) -> Result<Self, String> {
        let absolute = |value: Option<&str>, name: &str| {
            let path = PathBuf::from(value.ok_or_else(|| format!("set {name}"))?);
            if !path.is_absolute() {
                return Err(format!("{name} must be absolute"));
            }
            Ok(path)
        };
        let root = absolute(root, "G16_ARTIFACTS")?;
        let output = absolute(output, "G16_PAIR_COUNT_OUTPUT")?;
        let max_fused = cap
            .ok_or("set G16_PAIR_MAX_FUSED_PASSES explicitly; no device is queried")?
            .parse::<u32>()
            .map_err(|e| format!("invalid fused pass cap: {e}"))?;
        probe_coefficient_schedule(1, max_fused).map_err(|e| e.to_string())?;
        Ok(Self {
            root,
            output,
            max_fused,
        })
    }
}

fn check_shape(n: usize, vars: usize, public: usize) -> Result<(), String> {
    probe_coefficient_schedule(n, 1).map_err(|e| e.to_string())?;
    if vars == 0 || u32::try_from(vars).is_err() || public >= vars {
        return Err("invalid variable/public dimensions".into());
    }
    Ok(())
}

fn sum_bytes(buffers: &[u64]) -> Result<u64, String> {
    buffers.iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or("byte count overflow".into())
    })
}

struct Inventory {
    n: usize,
    vars: usize,
    public: usize,
    tables: Vec<u64>,
    bases_and_live: Vec<u64>,
    scratch: Vec<u64>,
    admission: u64,
    batches: Vec<(u32, u32)>,
    launches: usize,
}

impl Inventory {
    fn new(pk: &ProvingKey, max_fused: u32) -> Result<Self, String> {
        check_shape(pk.domain_size, pk.n_vars, pk.n_public)?;
        if pk.h_basis != HBasis::SnarkjsOddCoset {
            return Err("this partial probe covers the zkey coefficient path only".into());
        }
        pk.coeffs
            .check_structure(pk.domain_size, pk.n_vars)
            .map_err(|e| e.to_string())?;
        for (name, got, want) in [
            ("a", pk.a_query.len(), pk.n_vars),
            ("b1", pk.b_g1_query.len(), pk.n_vars),
            ("b2", pk.b_g2_query.len(), pk.n_vars),
            ("l", pk.l_query.len(), pk.n_vars - pk.n_public - 1),
            ("h", pk.h_query.len(), pk.domain_size),
        ] {
            if got != want {
                return Err(format!("{name} query has {got} entries, expected {want}"));
            }
        }
        let tables = CudaStages::key_buffers(pk);
        let bases_and_live = CudaMsm::key_buffers(pk);
        let scratch = CudaStages::scratch_buffers(pk.domain_size, pk.n_vars);
        let (batches, launches) =
            probe_coefficient_schedule(pk.domain_size, max_fused).map_err(|e| e.to_string())?;
        // Validate sums before using the production admission estimator.
        let mut all = tables.clone();
        all.extend(&bases_and_live);
        all.extend(&scratch);
        sum_bytes(&all)?;
        Ok(Self {
            n: pk.domain_size,
            vars: pk.n_vars,
            public: pk.n_public,
            tables,
            bases_and_live,
            scratch,
            admission: resident_bytes(pk),
            batches,
            launches,
        })
    }

    fn json(&self, name: &str, cap: u32) -> String {
        // Names are the two fixed fixture names, not caller-provided JSON text.
        assert!(["tiny_mul", "sha256"].contains(&name));
        let batches = self
            .batches
            .iter()
            .map(|(s0, k)| format!("[{s0},{k}]"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"fixture\":\"{}\",\"domain\":{},\"variables\":{},\"public\":{},",
                "\"prepared_circuits\":1,\"serial_proofs\":2,",
                "\"caller_supplied_fused_pass_cap\":{},\"pass_batches\":[{}],",
                "\"fused_coefficient_launches_per_proof\":{},",
                "\"fused_coefficient_launches_per_pair\":{},",
                "\"stage_key_buffer_bytes\":{:?},",
                "\"query_and_optional_live_index_buffer_bytes\":{:?},",
                "\"one_reusable_stage_scratch_buffer_bytes\":{:?},",
                "\"production_resident_admission_estimate_bytes\":{},",
                "\"resident_estimate_scope\":\"one-key-one-stage-scratch-each-buffer-rounded-2MiB\",",
                "\"unknown\":{{\"msm_schedule\":null,\"msm_scratch_and_arena_bytes\":null,",
                "\"pinned_host_staging_total_bytes\":null,\"module_and_context_bytes\":null,",
                "\"total_pair_traffic_bytes\":null,\"total_pair_launches\":null,",
                "\"synchronizations\":null,\"host_pack_and_combine_work\":null,",
                "\"primitive_floors\":null,\"two_lane_admission_bytes\":null}}}}"
            ),
            name,
            self.n,
            self.vars,
            self.public,
            cap,
            batches,
            self.launches,
            self.launches * 2,
            self.tables,
            self.bases_and_live,
            self.scratch,
            self.admission,
        )
    }
}

fn fixture_inventory(root: &Path, name: &str, cap: u32) -> Result<String, String> {
    let dir = root.join(name);
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).map_err(|e| e.to_string())?;
    let inventory = Inventory::new(&pk, cap)?;
    let witness = Witness::load(&dir.join("circuit.wtns")).map_err(|e| e.to_string())?;
    let vk = VerifyingKey::from_json(&dir.join("vkey.json")).map_err(|e| e.to_string())?;
    if witness.0.len() != pk.n_vars || vk.ic.len() != pk.n_public + 1 {
        return Err(format!(
            "{name}: witness or verifying key dimensions disagree"
        ));
    }
    Ok(inventory.json(name, cap))
}

fn write_inventory(config: &Config) -> Result<(), String> {
    // No CUDA constructor, clocks, proof generation or performance acceptance here.
    let tiny = fixture_inventory(&config.root, "tiny_mul", config.max_fused)?;
    let sha = fixture_inventory(&config.root, "sha256", config.max_fused)?;
    let json = format!(
        concat!(
            "{{\"schema\":1,\"mode\":\"partial-serial-pair-inventory\",",
            "\"complete_workload_trace\":false,\"device_opened\":false,",
            "\"device_profile_observed\":false,\"timings\":null,",
            "\"buffer_order\":{{",
            "\"stage_key\":[\"tw_fwd\",\"tw_inv\",\"coset_pows\",\"values\",",
            "\"a_ids_or_placeholder\",\"a_row_ptr\",\"a_signal\",",
            "\"b_ids_or_placeholder\",\"b_row_ptr\",\"b_signal\"],",
            "\"query\":[\"a\",\"b1\",\"l\",\"h\",\"b2\",\"optional_live_indices_in_a_b_order\"],",
            "\"stage_scratch\":[\"witness\",\"witness_std\",\"a\",\"b\",\"c\",",
            "\"t\",\"h_mont\",\"h_std\",\"unused\"]}},",
            "\"fixtures\":[{},{}]}}\n"
        ),
        tiny, sha
    );
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&config.output)
        .and_then(|mut file| file.write_all(json.as_bytes()))
        .map_err(|e| format!("{}: {e}", config.output.display()))
}

#[test]
#[ignore = "host-only partial inventory; requires explicit fixture, output and fused-cap configuration"]
fn pair_inventory_count_only() {
    let get = |key| std::env::var(key).ok();
    let (root, output, cap) = (
        get("G16_ARTIFACTS"),
        get("G16_PAIR_COUNT_OUTPUT"),
        get("G16_PAIR_MAX_FUSED_PASSES"),
    );
    let config = Config::parse(root.as_deref(), output.as_deref(), cap.as_deref())
        .expect("count-only configuration; never skips even without G16_REQUIRE_TESTS");
    write_inventory(&config).expect("both original fixtures must have a complete inventory");
}

#[test]
fn count_configuration_requires_explicit_inputs_and_cap() {
    for (root, output, cap) in [
        (None, Some("/output"), Some("10")),
        (Some("/fixtures"), None, Some("10")),
        (Some("/fixtures"), Some("/output"), None),
        (Some("relative"), Some("/output"), Some("10")),
        (Some("/fixtures"), Some("relative"), Some("10")),
        (Some("/fixtures"), Some("/output"), Some("0")),
        (Some("/fixtures"), Some("/output"), Some("11")),
        (Some("/fixtures"), Some("/output"), Some("4294967296")),
    ] {
        assert!(Config::parse(root, output, cap).is_err());
    }
    assert!(Config::parse(Some("/fixtures"), Some("/output"), Some("1")).is_ok());
}

#[test]
fn count_geometry_zero_and_overflow_are_not_workloads() {
    for n in [0, 3, 1 << 29, 1 << 31, usize::MAX] {
        assert!(check_shape(n, 1, 0).is_err());
        assert!(probe_coefficient_schedule(n, 1).is_err());
    }
    assert!(check_shape(1, 0, 0).is_err());
    assert!(check_shape(1, 1, 1).is_err());
    assert!(check_shape(1, usize::MAX, 0).is_err());
    assert!(check_shape(1, 1, usize::MAX).is_err());
    assert!(check_shape(1, 1, 0).is_ok());
    assert_eq!(sum_bytes(&[]).unwrap(), 0);
    assert_eq!(sum_bytes(&[0, 1, 2]).unwrap(), 3);
    assert!(sum_bytes(&[u64::MAX, 1]).is_err());
    assert_eq!(CudaStages::scratch_buffers(0, 0), vec![32; 9]);
}

#[test]
fn synthetic_inventory_keeps_unmeasured_work_unknown() {
    let synthetic = Inventory {
        n: 1,
        vars: 1,
        public: 0,
        tables: vec![4],
        bases_and_live: vec![4],
        scratch: vec![32; 9],
        admission: 0,
        batches: vec![(0, 0)],
        launches: 8,
    };
    let json = synthetic.json("tiny_mul", 1);
    for field in [
        "msm_schedule",
        "msm_scratch_and_arena_bytes",
        "pinned_host_staging_total_bytes",
        "module_and_context_bytes",
        "total_pair_traffic_bytes",
        "total_pair_launches",
        "synchronizations",
        "host_pack_and_combine_work",
        "primitive_floors",
        "two_lane_admission_bytes",
    ] {
        assert!(json.contains(&format!("\"{field}\":null")), "{field}");
    }
    assert!(json.contains("\"serial_proofs\":2"));
    assert!(json.contains("\"prepared_circuits\":1"));
    assert!(json.contains("\"fused_coefficient_launches_per_pair\":16"));
}

#[test]
fn count_schedule_uses_actual_partition_at_each_supported_cap() {
    for cap in 1..=10 {
        for log_n in 0..=28 {
            let n = 1 << log_n;
            if log_n == 28 {
                let domain = snarkrs_field::Domain::new(n).unwrap();
                assert!(domain.proving_coset_shift().is_err());
                assert!(probe_coefficient_schedule(n, cap).is_err());
                assert!(check_shape(n, 1, 0).is_err());
                continue;
            }
            let (batches, launches) = probe_coefficient_schedule(n, cap).unwrap();
            let mut at = 0;
            for &(s0, k) in &batches {
                assert_eq!(s0, at);
                assert!(k <= cap);
                at += k;
            }
            assert_eq!(at, log_n);
            assert_eq!(launches, 2 + 6 * batches.len());
        }
    }
    assert!(probe_coefficient_schedule(8, 0).is_err());
    assert!(probe_coefficient_schedule(8, u32::MAX).is_err());
    assert_eq!(probe_coefficient_schedule(1, 1).unwrap(), (vec![(0, 0)], 8));
    assert_eq!(probe_coefficient_schedule(8, 2).unwrap().1, 14);
    assert_eq!(probe_coefficient_schedule(8, 10).unwrap().1, 8);
    assert_eq!(probe_coefficient_schedule(65536, 10).unwrap().1, 14);
}
