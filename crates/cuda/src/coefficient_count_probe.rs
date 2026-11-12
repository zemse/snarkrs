//! Logical source work for the successful warm fused coefficient path, not DRAM traffic.

use super::*;
use snarkrs_field::Zero;
use snarkrs_formats::json_out::JsonWriter;
use std::io::{self, Write};

fn product(a: u64, b: u64) -> Result<u64, ProveError> {
    a.checked_mul(b)
        .ok_or_else(|| bad("coefficient count overflow"))
}

struct Counts {
    n: u64,
    vars: u64,
    nonzeros: [u64; 2],
    by_id: bool,
    nonzero_witness: u64,
    batches: Vec<(u32, u32)>,
}

impl Counts {
    fn new(c: &Coefficients, witness: &[Fr], n: usize, cap: u32) -> Result<Self, ProveError> {
        let (batches, _) = probe_coefficient_schedule(n, cap)?;
        if witness.is_empty() || u32::try_from(witness.len()).is_err() {
            return Err(bad(
                "coefficient count requires a nonempty u32-indexed witness",
            ));
        }
        c.check_structure(n, witness.len())
            .map_err(|e| bad(e.to_string()))?;
        Ok(Self {
            n: n as u64,
            vars: witness.len() as u64,
            nonzeros: c.signal.each_ref().map(|v| v.len() as u64),
            by_id: !ids_are_positions(c),
            nonzero_witness: witness.iter().filter(|s| !s.is_zero()).count() as u64,
            batches,
        })
    }

    fn gather_input_bytes(&self) -> Result<u64, ProveError> {
        let nz = self.nonzeros[0] + self.nonzeros[1];
        let index_reads = 4 * self.n + nz + if self.by_id { nz } else { 0 };
        let indices = product(index_reads, 4)?;
        let fields = product(product(2, nz)?, core::mem::size_of::<PackedFr>() as u64)?;
        indices
            .checked_add(fields)
            .ok_or_else(|| bad("gather byte count overflow"))
    }

    fn butterflies(&self) -> Result<u64, ProveError> {
        let passes: u64 = self.batches.iter().map(|(_, k)| u64::from(*k)).sum();
        product(product(self.n / 2, passes)?, 6)
    }

    fn ntt_global_fields(&self) -> Result<(u64, u64), ProveError> {
        let batches = self.batches.len() as u64;
        // Every batch loads n. JOIN adds A/B reads and replaces one n-store by two H stores.
        Ok((
            product(6 * batches + 2, self.n)?,
            product(6 * batches + 1, self.n)?,
        ))
    }
}

fn number<W: Write>(json: &mut JsonWriter<W>, key: &str, value: u64) -> io::Result<()> {
    json.field_raw(key, &value.to_string())
}

pub(crate) fn canonical_json(
    pk: &ProvingKey,
    witness: &[Fr],
    cap: u32,
) -> Result<String, ProveError> {
    if pk.h_basis != HBasis::SnarkjsOddCoset || witness.len() != pk.n_vars {
        return Err(bad(
            "coefficient trace requires original zkey witness dimensions and odd-coset basis",
        ));
    }
    let c = Counts::new(&pk.coeffs, witness, pk.domain_size, cap)?;
    let fr = core::mem::size_of::<PackedFr>() as u64;
    let scalar = core::mem::size_of::<PackedScalar>() as u64;
    let host_fr = core::mem::size_of::<Fr>() as u64;
    let limb64 = Fr::from(0u64).0 .0.len() as u64;
    let nz = c.nonzeros[0]
        .checked_add(c.nonzeros[1])
        .ok_or_else(|| bad("nonzero count overflow"))?;
    let butterflies = c.butterflies()?;
    let (ntt_reads, ntt_writes) = c.ntt_global_fields()?;
    let mut bytes = Vec::new();
    let mut json = JsonWriter::new(&mut bytes);
    let render = |json: &mut JsonWriter<&mut Vec<u8>>| -> io::Result<()> {
        json.begin_object()?;
        json.field_string("scope", "successful native warm fused coefficient proof; repeat twice serially; no clocks, CUDA objects, proofs or oracle calls")?;
        json.field_string("units", "source-level logical accesses/calls, not compiler instructions, cache transactions, DRAM bytes, allocation peak or measured work")?;
        number(json, "serial_pair_repetitions", 2)?;
        number(json, "caller_supplied_max_fused_passes", u64::from(cap))?;
        json.field_raw("hardware_profile_observed", "false")?;
        json.key("host_witness_per_proof")?;
        json.begin_object()?;
        for (key, value) in [
            ("PackedFr_pack_field_visits", c.vars),
            ("pack_source_u64_limb_reads", c.vars * limb64),
            ("pack_source_logical_bytes", c.vars * host_fr),
            ("pack_pinned_u32_word_writes", c.vars * fr / 4),
            ("pack_pinned_logical_bytes", c.vars * fr),
            ("prefix_classification_scalar_visits", c.vars),
            ("prefix_is_zero_calls", c.vars),
            (
                "prefix_is_one_calls_after_zero_short_circuit",
                c.nonzero_witness,
            ),
            ("prefix_u32_entries_written", c.vars + 1),
            ("prefix_requested_payload_bytes", (c.vars + 1) * 4),
            ("prefix_vec_constructions", 1),
            ("prefix_payload_clone_calls", 0),
            ("ordered_fingerprint_calls", 2),
            ("fingerprint_u64_limb_visits", 2 * c.vars * limb64),
            (
                "fingerprint_rotate_xor_wrapping_multiply_steps",
                2 * c.vars * limb64,
            ),
            ("reuse_fingerprint_u64_equality_calls", 1),
        ] {
            number(json, key, value)?;
        }
        json.field_string("prefix_lifetime", "scalar_prefix moves into HHandle; witness_std_for and Scalars borrow it, no payload clone on canonical reuse")?;
        json.field_string("classification_and_fingerprint_bytes", "field/limb visits only; short-circuit limb loads, cache reuse and physical memory instructions unknown")?;
        json.end()?;
        json.key("witness_upload_and_conversion_per_proof")?;
        json.begin_object()?;
        number(json, "pinned_H2D_copies", 1)?;
        number(json, "H2D_bytes", c.vars * fr)?;
        number(json, "H2D_bytes_per_serial_pair", 2 * c.vars * fr)?;
        number(json, "mont_to_std_launches", 1)?;
        number(json, "mont_to_std_elements", c.vars)?;
        number(json, "mont_to_std_logical_input_bytes", c.vars * fr)?;
        number(json, "mont_to_std_logical_output_bytes", c.vars * scalar)?;
        number(json, "fr_from_mont_calls", c.vars)?;
        json.end()?;
        json.key("CSR_gather_per_proof")?;
        json.begin_object()?;
        number(json, "A_nonzeros", c.nonzeros[0])?;
        number(json, "B_nonzeros", c.nonzeros[1])?;
        json.field_raw("by_id", if c.by_id { "true" } else { "false" })?;
        for (key, value) in [
            ("active_row_threads", c.n),
            ("row_endpoint_u32_reads", 4 * c.n),
            ("signal_index_u32_reads", nz),
            ("optional_value_id_u32_reads", if c.by_id { nz } else { 0 }),
            ("coefficient_value_Fr_lookups", nz),
            ("witness_Fr_lookups", nz),
            ("output_Fr_writes", 3 * c.n),
            ("Fr_mul_calls_including_C", nz + c.n),
            ("Fr_add_calls", nz),
            (
                "logical_input_bytes",
                c.gather_input_bytes().map_err(io::Error::other)?,
            ),
            ("logical_output_bytes", 3 * c.n * fr),
        ] {
            number(json, key, value)?;
        }
        json.field_string("lookup_scope", "per-nonzero values and witness lookups, NOT unique table size; C=A*B uses register results with no extra A/B global read")?;
        json.end()?;
        json.key("six_transform_batches_in_submission_order")?;
        json.begin_array()?;
        for vector in ["A", "B", "C"] {
            for inverse in [true, false] {
                for (index, &(s0, k)) in c.batches.iter().enumerate() {
                    let bf = c.n / 2 * u64::from(k);
                    let head = index == 0;
                    let join = vector == "C" && !inverse && index + 1 == c.batches.len();
                    let block_elements = 1u64 << k;
                    let grid = (c.n >> k).max(1);
                    json.begin_object()?;
                    json.field_string("vector", vector)?;
                    json.field_string("direction", if inverse { "inverse" } else { "forward" })?;
                    json.field_string(
                        "kernel",
                        if head { "g16_ntt_head" } else { "g16_ntt_tail" },
                    )?;
                    json.field_string(
                        "store",
                        if join {
                            "STORE_JOIN replaces C store"
                        } else {
                            "STORE_PLAIN"
                        },
                    )?;
                    for (key, value) in [
                        ("s0", u64::from(s0)),
                        ("k", u64::from(k)),
                        ("grid_blocks", grid),
                        ("shared_bytes_per_block", block_elements * fr),
                        (
                            "threads_per_block_upper_bound",
                            u64::from(PREFERRED_BLOCK).min(block_elements),
                        ),
                        ("butterflies", bf),
                        ("twiddle_Fr_reads", bf),
                        ("butterfly_Fr_mul_calls", bf),
                        ("butterfly_Fr_add_calls", bf),
                        ("butterfly_Fr_sub_calls", bf),
                        ("main_vector_Fr_reads", c.n),
                        ("main_vector_Fr_writes", if join { 0 } else { c.n }),
                        ("join_A_B_Fr_reads", if join { 2 * c.n } else { 0 }),
                        ("H_mont_Fr_writes", if join { c.n } else { 0 }),
                        ("H_std_scalar_writes", if join { c.n } else { 0 }),
                        (
                            "normalization_Fr_mul_calls",
                            if head && inverse { c.n } else { 0 },
                        ),
                        ("coset_Fr_mul_calls", if head && !inverse { c.n } else { 0 }),
                        (
                            "coset_table_Fr_reads",
                            if head && !inverse { c.n } else { 0 },
                        ),
                        ("shared_Fr_reads", c.n + 2 * bf),
                        ("shared_Fr_writes", c.n + 2 * bf),
                        ("source_block_barrier_calls", grid * (1 + u64::from(k))),
                    ] {
                        number(json, key, value)?;
                    }
                    json.field_raw("actual_threads_per_block", "null")?;
                    json.field_string("thread_geometry", "min(preferred block, device max threads, 2^k); device attribute unobserved")?;
                    json.end()?;
                }
            }
        }
        json.end()?;
        json.key("NTT_and_fused_join_totals_per_proof")?;
        json.begin_object()?;
        for (key, value) in [
            ("transforms", 6),
            ("kernel_launches", 6 * c.batches.len() as u64),
            ("standalone_join_launches", 0),
            ("butterflies", butterflies),
            ("butterflies_per_serial_pair", 2 * butterflies),
            ("twiddle_Fr_reads", butterflies),
            ("coset_table_Fr_reads", 3 * c.n),
            ("normalization_Fr_mul_calls", 3 * c.n),
            ("coset_Fr_mul_calls", 3 * c.n),
            ("join_Fr_mul_calls", c.n),
            ("join_Fr_sub_calls", c.n),
            ("g16_store_h_calls", c.n),
            ("H_fr_from_mont_calls", c.n),
            ("main_and_join_logical_read_bytes", ntt_reads * fr),
            ("main_and_H_logical_write_bytes", ntt_writes * fr),
            (
                "main_and_join_logical_read_bytes_per_pair",
                2 * ntt_reads * fr,
            ),
            (
                "main_and_H_logical_write_bytes_per_pair",
                2 * ntt_writes * fr,
            ),
            (
                "twiddle_and_coset_logical_read_bytes",
                (butterflies + 3 * c.n) * fr,
            ),
            (
                "shared_logical_read_bytes",
                (6 * c.batches.len() as u64 * c.n + 2 * butterflies) * fr,
            ),
            (
                "shared_logical_write_bytes",
                (6 * c.batches.len() as u64 * c.n + 2 * butterflies) * fr,
            ),
        ] {
            number(json, key, value)?;
        }
        json.field_string("logical_array_scope", "head and tail each load n and store n; final forward C store instead reads A/B and writes both H encodings; shared and table accesses separate; no standalone join or C write double count")?;
        json.end()?;
        json.key("native_host_proof_calls_per_successful_proof")?;
        json.begin_object()?;
        for (key, value) in [
            ("Fr_random_sample_calls_before_spawn", 2),
            ("nonzero_blinder_checks_on_success", 2),
            ("scoped_plain_thread_spawns", 1),
            ("scoped_plain_thread_joins", 1),
            ("overlapped_blinder_G1_scalar_products", 3),
            ("overlapped_blinder_G2_scalar_products", 1),
            ("overlapped_blinder_Fr_r_times_s", 1),
            ("post_join_Rayon_join_calls", 1),
            ("post_MSM_G1_scalar_products", 2),
            ("assembly_G1_add_calls", 7),
            ("assembly_G2_add_calls", 2),
            ("assembly_G1_sub_calls", 1),
            ("G1_batch_normalize_calls", 1),
            ("G1_points_to_normalize", 2),
            ("G2_batch_normalize_calls", 1),
            ("G2_points_to_normalize", 1),
        ] {
            number(json, key, value)?;
        }
        json.field_string("DAG", "two Fr samples and zero checks BEFORE spawn -> caller stages0..9 || one scoped plain thread (3G1+1G2 products and r*s) -> thread join -> pi_a/pi_b/pib1 adds -> Rayon join (2G1 MSM-dependent products) -> pi_c adds/sub -> two normalization batches -> proof; both calls finish before outside-clock verification/serialization")?;
        json.field_string("clock_warning", "assemble_us covers final assembly, not earlier overlapped terms/RNG/spawn/join; no additivity or zero-cost late join assumption")?;
        json.end()?;
        json.key("unknown_not_zero")?;
        json.begin_object()?;
        for key in [
            "physical_memory_instructions_DRAM_cache_transactions",
            "device_max_threads",
            "host_thread_pool_allocator_costs",
            "random_bytes_and_rejection_draws",
            "exact_group_Fq_operations_and_normalization_inversions",
            "MSM_data_dependent_digit_bucket_traffic",
            "physical_live_set_and_admission",
            "primitive_floors_and_headroom",
        ] {
            json.field_raw(key, "null")?;
        }
        json.end()?;
        json.field_raw("full_workload_trace", "false")?;
        json.end()
    };
    render(&mut json).map_err(|e| bad(e.to_string()))?;
    json.finish().map_err(|e| bad(e.to_string()))?;
    String::from_utf8(bytes).map_err(|e| bad(e.to_string()))
}

#[test]
fn logical_n1_n2_and_cap_splits_preserve_join_and_butterflies() {
    assert_eq!(
        core::mem::size_of::<PackedFr>(),
        core::mem::size_of::<PackedScalar>()
    );
    let empty = |n: usize| Coefficients {
        row_ptr: [vec![0; n + 1], vec![0; n + 1]],
        signal: [vec![], vec![]],
        value_id: [vec![], vec![]],
        values: vec![],
    };
    let one = Counts::new(&empty(1), &[Fr::from(1u64)], 1, 1).unwrap();
    assert_eq!(one.batches, [(0, 0)]);
    assert_eq!(one.gather_input_bytes().unwrap(), 16);
    assert_eq!(one.butterflies().unwrap(), 0);
    assert_eq!(one.ntt_global_fields().unwrap(), (8, 7));
    let two = Counts::new(&empty(2), &[Fr::from(1u64)], 2, 1).unwrap();
    assert_eq!(two.butterflies().unwrap(), 6);
    assert_eq!(two.ntt_global_fields().unwrap(), (16, 14));
    for cap in [1, 2, 3, 10] {
        let c = Counts::new(&empty(8), &[Fr::from(1u64)], 8, cap).unwrap();
        assert_eq!(c.butterflies().unwrap(), 72);
        let b = c.batches.len() as u64;
        assert_eq!(
            c.ntt_global_fields().unwrap(),
            ((6 * b + 2) * 8, (6 * b + 1) * 8)
        );
    }
}

#[test]
fn logical_csr_interned_direct_empty_and_bad_shapes() {
    let mut c = Coefficients {
        row_ptr: [vec![0, 1, 1], vec![0, 1, 1]],
        signal: [vec![0], vec![0]],
        value_id: [vec![0], vec![1]],
        values: vec![Fr::from(1u64); 2],
    };
    let direct = Counts::new(&c, &[Fr::from(1u64)], 2, 10).unwrap();
    assert!(!direct.by_id);
    assert_eq!(direct.nonzeros, [1, 1]);
    assert_eq!(direct.gather_input_bytes().unwrap(), 168);
    c.value_id[1][0] = 0;
    c.values.pop();
    let interned = Counts::new(&c, &[Fr::from(1u64)], 2, 10).unwrap();
    assert!(interned.by_id);
    assert_eq!(interned.gather_input_bytes().unwrap(), 176);
    assert!(Counts::new(&c, &[], 2, 10).is_err());
    for n in [0, 3, usize::MAX] {
        assert!(Counts::new(&c, &[Fr::from(1u64)], n, 10).is_err());
    }
    c.signal[0][0] = 1;
    assert!(Counts::new(&c, &[Fr::from(1u64)], 2, 10).is_err());
    assert!(product(u64::MAX, 2).is_err());
}
