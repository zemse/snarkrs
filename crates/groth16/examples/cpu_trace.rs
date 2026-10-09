//! Input dimensions and logical work counts, without proving or timings.
//! PROBE_FIXTURE names one artifact directory. See msm/examples/cpu_intrinsics.md.

use std::{env, error::Error, mem::size_of, path::PathBuf};

use serde_json::{json, Value};
use snarkrs_field::{AffineRepr, Fr, G1Affine, G2Affine, One, PrimeField, Zero};
use snarkrs_formats::{wtns::Witness, ProvingKey};
use snarkrs_msm::{window_size, RECODE_BITS};

fn msm<A: AffineRepr>(name: &str, bases: &[A], scalars: &[Fr]) -> Value {
    let mut infinity = 0usize;
    let mut zero = 0usize;
    let mut one = 0usize;
    let mut general = 0usize;
    for (p, s) in bases.iter().zip(scalars) {
        if p.is_zero() {
            infinity += 1;
        } else if s.is_zero() {
            zero += 1;
        } else if s.is_one() {
            one += 1;
        } else {
            general += 1;
        }
    }
    let width = window_size(general);
    json!({
        "name": name, "query_entries": bases.len(), "scalar_entries": scalars.len(),
        "host_query_bytes": std::mem::size_of_val(bases),
        "variable_prescan": { "infinity": infinity, "zero_noninfinity": zero,
            "one_noninfinity": one, "general_noninfinity": general },
        "window_bits_variable": (general > 0).then_some(width),
        "windows_variable": if general > 0 { RECODE_BITS.div_ceil(width as usize) } else { 0 },
        "buckets_per_window": if general > 0 { 1usize << (width - 1) } else { 0 },
        "bucket_add_count": null, "shared_inversion_rounds": null,
        "reason": "digit occupancy, batch conflicts and dispatch splitting not traced"
    })
}

fn trace(pk: &ProvingKey, witness: &[Fr]) -> Result<Value, Box<dyn Error>> {
    if witness.len() != pk.n_vars || pk.n_public >= pk.n_vars {
        return Err("witness/key dimensions disagree".into());
    }
    if pk.a_query.len() != witness.len()
        || pk.b_g1_query.len() != witness.len()
        || pk.b_g2_query.len() != witness.len()
        || pk.l_query.len() != witness.len() - pk.n_public - 1
        || pk.h_query.len() != pk.domain_size
    {
        return Err("query/key dimensions disagree".into());
    }
    let n = pk.domain_size;
    if !n.is_power_of_two() {
        return Err("domain must be a nonzero power of two".into());
    }
    let mut gathers = Vec::new();
    for m in 0..2 {
        let signals = &pk.coeffs.signal[m];
        let mut zeros = 0usize;
        let mut ones = 0usize;
        for &signal in signals {
            let w = witness
                .get(signal as usize)
                .ok_or("coefficient signal out of bounds")?;
            zeros += usize::from(w.is_zero());
            ones += usize::from(w.is_one());
        }
        gathers.push(json!({
            "matrix": m, "coefficient_visits": signals.len(),
            "zero_skips": zeros, "one_shortcuts": ones,
            "Fr_mul": signals.len() - zeros - ones, "Fr_add": signals.len() - zeros,
            "coefficient_storage_bytes": std::mem::size_of_val(pk.coeffs.value[m].as_slice()),
            "signal_storage_bytes": std::mem::size_of_val(signals.as_slice()),
            "row_pointer_storage_bytes": std::mem::size_of_val(pk.coeffs.row_ptr[m].as_slice())
        }));
    }
    let h_width = window_size(n);
    Ok(json!({
        "n_vars": pk.n_vars, "n_public": pk.n_public, "domain_size": n,
        "host_Fr_bytes": size_of::<Fr>(), "host_G1Affine_bytes": size_of::<G1Affine>(),
        "host_G2Affine_bytes": size_of::<G2Affine>(),
        "segments": [
            {"name": "gather", "matrices": gathers, "parallel": "A and B join; parallel rows"},
            {"name": "C=A*B", "Fr_mul": n, "logical_read_write_bytes": 3*n*size_of::<Fr>()},
            {"name": "three_intt_coset_ntt", "vectors": 3, "transforms": 6,
                "radix2_equivalent_butterflies": 6*(n/2)*n.trailing_zeros() as usize,
                "coset_products": 3*n, "full_bit_reversal_passes": 0,
                "live_vector_bytes": 3*n*size_of::<Fr>(),
                "actual_Fr_operations": null, "physical_memory_bytes": null,
                "parallel": "three pipelines joined; blocked parallel kernels",
                "reason": "compact twiddles and fused kernels require kernel-specific counts"},
            {"name": "H=A*B-C", "Fr_mul": n, "Fr_sub": n,
                "logical_read_write_bytes": 4*n*size_of::<Fr>()},
            {"name": "scrub_B_C", "logical_write_bytes": 2*n*size_of::<Fr>()},
            {"name": "five_MSMs", "parallel": "nested joins after H; internal window work",
                "queries": [
                    msm("A_G1", &pk.a_query, witness),
                    msm("B_G2", &pk.b_g2_query, witness),
                    msm("B_G1", &pk.b_g1_query, witness),
                    msm("L_G1", &pk.l_query, &witness[pk.n_public+1..]),
                    {"name": "H_G1", "query_entries": pk.h_query.len(),
                        "host_query_bytes": std::mem::size_of_val(pk.h_query.as_slice()),
                        "general_scalars": null, "general_scalar_upper_bound": n,
                        "dense_candidate_window_bits": h_width,
                        "dense_candidate_windows": RECODE_BITS.div_ceil(h_width as usize),
                        "reason": "H not computed by this count-only probe"}
                ]},
            {"name": "assembly_and_checked_verification", "actual_operations": null,
                "reason": "not traced; must be included in end-to-end timing"}
        ]
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(env::var("PROBE_FIXTURE")?);
    let pk = ProvingKey::load(&dir.join("circuit.zkey"))?;
    let witness = Witness::load(&dir.join("circuit.wtns"))?.0;
    let checksum = witness
        .iter()
        .flat_map(|w| w.into_bigint().0)
        .fold(0u64, |a, b| a.rotate_left(7) ^ b);
    let output = json!({
        "kind": "computation_trace", "schema": 1, "fixture": dir,
        "revision_declared": env::var("PROBE_REV").unwrap_or_else(|_| "unknown".into()),
        "work_mode": "variable", "unit_policy": "logical bytes, not DRAM traffic",
        "witness_checksum_noncryptographic": format!("{checksum:016x}"),
        "counts": trace(&pk, &witness)?
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prescan_excludes_infinity_before_scalar_categories() {
        let g = G1Affine::generator();
        let bases = [G1Affine::zero(), g, g, g];
        let scalars = [Fr::from(11u64), Fr::zero(), Fr::one(), Fr::from(2u64)];
        let row = msm("test", &bases, &scalars);
        for category in [
            "infinity",
            "zero_noninfinity",
            "one_noninfinity",
            "general_noninfinity",
        ] {
            assert_eq!(row["variable_prescan"][category], 1);
        }
        let empty = msm::<G1Affine>("empty", &[], &[]);
        assert_eq!(empty["windows_variable"], 0);
        assert!(empty["window_bits_variable"].is_null());
    }
}
