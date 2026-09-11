//! Host-only counts for Metal's MSM plans and the CSR gather. No device or timings.
//!
//! `cargo run --release -p g16-metal --example msm_audit -- bench/artifacts/csp/sha256_128`
//!
//! Pass circuit directories, not their parent. H is computed by the CPU backend;
//! its plan still uses the full domain length, as device-resident scalars do. The
//! window comes from Metal's selector, including `G16_METAL_MSM_C` when set.

#[cfg(target_os = "macos")]
mod audit {
    use std::path::Path;

    use g16_core::{cpu::CpuBackend, Backend, StageTimings};
    use g16_field::{AffineRepr, BigInteger, Fr, One, PrimeField, Zero};
    use g16_metal::msm::window_size_for;
    use g16_zkey::{wtns::Witness, ProvingKey};

    // Same carry-free recoding as sc_signed_digit. Reconstructing each scalar below
    // checks the host count independently of the GPU and catches a lost top carry.
    fn digit(limbs: &[u64], w: usize, c: usize) -> i64 {
        let off = w * c;
        let word = off / 64;
        let shift = off % 64;
        let mut bits = limbs.get(word).copied().unwrap_or(0) >> shift;
        if shift + c > 64 {
            bits |= limbs.get(word + 1).copied().unwrap_or(0) << (64 - shift);
        }
        let b = bits & ((1 << c) - 1);
        let carry = if off == 0 {
            0
        } else {
            (limbs[(off - 1) / 64] >> ((off - 1) % 64)) & 1
        };
        b as i64 + carry as i64 - (((b >> (c - 1)) << c) as i64)
    }

    fn plan<A: AffineRepr<ScalarField = Fr>>(
        name: &str,
        bases: &[A],
        scalars: &[Fr],
        device_scalars: bool,
    ) {
        assert_eq!(bases.len(), scalars.len());
        let general = scalars
            .iter()
            .filter(|s| !s.is_zero() && !s.is_one())
            .count();
        let cap = if device_scalars {
            scalars.len()
        } else {
            general
        };
        // Same per-upload bit bound as `Plan::new`: one past the longest scalar for
        // the signed carry. Device-resident scalars are unclassified and keep the
        // full-width layout.
        let recode_bits = if device_scalars {
            255
        } else {
            scalars
                .iter()
                .map(|s| s.into_bigint().num_bits() as usize + 1)
                .max()
                .unwrap_or(1)
                .min(255)
        };
        let c = window_size_for(cap, recode_bits) as usize;
        let cap = cap.max(1);
        let windows = recode_bits.div_ceil(c);
        let buckets = 1 << (c - 1);
        let mut rows = vec![0usize; windows * buckets];
        let mut live_entries = 0usize;
        let mut ones = 0usize;
        let mut live_ones = 0usize;
        let mut weights = vec![Fr::one(); windows];
        for w in 1..windows {
            weights[w] = weights[w - 1] * Fr::from(1u64 << c);
        }
        for (s, base) in scalars.iter().zip(bases) {
            if s.is_zero() {
                continue;
            }
            if s.is_one() {
                ones += 1;
                live_ones += usize::from(!base.is_zero());
                continue;
            }
            let limbs = s.into_bigint();
            let mut reconstructed = Fr::zero();
            for (w, weight) in weights.iter().enumerate() {
                let d = digit(limbs.as_ref(), w, c);
                let magnitude = d.unsigned_abs() as usize;
                let term = Fr::from(magnitude as u64) * weight;
                reconstructed += if d < 0 { -term } else { term };
                if magnitude != 0 {
                    rows[w * buckets + magnitude - 1] += 1;
                    live_entries += usize::from(!base.is_zero());
                }
            }
            assert_eq!(reconstructed, *s, "signed digits for {name}");
        }
        println!(
            "PLAN,{name},{},{cap},{general},{ones},{live_ones},{c},{windows},{buckets},{},{live_entries},{},{},{},{}",
            scalars.len(),
            rows.iter().sum::<usize>(),
            rows.iter().filter(|n| **n != 0).count(),
            rows.iter().max().unwrap(),
            rows[(windows - 1) * buckets..].iter().max().unwrap(),
            rows.chunks(buckets).filter(|w| w.iter().any(|n| *n != 0)).count(),
        );
    }

    fn gather(pk: &ProvingKey, witness: &[Fr]) {
        for m in 0..2 {
            let ptr = &pk.coeffs.row_ptr[m];
            let lengths: Vec<_> = ptr.windows(2).map(|p| (p[1] - p[0]) as usize).collect();
            // Loop slots with one row per lane and 32 lanes per SIMD group. A slot
            // in a lane whose row has ended is inactive, not another field product.
            let slots: usize = lengths
                .chunks(32)
                .map(|g| 32 * g.iter().max().unwrap())
                .sum();
            let values = &pk.coeffs.value[m];
            let trivial_coef = values
                .iter()
                .filter(|v| v.is_zero() || **v == Fr::one() || **v == -Fr::one())
                .count();
            let trivial_witness = pk.coeffs.signal[m]
                .iter()
                .filter(|s| {
                    let w = witness[**s as usize];
                    w.is_zero() || w.is_one()
                })
                .count();
            let mut general_witness_slots = 0usize;
            for (group, lengths) in lengths.chunks(32).enumerate() {
                for k in 0..*lengths.iter().max().unwrap() {
                    if lengths.iter().enumerate().any(|(lane, len)| {
                        if k >= *len {
                            return false;
                        }
                        let entry = ptr[group * 32 + lane] as usize + k;
                        let w = witness[pk.coeffs.signal[m][entry] as usize];
                        !w.is_zero() && !w.is_one()
                    }) {
                        general_witness_slots += 32;
                    }
                }
            }
            println!(
                "GATHER,{m},{},{},{slots},{trivial_coef},{trivial_witness},{general_witness_slots}",
                values.len(),
                lengths.iter().max().unwrap(),
            );
        }
    }

    pub fn run() {
        let dirs: Vec<_> = std::env::args_os().skip(1).collect();
        assert!(!dirs.is_empty(), "pass one or more circuit directories");
        println!("PLAN,job,n,cap,general,ones,live_ones,c,windows,buckets,entries,live_entries,used_rows,max_row,top_max_row,active_windows");
        println!("GATHER,matrix,nnz,max_row,simd_loop_slots,trivial_coefficients,trivial_witness_references,general_witness_slots");
        for dir in dirs {
            let dir = Path::new(&dir);
            println!("CIRCUIT,{}", dir.display());
            let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
            let witness = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
            gather(&pk, &witness);
            plan("A_g1", &pk.a_query, &witness, false);
            plan("B_g2", &pk.b_g2_query, &witness, false);
            plan("B_g1", &pk.b_g1_query, &witness, false);
            plan("L_g1", &pk.l_query, &witness[pk.n_public + 1..], false);
            let h_bases = pk.h_query.clone();
            let cpu = CpuBackend::new().prepare(pk).unwrap();
            let h = cpu
                .compute_h(&witness, &mut StageTimings::default())
                .unwrap();
            plan("H_g1", &h_bases, &h.to_host().unwrap(), true);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    audit::run();
    #[cfg(not(target_os = "macos"))]
    panic!("the Metal window selector is only built on macOS");
}
