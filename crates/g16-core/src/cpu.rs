//! The CPU backend: composes `g16-ntt` and `g16-msm` into stages 0-9.

use std::time::Instant;

use crate::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::*;
use g16_msm::{CpuMsm, MsmBackend};
use g16_ntt::{CpuNtt, Direction, NttBackend};
use g16_zkey::ProvingKey;
use rayon::prelude::*;

pub struct CpuBackend {
    pub threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        // The NTT and MSM crates both size their task counts against the global rayon
        // pool, so reporting anything else here would describe a pool nobody uses.
        Self {
            threads: rayon::current_num_threads().max(1),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        Ok(Box::new(CpuCircuit::new(pk)?))
    }
}

pub struct CpuCircuit {
    pk: ProvingKey,
    domain: Domain,
    /// snarkjs' `inc`: a primitive `2 * domain_size`-th root of unity. See
    /// [`CpuCircuit::new`] for why this and not `domain.coset_gen`.
    coset_shift: Fr,
    /// One instance for the life of the circuit: the twiddle cache is per instance, and
    /// a fresh one would rebuild the size/2 table on every one of the six transforms.
    ntt: CpuNtt,
    msm: CpuMsm,
}

impl CpuCircuit {
    fn new(pk: ProvingKey) -> Result<Self, ProveError> {
        let bad = |reason: String| ProveError::Backend {
            backend: "cpu",
            reason,
        };

        let domain = Domain::new(pk.domain_size).map_err(|e| bad(e.to_string()))?;
        // The CSR rows are indexed by evaluation point, so a domain that rounded up would
        // silently shorten every gather. Better to refuse the key than to prove garbage.
        if domain.size != pk.domain_size {
            return Err(bad(format!(
                "domain size {} is not a power of two",
                pk.domain_size
            )));
        }

        // Why a 2n-th root of unity rather than `domain.coset_gen` (= Fr::GENERATOR):
        // snarkjs evaluates A, B and C on the *odd* points of the 2n-th roots of unity,
        // `inc = Fr.w[power+1]` in groth16_prove.js, and section 9 of the zkey holds the
        // Lagrange basis of that same 2n domain restricted to its odd indices (see
        // `writeHs` in zkey_new.js, which takes `sTauG1[2i+1]`). The bases are tied to
        // that specific point set, so any other coset would pair evaluations against the
        // wrong Lagrange polynomials. `Domain::new(2n).group_gen` squares to
        // `domain.group_gen` by construction, which is exactly the relation
        // `Fr.w[power] == inc^2` that makes evaluation index i land on `inc^(2i+1)`.
        let coset_shift = Domain::new(
            domain
                .size
                .checked_mul(2)
                .ok_or_else(|| bad("domain size overflows".into()))?,
        )
        .map_err(|e| bad(format!("no 2n-th root of unity: {e}")))?
        .group_gen;

        for (m, name) in [(0usize, "A"), (1, "B")] {
            let row_ptr = &pk.coeffs.row_ptr[m];
            if row_ptr.len() != domain.size + 1 {
                return Err(bad(format!(
                    "matrix {name} has {} rows, domain size is {}",
                    row_ptr.len().saturating_sub(1),
                    domain.size
                )));
            }
            // Paid once per key, not per proof, and it turns a would-be panic deep inside
            // the parallel gather into an error at load time.
            if pk.coeffs.signal[m].iter().any(|&s| s as usize >= pk.n_vars) {
                return Err(bad(format!(
                    "matrix {name} references a signal beyond n_vars {}",
                    pk.n_vars
                )));
            }
        }

        Ok(Self {
            pk,
            domain,
            coset_shift,
            ntt: CpuNtt::new(),
            msm: CpuMsm::new(),
        })
    }

    /// Stage 0 for one matrix: `out[c] = sum over CSR row c of value * witness[signal]`.
    ///
    /// A gather rather than snarkjs' scatter. The CSR sort in `g16-zkey` is what buys
    /// this: rows are disjoint, so the loop is embarrassingly parallel with no atomics.
    fn gather(&self, m: usize, witness: &[Fr]) -> Vec<Fr> {
        let row_ptr = &self.pk.coeffs.row_ptr[m];
        let signal = &self.pk.coeffs.signal[m];
        let value = &self.pk.coeffs.value[m];

        (0..self.domain.size)
            .into_par_iter()
            .map(|c| {
                let lo = row_ptr[c] as usize;
                let hi = row_ptr[c + 1] as usize;
                let mut acc = Fr::zero();
                for k in lo..hi {
                    acc += value[k] * witness[signal[k] as usize];
                }
                acc
            })
            .collect()
    }

    fn check_witness(&self, witness: &[Fr]) -> Result<(), ProveError> {
        if witness.len() != self.pk.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.pk.n_vars,
            });
        }
        Ok(())
    }
}

impl PreparedCircuit for CpuCircuit {
    fn backend_name(&self) -> &'static str {
        "cpu"
    }
    fn n_vars(&self) -> usize {
        self.pk.n_vars
    }
    fn n_public(&self) -> usize {
        self.pk.n_public
    }
    fn domain_size(&self) -> usize {
        self.domain.size
    }
    fn key(&self) -> &ProvingKey {
        &self.pk
    }

    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        self.check_witness(witness)?;

        // Stage 0. There is no C matrix in the zkey: snarkjs' buildABC1 fills C by
        // multiplying the A and B evaluations pointwise, which is exact because the R1CS
        // constraint *is* a*b = c, so on the domain the C evaluations are that product.
        let start = Instant::now();
        let (mut a, mut b) = stage!(
            "s0 gather A and B (concurrent)",
            rayon::join(|| self.gather(0, witness), || self.gather(1, witness))
        );
        let mut c: Vec<Fr> = stage!(
            "s0 build C = A*B pointwise",
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| *x * y)
                .collect()
        );
        t.gather_us += start.elapsed().as_micros() as u64;

        // Stages 1-3, three vectors through iNTT -> coset shift -> NTT. Each transform is
        // already internally parallel, so the three run one after another rather than
        // fighting each other for the same pool.
        let mut ntt_us = 0u64;
        let mut pointwise_us = 0u64;
        for v in [&mut a, &mut b, &mut c] {
            let start = Instant::now();
            stage!("s1 iNTT", self.ntt.ntt(&self.domain, v, Direction::Inverse));
            ntt_us += start.elapsed().as_micros() as u64;

            let start = Instant::now();
            stage!(
                "s2 coset shift (distribute_powers)",
                self.ntt.distribute_powers(v, self.coset_shift)
            );
            pointwise_us += start.elapsed().as_micros() as u64;

            let start = Instant::now();
            stage!("s3 NTT", self.ntt.ntt(&self.domain, v, Direction::Forward));
            ntt_us += start.elapsed().as_micros() as u64;
        }
        t.ntt_us += ntt_us;

        // Stage 4. No division by Z. snarkjs' `joinABC` computes exactly `a*b - c` and
        // feeds it straight to the H multiexp, because the Z division is folded into the
        // section 9 bases at setup: they are the odd Lagrange polynomials of the 2n
        // domain, and P = A*B - C vanishes on the even points (those are the constraint
        // rows), so `sum_i P(inc^(2i+1)) * hExps[i]` already equals `[P(tau)]_1`, which
        // is `[H(tau) * Z(tau)]_1`. Dividing here as well would double-count the Z.
        let start = Instant::now();
        let h: Vec<Fr> = stage!(
            "s4 H = A*B - C",
            a.par_iter()
                .zip(b.par_iter())
                .zip(c.par_iter())
                .map(|((x, y), z)| *x * y - z)
                .collect()
        );
        pointwise_us += start.elapsed().as_micros() as u64;
        t.pointwise_us += pointwise_us;

        Ok(HPoly::Host(h))
    }

    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        // The CPU backend has nowhere else to keep H, so a device handle here can only
        // have come from another backend and there is nothing sane to do with it.
        let h = h.to_host().ok_or_else(|| ProveError::Backend {
            backend: "cpu",
            reason: "compute_h output came from another backend".to_string(),
        })?;
        if h.len() != self.domain.size {
            return Err(ProveError::Backend {
                backend: "cpu",
                reason: format!(
                    "h has {} entries, domain size is {}",
                    h.len(),
                    self.domain.size
                ),
            });
        }

        // Section 8 covers the private wires only: witness[0] is the constant 1 and
        // witness[1..=n_public] are the public inputs, both of which the verifier folds
        // into L_bar through IC instead.
        let l_scalars = &witness[self.pk.n_public + 1..];
        if l_scalars.len() != self.pk.l_query.len() {
            return Err(ProveError::Backend {
                backend: "cpu",
                reason: format!(
                    "l_query has {} bases, private witness is {} long",
                    self.pk.l_query.len(),
                    l_scalars.len()
                ),
            });
        }

        let start = Instant::now();
        // The five MSMs overlap through nested joins. Keep the nesting, but not for the
        // reason this comment used to give: it claimed four of the MSMs were cheap
        // because their scalars are mostly 0 or 1, leaving the dense H MSM running alone.
        // Both halves are false, and bench/results/profiling/ has the numbers.
        //
        // Only 1.80% of witness scalars are 0 or 1 on the two largest circuits, 4.12% at
        // the sparsest point on the ladder. So no witness MSM is cheap and none of them
        // drains early. The nesting itself is worth 1.01x at 140,261 constraints, not the
        // structural win implied here. It earns its keep at the bottom of the ladder
        // instead, 1.77x at 2 constraints, and by lifting occupancy on a pool that
        // otherwise idles: a proof measures 9.41 busy cores out of 12.
        //
        // It therefore stays because it raises occupancy and costs nothing, which is a
        // much weaker claim than the one it replaces.
        // The five annotations below overlap: these MSMs run at the same time. Their
        // durations are wall-clock windows, not disjoint costs, and they will sum to more
        // than the enclosing region. `examples/msm_shape.rs` measures them one at a time
        // when the disjoint cost is what is wanted.
        let ((a_g1, b_g2), ((b_g1, l_g1), h_g1)) = rayon::join(
            || {
                rayon::join(
                    || stage!("s5 MSM A -> G1", self.msm.msm_g1(&self.pk.a_query, witness)),
                    || {
                        stage!(
                            "s6 MSM B -> G2",
                            self.msm.msm_g2(&self.pk.b_g2_query, witness)
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || {
                                stage!(
                                    "s7 MSM B -> G1",
                                    self.msm.msm_g1(&self.pk.b_g1_query, witness)
                                )
                            },
                            || {
                                stage!(
                                    "s8 MSM L -> G1",
                                    self.msm.msm_g1(&self.pk.l_query, l_scalars)
                                )
                            },
                        )
                    },
                    // Every one of the `domain_size` bases is used. The "only n-1 are
                    // nonzero" rule belongs to the coefficient-form convention; in
                    // snarkjs' evaluation form all n entries are generically nonzero.
                    || stage!("s9 MSM H -> G1", self.msm.msm_g1(&self.pk.h_query, h)),
                )
            },
        );
        t.msm_us += start.elapsed().as_micros() as u64;

        Ok(MsmOutputs {
            a_g1,
            b_g2,
            b_g1,
            l_g1,
            h_g1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coset the prover evaluates on is fixed by the zkey's section 9 bases, so these
    /// two identities are a contract with snarkjs, not an implementation detail:
    /// `shift^2` must be the domain's own generator (so evaluation index `i` lands on
    /// `shift^(2i+1)`) and `shift^n` must be -1 (so the coset is the *odd* half of the
    /// 2n-th roots of unity, which is the half section 9 was built from).
    #[test]
    fn the_coset_shift_is_the_odd_half_of_the_2n_th_roots() {
        for log_n in 0..12u32 {
            let n = 1usize << log_n;
            let domain = Domain::new(n).unwrap();
            let shift = Domain::new(2 * n).unwrap().group_gen;
            assert_eq!(shift * shift, domain.group_gen, "n = {n}");
            assert_eq!(shift.pow([n as u64]), -Fr::one(), "n = {n}");
            // And therefore the coset misses the domain entirely, which is what makes
            // A*B - C nonzero on it.
            assert_ne!(shift.pow([n as u64]), Fr::one(), "n = {n}");
        }
    }
}
