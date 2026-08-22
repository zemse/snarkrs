//! The WebGPU implementation of `g16_core::Backend`: the file that turns nine stages of
//! kernels into a proof.
//!
//! Everything here is wiring. Stages 0 to 4 are [`crate::stages`], stages 5 to 9 are
//! [`crate::batch`], and what happens in this file is the split between witness-independent
//! work, which must all be hoisted into [`WgpuProver::prepare`], and per-proof work, which
//! must touch nothing shared and mutable.
//!
//! # Native only, and why that is not a hedge
//!
//! `g16_core::PreparedCircuit` is a synchronous trait and every wgpu readback is
//! fundamentally asynchronous, so something has to block. On native, `pollster::block_on`
//! blocks a thread while `device.poll` drives the queue, which is exactly what the readback
//! is documented to need. In a browser there is no thread to block: `poll` is a no-op, the
//! only thing that fires a `mapAsync` callback is a yield to the event loop, and
//! `pollster::block_on` compiles for wasm32 and then hangs the tab.
//!
//! So this module is `cfg`-gated off wasm32 rather than made to compile and deadlock. The
//! browser gets an `async fn prove` of its own at U13, over the same [`crate::stages`] and
//! [`crate::batch`] that this file calls; nothing below is on the browser's path and nothing
//! below has to be rewritten for it.
//!
//! # What `prepare` hoists, and what it cannot
//!
//! The trait's own words: "Witness-independent work: sort section 4 into CSR, build twiddles,
//! and for a GPU backend upload every base vector and keep it resident."  All of that happens
//! in [`WgpuCircuit::new`]. At `js_16x16_d32` it is about 26 MB of CSR, 24 MB of twiddle and
//! coset tables and 62 MB of base vectors.
//!
//! One thing that cannot be hoisted to the *backend* the way `g16-metal` hoists it: the NTT
//! pipelines. WGSL has no runtime-sized `var<workgroup>`, so the tile size is a compile-time
//! constant and the six transforms are compiled for one specific `log_n`. `HStages` therefore
//! lives on the circuit and two keys of different domain sizes compile it twice.
//! [`crate::batch::MsmBatch`] does not have that problem, so the three MSM modules and their
//! fifteen pipelines are built once per process and shared through an `Arc`.
//!
//! # There is no CPU fallback, and no size gate
//!
//! Every stage runs on the device and every one is checked against the CPU backend element by
//! element, so [`WgpuCircuit::backend_name`] can honestly say "wgpu" for the whole proof.
//! There is deliberately no "small circuits go to the CPU" gate either, for the reason
//! `g16-metal` gives: a gate that quietly ran the other backend would make
//! `g16 prove --backend wgpu` report a number that belongs to somebody else. The crossover is
//! a measurement to publish, and `tests/proof.rs` publishes it.

use std::sync::Arc;
use std::time::Instant;

use g16_core::{Backend, HPoly, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::{Fr, One, Zero};
use g16_zkey::ProvingKey;

use crate::batch::{G1Bases, G2Bases, Group, Job, MontConvert, MsmBatch, Source};
use crate::device::{bad, LimitsProfile, WgpuBackend};
use crate::pipelines::PrepareCost;
use crate::stages::{HStages, Stage4, WgpuHandle};

/// Where the time in [`WgpuProver::prepare`] went, in microseconds.
///
/// Kept per circuit rather than printed, so a caller can report it without re-running the
/// preparation to time it. `G16_WGPU_PREPARE=1` prints it as well.
#[derive(Clone, Copy, Debug, Default)]
pub struct CircuitCost {
    /// [`HStages::new`]: three shader modules, the CSR, both twiddle tables and the coset
    /// powers.
    pub stages_us: u64,
    /// The five base vectors repacked from ark's 72 and 136 byte affines and uploaded.
    pub bases_us: u64,
    pub total_us: u64,
}

/// One device, one queue, and the three MSM shader modules, built once per process.
///
/// Cloning is not offered: what makes this expensive is the pipeline compiles hanging off it,
/// and [`Self::prepare`] hands them to every circuit through an `Arc` rather than duplicating
/// anything.
pub struct WgpuProver {
    device: Arc<WgpuBackend>,
    msm: Arc<MsmBatch>,
    cost: PrepareCost,
}

impl WgpuProver {
    /// Opens an adapter and a device at the profile named by `G16_WGPU_LIMITS`, which
    /// defaults to the **browser floor** and not this adapter's own limits.
    ///
    /// That default is the point of the crate and it costs throughput: `Raised` on this M2
    /// Max is 4 GiB of storage binding against 128 MiB and 32 KiB of workgroup storage
    /// against 16 KiB. A kernel that only fits the raised profile is a kernel that fails in a
    /// stock browser, and native `cargo test` will not notice, so the number this backend
    /// reports by default is the one a browser could reproduce.
    ///
    /// Fails rather than falling back to anything. A benchmark that quietly measures a
    /// different backend is worse than no number at all.
    pub fn new() -> Result<Self, ProveError> {
        Self::with_profile(LimitsProfile::from_env()?)
    }

    pub fn with_profile(profile: LimitsProfile) -> Result<Self, ProveError> {
        let device = pollster::block_on(WgpuBackend::with_profile(profile))?;
        Self::with_device(Arc::new(device))
    }

    /// Same, on a device the caller already opened. `tests/proof.rs` uses it to share one
    /// adapter across every artifact in a binary, which is 5 fewer device creations.
    pub fn with_device(device: Arc<WgpuBackend>) -> Result<Self, ProveError> {
        let start = Instant::now();
        let msm = MsmBatch::new(&device)?;
        let cost = msm.cost();
        if std::env::var_os("G16_WGPU_PREPARE").is_some() {
            eprintln!(
                "wgpu backend: {} MSM modules, {} pipelines, {} us of naga and {} us of \
                 pipeline creation, {} us wall",
                cost.modules,
                cost.pipelines,
                cost.module_us,
                cost.pipeline_us,
                start.elapsed().as_micros(),
            );
        }
        Ok(Self {
            device,
            msm: Arc::new(msm),
            cost,
        })
    }

    pub fn device(&self) -> &Arc<WgpuBackend> {
        &self.device
    }

    /// What the three MSM modules cost to compile, which is paid once per process.
    pub fn msm_compile_cost(&self) -> PrepareCost {
        self.cost
    }

    /// The shared MSM pipelines, for a report. `tests/proof.rs` reads
    /// [`MsmBatch::last_readback_bytes`] through it, which is the only way to hold this
    /// backend to design §3's 64 KiB per-proof readback budget from outside.
    pub fn msm(&self) -> &MsmBatch {
        &self.msm
    }
}

impl Backend for WgpuProver {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn prepare(&self, pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        Ok(Box::new(WgpuCircuit::new(
            Arc::clone(&self.device),
            Arc::clone(&self.msm),
            pk,
        )?))
    }
}

/// A key whose witness-independent data is device resident: the CSR, both twiddle tables, the
/// coset powers and all five base vectors.
///
/// # Concurrency
///
/// Nothing here is mutated by a proof. The per-proof scratch is pooled behind two mutexes,
/// one owned by [`HStages`] and one by [`MsmBatch`], so two concurrent proofs cannot be
/// handed the same buffer. What they *would* share is the device's single uncaptured-error
/// slot and the device-wide `onSubmittedWorkDone` fence, so both `compute_h` and `msms` hold
/// [`WgpuBackend::exclusive`] for their whole GPU section. Concurrent proofs against one
/// circuit are therefore correct and serialised rather than parallel, which is what that
/// method's doc comment argues is the only honest option on one queue.
pub struct WgpuCircuit {
    pk: ProvingKey,
    device: Arc<WgpuBackend>,
    msm: Arc<MsmBatch>,
    stages: HStages,
    a_bases: G1Bases,
    b_g1_bases: G1Bases,
    b_g2_bases: G2Bases,
    l_bases: G1Bases,
    h_bases: G1Bases,
    cost: CircuitCost,
}

impl WgpuCircuit {
    fn new(
        device: Arc<WgpuBackend>,
        msm: Arc<MsmBatch>,
        pk: ProvingKey,
    ) -> Result<Self, ProveError> {
        // Shape checks first. `crate::batch` would reject an out-of-range job later, but by
        // then the message names buffer offsets rather than the section of the zkey that is
        // the wrong length, and the mismatch is a property of the key and not of the proof.
        let n_vars = pk.n_vars;
        for (name, got) in [
            ("a_query", pk.a_query.len()),
            ("b_g1_query", pk.b_g1_query.len()),
            ("b_g2_query", pk.b_g2_query.len()),
        ] {
            if got != n_vars {
                return Err(bad(format!("{name} has {got} bases, n_vars is {n_vars}")));
            }
        }
        if pk.h_query.len() != pk.domain_size {
            return Err(bad(format!(
                "h_query has {} bases, domain size is {}",
                pk.h_query.len(),
                pk.domain_size
            )));
        }
        let want_l = n_vars
            .checked_sub(pk.n_public + 1)
            .ok_or_else(|| bad(format!("n_public {} exceeds n_vars {n_vars}", pk.n_public)))?;
        if pk.l_query.len() != want_l {
            return Err(bad(format!(
                "l_query has {} bases, the private witness is {want_l} long",
                pk.l_query.len()
            )));
        }

        let t_all = Instant::now();
        // Stages 0 to 4, including the three shader modules. This also validates the key the
        // way the CPU backend does (power-of-two domain, CSR row count, every signal index
        // below n_vars), so a bad key is an error here rather than an out-of-range GPU read
        // later.
        let t0 = Instant::now();
        let stages = HStages::new(&device, &pk)?;
        let stages_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let a_bases = G1Bases::upload(&device, &pk.a_query)?;
        let b_g1_bases = G1Bases::upload(&device, &pk.b_g1_query)?;
        let b_g2_bases = G2Bases::upload(&device, &pk.b_g2_query)?;
        let l_bases = G1Bases::upload(&device, &pk.l_query)?;
        let h_bases = G1Bases::upload(&device, &pk.h_query)?;
        let bases_us = t1.elapsed().as_micros() as u64;

        let cost = CircuitCost {
            stages_us,
            bases_us,
            total_us: t_all.elapsed().as_micros() as u64,
        };
        if std::env::var_os("G16_WGPU_PREPARE").is_some() {
            eprintln!(
                "wgpu prepare: stages {} us, bases {} us ({} MB), total {} us (domain {}, \
                 n_vars {n_vars})",
                cost.stages_us,
                cost.bases_us,
                (a_bases.bytes()
                    + b_g1_bases.bytes()
                    + b_g2_bases.bytes()
                    + l_bases.bytes()
                    + h_bases.bytes())
                    / 1_000_000,
                cost.total_us,
                pk.domain_size,
            );
        }

        Ok(Self {
            pk,
            device,
            msm,
            stages,
            a_bases,
            b_g1_bases,
            b_g2_bases,
            l_bases,
            h_bases,
            cost,
        })
    }

    /// What [`WgpuProver::prepare`] cost for this key.
    pub fn prepare_cost(&self) -> CircuitCost {
        self.cost
    }

    /// Device bytes this key holds resident: the five base vectors.
    pub fn base_bytes(&self) -> u64 {
        self.a_bases.bytes()
            + self.b_g1_bases.bytes()
            + self.b_g2_bases.bytes()
            + self.l_bases.bytes()
            + self.h_bases.bytes()
    }

    pub fn stages(&self) -> &HStages {
        &self.stages
    }

    pub fn msm(&self) -> &MsmBatch {
        &self.msm
    }

    /// Stages 0 to 4 with stage 4's implementation chosen, for the test that runs both.
    pub fn compute_h_with(
        &self,
        witness: &[Fr],
        mode: Stage4,
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        let _gpu = self.device.exclusive();
        pollster::block_on(self.stages.compute_h_with(&self.device, witness, mode, t))
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

/// Scalars that are neither 0 nor 1, over the whole witness and over the private tail.
///
/// One pass rather than two, and it is the only host work `msms` does on the witness: the
/// limbs are already on the device from stage 0. `crate::msm::DigitPlan` sizes both the
/// window width and `cap` from this count, and `cap` bounds a storage write that WebGPU drops
/// in silence when it goes out of range, so **an undercount loses entries with no error
/// anywhere**. Counting rather than estimating is not optional. At 140,824 variables this is
/// about 0.2 ms.
fn general_counts(witness: &[Fr], private_from: usize) -> (u32, u32) {
    let mut all = 0u32;
    let mut private = 0u32;
    for (i, x) in witness.iter().enumerate() {
        if !(x.is_zero() || x.is_one()) {
            all += 1;
            if i >= private_from {
                private += 1;
            }
        }
    }
    (all, private)
}

impl PreparedCircuit for WgpuCircuit {
    /// Always "wgpu", and that is a claim about what ran: no stage in this backend has a CPU
    /// fallback, so the name cannot be describing a proof the CPU did.
    fn backend_name(&self) -> &'static str {
        "wgpu"
    }
    fn n_vars(&self) -> usize {
        self.pk.n_vars
    }
    fn n_public(&self) -> usize {
        self.pk.n_public
    }
    fn domain_size(&self) -> usize {
        self.pk.domain_size
    }
    fn key(&self) -> &ProvingKey {
        &self.pk
    }

    fn compute_h(&self, witness: &[Fr], t: &mut StageTimings) -> Result<HPoly, ProveError> {
        let _gpu = self.device.exclusive();
        pollster::block_on(self.stages.compute_h(&self.device, witness, t))
    }

    fn msms(
        &self,
        witness: &[Fr],
        h: &HPoly,
        t: &mut StageTimings,
    ) -> Result<MsmOutputs, ProveError> {
        self.check_witness(witness)?;
        if h.len() != self.pk.domain_size {
            return Err(bad(format!(
                "h has {} entries, the domain size is {}",
                h.len(),
                self.pk.domain_size
            )));
        }
        let start = Instant::now();
        let _gpu = self.device.exclusive();

        let private_from = self.pk.n_public + 1;
        let (g_all, g_private) = general_counts(witness, private_from);
        let n_vars = self.pk.n_vars as u32;
        let l_len = self.l_bases.len() as u32;
        let domain = self.pk.domain_size as u32;

        // Four G1 jobs and one G2, in stage order, so the results come back as
        // [A, B-G2, B-G1, L, H] and `MsmOutputs` is filled positionally.
        let witness_jobs = [
            Job::G1 {
                bases: &self.a_bases,
                base_off: 0,
            },
            Job::G2 {
                bases: &self.b_g2_bases,
                base_off: 0,
            },
            Job::G1 {
                bases: &self.b_g1_bases,
                base_off: 0,
            },
        ];
        let l_jobs = [Job::G1 {
            bases: &self.l_bases,
            base_off: 0,
        }];
        let h_jobs = [Job::G1 {
            bases: &self.h_bases,
            base_off: 0,
        }];

        // The normal path: `H` is the buffer stage 4 wrote and the witness is the one stage 0
        // uploaded, so nothing crosses the host boundary and stage 9's scalars never exist in
        // host memory at all. `HPoly::device_handle` returns `None` for a handle from another
        // backend, so a Metal handle reaching here is a fallback rather than a reinterpreted
        // pointer.
        let handle = h.device_handle::<WgpuHandle>(crate::stages::TAG);
        let host_h = h.to_host();
        let (mont, groups) = match handle {
            Some(handle) => {
                if handle.len() != self.pk.domain_size {
                    return Err(bad(format!(
                        "device h holds {} entries, the domain size is {}",
                        handle.len(),
                        self.pk.domain_size
                    )));
                }
                let w = handle.witness_std();
                (
                    Some(MontConvert {
                        src: handle.witness_mont(),
                        dst: w,
                        n: n_vars,
                    }),
                    vec![
                        Group {
                            scalars: Source::Device {
                                buf: w,
                                general: Some(g_all),
                            },
                            scalar_off: 0,
                            n: n_vars,
                            jobs: &witness_jobs,
                        },
                        Group {
                            scalars: Source::Device {
                                buf: w,
                                general: Some(g_private),
                            },
                            // Section 8 covers the private wires only: witness[0] is the
                            // constant 1 and witness[1..=n_public] are the public inputs,
                            // which the verifier folds in through IC instead.
                            scalar_off: private_from as u32,
                            n: l_len,
                            jobs: &l_jobs,
                        },
                        Group {
                            scalars: Source::Device {
                                buf: handle.h_std(),
                                // Nobody on the host has seen these values, so the only safe
                                // answer is "all of them are general". Overestimating costs a
                                // wider window and a larger entry array; underestimating
                                // loses entries in silence. See `crate::batch::Source`.
                                general: None,
                            },
                            scalar_off: 0,
                            n: domain,
                            jobs: &h_jobs,
                        },
                    ],
                )
            }
            None => {
                // A `compute_h` from another backend. Not the proving path, and the only
                // thing it is really for is letting a test hold stages 5 to 9 to the CPU's
                // stages 0 to 4 without the device's own H in the way.
                let host_h = host_h.ok_or_else(|| {
                    bad("compute_h output is neither a wgpu handle nor a host vector")
                })?;
                (
                    None,
                    vec![
                        Group {
                            scalars: Source::Host(witness),
                            scalar_off: 0,
                            n: n_vars,
                            jobs: &witness_jobs,
                        },
                        Group {
                            scalars: Source::Host(witness),
                            scalar_off: private_from as u32,
                            n: l_len,
                            jobs: &l_jobs,
                        },
                        Group {
                            scalars: Source::Host(host_h),
                            scalar_off: 0,
                            n: domain,
                            jobs: &h_jobs,
                        },
                    ],
                )
            }
        };

        let out = pollster::block_on(self.msm.run(&self.device, mont, &groups))?;
        if out.len() != 5 {
            return Err(bad(format!(
                "the MSM batch returned {} results for 5 jobs",
                out.len()
            )));
        }
        let outputs = MsmOutputs {
            a_g1: out[0].g1()?,
            b_g2: out[1].g2()?,
            b_g1: out[2].g1()?,
            l_g1: out[3].g1()?,
            h_g1: out[4].g1()?,
        };
        t.msm_us += start.elapsed().as_micros() as u64;
        Ok(outputs)
    }
}

/// `Backend` and `PreparedCircuit` both require `Send + Sync`, and the whole reason `prepare`
/// is a separate call is that one resident key is proved against from many threads. If a
/// `!Sync` field ever lands on either struct this stops compiling here rather than at the
/// `Box<dyn PreparedCircuit>` coercion, where the error names the trait object instead of the
/// field.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WgpuProver>();
    assert_send_sync::<WgpuCircuit>();
};
