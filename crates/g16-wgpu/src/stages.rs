//! Stages 0 to 4 as one submit: the CSR gather, the six NTTs, the coset shift and
//! `H = A*B - C`, with everything resident on the device.
//!
//! # Why this is a stage *group* and not five methods
//!
//! `g16_core::PreparedCircuit::compute_h` is a group because a trait with an `ntt()` method
//! would force the three domain vectors back through host memory between every one of the
//! six transforms, and a GPU backend built against it would measure the bus. So everything
//! below runs in one command encoder with one submit, and the result never leaves the
//! device: the value handed back is [`g16_core::HPoly::Device`] carrying a [`WgpuHandle`],
//! and stage 9's MSM will read the buffer stage 4 wrote.
//!
//! # The dispatch budget, which is what decides whether that is worth doing
//!
//! Design §3 measures an empty submit plus `onSubmittedWorkDone` at 0.1 to 0.3 ms and an
//! extra dispatch in an already open encoder at 2 to 3 microseconds, a factor of 50 to 100.
//! An empty submit plus its fence re-measures here at **22 to 24 microseconds in release**
//! and about 70 in debug, so the design's figure is pessimistic by 4x and the ratio it rests
//! on is intact. At the largest artifact (2^18) this encodes **20 dispatches**: one gather,
//! three per transform for six transforms, and one `h_join`. Twenty separate submits would
//! be 0.5 ms of pure fence latency against about 0.05 ms of extra encoding.
//! `tests/stages.rs` asserts the submit count is exactly one, counted through
//! [`crate::device::WgpuBackend::submit`] rather than asserted in a comment.
//!
//! Note the 20 and not Metal's 13. Two differences, both measured rather than chosen: the
//! NTT tile here is 8 rather than Metal's 10, so a 2^18 transform is 6 + 6 + 6 rather than
//! 9 + 9 (`gen::ntt::PREFERRED_FUSED`, where filling the workgroup budget measured 5x
//! slower), and stage 4 is its own dispatch rather than fused ([`Stage4`], where the fusion
//! measured 5% to 9% slower).
//!
//! # What the host still does per proof
//!
//! One limb split per witness entry (`PackedFr::from_fr`, four `u64` into eight `u32`, no
//! arithmetic) and one `queue.write_buffer`. Design §3 leans on that being *only* a limb
//! split: `g16-metal` uploads `PackedScalar`, which is a Montgomery reduction per scalar,
//! and in the browser rayon-core falls back to a sequential pool so 140,000 reductions would
//! run on one thread. Standard form is derived on the device instead, by `fr_mont_to_std` at
//! U8 for the witness and by stage 4 here for `H`.
//!
//! Everything else the kernels read is witness independent and is uploaded once in
//! [`HStages::new`]: the concatenated CSR, both twiddle tables and the coset power table.
//!
//! # Ordering, which is this module's responsibility on both paths
//!
//! Stage 4 computes `H[i] = A[i]*B[i] - C[i]` where `A` and `B` must already hold their
//! **coset evaluations**. The encoder runs A's chain, then B's, then C's, and WebGPU orders
//! dispatches within one compute pass and makes each one's writes visible to the next, so by
//! the time stage 4 runs the other two are finished. Nothing in [`crate::ntt`] can check
//! this: `Epilogue::Join` will happily join against a half-transformed A, and `h_join` will
//! happily read one.

use std::sync::{Arc, Mutex};

use g16_core::{HPoly, ProveError, StageTimings};
use g16_field::{Domain, Fr};
use g16_gpu_layout::{PackedFr, PackedScalar, LIMBS};
use g16_zkey::ProvingKey;
use web_time::Instant;

use crate::device::{bad, WgpuBackend};
use crate::gather::{fr_buffer, fr_from_words, CsrTables, GatherAbc};
use crate::ntt::{Direction, Epilogue, Ntt, NttTables, Scale, Transform};
use crate::params::ParamRing;
use crate::pointwise::HJoin;
use crate::readback::Readback;

/// The tag on [`g16_core::HPoly::Device`] values produced here.
///
/// A handle carrying any other tag came from a different backend and must not be
/// dereferenced as a [`WgpuHandle`]; `HPoly::device_handle` enforces that, so a Metal handle
/// reaching this backend is a `None` rather than a reinterpreted pointer.
pub const TAG: &str = "wgpu";

/// Bytes one `Fr` occupies on the device.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// Which implementation of stage 4 runs.
///
/// # The default is not the one design §4 and §8 specify, because the fusion loses
///
/// Both are implemented, both are checked against `g16_core::cpu` on every artifact, and
/// the design says to ship [`Self::Fused`]. Measured GPU wall time for one whole
/// `compute_h`, interleaved, nine reps each, medians, M2 Max through naga to MSL,
/// `tests/stages.rs::the_fusion_is_measured_and_not_assumed`:
///
/// ```text
/// artifact       domain  batches   fused us   alone us   ratio
/// tiny_mul       2^3           1        744        739   1.007
/// js_1x1_d8      2^12          2       2832       2693   1.052
/// js_2x2_d16     2^14          2       3138       2933   1.070
/// js_2x2_d32     2^15          2       5002       4596   1.088
/// js_8x8_d32     2^17          3      13949      12857   1.085
/// js_16x16_d32   2^18          3      22469      20783   1.081
/// ```
///
/// The fusion is 5.2% to 8.8% slower on every artifact above 2^3, reproducibly. Six runs
/// (three debug, three release) put the median between 1.072 and 1.081 and none has put any
/// artifact above 2^3 below 1.04. The 2^3 row is nearly all fence and dispatch overhead (an
/// empty submit plus its fence measures 22 us in release and 70 in debug) and has swung
/// between 1.01 and 2.04 across runs, so it is printed and not read. The rows that carry
/// the result are 2^17 and 2^18, which have reproduced at 1.08 in every run.
///
/// This is the fourth design decision in a row inherited from `g16-metal` that did not
/// survive being measured, after `gather_abc`'s workgroup size, the NTT's, and the NTT's
/// fused-pass count.
///
/// The mechanism is in the doc comment on that test. The short version is that the design
/// counts bytes moved and the last batch of a multi-batch transform is **strided**, so the
/// four extra streams the join adds are uncoalesced where `h_join` reads and writes them
/// contiguously. That is a hypothesis with the right sign, not a proof.
///
/// naga to MSL on Apple silicon. Chrome compiles through Tint, so U14 re-runs this; the test
/// asserts that the *default* is the faster of the two rather than asserting today's answer,
/// so a flip fails loudly instead of shipping quietly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage4 {
    /// Fused into the store epilogue of the last batch of C's forward transform.
    ///
    /// Saves one write of C and one read of A, B and C on paper, one dispatch, and the
    /// whole `h_join` module. Measured slower anyway; see the table above. Kept, selectable
    /// and tested, because the ranking is a naga-to-MSL measurement on one machine and
    /// because it is the shape `g16-metal` ships.
    Fused,
    /// The standalone `h_join` kernel, with C's forward transform written out in full first.
    ///
    /// The default, on the measurement above. Also the only device implementation of stage 4
    /// that shares nothing with the NTT but the `Fr` prelude, which is what makes running
    /// both paths in `tests/stages.rs` two independent statements rather than one repeated.
    ///
    /// The cost it does carry is at prepare time, not per proof: its module is 30.3 KiB and
    /// takes **76 ms of naga cold and 2.2 ms warm**, against 1.7 ms saved per proof at 2^18.
    /// Both numbers are real and they measure different things: Metal keeps an on-disk
    /// function cache, so 76 ms is what a shader nobody has ever compiled on this machine
    /// costs and 2.2 ms is every run after that. U6 hit the same 25x gap on the NTT module.
    /// Break-even against the fused path is one proof warm and about 45 cold. In Chrome,
    /// where a 30 KiB module measures 7 to 9 ms, it is under 5 either
    /// way. A prover that only ever proves once, on a cold cache, natively, would be better
    /// off with [`Self::Fused`] and no `HJoin` at all; U11 can make that call with a
    /// whole-proof denominator in front of it.
    #[default]
    Standalone,
}

// ---------------------------------------------------------------------------
// Per-proof scratch, pooled
// ---------------------------------------------------------------------------

/// One in-flight proof's device memory: the three domain vectors, one transform temporary,
/// the packed witness, the two H outputs, and the parameter ring.
///
/// **`t` is not an optimisation, it is required.** The NTT head reads `SRC[reverse(i)]` and
/// writes `DST[i]`, and the reversed index of one workgroup's slice lands in another's, so
/// an in-place head reads values a different workgroup has already overwritten. Nothing in
/// `crate::ntt` can check that `src != dst`: wgpu offers no buffer identity comparison and
/// WebGPU will happily bind one buffer twice. So each vector ping-pongs `v -> t` on the
/// inverse transform and `t -> v` on the forward one.
///
/// The **parameter ring lives here rather than on the circuit**, and that is the
/// concurrency-correctness point of this whole struct. A ring has a write cursor, so two
/// proofs sharing one would interleave their parameter blocks and each would dispatch with
/// the other's row range. `PreparedCircuit` is `&self` and explicitly safe to prove with
/// concurrently, so anything with per-proof state has to be checked out, not shared.
///
/// Pooled rather than allocated per proof. Design §3 keeps the pool even though heliax
/// measured wgpu buffer creation as sub-microsecond, because the cost being avoided is
/// first-touch page faulting, which Metal measured at 15.2 GB/s cold against 54.8 GB/s warm.
/// That figure has not been re-measured on WebGPU and pooling costs nothing either way; what
/// it definitely buys is the concurrency property above.
struct Scratch {
    witness: wgpu::Buffer,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    c: wgpu::Buffer,
    t: wgpu::Buffer,
    h_mont: wgpu::Buffer,
    h_std: wgpu::Buffer,
    ring: ParamRing,
    /// Host staging for the witness limb split, kept across proofs.
    ///
    /// Same reasoning as `ParamRing`'s own staging vector: this is 4.5 MB at 140,824
    /// variables and reallocating it per proof would be a larger share of the 0.9 ms of
    /// host work than the copy itself. It is also per proof, so it belongs here and not on
    /// the circuit.
    pack: Vec<u32>,
}

type Pool = Arc<Mutex<Vec<Scratch>>>;

// ---------------------------------------------------------------------------
// Per-key pipelines and tables
// ---------------------------------------------------------------------------

/// Everything stages 0 to 4 need that does not depend on the witness: three shader modules,
/// the CSR, the twiddle and coset tables, and the scratch pool.
///
/// Built once per key. The NTT pipelines are compiled for one specific `log_n`, because
/// WGSL has no runtime-sized `var<workgroup>` and the tile size is therefore a compile-time
/// constant, so this cannot be shared across keys of different domains the way `g16-metal`'s
/// `HStages` is.
///
/// Nothing here is mutated by a proof. The pool is behind a mutex and hands out an exclusive
/// [`Scratch`] per in-flight proof, so two concurrent `compute_h` calls against one
/// `HStages` touch no common mutable state.
pub struct HStages {
    gather: GatherAbc,
    ntt: Ntt,
    join: HJoin,
    csr: CsrTables,
    tables: NttTables,
    domain: Domain,
    n_vars: usize,
    pool: Pool,
}

impl HStages {
    /// Compiles the three modules (stage 0, the NTT, stage 4) and uploads every
    /// witness-independent table for `pk`.
    ///
    /// This is the whole of the "warm versus cold" distinction for stages 0 to 4: at
    /// `js_16x16_d32` it uploads roughly 26 MB of CSR plus 24 MB of tables, and a prover
    /// that called it per proof would measure its own key upload.
    pub fn new(backend: &WgpuBackend, pk: &ProvingKey) -> Result<Self, ProveError> {
        let domain = Domain::new(pk.domain_size).map_err(|e| bad(e.to_string()))?;
        // The CSR rows are indexed by evaluation point, so a domain that rounded up would
        // silently shorten every gather. Refuse the key instead of proving garbage.
        if domain.size != pk.domain_size {
            return Err(bad(format!(
                "domain size {} is not a power of two",
                pk.domain_size
            )));
        }
        if pk.n_vars == 0 {
            return Err(bad("the proving key has no variables"));
        }

        let csr = CsrTables::from_coefficients(backend, &pk.coeffs, domain.size, pk.n_vars)?;
        let tables = NttTables::new(backend, domain.size)?;
        let gather = GatherAbc::new(backend)?;
        let ntt = Ntt::new(backend, domain.log_size)?;
        let join = HJoin::new(backend)?;

        Ok(Self {
            gather,
            ntt,
            join,
            csr,
            tables,
            domain,
            n_vars: pk.n_vars,
            pool: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn domain_size(&self) -> usize {
        self.domain.size
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// snarkjs' `inc`, a primitive `2n`-th root of unity. Not `Domain::coset_gen`; see
    /// `g16_core::cpu::CpuCircuit::new` for why, which is a contract with the zkey's section
    /// 9 bases rather than a free choice.
    pub fn coset_shift(&self) -> Fr {
        self.tables.coset_shift()
    }

    /// The `(s0, k)` pass split each of the six transforms runs, for reporting.
    pub fn pass_batches(&self) -> Vec<(u32, u32)> {
        self.ntt.batches().iter().map(|b| (b.s0, b.k)).collect()
    }

    /// Dispatches one [`Self::compute_h`] encodes at `mode`.
    ///
    /// The gather is one dispatch below a 2^23 domain and the standalone join one below
    /// 2^23, but both are computed rather than assumed so the number stays right when a
    /// bigger artifact arrives.
    pub fn dispatches(&self, mode: Stage4) -> u32 {
        let n = self.domain.size as u32;
        let joins = match mode {
            Stage4::Fused => 0,
            Stage4::Standalone => self.join.dispatches(n),
        };
        self.gather.dispatches(n) + 6 * self.ntt.dispatches() as u32 + joins
    }

    /// Parameter ring slots one proof consumes. Same count as [`Self::dispatches`], because
    /// every dispatch takes exactly one block.
    fn ring_slots(&self) -> u32 {
        self.dispatches(Stage4::Standalone).max(1)
    }

    /// Scratch for one proof, from the pool or newly allocated.
    ///
    /// Sized for the standalone path's ring even when the fused path runs, because a pooled
    /// scratch outlives the mode it was first created under and a ring that is one slot
    /// short fails at `push` rather than at allocation.
    fn take_scratch(&self, backend: &WgpuBackend) -> Result<Scratch, ProveError> {
        if let Some(s) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(s);
        }
        let n = self.domain.size as u32;
        Ok(Scratch {
            witness: fr_buffer(backend, "g16 witness", self.n_vars as u32)?,
            a: fr_buffer(backend, "g16 a", n)?,
            b: fr_buffer(backend, "g16 b", n)?,
            c: fr_buffer(backend, "g16 c", n)?,
            t: fr_buffer(backend, "g16 t", n)?,
            h_mont: fr_buffer(backend, "g16 h_mont", n)?,
            h_std: fr_buffer(backend, "g16 h_std", n)?,
            ring: ParamRing::new(backend, "g16 stage 0-4 params", self.ring_slots())?,
            pack: Vec::with_capacity(self.n_vars * LIMBS),
        })
    }

    /// Device bytes one pooled scratch set holds: six domain vectors plus the witness, at 32
    /// bytes an element. 52.3 MiB at `js_16x16_d32`.
    pub fn scratch_bytes(&self) -> u64 {
        (6 * self.domain.size as u64 + self.n_vars as u64) * FR_BYTES
    }

    /// Stages 0 to 4 in one submit, at [`Stage4::default`]. `H` stays on the device.
    pub async fn compute_h(
        &self,
        backend: &WgpuBackend,
        witness: &[Fr],
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        self.compute_h_with(backend, witness, Stage4::default(), t)
            .await
    }

    /// Same, with stage 4's implementation chosen.
    ///
    /// # Timing attribution, stated because it would otherwise be invented
    ///
    /// There is one submission, so there is one GPU measurement to make and the split is
    /// host against device rather than stage against stage:
    ///
    /// * `gather_us` is **all** the host work: the witness limb split, the `write_buffer`,
    ///   the parameter blocks and the bind groups, and the encoding.
    /// * `ntt_us` is submit to fence, so it is GPU wall time and nothing else.
    /// * `pointwise_us` is **zero on both paths**, because stage 4 is not a separate
    ///   submission on either.
    ///
    /// The host half is not a rounding error and lumping it into `ntt_us` produced a wrong
    /// conclusion once already while this unit was being written: with the two together, the
    /// fused path measured slower than the standalone one at 2^3, where there is no GPU work
    /// to speak of and the whole number is host time.
    ///
    /// Per-stage numbers need GPU timestamp queries, which is U11's, and design §7 already
    /// records that Dawn quantises resolved timestamps to 65,536 ns so even those will not
    /// resolve a sub-millisecond stage.
    pub async fn compute_h_with(
        &self,
        backend: &WgpuBackend,
        witness: &[Fr],
        mode: Stage4,
        t: &mut StageTimings,
    ) -> Result<HPoly, ProveError> {
        if witness.len() != self.n_vars {
            return Err(ProveError::WitnessLength {
                got: witness.len(),
                want: self.n_vars,
            });
        }
        let n = self.domain.size as u32;
        let mut sc = self.take_scratch(backend)?;

        // Host work, and all of it: a limb split per entry into a reused staging vector,
        // then one `write_buffer`. No Montgomery reduction anywhere; see the module docs.
        let start = Instant::now();
        sc.pack.clear();
        for x in witness {
            sc.pack.extend_from_slice(&PackedFr::from_fr(x).v);
        }
        backend
            .queue()
            .write_buffer(&sc.witness, 0, bytemuck::cast_slice(&sc.pack));

        // Every parameter block for the whole call is pushed before the encoder exists and
        // uploaded in one `write_buffer`, which is design §3's rule and the reason `plan`
        // and `encode` are separate on all three kernels.
        sc.ring.reset();
        let gather_offsets = self.gather.plan(&self.csr, &mut sc.ring)?;
        let gather_bind = self.gather.bind(
            backend,
            &self.csr,
            &sc.ring,
            &sc.witness,
            &sc.a,
            &sc.b,
            &sc.c,
        )?;

        // The six transforms, in the order the fusion depends on: A's pair, then B's, then
        // C's, so that A and B hold their coset evaluations before C's last batch joins.
        let mut planned = Vec::with_capacity(6);
        for vi in 0..3usize {
            let v = match vi {
                0 => &sc.a,
                1 => &sc.b,
                _ => &sc.c,
            };
            // Stage 1: the iNTT, out of place v -> t, with the 1/n normalisation riding in
            // on the load. The transform is Fr-linear, so scaling the input is the same map
            // as scaling the output, which is what makes that fusion exact rather than
            // approximate.
            planned.push(self.ntt.plan(
                backend,
                &self.tables,
                &mut sc.ring,
                &Transform {
                    dir: Direction::Inverse,
                    scale: Scale::SizeInv,
                    src: v,
                    dst: &sc.t,
                    epilogue: Epilogue::Plain,
                },
            )?);
            // Stages 2 and 3: the coset shift applied on load, then the forward transform
            // back into v. Stage 4 rides out on the last store of the last vector.
            let join = mode == Stage4::Fused && vi == 2;
            planned.push(self.ntt.plan(
                backend,
                &self.tables,
                &mut sc.ring,
                &Transform {
                    dir: Direction::Forward,
                    scale: Scale::CosetPowers,
                    src: &sc.t,
                    dst: v,
                    epilogue: if join {
                        Epilogue::Join {
                            a: &sc.a,
                            b: &sc.b,
                            h_mont: &sc.h_mont,
                            h_std: &sc.h_std,
                        }
                    } else {
                        Epilogue::Plain
                    },
                },
            )?);
        }

        let standalone = (mode == Stage4::Standalone)
            .then(|| -> Result<_, ProveError> {
                let offsets = self.join.plan(n, &mut sc.ring)?;
                let bind = self.join.bind(
                    backend, &sc.ring, n, &sc.a, &sc.b, &sc.c, &sc.h_mont, &sc.h_std,
                )?;
                Ok((offsets, bind))
            })
            .transpose()?;

        sc.ring.flush(backend);

        let mut enc = backend
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("g16 stages 0-4"),
            });
        {
            // One compute pass for all of it. WebGPU orders dispatches within a pass and
            // makes each one's writes visible to the next, which is what lets the six
            // transforms and the join share an encoder with no explicit barrier.
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g16 stages 0-4"),
                timestamp_writes: None,
            });
            self.gather
                .encode(&mut pass, &gather_bind, &self.csr, &gather_offsets)?;
            for p in &planned {
                self.ntt.encode(&mut pass, p)?;
            }
            if let Some((offsets, bind)) = &standalone {
                self.join.encode(&mut pass, bind, n, offsets)?;
            }
        }
        let cmd = enc.finish();
        // Everything above this line is host work, and it is charged as such. See the
        // attribution note on this method for why that separation is load-bearing.
        t.gather_us += start.elapsed().as_micros() as u64;

        let start = Instant::now();
        // Exactly one submit, and counted. `tests/stages.rs` reads the counter either side
        // of this call and requires the difference to be one.
        backend.submit([cmd]);
        // Awaited so that `ntt_us` is a GPU wall time and not an encode time. See
        // `WgpuBackend::wait_for_submitted_work` for what that costs.
        backend.wait_for_submitted_work().await?;
        t.ntt_us += start.elapsed().as_micros() as u64;

        if let Some(e) = backend.take_error() {
            // `sc` is dropped rather than returned to the pool, deliberately. A device error
            // means some dispatch in that encoder did not run, so what is in those buffers is
            // unknown, and the next proof would inherit it. Every kernel here overwrites what
            // it reads, so that would be harmless in practice, and "harmless in practice" is
            // not a reason to hand a proof a buffer nobody can describe. The cost is one
            // reallocation on the next proof after a failure.
            return Err(bad(format!("device error in stages 0 to 4: {e}")));
        }

        Ok(HPoly::Device {
            tag: TAG,
            len: self.domain.size,
            data: Arc::new(WgpuHandle {
                scratch: Some(sc),
                pool: Arc::clone(&self.pool),
                len: self.domain.size,
            }),
        })
    }

    /// Scratch sets sitting idle in the pool.
    ///
    /// Public so `tests/stages.rs` can assert the thing the pool is actually for: that a
    /// set is checked out for the life of the [`WgpuHandle`] and returns on its drop, and
    /// that two concurrent proofs therefore hold two different sets rather than racing on
    /// one. There is no way to observe that from the outside otherwise, because wgpu has no
    /// buffer identity comparison.
    pub fn pooled(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn gather(&self) -> &GatherAbc {
        &self.gather
    }

    pub fn ntt(&self) -> &Ntt {
        &self.ntt
    }

    pub fn h_join(&self) -> &HJoin {
        &self.join
    }
}

// ---------------------------------------------------------------------------
// The device-resident result
// ---------------------------------------------------------------------------

/// What [`g16_core::HPoly::Device`] carries out of stage 4, under the tag [`TAG`].
///
/// Recover it with `h.device_handle::<WgpuHandle>(g16_wgpu::stages::TAG)`. It owns the whole
/// scratch set for the proof, which is what keeps the buffers alive and exclusive until the
/// caller drops the `HPoly`: the scratch returns to the pool at that point and not before,
/// so a second concurrent proof cannot be handed buffers stage 9 is still reading.
///
/// **The handle is never stashed on the circuit.** `PreparedCircuit` takes `&self` and is
/// documented as safe to prove with concurrently, so a circuit holding "the last H buffer"
/// would race between two in-flight proofs and the race would produce a proof that simply
/// fails to verify, with nothing else to go on. The handle travels through the value.
///
/// # The one rule a caller has to keep
///
/// Do not drop the `HPoly` while GPU work that reads these buffers is still in flight.
/// Returning the scratch to the pool makes it available to the next proof, and WebGPU will
/// not stop that next proof from overwriting a buffer an unfinished dispatch is reading.
/// `compute_h` waits for its own submission before returning, and U11's `msms` waits for its
/// readback, so in the intended sequence this holds automatically. Nothing enforces it.
pub struct WgpuHandle {
    scratch: Option<Scratch>,
    pool: Pool,
    len: usize,
}

impl WgpuHandle {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn sc(&self) -> &Scratch {
        // Only `drop` clears the option, and it cannot run while a `&self` borrow lives.
        self.scratch.as_ref().expect("handle used after drop")
    }

    /// `H` in **standard form**, the integer in `[0, r)`.
    ///
    /// This is what stage 9's Pippenger must read: a window digit of a Montgomery
    /// representative is a digit of `a*R mod r`, which is a different number, and a proof
    /// built from the wrong one is wrong by a factor of R and fails verification with
    /// nothing else to go on.
    pub fn h_std(&self) -> &wgpu::Buffer {
        &self.sc().h_std
    }

    /// `H` in **Montgomery form**, which is `g16_gpu_layout::PackedFr` and arkworks' own
    /// internal representation. For further field arithmetic and for host comparisons.
    pub fn h_mont(&self) -> &wgpu::Buffer {
        &self.sc().h_mont
    }

    /// The three coset vectors A, B and C, for a debugging dump.
    ///
    /// C holds its finished forward transform only on the [`Stage4::Standalone`] path. On
    /// the fused path the last batch writes `H` instead of C, which is the write the fusion
    /// exists to save, so C holds the half-transformed vector (or, on a single-batch domain,
    /// the raw gather output).
    pub fn coset(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        let s = self.sc();
        (&s.a, &s.b, &s.c)
    }

    /// The raw device words of `H` in Montgomery form, undecoded.
    ///
    /// Separate from [`Self::to_host`] because a test comparing against
    /// `PackedFr::from_fr` wants the bytes: `PackedFr::to_fr` is `new_unchecked`, a bit
    /// reinterpretation, so decoding first would be harmless here but would make the
    /// comparison read as if a conversion had been trusted.
    pub async fn h_mont_words(&self, backend: &WgpuBackend) -> Result<Vec<u32>, ProveError> {
        self.read_words(backend, self.h_mont()).await
    }

    /// The raw device words of `H` in standard form, undecoded.
    ///
    /// The undecoded form is the only honest one for `h_std`: `PackedScalar::to_fr` returns
    /// `None` on limbs that are not a canonical residue, so decoding first would turn a
    /// wrong reduction into a `None` rather than into a visible pair of limb arrays.
    pub async fn h_std_words(&self, backend: &WgpuBackend) -> Result<Vec<u32>, ProveError> {
        self.read_words(backend, self.h_std()).await
    }

    /// Copies `H` down to the host in Montgomery form.
    ///
    /// **Tests and cross-checks only.** Forcing this copy is exactly the round trip
    /// [`g16_core::HPoly`] exists to avoid, and at 2^18 it moves 8 MB where design §3 caps
    /// the whole per-proof readback at 64 KiB. On wasm `get_mapped_range` copies the entire
    /// mapped `ArrayBuffer` into linear memory, measured at 33.8 ms for 64 MiB, so this is
    /// also the wrong tool there.
    pub async fn to_host(&self, backend: &WgpuBackend) -> Result<Vec<Fr>, ProveError> {
        let words = self.h_mont_words(backend).await?;
        fr_from_words(&words).ok_or_else(|| bad("h_mont is not a whole number of elements"))
    }

    /// The standard-form copy, read back and validated.
    ///
    /// `None` at any index whose limbs are not a canonical residue below `r`, which is what
    /// a wrong reduction in `fr_from_mont` looks like. Silently reducing here would hide
    /// exactly that bug, so the check is the point of the return type.
    pub async fn to_host_std(&self, backend: &WgpuBackend) -> Result<Option<Vec<Fr>>, ProveError> {
        let words = self.h_std_words(backend).await?;
        if !words.len().is_multiple_of(LIMBS) {
            return Err(bad("h_std is not a whole number of elements"));
        }
        Ok(words
            .chunks_exact(LIMBS)
            .map(|c| {
                let mut v = [0u32; LIMBS];
                v.copy_from_slice(c);
                PackedScalar { v }.to_fr()
            })
            .collect())
    }

    async fn read_words(
        &self,
        backend: &WgpuBackend,
        buf: &wgpu::Buffer,
    ) -> Result<Vec<u32>, ProveError> {
        let bytes = self.len as u64 * FR_BYTES;
        let rb = Readback::new(backend, "g16 h readback", bytes)?;
        let mut enc = backend
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("g16 h readback"),
            });
        rb.copy_from(&mut enc, buf, 0, bytes)?;
        let raw = rb.submit_and_read(backend, enc, bytes).await?;
        Ok(bytemuck::cast_slice::<u8, u32>(&raw).to_vec())
    }
}

impl Drop for WgpuHandle {
    fn drop(&mut self) {
        if let Some(s) = self.scratch.take() {
            self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(s);
        }
    }
}
