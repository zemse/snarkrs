//! Adapter, device and limits profile: the one place that decides what this backend is
//! allowed to assume about the hardware.
//!
//! # Why the default profile is the browser floor and not the adapter's own limits
//!
//! A WebGPU device created without `requiredLimits` gets the spec defaults even on an
//! adapter advertising far more, with no warning anywhere. The gap measured on this M2 Max:
//! 4 GiB of storage-buffer binding against the spec's
//! 128 MiB (32x), 32 KiB of workgroup storage against 16 KiB (2x), 1024 invocations per
//! workgroup against 256 (4x), and one extra storage buffer per stage, 9 against 8 (see the
//! correction below). Every one is a cliff a kernel can be written off the edge of, and
//! native `cargo test` will not notice: `SHADER_INT64` is even reported as available on
//! native Metal through wgpu and does not exist in the WebGPU specification at all.
//!
//! Prior art records heliax shipping exactly that bug
//! twice. So [`LimitsProfile::Floor`] is the default, it is what tests use, and
//! [`LimitsProfile::Raised`] is opt-in through `G16_WGPU_LIMITS=raised` and buys throughput
//! only, never correctness.
//!
//! # Three limits that are physics
//!
//! `maxComputeWorkgroupsPerDimension` is 65535, `maxBindGroups` is 4 and
//! `minStorageBufferOffsetAlignment` is 256 at every tier of every browser, so no browser
//! profile ever improves them. Kernels are designed against those three as constants. Native
//! Metal does report a finer 32-byte storage alignment here, which is exactly the kind of
//! headroom that must not be leaned on.
//!
//! # Measured here, and it corrects that: `Raised` buys almost nothing on storage buffers
//!
//! An earlier probe recorded 29 storage buffers per shader stage on
//! this M2 Max, 3.6x the spec's 8, which would make design §4's 8-buffer squeeze look like a
//! Floor-only problem. It is not. That 29 was measured on an instance without
//! `STRICT_WEBGPU_COMPLIANCE`. With the flag on, which is unconditional here and in every
//! test, `adapter.limits()` reports **9**, so `Raised` is worth exactly one extra storage
//! buffer per stage. Both numbers taken back to back on this machine, wgpu 30.0.1, Metal:
//!
//! ```text
//! strict=false: storage_buffers_per_shader_stage 29
//! strict=true:  storage_buffers_per_shader_stage  9
//! ```
//!
//! Nothing else in the table moves with the flag. Plan the kernels for 8 and treat the
//! ninth as slack, not as headroom.
//!
//! `RequestAdapterOptions::apply_limit_buckets`, wgpu's anti-fingerprinting rounding that
//! browsers apply to adapter limits, changes nothing at all on this adapter (every field
//! identical with it on and off), so it is left off and the adapter column stays raw.
//!
//! # What "granted" means, and why the native column is a tautology
//!
//! `wgpu::DeviceDescriptor::required_limits` is documented as "exactly the specified limits,
//! and no better or worse, will be allowed in validation", and `Device::limits()` echoes the
//! request back. So on native the granted column always equals the requested column and
//! proves nothing. The column that carries information is the adapter's, which is why
//! [`WgpuBackend::limits_table`] prints all three.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use g16_core::ProveError;

use crate::pipelines::PrepareCost;

pub(crate) fn bad(reason: impl Into<String>) -> ProveError {
    ProveError::Backend {
        backend: "wgpu",
        reason: reason.into(),
    }
}

/// Which set of limits the device is requested with.
///
/// Not a performance knob with a safe default: picking [`Self::Raised`] changes what will
/// compile, and a kernel that only fits `Raised` is a kernel that fails in a stock browser.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimitsProfile {
    /// `wgpu::Limits::default()`, which is the WebGPU specification's defaults verbatim:
    /// 128 MiB storage binding, 256 MiB buffer, 64 KiB uniform binding, 16 KiB workgroup
    /// storage, 256 invocations per workgroup, 8 storage buffers per shader stage, 4 bind
    /// groups, 65535 workgroups per dimension, 256-byte binding alignment.
    #[default]
    Floor,
    /// Whatever the adapter reports. Larger buffers and wider workgroups, and nothing this
    /// backend needs in order to be correct. On this M2 Max, with strict WebGPU compliance
    /// on: 4 GiB buffer and storage binding against 256 MiB and 128 MiB, 32 KiB workgroup
    /// storage against 16 KiB, 1024 invocations against 256, and 9 storage buffers per stage
    /// against 8. That last one is the correction in the module docs, and it is why no
    /// kernel is allowed to need a ninth.
    Raised,
}

impl LimitsProfile {
    /// Reads `G16_WGPU_LIMITS`, defaulting to [`Self::Floor`].
    ///
    /// `std::env::var` is not a `cfg` hazard here: on `wasm32-unknown-unknown` std compiles
    /// it against an empty environment and it returns `NotPresent`, so the browser build
    /// gets `Floor` and never has to be special-cased. A browser that wants `Raised` will
    /// get an explicit constructor at U13, not an environment variable it cannot set.
    pub fn from_env() -> Result<Self, ProveError> {
        match std::env::var("G16_WGPU_LIMITS") {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Floor),
        }
    }

    /// Rejects an unrecognised value rather than falling back to `Floor`.
    ///
    /// A typo that silently selects the default would make `G16_WGPU_LIMITS=rasied` report a
    /// Floor number as a Raised one, which is the class of measurement error this repo keeps
    /// finding.
    pub fn parse(v: &str) -> Result<Self, ProveError> {
        match v.trim().to_ascii_lowercase().as_str() {
            "floor" | "" => Ok(Self::Floor),
            "raised" => Ok(Self::Raised),
            other => Err(bad(format!(
                "G16_WGPU_LIMITS={other:?} is not a profile, expected \"floor\" or \"raised\""
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Raised => "raised",
        }
    }

    /// The limits to request from this adapter under this profile.
    pub fn required_limits(self, adapter: &wgpu::Adapter) -> wgpu::Limits {
        match self {
            Self::Floor => wgpu::Limits::default(),
            Self::Raised => adapter.limits(),
        }
    }
}

/// One adapter, one device, one queue, and the running total of what compiling cost.
///
/// Built once per process. There is no `Clone`: what makes it expensive is the pipeline
/// compiles hanging off it, and U11 will share those by handing every circuit an `Arc` of
/// the same [`crate::pipelines::Kernels`] rather than by duplicating anything.
///
/// This is not yet a `g16_core::Backend`. Stages 0 to 9 do not exist, so an implementation
/// would have to either lie or panic; U11 adds it once there is something to run.
pub struct WgpuBackend {
    // Held because dropping the instance while a device is alive is not something wgpu
    // promises anything about, and on wasm it owns the `GPU` handle.
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    profile: LimitsProfile,
    requested: wgpu::Limits,
    /// First uncaptured device error, if any. See [`Self::take_error`].
    error: Arc<Mutex<Option<String>>>,
    /// Held for the whole of one proof's GPU section. See [`Self::exclusive`].
    gpu: Mutex<()>,
    cost: Mutex<PrepareCost>,
    /// Command buffers handed to [`WgpuBackend::submit`] since this backend was created.
    /// See that method for why the count exists and what it cannot see.
    submits: AtomicU64,
}

/// `g16_core::Backend` and `PreparedCircuit` are both `Send + Sync`, so U11 cannot land
/// unless this holds. On wasm it holds only through wgpu's `fragile-send-sync-non-atomic-wasm`
/// feature, whose own cfg is `not(target_feature = "atomics")`, so dropping the feature or
/// turning on browser threads breaks it. Checking it here makes that a compile error in this
/// crate rather than a trait-bound error five units later.
const fn assert_send_sync<T: Send + Sync>() {}
const _: () = assert_send_sync::<WgpuBackend>();

impl WgpuBackend {
    /// Opens an adapter and a device at the profile named by `G16_WGPU_LIMITS`.
    ///
    /// Fails rather than falling back to anything: a benchmark that quietly measures a
    /// different backend is the exact failure this repo exists to avoid.
    pub async fn new() -> Result<Self, ProveError> {
        Self::with_profile(LimitsProfile::from_env()?).await
    }

    /// Same, at a caller-chosen profile.
    pub async fn with_profile(profile: LimitsProfile) -> Result<Self, ProveError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // The same switch as `WGPU_STRICT_WEBGPU_COMPLIANCE=1`, set here so a run that forgot
        // the variable still fails on a Metal-only construct instead of passing natively and
        // failing in Chrome at U13. It is on under `Raised` too: `Raised` widens limits, it
        // does not license a kernel that only one implementation can compile.
        desc.flags |= wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
        let instance = wgpu::Instance::new(desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| bad(format!("no wgpu adapter on this machine: {e}")))?;
        let info = adapter.get_info();
        let requested = profile.required_limits(&adapter);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("g16-wgpu"),
                // Empty on purpose and at both profiles. Every feature beyond the WebGPU
                // core set is one no browser has, and `SHADER_INT64` in particular is
                // reported as present on native Metal here while not existing in the
                // specification at all.
                required_features: wgpu::Features::empty(),
                required_limits: requested.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| {
                bad(format!(
                    "adapter {:?} refused a device at the {} profile: {e}",
                    info.name,
                    profile.as_str()
                ))
            })?;

        // wgpu's default uncaptured-error handler panics on native and logs to the console on
        // the web, and the web half is the problem: a shader that fails validation in Chrome
        // otherwise leaves no trace in Rust and the dispatch simply produces garbage.
        // Recording the first one makes it a `ProveError` on both targets.
        //
        // The trade is that a native validation error stops being an immediate panic, so it
        // is also written to stderr here. That keeps the loud native behaviour (the message
        // appears in test output at the moment it happens) while still letting
        // [`Self::take_error`] turn it into a returned error. `eprintln!` is a no-op on
        // wasm32-unknown-unknown rather than a compile error, so no `cfg` is needed.
        let error = Arc::new(Mutex::new(None::<String>));
        let sink = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            eprintln!("wgpu device error: {e}");
            let mut slot = sink.lock().unwrap();
            if slot.is_none() {
                *slot = Some(e.to_string());
            }
        }));

        // Device loss is a *different* channel from `on_uncaptured_error`, and it is the one
        // that matters for a wrong answer rather than a rejected call. An uncaptured error is
        // raised when the API refuses something; a lost device is what happens when work that
        // was accepted does not complete, and the buffers it was going to write keep whatever
        // they held. `wait_for_submitted_work` returns normally in that case and the proof
        // comes out wrong with nothing else to go on, which is exactly the failure `TASKS.md`
        // records against `g16-metal` for never checking command buffer status.
        let sink = error.clone();
        device.set_device_lost_callback(move |reason, msg| {
            eprintln!("wgpu device lost: {reason:?}: {msg}");
            let mut slot = sink.lock().unwrap();
            if slot.is_none() {
                *slot = Some(format!("the device was lost ({reason:?}): {msg}"));
            }
        });

        Ok(Self {
            _instance: instance,
            adapter,
            device,
            queue,
            info,
            profile,
            requested,
            error,
            gpu: Mutex::new(()),
            cost: Mutex::new(PrepareCost::default()),
            submits: AtomicU64::new(0),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn profile(&self) -> LimitsProfile {
        self.profile
    }

    /// What was asked for. Under [`LimitsProfile::Floor`] this is `wgpu::Limits::default()`
    /// field for field, which is what `tests/device.rs` asserts.
    pub fn requested_limits(&self) -> &wgpu::Limits {
        &self.requested
    }

    /// What the device says it has. Equal to [`Self::requested_limits`] by construction on
    /// native wgpu; see the module docs.
    pub fn granted_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// What the hardware offers, which is the only one of the three that carries information
    /// on native.
    pub fn adapter_limits(&self) -> wgpu::Limits {
        self.adapter.limits()
    }

    /// Invocations per workgroup a kernel is allowed to declare: the smaller of what this
    /// device grants and [`crate::gen::FLOOR_INVOCATIONS`].
    ///
    /// # Why this is not just the granted limit
    ///
    /// The generators hard-assert the floor, because the WGSL they emit has to compile in a
    /// stock browser. The host constructors used to check the *granted* limit instead, which
    /// is 1024 on this M2 Max under `G16_WGPU_LIMITS=raised`. A `with_shape(.., 512)` call
    /// therefore passed its own validation and then panicked inside the generator, where a
    /// `ProveError` was the documented behaviour. One ceiling, read by both sides, is the
    /// fix; the floor is the side to keep, because the generated source is what has to run
    /// in Chrome. Filed against U11 in `TASKS.md` by U9's verification pass.
    pub fn ceiling_invocations(&self) -> u32 {
        self.granted_limits()
            .max_compute_invocations_per_workgroup
            .min(crate::gen::FLOOR_INVOCATIONS)
    }

    /// Workgroup storage a kernel is allowed to declare, in bytes: the smaller of what this
    /// device grants and [`crate::gen::FLOOR_WORKGROUP_BYTES`]. Same reasoning as
    /// [`Self::ceiling_invocations`]; this adapter grants 32768 against the floor's 16384,
    /// which is exactly one extra NTT tile.
    pub fn ceiling_workgroup_bytes(&self) -> u64 {
        u64::from(self.granted_limits().max_compute_workgroup_storage_size)
            .min(crate::gen::FLOOR_WORKGROUP_BYTES)
    }

    /// Takes the first uncaptured device error since the last call, if there was one.
    ///
    /// Errors arrive asynchronously on both targets, so this is a poll and not a barrier:
    /// call it after a submit has been waited on, not immediately after encoding.
    ///
    /// **It is a single slot on the whole device, so it is only attributable while one proof
    /// at a time is submitting.** [`Self::exclusive`] is what arranges that.
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap().take()
    }

    /// Exclusive use of this device for one proof's GPU section: hold it from before the
    /// first `write_buffer` until after [`Self::take_error`].
    ///
    /// # Why a proof has to hold a lock, when nothing here is otherwise shared
    ///
    /// `g16_core::PreparedCircuit` takes `&self` and is documented as safe to prove with from
    /// several threads at once. Every piece of per-proof *state* is already per proof: the
    /// scratch buffers and the parameter ring are pooled and checked out
    /// (`crate::stages::Scratch`, `crate::batch`), so two concurrent proofs share no buffer.
    /// Two things are still device-global and cannot be made per proof:
    ///
    /// 1. **The uncaptured-error slot.** `Device::on_uncaptured_error` takes one callback for
    ///    the whole device and this backend funnels it into one `Mutex<Option<String>>`.
    ///    With two proofs in flight, whichever calls [`Self::take_error`] first takes
    ///    whichever error arrived, so proof A's dropped dispatch is reported against proof B
    ///    and A returns `Ok` over buffers some of whose dispatches never ran. That is a proof
    ///    that fails verification with nothing else to go on, which is the worst failure mode
    ///    this crate has. WebGPU's `pushErrorScope`/`popErrorScope` does not fix it either:
    ///    the scope stack is itself device-wide, so two interleaved proofs nest each other's
    ///    scopes.
    /// 2. **`onSubmittedWorkDone`.** [`Self::wait_for_submitted_work`] waits for *every*
    ///    submission on the device, so under concurrency `StageTimings::ntt_us` would include
    ///    another proof's GPU time and the benchmark would report a number nobody can
    ///    attribute.
    ///
    /// Both were filed against U11 in `TASKS.md`. The honest options were a lock from submit
    /// to poll or one device per in-flight proof, and a second device means a second copy of
    /// every base vector, 62 MB at `js_16x16_d32`, plus a second set of pipeline compiles.
    /// So: a lock, held across the whole of `compute_h` and the whole of `msms`.
    ///
    /// **What that costs, said plainly.** Concurrent proofs against one circuit are correct
    /// and serialised, not parallel. Nothing is lost that a single queue was ever going to
    /// give: this is one GPU with one queue, and two proofs interleaved on it finish in the
    /// same total time as two run back to back. What is lost is the host half, the witness
    /// limb split and the Horner tail, roughly 1 ms of the 100 at 2^18, which could in
    /// principle overlap another proof's GPU work and now does not.
    ///
    /// The guard is deliberately taken at the synchronous `PreparedCircuit` boundary in
    /// [`crate::backend`] rather than inside the `async fn`s here, so it is never held across
    /// an `.await` and the futures stay `Send`. A caller driving [`crate::stages::HStages`]
    /// or [`crate::batch::MsmBatch`] directly from two tasks has to take it itself; the
    /// browser entry point at U13 runs one proof per worker and does not.
    ///
    /// **Measured, after the fact: this is load bearing for correctness and not only for
    /// error attribution and timing.** U11's mutation round deleted the two calls to it and
    /// ran `tests/proof.rs::two_proofs_in_parallel_on_one_prepared_circuit_both_verify` on
    /// its own, which is four threads proving `js_16x16_d32` against one circuit on one
    /// device: `verify: PairingFailed` on run 2 of 3. The reasoning above argued for the
    /// guard from the error slot and the fence; four concurrent unguarded proofs on one
    /// device also simply produce a wrong proof, about one run in three, with no error
    /// raised anywhere. Reproduce by removing both `let _gpu = self.device.exclusive();`
    /// lines in [`crate::backend`].
    pub fn exclusive(&self) -> std::sync::MutexGuard<'_, ()> {
        self.gpu.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `queue.submit`, counted.
    ///
    /// Design §3 puts a whole proof in **two** submits, one for `compute_h` and one for
    /// `msms`, because an empty submit plus its fence measured 0.1 to 0.3 ms on this
    /// platform against 2 to 3 microseconds for an extra dispatch in an already open
    /// encoder. That is a claim about the code, so it is counted rather than asserted in a
    /// comment: `tests/stages.rs` reads [`Self::submits`] either side of `compute_h` and
    /// requires the difference to be exactly one.
    ///
    /// What the counter cannot see, stated plainly: [`Self::queue`] is public and a caller
    /// that submits through it directly is invisible here. The counter is a regression
    /// guard on code that already routes through this method, not a sandbox. Everything in
    /// `stages.rs` and `readback.rs` does route through it; the older tests submit
    /// directly and are not counted, which is harmless because they are not the thing being
    /// measured.
    ///
    /// `queue.write_buffer` is deliberately not counted. It is a queue *write*, staged and
    /// flushed with the next submit rather than a submission of its own, so counting it
    /// would report a number the design's budget is not in terms of.
    pub fn submit(
        &self,
        buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        self.submits.fetch_add(1, Ordering::Relaxed);
        self.queue.submit(buffers)
    }

    /// Submissions made through [`Self::submit`] since this backend was created.
    pub fn submits(&self) -> u64 {
        self.submits.load(Ordering::Relaxed)
    }

    /// Waits until everything submitted so far has finished on the GPU.
    ///
    /// The same two-target shape as [`crate::readback::Readback::submit_and_read`] and for
    /// the same reason: `on_submitted_work_done`'s own documentation says the callback runs
    /// only when `submit`, `poll_all` or `device.poll` is called elsewhere, so the `poll`
    /// below is what fires it on native, and on the web it is a documented no-op and the
    /// browser event loop fires it while this future is suspended. One function, no `cfg`.
    ///
    /// `compute_h` awaits this so that its `ntt_us` is a GPU wall time rather than an
    /// encode time. The cost is one fence, measured at 22 to 24 microseconds in release and
    /// about 70 in debug by
    /// `tests/stages.rs::the_fusion_is_measured_and_not_assumed`, against 0.7 ms of GPU work
    /// at the smallest artifact and 21 ms at the largest. So it is 0.1% to 3% of the number
    /// it makes honest, and design §3's 0.1 to 0.3 ms for the same thing is pessimistic by
    /// about 4x. U11 can drop the wait once GPU timestamp queries land, at which point the
    /// number stops being a wall clock at all.
    pub async fn wait_for_submitted_work(&self) -> Result<(), ProveError> {
        // flume and not std::sync::mpsc: the receiver has to be awaited rather than blocked
        // on, and `mpsc::Receiver` has no async form. Same choice as `readback.rs`.
        let (tx, rx) = flume::bounded(1);
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| bad(format!("waiting for the GPU failed: {e}")))?;
        rx.recv_async()
            .await
            .map_err(|_| bad("the submitted-work callback was dropped before it fired"))
    }

    /// Runs `f` inside a WebGPU validation error scope and reports what the scope caught.
    ///
    /// # Why this exists on top of `on_uncaptured_error`
    ///
    /// Because on Safari 26.6 the uncaptured-error channel reported nothing at all for a
    /// device error that a scope catches in full. WebKit's WGSL to MSL translation emitted a
    /// Metal construct that does not compile (`gen::field::Field::ret_limbs` has the detail),
    /// every `createComputePipeline` in the prover failed, every submit became a no-op, and
    /// [`Self::take_error`] returned `None` at all five places this crate calls it. The
    /// proof came out in 3 ms and snarkjs rejected it.
    ///
    /// A scope is the channel that is *specified* to answer: `popErrorScope` resolves after
    /// the operations inside it have finished their error checking, so it does not race the
    /// way polling a slot filled by an event does.
    ///
    /// # The concurrency rule this owes [`Self::exclusive`]
    ///
    /// That method argues error scopes cannot attribute a *proof's* errors, because the scope
    /// stack is device-wide and two proofs in flight would nest each other's. That argument
    /// stands and this does not contradict it: **the caller must hold the device exclusively
    /// for the whole of `f`**, which is why the only callers are the two build paths
    /// (`create_prover` and `prepare`), each of which runs before any proof exists on this
    /// device. Do not reach for this inside `compute_h` or `msms`.
    ///
    /// Not used natively for anything, and harmless there: wgpu implements scopes on every
    /// backend, and a native validation error is reported through both channels.
    pub async fn scoped<T>(
        &self,
        what: &str,
        f: impl FnOnce() -> Result<T, ProveError>,
    ) -> Result<T, ProveError> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        // Popped on both paths. A scope left on the stack is one the next `scoped` call pops
        // instead, which would report this section's error against the next one.
        let caught = scope.pop().await;
        let out = out?;
        match caught {
            Some(e) => Err(bad(format!("{what} was rejected by this device: {e}"))),
            None => Ok(out),
        }
    }

    pub(crate) fn charge(&self, cost: PrepareCost) {
        self.cost.lock().unwrap().add(cost);
    }

    /// Everything [`crate::pipelines::Kernels::build`] has cost on this device so far.
    ///
    /// Tracked from this unit rather than discovered at the end:
    /// An earlier measurement records 129 s of pipeline creation for one
    /// monolithic shader with an unrolled 20-limb multiply inlined at a dozen sites, which is
    /// precisely the shape the MSM kernels take at U9 and U10.
    pub fn prepare_cost(&self) -> PrepareCost {
        *self.cost.lock().unwrap()
    }

    /// The limits that matter to this backend, as requested / granted / adapter.
    ///
    /// Twelve rows out of the roughly sixty fields in `wgpu::Limits`, chosen as the ones
    /// measured to differ from the spec floor, plus the two alignments
    /// the parameter ring is built around. Printing all sixty would bury them.
    pub fn limits_table(&self) -> String {
        let (r, g, a) = (
            self.requested_limits(),
            self.granted_limits(),
            self.adapter_limits(),
        );
        let rows: [(&str, u64, u64, u64); 12] = [
            (
                "max_buffer_size",
                r.max_buffer_size,
                g.max_buffer_size,
                a.max_buffer_size,
            ),
            (
                "max_storage_buffer_binding_size",
                r.max_storage_buffer_binding_size,
                g.max_storage_buffer_binding_size,
                a.max_storage_buffer_binding_size,
            ),
            (
                "max_uniform_buffer_binding_size",
                r.max_uniform_buffer_binding_size,
                g.max_uniform_buffer_binding_size,
                a.max_uniform_buffer_binding_size,
            ),
            (
                "max_compute_workgroup_storage_size",
                r.max_compute_workgroup_storage_size as u64,
                g.max_compute_workgroup_storage_size as u64,
                a.max_compute_workgroup_storage_size as u64,
            ),
            (
                "max_compute_invocations_per_workgroup",
                r.max_compute_invocations_per_workgroup as u64,
                g.max_compute_invocations_per_workgroup as u64,
                a.max_compute_invocations_per_workgroup as u64,
            ),
            (
                "max_compute_workgroup_size_x",
                r.max_compute_workgroup_size_x as u64,
                g.max_compute_workgroup_size_x as u64,
                a.max_compute_workgroup_size_x as u64,
            ),
            (
                "max_compute_workgroups_per_dimension",
                r.max_compute_workgroups_per_dimension as u64,
                g.max_compute_workgroups_per_dimension as u64,
                a.max_compute_workgroups_per_dimension as u64,
            ),
            (
                "max_storage_buffers_per_shader_stage",
                r.max_storage_buffers_per_shader_stage as u64,
                g.max_storage_buffers_per_shader_stage as u64,
                a.max_storage_buffers_per_shader_stage as u64,
            ),
            (
                "max_bind_groups",
                r.max_bind_groups as u64,
                g.max_bind_groups as u64,
                a.max_bind_groups as u64,
            ),
            (
                "max_dynamic_uniform_buffers_per_pipeline_layout",
                r.max_dynamic_uniform_buffers_per_pipeline_layout as u64,
                g.max_dynamic_uniform_buffers_per_pipeline_layout as u64,
                a.max_dynamic_uniform_buffers_per_pipeline_layout as u64,
            ),
            (
                "min_uniform_buffer_offset_alignment",
                r.min_uniform_buffer_offset_alignment as u64,
                g.min_uniform_buffer_offset_alignment as u64,
                a.min_uniform_buffer_offset_alignment as u64,
            ),
            (
                "min_storage_buffer_offset_alignment",
                r.min_storage_buffer_offset_alignment as u64,
                g.min_storage_buffer_offset_alignment as u64,
                a.min_storage_buffer_offset_alignment as u64,
            ),
        ];
        let mut s = format!(
            "{:47} {:>12} {:>12} {:>12}\n",
            format!("limit ({})", self.profile.as_str()),
            "requested",
            "granted",
            "adapter"
        );
        for (name, req, got, adp) in rows {
            s.push_str(&format!("{name:47} {req:>12} {got:>12} {adp:>12}\n"));
        }
        s
    }
}
