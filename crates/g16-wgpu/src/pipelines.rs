//! Shader modules and compute pipelines, built once and timed.
//!
//! # Why the cost is tracked from this unit rather than measured at the end
//!
//! An earlier measurement records 129 s of pipeline creation for one
//! monolithic WGSL shader with an unrolled 20-limb Montgomery multiply inlined at a dozen
//! call sites. That is the shape U9 and U10's curve kernels take, so the number is either
//! watched from the first commit or discovered when the browser story is already dead.
//! §9 risk 2 of the design puts the kill line at 10 s of MSM pipeline creation in Chrome.
//!
//! Measured on this M2 Max so far, naga to MSL, best of three with a nonce comment forcing a
//! genuinely new shader hash: **26 to 28 ms** for the 83.4 KiB, 25 entry point field module
//! (`crates/g16-wgpu/tests/field.rs`), and 155 to 174 ms for the 29 KiB single-field module
//! through the bench harness. The one 1968 ms figure the last unit saw was process or driver
//! first touch, not compilation, and attributing it to a shader would be wrong by two orders
//! of magnitude.
//!
//! # The number means different things on the two targets, and that has to be said
//!
//! On native, `create_shader_module` runs naga's parse, validate and MSL emit synchronously,
//! and `create_compute_pipeline` runs Metal's own compiler synchronously. Both halves of
//! [`PrepareCost`] are real work.
//!
//! In a browser neither call is required to have finished anything when it returns:
//! `createShaderModule` reports errors through `getCompilationInfo()`, and Dawn compiles
//! lazily behind `createComputePipeline`. The sweep measured
//! 0.9 to 4.6 ms in Chrome for the same 29 KiB module that costs 155 to 174 ms natively,
//! roughly 40x, and that ratio is too good to believe as a compiler comparison. Read the
//! browser figure as a lower bound until U13 takes it again through
//! `createComputePipelineAsync`, which is the call that actually waits.

use std::collections::HashMap;

use g16_core::ProveError;
// Not `std::time::Instant`: that panics at run time on `wasm32-unknown-unknown` with "time
// not implemented on this platform", in the browser, after everything has linked. `web-time`
// is `performance.now()` there and a plain re-export of `std::time` everywhere else, which
// is what keeps this file free of a target `cfg`. wgpu already pulls it on wasm.
use web_time::Instant;

use crate::device::{bad, WgpuBackend};

/// Where the time in pipeline construction went, in microseconds.
///
/// Accumulated on the backend rather than printed, so a caller can report it without
/// re-running the build in order to time it. Same shape and same reason as
/// `g16_cuda::PrepareCost`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrepareCost {
    /// `Device::create_shader_module`: naga's parse, validate and backend emit.
    pub module_us: u64,
    /// `Device::create_compute_pipeline`: the platform compiler behind it.
    pub pipeline_us: u64,
    /// `module_us + pipeline_us`, which is the figure §9 risk 2's kill line is in terms of.
    pub compile_us: u64,
    pub modules: u32,
    pub pipelines: u32,
}

impl PrepareCost {
    pub fn add(&mut self, other: PrepareCost) {
        self.module_us += other.module_us;
        self.pipeline_us += other.pipeline_us;
        self.compile_us += other.compile_us;
        self.modules += other.modules;
        self.pipelines += other.pipelines;
    }
}

/// One WGSL module and every compute pipeline built from it.
///
/// Deliberately one module per group of kernels rather than one for the whole backend: the
/// 129 s figure in the module docs was a single fused module, and the mitigation the design
/// commits to is many small ones. Splitting also means a change to the NTT does not
/// re-trigger the compile over the G2 curve arithmetic.
pub struct Kernels {
    label: String,
    source_len: usize,
    module: wgpu::ShaderModule,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    cost: PrepareCost,
}

impl Kernels {
    /// Compiles `source` once and builds one pipeline per name in `entries`.
    ///
    /// `layout` is explicit rather than `None`. An inferred layout is derived per entry point
    /// from what that entry point happens to touch, so two kernels meant to share a bind
    /// group can silently end up with incompatible layouts, and a binding a kernel does not
    /// read disappears from the layout entirely. An explicit layout is allowed to declare
    /// bindings a shader ignores, which is what lets one bind group serve every entry point
    /// in a module.
    ///
    /// The cost is charged to `backend` as a side effect, so
    /// [`WgpuBackend::prepare_cost`] is the total across every module without the caller
    /// having to add them up.
    pub fn build(
        backend: &WgpuBackend,
        label: &str,
        source: &str,
        layout: &wgpu::PipelineLayout,
        entries: &[&str],
    ) -> Result<Self, ProveError> {
        let paired: Vec<(&str, &wgpu::PipelineLayout)> =
            entries.iter().map(|&e| (e, layout)).collect();
        Self::build_with_layouts(backend, label, source, &paired)
    }

    /// Same, with a separate pipeline layout per entry point.
    ///
    /// One module, several layouts, which is what design §4's "one entry point per mode"
    /// needs: `ntt_head_plain` declares 4 storage buffers and `ntt_head_join` declares 7, and
    /// giving both the union of 8 would put the pair exactly at the browser floor with no
    /// margin, which is the situation the split exists to escape. The module is still
    /// compiled once, which matters because the shared `Fr` prelude is most of its bytes.
    pub fn build_with_layouts(
        backend: &WgpuBackend,
        label: &str,
        source: &str,
        entries: &[(&str, &wgpu::PipelineLayout)],
    ) -> Result<Self, ProveError> {
        let device = backend.device();

        let t0 = Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let module_us = t0.elapsed().as_micros() as u64;

        let t1 = Instant::now();
        let mut pipelines = HashMap::with_capacity(entries.len());
        for &(entry, layout) in entries {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                // No pipeline cache. `Device::create_pipeline_cache` needs the
                // `PIPELINE_CACHE` feature, which does not exist in WebGPU, so caching here
                // would be a native-only speedup on the one target where the compile is
                // already fast enough not to matter.
                cache: None,
            });
            if pipelines.insert(entry.to_string(), p).is_some() {
                return Err(bad(format!(
                    "module {label:?} lists entry point {entry:?} twice"
                )));
            }
        }
        let pipeline_us = t1.elapsed().as_micros() as u64;

        // A WGSL error in Chrome arrives as an uncaptured device error rather than as a
        // failed call, so without this the first sign of a broken shader is a dispatch that
        // writes nothing. Natively wgpu's default handler would already have panicked, so
        // this only ever fires on the web, which is exactly where it is needed.
        if let Some(e) = backend.take_error() {
            return Err(bad(format!("module {label:?} failed to build: {e}")));
        }

        let cost = PrepareCost {
            module_us,
            pipeline_us,
            compile_us: module_us + pipeline_us,
            modules: 1,
            pipelines: entries.len() as u32,
        };
        backend.charge(cost);

        Ok(Self {
            label: label.to_string(),
            source_len: source.len(),
            module,
            pipelines,
            cost,
        })
    }

    /// The pipeline for one entry point, or an error naming the module.
    ///
    /// Returns a `Result` rather than panicking on a missing key because the caller is a
    /// backend method that already returns `ProveError`, and a typo'd entry point should not
    /// take the process down.
    pub fn get(&self, entry: &str) -> Result<&wgpu::ComputePipeline, ProveError> {
        self.pipelines.get(entry).ok_or_else(|| {
            bad(format!(
                "module {:?} has no pipeline for entry point {entry:?}",
                self.label
            ))
        })
    }

    pub fn module(&self) -> &wgpu::ShaderModule {
        &self.module
    }

    /// The label this module was built under, which is what a diagnostic dump names it by.
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn cost(&self) -> PrepareCost {
        self.cost
    }

    /// One line for a log or a test: what was compiled and what it cost.
    pub fn summary(&self) -> String {
        format!(
            "{}: {:.1} KiB, {} entry points, module {:.1} ms + pipelines {:.1} ms = {:.1} ms",
            self.label,
            self.source_len as f64 / 1024.0,
            self.cost.pipelines,
            self.cost.module_us as f64 / 1e3,
            self.cost.pipeline_us as f64 / 1e3,
            self.cost.compile_us as f64 / 1e3,
        )
    }
}
