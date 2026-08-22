//! Stage 4 on the host, standalone: the module, the layout and the dispatch for `h_join`.
//!
//! [`crate::gen::pointwise`] explains the kernel. This file is the other half, and it is
//! small on purpose: one pipeline, one bind group, one parameter block per dispatch.
//!
//! # What this is for
//!
//! Two things, and the first one changed while U7 was being written.
//!
//! **It is the path the prover takes.** Design §4 and §8 both say to fuse stage 4 into the
//! last NTT batch, and measuring the two put the fusion 5% to 9% behind on every
//! artifact. [`crate::stages::Stage4`] carries the table. Both paths ship.
//!
//! **It is the independent implementation the fused one is checked against.** The fused path
//! writes `H` out of the store epilogue of the last NTT batch, so the only thing that ever
//! sees `H` there is the same kernel that computed the transform, and comparing that path
//! against itself would prove nothing. `h_join` shares the `Fr` prelude with the NTT and
//! nothing else, so running both in
//! `tests/stages.rs::compute_h_matches_the_cpu_backend_on_every_artifact` is two independent
//! statements rather than one repeated.
//!
//! It is also the only way to get the three coset vectors materialised on the device, since
//! the fused path deliberately never writes C's final transform.

use bytemuck::{Pod, Zeroable};
use g16_core::ProveError;
use g16_gpu_layout::LIMBS;

use crate::device::{bad, WgpuBackend};
use crate::gather::storage_entry;
use crate::gen::field::Variant;
use crate::gen::pointwise as wgsl;
use crate::params::ParamRing;
use crate::pipelines::Kernels;

/// Bytes one `Fr` occupies on the device.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// Mirrors `struct HJoinParams` in [`crate::gen::pointwise`]. 16 bytes.
///
/// The padding is written out rather than left to WGSL's 16-byte uniform struct alignment,
/// so the two declarations are obviously the same size. `ParamRing::push` cannot check the
/// correspondence and nothing else will either: a mismatch reads plausible garbage with no
/// validation error anywhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct HJoinParams {
    /// First element this dispatch computes.
    pub lo: u32,
    /// One past the last element this dispatch computes.
    pub hi: u32,
    pub pad0: u32,
    pub pad1: u32,
}

const _: () = assert!(core::mem::size_of::<HJoinParams>() == 16);

/// The standalone stage 4 module, its bind group layout and its one pipeline.
pub struct HJoin {
    kernels: Kernels,
    bgl: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
    elems_per_dispatch: u32,
    workgroup: u32,
    source_len: usize,
}

impl HJoin {
    /// Compiles at [`wgsl::WORKGROUP`] and this device's own workgroup-per-dimension limit.
    pub fn new(backend: &WgpuBackend) -> Result<Self, ProveError> {
        let max_wg = backend
            .granted_limits()
            .max_compute_workgroups_per_dimension;
        Self::with_shape(
            backend,
            max_wg.saturating_mul(wgsl::WORKGROUP),
            wgsl::WORKGROUP,
        )
    }

    /// Same, with the per-dispatch element cap and the workgroup size forced.
    ///
    /// Public for two tests that cannot be written otherwise: the workgroup sweep that made
    /// [`wgsl::WORKGROUP`] a measured number, and the multi-dispatch path, which no artifact
    /// reaches because a single dispatch covers 16,776,960 elements and the largest domain
    /// here is 2^18. A code path no test can reach is a code path that ships untested.
    ///
    /// The element cap is rounded down to a whole number of workgroups, because a partial
    /// workgroup at a chunk boundary would leave elements uncovered between chunks.
    pub fn with_shape(
        backend: &WgpuBackend,
        elems_per_dispatch: u32,
        workgroup: u32,
    ) -> Result<Self, ProveError> {
        // The browser floor, not the granted limit; see `WgpuBackend::ceiling_invocations`.
        let limits = backend.granted_limits();
        let ceiling = backend.ceiling_invocations();
        if workgroup == 0 || workgroup > ceiling {
            return Err(bad(format!(
                "workgroup size {workgroup} is outside 1..={ceiling}"
            )));
        }
        let elems = elems_per_dispatch - elems_per_dispatch % workgroup;
        if elems == 0 {
            return Err(bad(format!(
                "elems_per_dispatch {elems_per_dispatch} is under one {workgroup} thread \
                 workgroup"
            )));
        }
        let max_wg = limits.max_compute_workgroups_per_dimension;
        if elems.div_ceil(workgroup) > max_wg {
            return Err(bad(format!(
                "elems_per_dispatch {elems} needs {} workgroups, over the {max_wg} limit",
                elems.div_ceil(workgroup)
            )));
        }

        let device = backend.device();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g16 h_join"),
            entries: &Self::bind_group_layout_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g16 h_join"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let source = wgsl::h_join_module_at(Variant::default(), workgroup);
        let source_len = source.len();
        let kernels = Kernels::build(backend, "h_join", &source, &layout, &[wgsl::ENTRY])?;

        Ok(Self {
            kernels,
            bgl,
            _layout: layout,
            elems_per_dispatch: elems,
            workgroup,
            source_len,
        })
    }

    /// The bind group layout entries, as data, so a test can count what the pipeline layout
    /// declares.
    ///
    /// This is the list the layout is built from, so the count a test takes here is the
    /// count the device enforces. wgpu exposes nothing readable off a constructed
    /// `BindGroupLayout`, and duplicating the list in the test would only prove the
    /// duplicate was right.
    pub fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 6] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_A, true),
            storage_entry(wgsl::BIND_B, true),
            storage_entry(wgsl::BIND_C, true),
            storage_entry(wgsl::BIND_H_MONT, false),
            storage_entry(wgsl::BIND_H_STD, false),
        ]
    }

    /// Storage buffers the pipeline layout declares, counted from
    /// [`Self::bind_group_layout_entries`]. Must be at most 8, which is the floor.
    pub fn storage_buffer_count() -> u32 {
        Self::bind_group_layout_entries()
            .iter()
            .filter(|e| {
                matches!(
                    e.ty,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    }
                )
            })
            .count() as u32
    }

    /// Binds three inputs and two outputs over `n` elements.
    ///
    /// Every buffer is length checked here, because WebGPU derives a runtime-sized array's
    /// length from the binding size and then silently drops out-of-range writes and returns
    /// zero for out-of-range reads. A short `h_std` would leave a correct `h_mont` next to a
    /// truncated `h_std` and the proof would fail with nothing in any log.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        backend: &WgpuBackend,
        ring: &ParamRing,
        n: u32,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        h_mont: &wgpu::Buffer,
        h_std: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup, ProveError> {
        let want = n as u64 * FR_BYTES;
        for (name, buf) in [
            ("a", a),
            ("b", b),
            ("c", c),
            ("h_mont", h_mont),
            ("h_std", h_std),
        ] {
            if buf.size() < want {
                return Err(bad(format!(
                    "h_join {name} is {} bytes, {n} elements need {want}",
                    buf.size()
                )));
            }
        }

        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        Ok(backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("g16 h_join"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: wgsl::BIND_PARAMS,
                        resource: wgpu::BindingResource::Buffer(ring.binding()),
                    },
                    entry(wgsl::BIND_A, a),
                    entry(wgsl::BIND_B, b),
                    entry(wgsl::BIND_C, c),
                    entry(wgsl::BIND_H_MONT, h_mont),
                    entry(wgsl::BIND_H_STD, h_std),
                ],
            }))
    }

    /// Pushes one parameter block per dispatch and returns their dynamic offsets.
    ///
    /// Separate from [`Self::encode`] because design §3 writes every parameter block for the
    /// whole proof in one `write_buffer` before encoding starts, so the pushes have to
    /// happen before the compute pass exists.
    pub fn plan(&self, n: u32, ring: &mut ParamRing) -> Result<Vec<u32>, ProveError> {
        let mut offsets = Vec::with_capacity(self.dispatches(n) as usize);
        let mut lo = 0u32;
        while lo < n {
            let hi = (lo + self.elems_per_dispatch).min(n);
            offsets.push(ring.push(&HJoinParams {
                lo,
                hi,
                pad0: 0,
                pad1: 0,
            })?);
            lo = hi;
        }
        Ok(offsets)
    }

    /// Records the dispatches. `offsets` is what [`Self::plan`] returned for the same `n`.
    pub fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        n: u32,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let want = self.dispatches(n) as usize;
        if offsets.len() != want {
            return Err(bad(format!(
                "stage 4 over {n} elements needs {want} dispatches, got {} parameter offsets",
                offsets.len()
            )));
        }
        pass.set_pipeline(self.kernels.get(wgsl::ENTRY)?);
        let mut lo = 0u32;
        for &off in offsets {
            let hi = (lo + self.elems_per_dispatch).min(n);
            pass.set_bind_group(0, bind, &[off]);
            pass.dispatch_workgroups((hi - lo).div_ceil(self.workgroup), 1, 1);
            lo = hi;
        }
        Ok(())
    }

    /// Dispatches `n` elements need, which is also the parameter ring slots stage 4 consumes
    /// on the standalone path.
    pub fn dispatches(&self, n: u32) -> u32 {
        n.div_ceil(self.elems_per_dispatch)
    }

    /// Elements one dispatch covers: 16,776,960 at the floor (256 threads x 65535
    /// workgroups), less if forced.
    pub fn elems_per_dispatch(&self) -> u32 {
        self.elems_per_dispatch
    }

    /// Threads per workgroup this pipeline was generated at.
    pub fn workgroup(&self) -> u32 {
        self.workgroup
    }

    /// Generated WGSL length in bytes, for the compile-cost report.
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }
}
