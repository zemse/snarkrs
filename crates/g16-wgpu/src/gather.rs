//! Stage 0 on the host: build the concatenated CSR once, then dispatch the gather per proof.
//!
//! [`crate::gen::gather`] explains the kernel and why it is a restructuring of
//! `g16-metal/src/shaders/gather.metal` rather than a translation. This file is the other
//! half: the buffers the restructuring needs, and the dispatch.
//!
//! # What is per key and what is per proof
//!
//! [`CsrTables`] is built from `pk.coeffs` alone. It contains no witness, so it is uploaded
//! once in `prepare` and never touched again, however many proofs run against the key. At
//! `js_16x16_d32` that is about 23 MB of coefficient values plus 3 MB of indices, and moving
//! it per proof would cost more than the gather itself.
//!
//! Per proof the only host work in this stage is packing the witness (one limb split per
//! entry, no Montgomery reduction, design §3) and pushing one 32-byte parameter block per
//! dispatch, which is one dispatch below a 2^23 domain.
//!
//! # Everything is validated on the host, because the device will not complain
//!
//! An out-of-range index in WGSL is not a fault. naga and every browser bounds-check storage
//! access, so `WITNESS[signal]` with a signal past the end of the witness quietly returns
//! zero and the proof comes out wrong with nothing in any log. [`CsrTables::build`] therefore
//! walks the CSR once at key load and rejects a non-monotone `row_ptr`, a `row_ptr` of the
//! wrong length, a nonzero count that disagrees with the arrays, and a signal at or past
//! `n_vars`. `g16_core::cpu::CpuCircuit::prepare` already checks two of those four, but this
//! backend must not depend on somebody else having looked.

use bytemuck::{Pod, Zeroable};
use g16_core::ProveError;
use g16_field::Fr;
use g16_gpu_layout::{PackedFr, LIMBS};
use g16_zkey::Coefficients;

use crate::device::{bad, WgpuBackend};
use crate::gen::field::Variant;
use crate::gen::gather as wgsl;
use crate::params::ParamRing;
use crate::pipelines::Kernels;

/// Bytes one `Fr` occupies on the device. 32, and the same 32 `g16-metal` uploads.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

/// The uniform block [`wgsl::ENTRY`] reads. Field for field the WGSL `GatherParams`.
///
/// Eight `u32` rather than six because WGSL rounds a uniform struct up to a multiple of its
/// 16-byte alignment, so the shader-side struct is 32 bytes whatever this one says. Writing
/// the padding out makes the two obviously the same size instead of accidentally the same
/// size. `ParamRing::push` cannot check this for us and nothing else will either.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GatherParams {
    /// First row this dispatch computes.
    pub row_lo: u32,
    /// One past the last row this dispatch computes.
    pub row_hi: u32,
    /// Index of A's `row_ptr[0]` inside the concatenated `ROW_PTR`. Always 0, passed
    /// anyway: a kernel that assumes it is zero is a kernel that breaks the day the
    /// concatenation order changes.
    pub row_base_a: u32,
    /// Index of B's `row_ptr[0]`, which is `domain_size + 1`.
    pub row_base_b: u32,
    /// Index of A's first nonzero inside `SIGNAL` and `VALUE`. Always 0.
    pub nz_base_a: u32,
    /// Index of B's first nonzero, which is A's nonzero count.
    pub nz_base_b: u32,
    pub pad0: u32,
    pub pad1: u32,
}

// ---------------------------------------------------------------------------
// The concatenated CSR, on the host
// ---------------------------------------------------------------------------

/// The three concatenated arrays and the four base offsets, before any GPU is involved.
///
/// Split out from [`CsrTables`] so the concatenation is testable without a device, which
/// matters because an off-by-one in `nz_base_b` produces a proof that is wrong rather than a
/// proof that fails, and a host test can compare against `pk.coeffs` byte for byte where a
/// device test can only compare the final vectors.
pub struct CsrHost {
    /// `row_ptr[0]` then `row_ptr[1]`, unbiased, `2 * (domain_size + 1)` entries.
    pub row_ptr: Vec<u32>,
    /// `signal[0]` then `signal[1]`.
    pub signal: Vec<u32>,
    /// `value[0]` then `value[1]`, as Montgomery limbs, 8 `u32` per element.
    pub value: Vec<u32>,
    /// Where each matrix's `row_ptr` starts inside [`Self::row_ptr`].
    pub row_base: [u32; 2],
    /// Where each matrix's nonzeros start inside [`Self::signal`] and [`Self::value`].
    pub nz_base: [u32; 2],
    /// Nonzeros per matrix.
    pub nnz: [u32; 2],
    /// Rows, which is the domain size.
    pub n_rows: u32,
    /// Witness length the signals were checked against.
    pub n_vars: u32,
}

impl CsrHost {
    /// Concatenates and validates. `O(domain_size + nnz)`, paid once per key.
    pub fn build(
        coeffs: &Coefficients,
        domain_size: usize,
        n_vars: usize,
    ) -> Result<Self, ProveError> {
        if domain_size == 0 {
            return Err(bad("domain size is zero"));
        }
        if n_vars == 0 {
            return Err(bad("n_vars is zero"));
        }
        // 2^23 rows is 8.4 million and already needs two dispatches; the u32 ceiling is far
        // beyond any circuit that fits in memory, but the casts below are only sound because
        // this is checked.
        let rows_plus_one = domain_size
            .checked_add(1)
            .filter(|&v| u32::try_from(v).is_ok())
            .ok_or_else(|| bad(format!("domain size {domain_size} overflows u32")))?;
        u32::try_from(n_vars).map_err(|_| bad(format!("n_vars {n_vars} overflows u32")))?;

        let mut row_ptr = Vec::with_capacity(2 * rows_plus_one);
        let mut nnz = [0u32; 2];
        for (m, name) in [(0usize, "A"), (1usize, "B")] {
            let rp = &coeffs.row_ptr[m];
            if rp.len() != rows_plus_one {
                return Err(bad(format!(
                    "matrix {name} has {} row_ptr entries, domain size {domain_size} needs {rows_plus_one}",
                    rp.len()
                )));
            }
            // Monotonicity is what makes `lo < hi` a bounded loop in the kernel. A single
            // decreasing pair would make one thread loop until it walked off the end of a
            // 4 GiB address space, which on a GPU is a hang and not an error.
            for c in 1..rp.len() {
                if rp[c] < rp[c - 1] {
                    return Err(bad(format!(
                        "matrix {name} row_ptr is not monotone at row {}: {} then {}",
                        c - 1,
                        rp[c - 1],
                        rp[c]
                    )));
                }
            }
            let total = rp[rp.len() - 1] as usize;
            if total != coeffs.signal[m].len() || total != coeffs.value[m].len() {
                return Err(bad(format!(
                    "matrix {name} row_ptr ends at {total} but has {} signals and {} values",
                    coeffs.signal[m].len(),
                    coeffs.value[m].len()
                )));
            }
            nnz[m] = rp[rp.len() - 1];
            row_ptr.extend_from_slice(rp);
        }

        let nz_base = [0u32, nnz[0]];
        let total_nz = nnz[0]
            .checked_add(nnz[1])
            .ok_or_else(|| bad("A and B together have more than 2^32 nonzeros"))?
            as usize;

        let mut signal = Vec::with_capacity(total_nz);
        let mut value = Vec::with_capacity(total_nz * LIMBS);
        for (m, name) in [(0usize, "A"), (1usize, "B")] {
            for (k, &s) in coeffs.signal[m].iter().enumerate() {
                if s as usize >= n_vars {
                    return Err(bad(format!(
                        "matrix {name} nonzero {k} references signal {s}, witness has {n_vars}"
                    )));
                }
            }
            signal.extend_from_slice(&coeffs.signal[m]);
            push_fr_words(&coeffs.value[m], &mut value);
        }

        Ok(Self {
            row_ptr,
            signal,
            value,
            row_base: [0, rows_plus_one as u32],
            nz_base,
            nnz,
            n_rows: domain_size as u32,
            n_vars: n_vars as u32,
        })
    }

    /// The four base offsets, as the kernel wants them. `row_lo` and `row_hi` are the
    /// caller's, because they change per dispatch and these do not.
    pub fn params(&self) -> GatherParams {
        GatherParams {
            row_lo: 0,
            row_hi: self.n_rows,
            row_base_a: self.row_base[0],
            row_base_b: self.row_base[1],
            nz_base_a: self.nz_base[0],
            nz_base_b: self.nz_base[1],
            pad0: 0,
            pad1: 0,
        }
    }
}

/// Montgomery limbs of `xs`, appended to `out`. Eight `u32` per element, little endian,
/// which is `PackedFr` verbatim.
///
/// A copy of `Fp::0.0` and nothing else: arkworks already stores an `Fr` in Montgomery form
/// with `R = 2^256`, the same radix the generated CIOS uses, so there is no conversion on
/// either side. Design §3 leans on this being nothing but a limb split, because in the
/// browser it runs single-threaded.
pub fn push_fr_words(xs: &[Fr], out: &mut Vec<u32>) {
    out.reserve(xs.len() * LIMBS);
    for x in xs {
        out.extend_from_slice(&PackedFr::from_fr(x).v);
    }
}

/// Montgomery limbs of `xs` as a fresh vector.
pub fn fr_words(xs: &[Fr]) -> Vec<u32> {
    let mut out = Vec::with_capacity(xs.len() * LIMBS);
    push_fr_words(xs, &mut out);
    out
}

/// The inverse of [`fr_words`], for reading a device buffer back.
///
/// Returns `None` on a word count that is not a whole number of elements, which is the shape
/// a wrong stride takes: silently dropping a trailing partial element would turn a length bug
/// into a comparison that passes on the elements it did read.
pub fn fr_from_words(words: &[u32]) -> Option<Vec<Fr>> {
    if !words.len().is_multiple_of(LIMBS) {
        return None;
    }
    Some(
        words
            .chunks_exact(LIMBS)
            .map(|c| {
                let mut v = [0u32; LIMBS];
                v.copy_from_slice(c);
                PackedFr { v }.to_fr()
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// The concatenated CSR, on the device
// ---------------------------------------------------------------------------

/// [`CsrHost`] uploaded. Witness independent, so this is built in `prepare` and outlives
/// every proof against the key.
pub struct CsrTables {
    row_ptr: wgpu::Buffer,
    signal: wgpu::Buffer,
    value: wgpu::Buffer,
    params: GatherParams,
    n_rows: u32,
    n_vars: u32,
    nnz: [u32; 2],
    bytes: u64,
}

impl CsrTables {
    /// Uploads a prepared [`CsrHost`].
    pub fn upload(backend: &WgpuBackend, host: &CsrHost) -> Result<Self, ProveError> {
        let limits = backend.granted_limits();
        let value_bytes = host.value.len() as u64 * 4;
        // The first thing that breaks at scale, so it is named rather than left to a wgpu
        // validation message about an anonymous buffer. 128 MiB at the Floor is 4.19 million
        // nonzeros, which is a 2^22 domain at the measured 1.13 to 1.69 nonzeros per row.
        if value_bytes > limits.max_storage_buffer_binding_size {
            return Err(bad(format!(
                "the concatenated coefficient values are {value_bytes} bytes ({} nonzeros at \
                 {FR_BYTES} each), over the {} byte storage binding limit; chunking the CSR is \
                 unit U15",
                host.signal.len(),
                limits.max_storage_buffer_binding_size
            )));
        }

        let row_ptr = storage_u32(backend, "g16 csr row_ptr", &host.row_ptr)?;
        let signal = storage_u32(backend, "g16 csr signal", &host.signal)?;
        // Padded to a whole `Fr` and not to one word, which is what `storage_u32` does and
        // what this call site used to rely on.
        //
        // Found at U7 by a 2^0 synthetic key whose B matrix has no nonzeros: WebGPU sizes a
        // runtime-sized array binding by its element stride, so binding a 4-byte buffer as
        // `array<Fr>` is a hard validation error, "the buffer bound at binding index 3 is
        // bound with size 4 where the shader expects 32", and the whole dispatch is dropped.
        // `signal` and `row_ptr` are `array<u32>` and one word really is enough for them.
        // `crate::ntt::NttTables` already pads its `Fr` tables to `LIMBS` words for the same
        // reason; this was the one place that did not.
        let value = if host.value.is_empty() {
            storage_u32(backend, "g16 csr value", &[0u32; LIMBS])?
        } else {
            storage_u32(backend, "g16 csr value", &host.value)?
        };
        let bytes = row_ptr.size() + signal.size() + value.size();

        Ok(Self {
            row_ptr,
            signal,
            value,
            params: host.params(),
            n_rows: host.n_rows,
            n_vars: host.n_vars,
            nnz: host.nnz,
            bytes,
        })
    }

    /// Build and upload in one step, which is what `prepare` calls.
    pub fn from_coefficients(
        backend: &WgpuBackend,
        coeffs: &Coefficients,
        domain_size: usize,
        n_vars: usize,
    ) -> Result<Self, ProveError> {
        Self::upload(backend, &CsrHost::build(coeffs, domain_size, n_vars)?)
    }

    /// Rows, which is the domain size.
    pub fn n_rows(&self) -> u32 {
        self.n_rows
    }

    /// Witness length the signals were validated against.
    pub fn n_vars(&self) -> u32 {
        self.n_vars
    }

    /// Nonzeros per matrix, `[A, B]`.
    pub fn nnz(&self) -> [u32; 2] {
        self.nnz
    }

    /// Device bytes the three buffers occupy.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// One `STORAGE | COPY_DST` buffer holding `data`, written with `queue.write_buffer`.
///
/// `write_buffer` and not `mapped_at_creation`: design §3 records staging through a mapped
/// buffer measuring 9x slower than `write_buffer` on this platform, and the WebGPU rule that
/// a mappable buffer can never also be a storage buffer means the mapped path needs a
/// device-to-device copy on top.
///
/// A zero-length allocation is illegal in WebGPU and does happen here: a key whose B matrix
/// has no nonzeros at all gives an empty `signal` array. One padding word costs nothing and
/// the kernel never reads it, because `lo == hi` for every row of such a matrix.
///
/// One word is enough only for an `array<u32>` binding. A buffer bound as `array<Fr>` must
/// be at least one 32-byte element or bind-group creation fails outright, so the caller
/// pads those itself; see `CsrTables::upload` and `NttTables::new`.
pub(crate) fn storage_u32(
    backend: &WgpuBackend,
    label: &str,
    data: &[u32],
) -> Result<wgpu::Buffer, ProveError> {
    let bytes = (data.len().max(1) as u64) * 4;
    if bytes > backend.granted_limits().max_buffer_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes, over the {} byte buffer limit",
            backend.granted_limits().max_buffer_size
        )));
    }
    let buf = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        backend
            .queue()
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
    }
    Ok(buf)
}

/// A device buffer of `count` `Fr`, `STORAGE | COPY_SRC | COPY_DST`.
///
/// Here rather than in a scratch pool because U7 owns the pool and this unit needs somewhere
/// for its outputs to land. U7 replaces the call site, not the layout.
pub fn fr_buffer(
    backend: &WgpuBackend,
    label: &str,
    count: u32,
) -> Result<wgpu::Buffer, ProveError> {
    let bytes = count as u64 * FR_BYTES;
    let limits = backend.granted_limits();
    if bytes == 0 {
        return Err(bad(format!("{label}: a zero length buffer is not legal")));
    }
    if bytes > limits.max_storage_buffer_binding_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes for {count} field elements, over the {} byte storage \
             binding limit",
            limits.max_storage_buffer_binding_size
        )));
    }
    Ok(backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Uploads `xs` as Montgomery limbs into a new storage buffer.
pub fn upload_fr(
    backend: &WgpuBackend,
    label: &str,
    xs: &[Fr],
) -> Result<wgpu::Buffer, ProveError> {
    let count =
        u32::try_from(xs.len()).map_err(|_| bad(format!("{label}: {} elements", xs.len())))?;
    let buf = fr_buffer(backend, label, count)?;
    backend
        .queue()
        .write_buffer(&buf, 0, bytemuck::cast_slice(&fr_words(xs)));
    Ok(buf)
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Stage 0's module, bind group layout and pipeline. Built once per device.
pub struct GatherAbc {
    kernels: Kernels,
    bgl: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
    rows_per_dispatch: u32,
    workgroup: u32,
    source_len: usize,
}

impl GatherAbc {
    /// Compiles stage 0 at the device's own workgroup-per-dimension limit.
    pub fn new(backend: &WgpuBackend) -> Result<Self, ProveError> {
        let max_wg = backend
            .granted_limits()
            .max_compute_workgroups_per_dimension;
        Self::with_rows_per_dispatch(backend, max_wg.saturating_mul(wgsl::WORKGROUP))
    }

    /// Same, with the per-dispatch row cap forced.
    ///
    /// Public because the only way to exercise the multi-dispatch path on a machine whose
    /// artifacts are all 2^18 is to shrink the cap, and a test that cannot reach a code path
    /// is a code path that ships untested. It is rounded down to a whole number of
    /// workgroups, because a partial workgroup would leave rows uncovered between chunks.
    pub fn with_rows_per_dispatch(
        backend: &WgpuBackend,
        rows_per_dispatch: u32,
    ) -> Result<Self, ProveError> {
        Self::with_shape(backend, rows_per_dispatch, wgsl::WORKGROUP)
    }

    /// Same, with the workgroup size forced as well.
    ///
    /// Public for `tests/gather.rs`'s sweep, which is how [`wgsl::WORKGROUP`] became a
    /// measured number instead of a copied one. Nothing in the backend calls it.
    pub fn with_shape(
        backend: &WgpuBackend,
        rows_per_dispatch: u32,
        workgroup: u32,
    ) -> Result<Self, ProveError> {
        let limits = backend.granted_limits();
        if workgroup == 0 || workgroup > limits.max_compute_invocations_per_workgroup {
            return Err(bad(format!(
                "workgroup size {workgroup} is outside 1..={}",
                limits.max_compute_invocations_per_workgroup
            )));
        }
        let rows = rows_per_dispatch - rows_per_dispatch % workgroup;
        if rows == 0 {
            return Err(bad(format!(
                "rows_per_dispatch {rows_per_dispatch} is under one {workgroup} thread workgroup"
            )));
        }
        let max_wg = limits.max_compute_workgroups_per_dimension;
        if rows.div_ceil(workgroup) > max_wg {
            return Err(bad(format!(
                "rows_per_dispatch {rows} needs {} workgroups, over the {max_wg} limit",
                rows.div_ceil(workgroup)
            )));
        }

        let device = backend.device();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("g16 gather"),
            entries: &Self::bind_group_layout_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("g16 gather"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let source = wgsl::gather_module_at(Variant::default(), workgroup);
        let source_len = source.len();
        let kernels = Kernels::build(backend, "gather", &source, &layout, &[wgsl::ENTRY])?;

        Ok(Self {
            kernels,
            bgl,
            _layout: layout,
            rows_per_dispatch: rows,
            workgroup,
            source_len,
        })
    }

    /// The bind group layout, as data, so a test can count what the pipeline layout declares.
    ///
    /// This is the list the layout is actually built from, so the count a test takes here is
    /// the count the device enforces. wgpu offers no way to read entries back off a
    /// constructed `BindGroupLayout`, and duplicating the list in the test would only prove
    /// the duplicate was right.
    pub fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
        [
            ParamRing::layout_entry(wgsl::BIND_PARAMS),
            storage_entry(wgsl::BIND_ROW_PTR, true),
            storage_entry(wgsl::BIND_SIGNAL, true),
            storage_entry(wgsl::BIND_VALUE, true),
            storage_entry(wgsl::BIND_WITNESS, true),
            storage_entry(wgsl::BIND_OUT_A, false),
            storage_entry(wgsl::BIND_OUT_B, false),
            storage_entry(wgsl::BIND_OUT_C, false),
        ]
    }

    /// Storage buffers the pipeline layout declares, counted from
    /// [`Self::bind_group_layout_entries`]. Must be at most 8, which is the Floor.
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

    /// Binds one CSR, one witness and three outputs. Valid until any of them is dropped.
    ///
    /// The three outputs are checked against `csr.n_rows()` and the witness against
    /// `csr.n_vars()` here, because WebGPU derives a runtime-sized array's length from the
    /// binding size and then silently drops out-of-range writes. A short `out_c` would
    /// otherwise produce a correct-looking A and B and a truncated C.
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        backend: &WgpuBackend,
        csr: &CsrTables,
        ring: &ParamRing,
        witness: &wgpu::Buffer,
        out_a: &wgpu::Buffer,
        out_b: &wgpu::Buffer,
        out_c: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup, ProveError> {
        let want_out = csr.n_rows as u64 * FR_BYTES;
        for (name, buf) in [("out_a", out_a), ("out_b", out_b), ("out_c", out_c)] {
            if buf.size() < want_out {
                return Err(bad(format!(
                    "{name} is {} bytes, the gather writes {} rows = {want_out} bytes",
                    buf.size(),
                    csr.n_rows
                )));
            }
        }
        let want_w = csr.n_vars as u64 * FR_BYTES;
        if witness.size() < want_w {
            return Err(bad(format!(
                "the witness buffer is {} bytes, the CSR indexes {} signals = {want_w} bytes",
                witness.size(),
                csr.n_vars
            )));
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
                label: Some("g16 gather"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: wgsl::BIND_PARAMS,
                        resource: wgpu::BindingResource::Buffer(ring.binding()),
                    },
                    entry(wgsl::BIND_ROW_PTR, &csr.row_ptr),
                    entry(wgsl::BIND_SIGNAL, &csr.signal),
                    entry(wgsl::BIND_VALUE, &csr.value),
                    entry(wgsl::BIND_WITNESS, witness),
                    entry(wgsl::BIND_OUT_A, out_a),
                    entry(wgsl::BIND_OUT_B, out_b),
                    entry(wgsl::BIND_OUT_C, out_c),
                ],
            }))
    }

    /// Pushes one parameter block per dispatch and returns their dynamic offsets.
    ///
    /// Separate from [`Self::encode`] because every parameter block for the whole proof is
    /// written in one `write_buffer` before encoding starts (design §3), so the pushes have
    /// to happen before the compute pass exists rather than inside it.
    pub fn plan(&self, csr: &CsrTables, ring: &mut ParamRing) -> Result<Vec<u32>, ProveError> {
        let mut offsets = Vec::with_capacity(self.dispatches(csr.n_rows) as usize);
        let mut lo = 0u32;
        while lo < csr.n_rows {
            let hi = (lo + self.rows_per_dispatch).min(csr.n_rows);
            offsets.push(ring.push(&GatherParams {
                row_lo: lo,
                row_hi: hi,
                ..csr.params
            })?);
            lo = hi;
        }
        Ok(offsets)
    }

    /// Records the dispatches. `offsets` is what [`Self::plan`] returned for the same `csr`.
    pub fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bind: &wgpu::BindGroup,
        csr: &CsrTables,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let want = self.dispatches(csr.n_rows) as usize;
        if offsets.len() != want {
            return Err(bad(format!(
                "the gather over {} rows needs {want} dispatches, got {} parameter offsets",
                csr.n_rows,
                offsets.len()
            )));
        }
        pass.set_pipeline(self.kernels.get(wgsl::ENTRY)?);
        let mut lo = 0u32;
        for &off in offsets {
            let hi = (lo + self.rows_per_dispatch).min(csr.n_rows);
            pass.set_bind_group(0, bind, &[off]);
            pass.dispatch_workgroups((hi - lo).div_ceil(self.workgroup), 1, 1);
            lo = hi;
        }
        Ok(())
    }

    /// Dispatches `n_rows` needs, which is also the parameter ring slots stage 0 consumes.
    pub fn dispatches(&self, n_rows: u32) -> u32 {
        n_rows.div_ceil(self.rows_per_dispatch)
    }

    /// Rows one dispatch covers: 8,388,480 at the Floor (128 threads x 65535 workgroups),
    /// less if forced.
    pub fn rows_per_dispatch(&self) -> u32 {
        self.rows_per_dispatch
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

/// The layout entry for one storage binding.
///
/// `pub(crate)` so `crate::pointwise` builds its layout from the same helper: a
/// `read_only: false` entry and a `var<storage, read>` declaration are not interchangeable
/// in WebGPU, and having one helper is one fewer place for the two to disagree.
pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
