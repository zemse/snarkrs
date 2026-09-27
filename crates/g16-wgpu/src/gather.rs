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
//! zero and the proof comes out wrong with nothing in any log. [`CsrHost::build`] therefore
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
    /// The first nonzero of A that the bound `SIGNAL` and `VALUE` hold. `row_ptr` numbers
    /// nonzeros from the start of the matrix and a chunked CSR binds only its own rows'
    /// nonzeros, so the kernel subtracts this before adding `nz_base_a`. Zero for a key that
    /// fits in one chunk.
    pub nz_lo_a: u32,
    /// Same for B.
    pub nz_lo_b: u32,
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
            nz_lo_a: 0,
            nz_lo_b: 0,
        }
    }

    /// Splits the rows into chunks of at most `chunk_nz` nonzeros, A's and B's together.
    ///
    /// Greedy from row 0: a chunk takes rows until the next one would push it over. The
    /// nonzeros of a row range are contiguous in both matrices, so a chunk's `signal` and
    /// `value` are two slices of the host arrays and no per-nonzero work happens here. A
    /// single row over `chunk_nz` on its own is an error, because no partition holds it.
    pub fn row_chunks(&self, chunk_nz: u32) -> Result<Vec<CsrChunkHost>, ProveError> {
        if chunk_nz == 0 {
            return Err(bad("a CSR chunk of zero nonzeros holds no row"));
        }
        let rows = self.n_rows as usize;
        let rp_a = &self.row_ptr[..rows + 1];
        let rp_b = &self.row_ptr[rows + 1..];
        let mut out = Vec::new();
        let mut r0 = 0usize;
        while r0 < rows {
            let (ka0, kb0) = (rp_a[r0], rp_b[r0]);
            let mut r1 = r0;
            while r1 < rows {
                let held = u64::from(rp_a[r1 + 1] - ka0) + u64::from(rp_b[r1 + 1] - kb0);
                if held > u64::from(chunk_nz) {
                    break;
                }
                r1 += 1;
            }
            if r1 == r0 {
                return Err(bad(format!(
                    "row {r0} has {} nonzeros across A and B, more than the {chunk_nz} a CSR \
                     chunk can bind",
                    (rp_a[r0 + 1] - ka0) + (rp_b[r0 + 1] - kb0)
                )));
            }
            out.push(CsrChunkHost {
                row_lo: r0 as u32,
                row_hi: r1 as u32,
                nz_lo: [ka0, kb0],
                nz_hi: [rp_a[r1], rp_b[r1]],
            });
            r0 = r1;
        }
        Ok(out)
    }
}

/// One row chunk of a [`CsrHost`]: which rows, and which nonzeros of each matrix they own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrChunkHost {
    pub row_lo: u32,
    pub row_hi: u32,
    /// First nonzero of each matrix in this chunk, in the matrix's own numbering.
    pub nz_lo: [u32; 2],
    /// One past the last.
    pub nz_hi: [u32; 2],
}

impl CsrChunkHost {
    /// Nonzeros this chunk binds, both matrices together.
    pub fn nnz(&self) -> u32 {
        (self.nz_hi[0] - self.nz_lo[0]) + (self.nz_hi[1] - self.nz_lo[1])
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

/// One row chunk of the CSR on the device: its rows' nonzeros of both matrices, A's first,
/// and the parameter block that tells the kernel where they start.
struct CsrChunk {
    row_lo: u32,
    row_hi: u32,
    signal: wgpu::Buffer,
    value: wgpu::Buffer,
    params: GatherParams,
}

/// [`CsrHost`] uploaded. Witness independent, so this is built in `prepare` and outlives
/// every proof against the key.
///
/// # Row chunks, for a CSR over the storage binding limit
///
/// The coefficient values are 32 bytes a nonzero and a storage binding is 128 MiB at the
/// floor, so a key past 4.19 million nonzeros cannot bind them at once: `js_384x384_d32` has
/// 17.7 million and wants 567 MB. The kernel binds seven storage buffers and the floor allows
/// eight, so the values cannot be spread over several bindings either. Instead the rows are
/// split into chunks whose nonzeros fit, each chunk gets its own `signal` and `value` buffers
/// and its own bind group, and the gather is dispatched per chunk. `row_ptr` stays whole and
/// unbiased; the chunk's `nz_lo_*` tells the kernel where its buffers start in the matrix's
/// numbering. A key that fits is one chunk, one bind group and the same dispatch as before.
pub struct CsrTables {
    row_ptr: wgpu::Buffer,
    chunks: Vec<CsrChunk>,
    n_rows: u32,
    n_vars: u32,
    nnz: [u32; 2],
    bytes: u64,
}

impl CsrTables {
    /// Uploads a prepared [`CsrHost`] in chunks that fit this device's storage binding.
    pub fn upload(backend: &WgpuBackend, host: &CsrHost) -> Result<Self, ProveError> {
        Self::upload_chunked(backend, host, Self::chunk_nz(backend))
    }

    /// Same, with at most `chunk_nz` nonzeros per chunk. Public so a test can force the
    /// chunked path on a key that would otherwise fit in one.
    pub fn upload_chunked(
        backend: &WgpuBackend,
        host: &CsrHost,
        chunk_nz: u32,
    ) -> Result<Self, ProveError> {
        let limit = Self::chunk_nz(backend);
        if chunk_nz > limit {
            return Err(bad(format!(
                "a CSR chunk of {chunk_nz} nonzeros is {} bytes of values, over the {} byte \
                 storage binding limit ({limit} nonzeros)",
                u64::from(chunk_nz) * FR_BYTES,
                backend.granted_limits().max_storage_buffer_binding_size
            )));
        }
        let row_ptr = storage_u32(backend, "g16 csr row_ptr", &host.row_ptr)?;
        let mut bytes = row_ptr.size();
        let nnz_a = host.nnz[0] as usize;
        let mut chunks = Vec::new();
        for c in host.row_chunks(chunk_nz)? {
            let (a, b) = (
                c.nz_lo[0] as usize..c.nz_hi[0] as usize,
                nnz_a + c.nz_lo[1] as usize..nnz_a + c.nz_hi[1] as usize,
            );
            let signal = storage_parts(
                backend,
                "g16 csr signal",
                &[&host.signal[a.clone()], &host.signal[b.clone()]],
                1,
            )?;
            // Padded to a whole `Fr` and not to one word, which is what `storage_parts`
            // would do on its own.
            //
            // Found at U7 by a 2^0 synthetic key whose B matrix has no nonzeros: WebGPU sizes
            // a runtime-sized array binding by its element stride, so binding a 4-byte buffer
            // as `array<Fr>` is a hard validation error, "the buffer bound at binding index 3
            // is bound with size 4 where the shader expects 32", and the whole dispatch is
            // dropped. `signal` and `row_ptr` are `array<u32>` and one word really is enough
            // for them. `crate::ntt::NttTables` already pads its `Fr` tables to `LIMBS` words
            // for the same reason; this was the one place that did not.
            let value = storage_parts(
                backend,
                "g16 csr value",
                &[
                    &host.value[a.start * LIMBS..a.end * LIMBS],
                    &host.value[b.start * LIMBS..b.end * LIMBS],
                ],
                LIMBS,
            )?;
            bytes += signal.size() + value.size();
            chunks.push(CsrChunk {
                row_lo: c.row_lo,
                row_hi: c.row_hi,
                signal,
                value,
                params: GatherParams {
                    // B's nonzeros follow A's chunk, so B's base is A's count in this chunk
                    // and not in the whole matrix.
                    nz_base_b: c.nz_hi[0] - c.nz_lo[0],
                    nz_lo_a: c.nz_lo[0],
                    nz_lo_b: c.nz_lo[1],
                    ..host.params()
                },
            });
        }

        Ok(Self {
            row_ptr,
            chunks,
            n_rows: host.n_rows,
            n_vars: host.n_vars,
            nnz: host.nnz,
            bytes,
        })
    }

    /// Nonzeros one chunk's value buffer can hold on this device: 4,194,304 at the floor.
    pub fn chunk_nz(backend: &WgpuBackend) -> u32 {
        let limit = backend.granted_limits().max_storage_buffer_binding_size / FR_BYTES;
        u32::try_from(limit).unwrap_or(u32::MAX)
    }

    /// Row chunks the CSR was uploaded in. One for every key that fits the binding limit.
    pub fn chunks(&self) -> usize {
        self.chunks.len()
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
/// pads those itself; see `CsrTables::upload_chunked` and `NttTables::new`.
pub(crate) fn storage_u32(
    backend: &WgpuBackend,
    label: &str,
    data: &[u32],
) -> Result<wgpu::Buffer, ProveError> {
    let bytes = (data.len().max(1) as u64) * 4;
    let limits = backend.granted_limits();
    if bytes > limits.max_buffer_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes, over the {} byte buffer limit",
            limits.max_buffer_size
        )));
    }
    // Every buffer from here is bound as `array<u32>` or `array<Fr>`, so the binding limit is
    // the one that actually bites: 128 MiB against 256 MiB at the Floor. Without this a 2^23
    // `coset_pows` allocates cleanly and then fails inside `create_bind_group` with wgpu's own
    // message about an anonymous buffer.
    if bytes > limits.max_storage_buffer_binding_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes, over the {} byte storage binding limit",
            limits.max_storage_buffer_binding_size
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

/// [`storage_u32`] over several slices laid end to end, without concatenating them on the
/// host: one buffer of their total length, at least `min_words`, and one `write_buffer` per
/// slice at its offset. A CSR chunk is two slices of the host arrays, A's nonzeros then B's,
/// and at 17.7 million nonzeros copying them into a temporary first would be 640 MB of
/// memcpy for nothing.
pub(crate) fn storage_parts(
    backend: &WgpuBackend,
    label: &str,
    parts: &[&[u32]],
    min_words: usize,
) -> Result<wgpu::Buffer, ProveError> {
    let words = parts.iter().map(|p| p.len()).sum::<usize>().max(min_words);
    let bytes = words as u64 * 4;
    let limits = backend.granted_limits();
    if bytes > limits.max_storage_buffer_binding_size {
        return Err(bad(format!(
            "{label} wants {bytes} bytes, over the {} byte storage binding limit",
            limits.max_storage_buffer_binding_size
        )));
    }
    let buf = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut at = 0u64;
    for p in parts {
        if !p.is_empty() {
            backend
                .queue()
                .write_buffer(&buf, at, bytemuck::cast_slice(p));
        }
        at += p.len() as u64 * 4;
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
        // The browser floor, not the granted limit. `gen::gather::gather_module_at` asserts
        // the same ceiling, and the two disagreed until U11: under `G16_WGPU_LIMITS=raised`
        // this adapter grants 1024, so a 512-thread request passed here and panicked in the
        // generator. See `WgpuBackend::ceiling_invocations`.
        let limits = backend.granted_limits();
        let ceiling = backend.ceiling_invocations();
        if workgroup == 0 || workgroup > ceiling {
            return Err(bad(format!(
                "workgroup size {workgroup} is outside 1..={ceiling}"
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

    /// Binds one CSR, one witness and three outputs, one bind group per CSR chunk. Valid
    /// until any of them is dropped.
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
    ) -> Result<Vec<wgpu::BindGroup>, ProveError> {
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
        Ok(csr
            .chunks
            .iter()
            .map(|c| {
                backend
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
                            entry(wgsl::BIND_SIGNAL, &c.signal),
                            entry(wgsl::BIND_VALUE, &c.value),
                            entry(wgsl::BIND_WITNESS, witness),
                            entry(wgsl::BIND_OUT_A, out_a),
                            entry(wgsl::BIND_OUT_B, out_b),
                            entry(wgsl::BIND_OUT_C, out_c),
                        ],
                    })
            })
            .collect())
    }

    /// The dispatches `csr` takes, in order: each chunk's rows in slices of
    /// `rows_per_dispatch`, as `(chunk, row_lo, row_hi)`.
    fn ranges(&self, csr: &CsrTables) -> Vec<(usize, u32, u32)> {
        let mut out = Vec::new();
        for (ci, c) in csr.chunks.iter().enumerate() {
            let mut lo = c.row_lo;
            while lo < c.row_hi {
                let hi = (lo + self.rows_per_dispatch).min(c.row_hi);
                out.push((ci, lo, hi));
                lo = hi;
            }
        }
        out
    }

    /// Pushes one parameter block per dispatch and returns their dynamic offsets.
    ///
    /// Separate from [`Self::encode`] because every parameter block for the whole proof is
    /// written in one `write_buffer` before encoding starts (design §3), so the pushes have
    /// to happen before the compute pass exists rather than inside it.
    pub fn plan(&self, csr: &CsrTables, ring: &mut ParamRing) -> Result<Vec<u32>, ProveError> {
        self.ranges(csr)
            .into_iter()
            .map(|(ci, lo, hi)| {
                ring.push(&GatherParams {
                    row_lo: lo,
                    row_hi: hi,
                    ..csr.chunks[ci].params
                })
            })
            .collect()
    }

    /// Records the dispatches. `binds` is what [`Self::bind`] returned for `csr` and
    /// `offsets` is what [`Self::plan`] returned for it.
    pub fn encode(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        binds: &[wgpu::BindGroup],
        csr: &CsrTables,
        offsets: &[u32],
    ) -> Result<(), ProveError> {
        let ranges = self.ranges(csr);
        if offsets.len() != ranges.len() {
            return Err(bad(format!(
                "the gather over {} rows needs {} dispatches, got {} parameter offsets",
                csr.n_rows,
                ranges.len(),
                offsets.len()
            )));
        }
        if binds.len() != csr.chunks.len() {
            return Err(bad(format!(
                "the gather over {} CSR chunks needs as many bind groups, got {}",
                csr.chunks.len(),
                binds.len()
            )));
        }
        pass.set_pipeline(self.kernels.get(wgsl::ENTRY)?);
        for (&off, (ci, lo, hi)) in offsets.iter().zip(ranges) {
            pass.set_bind_group(0, &binds[ci], &[off]);
            crate::readback::dispatch(pass, (hi - lo).div_ceil(self.workgroup));
        }
        Ok(())
    }

    /// Dispatches `csr` needs, which is also the parameter ring slots stage 0 consumes: one
    /// per `rows_per_dispatch` rows of each chunk.
    pub fn dispatches(&self, csr: &CsrTables) -> u32 {
        self.ranges(csr).len() as u32
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
