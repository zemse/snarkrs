//! Stages 1 to 3 on the host: the pass split, the two twiddle tables, the coset power table,
//! and the encoding of one transform as a chain of batch dispatches.
//!
//! [`crate::gen::ntt`] explains the kernels. This file is the other half: what is uploaded
//! once per key, how eighteen passes become three dispatches, and the bind groups the four
//! entry points need.
//!
//! # Why passes are batched at all
//!
//! One dispatch per pass is `log n` round trips through device memory. At 2^18 that is 18
//! passes over 8 MB read and 8 MB written, and the transform is memory bound with the ALU
//! idle. A batch instead pulls a slice into workgroup memory, runs several passes there with
//! only a `workgroupBarrier()` between them, and writes back once.
//!
//! Which passes can share a slice is fixed by the index algebra, not by taste. Pass `t`
//! flips bit `t` of an element index, so passes `[s0, s0+k)` touch only bits `[s0, s0+k)`,
//! and a group is any set of indices agreeing on every other bit. That is a contiguous run
//! of `2^k` elements when `s0 = 0` and a run strided by `2^s0` otherwise. Standard six-step
//! decomposition, written as one kernel shape taking `(s0, k)`.
//!
//! # The floor is not what decides the tile, and that is the surprise
//!
//! `maxComputeWorkgroupStorageSize` is 16384 at the floor against 32768 on this M2 Max, so
//! the *ceiling* on fused passes is 9 here and 10 on Metal. Neither is what ships. Filling
//! the budget measured 5x slower at 2^18 than leaving three quarters of it unused, so
//! [`crate::gen::ntt::PREFERRED_FUSED`] caps the tile at 8 and a 2^18 domain runs 6 + 6 + 6
//! rather than the 9 + 9 design §4 specifies: one extra dispatch per transform, 68 ms back
//! across a proof. The table and the argument are on that constant.
//!
//! So the browser's storage floor costs this stage nothing at any size we ship, because the
//! shape it would force is slower than the shape that was chosen for other reasons. That is
//! the opposite of what the storage-*buffer* floor did to stage 0, which had to be
//! restructured to fit inside it.
//!
//! # What the host still does per key
//!
//! Three tables, all witness independent, all uploaded once: forward twiddles (`n/2`
//! elements), inverse twiddles (`n/2`), and `shift^j` for `j` in `[0, n)`. At 2^18 that is
//! 16 MB, against about 26 MB for the CSR. The coset powers are **not** the twiddles: the
//! twiddles are powers of the domain's own `n`-th root and the shift is a primitive `2n`-th
//! root, so no table `g16-field` builds can be reused. See
//! `g16_core::cpu::CpuCircuit::new` for why that specific element and not `Fr::GENERATOR`.

use bytemuck::{Pod, Zeroable};
use g16_core::ProveError;
use g16_field::{Domain, Field as _, Fr};
use g16_gpu_layout::{PackedFr, LIMBS};

use crate::device::{bad, WgpuBackend};
use crate::gather::{fr_words, storage_u32};
use crate::gen::field::Variant;
use crate::gen::ntt::{self as wgsl, Mode};
use crate::params::ParamRing;
use crate::pipelines::Kernels;

/// Bytes one `Fr` occupies on the device.
const FR_BYTES: u64 = (LIMBS * 4) as u64;

// ---------------------------------------------------------------------------
// The pass split
// ---------------------------------------------------------------------------

/// One batch: `k` consecutive passes starting at pass `s0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    pub s0: u32,
    pub k: u32,
}

/// Splits `log_n` passes into as few batches as the workgroup budget allows, sized as evenly
/// as possible.
///
/// Evenly, not greedily. A greedy fill at 2^18 with a cap of 10 gives 10 + 8, where the
/// second dispatch has only 2^8 workgroups sharing 2^17 butterflies; an even 9 + 9 gives both
/// dispatches 2^9 groups. The pathological version of the greedy split is what profiling of
/// bellperson caught at 2^26, where the leftover kernel launched 16 million workgroups of two
/// threads each.
///
/// Ported unchanged from `g16_metal::stages::split_passes`, and it must stay unchanged: the
/// two backends producing the same `(s0, k)` sequence is what makes a cross-backend
/// disagreement a real arithmetic bug rather than a different decomposition.
pub fn split_passes(log_n: u32, max_fused: u32) -> Vec<Batch> {
    if log_n == 0 {
        // A one-point domain has no butterflies, but the head still runs so the load scale
        // and the store epilogue happen.
        return vec![Batch { s0: 0, k: 0 }];
    }
    let batches = log_n.div_ceil(max_fused.max(1));
    let base = log_n / batches;
    let rem = log_n % batches;
    let mut out = Vec::with_capacity(batches as usize);
    let mut s0 = 0;
    for i in 0..batches {
        let k = base + u32::from(i < rem);
        out.push(Batch { s0, k });
        s0 += k;
    }
    out
}

// ---------------------------------------------------------------------------
// The kernel argument block
// ---------------------------------------------------------------------------

/// Mirrors `struct NttParams` in [`crate::gen::ntt`]. 48 bytes.
///
/// `kscale` is eight bare `u32` and not a `PackedFr` because the WGSL side cannot be an
/// `Fr`: in the uniform address space every array element's stride is rounded up to 16
/// bytes, so `array<u32, 8>` there occupies 128 bytes and would read one limb per 16 with no
/// validation error anywhere. The two structs are the same shape only because both spell the
/// scale out flat.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct NttParams {
    pub log_n: u32,
    pub s0: u32,
    pub scale_mode: u32,
    pub pad0: u32,
    pub kscale: [u32; LIMBS],
}

const _: () = assert!(core::mem::size_of::<NttParams>() == 48);

/// Which twiddle table a transform reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Inverse,
}

/// What the head multiplies into every element as it loads it.
///
/// Both of the non-trivial cases are exact rather than approximate, and both are free
/// because the element is already in a register:
///
/// * [`Self::SizeInv`] is the iNTT's `1/n`. The CPU backend multiplies every *output* by it;
///   the transform is `Fr`-linear, so scaling the input is the same map.
/// * [`Self::CosetPowers`] is stage 2, `x[j] *= shift^j`, applied to the input of the
///   forward transform. Indexed by the natural-order source position, which is what the
///   kernel's `PTAB[src_i]` is careful about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    None,
    SizeInv,
    CosetPowers,
}

/// Where the last batch of a transform writes.
#[derive(Clone, Copy)]
pub enum Epilogue<'a> {
    /// The transform's own buffer.
    Plain,
    /// Stage 4 fused in: write `H = a*b - x` to `h_mont` and `from_mont(H)` to `h_std`,
    /// and do not write the transform's own buffer at all.
    ///
    /// Legal only on the last transform of the three, because `a` and `b` must already hold
    /// their coset evaluations by then. [`crate::stages`] owns that ordering; this module
    /// only provides the kernel and checks it against a host oracle.
    ///
    /// Not the path the prover takes. The standalone `crate::pointwise::HJoin` measured 5% to
    /// 9% faster on every artifact; see [`crate::stages::Stage4`].
    Join {
        a: &'a wgpu::Buffer,
        b: &'a wgpu::Buffer,
        h_mont: &'a wgpu::Buffer,
        h_std: &'a wgpu::Buffer,
    },
}

// ---------------------------------------------------------------------------
// Per-key tables
// ---------------------------------------------------------------------------

/// The three witness-independent tables, on the device.
pub struct NttTables {
    domain: Domain,
    coset_shift: Fr,
    tw_fwd: wgpu::Buffer,
    tw_inv: wgpu::Buffer,
    coset_pows: wgpu::Buffer,
    bytes: u64,
}

impl NttTables {
    /// Builds and uploads for a domain of exactly `domain_size` points.
    ///
    /// Rejects a `domain_size` that is not a power of two rather than rounding up, because
    /// the CSR rows are indexed by evaluation point and a domain that grew would silently
    /// shorten every gather.
    pub fn new(backend: &WgpuBackend, domain_size: usize) -> Result<Self, ProveError> {
        let domain = Domain::new(domain_size).map_err(|e| bad(e.to_string()))?;
        if domain.size != domain_size {
            return Err(bad(format!(
                "domain size {domain_size} is not a power of two"
            )));
        }

        // The same derivation as the CPU and Metal backends, and it has to stay the same:
        // the coset is fixed by the zkey's section 9 bases, so any other shift pairs the
        // evaluations against the wrong Lagrange polynomials.
        let coset_shift = Domain::new(
            domain
                .size
                .checked_mul(2)
                .ok_or_else(|| bad("domain size overflows"))?,
        )
        .map_err(|e| bad(format!("no 2n-th root of unity: {e}")))?
        .group_gen;

        // A running product, one multiply per entry, rather than a square-and-multiply
        // ladder per index. At 2^18 that is 262,144 multiplies against about 4.7 million.
        let mut pows = Vec::with_capacity(domain.size);
        let mut acc = Fr::ONE;
        for _ in 0..domain.size {
            pows.push(acc);
            acc *= coset_shift;
        }

        // Padded to one element when the domain is a single point, where `twiddles()` is
        // empty. A bare `array<Fr>` binding's minimum size is one 32-byte element, so a
        // 4-byte buffer would be a bind group validation failure rather than a harmless
        // empty table. The kernel never reads it: a 2^0 domain has no butterflies.
        let pad = |mut v: Vec<u32>| {
            if v.is_empty() {
                v = vec![0u32; LIMBS];
            }
            v
        };
        let tw_fwd = storage_u32(
            backend,
            "g16 ntt tw_fwd",
            &pad(fr_words(&domain.twiddles())),
        )?;
        let tw_inv = storage_u32(
            backend,
            "g16 ntt tw_inv",
            &pad(fr_words(&domain.twiddles_inv())),
        )?;
        let coset_pows = storage_u32(backend, "g16 ntt coset_pows", &pad(fr_words(&pows)))?;
        let bytes = tw_fwd.size() + tw_inv.size() + coset_pows.size();

        Ok(Self {
            domain,
            coset_shift,
            tw_fwd,
            tw_inv,
            coset_pows,
            bytes,
        })
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// snarkjs' `inc`, a primitive `2n`-th root of unity. Not `Domain::coset_gen`.
    pub fn coset_shift(&self) -> Fr {
        self.coset_shift
    }

    /// Device bytes the three tables occupy.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn twiddles(&self, dir: Direction) -> &wgpu::Buffer {
        match dir {
            Direction::Forward => &self.tw_fwd,
            Direction::Inverse => &self.tw_inv,
        }
    }
}

// ---------------------------------------------------------------------------
// The pipelines
// ---------------------------------------------------------------------------

/// One of the six transforms, as the caller describes it.
///
/// A struct rather than five more arguments because `dir`, `scale`, `src` and `dst` are all
/// easy to transpose at a call site and none of them would fail loudly if they were: an iNTT
/// run with the forward twiddles is a wrong answer, not an error.
#[derive(Clone, Copy)]
pub struct Transform<'a> {
    /// Which twiddle table.
    pub dir: Direction,
    /// What the head multiplies in as it loads. Only the first batch applies it.
    pub scale: Scale,
    /// The head's input, read through the bit-reverse.
    ///
    /// Must not be `dst`. The head reads `SRC[reverse(i)]` and writes `DST[i]`, and the
    /// reversed index of one workgroup's slice lands in another's, so an in-place head would
    /// read values a different workgroup had already overwritten. Nothing here can check it:
    /// wgpu offers no buffer identity comparison and WebGPU is happy to bind one buffer
    /// twice. `g16-metal`'s `stages.rs` keeps a separate `t` buffer for exactly this reason,
    /// and so must U7.
    pub src: &'a wgpu::Buffer,
    /// The head's output, and every later batch's in-place buffer.
    pub dst: &'a wgpu::Buffer,
    /// What the last batch does instead of writing `dst`.
    pub epilogue: Epilogue<'a>,
}

/// One transform's bind groups and parameter offsets, ready to encode.
pub struct Planned {
    head: wgpu::BindGroup,
    /// `None` when the transform is a single batch, so the head is also the last.
    tail_plain: Option<wgpu::BindGroup>,
    /// `None` unless the epilogue is a join and there is more than one batch.
    tail_join: Option<wgpu::BindGroup>,
    head_mode: Mode,
    join: bool,
    offsets: Vec<u32>,
}

impl Planned {
    /// Dispatches this transform will encode, which is the batch count.
    pub fn dispatches(&self) -> usize {
        self.offsets.len()
    }
}

/// The stages 1 to 3 module, its four pipeline layouts, and one pipeline per
/// `(mode, tile size)`.
///
/// Built for a specific `log_n`, because the tile sizes are compile-time constants. Two
/// distinct tile sizes at most, since [`split_passes`] only ever emits `base` and `base + 1`.
pub struct Ntt {
    kernels: Kernels,
    layouts: [wgpu::BindGroupLayout; 4],
    _pipeline_layouts: [wgpu::PipelineLayout; 4],
    batches: Vec<Batch>,
    log_n: u32,
    n: u32,
    max_fused: u32,
    workgroup: Option<u32>,
    source_len: usize,
}

impl Ntt {
    /// Compiles for a domain of `2^log_n` points at the measured shape.
    ///
    /// [`Self::preferred_fused`] rather than [`Self::max_fused`], and one workgroup size per
    /// tile rather than a flat one. Both differ from what design §4 specifies and both are
    /// swept in `tests/ntt.rs`; the tables and the argument are on
    /// [`wgsl::PREFERRED_FUSED`] and [`wgsl::workgroup_for`].
    pub fn new(backend: &WgpuBackend, log_n: u32) -> Result<Self, ProveError> {
        Self::with_shape(backend, log_n, Self::preferred_fused(backend), None)
    }

    /// Passes per dispatch this backend actually uses: the smaller of the memory ceiling and
    /// the measured [`wgsl::PREFERRED_FUSED`].
    ///
    /// 8 at the floor, where the ceiling is 9. Filling the workgroup budget is 4.7x slower at
    /// 2^18 than leaving a quarter of it unused, which is the opposite of what "as few
    /// dispatches as the budget allows" predicts.
    pub fn preferred_fused(backend: &WgpuBackend) -> u32 {
        Self::max_fused(backend).min(wgsl::PREFERRED_FUSED)
    }

    /// Largest `k` this device's workgroup storage allows, capped at
    /// [`wgsl::MAX_FUSED_PASSES`].
    ///
    /// 9 at the floor (16384 / 32 = 512 elements), 10 on this M2 Max at `Raised`. Derived
    /// from the granted limit and never from `Limits::default()`, so `Raised` gets the extra
    /// pass and a stricter future device gets one fewer instead of a shader that will not
    /// build.
    pub fn max_fused(backend: &WgpuBackend) -> u32 {
        let elems = backend.granted_limits().max_compute_workgroup_storage_size
            / crate::gen::field::Variant::default().bytes_per_elem() as u32;
        if elems == 0 {
            return 0;
        }
        (u32::BITS - 1 - elems.leading_zeros()).min(wgsl::MAX_FUSED_PASSES)
    }

    /// Same, with the pass cap forced and the workgroup size optionally forced flat.
    ///
    /// Public because `tests/ntt.rs` needs both. A forced `max_fused` is the only way to
    /// reach three and four batch splits at a domain that would otherwise fit in two, and a
    /// forced workgroup size is how [`wgsl::workgroup_for`] became a measured rule instead of
    /// a number copied from Metal. `None` is the per-tile rule, which is what ships.
    pub fn with_shape(
        backend: &WgpuBackend,
        log_n: u32,
        max_fused: u32,
        workgroup: Option<u32>,
    ) -> Result<Self, ProveError> {
        let limits = backend.granted_limits();
        if let Some(w) = workgroup {
            if w == 0 || w > limits.max_compute_invocations_per_workgroup {
                return Err(bad(format!(
                    "workgroup size {w} is outside 1..={}",
                    limits.max_compute_invocations_per_workgroup
                )));
            }
        }
        if log_n > 32 {
            return Err(bad(format!(
                "log_n {log_n} is not a domain any Fr supports"
            )));
        }
        let ceiling = Self::max_fused(backend);
        if max_fused == 0 || max_fused > ceiling {
            return Err(bad(format!(
                "max_fused {max_fused} is outside 1..={ceiling}; this device grants {} bytes \
                 of workgroup storage and an Fr is {FR_BYTES}",
                limits.max_compute_workgroup_storage_size
            )));
        }

        let batches = split_passes(log_n, max_fused);
        // The head's `base = wid.x << k` and `low = 0u` are only right for a batch starting
        // at pass 0. `split_passes` always starts there, and this is what lets the generator
        // say so rather than hope.
        if batches[0].s0 != 0 {
            return Err(bad(format!(
                "split_passes started the first batch at pass {}, and the head kernel assumes 0",
                batches[0].s0
            )));
        }
        let n = 1u32
            .checked_shl(log_n)
            .ok_or_else(|| bad(format!("2^{log_n} overflows u32")))?;
        // The largest batch has the fewest groups, so the smallest k is what can overrun
        // maxComputeWorkgroupsPerDimension. 65535 never rises in any browser at any tier.
        let max_groups = limits.max_compute_workgroups_per_dimension;
        for b in &batches {
            let groups = (n >> b.k).max(1);
            if groups > max_groups {
                return Err(bad(format!(
                    "batch (s0 {}, k {}) of a 2^{log_n} domain needs {groups} workgroups, over \
                     the {max_groups} limit",
                    b.s0, b.k
                )));
            }
        }

        let mut ks: Vec<u32> = batches.iter().map(|b| b.k).collect();
        ks.sort_unstable();
        ks.dedup();

        let device = backend.device();
        let layouts: [wgpu::BindGroupLayout; 4] = std::array::from_fn(|i| {
            let m = Mode::ALL[i];
            let entries: Vec<wgpu::BindGroupLayoutEntry> =
                m.bindings().into_iter().map(layout_entry).collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!(
                    "g16 ntt {}{}",
                    if m.head { "head " } else { "tail " },
                    if m.join { "join" } else { "plain" }
                )),
                entries: &entries,
            })
        });
        let pipeline_layouts: [wgpu::PipelineLayout; 4] = std::array::from_fn(|i| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("g16 ntt"),
                bind_group_layouts: &[Some(&layouts[i])],
                immediate_size: 0,
            })
        });

        let source = wgsl::ntt_module_at(Variant::default(), &ks, workgroup);
        let source_len = source.len();
        let names: Vec<(String, usize)> = ks
            .iter()
            .flat_map(|&k| (0..4).map(move |i| (Mode::ALL[i].entry(k), i)))
            .collect();
        let entries: Vec<(&str, &wgpu::PipelineLayout)> = names
            .iter()
            .map(|(name, i)| (name.as_str(), &pipeline_layouts[*i]))
            .collect();
        let kernels = Kernels::build_with_layouts(backend, "ntt", &source, &entries)?;

        Ok(Self {
            kernels,
            layouts,
            _pipeline_layouts: pipeline_layouts,
            batches,
            log_n,
            n,
            max_fused,
            workgroup,
            source_len,
        })
    }

    /// The `(s0, k)` split this was compiled for.
    pub fn batches(&self) -> &[Batch] {
        &self.batches
    }

    /// Dispatches one transform costs. Six of these per proof.
    pub fn dispatches(&self) -> usize {
        self.batches.len()
    }

    pub fn log_n(&self) -> u32 {
        self.log_n
    }

    pub fn domain_size(&self) -> u32 {
        self.n
    }

    pub fn max_fused_passes(&self) -> u32 {
        self.max_fused
    }

    /// The forced flat workgroup size, or `None` when each tile got its own.
    pub fn forced_workgroup(&self) -> Option<u32> {
        self.workgroup
    }

    /// Threads per workgroup the entry points at tile size `k` were generated at.
    pub fn workgroup_for(&self, k: u32) -> u32 {
        self.workgroup.unwrap_or_else(|| wgsl::workgroup_for(k))
    }

    pub fn source_len(&self) -> usize {
        self.source_len
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    /// The bind group layout entries for one mode, as data, so a test can count the storage
    /// buffers the pipeline layout actually declares.
    ///
    /// This is the same list [`Self::with_shape`] builds the layout from. wgpu exposes
    /// nothing readable off a constructed `BindGroupLayout`, and duplicating the list in a
    /// test would only prove the duplicate was right.
    pub fn bind_group_layout_entries(m: Mode) -> Vec<wgpu::BindGroupLayoutEntry> {
        m.bindings().into_iter().map(layout_entry).collect()
    }

    /// Builds the bind groups and pushes the parameter blocks for one whole transform.
    ///
    /// `src` and `dst` must be different buffers. The head reads `SRC[reverse(i)]` and writes
    /// `DST[i]`, and the reversed index of one workgroup's slice lands in another's, so an
    /// in-place head would read values a different workgroup had already overwritten. Nothing
    /// here can check it: wgpu offers no buffer identity comparison, and WebGPU is happy to
    /// bind one buffer twice. Metal's `stages.rs` keeps a separate `t` buffer for exactly
    /// this reason and so must U7.
    ///
    /// Separate from [`Self::encode`] because design §3 writes every parameter block for the
    /// whole proof in one `write_buffer` before encoding starts, so the pushes have to happen
    /// before the compute pass exists.
    pub fn plan(
        &self,
        backend: &WgpuBackend,
        tables: &NttTables,
        ring: &mut ParamRing,
        tf: &Transform<'_>,
    ) -> Result<Planned, ProveError> {
        if tables.domain.log_size != self.log_n {
            return Err(bad(format!(
                "the tables are a 2^{} domain and these pipelines are a 2^{} one",
                tables.domain.log_size, self.log_n
            )));
        }
        let join = matches!(tf.epilogue, Epilogue::Join { .. });
        let single = self.batches.len() == 1;
        let head_mode = Mode {
            head: true,
            join: join && single,
        };

        let head = self.bind(backend, tables, ring, head_mode, tf, tf.epilogue)?;
        let tail_plain = (!single)
            .then(|| self.bind(backend, tables, ring, Mode::TAIL_PLAIN, tf, Epilogue::Plain))
            .transpose()?;
        let tail_join = (!single && join)
            .then(|| self.bind(backend, tables, ring, Mode::TAIL_JOIN, tf, tf.epilogue))
            .transpose()?;

        let kscale = match tf.scale {
            Scale::SizeInv => PackedFr::from_fr(&tables.domain.size_inv).v,
            _ => [0u32; LIMBS],
        };
        let scale_mode = match tf.scale {
            Scale::None => wgsl::SCALE_NONE,
            Scale::SizeInv => wgsl::SCALE_CONST,
            Scale::CosetPowers => wgsl::SCALE_TABLE,
        };
        let mut offsets = Vec::with_capacity(self.batches.len());
        for (bi, b) in self.batches.iter().enumerate() {
            offsets.push(ring.push(&NttParams {
                log_n: self.log_n,
                s0: b.s0,
                // The scale rides in on the load, so only the first batch applies it.
                //
                // Not load-bearing, and saying so because a mutation test proved it: the
                // tail entry points never declare `scale_mode` at all, so setting it on a
                // tail block changes nothing and no test can see the difference. It is here
                // so the parameter ring reads honestly under a debugger, and so a tail that
                // ever grows a load scale cannot silently inherit the head's.
                scale_mode: if bi == 0 {
                    scale_mode
                } else {
                    wgsl::SCALE_NONE
                },
                pad0: 0,
                kscale,
            })?);
        }

        Ok(Planned {
            head,
            tail_plain,
            tail_join,
            head_mode,
            join,
            offsets,
        })
    }

    /// Records one transform's batches, in order.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, t: &Planned) -> Result<(), ProveError> {
        if t.offsets.len() != self.batches.len() {
            return Err(bad(format!(
                "the transform was planned for {} batches, these pipelines have {}",
                t.offsets.len(),
                self.batches.len()
            )));
        }
        let last = self.batches.len() - 1;
        for (bi, b) in self.batches.iter().enumerate() {
            let (mode, bind) = if bi == 0 {
                (t.head_mode, &t.head)
            } else if bi == last && t.join {
                (
                    Mode::TAIL_JOIN,
                    t.tail_join.as_ref().ok_or_else(|| {
                        bad("a joined transform was planned without its join tail bind group")
                    })?,
                )
            } else {
                (
                    Mode::TAIL_PLAIN,
                    t.tail_plain.as_ref().ok_or_else(|| {
                        bad("a multi-batch transform was planned without its tail bind group")
                    })?,
                )
            };
            pass.set_pipeline(self.kernels.get(&mode.entry(b.k))?);
            pass.set_bind_group(0, bind, &[t.offsets[bi]]);
            pass.dispatch_workgroups((self.n >> b.k).max(1), 1, 1);
        }
        Ok(())
    }

    /// One mode's bind group.
    ///
    /// `epilogue` is taken separately from `tf.epilogue` because a multi-batch joined
    /// transform still needs a *plain* tail bind group for its middle batches, and that one
    /// must not carry the join buffers.
    fn bind(
        &self,
        backend: &WgpuBackend,
        tables: &NttTables,
        ring: &ParamRing,
        m: Mode,
        tf: &Transform<'_>,
        epilogue: Epilogue<'_>,
    ) -> Result<wgpu::BindGroup, ProveError> {
        let dst = tf.dst;
        let src = m.head.then_some(tf.src);
        // Checked here because WebGPU derives a runtime-sized array's length from the
        // binding size and then silently drops out-of-range writes and returns zero for
        // out-of-range reads. A short `dst` would produce a transform that is correct on its
        // first half and zero on the rest, with nothing in any log.
        let want = self.n as u64 * FR_BYTES;
        let mut need: Vec<(&str, &wgpu::Buffer, u64)> = vec![("dst", dst, want)];
        if let Some(s) = src {
            need.push(("src", s, want));
        }
        if let Epilogue::Join {
            a,
            b,
            h_mont,
            h_std,
        } = epilogue
        {
            if m.join {
                need.extend([
                    ("join_a", a, want),
                    ("join_b", b, want),
                    ("h_mont", h_mont, want),
                    ("h_std", h_std, want),
                ]);
            }
        }
        for (name, buf, bytes) in need {
            if buf.size() < bytes {
                return Err(bad(format!(
                    "{name} is {} bytes, a 2^{} domain needs {bytes}",
                    buf.size(),
                    self.log_n
                )));
            }
        }

        let tw = tables.twiddles(tf.dir);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = m
            .bindings()
            .into_iter()
            .map(|binding| {
                let resource = match binding {
                    wgsl::BIND_PARAMS => wgpu::BindingResource::Buffer(ring.binding()),
                    wgsl::BIND_SRC => src
                        .expect("Mode::bindings lists SRC only for the head")
                        .as_entire_binding(),
                    wgsl::BIND_DST => dst.as_entire_binding(),
                    wgsl::BIND_TW => tw.as_entire_binding(),
                    wgsl::BIND_PTAB => tables.coset_pows.as_entire_binding(),
                    other => match epilogue {
                        Epilogue::Join {
                            a,
                            b,
                            h_mont,
                            h_std,
                        } => match other {
                            wgsl::BIND_JOIN_A => a.as_entire_binding(),
                            wgsl::BIND_JOIN_B => b.as_entire_binding(),
                            wgsl::BIND_H_MONT => h_mont.as_entire_binding(),
                            wgsl::BIND_H_STD => h_std.as_entire_binding(),
                            _ => unreachable!("Mode::bindings emits no other binding"),
                        },
                        Epilogue::Plain => {
                            unreachable!("Mode::bindings lists the join buffers only when joining")
                        }
                    },
                };
                wgpu::BindGroupEntry { binding, resource }
            })
            .collect();

        Ok(backend
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("g16 ntt"),
                layout: &self.layouts[Mode::ALL.iter().position(|x| *x == m).unwrap()],
                entries: &entries,
            }))
    }
}

/// The layout entry for one binding number, matching what [`crate::gen::ntt`] declares.
///
/// A `read_only: false` entry and a `var<storage, read>` declaration are not interchangeable
/// in WebGPU: the binding types must match exactly, so this table and the generated shader
/// have to agree binding for binding. They are next to each other for that reason.
fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    if binding == wgsl::BIND_PARAMS {
        return ParamRing::layout_entry(binding);
    }
    let read_only = !matches!(
        binding,
        wgsl::BIND_DST | wgsl::BIND_H_MONT | wgsl::BIND_H_STD
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_batches_partition_the_transform() {
        for max_fused in 1..=10u32 {
            for log_n in 0..=20u32 {
                let bs = split_passes(log_n, max_fused);
                assert!(!bs.is_empty(), "log_n {log_n} max {max_fused}: no batches");
                let mut at = 0;
                for b in &bs {
                    assert_eq!(b.s0, at, "log_n {log_n} max {max_fused}: gap or overlap");
                    assert!(b.k <= max_fused, "log_n {log_n} max {max_fused}: k too big");
                    at += b.k;
                }
                assert_eq!(at, log_n, "log_n {log_n} max {max_fused}: passes lost");
                assert_eq!(bs.len() as u32, log_n.div_ceil(max_fused).max(1));
                // Even, not greedy: no two batches differ by more than one pass.
                let lo = bs.iter().map(|b| b.k).min().unwrap();
                let hi = bs.iter().map(|b| b.k).max().unwrap();
                assert!(hi - lo <= 1, "log_n {log_n} max {max_fused}: uneven split");
            }
        }
    }

    #[test]
    fn a_2_18_domain_splits_nine_and_nine_at_the_browser_floor() {
        // 16384 bytes of workgroup storage / 32 bytes per Fr = 512 elements = 2^9.
        assert_eq!(
            split_passes(18, 9),
            vec![Batch { s0: 0, k: 9 }, Batch { s0: 9, k: 9 }]
        );
    }
}
