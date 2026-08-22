//! The uniform parameter ring: how a per-dispatch scalar reaches a WGSL kernel.
//!
//! # Why this exists at all
//!
//! Metal has `setBytes`, Vulkan has push constants, CUDA has kernel arguments. WebGPU has
//! none of the three, and `wgpu`'s `immediate_size` (its push-constant equivalent) is a
//! native extension the browser does not implement. The only portable channel for "this
//! dispatch works on rows 8192 to 12287 with stride 20" is a uniform buffer.
//!
//! Creating one small uniform buffer per dispatch would be 51 buffer creations and 51
//! `writeBuffer` calls per proof (design §3: 13 dispatches for `compute_h`, 38 for the
//! MSMs). Instead this is **one** buffer holding every block, written in **one**
//! `writeBuffer` before encoding starts, with each dispatch selecting its own block through
//! the dynamic offset argument of `setBindGroup`.
//!
//! # The 256 that is not negotiable
//!
//! A dynamic offset must be a multiple of `minUniformBufferOffsetAlignment`, which is 256 in
//! the WebGPU defaults and never improves at
//! any tier in any browser. So a 16-byte parameter block still costs a 256-byte slot. That
//! is not worth optimising: 51 slots is 13,056 bytes, against hundreds of megabytes of point
//! data. The 64 KiB `maxUniformBufferBindingSize` floor caps a single slot, not the ring,
//! because each dispatch binds one 256-byte window rather than the whole buffer.
//!
//! The stride is read from the granted limits rather than hard-coded, because an adapter is
//! allowed to report a *smaller* alignment and `Raised` then gets denser packing for free.
//! It is never read from `Limits::default()`, which would silently over-align on such a
//! device and, worse, silently under-align if a future adapter went the other way.
//!
//! # Why uniform and not storage
//!
//! Two reasons, both hard limits rather than preferences. A storage buffer would consume one
//! of the 8 `maxStorageBuffersPerShaderStage` slots that design §4's kernels are already
//! tight against, while `maxUniformBuffersPerShaderStage` is 12 and unused. And
//! `maxDynamicStorageBuffersPerPipelineLayout` is 4 against
//! `maxDynamicUniformBuffersPerPipelineLayout`'s 8.

use bytemuck::Pod;
use g16_core::ProveError;

use crate::device::{bad, WgpuBackend};

/// One uniform buffer of `slots` parameter blocks, addressed by dynamic offset.
///
/// Host-side staging is a plain `Vec<u8>` that is filled during encoding and uploaded once.
/// It is kept across [`Self::reset`] so a per-proof ring allocates nothing after the first
/// proof, which matters because `writeBuffer` on this platform runs at 4 to 6 GB/s and the
/// allocation would be a larger share of the cost than the copy.
pub struct ParamRing {
    buffer: wgpu::Buffer,
    staging: Vec<u8>,
    stride: u32,
    slots: u32,
    used: u32,
}

impl ParamRing {
    /// Allocates a ring with room for `slots` blocks.
    ///
    /// The ring is checked against `maxBufferSize` and not against
    /// `maxUniformBufferBindingSize`. Those bound different things: the binding this hands
    /// out is one `stride` window, so the 64 KiB uniform binding floor caps the *slot*, not
    /// the ring, and a ring of a million slots would be legal. The check is here rather than
    /// left to wgpu because a validation error at bind-group creation names a byte offset,
    /// not the ring that was sized wrong.
    pub fn new(backend: &WgpuBackend, label: &str, slots: u32) -> Result<Self, ProveError> {
        let limits = backend.granted_limits();
        let stride = limits.min_uniform_buffer_offset_alignment;
        let bytes = (stride as u64)
            .checked_mul(slots as u64)
            .ok_or_else(|| bad(format!("parameter ring {label:?}: {slots} slots overflows")))?;
        if u64::from(stride) > limits.max_uniform_buffer_binding_size {
            return Err(bad(format!(
                "parameter ring {label:?}: a {stride} byte slot is over the {} byte uniform \
                 binding limit",
                limits.max_uniform_buffer_binding_size
            )));
        }
        if bytes > limits.max_buffer_size {
            return Err(bad(format!(
                "parameter ring {label:?} wants {bytes} bytes ({slots} slots x {stride}), \
                 over the {} byte buffer limit",
                limits.max_buffer_size
            )));
        }
        let buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            buffer,
            staging: vec![0u8; bytes as usize],
            stride,
            slots,
            used: 0,
        })
    }

    /// Appends one parameter block and returns the dynamic offset that selects it.
    ///
    /// The bound is `Pod` and not `Copy` because this reinterprets `T` as bytes: `Pod` is what
    /// rules out uninitialised padding, which is undefined behaviour to read, and `repr(C)` is
    /// what gives the field order a meaning at all.
    ///
    /// Nothing checks that `T` matches the WGSL `struct` the kernel declares, and nothing can.
    /// A mismatch reads plausible garbage and no validation layer will say a word, so keep the
    /// two next to each other. WGSL's uniform address space is stricter than `repr(C)`: a
    /// struct is aligned to 16 bytes and every array member has a stride that is a multiple of
    /// 16, so `vec3<u32>` occupies 16 bytes and a `u32` following one starts at offset 16, not
    /// 12. Prefer flat `u32` fields with explicit padding, as `tests/device.rs` does.
    pub fn push<T: Pod>(&mut self, block: &T) -> Result<u32, ProveError> {
        let bytes = bytemuck::bytes_of(block);
        if bytes.len() > self.stride as usize {
            return Err(bad(format!(
                "parameter block is {} bytes, over the {} byte slot stride",
                bytes.len(),
                self.stride
            )));
        }
        if self.used >= self.slots {
            return Err(bad(format!(
                "parameter ring is full at {} slots; size it for the dispatch count",
                self.slots
            )));
        }
        let offset = self.used * self.stride;
        self.staging[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes);
        self.used += 1;
        Ok(offset)
    }

    /// Uploads every block pushed since the last [`Self::reset`], in one call.
    ///
    /// Only the used prefix is sent, so a ring sized for the worst case does not pay for the
    /// slots a smaller proof did not touch.
    pub fn flush(&self, backend: &WgpuBackend) {
        let end = (self.used * self.stride) as usize;
        if end == 0 {
            return;
        }
        backend
            .queue()
            .write_buffer(&self.buffer, 0, &self.staging[..end]);
    }

    /// Rewinds to slot zero. The staging allocation and the GPU buffer are kept.
    ///
    /// Stale bytes past the new write cursor are deliberately not zeroed: every dispatch
    /// reads only the slot it was given an offset for, and clearing 13 KB per proof to
    /// defend against a bug that would be a wrong offset rather than stale data is work for
    /// nothing.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// The binding for a bind group entry, one slot wide.
    ///
    /// `offset` is zero and `size` is one stride: with `has_dynamic_offset` the offset comes
    /// from `set_bind_group` at dispatch time, and the size is what bounds the read. WebGPU
    /// validates `dynamic_offset + size <= buffer size`, which is why the ring is sized as
    /// `slots * stride` exactly.
    pub fn binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(self.stride as u64),
        }
    }

    /// The bind group layout entry a kernel using this ring must declare.
    ///
    /// `min_binding_size` is left `None` on purpose. Setting it to the parameter struct's
    /// size would be a stronger check, but the ring is shared by kernels whose parameter
    /// structs differ in size, and one layout has to serve all of them.
    pub fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Bytes per slot: `minUniformBufferOffsetAlignment` on this device, 256 at the Floor.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Slots pushed since the last reset.
    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }
}
