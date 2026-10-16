//! The Metal handles this crate holds, and the calls on them every module makes.
//!
//! objc2-metal hands every object out as a `Retained<ProtocolObject<dyn MTL...>>`; the
//! aliases below are those types under the names the rest of the crate uses. The
//! helpers wrap the few calls objc2-metal marks `unsafe` and every module needs, so the
//! safety argument is made once, here.

use core::ffi::c_void;
use core::ops::Deref;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLDispatchType, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
};

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type DeviceRef = ProtocolObject<dyn MTLDevice>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type CommandQueueRef = ProtocolObject<dyn MTLCommandQueue>;
pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
pub type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;
pub type ComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
pub type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type Function = Retained<ProtocolObject<dyn MTLFunction>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// An `MTLBuffer`. Cloning it is a retain.
///
/// objc2-metal leaves `MTLBuffer` neither `Send` nor `Sync`, because nothing stops two
/// threads writing its contents at once. Apple documents the object itself as
/// thread-safe, and every access to the contents here is ordered by a command buffer
/// wait or a pool lock, which is the contract the buffers' `unsafe` readers already
/// state.
#[derive(Clone)]
pub struct Buffer(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: see the type's documentation.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl Deref for Buffer {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The system default device, `None` on a machine without Metal.
pub fn system_default() -> Option<Device> {
    objc2_metal::MTLCreateSystemDefaultDevice()
}

/// A command queue, `None` if the device refused one.
pub fn command_queue(device: &DeviceRef) -> Option<CommandQueue> {
    device.newCommandQueue()
}

/// `bytes` of storage with `options`, `None` when the allocation failed.
pub fn buffer(device: &DeviceRef, bytes: usize, options: MTLResourceOptions) -> Option<Buffer> {
    device
        .newBufferWithLength_options(bytes, options)
        .map(Buffer)
}

/// `bytes` of storage with `options`, holding a copy of what `src` points at, `None` when
/// the allocation failed.
///
/// # Safety
///
/// `src` must be readable for `bytes`.
pub unsafe fn buffer_with_data(
    device: &DeviceRef,
    src: NonNull<c_void>,
    bytes: usize,
    options: MTLResourceOptions,
) -> Option<Buffer> {
    device
        .newBufferWithBytes_length_options(src, bytes, options)
        .map(Buffer)
}

/// Compiles MSL source with the default options. The error is the compiler's diagnostic.
pub fn library(device: &DeviceRef, source: &str) -> Result<Library, String> {
    device
        .newLibraryWithSource_options_error(
            &NSString::from_str(source),
            Some(&MTLCompileOptions::new()),
        )
        .map_err(|e| e.localizedDescription().to_string())
}

/// The kernel `name` in `library`.
pub fn function(library: &ProtocolObject<dyn MTLLibrary>, name: &str) -> Result<Function, String> {
    library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("Function '{name}' does not exist"))
}

/// A compute pipeline for `function`.
pub fn pipeline(
    device: &DeviceRef,
    function: &ProtocolObject<dyn MTLFunction>,
) -> Result<ComputePipelineState, String> {
    device
        .newComputePipelineStateWithFunction_error(function)
        .map_err(|e| e.localizedDescription().to_string())
}

/// A serial compute encoder on `cb`.
pub fn encoder(cb: &CommandBufferRef) -> ComputeCommandEncoder {
    cb.computeCommandEncoder()
        .expect("a command buffer with no open encoder gives a compute encoder")
}

/// A compute encoder of `dispatch` type on `cb`.
pub fn encoder_with(cb: &CommandBufferRef, dispatch: MTLDispatchType) -> ComputeCommandEncoder {
    cb.computeCommandEncoderWithDispatchType(dispatch)
        .expect("a command buffer with no open encoder gives a compute encoder")
}

/// Names `enc` in GPU captures and in a failed buffer's per-encoder states.
pub fn set_label(enc: &ComputeCommandEncoderRef, label: &str) {
    enc.setLabel(Some(&NSString::from_str(label)));
}

/// Binds `buf` from byte `offset` at buffer `index`.
pub fn set_buffer(enc: &ComputeCommandEncoderRef, index: usize, buf: &Buffer, offset: usize) {
    // SAFETY: every caller binds an index its kernel declares, at an offset inside the
    // buffer, and keeps `buf` alive until the command buffer it is encoded into has
    // completed.
    unsafe { enc.setBuffer_offset_atIndex(Some(buf), offset, index) }
}

/// Copies `len` bytes from `bytes` into the argument table at buffer `index`.
///
/// # Safety
///
/// `bytes` must be readable for `len`.
pub unsafe fn set_bytes(
    enc: &ComputeCommandEncoderRef,
    index: usize,
    len: usize,
    bytes: *const c_void,
) {
    let bytes = NonNull::new(bytes.cast_mut()).expect("set_bytes from a null pointer");
    enc.setBytes_length_atIndex(bytes, len, index)
}

/// `MTLCommandBufferStatus` by its name, for an error message.
pub fn status_name(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "?",
    }
}

/// A one-dimensional grid or threadgroup of `width`.
pub fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}
