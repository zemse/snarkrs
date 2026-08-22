//! The one way results leave the GPU, written once so that it is correct on native and in a
//! browser.
//!
//! # Why there is exactly one of these and why it is async
//!
//! `Device::poll` is documented as a no-op on WebGPU ("Devices are automatically polled",
//! `wgpu-30.0.1/src/backend/webgpu.rs:2698`) and `PollType::Wait` has no effect there either.
//! The only thing that ever fires a `map_async` callback in a browser is a yield to the JS
//! event loop, so any code that blocks a thread waiting for the GPU deadlocks the tab. That
//! rules out a synchronous readback, and `pollster::block_on` compiles for wasm32 and then
//! hangs, which is worse than not compiling.
//!
//! # What the design said, what the API actually does, and the correction
//!
//! Design §8 U4 says to build this on `CommandEncoder::map_buffer_on_submit` "plus an async
//! channel", and adds that `device.poll` "cannot be the mechanism". The first half is right
//! and the API exists (wgpu 30.0.1, `src/api/command_buffer_actions.rs:103`, from
//! [wgpu#8125](https://github.com/gfx-rs/wgpu/pull/8125)). The second half is only right for
//! the web.
//!
//! Read what `map_buffer_on_submit` does: it pushes a `DeferredBufferMapping` onto the
//! encoder, and `Queue::submit` drains those and calls plain `Buffer::map_async` on each
//! (`src/api/command_buffer_actions.rs:31-42`). It is `map_async`, scheduled for you at the
//! right moment. Its own doc comment then says, in full: "For the callback to run, either
//! `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)` must be called elsewhere
//! in the runtime." On native, nothing after the submit drives the queue on its own, so the
//! `poll` is still required and this function would hang without it.
//!
//! So `map_buffer_on_submit` is worth using, but for the reason the wgpu PR gives rather than
//! the one the design gives: it removes the submit-then-map ordering hazard, where a
//! `map_async` issued before the submit is a validation error and one issued after has to be
//! sequenced by hand. The `poll` stays, unconditionally, with no `cfg`: it is what makes the
//! callback fire on native and it is a documented no-op on the web, which is exactly the
//! shape "one function correct on both targets" needs.
//!
//! This was tested rather than reasoned about. Deleting the `poll` below and running
//! `tests/device.rs::one_uniform_ring_feeds_three_dispatches_from_three_dynamic_offsets`
//! hangs the test past 60 s instead of finishing in 40 ms. Anyone who reads the design's
//! "device.poll cannot be the mechanism" and removes it will reproduce that.
//!
//! # Cost, and the size ceiling that keeps it irrelevant
//!
//! One `mapAsync` round trip is about 0.3 ms. Design §3 caps the whole per-proof readback at
//! 64 KiB (20 window sums plus up to 64 `ones_groups` partials per MSM, four G1 and one G2),
//! so this is paid once per proof and nothing is chunked through it.
//!
//! That ceiling is load-bearing on wasm. `get_mapped_range` there copies the entire mapped
//! `ArrayBuffer` into linear memory, measured at
//! **33.8 ms for 64 MiB** against 2.6 ms for the GPU copy that produced it. At 64 KiB that is
//! three orders of magnitude clear. If a readback ever grows past a megabyte, this function is
//! the wrong tool and `as_uint8array()` is the right one.

use g16_core::ProveError;

use crate::device::{bad, WgpuBackend};

/// A staging buffer plus the copy-and-map dance around it.
///
/// Allocated once and reused: the buffer is `MAP_READ | COPY_DST`, which is the one usage
/// combination a mappable buffer is allowed, and it can never also be a storage buffer. That
/// is a WebGPU rule and not a wgpu limitation, so the copy from the storage buffer into this
/// one is unavoidable rather than a missed optimisation.
pub struct Readback {
    staging: wgpu::Buffer,
    bytes: u64,
}

impl Readback {
    /// Allocates a staging buffer of exactly `bytes`.
    ///
    /// `bytes` must be a multiple of 4, which WebGPU requires of every buffer-to-buffer copy
    /// size and of every mapped range length.
    pub fn new(backend: &WgpuBackend, label: &str, bytes: u64) -> Result<Self, ProveError> {
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(bad(format!(
                "readback {label:?} of {bytes} bytes: a mapped range must be a nonzero \
                 multiple of 4"
            )));
        }
        let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self { staging, bytes })
    }

    /// Queues the device-to-staging copy into an encoder the caller is still building.
    ///
    /// Separate from [`Self::submit_and_read`] so the copy lands in the same command encoder
    /// as the dispatches that produced the data. Design §3 puts the whole proof in two
    /// submits, and a readback that opened its own encoder would make it three.
    pub fn copy_from(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        src_offset: u64,
        bytes: u64,
    ) -> Result<(), ProveError> {
        if bytes > self.bytes {
            return Err(bad(format!(
                "readback buffer holds {} bytes, asked to copy {bytes}",
                self.bytes
            )));
        }
        enc.copy_buffer_to_buffer(src, src_offset, &self.staging, 0, bytes);
        Ok(())
    }

    /// Finishes `enc`, submits it, waits for the GPU, and returns the first `bytes` of the
    /// staging buffer.
    ///
    /// Takes the encoder by value because `map_buffer_on_submit` has to be registered before
    /// `finish()`, which is the whole point of using it over a bare `map_async`.
    ///
    /// Returns an owned `Vec` rather than the mapped view. A `BufferView` borrows the buffer
    /// and would have to stay alive across the caller's parsing, which forbids reusing this
    /// `Readback` and, on wasm, holds a copy of the `ArrayBuffer` anyway. At the 64 KiB
    /// ceiling in the module docs the extra copy is not measurable.
    pub async fn submit_and_read(
        &self,
        backend: &WgpuBackend,
        enc: wgpu::CommandEncoder,
        bytes: u64,
    ) -> Result<Vec<u8>, ProveError> {
        if bytes > self.bytes {
            return Err(bad(format!(
                "readback buffer holds {} bytes, asked to read {bytes}",
                self.bytes
            )));
        }
        // `flume` and not `std::sync::mpsc`: the receiver has to be awaited, not blocked on,
        // and `mpsc::Receiver` has no async form. This is wgpu's own choice in
        // `examples/features/src/repeated_compute`.
        let (tx, rx) = flume::bounded(1);
        enc.map_buffer_on_submit(&self.staging, wgpu::MapMode::Read, 0..bytes, move |r| {
            let _ = tx.send(r);
        });
        backend.queue().submit([enc.finish()]);

        // Native: this is what runs the callback, and without it the await below never
        // resolves. Web: a documented no-op, and the browser event loop runs the callback
        // while the await is suspended. See the module docs.
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| bad(format!("waiting for the GPU failed: {e}")))?;

        rx.recv_async()
            .await
            .map_err(|_| bad("the map callback was dropped before it fired"))?
            .map_err(|e| bad(format!("mapping the readback buffer failed: {e}")))?;

        let slice = self.staging.slice(0..bytes);
        let view = slice
            .get_mapped_range()
            .map_err(|e| bad(format!("the readback buffer mapped but did not read: {e}")))?;
        let out = view.to_vec();
        // Both in this order and both required: the view borrows the mapping, and a buffer
        // left mapped cannot be used by any later command.
        drop(view);
        self.staging.unmap();

        if let Some(e) = backend.take_error() {
            return Err(bad(format!("device error during readback: {e}")));
        }
        Ok(out)
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.bytes
    }
}
