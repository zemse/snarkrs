//! Can a Metal command buffer report `Completed` after the GPU cut one of its compute
//! encoders short?
//!
//! Every command buffer here holds two compute encoders. The first runs a busy kernel of
//! about `--ms` milliseconds (the size macOS kills for impacting interactivity when the
//! display wants the GPU) and then, as its last dispatch, writes this buffer's epoch to
//! word A. The second encoder holds one dispatch that writes the epoch to word B. After
//! `waitUntilCompleted` the buffer's status, its NSError, the per-encoder execution
//! status (`MTLCommandBufferErrorOptionEncoderExecutionStatus`) and both words are read
//! and tallied. A `Completed` with A missing is the case BUG-28 suspects; B present with
//! A missing is the wgpu lane's measurement that an abort ends only the encoder it lands
//! on. Run it beside three proof loops to get kills.
//!
//! Each measured submission starts from two alternating, epoch-dependent residues. The
//! host checks every output against CPU squaring as well as the markers and run counts:
//! a full dispatch with a token can still leave wrong data. This is a diagnostic for this
//! kernel, not a correctness guarantee for proving kernels or their completion tokens.
//!
//! `cargo run --release -p snarkrs-metal --example cb_seal_probe -- [--ms 100] [--n 200]
//! [--no-encoder-status]`

#[cfg(target_os = "macos")]
mod probe {
    use std::time::Instant;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSArray;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor, MTLCommandBufferEncoderInfo,
        MTLCommandBufferEncoderInfoErrorKey, MTLCommandBufferErrorOption, MTLCommandBufferStatus,
        MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLResourceOptions,
    };
    use snarkrs_field::{Field, Fr};
    use snarkrs_metal::kernels::FR_MSL;
    use snarkrs_metal::layout::PackedFr;
    use snarkrs_metal::mtl::{
        self, Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
        ComputePipelineState, Device,
    };

    /// Fresh inputs and their CPU reference, computed before the timed submission.
    fn prepare_data(data: &mut [PackedFr], epoch: u32, iters: u32) -> [PackedFr; 2] {
        let inputs = [
            Fr::from(2 * epoch as u64 + 2),
            Fr::from(2 * epoch as u64 + 3),
        ];
        for (i, word) in data.iter_mut().enumerate() {
            *word = PackedFr::from_fr(&inputs[i % 2]);
        }
        inputs.map(|mut x| {
            for _ in 0..iters {
                x.square_in_place();
            }
            PackedFr::from_fr(&x)
        })
    }

    /// Number of wrong stores and the first one's index, including canonical wrong values.
    fn wrong_data(data: &[PackedFr], expected: &[PackedFr; 2]) -> (usize, Option<usize>) {
        let mut count = 0;
        let mut first = None;
        for (i, word) in data.iter().enumerate() {
            if *word != expected[i % 2] {
                count += 1;
                first.get_or_insert(i);
            }
        }
        (count, first)
    }

    const PROBE_MSL: &str = r#"
// Every thread leaves the epoch in its marker and bumps its run count once, so the host
// can tell a dispatch that ran in full (every marker current, every count up by one)
// from one the GPU cut short (stale markers) or ran twice after a recovery (counts up
// by two).
kernel void busy(device Fr* buf [[buffer(0)]],
                 constant uint& iters [[buffer(1)]],
                 device uint* marker [[buffer(2)]],
                 device uint* count [[buffer(3)]],
                 constant uint& epoch [[buffer(4)]],
                 uint gid [[thread_position_in_grid]]) {
    Fr x = buf[gid];
    for (uint i = 0; i < iters; i++) {
        x = fr_mul(x, x);
    }
    buf[gid] = x;
    marker[gid] = epoch;
    count[gid] += 1u;
}

kernel void seal(device uint* word [[buffer(0)]],
                 constant uint& tok [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        word[0] = tok;
    }
}
"#;

    const BUSY_THREADS: usize = 1 << 18;

    fn error_text(cb: &CommandBufferRef) -> String {
        cb.error()
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_default()
    }

    /// `(label, errorState)` per encoder, from the NSError's user info. The states are
    /// MTLCommandEncoderErrorState: 0 unknown, 1 completed, 2 affected, 3 pending,
    /// 4 faulted.
    fn encoder_status(cb: &CommandBufferRef) -> Vec<(String, i64)> {
        let Some(err) = cb.error() else {
            return Vec::new();
        };
        // SAFETY: a constant NSString Metal exports.
        let key = unsafe { MTLCommandBufferEncoderInfoErrorKey };
        let Some(arr) = err.userInfo().objectForKey(key) else {
            return Vec::new();
        };
        let Ok(arr) = arr.downcast::<NSArray>() else {
            return Vec::new();
        };
        (0..arr.count())
            .map(|i| {
                // SAFETY: the array under this key holds MTLCommandBufferEncoderInfo
                // objects.
                let e: Retained<ProtocolObject<dyn MTLCommandBufferEncoderInfo>> =
                    unsafe { Retained::cast_unchecked(arr.objectAtIndex(i)) };
                (e.label().to_string(), e.errorState().0 as i64)
            })
            .collect()
    }

    fn state_name(s: i64) -> &'static str {
        match s {
            0 => "unknown",
            1 => "completed",
            2 => "affected",
            3 => "pending",
            4 => "faulted",
            _ => "?",
        }
    }

    /// A command buffer with per-encoder error reporting on, or a plain one.
    fn command_buffer(queue: &CommandQueue, encoder_status: bool) -> CommandBuffer {
        if !encoder_status {
            return queue.commandBuffer().unwrap();
        }
        let desc = MTLCommandBufferDescriptor::new();
        desc.setErrorOptions(MTLCommandBufferErrorOption::EncoderExecutionStatus);
        queue.commandBufferWithDescriptor(&desc).unwrap()
    }

    struct Gpu {
        device: Device,
        queue: CommandQueue,
        busy: ComputePipelineState,
        seal: ComputePipelineState,
    }

    impl Gpu {
        fn new() -> Self {
            let device = mtl::system_default().expect("no Metal device");
            let src = format!("{FR_MSL}\n{PROBE_MSL}");
            let library =
                mtl::library(&device, &src).unwrap_or_else(|e| panic!("MSL compile failed:\n{e}"));
            let pso = |name: &str| {
                let f = mtl::function(&library, name).expect(name);
                mtl::pipeline(&device, &f).expect(name)
            };
            let busy = pso("busy");
            let seal = pso("seal");
            let queue = mtl::command_queue(&device).unwrap();
            Self {
                device,
                queue,
                busy,
                seal,
            }
        }

        fn buffer(&self, bytes: usize) -> Buffer {
            mtl::buffer(&self.device, bytes, MTLResourceOptions::StorageModeShared).unwrap()
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_busy(
            &self,
            enc: &ComputeCommandEncoderRef,
            data: &Buffer,
            iters: u32,
            marker: &Buffer,
            count: &Buffer,
            epoch: u32,
        ) {
            enc.setComputePipelineState(&self.busy);
            mtl::set_buffer(enc, 0, data, 0);
            // SAFETY: a `u32` is readable for 4 bytes.
            unsafe { mtl::set_bytes(enc, 1, 4, (&iters as *const u32).cast()) };
            mtl::set_buffer(enc, 2, marker, 0);
            mtl::set_buffer(enc, 3, count, 0);
            // SAFETY: as above.
            unsafe { mtl::set_bytes(enc, 4, 4, (&epoch as *const u32).cast()) };
            let tg = self.busy.maxTotalThreadsPerThreadgroup().min(256);
            enc.dispatchThreads_threadsPerThreadgroup(mtl::size(BUSY_THREADS), mtl::size(tg));
        }

        fn encode_seal(&self, enc: &ComputeCommandEncoderRef, word: &Buffer, tok: u32) {
            enc.setComputePipelineState(&self.seal);
            mtl::set_buffer(enc, 0, word, 0);
            // SAFETY: a `u32` is readable for 4 bytes.
            unsafe { mtl::set_bytes(enc, 1, 4, (&tok as *const u32).cast()) };
            enc.dispatchThreads_threadsPerThreadgroup(mtl::size(1), mtl::size(1));
        }
    }

    fn read_word(b: &Buffer) -> u32 {
        // SAFETY: a 4-byte shared buffer, read after the command buffer completed.
        unsafe { *(b.contents().as_ptr() as *const u32) }
    }

    fn write_word(b: &Buffer, v: u32) {
        // SAFETY: a 4-byte shared buffer with nothing in flight against it.
        unsafe { *(b.contents().as_ptr() as *mut u32) = v }
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut target_ms = 100.0f64;
        let mut n = 200usize;
        let mut with_status = true;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--ms" => {
                    target_ms = args[i + 1].parse().expect("--ms");
                    i += 1;
                }
                "--n" => {
                    n = args[i + 1].parse().expect("--n");
                    i += 1;
                }
                "--no-encoder-status" => with_status = false,
                a => panic!("unknown argument {a}"),
            }
            i += 1;
        }

        let gpu = Gpu::new();
        let data = gpu.buffer(BUSY_THREADS * 32);
        // Any residue in Montgomery form squares to a residue; ones are fine.
        {
            // SAFETY: freshly allocated shared buffer of BUSY_THREADS Fr.
            let words = unsafe {
                std::slice::from_raw_parts_mut(
                    data.contents().as_ptr() as *mut u32,
                    BUSY_THREADS * 8,
                )
            };
            for (i, w) in words.iter_mut().enumerate() {
                *w = if i % 8 == 0 { 1 } else { 0 };
            }
        }
        let word_a = gpu.buffer(4);
        let word_b = gpu.buffer(4);
        let marker = gpu.buffer(BUSY_THREADS * 4);
        let count = gpu.buffer(BUSY_THREADS * 4);
        // SAFETY: both are BUSY_THREADS u32s in shared storage, read and zeroed only
        // between command buffers.
        let (markers, counts) = unsafe {
            (
                std::slice::from_raw_parts_mut(
                    marker.contents().as_ptr() as *mut u32,
                    BUSY_THREADS,
                ),
                std::slice::from_raw_parts_mut(count.contents().as_ptr() as *mut u32, BUSY_THREADS),
            )
        };
        markers.fill(0);
        counts.fill(0);

        // Calibrate `iters` so one busy dispatch takes about `target_ms`. The best of
        // three, since the GPU may already be shared.
        let mut iters = 64u32;
        loop {
            let mut ms = f64::MAX;
            for _ in 0..3 {
                let cb = gpu.queue.commandBuffer().unwrap();
                let enc = mtl::encoder(&cb);
                gpu.encode_busy(&enc, &data, iters, &marker, &count, 0);
                enc.endEncoding();
                let t = Instant::now();
                cb.commit();
                cb.waitUntilCompleted();
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            if ms >= target_ms * 0.8 || iters >= 1 << 24 {
                let scaled = (iters as f64 * target_ms / ms).round().max(1.0) as u32;
                eprintln!("calibrated: {iters} iters = {ms:.1} ms, using {scaled}");
                iters = scaled;
                break;
            }
            iters *= 2;
        }

        let mut tally: std::collections::BTreeMap<String, usize> = Default::default();
        let mut completed_a_missing = 0usize;
        let mut sealed_but_partial = 0usize;
        let mut sealed_but_replayed = 0usize;
        let mut sealed_but_wrong = 0usize;
        let mut error_b_present = 0usize;
        let mut errors = 0usize;
        for i in 0..n {
            let epoch = i as u32 + 1;
            write_word(&word_a, 0);
            write_word(&word_b, 0);
            markers.fill(0);
            counts.fill(0);
            // SAFETY: BUSY_THREADS packed residues in shared storage, with the previous
            // command buffer finished and the next one not yet committed.
            let expected = prepare_data(
                unsafe {
                    std::slice::from_raw_parts_mut(
                        data.contents().cast::<PackedFr>().as_ptr(),
                        BUSY_THREADS,
                    )
                },
                epoch,
                iters,
            );
            let cb = command_buffer(&gpu.queue, with_status);
            let enc = mtl::encoder(&cb);
            mtl::set_label(&enc, "busy+sealA");
            gpu.encode_busy(&enc, &data, iters, &marker, &count, epoch);
            gpu.encode_seal(&enc, &word_a, epoch);
            enc.endEncoding();
            let enc = mtl::encoder(&cb);
            mtl::set_label(&enc, "sealB");
            gpu.encode_seal(&enc, &word_b, epoch);
            enc.endEncoding();
            let t = Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            let wall_ms = t.elapsed().as_secs_f64() * 1e3;
            let status = cb.status();
            let a = read_word(&word_a) == epoch;
            let b = read_word(&word_b) == epoch;
            // What the busy dispatch itself did, thread by thread.
            let stale = markers.iter().filter(|&&m| m != epoch).count();
            let skipped = counts.iter().filter(|&&c| c == 0).count();
            let twice = counts.iter().filter(|&&c| c >= 2).count();
            // SAFETY: the shared data buffer holds BUSY_THREADS packed residues, and
            // the command buffer that wrote them has completed.
            let (wrong, first_wrong) = wrong_data(
                unsafe {
                    std::slice::from_raw_parts(
                        data.contents().cast::<PackedFr>().as_ptr(),
                        BUSY_THREADS,
                    )
                },
                &expected,
            );
            let busy = if stale == 0 && skipped == 0 && twice == 0 {
                "full".to_string()
            } else {
                format!("stale={stale} skipped={skipped} twice={twice}")
            };
            let key = format!(
                "status={} A={} B={} busy={busy} wrong={wrong}",
                mtl::status_name(status),
                if a { "ok" } else { "MISSING" },
                if b { "ok" } else { "MISSING" }
            );
            *tally.entry(key.clone()).or_default() += 1;
            let anomaly = (status == MTLCommandBufferStatus::Completed && !(a && b))
                || (status != MTLCommandBufferStatus::Completed)
                || busy != "full"
                || wrong > 0;
            if status == MTLCommandBufferStatus::Completed && !a {
                completed_a_missing += 1;
            }
            if status == MTLCommandBufferStatus::Completed && a && (stale > 0 || skipped > 0) {
                sealed_but_partial += 1;
            }
            if status == MTLCommandBufferStatus::Completed && a && twice > 0 {
                sealed_but_replayed += 1;
            }
            if status == MTLCommandBufferStatus::Completed && a && wrong > 0 {
                sealed_but_wrong += 1;
            }
            if status != MTLCommandBufferStatus::Completed {
                errors += 1;
                if b {
                    error_b_present += 1;
                }
            }
            if anomaly {
                let enc_states: Vec<String> = encoder_status(&cb)
                    .into_iter()
                    .map(|(l, s)| format!("{l}:{}", state_name(s)))
                    .collect();
                eprintln!(
                    "[{i}] {key} first_wrong={first_wrong:?} wall {wall_ms:.1} ms error \"{}\" encoders [{}]",
                    error_text(&cb),
                    enc_states.join(", ")
                );
            } else if i % 25 == 0 {
                eprintln!("[{i}] ok, wall {wall_ms:.1} ms");
            }
        }
        println!(
            "--- {n} command buffers, busy target {target_ms} ms, encoder status {with_status}"
        );
        for (k, v) in &tally {
            println!("{v:6}  {k}");
        }
        println!(
            "Completed with token A missing: {completed_a_missing}; Completed with token A \
             present but the busy dispatch partial: {sealed_but_partial}, run twice: \
             {sealed_but_replayed}, wrong arithmetic: {sealed_but_wrong}; not Completed: \
             {errors}, of which token B (second \
             encoder) present: {error_b_present}"
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn canonical_wrong_stores_are_detected() {
            let mut data = vec![PackedFr::ZERO; 8];
            let expected = prepare_data(&mut data, 7, 3);
            for (i, word) in data.iter_mut().enumerate() {
                *word = expected[i % 2];
            }
            assert_eq!(wrong_data(&data, &expected), (0, None));
            // Lost, crossed and replayed stores can all be canonical residues.
            data[1] = PackedFr::from_fr(&Fr::from(17u64));
            data[3] = expected[0];
            data[5] = PackedFr::from_fr(&expected[1].to_fr().square());
            assert_eq!(wrong_data(&data, &expected), (3, Some(1)));
            let next = prepare_data(&mut data, 8, 3);
            assert_ne!(expected, next);
        }

        /// One short dispatch, no overload or recovery: pins the CPU/GPU wire format
        /// and demonstrates that current markers and counts alone miss a wrong store.
        #[test]
        fn sealed_arithmetic_matches_cpu_and_corruption_is_detected() {
            objc2::rc::autoreleasepool(|_| {
                let gpu = Gpu::new();
                let data = gpu.buffer(BUSY_THREADS * 32);
                let marker = gpu.buffer(BUSY_THREADS * 4);
                let count = gpu.buffer(BUSY_THREADS * 4);
                let token = gpu.buffer(4);
                let epoch = 11;
                write_word(&token, 0);
                // SAFETY: fresh shared buffers, no work in flight. Each is sized for
                // BUSY_THREADS elements of its respective type.
                let expected = unsafe {
                    std::slice::from_raw_parts_mut(
                        count.contents().cast::<u32>().as_ptr(),
                        BUSY_THREADS,
                    )
                    .fill(0);
                    std::slice::from_raw_parts_mut(
                        marker.contents().cast::<u32>().as_ptr(),
                        BUSY_THREADS,
                    )
                    .fill(0);
                    prepare_data(
                        std::slice::from_raw_parts_mut(
                            data.contents().cast::<PackedFr>().as_ptr(),
                            BUSY_THREADS,
                        ),
                        epoch,
                        2,
                    )
                };
                let cb = command_buffer(&gpu.queue, true);
                let enc = mtl::encoder(&cb);
                gpu.encode_busy(enc, &data, 2, &marker, &count, epoch);
                gpu.encode_seal(enc, &token, epoch);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(read_word(&token), epoch);
                // SAFETY: correctly sized shared buffers, command buffer completed.
                unsafe {
                    let markers = std::slice::from_raw_parts(
                        marker.contents().cast::<u32>().as_ptr(),
                        BUSY_THREADS,
                    );
                    let counts = std::slice::from_raw_parts(
                        count.contents().cast::<u32>().as_ptr(),
                        BUSY_THREADS,
                    );
                    let words = std::slice::from_raw_parts_mut(
                        data.contents().cast::<PackedFr>().as_ptr(),
                        BUSY_THREADS,
                    );
                    assert!(markers.iter().all(|&m| m == epoch));
                    assert!(counts.iter().all(|&c| c == 1));
                    assert_eq!(wrong_data(words, &expected), (0, None));
                    words[BUSY_THREADS - 1] = PackedFr::from_fr(&Fr::from(23u64));
                    assert_eq!(wrong_data(words, &expected), (1, Some(BUSY_THREADS - 1)));
                    assert_eq!(read_word(&token), epoch);
                    assert!(markers.iter().all(|&m| m == epoch));
                    assert!(counts.iter().all(|&c| c == 1));
                }
            });
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    probe::main();
    #[cfg(not(target_os = "macos"))]
    eprintln!("macOS only");
}
