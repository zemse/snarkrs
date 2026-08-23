//! The browser entry point: this prover, as an ES module, driven from a dedicated Web Worker.
//!
//! Everything here is boundary work. The proof
//! itself is [`crate::backend::WgpuCircuit`], which compiles for both targets; what this
//! module adds is the four things a browser needs and a `cargo test` never exercises: a way
//! to get a 94 MB zkey into linear memory without peaking at twice its size, a CSPRNG that is
//! the browser's and not a stub, an async `prove` that yields to the event loop instead of
//! blocking the only thread there is, and a proof encoded exactly the way
//! `snarkjs.groth16.verify` reads it.
//!
//! # Free functions over one worker-local state, not an exported object
//!
//! Design §6 lists this surface as free functions and that is what it is. The reason is not
//! style: `#[wasm_bindgen]` async methods that borrow `&self` across an `.await` are a
//! standing source of trouble, and the object would be a singleton anyway. Design §6 puts the
//! whole prover in **one dedicated Web Worker** because `wgpu`'s
//! `fragile-send-sync-non-atomic-wasm` is what makes `wgpu::Device` satisfy
//! `Backend: Send + Sync` and it is sound only without `+atomics`, which rules out
//! wasm-bindgen-rayon. One worker, one device, one state.
//!
//! [`prove`] refuses to run twice at once rather than serialising on
//! [`crate::device::WgpuBackend::exclusive`]. That guard is a `std::sync::Mutex`, and held
//! across an `.await` on a browser's single thread a second caller would not queue behind the
//! first, it would deadlock the worker with no error anywhere.
//!
//! # The zkey path, and why it is not a `&[u8]` parameter
//!
//! `wasm-bindgen`'s `passArray8ToWasm0` copies the whole array into linear memory, so
//! `prove(zkey: &[u8])` peaks at the JS copy plus the wasm copy: 189 MB for the 94.4 MB
//! `js_16x16_d32` key, before anything is parsed. Instead [`zkey_alloc`] returns a pointer
//! into wasm memory and the page streams `fetch` chunks straight into it, then [`zkey_take`]
//! reconstitutes the `Vec` and parses it. One allocation, no JS-side copy.
//!
//! The rule the page must follow, and the reason `worker.js` re-derives its view on every
//! chunk: **any** wasm allocation can grow the memory, and growing detaches the old
//! `ArrayBuffer`. Writes through a detached view are not an error in JavaScript, they are
//! silently discarded, and the failure surfaces as a zkey that fails to parse somewhere in
//! the middle. [`wasm_memory_bytes`] is exported so the page can show that the peak is one
//! copy and not two.
//!
//! # Randomness
//!
//! `crypto.getRandomValues`, resolved from the global scope at [`create_prover`] so that a
//! context without it fails at start-up rather than at the first proof. This is the blinder
//! sampling of stage 10, `g16_core::prove`'s bound is `CryptoRng` for that reason, and there
//! is deliberately no fallback: a "temporary" stub here silently removes the zero-knowledge
//! property while every test still passes.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use g16_core::{json, verify::verify as verify_proof, ProveError, StageTimings};
use g16_field::Fr;
use g16_zkey::{wtns::Witness, ProvingKey, VerifyingKey};
use wasm_bindgen::prelude::*;
use web_time::Instant;

use crate::backend::WgpuCircuit;
use crate::batch::MsmBatch;
use crate::device::{LimitsProfile, WgpuBackend};

/// Everything one worker holds. There is exactly one, in a `thread_local`, because there is
/// exactly one device and wasm32 without `+atomics` has exactly one thread.
struct State {
    device: Arc<WgpuBackend>,
    msm: Arc<MsmBatch>,
    /// Resolved once at start-up. See the module docs on why there is no fallback.
    crypto: web_sys::Crypto,
    /// Parsed but not yet uploaded. Taken by [`prepare`], which consumes it.
    pk: Option<ProvingKey>,
    /// `Rc` so [`prove`] can clone a handle out of the `RefCell` and drop the borrow before
    /// it awaits. Holding a `Ref` across an `.await` is how a re-entrant call panics.
    circuit: Option<Rc<WgpuCircuit>>,
    witness: Option<Rc<Vec<Fr>>>,
    /// Set for the duration of a proof. See the module docs: the alternative is a deadlock.
    busy: bool,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
    /// Live allocations handed out by [`zkey_alloc`] / [`wtns_alloc`], keyed by pointer.
    ///
    /// The length is kept here rather than trusted from JS. `Vec::from_raw_parts` with a
    /// length that is not the one that was allocated is undefined behaviour, and the caller
    /// supplying it is a JavaScript file.
    static ALLOCS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

fn js(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn with_state<T>(f: impl FnOnce(&mut State) -> Result<T, JsError>) -> Result<T, JsError> {
    STATE.with(|s| match s.borrow_mut().as_mut() {
        Some(state) => f(state),
        None => Err(JsError::new(
            "create_prover() has not run, or it failed; there is no device",
        )),
    })
}

/// Installs the panic hook. Call once, before anything else.
///
/// Without it a Rust panic in a worker is `unreachable executed` with no message and no
/// stack, which during U13 cost more time than any other single thing.
#[wasm_bindgen]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Bytes of wasm linear memory, right now.
///
/// Exported so the page can show that streaming the zkey peaks at one copy of it and not two.
/// `performance.memory` is the wrong instrument: it reports the JS heap, and the whole point
/// of [`zkey_alloc`] is that the key never lands there.
#[wasm_bindgen]
pub fn wasm_memory_bytes() -> f64 {
    // `memory_size` counts 64 KiB pages. f64 rather than u32 because a 4 GiB wasm32 memory
    // does not fit in a u32 and JS has no u64 anyway.
    (core::arch::wasm32::memory_size(0) as f64) * 65536.0
}

// ---------------------------------------------------------------------------------------
// The streaming byte path.
// ---------------------------------------------------------------------------------------

fn alloc(len: usize) -> *mut u8 {
    // `vec![0; len].into_boxed_slice()` and not `Vec::with_capacity`: the capacity has to be
    // exactly `len` when the box is reconstituted, and `with_capacity` is allowed to give
    // more. The zeroing pass is about 10 ms on the 94 MB key and buys a `Vec` that is sound
    // to rebuild.
    let b = vec![0u8; len].into_boxed_slice();
    let ptr = Box::into_raw(b) as *mut u8;
    ALLOCS.with(|a| a.borrow_mut().push((ptr as usize, len)));
    ptr
}

/// Reclaims an allocation, checking it against the ones actually handed out.
///
/// # Safety
///
/// `ptr` must be one [`alloc`] returned and not yet taken. That is checked against `ALLOCS`
/// rather than trusted, because the caller is a JavaScript file and the alternative is
/// `Vec::from_raw_parts` on an arbitrary number.
fn take(ptr: *mut u8, len: usize) -> Result<Vec<u8>, JsError> {
    let found = ALLOCS.with(|a| {
        let mut a = a.borrow_mut();
        match a.iter().position(|&(p, l)| p == ptr as usize && l == len) {
            Some(i) => {
                a.swap_remove(i);
                true
            }
            None => false,
        }
    });
    if !found {
        return Err(JsError::new(
            "that pointer and length are not a live allocation from zkey_alloc or wtns_alloc",
        ));
    }
    // SAFETY: checked above to be exactly what `alloc` returned, and removed from the table
    // so it cannot be taken twice.
    let b = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
    Ok(b.into_vec())
}

/// Reserves `len` bytes of wasm memory for a zkey and returns a pointer into it.
///
/// The page writes the `fetch` stream into `new Uint8Array(memory.buffer, ptr, len)` and must
/// re-derive that view after every chunk. See the module docs.
#[wasm_bindgen]
pub fn zkey_alloc(len: usize) -> *mut u8 {
    alloc(len)
}

/// Reserves `len` bytes for a witness. Identical to [`zkey_alloc`]; both names exist because
/// design §6 names both and a page reading the design should find them.
#[wasm_bindgen]
pub fn wtns_alloc(len: usize) -> *mut u8 {
    alloc(len)
}

/// Releases an allocation the page decided not to fill, after an aborted fetch.
#[wasm_bindgen]
pub fn bytes_free(ptr: *mut u8, len: usize) -> Result<(), JsError> {
    take(ptr, len).map(drop)
}

/// Parses the zkey the page streamed in. The bytes are consumed; the pointer is dead after.
#[wasm_bindgen]
pub fn zkey_take(ptr: *mut u8, len: usize) -> Result<(), JsError> {
    let bytes = take(ptr, len)?;
    let pk = ProvingKey::from_bytes(bytes).map_err(js)?;
    with_state(|s| {
        s.pk = Some(pk);
        // A new key invalidates the uploaded one. Silently keeping the old circuit is how a
        // benchmark reports the wrong artifact's time.
        s.circuit = None;
        Ok(())
    })
}

/// Parses the witness the page streamed in.
#[wasm_bindgen]
pub fn wtns_take(ptr: *mut u8, len: usize) -> Result<(), JsError> {
    let bytes = take(ptr, len)?;
    let w = Witness::from_bytes(bytes).map_err(js)?;
    with_state(|s| {
        s.witness = Some(Rc::new(w.0));
        Ok(())
    })
}

// ---------------------------------------------------------------------------------------
// Device, preparation, proving.
// ---------------------------------------------------------------------------------------

/// Opens an adapter and a device and compiles the three MSM shader modules.
///
/// `profile` is `"floor"` or `"raised"`, and it is a parameter rather than an environment
/// variable because a browser has no environment. **Floor is the default and it is the honest
/// one**: it requests the WebGPU specification's own defaults (128 MiB storage binding,
/// 16 KiB workgroup storage, 256 invocations) rather than what this adapter happens to offer,
/// so a number taken here is one a stock browser on other hardware could reproduce. Chrome on
/// an M2 Max grants far more if asked. Whichever ran is reported by [`caps`] and belongs in
/// the CSV.
#[wasm_bindgen]
pub async fn create_prover(profile: Option<String>) -> Result<(), JsError> {
    let profile = LimitsProfile::parse(profile.as_deref().unwrap_or("floor")).map_err(js)?;

    // Before the device, because a context with no CSPRNG must fail at start-up and not at
    // stage 10 of the first proof. `js_sys::global()` covers Window and WorkerGlobalScope
    // with no cfg; `web_sys::window()` would be `None` in the worker this runs in.
    let global = js_sys::global();
    let crypto: web_sys::Crypto = js_sys::Reflect::get(&global, &JsValue::from_str("crypto"))
        .map_err(|_| JsError::new("this context has no `crypto`"))?
        .dyn_into()
        .map_err(|_| JsError::new("`crypto` is not a Crypto object"))?;

    let device = Arc::new(WgpuBackend::with_profile(profile).await.map_err(js)?);
    let msm = Arc::new(MsmBatch::new(&device).map_err(js)?);

    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            device,
            msm,
            crypto,
            pk: None,
            circuit: None,
            witness: None,
            busy: false,
        });
    });
    Ok(())
}

/// Adapter identity, the profile that was requested, and requested against granted limits.
///
/// Returned as a JSON string rather than an object, so this crate needs no
/// `serde-wasm-bindgen` and the page can put the text in a CSV cell unchanged. Design §7 rule
/// 5 requires every row to carry these; without them a ratio is not falsifiable.
#[wasm_bindgen]
pub fn caps() -> Result<String, JsError> {
    with_state(|s| {
        let d = &s.device;
        let info = d.adapter_info();
        let (r, g) = (d.requested_limits().clone(), d.granted_limits());
        let compile = d.prepare_cost();
        Ok(format!(
            concat!(
                r#"{{"profile":"{}","adapter":{{"name":{},"vendor":"{}","device":"{}","#,
                r#""backend":"{:?}","device_type":"{:?}","driver":{}}},"#,
                r#""limits":{{"#,
                r#""max_buffer_size":[{},{}],"#,
                r#""max_storage_buffer_binding_size":[{},{}],"#,
                r#""max_compute_workgroup_storage_size":[{},{}],"#,
                r#""max_compute_invocations_per_workgroup":[{},{}],"#,
                r#""max_storage_buffers_per_shader_stage":[{},{}]}},"#,
                r#""msm_modules":{},"msm_pipelines":{},"msm_compile_us":{}}}"#,
            ),
            d.profile().as_str(),
            json_string(&info.name),
            info.vendor,
            info.device,
            info.backend,
            info.device_type,
            json_string(&info.driver),
            r.max_buffer_size,
            g.max_buffer_size,
            r.max_storage_buffer_binding_size,
            g.max_storage_buffer_binding_size,
            r.max_compute_workgroup_storage_size,
            g.max_compute_workgroup_storage_size,
            r.max_compute_invocations_per_workgroup,
            g.max_compute_invocations_per_workgroup,
            r.max_storage_buffers_per_shader_stage,
            g.max_storage_buffers_per_shader_stage,
            compile.modules,
            compile.pipelines,
            compile.module_us,
        ))
    })
}

/// Minimal JSON string escaping, for adapter strings that come from a driver.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Uploads everything witness independent: the CSR, both twiddle tables, the coset powers and
/// all five base vectors. This is the "warm" half of every number this repo publishes.
///
/// Consumes the parsed zkey, so the page must call [`zkey_take`] again to prepare a different
/// key. Returns its own cost as JSON.
#[wasm_bindgen]
pub async fn prepare() -> Result<String, JsError> {
    let (device, msm, pk) = with_state(|s| {
        let pk =
            s.pk.take()
                .ok_or_else(|| JsError::new("no zkey; call zkey_alloc and zkey_take first"))?;
        Ok((Arc::clone(&s.device), Arc::clone(&s.msm), pk))
    })?;

    let t0 = Instant::now();
    // Synchronous: every upload is `Queue::write_buffer`, which copies into a staging belt
    // and needs no readback. Nothing here awaits, and that is a property of `WgpuCircuit`
    // rather than an accident, so the browser and the native path prepare identically.
    let circuit = WgpuCircuit::new(device, msm, pk).map_err(js)?;
    let total_us = t0.elapsed().as_micros() as u64;
    let cost = circuit.prepare_cost();
    let base_bytes = circuit.base_bytes();
    let (n_vars, n_public, domain) = (
        circuit.key().n_vars,
        circuit.key().n_public,
        circuit.key().domain_size,
    );

    with_state(|s| {
        s.circuit = Some(Rc::new(circuit));
        Ok(())
    })?;

    Ok(format!(
        r#"{{"stages_us":{},"bases_us":{},"total_us":{},"base_bytes":{},"n_vars":{},"n_public":{},"domain_size":{},"wasm_memory_bytes":{}}}"#,
        cost.stages_us,
        cost.bases_us,
        total_us,
        base_bytes,
        n_vars,
        n_public,
        domain,
        wasm_memory_bytes(),
    ))
}

/// One proof. Stages 0 to 11, on the device except for the blinding.
///
/// Returns JSON: `proof` in snarkjs' `proof.json` shape, `publicSignals` in `public.json`
/// shape, and the [`StageTimings`] the native harness records, so a browser row and a native
/// row have the same columns.
#[wasm_bindgen]
pub async fn prove() -> Result<String, JsError> {
    // Everything needed is cloned out under a short borrow, and the borrow is dropped before
    // the first `.await`. A `RefCell` borrow held across an await panics the moment the page
    // calls anything else on this module.
    let (circuit, witness) = with_state(|s| {
        if s.busy {
            // Not a queue. See the module docs: `exclusive()` is a std Mutex and on one
            // browser thread a second caller would deadlock rather than wait.
            return Err(JsError::new(
                "a proof is already running in this worker; prove() is not re-entrant",
            ));
        }
        let circuit = s
            .circuit
            .clone()
            .ok_or_else(|| JsError::new("not prepared; call prepare() first"))?;
        let witness = s
            .witness
            .clone()
            .ok_or_else(|| JsError::new("no witness; call wtns_alloc and wtns_take first"))?;
        s.busy = true;
        Ok((circuit, witness))
    })?;

    let out = prove_inner(&circuit, &witness).await;
    with_state(|s| {
        s.busy = false;
        Ok(())
    })?;
    out
}

async fn prove_inner(circuit: &WgpuCircuit, witness: &[Fr]) -> Result<String, JsError> {
    let mut t = StageTimings::default();
    let t0 = Instant::now();

    let h = circuit.compute_h_async(witness, &mut t).await.map_err(js)?;
    let m = circuit.msms_async(witness, &h, &mut t).await.map_err(js)?;

    // Stage 10, the trust boundary. `Fr::rand` over the browser's CSPRNG; see the module docs.
    let (r, s) = with_state(|st| {
        let mut rng = BrowserRng {
            crypto: st.crypto.clone(),
        };
        Ok((
            <Fr as ark_std::UniformRand>::rand(&mut rng),
            <Fr as ark_std::UniformRand>::rand(&mut rng),
        ))
    })?;

    // Stage 11, from `g16-core`, so the browser and the CLI blind identically.
    let proof = g16_core::prove::assemble(circuit.key(), &m, r, s, &mut t);
    let total_us = t0.elapsed().as_micros() as u64;

    let n_public = circuit.key().n_public;
    if witness.len() < n_public + 1 {
        return Err(js(ProveError::WitnessLength {
            got: witness.len(),
            want: n_public + 1,
        }));
    }
    let public = &witness[1..=n_public];

    Ok(format!(
        r#"{{"proof":{},"publicSignals":{},"timings":{{"gather_us":{},"ntt_us":{},"pointwise_us":{},"msm_us":{},"assemble_us":{},"total_us":{}}}}}"#,
        json::proof_to_string(&proof).trim_end(),
        json::public_to_string(public).trim_end(),
        t.gather_us,
        t.ntt_us,
        t.pointwise_us,
        t.msm_us,
        t.assemble_us,
        total_us,
    ))
}

/// Our own verifier, over snarkjs' own JSON. Returns `true` only if the pairing check holds.
///
/// Design §7 rule 8: the page cross-verifies both ways before recording a timing, ours
/// through `snarkjs.groth16.verify` and snarkjs' through this. A verifier that shares no code
/// with the prover is the only oracle that catches the two of them being wrong together.
#[wasm_bindgen]
pub fn verify(vkey_json: &str, public_json: &str, proof_json: &str) -> Result<bool, JsError> {
    let vk = VerifyingKey::from_json_str(vkey_json).map_err(js)?;
    let public = json::public_from_str(public_json).map_err(js)?;
    let proof = json::proof_from_str(proof_json).map_err(js)?;
    // A rejected proof is `false` and not an error: the page distinguishes "did not verify"
    // from "could not be read", and only the first one is a result.
    Ok(verify_proof(&vk, &public, &proof).is_ok())
}

// ---------------------------------------------------------------------------------------
// Stage 10's CSPRNG.
// ---------------------------------------------------------------------------------------

/// `crypto.getRandomValues`, as an `ark_std` `RngCore`.
///
/// `getrandom` is deliberately not linked: `cargo tree --target wasm32-unknown-unknown -p
/// g16-core -i getrandom` matches nothing, and keeping it that way means no crate in this
/// graph can quietly acquire a wasm entropy source we did not choose.
struct BrowserRng {
    crypto: web_sys::Crypto,
}

impl BrowserRng {
    /// `getRandomValues` throws on a request over 65,536 bytes, so this chunks. Stage 10 asks
    /// for 64, but a silent truncation here would be a silent loss of zero knowledge.
    fn fill(&self, dest: &mut [u8]) -> Result<(), JsValue> {
        for chunk in dest.chunks_mut(65_536) {
            self.crypto.get_random_values_with_u8_array(chunk)?;
        }
        Ok(())
    }
}

impl ark_std::rand::RngCore for BrowserRng {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.fill_bytes(&mut b);
        u32::from_le_bytes(b)
    }

    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.fill_bytes(&mut b);
        u64::from_le_bytes(b)
    }

    /// Panics rather than degrading. This is the blinder sampling; the caller has no
    /// meaningful recovery and a zeroed buffer would produce a proof that verifies and leaks.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.fill(dest)
            .expect("crypto.getRandomValues failed; refusing to blind with anything else");
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), ark_std::rand::Error> {
        match self.fill(dest) {
            Ok(()) => Ok(()),
            // `Error::new` would be the obvious call and it does not exist here:
            // `ark-std` is `default-features = false`, so `rand_core`'s `std` is off and the
            // boxed-payload constructor is compiled out. The numeric form is the one that
            // survives, and `CUSTOM_START` is the range rand_core reserves for callers.
            Err(_) => Err(ark_std::rand::Error::from(
                core::num::NonZeroU32::new(ark_std::rand::Error::CUSTOM_START)
                    .expect("CUSTOM_START is 0x8000_0000 and therefore nonzero"),
            )),
        }
    }
}

/// The bound `g16_core::prove` requires, and the claim it encodes: these bytes come from the
/// platform CSPRNG.
impl ark_std::rand::CryptoRng for BrowserRng {}
