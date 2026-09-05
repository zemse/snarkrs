/// <reference lib="webworker" />
/// The prover's host. Everything expensive on our side of the comparison happens in here.
///
/// **Why a worker.** The clock is on the main thread and the run refuses any timing taken
/// while the tab is hidden. A one-second synchronous stretch of wasm on the main thread does
/// not merely block paint, it blocks the check that is supposed to notice. It also freezes
/// the progress ring, which for a page whose whole job is to look like it is working is not
/// a cosmetic problem.
///
/// **Why exactly one worker and not a pool.** wgpu's `fragile-send-sync-non-atomic-wasm` is
/// what lets `wgpu::Device` satisfy `Backend: Send + Sync`, and it is sound only in a build
/// without `+atomics`. `wasm-bindgen-rayon` requires `+atomics`. Browser threads and a Send
/// GPU backend are mutually exclusive in one module, so the prover lives in one worker and
/// the zkey never crosses a postMessage boundary.
///
/// The protocol is request/response keyed on an id, with the handler's own data nested under
/// `value` rather than spread into the envelope. It was spread once, and a handler returning
/// its own `ok` field overwrote the envelope's: `verify` reporting a good proof came out as
/// a rejected promise. Nesting makes the collision impossible rather than documented.

type Wasm = {
  default: (m?: unknown) => Promise<unknown>;
  start: () => void;
  create_prover: (profile?: string) => Promise<void>;
  caps: () => string;
  zkey_alloc: (n: number) => number;
  zkey_take: (p: number, n: number) => void;
  wtns_alloc: (n: number) => number;
  wtns_take: (p: number, n: number) => void;
  bytes_free: (p: number, n: number) => void;
  unload: () => void;
  prepare: () => Promise<string>;
  prove: () => Promise<string>;
  verify: (vkey: string, pub: string, proof: string) => boolean;
  wasm_memory: () => WebAssembly.Memory;
  wasm_memory_bytes: () => number;
};

let wasm: Wasm | null = null;
let ready = false;

function w(): Wasm {
  if (!wasm || !ready) throw new Error('the prover worker is not initialised');
  return wasm;
}

/// Streams a URL straight into wasm linear memory and returns the pointer.
///
/// The two rules here are the entire reason this is not
/// `new Uint8Array(await r.arrayBuffer())` handed to a `&[u8]` parameter.
///
/// 1. **Never pass the bytes to wasm-bindgen as a slice.** `passArray8ToWasm0` allocates
///    inside wasm and copies, so the peak is the JS ArrayBuffer plus the wasm copy. For the
///    108 MB keccak key that is 216 MB before a single section has been parsed. Measured on
///    a 94.4 MB key: allocating and streaming grew linear memory by 94,437,376 bytes, one
///    copy, and the slice form would have doubled it.
/// 2. **Re-derive the view after every chunk.** Any wasm allocation can grow the memory, and
///    growing detaches the old ArrayBuffer. A `.set()` through a detached view is not an
///    error in JavaScript, it is a silent no-op, and the symptom is a zkey that fails to
///    parse somewhere in the middle with nothing pointing at the cause. Nothing in this loop
///    allocates today; the view is rebuilt anyway, because "nothing allocates today" is not
///    a property anyone will re-check.
async function streamInto(
  url: string,
  alloc: (n: number) => number,
  onProgress: (done: number, total: number) => void
) {
  const t0 = performance.now();
  const r = await fetch(url).catch((e) => {
    // A CORS rejection reaches JavaScript as a bare TypeError reading "Failed to fetch",
    // with the real reason left in the console where a visitor will not look. Since the one
    // way this page fails on a fresh deployment is a bucket without a CORS rule, say so
    // rather than making someone match a generic network error to a config file.
    const cross = new URL(url, self.location.href).origin !== self.location.origin;
    throw new Error(
      cross
        ? `could not fetch ${url}: ${e?.message ?? e}. The artifact host has to allow ` +
          `cross-origin reads from ${self.location.origin}; see web/s3-cors.json.`
        : `could not fetch ${url}: ${e?.message ?? e}`
    );
  });
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  const len = Number(r.headers.get('content-length') ?? 0);
  if (!len) {
    // Without a length there is nothing to reserve up front, and buffering the body to find
    // out is exactly the second copy this function exists to avoid. S3 always sends one; a
    // proxy that strips it is a misconfiguration worth failing loudly on.
    throw new Error(`${url}: no content-length, so the size cannot be reserved up front`);
  }

  const ptr = alloc(len);
  try {
    const reader = r.body!.getReader();
    let off = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (off + value.length > len) {
        throw new Error(`${url}: body is longer than the ${len} bytes it declared`);
      }
      new Uint8Array(w().wasm_memory().buffer, ptr, len).set(value, off);
      off += value.length;
      onProgress(off, len);
    }
    if (off !== len) throw new Error(`${url}: got ${off} bytes, content-length said ${len}`);
  } catch (e) {
    w().bytes_free(ptr, len);
    throw e;
  }
  return { ptr, len, ms: performance.now() - t0 };
}

type Args = Record<string, any>;

const handlers: Record<string, (a: Args, emit: (p: unknown) => void) => Promise<unknown>> = {
  /// Loads the wasm module and opens the adapter. Both are once per page rather than once
  /// per proof, and both are reported rather than swallowed, because they are the part of a
  /// cold start that this page's numbers deliberately exclude. snarkjs pays the equivalent
  /// (building its BN254 module with wasmbuilder, then compiling it again inside every
  /// worker it spawns) and it is not in snarkjs' number either.
  async init({ pkgBase, profile }) {
    const t0 = performance.now();
    // A variable URL, so Vite leaves it alone. static/pkg is produced by
    // scripts/build-wasm.sh and is not part of the module graph.
    const mod = (await import(/* @vite-ignore */ `${pkgBase}/g16_wasm.js`)) as Wasm;
    await mod.default();
    mod.start();
    const t1 = performance.now();
    wasm = mod;
    await mod.create_prover(profile ?? 'floor');
    ready = true;
    return {
      caps: JSON.parse(mod.caps()),
      moduleMs: t1 - t0,
      deviceMs: performance.now() - t1,
      memBytes: mod.wasm_memory_bytes()
    };
  },

  /// Fetches one circuit's proving key and witness and parses both.
  ///
  /// The fetch is not inside any timed proving region, and deliberately so: fetching is not
  /// something snarkjs does inside `groth16.prove` either. It is timed separately because
  /// the page shows it, and because the observed throughput is what the ETA runs on.
  ///
  /// Returns the raw bytes as well, because snarkjs needs the same two files and downloading
  /// them twice would be both slow and a different test. They leave wasm as a copy; that is
  /// unavoidable, since snarkjs cannot read another module's linear memory.
  async load({ base, name }, emit) {
    const t0 = performance.now();
    const z = await streamInto(`${base}/${name}/circuit.zkey`, w().zkey_alloc, (d, t) =>
      emit({ file: 'zkey', done: d, total: t })
    );
    // Copied out before `zkey_take` consumes the allocation. snarkjs gets this array.
    const zkeyBytes = new Uint8Array(
      new Uint8Array(w().wasm_memory().buffer, z.ptr, z.len)
    );
    const tz = performance.now();
    w().zkey_take(z.ptr, z.len);
    const zkeyParseMs = performance.now() - tz;

    const wt = await streamInto(`${base}/${name}/circuit.wtns`, w().wtns_alloc, (d, t) =>
      emit({ file: 'wtns', done: d, total: t })
    );
    const wtnsBytes = new Uint8Array(
      new Uint8Array(w().wasm_memory().buffer, wt.ptr, wt.len)
    );
    w().wtns_take(wt.ptr, wt.len);

    // Small enough to keep as JSON. The vkey is what both provers get verified against.
    const [vkey, publicSignals] = await Promise.all(
      ['vkey.json', 'public.json'].map(async (f) => {
        const url = `${base}/${name}/${f}`;
        const r = await fetch(url).catch((e) => {
          throw new Error(`could not fetch ${url}: ${e?.message ?? e}`);
        });
        if (!r.ok) throw new Error(`${name}/${f}: ${r.status} ${r.statusText}`);
        return r.json();
      })
    );

    return {
      zkeyBytes,
      wtnsBytes,
      vkey,
      publicSignals,
      zkeyParseMs,
      downloadMs: z.ms + wt.ms,
      downloadedBytes: z.len + wt.len,
      totalMs: performance.now() - t0,
      memBytes: w().wasm_memory_bytes()
    };
  },

  /// Uploads every base vector and both twiddle tables to the GPU. This is the cost snarkjs
  /// has no equivalent of, so it is reported on its own line and never folded into a proof.
  async prepare() {
    const t0 = performance.now();
    const r = JSON.parse(await w().prepare());
    return { ...r, wallMs: performance.now() - t0 };
  },

  async prove() {
    const t0 = performance.now();
    const r = JSON.parse(await w().prove());
    return { ...r, wallMs: performance.now() - t0 };
  },

  /// Our verifier over snarkjs' proof, so the cross-check runs both ways. `verified: false`
  /// means the pairing check failed; a malformed input throws instead.
  async verify({ vkey, publicSignals, proof }) {
    return {
      verified: w().verify(
        JSON.stringify(vkey),
        JSON.stringify(publicSignals),
        JSON.stringify(proof)
      )
    };
  },

  /// Drops the key, the witness and every device buffer, between circuits. Without it the
  /// peak is two circuits rather than one, and a sweep dies at a circuit that fits on its own.
  async unload() {
    w().unload();
    return { memBytes: w().wasm_memory_bytes() };
  }
};

self.onmessage = async (ev: MessageEvent) => {
  const { id, type, ...args } = ev.data;
  const emit = (progress: unknown) => self.postMessage({ id, progress });
  try {
    const h = handlers[type];
    if (!h) throw new Error(`unknown message type ${JSON.stringify(type)}`);
    const value = await h(args, emit);
    // The raw key and witness are ~108 MB for the largest circuit here and the main thread
    // needs them to feed snarkjs. Transferring the backing buffers hands them over instead
    // of structured-cloning them, which would put a second copy of the key in the process
    // at the exact moment the first one is still live.
    const transfer: ArrayBuffer[] = [];
    for (const v of Object.values(value ?? {})) {
      if (ArrayBuffer.isView(v)) transfer.push(v.buffer as ArrayBuffer);
    }
    (self as unknown as Worker).postMessage({ id, ok: true, value }, transfer);
  } catch (e: any) {
    // `e` may be a JsError from Rust, whose message carries the Rust error text.
    self.postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
