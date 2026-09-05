# Groth16 speed test

One page, one button. It downloads real proving keys from the artifact bucket smallest-first,
proves each circuit on the GPU through `crates/g16-wgpu` compiled to wasm, proves the same
circuit with snarkjs, makes each prover verify the other's proof, and shows the ratio.

    npm install
    npm run wasm      # builds crates/g16-wasm into static/pkg
    npm run dev       # http://localhost:5173

`npm run build` writes a fully static `build/` directory. There is no server side.

## The bucket needs a CORS rule

Every proving key is fetched cross-origin from `gpu-snark-bench`, so the bucket has to
say browsers may read it. Without the rule the deployed site fails at the first fetch with an
opaque CORS error, while `curl` on the same URL works fine, which is a confusing half hour.

    aws s3api put-bucket-cors --bucket gpu-snark-bench \
      --cors-configuration file://s3-cors.json

Dev does not need it: `vite.config.ts` proxies `/s3/*` to the bucket, so a checkout with no
AWS access still runs the whole benchmark. That proxy is dev-only by design, and it is also
why "works locally" is not evidence the rule is in place.

## What is measured, and what is deliberately not

Both provers get the identical `circuit.zkey` and `circuit.wtns`, already in memory. Nothing
fetches inside a timed call: hand `fastfile.readExisting` a URL string and snarkjs will fetch
inside `prove()`, and the row becomes a bandwidth measurement wearing a prover's name.

Each number is the **median** of the timed reps after a warm-up, never the mean, because the
mean of a run that contained a garbage collection is a number about the garbage collector.

snarkjs has no `prepare()`. It re-reads zkey sections 4 through 9 inside every `prove()`, so
its one number is both its cold and its warm one. Ours is warm, and the upload it warms up
with is shown as its own **GPU upload** column rather than folded into the proof. Comparing a
warm number of ours against snarkjs' only number without saying so is the exact vendor-chart
dishonesty this page exists to avoid.

Excluded from both sides, on purpose: witness generation (stage -1, CPU-only forever, and
neither prover can act on it), fetching, and the page-lifetime costs each prover pays once
(our wasm module and adapter; snarkjs building its BN254 module with wasmbuilder and
compiling it again inside every worker it spawns).

A rep that spans a hidden tab is discarded. Chrome throttles timers to 1 Hz in a hidden tab
and to 1/min after five minutes: that does not slow the wasm, but it moves the clock the wasm
is measured against.

The snarkjs worker count is read back from `curve.tm.concurrency` after a prove and refused
if it is 1 on a multi-core machine. A bundler that resolves snarkjs' `node` export condition
gives a silently single-threaded prover and a completely fake speedup for us. That is why
`static/vendor/snarkjs.min.js` is the vendored UMD bundle loaded by a `<script>` tag
(snarkjs 0.7.6, sha256 `3f61bbd9ac0a10173902eaef65b510fa4e9a2c057f759c7f18a6d0446b20fd06`)
rather than an npm import.

snarkjs is not "pure JS", whatever its reputation: `snarkjs -> ffjavascript -> wasmcurves` is
hand-written WebAssembly BN254 arithmetic spread across `navigator.hardwareConcurrency`
workers. It is a genuine multi-threaded wasm baseline, not a strawman.

## Circuits

Measured on an M2 Max in Chrome 152, warm, three untimed warm-ups then five timed reps, every
proof cross-verified in both directions. These are the numbers baked into the estimator as its
reference.

| circuit | constraints | key | snarkjs | WebGPU | GPU upload | |
|---|---:|---:|---:|---:|---:|---:|
| Railgun 1x1 | 20,135 | 10 MB | 660 ms | 347 ms | 13 ms | 1.9x |
| Tornado Cash | 28,275 | 15 MB | 1.05 s | 329 ms | 11 ms | 3.2x |
| SHA-256 | 59,281 | 34 MB | 910 ms | 145 ms | 24 ms | 6.3x |
| Railgun 13x1 | 141,276 | 68 MB | 4.13 s | 939 ms | 41 ms | 4.4x |
| RSA-2048 | 190,945 | 104 MB | 3.63 s | 488 ms | 44 ms | 7.4x |
| Keccak-256 | 239,176 | 108 MB | 3.07 s | 261 ms | 42 ms | 12x |
| Anon Aadhaar | 1,115,080 | 631 MB | — | — | — | listed, not run |

Median ratio 5.3x. Read the table before trusting constraint count as a proxy for cost: it is
not even monotonic. Tornado has less than half of SHA-256's constraints and is slower on both
provers, Railgun 13x1 is smaller than RSA on both counts and slower than it on both provers,
and Keccak is 1.25x RSA's constraint count and proves in 53% of the time on our side. Cost
tracks non-zero entries in the A/B/C matrices rather than rows of them, and on our side the
MSM additionally skips zero digits, so a bit-valued witness leaves most high windows empty.
That is why the estimator keys its reference by circuit instead of fitting a curve.

The WebGPU column also shows where this backend stops winning. It does not go below about
330 ms however small the circuit is, because that floor is fixed per-proof cost rather than
arithmetic: Railgun 1x1 is only 1.9x, and something a little smaller would lose outright.

Anon Aadhaar is shown as a skipped row rather than omitted. Its 1,101,048 wires need 134.4 MB
of G2 bases in a single storage binding, against the 128 MB WebGPU guarantees, so the backend
refuses it before it reaches the GPU. Chunked base bindings would fix it and are not written
yet. A prover's ceiling is a result; dropping the row would leave a page whose largest circuit
is the largest one that happens to work.

`?circuits=tornado,sha256` picks a subset, `?reps=5` changes the rep count, `?warmup=N` the
number of untimed reps before it, and `?profile=raised` asks the adapter for limits above the
floor, which it is free to decline.

## Estimates

The ETA has to guess at two things and guesses at them differently. **Download** is measured:
the estimator starts at a pessimistic 12 Mbit/s and is overwritten by observed throughput
within the first second of the first fetch. **Proving** is interpolated from a real measured
curve (this prover and snarkjs 0.7.6, Chrome 152, M2 Max, median of 15 reps) and then
rescaled by the first real measurement
taken on the machine actually running. Until that lands, the page says the estimate is not
from this machine.

Constraint count does not fully determine proving time, which is why the estimate is a
starting point rather than a prediction: Keccak-256 has 1.7x the wires of a synthetic circuit
that takes three times as long, because the MSM skips zero digits and a bit-valued witness
leaves most high windows empty.

## Deployment notes

WebGPU needs a secure context: HTTPS in production, with `localhost` and `127.0.0.1` exempt.
Chrome 113+ on desktop, 121+ on Android. Feature-detect `navigator.gpu` **and** null-check the
adapter — browsers with the property and a blocklisted driver hand back null, and a page that
only checks the first reports "supported" and then dies at the first proof.

`.wasm` must be served as `application/wasm` or `WebAssembly.instantiateStreaming` refuses it.
`static/_headers` covers Netlify and Cloudflare Pages, `vercel.json` covers Vercel.
