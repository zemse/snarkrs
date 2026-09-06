/// The state machine behind the button.
///
/// For each circuit, smallest first: download the proving key and witness, prove it on the
/// GPU, then prove the same thing with snarkjs, cross-verify both proofs in both directions,
/// publish the row, drop everything, move on. Partial results are visible the whole way
/// through, because at real circuit sizes the last row can be two minutes after the first
/// and a page that shows nothing until the end looks broken.

import {
  CIRCUITS,
  FLOOR_STORAGE_BINDING,
  cannotRun,
  selectCircuits,
  totalBytes,
  type Circuit
} from './circuits';
import { Estimator } from './estimate';
import { Prover } from './prover';
import { snarkjsProve, snarkjsVerify } from './snarkjs';
import { median } from './format';
import { visible, Discards } from './visibility';

export type RowStatus = 'pending' | 'downloading' | 'webgpu' | 'snarkjs' | 'done' | 'error' | 'skipped';

export type Row = {
  circuit: Circuit;
  status: RowStatus;
  /// Bytes of this circuit's two files that have landed, for the download bar.
  downloaded: number;
  downloadMs?: number;
  /// Uploading the bases to the GPU. Reported on its own line, never folded into a proof:
  /// snarkjs has no equivalent step and burying ours inside our number would be the exact
  /// vendor-chart dishonesty this page exists to avoid.
  prepareMs?: number;
  webgpuMs?: number;
  snarkjsMs?: number;
  /// Every timed rep, not just the median. Kept so the page can show how tight the
  /// measurement was, and so a reader can tell a stable number from a lucky one.
  webgpuReps?: number[];
  snarkjsReps?: number[];
  /// The prover's own per-stage breakdown from the last rep, in microseconds. The same
  /// columns the native harness records, so a browser row and a native row are comparable,
  /// and the first place to look when a proof comes back wrong: a stage reporting zero did
  /// not run.
  stages?: Record<string, number>;
  snarkjsWorkers?: number | null;
  /// Each prover's proof, checked by the other one's verifier.
  crossVerified?: boolean;
  note?: string;
  error?: string;
  /// Reps finished so far on each side, warm-ups included. Only the ETA reads these: a
  /// circuit that is four proofs into a five-proof block has four proofs less left to do,
  /// and without counting them the estimate sits still for the whole block and then drops
  /// off a cliff when the median lands.
  webgpuDone?: number;
  snarkjsDone?: number;
};

const q = () => (typeof location === 'undefined' ? '' : location.search);
const param = (k: string, d: number) => {
  const v = Number(new URLSearchParams(q()).get(k));
  return Number.isFinite(v) && v > 0 ? v : d;
};

export class Run {
  rows = $state<Row[]>(
    selectCircuits(typeof location === 'undefined' ? '' : location.search).map((circuit) => ({
      circuit,
      // Every row starts runnable. What this device can actually hold is not known until an
      // adapter has been asked, so `plan()` is what marks rows skipped, and it is called
      // again once the device reports what it really granted.
      status: 'pending' as const,
      downloaded: 0
    }))
  );
  phase = $state<'idle' | 'starting' | 'running' | 'done' | 'error'>('idle');
  /// One line under the ring saying what is happening right now.
  activity = $state('');
  etaMs = $state<number | null>(null);
  /// 0..1 across the whole run. The one progress number the page draws; see `work()`.
  progress = $state(0);
  calibrated = $state(false);
  mbits = $state(0);
  env = $state<any>(null);
  fatal = $state<string | null>(null);

  private est = new Estimator();
  private prover: Prover | null = null;
  private cancelled = false;
  private startedAt = 0;
  /// The operation currently being awaited, and what it was estimated to cost. Everything
  /// on the proving path is one long await with no progress events inside it, so without
  /// this the ETA would be a staircase: frozen for the eleven seconds of a snarkjs proof,
  /// then eleven seconds lower. Subtracting the time already spent inside the current
  /// operation is what lets the number move while nothing is reporting.
  private inFlight: { estMs: number; startedAt: number } | null = null;
  private ticker: ReturnType<typeof setInterval> | null = null;

  /// Downloads and proofs are timed for real; the ETA has to guess. Both curves live in
  /// Estimator and both get re-anchored to this machine as measurements come in.
  private readonly reps = param('reps', 3);
  /// Three, not one. The first `prove()` on a page builds snarkjs' BN254 module with
  /// wasmbuilder, compiles it again inside each of the twelve workers it spawns, and runs
  /// while the JIT is still tiering up: on a cold page with one warm-up the first circuit
  /// came out at 4.09 s, against 1.03 s for the same circuit once the page was warm. Three
  /// untimed reps is what the native harness settled on for the same reason.
  private readonly warmup = param('warmup', 3);
  /// `auto` by default: the floor for everything that shapes a kernel, and the adapter's own
  /// number for the two limits that only decide how much fits. `?profile=floor` forces the
  /// specification's guarantees, which is what to use when the question is what a stock
  /// browser elsewhere would do rather than what this machine can do.
  private readonly profile = new URLSearchParams(q()).get('profile') ?? 'auto';
  /// Where the artifacts come from. In dev this is the Vite proxy, which exists so a
  /// checkout with no AWS access still runs the whole benchmark; in a build it is the bucket
  /// directly, which is what makes the CORS rule in s3-cors.json load-bearing.
  ///
  /// `VITE_ARTIFACT_BASE` overrides both, for a deployment that fronts the bucket with a CDN
  /// or serves the keys from its own origin. Note that pointing it at the site's own origin
  /// is also the way out if the bucket cannot be given a CORS rule at all.
  private readonly base =
    import.meta.env.VITE_ARTIFACT_BASE ??
    (import.meta.env.DEV
      ? '/s3/artifacts'
      : 'https://gpu-snark-bench.s3.amazonaws.com/artifacts');

  get circuits() {
    return selectCircuits(q());
  }
  get totalDownload() {
    return totalBytes(this.circuits, this.bindingLimit);
  }
  /// How many circuits will actually be proved. Not `circuits.length`, which counts the
  /// listed-but-refused rows too, and would promise work the run is not going to do.
  /// What one storage binding may hold on this device, in bytes.
  ///
  /// Starts at the WebGPU floor, which is what the specification guarantees and therefore
  /// the only safe assumption before an adapter has been asked. The page raises it from
  /// `checkSupport()` as soon as that resolves, and the runner lowers it again if the device
  /// turns out to have granted less than the adapter advertised.
  bindingLimit = $state(FLOOR_STORAGE_BINDING);

  /// Marks the rows this device cannot hold, and un-marks the ones it can.
  ///
  /// Idempotent and safe to call repeatedly: it only ever moves a row between `pending` and
  /// `skipped`, never touching one that has already run. That matters because it is called
  /// three times, on three progressively better answers to "how big a buffer is allowed":
  /// the floor at construction, the adapter's claim once `checkSupport()` returns, and the
  /// device's actual grant once the worker has opened one.
  plan(limit: number) {
    this.bindingLimit = limit;
    for (const r of this.rows) {
      if (r.status !== 'pending' && r.status !== 'skipped') continue;
      const why = cannotRun(r.circuit, limit);
      r.status = why ? 'skipped' : 'pending';
      // Only a row that cannot run says why. A row that runs because this GPU offers more
      // than the specification's floor used to carry a note saying so, and on the machines
      // that can run everything it read as a warning attached to the one circuit that had
      // nothing wrong with it. The comparability point it was making is real but belongs
      // once in the footer, not per row: see `aboveFloor`.
      r.note = why ?? undefined;
    }
  }

  /// Circuits this run will prove that a stock floor-only device could not hold.
  ///
  /// Worth stating once, because a seven-circuit result from here and a six-circuit result
  /// from a floor-only device are not the same measurement of the same ladder, and nothing
  /// else on the page would say so.
  get aboveFloor() {
    return this.rows.filter(
      (r) => r.status !== 'skipped' && cannotRun(r.circuit, FLOOR_STORAGE_BINDING)
    );
  }

  get willRun() {
    return this.rows.filter((r) => r.status !== 'skipped').length;
  }

  async start() {
    if (this.phase === 'running' || this.phase === 'starting') return;
    this.cancelled = false;
    this.fatal = null;
    this.phase = 'starting';
    // A handle for reading the raw per-rep timings out of the console. The page shows a
    // median and a spread; this is how the rep counts in `reps`/`warmup` were chosen, and
    // how they should be re-chosen on a machine that behaves differently.
    (globalThis as Record<string, unknown>).__g16run = this;
    this.startedAt = performance.now();
    // Reset rather than rebuild, so a second run reuses the same objects and the table does
    // not flash empty between runs.
    for (const r of this.rows) {
      r.status = 'pending';
      r.downloaded = 0;
      r.downloadMs = r.prepareMs = r.webgpuMs = r.snarkjsMs = undefined;
      r.crossVerified = undefined;
      r.webgpuReps = r.snarkjsReps = undefined;
      r.webgpuDone = r.snarkjsDone = undefined;
      r.stages = undefined;
      r.error = undefined;
      r.note = undefined;
    }
    this.plan(this.bindingLimit);
    this.progress = 0;
    this.recomputeEta();
    // The ETA is the only thing on the page that moves while a proof is running, and the
    // ring is drawn from it, so it is driven by a clock rather than by work completing.
    // 200 ms is under the ~250 ms at which a countdown starts to look broken and far above
    // anything this costs to recompute.
    this.ticker ??= setInterval(() => this.recomputeEta(), 200);

    try {
      this.activity = 'opening the GPU';
      this.prover = new Prover();
      const init = await this.prover.init(`${location.origin}/pkg`, this.profile);
      this.env = {
        ...init.caps,
        moduleMs: init.moduleMs,
        deviceMs: init.deviceMs,
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated: self.crossOriginIsolated,
        // wgpu reports an empty AdapterInfo through WebGPU, so the adapter's own strings
        // have to come from JS rather than from the prover.
        adapter: await adapterInfo()
      };

      // Re-plan against what the device actually granted, which is the first authoritative
      // answer. Until now the ladder was sized from the adapter's *claim*, and the two can
      // differ: `auto` asks for the adapter's number, but a driver may refuse it and the
      // backend then falls back to the floor rather than failing outright. When that
      // happens the fallback reason is carried in `caps()` so the row can say the circuit
      // was dropped because this device would not grant the memory, not because the circuit
      // is too big in principle.
      const granted = this.env?.limits?.max_storage_buffer_binding_size?.[1];
      if (typeof granted === 'number' && granted > 0) this.plan(granted);
      if (this.env?.auto_fallback) {
        this.activity = 'this device refused the larger limits; running what fits';
      }
      // `?selftest=1` runs the GPU known-answer battery before any proving and puts the
      // result in `env`, which is what `?report=1` posts. It is off by default because it
      // compiles seven throwaway shader modules; it is the first thing to turn on when a
      // browser produces a proof nobody can explain.
      if (new URLSearchParams(q()).get('selftest') === '1') {
        this.activity = 'running the GPU self-test';
        try {
          this.env.selftest = await this.prover.selftest();
        } catch (e: any) {
          this.env.selftest = { error: String(e?.message ?? e) };
        }
      }
      this.phase = 'running';

      for (const row of this.rows) {
        if (this.cancelled) break;
        if (row.status === 'skipped') continue;
        await this.runOne(row);
        this.recomputeEta();
      }
      this.phase = this.cancelled ? 'idle' : 'done';
      await this.report();
      this.activity = '';
      this.etaMs = null;
      if (this.phase === 'done') this.progress = 1;
    } catch (e: any) {
      this.phase = 'error';
      this.fatal = String(e?.message ?? e);
      // Reported too, and not only on the happy path. The failure worth reading is usually
      // the one that stopped the run before it produced a row: a device that refused to
      // open, or a GPU self-test that refused to let it prove. Without this, `?report=1`
      // posts nothing at all and the log looks like the page never ran.
      await this.report();
    } finally {
      if (this.ticker != null) {
        clearInterval(this.ticker);
        this.ticker = null;
      }
      this.inFlight = null;
      this.prover?.terminate();
      this.prover = null;
    }
  }

  /// Posts the finished rows to the dev server when `?report=1`. See the `g16-report-sink`
  /// plugin in vite.config.ts for why: it is the only way to read a result out of a browser
  /// this machine cannot drive.
  private async report() {
    if (new URLSearchParams(q()).get('report') !== '1') return;
    const body = {
      userAgent: navigator.userAgent,
      env: this.env,
      phase: this.phase,
      fatal: this.fatal,
      rows: this.rows.map((r) => ({
        circuit: r.circuit.name,
        status: r.status,
        snarkjsMs: r.snarkjsMs,
        webgpuMs: r.webgpuMs,
        prepareMs: r.prepareMs,
        crossVerified: r.crossVerified,
        webgpuReps: r.webgpuReps,
        snarkjsReps: r.snarkjsReps,
        stages: r.stages,
        error: r.error,
        note: r.note
      }))
    };
    try {
      await fetch('/__report', { method: 'POST', body: JSON.stringify(body, null, 2) });
    } catch {
      /* reporting is a debugging aid; never let it take a completed run down */
    }
  }

  stop() {
    this.cancelled = true;
    this.activity = 'stopping after this circuit';
  }

  private async runOne(row: Row) {
    const p = this.prover!;
    const c = row.circuit;
    try {
      row.status = 'downloading';
      this.activity = `downloading ${c.label} (${(c.zkeyBytes / 1024 ** 2).toFixed(0)} MB)`;
      const loaded = await p.load(this.base, c.name, (pr: any) => {
        // zkey then witness, so the bar covers both files rather than resetting between them.
        const done = pr.file === 'zkey' ? pr.done : c.zkeyBytes + pr.done;
        row.downloaded = done;
        this.recomputeEta();
      });
      row.downloaded = loaded.downloadedBytes;
      row.downloadMs = loaded.downloadMs;
      this.est.observeDownload(loaded.downloadedBytes, loaded.downloadMs);
      this.mbits = this.est.mbitsPerSecond;

      // ---- our side first, as asked. -------------------------------------------------
      row.status = 'webgpu';
      this.activity = `${c.label}: uploading ${(c.zkeyBytes / 1024 ** 2).toFixed(0)} MB to the GPU`;
      const prep = await this.timed(this.est.prepareMs(c), () => p.prepare());
      row.prepareMs = prep.wallMs;
      this.est.observePrepare();

      const ours: number[] = [];
      let ourProof: any = null;
      const gpuDiscards = new Discards(this.reps + 5);
      // `i` advances only on a rep that counted. A warm-up is never discarded: it is not
      // measured, so a hidden tab does not spoil it, and re-taking it would only be slower.
      for (let i = 0; i < this.warmup + this.reps; ) {
        await visible();
        // Warm-ups counted, and counted the same way snarkjs counts them, because the two
        // lines sit in the same place on screen one after the other and a reader comparing
        // them should not have to know that one of them hides its warm-up.
        this.activity = `${c.label}: proving on the GPU (${i + 1}/${this.warmup + this.reps})`;
        const r = await this.timed(this.est.proveMs('webgpu', c), () => p.prove());
        if (i >= this.warmup && document.visibilityState !== 'visible') {
          gpuDiscards.count(`${c.label} on the GPU`);
          continue;
        }
        ourProof = r;
        if (i >= this.warmup) ours.push(r.wallMs);
        i++;
        row.webgpuDone = i;
      }
      row.webgpuReps = ours;
      row.stages = ourProof?.timings;
      row.webgpuMs = median(ours);
      this.est.observeProof('webgpu', c, row.webgpuMs);
      this.recomputeEta();

      // ---- then snarkjs, on the identical bytes. ---------------------------------------
      row.status = 'snarkjs';
      this.activity = `${c.label}: proving with snarkjs`;
      // snarkjs runs its whole rep loop inside one call and reports back per rep, so the
      // in-flight window is re-armed from the callback rather than wrapped around the call.
      this.inFlight = { estMs: this.est.proveMs('snarkjs', c), startedAt: performance.now() };
      const sj = await snarkjsProve(
        loaded.zkeyBytes,
        loaded.wtnsBytes,
        loaded.vkey,
        this.reps,
        this.warmup,
        (done, total) => {
          row.snarkjsDone = done;
          this.inFlight = { estMs: this.est.proveMs('snarkjs', c), startedAt: performance.now() };
          this.activity = `${c.label}: proving with snarkjs (${done}/${total})`;
          this.recomputeEta();
        }
      );
      if (!sj.ms.length) throw new Error('every snarkjs rep was discarded (tab hidden?)');
      row.snarkjsReps = sj.ms;
      row.snarkjsMs = median(sj.ms);
      row.snarkjsWorkers = sj.workers;
      if (sj.warning) row.note = sj.warning;
      this.est.observeProof('snarkjs', c, row.snarkjsMs);

      // ---- neither prover is allowed to be its own judge. ------------------------------
      //
      // Three checks, not one, because "cross-verification failed" on its own sends the
      // reader looking in the wrong place. Each of the three fails for a different reason:
      //
      //  * public signals differ  -> the two provers did not prove the same statement, so
      //    one of them read the witness or the key differently. Nothing downstream of this
      //    means anything, so it is checked first.
      //  * snarkjs rejects ours   -> our prover is wrong on this browser.
      //  * we reject snarkjs'     -> our verifier is wrong on this browser. snarkjs already
      //    verified this proof itself during the warm-up, so the proof is good.
      //
      // The distinction is not hypothetical: a report of this failing on Safari is exactly
      // the case where knowing which of the three broke is the whole diagnosis.
      const samePublic =
        JSON.stringify(ourProof.publicSignals) === JSON.stringify(sj.publicSignals);
      const theirsChecksOurs = await snarkjsVerify(
        loaded.vkey,
        ourProof.publicSignals,
        ourProof.proof
      );
      const oursChecksTheirs = await p.verify(loaded.vkey, sj.publicSignals, sj.proof);
      row.crossVerified = samePublic && theirsChecksOurs && oursChecksTheirs.verified;
      if (!row.crossVerified) {
        const why = !samePublic
          ? 'the two provers produced different public signals, so they did not prove the ' +
            'same statement; the key or the witness is being read differently'
          : !theirsChecksOurs && !oursChecksTheirs.verified
            ? 'snarkjs rejects our proof and we reject snarkjs\u2019, so both our prover and ' +
              'our verifier disagree with snarkjs on this browser'
            : !theirsChecksOurs
              ? 'snarkjs rejects our proof, so our prover is producing a bad proof on this ' +
                'browser (our verifier accepts snarkjs\u2019 proof, so the verifier is fine)'
              : 'we reject snarkjs\u2019 proof, which snarkjs itself verified, so our ' +
                'verifier is wrong on this browser and the prover may be fine';
        throw new Error(`cross-verification failed: ${why}`);
      }

      row.status = 'done';
    } catch (e: any) {
      const msg = String(e?.message ?? e);
      // The backend refuses an oversized binding before it allocates anything, with a
      // message naming the limit. That is this device declining the circuit, which the
      // pre-flight `plan()` normally catches first; reaching it here means the device
      // granted less than it advertised, or some other buffer was the one that did not fit.
      // Either way it is a capacity fact, not a failure, and a red error row would say the
      // wrong thing about a prover that behaved correctly.
      if (/over the \d+ byte (storage binding|buffer) limit/.test(msg)) {
        row.status = 'skipped';
        row.note = `could not run on this device: ${msg}`;
      } else {
        row.status = 'error';
        row.error = msg;
      }
    } finally {
      this.inFlight = null;
      // Always, including after a failure. Otherwise the key that just failed is still
      // resident when the next circuit allocates its own and the whole run dies at a
      // circuit that would have been fine on its own.
      try {
        await p.unload();
      } catch {
        /* the worker is already gone; nothing left to free */
      }
    }
  }

  /// Runs one awaited step with its estimated cost declared, so the ETA can keep counting
  /// down while the step is in progress instead of freezing until it returns.
  private async timed<T>(estMs: number, f: () => Promise<T>): Promise<T> {
    this.inFlight = { estMs, startedAt: performance.now() };
    try {
      return await f();
    } finally {
      this.inFlight = null;
    }
  }

  /// Credit for the operation currently being awaited, in milliseconds.
  ///
  /// Asymptotic rather than clamped at the estimate. A hard clamp is the honest answer to
  /// "how much of this step is done" once the step has outlasted its prediction, but it
  /// means the countdown and the ring both stop dead, and a prediction being wrong is not
  /// rare: snarkjs saturates every core, so a machine with anything else running takes
  /// several times the reference and the very first circuit is estimated before anything
  /// has been measured on it. This curve credits the step at real time to begin with and
  /// then ever more slowly, approaching the estimate without reaching it, so an overrun
  /// crawls instead of freezing and can never claim more than the step is worth.
  private inFlightCredit() {
    const f = this.inFlight;
    if (!f || f.estMs <= 0) return 0;
    return f.estMs * (1 - Math.exp(-(performance.now() - f.startedAt) / f.estMs));
  }

  /// The whole run priced in milliseconds, split into what is behind us and what is not.
  ///
  /// Both halves are priced with the estimates as they stand right now. That is the point:
  /// when the estimator re-anchors to this machine partway through, the two move together
  /// and the ring stays where it is, instead of lurching because the denominator changed
  /// under it. It is also why the ring is drawn from this rather than from the ETA. An ETA
  /// can go up, and a progress ring driven off one goes backwards when it does, which reads
  /// as the run losing work it had already done.
  private work() {
    let done = 0;
    let total = 0;
    const all = this.reps + this.warmup;
    for (const row of this.rows) {
      if (row.status === 'skipped') continue;
      const c = row.circuit;
      const dl = this.est.downloadMs(c.zkeyBytes + c.wtnsBytes);
      const prep = this.est.prepareMs(c);
      const gpu = this.est.proveMs('webgpu', c) * all;
      const sj = this.est.proveMs('snarkjs', c) * all;
      total += dl + prep + gpu + sj;
      if (row.status === 'done' || row.status === 'error') {
        // Priced at the prediction, not at what it actually cost, so the two sides of the
        // ratio stay in the same units.
        done += dl + prep + gpu + sj;
        continue;
      }
      done += this.est.downloadMs(Math.min(row.downloaded, c.zkeyBytes + c.wtnsBytes));
      if (row.prepareMs != null) done += prep;
      done += this.est.proveMs('webgpu', c) * Math.min(all, row.webgpuDone ?? 0);
      done += this.est.proveMs('snarkjs', c) * Math.min(all, row.snarkjsDone ?? 0);
    }
    done = Math.min(total, done + this.inFlightCredit());
    return { done, total };
  }

  /// Everything still to do, in milliseconds, and how much of the run is behind us.
  ///
  /// Called on a 200 ms clock as well as on every event, which is the only reason either
  /// number moves during a proof. The clock is why everything counted here is counted from
  /// state a timer can re-read: bytes downloaded, reps finished, and the elapsed part of
  /// the one operation currently being awaited.
  private recomputeEta() {
    const { done, total } = this.work();
    this.etaMs = Math.max(0, total - done);
    this.progress = total > 0 ? Math.min(1, done / total) : 0;
    this.calibrated = this.est.calibrated;
  }

  /// Median of the per-circuit ratios, over the circuits that finished. Not the ratio of the
  /// sums: that would let the single largest circuit decide the headline, which is a
  /// different claim from "how much faster is this, typically".
  get headline() {
    const rs = this.rows
      .filter((r) => r.status === 'done' && r.webgpuMs && r.snarkjsMs)
      .map((r) => r.snarkjsMs! / r.webgpuMs!);
    return rs.length ? median(rs) : null;
  }

  get totals() {
    const done = this.rows.filter((r) => r.status === 'done');
    return {
      snarkjs: done.reduce((n, r) => n + (r.snarkjsMs ?? 0), 0),
      webgpu: done.reduce((n, r) => n + (r.webgpuMs ?? 0), 0),
      n: done.length
    };
  }

  get elapsedMs() {
    return this.startedAt ? performance.now() - this.startedAt : 0;
  }
}

/// The adapter's own description of itself. `wgpu::AdapterInfo` comes back empty on WebGPU,
/// so without this a result cannot name the GPU that produced it.
async function adapterInfo() {
  if (!navigator.gpu) return null;
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!a) return null;
  const i: any = (a as any).info ?? {};
  return {
    vendor: i.vendor ?? '',
    architecture: i.architecture ?? '',
    device: i.device ?? '',
    description: i.description ?? '',
    maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
    maxBufferSize: a.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage
  };
}

export { CIRCUITS };
