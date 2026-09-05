/// The state machine behind the button.
///
/// For each circuit, smallest first: download the proving key and witness, prove it on the
/// GPU, then prove the same thing with snarkjs, cross-verify both proofs in both directions,
/// publish the row, drop everything, move on. Partial results are visible the whole way
/// through, because at real circuit sizes the last row can be two minutes after the first
/// and a page that shows nothing until the end looks broken.

import { CIRCUITS, selectCircuits, totalBytes, type Circuit } from './circuits';
import { Estimator } from './estimate';
import { Prover } from './prover';
import { snarkjsProve, snarkjsVerify } from './snarkjs';
import { median } from './format';

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
      status: circuit.refuses ? ('skipped' as const) : ('pending' as const),
      downloaded: 0,
      note: circuit.refuses
    }))
  );
  phase = $state<'idle' | 'starting' | 'running' | 'done' | 'error'>('idle');
  /// Bumped on every start. The page watches it to reset the ring's scale, which is
  /// anchored to the largest ETA seen and would otherwise carry over from the last run.
  runId = $state(0);
  /// One line under the ring saying what is happening right now.
  activity = $state('');
  /// 0..1 for the current sub-step, or null when there is nothing meaningful to show.
  subProgress = $state<number | null>(null);
  etaMs = $state<number | null>(null);
  calibrated = $state(false);
  mbits = $state(0);
  env = $state<any>(null);
  fatal = $state<string | null>(null);

  private est = new Estimator();
  private prover: Prover | null = null;
  private cancelled = false;
  private startedAt = 0;

  /// Downloads and proofs are timed for real; the ETA has to guess. Both curves live in
  /// Estimator and both get re-anchored to this machine as measurements come in.
  private readonly reps = param('reps', 3);
  /// Three, not one. The first `prove()` on a page builds snarkjs' BN254 module with
  /// wasmbuilder, compiles it again inside each of the twelve workers it spawns, and runs
  /// while the JIT is still tiering up: on a cold page with one warm-up the first circuit
  /// came out at 4.09 s, against 1.03 s for the same circuit once the page was warm. Three
  /// untimed reps is what the native harness settled on for the same reason.
  private readonly warmup = param('warmup', 3);
  private readonly profile = new URLSearchParams(q()).get('profile') ?? 'floor';
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
    return totalBytes(this.circuits);
  }
  /// How many circuits will actually be proved. Not `circuits.length`, which counts the
  /// listed-but-refused rows too, and would promise work the run is not going to do.
  get willRun() {
    return this.circuits.filter((c) => !c.refuses).length;
  }

  async start() {
    if (this.phase === 'running' || this.phase === 'starting') return;
    this.cancelled = false;
    this.fatal = null;
    this.phase = 'starting';
    this.runId++;
    // A handle for reading the raw per-rep timings out of the console. The page shows a
    // median and a spread; this is how the rep counts in `reps`/`warmup` were chosen, and
    // how they should be re-chosen on a machine that behaves differently.
    (globalThis as Record<string, unknown>).__g16run = this;
    this.startedAt = performance.now();
    // Reset rather than rebuild, so a second run reuses the same objects and the table does
    // not flash empty between runs.
    for (const r of this.rows) {
      r.status = r.circuit.refuses ? 'skipped' : 'pending';
      r.downloaded = 0;
      r.downloadMs = r.prepareMs = r.webgpuMs = r.snarkjsMs = undefined;
      r.crossVerified = undefined;
      r.webgpuReps = r.snarkjsReps = undefined;
      r.stages = undefined;
      r.error = undefined;
      r.note = r.circuit.refuses;
    }
    this.recomputeEta();

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
      this.subProgress = null;
      this.etaMs = null;
    } catch (e: any) {
      this.phase = 'error';
      this.fatal = String(e?.message ?? e);
      // Reported too, and not only on the happy path. The failure worth reading is usually
      // the one that stopped the run before it produced a row: a device that refused to
      // open, or a GPU self-test that refused to let it prove. Without this, `?report=1`
      // posts nothing at all and the log looks like the page never ran.
      await this.report();
    } finally {
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
        this.subProgress = done / (c.zkeyBytes + c.wtnsBytes);
        this.recomputeEta();
      });
      row.downloaded = loaded.downloadedBytes;
      row.downloadMs = loaded.downloadMs;
      this.est.observeDownload(loaded.downloadedBytes, loaded.downloadMs);
      this.mbits = this.est.mbitsPerSecond;

      // ---- our side first, as asked. -------------------------------------------------
      row.status = 'webgpu';
      this.subProgress = null;
      this.activity = `${c.label}: uploading ${(c.zkeyBytes / 1024 ** 2).toFixed(0)} MB to the GPU`;
      const prep = await p.prepare();
      row.prepareMs = prep.wallMs;

      const ours: number[] = [];
      let ourProof: any = null;
      for (let i = 0; i < this.warmup + this.reps; i++) {
        if (document.visibilityState !== 'visible') {
          // Same rule the snarkjs side follows: a rep spanning a hidden tab is measured
          // against a clock Chrome has throttled to 1 Hz, so it is dropped, not recorded.
          await visible();
        }
        this.activity = `${c.label}: proving on the GPU (${Math.max(1, i - this.warmup + 1)}/${this.reps})`;
        this.subProgress = i / (this.warmup + this.reps);
        const r = await p.prove();
        ourProof = r;
        if (i >= this.warmup) ours.push(r.wallMs);
      }
      row.webgpuReps = ours;
      row.stages = ourProof?.timings;
      row.webgpuMs = median(ours);
      this.est.observeProof('webgpu', c, row.webgpuMs);
      this.recomputeEta();

      // ---- then snarkjs, on the identical bytes. ---------------------------------------
      row.status = 'snarkjs';
      this.subProgress = null;
      this.activity = `${c.label}: proving with snarkjs`;
      const sj = await snarkjsProve(
        loaded.zkeyBytes,
        loaded.wtnsBytes,
        loaded.vkey,
        this.reps,
        this.warmup,
        (i) => {
          this.subProgress = (i + 1) / this.reps;
          this.activity = `${c.label}: proving with snarkjs (${i + 1}/${this.reps})`;
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
      row.status = 'error';
      row.error = String(e?.message ?? e);
    } finally {
      this.subProgress = null;
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

  /// Everything still to do, in milliseconds: the downloads not yet fetched plus the
  /// proving not yet run, for every circuit that has not finished.
  private recomputeEta() {
    let total = 0;
    for (const row of this.rows) {
      if (row.status === 'done' || row.status === 'skipped' || row.status === 'error') continue;
      const c = row.circuit;
      const remaining = Math.max(0, c.zkeyBytes + c.wtnsBytes - row.downloaded);
      total += this.est.downloadMs(remaining);
      if (row.prepareMs == null) total += this.est.prepareMs(c);
      if (row.webgpuMs == null) {
        total += this.est.proveMs('webgpu', c) * (this.reps + this.warmup);
      }
      if (row.snarkjsMs == null) {
        total += this.est.proveMs('snarkjs', c) * (this.reps + this.warmup);
      }
    }
    this.etaMs = total;
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

function visible(): Promise<void> {
  if (document.visibilityState === 'visible') return Promise.resolve();
  return new Promise((res) => {
    const on = () => {
      if (document.visibilityState === 'visible') {
        document.removeEventListener('visibilitychange', on);
        res();
      }
    };
    document.addEventListener('visibilitychange', on);
  });
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
