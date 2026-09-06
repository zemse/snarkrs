/// snarkjs, timed the only honest way.
///
/// Four rules, each of which was a wrong number first.
///
/// 1. `groth16.prove(zkeyU8, wtnsU8)`, never `fullProve`. Witness generation is CPU-only for
///    both provers (it is stage -1 and will never be on a GPU) and folding it in adds a cost
///    neither prover can act on.
/// 2. Both arguments are `Uint8Array` and never a URL. Hand `fastfile.readExisting` a string
///    and it will `fetch` inside the timed call, so the row becomes a bandwidth measurement.
/// 3. The worker count is read back after a prove and refused if it is 1 on a multi-core
///    machine. A bundler that resolves snarkjs' `node` export condition yields a silently
///    single-threaded prover and a speedup of ours that is entirely fake. The vendored UMD
///    bundle removes the cause; this catches it anyway.
/// 4. Every proof is verified before its time is recorded. A timing from an unverified proof
///    is not a timing.
///
/// The hidden-tab rule is rule 5 and lives in `visibility.ts`, because the GPU side has to
/// follow exactly the same one for the two columns to be comparable.

import { visible, Discards } from './visibility';

declare const snarkjs: any;

export type SnarkjsRun = {
  ms: number[];
  workers: number | null;
  proof: unknown;
  publicSignals: unknown;
  warning?: string;
};

export async function snarkjsProve(
  zkey: Uint8Array,
  wtns: Uint8Array,
  vkey: unknown,
  reps: number,
  warmup: number,
  onRep: (i: number, ms: number) => void
): Promise<SnarkjsRun> {
  if (typeof snarkjs === 'undefined') {
    throw new Error('snarkjs did not load; static/vendor/snarkjs.min.js is missing');
  }

  let last: any = null;
  for (let i = 0; i < warmup; i++) {
    last = await snarkjs.groth16.prove(zkey, wtns);
    if (i === 0 && !(await snarkjs.groth16.verify(vkey, last.publicSignals, last.proof))) {
      throw new Error('snarkjs did not verify its own proof');
    }
  }

  // Read the thread manager only after a prove, because that is when it exists.
  let workers: number | null = null;
  try {
    workers = (await snarkjs.curves.getCurveFromName('bn128'))?.tm?.concurrency ?? null;
  } catch {
    workers = null;
  }

  const ms: number[] = [];
  const discards = new Discards(reps + 5);
  // `i` advances only on a rep that counted, so a discarded one is re-taken rather than lost.
  // See `visibility.ts` for why a hidden tab invalidates a measurement, and for what this
  // loop used to do instead.
  for (let i = 0; i < reps; ) {
    await visible();
    const t0 = performance.now();
    const r = await snarkjs.groth16.prove(zkey, wtns);
    const t1 = performance.now();
    if (document.visibilityState !== 'visible') {
      discards.count('snarkjs');
      continue;
    }
    if (!(await snarkjs.groth16.verify(vkey, r.publicSignals, r.proof))) {
      throw new Error(`snarkjs rep ${i} produced a proof that does not verify`);
    }
    last = r;
    ms.push(t1 - t0);
    onRep(i, t1 - t0);
    i++;
  }

  return {
    ms,
    workers,
    proof: last?.proof,
    publicSignals: last?.publicSignals,
    warning:
      workers === 1 && navigator.hardwareConcurrency > 1
        ? 'snarkjs ran single-threaded on a multi-core machine, so this baseline is invalid'
        : undefined
  };
}

/// Our proof, checked by snarkjs' verifier. The other direction is in the worker. Both run,
/// because a prover that agrees only with itself has proved nothing.
export async function snarkjsVerify(vkey: unknown, publicSignals: unknown, proof: unknown) {
  return (await snarkjs.groth16.verify(vkey, publicSignals, proof)) as boolean;
}
