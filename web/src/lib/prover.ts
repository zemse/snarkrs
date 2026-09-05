/// The main thread's handle on the prover worker.

import ProverWorker from './worker/prover?worker';

type Pending = {
  resolve: (v: any) => void;
  reject: (e: Error) => void;
  onProgress?: (p: any) => void;
};

export class Prover {
  private worker: Worker;
  private pending = new Map<number, Pending>();
  private next = 1;

  constructor() {
    this.worker = new ProverWorker();
    this.worker.onmessage = (ev: MessageEvent) => {
      const { id, ok, error, value, progress } = ev.data;
      const p = this.pending.get(id);
      if (!p) return;
      // A progress frame is not a reply; the entry stays until the real one arrives.
      if (progress !== undefined) return p.onProgress?.(progress);
      this.pending.delete(id);
      ok ? p.resolve(value) : p.reject(new Error(error));
    };
    // A module-load failure inside a worker surfaces here and nowhere else. Without this,
    // a typo in an import path is a page that sits at "starting" forever with a clean
    // console on the main thread.
    this.worker.onerror = (e) => {
      const err = new Error(`prover worker: ${e.message ?? 'failed to load'}`);
      for (const [, p] of this.pending) p.reject(err);
      this.pending.clear();
    };
  }

  private call<T = any>(type: string, args: object = {}, onProgress?: (p: any) => void) {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, type, ...args });
    });
  }

  init(pkgBase: string, profile: string) {
    return this.call('init', { pkgBase, profile });
  }
  load(base: string, name: string, onProgress: (p: any) => void) {
    return this.call('load', { base, name }, onProgress);
  }
  prepare() {
    return this.call('prepare');
  }
  prove() {
    return this.call('prove');
  }
  verify(vkey: unknown, publicSignals: unknown, proof: unknown) {
    return this.call<{ verified: boolean }>('verify', { vkey, publicSignals, proof });
  }
  unload() {
    return this.call('unload');
  }
  terminate() {
    this.worker.terminate();
  }
}
