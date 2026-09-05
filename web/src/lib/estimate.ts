/// The ETA the page shows while it works.
///
/// Two things get estimated, and they are estimated differently on purpose.
///
/// **Download** is measured, not predicted. A guess at someone's bandwidth is worthless and
/// the real number arrives within the first second of the first fetch, so the estimator
/// starts at a deliberately pessimistic default and is overwritten by observed throughput as
/// soon as there is any.
///
/// **Proving** cannot be observed before it is run, so it starts from measurements taken on
/// a reference machine and is rescaled by the first real measurement of the run.

import type { Circuit } from './circuits';

/// Median proof time in milliseconds for each circuit, on an M2 Max in Chrome 152, warm,
/// three untimed warm-ups then five timed reps. Taken with this page, on these exact
/// artifacts.
///
/// Keyed by circuit rather than fitted against constraint count, because constraint count
/// does not predict proving time and this table is the evidence. Tornado has less than half
/// Keccak's constraints and takes four times as long on snarkjs; it is slower than SHA-256,
/// which is twice its size, on **both** provers. The reason is that the cost tracks non-zero
/// entries in the A/B/C matrices, not rows of them: Tornado is Poseidon-dense, where every
/// constraint has many terms, while the hash circuits are bit operations with two or three
/// terms each. On our side there is a second effect on top, that the MSM skips zero digits,
/// so a bit-valued witness leaves most high windows empty and Keccak comes out cheaper than
/// a circuit two thirds its size.
///
/// A fitted curve through these points would be a curve through noise. An unknown circuit
/// falls back to `interpolate` below, which is honest about being a rough shape.
const REFERENCE: Record<string, { snarkjs: number; webgpu: number; prepare: number }> = {
  tornado: { snarkjs: 1030, webgpu: 334, prepare: 15 },
  sha256: { snarkjs: 890, webgpu: 149, prepare: 30 },
  rsa2048: { snarkjs: 3550, webgpu: 491, prepare: 50 },
  keccak256: { snarkjs: 3090, webgpu: 259, prepare: 46 }
};

type Point = { c: number; ms: number };

/// The fallback for a circuit with no measured row. This is the synthetic ladder the native
/// harness uses: one circuit shape at six sizes, so a curve through it is a curve in
/// constraint count and nothing else, which is exactly what makes it a poor predictor of a
/// real circuit and a reasonable last resort. Same machine and browser as REFERENCE.
const WEBGPU: Point[] = [
  { c: 3359, ms: 258.5 },
  { c: 17929, ms: 275.6 },
  { c: 70357, ms: 651.3 },
  { c: 140261, ms: 940.8 },
  { c: 1115080, ms: 2302.7 }
];

// snarkjs has no prepare(); it re-reads zkey sections 4 to 9 inside every prove(), so this
// one curve is both its cold and its warm number.
const SNARKJS: Point[] = [
  { c: 3359, ms: 183.2 },
  { c: 17929, ms: 837.8 },
  { c: 70357, ms: 2990.2 },
  { c: 140261, ms: 5767.1 },
  { c: 1115080, ms: 23642.9 }
];

/// Piecewise linear through the measured points, extrapolating along the nearest segment
/// past either end. Not a fitted polynomial: the curve has a visible kink where the domain
/// size steps to the next power of two, and a smooth fit through a step function is worse
/// than a straight line between two things that were actually measured.
function interpolate(pts: Point[], c: number): number {
  if (c <= pts[0].c) return (pts[0].ms * c) / pts[0].c;
  for (let i = 1; i < pts.length; i++) {
    if (c <= pts[i].c) {
      const a = pts[i - 1];
      const b = pts[i];
      return a.ms + ((c - a.c) / (b.c - a.c)) * (b.ms - a.ms);
    }
  }
  const a = pts[pts.length - 2];
  const b = pts[pts.length - 1];
  return b.ms + ((c - b.c) / (b.c - a.c)) * (b.ms - a.ms);
}

/// Before anything has been downloaded there is no measurement to use. 12 Mbit/s: low enough
/// that the first correction almost always revises the estimate down, which is the direction
/// a wrong ETA should be wrong in.
const DEFAULT_BYTES_PER_MS = (12 * 1_000_000) / 8 / 1000;

/// Uploading the bases for a circuit with no measured row. The measured ones range from 0.4
/// to 1.0 ms per MB with a fixed cost on top, so this is a shape rather than a law; it is
/// also the smallest term in the estimate by an order of magnitude.
const PREPARE_MS_PER_MB = 0.8;

export class Estimator {
  private bytesPerMs = DEFAULT_BYTES_PER_MS;
  private downloadObserved = false;
  /// Multiplies the reference numbers to fit the machine actually running. One scale per
  /// prover, because a fast GPU and a slow CPU are a normal combination and a single fudge
  /// factor would smear one into the other.
  private scale = { webgpu: 1, snarkjs: 1 };
  private scaled = { webgpu: 0, snarkjs: 0 };

  /// Called with each completed download. Only whole files, and only ones big enough for the
  /// number to mean something: a 241-byte public.json over a warm connection reports an
  /// absurd throughput that would wreck the estimate for the 108 MB fetch behind it.
  observeDownload(bytes: number, ms: number) {
    if (bytes < 1_000_000 || ms <= 0) return;
    const rate = bytes / ms;
    // The first real sample replaces the default outright; later ones move it gently, so one
    // slow chunk on a shared connection does not swing the whole remaining estimate.
    this.bytesPerMs = this.downloadObserved ? this.bytesPerMs * 0.6 + rate * 0.4 : rate;
    this.downloadObserved = true;
  }

  /// Called with each completed proof, to re-anchor the reference to this machine.
  observeProof(prover: 'webgpu' | 'snarkjs', circuit: Circuit, ms: number) {
    const predicted = this.reference(prover, circuit);
    if (predicted <= 0) return;
    const s = ms / predicted;
    const n = ++this.scaled[prover];
    // A running mean of the ratio: every circuit measured so far gets an equal say, rather
    // than the newest one taking over, because the thing being corrected is the machine and
    // the machine does not change between circuits.
    this.scale[prover] += (s - this.scale[prover]) / n;
  }

  private reference(prover: 'webgpu' | 'snarkjs', c: Circuit) {
    const row = REFERENCE[c.name];
    if (row) return row[prover];
    return interpolate(prover === 'webgpu' ? WEBGPU : SNARKJS, c.constraints);
  }

  downloadMs(bytes: number) {
    return bytes / this.bytesPerMs;
  }

  proveMs(prover: 'webgpu' | 'snarkjs', c: Circuit) {
    return this.reference(prover, c) * this.scale[prover];
  }

  prepareMs(c: Circuit) {
    const row = REFERENCE[c.name];
    const base = row ? row.prepare : (c.zkeyBytes / 1_000_000) * PREPARE_MS_PER_MB;
    return base * this.scale.webgpu;
  }

  /// Whether the numbers above still rest on the reference machine rather than this one. The
  /// UI says which, because an ETA computed from somebody else's GPU should not be presented
  /// with the same confidence as one computed from measurements taken here.
  get calibrated() {
    return this.downloadObserved && this.scaled.webgpu > 0;
  }

  get mbitsPerSecond() {
    return (this.bytesPerMs * 1000 * 8) / 1_000_000;
  }
}
