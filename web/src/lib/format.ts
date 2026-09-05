export function ms(x: number | undefined | null): string {
  if (x == null || !isFinite(x)) return '—';
  if (x < 1000) return `${Math.round(x)} ms`;
  if (x < 10_000) return `${(x / 1000).toFixed(2)} s`;
  return `${(x / 1000).toFixed(1)} s`;
}

export function duration(x: number): string {
  if (!isFinite(x) || x < 0) return '—';
  const s = Math.round(x / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export const count = (n: number) => n.toLocaleString('en-US');

/// Median, and never the mean. The mean of a run that contained a garbage collection is a
/// number about the garbage collector.
export function median(xs: number[]): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

/// "4.6x faster" / "1.4x slower", from snarkjs' time over ours.
export function ratioLabel(r: number): { x: string; faster: boolean } {
  return r >= 1
    ? { x: `${r.toFixed(r < 10 ? 1 : 0)}x faster`, faster: true }
    : { x: `${(1 / r).toFixed(1)}x slower`, faster: false };
}
