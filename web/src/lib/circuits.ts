/// The circuit ladder, smallest first.
///
/// Only circuits that are actually in the artifact bucket are here. The synthetic `js_*`
/// ladder the native harness uses is local-only, and `railgun-*` was never uploaded, so
/// neither can be part of a test a stranger's browser runs.
///
/// `constraints` and `wires` are snarkjs' own counts from each artifact's `r1cs-info.txt`.
/// `zkeyBytes` and `wtnsBytes` are the S3 objects' `Content-Length`, recorded here rather
/// than discovered with a HEAD request so the page can show a total download size and an
/// ETA before it has made a single request.

export type Circuit = {
  name: string;
  label: string;
  blurb: string;
  constraints: number;
  wires: number;
  zkeyBytes: number;
  wtnsBytes: number;
};

/// The storage-binding size every WebGPU implementation guarantees, and the only size that
/// may be assumed before an adapter has answered. 128 MiB.
export const FLOOR_STORAGE_BINDING = 128 * 1024 * 1024;

/// Bytes of one G2 base point: `(x, y)` with both in Fq2, so four Fq, each a 254-bit number
/// stored in 32 bytes.
const G2_POINT_BYTES = 128;

/// The largest single storage binding a circuit needs, which is stage 6's G2 base vector:
/// one point per wire, contiguous, bound whole.
///
/// This is the ceiling that decides whether a device can prove a circuit at all, and it is
/// computed rather than recorded per circuit so a new row cannot be added with a stale
/// answer. The G1 bases are half the size per point and never the binding that overflows.
export const g2BindingBytes = (c: Circuit) => c.wires * G2_POINT_BYTES;

/// Why this device cannot prove this circuit, or `null` if it can.
///
/// `granted` is the adapter's own `maxStorageBufferBindingSize`, not the WebGPU floor. The
/// prover opens its device at the `auto` profile, which asks for exactly that number, so a
/// GPU with headroom gets to use it. A stock floor-only device gets the 128 MiB the
/// specification guarantees and the largest circuits are simply out of reach there, which is
/// a fact about the device and is reported as one.
export function cannotRun(c: Circuit, granted: number): string | null {
  const need = g2BindingBytes(c);
  if (need <= granted) return null;
  const mib = (n: number) => `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return (
    `needs ${mib(need)} in one GPU buffer binding for its ${c.wires.toLocaleString()} G2 ` +
    `base points, and this device grants ${mib(granted)}`
  );
}

export const CIRCUITS: Circuit[] = [
  {
    name: 'railgun-01x01',
    label: 'Railgun 1x1',
    blurb: 'A private transfer with one input note and one output. The small end.',
    constraints: 20135,
    wires: 20154,
    zkeyBytes: 10010885,
    wtnsBytes: 645004
  },
  {
    name: 'tornado',
    label: 'Tornado Cash',
    blurb: 'A Merkle membership proof, the shape most mixers and airdrops use.',
    constraints: 28275,
    wires: 28300,
    zkeyBytes: 15025241,
    wtnsBytes: 905676
  },
  {
    name: 'sha256',
    label: 'SHA-256',
    blurb: 'One SHA-256 compression, proved in-circuit. Bit-heavy and unfriendly.',
    constraints: 59281,
    wires: 59170,
    zkeyBytes: 33867697,
    wtnsBytes: 1893516
  },
  {
    name: 'railgun-13x01',
    label: 'Railgun 13x1',
    blurb: 'The same private transfer with thirteen input notes. Five times the work.',
    constraints: 141276,
    wires: 141499,
    zkeyBytes: 68499861,
    wtnsBytes: 4528044
  },
  {
    name: 'rsa2048',
    label: 'RSA-2048',
    blurb: 'A 2048-bit RSA signature check, as used for passport and email proofs.',
    constraints: 190945,
    wires: 190035,
    zkeyBytes: 103530421,
    wtnsBytes: 6081196
  },
  {
    name: 'keccak256',
    label: 'Keccak-256',
    blurb: "Ethereum's hash, the expensive one. 239k constraints of bit twiddling.",
    constraints: 239176,
    wires: 240257,
    zkeyBytes: 108112489,
    wtnsBytes: 7688300
  },
  {
    // 1,101,048 wires, so 140,934,144 bytes of G2 bases against WebGPU's guaranteed
    // 134,217,728 byte storage binding: 5% too big for a device that only offers the floor,
    // and comfortable on one that offers more. It proves in about 1.8 s on an M2 Max, whose
    // adapter grants 4 GiB. `cannotRun` decides per device rather than per circuit, which is
    // why there is no hardcoded refusal here any more.
    //
    // Chunked base bindings would put it in reach of floor-only devices too, and are not
    // written yet.
    name: 'anon-aadhaar',
    label: 'Anon Aadhaar',
    blurb: 'India\u2019s identity circuit. 1.1M constraints and a 631 MB proving key.',
    constraints: 1115080,
    wires: 1101048,
    zkeyBytes: 631413453,
    wtnsBytes: 35233612
  }
];

/// Every circuit, including any this device cannot hold. Those are shown as a skipped row
/// rather than filtered out, which costs nothing (a skipped row downloads nothing) and keeps
/// the prover's ceiling on the page. `?circuits=a,b` narrows it.
///
/// A refusal is a result. Dropping the row would leave a page whose largest circuit is the
/// largest one that happens to work, which is how a benchmark ends up flattering itself
/// without anybody deciding to.
export function selectCircuits(search: string): Circuit[] {
  const only = new URLSearchParams(search).get('circuits');
  if (!only) return CIRCUITS;
  const want = only.split(',').map((s) => s.trim());
  return CIRCUITS.filter((c) => want.includes(c.name));
}

/// What a run will actually fetch on a device with this binding limit. The rows that cannot
/// run are excluded, so the figure on the button is what the visitor is agreeing to download
/// and not a number that includes a 631 MB key nothing will ask for.
export const totalBytes = (cs: Circuit[], granted: number) =>
  cs
    .filter((c) => !cannotRun(c, granted))
    .reduce((n, c) => n + c.zkeyBytes + c.wtnsBytes, 0);
