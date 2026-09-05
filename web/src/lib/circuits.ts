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
  /// Set when the prover is known to refuse the circuit, with the reason. The row is still
  /// listed: a prover's ceiling is a result, and hiding it is how a benchmark flatters
  /// itself. Nothing here runs by default.
  refuses?: string;
};

export const CIRCUITS: Circuit[] = [
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
    // 1,101,048 wires. The G2 base vector alone is 128 bytes per wire, so 140,934,144 bytes
    // against WebGPU's guaranteed 134,217,728 byte storage-binding limit: the backend
    // refuses it before it touches the GPU, on every browser that only offers the floor.
    // Chunked base bindings would fix it and are not written yet.
    name: 'anon-aadhaar',
    label: 'Anon Aadhaar',
    blurb: 'India’s identity circuit. 1.1M constraints and a 631 MB proving key.',
    constraints: 1115080,
    wires: 1101048,
    zkeyBytes: 631413453,
    wtnsBytes: 35233612,
    refuses:
      'its G2 bases need 134.4 MB in one binding, over the 128 MB WebGPU guarantees; needs chunked bindings'
  }
];

/// What a run covers by default: everything the prover will actually accept, which today is
/// the four that fit inside the floor limits. 262 MB of downloads and a couple of minutes.
/// `?circuits=a,b` overrides it, and `?all=1` adds the ones expected to refuse.
export function selectCircuits(search: string): Circuit[] {
  const q = new URLSearchParams(search);
  const only = q.get('circuits');
  if (only) {
    const want = only.split(',').map((s) => s.trim());
    return CIRCUITS.filter((c) => want.includes(c.name));
  }
  if (q.get('all') === '1') return CIRCUITS;
  return CIRCUITS.filter((c) => !c.refuses);
}

export const totalBytes = (cs: Circuit[]) =>
  cs.reduce((n, c) => n + c.zkeyBytes + c.wtnsBytes, 0);
