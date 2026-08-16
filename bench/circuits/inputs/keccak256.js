// Emit input.json for keccak256.circom.
//
// Bit order is the part that silently produces a wrong-but-satisfiable witness
// if you get it wrong: the circuit wants LSB-first WITHIN each byte, with
// bytes in normal message order. That is what vocdoni's own test helper
// bytesToBits does, and it is the opposite of the MSB-first convention
// circomlib's Sha256 uses, so the two circuits in this bench suite disagree
// on purpose.
const fs = require("fs");

const N_BYTES = 135;

// A fixed, boring message. Benchmark inputs must be deterministic: a random
// witness would make constraint counts stable but timings jitter with the
// sparsity of the witness, and this prover's MSM skips zero scalars.
const msg = Buffer.alloc(N_BYTES);
for (let i = 0; i < N_BYTES; i++) msg[i] = (i * 7 + 13) & 0xff;

const bits = [];
for (const byte of msg) {
  for (let j = 0; j < 8; j++) bits.push((byte >> j) & 1);
}

if (bits.length !== N_BYTES * 8) throw new Error("bit length mismatch");
fs.writeFileSync(process.argv[2] || "input.json", JSON.stringify({ in: bits }));
console.log(`wrote ${bits.length} bits for a ${N_BYTES}-byte message`);
