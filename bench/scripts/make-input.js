#!/usr/bin/env node
// Build a VALID witness input for JoinSplit(nIns, nOuts, depth, valueBits).
// Everything must actually be consistent: all input notes live in one Merkle
// tree under a single root, nullifiers derive from the real nullifying key,
// and value is conserved. An inconsistent input fails witness generation, so
// this is also the correctness check on the circuit itself.
//
// usage: node make-input.js <nIns> <nOuts> <depth> [valueBits]
const { buildPoseidon } = require('circomlibjs');

async function main() {
  const nIns  = parseInt(process.argv[2], 10);
  const nOuts = parseInt(process.argv[3], 10);
  const depth = parseInt(process.argv[4], 10);

  const poseidon = await buildPoseidon();
  const F = poseidon.F;
  const H = (xs) => F.toObject(poseidon(xs));

  const spendingKey = 12345678901234567890n;
  const tokenId     = 1n;

  const nk  = H([spendingKey]);
  const npk = H([nk, spendingKey]);

  // input notes at leaf indices 0..nIns-1, values 1000, 2000, ...
  const inValue = [], inRandom = [], inLeafIndex = [], commitments = [], nullifiers = [];
  for (let i = 0; i < nIns; i++) {
    const value = BigInt((i + 1) * 1000);
    const rand  = BigInt(1000000 + i);
    inValue.push(value);
    inRandom.push(rand);
    inLeafIndex.push(BigInt(i));
    commitments.push(H([npk, tokenId, value, rand]));
    nullifiers.push(H([nk, BigInt(i)]));
  }

  // zero hashes per level, so a depth-32 tree with a handful of leaves is cheap
  const zh = [0n];
  for (let l = 1; l <= depth; l++) zh.push(H([zh[l - 1], zh[l - 1]]));

  // sparse tree: level 0 holds the commitments, everything else is a zero subtree
  let level = new Map();
  commitments.forEach((c, i) => level.set(BigInt(i), c));
  const levels = [level];
  for (let l = 0; l < depth; l++) {
    const cur = levels[l], next = new Map();
    const parents = new Set([...cur.keys()].map((k) => k / 2n));
    for (const p of parents) {
      const left  = cur.get(2n * p)      ?? zh[l];
      const right = cur.get(2n * p + 1n) ?? zh[l];
      next.set(p, H([left, right]));
    }
    levels.push(next);
  }
  const merkleRoot = levels[depth].get(0n) ?? zh[depth];

  // sibling path and direction bits per input leaf
  const inSiblings = [], inPathBits = [];
  for (let i = 0; i < nIns; i++) {
    let idx = BigInt(i);
    const sibs = [], bits = [];
    for (let l = 0; l < depth; l++) {
      const sib = idx ^ 1n;
      sibs.push((levels[l].get(sib) ?? zh[l]).toString());
      bits.push((idx & 1n).toString());   // 0 = cur is the left child
      idx >>= 1n;
    }
    inSiblings.push(sibs);
    inPathBits.push(bits);
  }

  // conserve value: split the input total across the outputs, remainder to the last
  const total = inValue.reduce((a, b) => a + b, 0n);
  const each  = total / BigInt(nOuts);
  const outValue = Array.from({ length: nOuts }, (_, k) =>
    k === nOuts - 1 ? total - each * BigInt(nOuts - 1) : each);

  const outRandom = [], outNpk = [], outCommitments = [];
  for (let k = 0; k < nOuts; k++) {
    const rand = BigInt(2000000 + k);
    const npkOut = H([BigInt(555 + k), BigInt(777 + k)]);
    outRandom.push(rand);
    outNpk.push(npkOut);
    outCommitments.push(H([npkOut, tokenId, outValue[k], rand]));
  }

  const s = (x) => x.toString();
  console.log(JSON.stringify({
    merkleRoot: s(merkleRoot),
    nullifiers: nullifiers.map(s),
    outCommitments: outCommitments.map(s),
    tokenId: s(tokenId),
    inValue: inValue.map(s),
    inRandom: inRandom.map(s),
    inLeafIndex: inLeafIndex.map(s),
    inSiblings,
    inPathBits,
    spendingKey: s(spendingKey),
    outValue: outValue.map(s),
    outRandom: outRandom.map(s),
    outNpk: outNpk.map(s),
  }, null, 1));
}
main().catch((e) => { console.error(e); process.exit(1); });
