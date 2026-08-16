pragma circom 2.1.4;

include "keccak256-circom/circuits/keccak.circom";

// Ethereum's keccak256 (legacy 0x01 domain byte, not NIST SHA3) over a single
// absorption block. This is the hash every EVM circuit ends up paying for, so
// it is the one worth benchmarking.
//
// 135 bytes, not 136. The rate of keccak-f[1600] is 136 bytes, but vocdoni's
// Pad() writes its 1-byte domain marker at out2[nBits .. nBits+7] inside a
// fixed 1088-bit block, so a full 136-byte message would index past the end.
// 135 bytes is the largest input this implementation absorbs in one
// permutation, and one permutation is what we want to measure: there is no
// multi-block path in this circuit.
component main = Keccak(135 * 8, 256);
