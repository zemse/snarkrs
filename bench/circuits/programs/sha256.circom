pragma circom 2.1.4;

include "sha256/sha256.circom";

// SHA-256 over a fixed-length message, the shape every "sha256 in a circuit"
// benchmark actually measures. circomlib's Sha256 does its own padding, so the
// message length in bits is a compile-time constant and the witness is just the
// message bits.
template Sha256Bench(nBits) {
    signal input in[nBits];
    signal output out[256];

    component sha = Sha256(nBits);
    for (var i = 0; i < nBits; i++) {
        sha.in[i] <== in[i];
    }
    for (var i = 0; i < 256; i++) {
        out[i] <== sha.out[i];
    }
}

component main = Sha256Bench(512);
