pragma circom 2.1.4;

include "poseidon.circom";

// Two-input Poseidon, not the distinct Poseidon2 permutation.
template PoseidonOnion(nHashes) {
    signal input seed;
    signal input salt[nHashes];
    signal output digest;

    signal h[nHashes + 1];
    component hash[nHashes];
    h[0] <== seed;
    for (var i = 0; i < nHashes; i++) {
        hash[i] = Poseidon(2);
        hash[i].inputs[0] <== h[i];
        hash[i].inputs[1] <== salt[i];
        h[i + 1] <== hash[i].out;
    }
    digest <== h[nHashes];
}
