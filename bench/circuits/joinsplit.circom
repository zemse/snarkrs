pragma circom 2.2.2;

include "poseidon.circom";
include "bitify.circom";
include "comparators.circom";
include "mux1.circom";

// One level of a binary Merkle inclusion proof.
// pathBit selects whether cur sits on the left (0) or the right (1).
template MerkleLevel() {
    signal input cur;
    signal input sibling;
    signal input pathBit;
    signal output out;

    pathBit * (pathBit - 1) === 0;

    component l = Mux1();
    component r = Mux1();
    l.c[0] <== cur;      l.c[1] <== sibling;  l.s <== pathBit;
    r.c[0] <== sibling;  r.c[1] <== cur;      r.s <== pathBit;

    component h = Poseidon(2);
    h.inputs[0] <== l.out;
    h.inputs[1] <== r.out;
    out <== h.out;
}

// Merkle inclusion proof of `leaf` under `root`, depth levels.
template MerkleProof(depth) {
    signal input leaf;
    signal input root;
    signal input siblings[depth];
    signal input pathBits[depth];

    component lv[depth];
    signal cur[depth + 1];
    cur[0] <== leaf;
    for (var i = 0; i < depth; i++) {
        lv[i] = MerkleLevel();
        lv[i].cur     <== cur[i];
        lv[i].sibling <== siblings[i];
        lv[i].pathBit <== pathBits[i];
        cur[i + 1]    <== lv[i].out;
    }
    root === cur[depth];
}

// The canonical shielded-pool joinsplit, the shape shared by Railgun's
// joinsplit.circom and Zeto's nullifier circuits:
//   1. every input note is a real commitment in the tree
//   2. the spender owns it (npk derived from the spending key)
//   3. each spend publishes a deterministic, unlinkable nullifier
//   4. value is conserved across inputs and outputs
//   5. output commitments are well formed
//   6. all notes share one token id
template JoinSplit(nIns, nOuts, depth, valueBits) {
    // public
    signal input merkleRoot;
    signal input nullifiers[nIns];
    signal input outCommitments[nOuts];
    signal input tokenId;

    // private, input notes
    signal input inValue[nIns];
    signal input inRandom[nIns];
    signal input inLeafIndex[nIns];
    signal input inSiblings[nIns][depth];
    signal input inPathBits[nIns][depth];
    signal input spendingKey;

    // private, output notes
    signal input outValue[nOuts];
    signal input outRandom[nOuts];
    signal input outNpk[nOuts];

    // nullifying key and note public key both derive from the spending key,
    // so ownership and nullifier derivation share one secret.
    component nk = Poseidon(1);
    nk.inputs[0] <== spendingKey;

    component npk = Poseidon(2);
    npk.inputs[0] <== nk.out;
    npk.inputs[1] <== spendingKey;

    component inCommit[nIns];
    component inNull[nIns];
    component inProof[nIns];
    component inRange[nIns];

    var sumIn = 0;
    for (var i = 0; i < nIns; i++) {
        // commitment = Poseidon(npk, token, value, random)
        inCommit[i] = Poseidon(4);
        inCommit[i].inputs[0] <== npk.out;
        inCommit[i].inputs[1] <== tokenId;
        inCommit[i].inputs[2] <== inValue[i];
        inCommit[i].inputs[3] <== inRandom[i];

        // nullifier = Poseidon(nk, leafIndex), unlinkable to the commitment
        inNull[i] = Poseidon(2);
        inNull[i].inputs[0] <== nk.out;
        inNull[i].inputs[1] <== inLeafIndex[i];
        nullifiers[i] === inNull[i].out;

        inProof[i] = MerkleProof(depth);
        inProof[i].leaf <== inCommit[i].out;
        inProof[i].root <== merkleRoot;
        for (var j = 0; j < depth; j++) {
            inProof[i].siblings[j] <== inSiblings[i][j];
            inProof[i].pathBits[j] <== inPathBits[i][j];
        }

        // keep values in range so the balance sum cannot wrap the field
        inRange[i] = Num2Bits(valueBits);
        inRange[i].in <== inValue[i];

        sumIn += inValue[i];
    }

    component outCommit[nOuts];
    component outRange[nOuts];

    var sumOut = 0;
    for (var k = 0; k < nOuts; k++) {
        outCommit[k] = Poseidon(4);
        outCommit[k].inputs[0] <== outNpk[k];
        outCommit[k].inputs[1] <== tokenId;
        outCommit[k].inputs[2] <== outValue[k];
        outCommit[k].inputs[3] <== outRandom[k];
        outCommitments[k] === outCommit[k].out;

        outRange[k] = Num2Bits(valueBits);
        outRange[k].in <== outValue[k];

        sumOut += outValue[k];
    }

    sumIn === sumOut;
}
