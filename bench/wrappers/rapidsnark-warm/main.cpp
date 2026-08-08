// Warm/steady-state rapidsnark: load the zkey ONCE, then prove N times.
//
//     rapidsnark-warm <zkey> <wtns> <proof.json> <public.json> <reps>
//
// This exists because rapidsnark's stock CLI cannot measure warm proving. Its
// `groth16_prover` entry point takes the raw zkey BUFFER on every call and
// parses it internally, so wrapping that in a loop would re-pay the setup cost
// each iteration and measure nothing new. The library's real warm path is the
// object API used by proverServer: groth16_prover_create parses the zkey once
// into a prover object, and groth16_prover_prove is then callable repeatedly.
//
// Nothing about the proving algorithm is modified. The only change from the
// stock CLI is which of rapidsnark's two published entry points is used, so
// that the zkey parse lands outside the timed region instead of inside it.
//
// Emits one line per rep: "rep <i> <ms>", and writes the last proof out so the
// result stays verifiable by snarkjs.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fileloader.hpp"
#include "prover.h"

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: rapidsnark-warm <zkey> <wtns> <proof.json> "
                     "<public.json> <reps>\n";
        return EXIT_FAILURE;
    }
    const std::string zkeyFilename = argv[1];
    const std::string wtnsFilename = argv[2];
    const int reps = std::atoi(argv[5]);

    try {
        BinFileUtils::FileLoader zkeyFile(zkeyFilename);
        BinFileUtils::FileLoader wtnsFile(wtnsFilename);
        char errorMsg[1024];

        unsigned long long publicSize = 0, proofSize = 0;
        if (groth16_public_size_for_zkey_buf(zkeyFile.dataBuffer(),
                                             zkeyFile.dataSize(), &publicSize,
                                             errorMsg, sizeof(errorMsg)) != PROVER_OK)
            throw std::runtime_error(errorMsg);
        groth16_proof_size(&proofSize);

        std::vector<char> publicBuffer(publicSize), proofBuffer(proofSize);

        // --- setup, deliberately OUTSIDE the timed region ---
        void *prover = nullptr;
        if (groth16_prover_create(&prover, zkeyFile.dataBuffer(),
                                  zkeyFile.dataSize(), errorMsg,
                                  sizeof(errorMsg)) != PROVER_OK)
            throw std::runtime_error(errorMsg);

        for (int i = 1; i <= reps; i++) {
            unsigned long long ps = proofSize, us = publicSize;
            auto t0 = std::chrono::steady_clock::now();
            int rc = groth16_prover_prove(prover, wtnsFile.dataBuffer(),
                                          wtnsFile.dataSize(), proofBuffer.data(),
                                          &ps, publicBuffer.data(), &us, errorMsg,
                                          sizeof(errorMsg));
            auto t1 = std::chrono::steady_clock::now();
            if (rc != PROVER_OK) throw std::runtime_error(errorMsg);
            printf("rep %d %lld\n", i,
                   (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                       t1 - t0).count());
            if (i == reps) {
                std::ofstream(argv[3]).write(proofBuffer.data(), ps);
                std::ofstream(argv[4]).write(publicBuffer.data(), us);
            }
        }
        groth16_prover_destroy(prover);
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
