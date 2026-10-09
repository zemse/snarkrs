// Warm/steady-state rapidsnark: load the zkey ONCE, then prove N times.
//
//     rapidsnark-warm <zkey> <wtns> <proof.json> <public.json> <reps> <vkey.json>
//                    [--proof-dir DIR]
//
// This exists because rapidsnark's stock CLI cannot measure warm proving. Its
// `groth16_prover` entry point takes the raw zkey BUFFER on every call and
// parses it internally, so wrapping that in a loop would re-pay the setup cost
// each iteration and measure nothing new. The library's real warm path is the
// object API used by proverServer: groth16_prover_create parses the zkey once
// into a prover object, and groth16_prover_prove is then callable repeatedly.
//
// Nothing about the proving algorithm is modified. The object API keeps the
// zkey parse outside the timed region instead of inside it.
//
// Verifies every proof outside the timed region before emitting "rep <i> <ms>".
// Writes the last proof and optional per-repetition proofs for independent checks.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "fileloader.hpp"
#include "misc.hpp"
#include "prover.h"
#include "verifier.h"

void write_json(const std::filesystem::path &path, const std::vector<char> &buffer,
                unsigned long long size) {
    std::ofstream file(path);
    file.write(buffer.data(), size);
    file.close();
    if (!file)
        throw std::runtime_error("cannot write JSON output: " + path.string());
}

int main(int argc, char **argv) {
    if ((argc != 7 && argc != 9) ||
        (argc == 9 && (std::string(argv[7]) != "--proof-dir" || !argv[8][0]))) {
        std::cerr << "Usage: rapidsnark-warm <zkey> <wtns> <proof.json> "
                     "<public.json> <reps> <vkey.json> [--proof-dir DIR]\n";
        return EXIT_FAILURE;
    }
    const std::string zkeyFilename = argv[1];
    const std::string wtnsFilename = argv[2];

    try {
        size_t consumed = 0;
        const int reps = std::stoi(argv[5], &consumed);
        if (reps <= 0 || consumed != std::string(argv[5]).size())
            throw std::runtime_error("reps must be a positive integer");
        if (std::filesystem::weakly_canonical(argv[3]) ==
            std::filesystem::weakly_canonical(argv[4]))
            throw std::runtime_error("proof and public outputs must differ");
        std::filesystem::path proofDir;
        if (argc == 9) {
            proofDir = argv[8];
            std::filesystem::create_directories(proofDir);
            if (!std::filesystem::is_directory(proofDir))
                throw std::runtime_error("proof-dir must be a directory");
            for (int i = 0; i < reps; i++) {
                for (const auto &name : {"proof-", "public-"}) {
                    auto output = proofDir /
                                  (std::string(name) + std::to_string(i + 1) + ".json");
                    if (std::filesystem::weakly_canonical(output) ==
                            std::filesystem::weakly_canonical(argv[3]) ||
                        std::filesystem::weakly_canonical(output) ==
                            std::filesystem::weakly_canonical(argv[4]))
                        throw std::runtime_error(
                            "final outputs must not overwrite repetition outputs");
                }
            }
        }
        std::ifstream vkeyFile(argv[6]);
        if (!vkeyFile)
            throw std::runtime_error("cannot open verification key");
        const std::string vkey((std::istreambuf_iterator<char>(vkeyFile)),
                               std::istreambuf_iterator<char>());
        if (vkey.empty() || vkeyFile.bad())
            throw std::runtime_error("cannot read verification key");
        BinFileUtils::FileLoader zkeyFile(zkeyFilename);
        BinFileUtils::FileLoader wtnsFile(wtnsFilename);
        char errorMsg[1024] = {};

        unsigned long long publicSize = 0, proofSize = 0;
        if (groth16_public_size_for_zkey_buf(zkeyFile.dataBuffer(), zkeyFile.dataSize(),
                                             &publicSize, errorMsg,
                                             sizeof(errorMsg)) != PROVER_OK)
            throw std::runtime_error(errorMsg);
        groth16_proof_size(&proofSize);

        if (!publicSize || !proofSize)
            throw std::runtime_error("invalid prover buffer size");
        std::vector<char> publicBuffer(publicSize + 1), proofBuffer(proofSize + 1);

        // Setup stays outside the timed region.
        void *prover = nullptr;
        if (groth16_prover_create(&prover, zkeyFile.dataBuffer(), zkeyFile.dataSize(),
                                  errorMsg, sizeof(errorMsg)) != PROVER_OK)
            throw std::runtime_error(errorMsg);
        std::unique_ptr<void, decltype(&groth16_prover_destroy)> owner(
            prover, &groth16_prover_destroy);
        std::cerr << "ffiasm_pool_threads="
                  << ThreadPool::defaultPool().getThreadCount() << '\n';

        for (int i = 0; i < reps; i++) {
            unsigned long long ps = proofSize, us = publicSize;
            auto t0 = std::chrono::steady_clock::now();
            int rc = groth16_prover_prove(
                prover, wtnsFile.dataBuffer(), wtnsFile.dataSize(), proofBuffer.data(),
                &ps, publicBuffer.data(), &us, errorMsg, sizeof(errorMsg));
            auto t1 = std::chrono::steady_clock::now();
            if (rc != PROVER_OK)
                throw std::runtime_error(errorMsg);
            if (ps > proofSize || us > publicSize)
                throw std::runtime_error("invalid prover output size");
            proofBuffer[ps] = '\0';
            publicBuffer[us] = '\0';
            errorMsg[0] = '\0';
            if (groth16_verify(proofBuffer.data(), publicBuffer.data(), vkey.c_str(),
                               errorMsg, sizeof(errorMsg)) != VERIFIER_VALID_PROOF)
                throw std::runtime_error(std::string("verification failed: ") +
                                         errorMsg);
            if (!proofDir.empty()) {
                const std::string suffix = std::to_string(i + 1) + ".json";
                write_json(proofDir / ("proof-" + suffix), proofBuffer, ps);
                write_json(proofDir / ("public-" + suffix), publicBuffer, us);
            }
            if (i + 1 == reps) {
                write_json(argv[3], proofBuffer, ps);
                write_json(argv[4], publicBuffer, us);
            }
            auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("rep %d %.6f\n", i + 1, ms);
        }
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
