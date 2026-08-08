# ffjavascript oracles

These three scripts replay snarkjs' own prover in ffjavascript and dump intermediate
values that the Rust tests compare against element by element. They exist because a
pairing check is a single bit of information: it cannot tell a wrong coset apart from a
wrong Z convention apart from a wrong l_query offset, and two of those can cancel.

    gen-h-expected.mjs   <artifact-dir> <artifact-dir>/h_expected.json
    gen-msm-expected.mjs <artifact-dir> <artifact-dir>/msm_expected.json
    gen-fft-vectors.mjs  bench/fft-vectors

Consumed by, respectively:

    crates/g16-core/tests/h_matches_snarkjs.rs
    crates/g16-core/tests/msms_match_snarkjs.rs
    crates/g16-core/tests/ntt_matches_ffjavascript.rs

Each test skips loudly when its fixture is absent.

The scripts need `@iden3/binfileutils` and `ffjavascript` resolvable. snarkjs' own install
carries both; with a pnpm global snarkjs the quickest route is to symlink them into a
scratch `node_modules` and run from there.
