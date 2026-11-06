//! SP1 v6.1.0's Groth16 wrap circuit, solved and compared byte for byte with what gnark's
//! own `Solve` returned for the same witness.
//!
//! The circuit is 2.4 GB and the solutions 1.5 GB each, none of it in this repo, so this
//! is opt in: `G16_SP1_CIRCUIT` names `groth16_circuit.bin` and `G16_SP1_CASES` lists
//! `WITNESS.json=SOLUTION.bin` pairs separated by commas, each solution being gnark-crypto's
//! `unsafe` dump of W, A and B. Unset, the test skips; `G16_REQUIRE_TESTS=1` makes that a
//! failure.
//!
//! ```sh
//! G16_SP1_CIRCUIT=~/.sp1/circuits/groth16/v6.1.0/groth16_circuit.bin \
//! G16_SP1_CASES=~/.sp1/circuits/groth16/v6.1.0/groth16_witness.json=sample.bin,job2/witness.json=job2/solution.bin \
//! cargo test --release -p snarkrs-gnark-solver --test sp1 -- --nocapture
//! ```

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use snarkrs_field::Fr;
use snarkrs_gnark_solver::{sp1, GnarkR1cs, Solver};
use std::path::PathBuf;
use std::time::Instant;

/// The index of the first element of `v` whose dump in `bytes` differs, after checking the
/// u64 length in front of it; and the bytes after it.
fn compare<'a>(what: &str, v: &[Fr], bytes: &'a [u8]) -> &'a [u8] {
    let (len, rest) = bytes.split_at(8);
    assert_eq!(
        u64::from_le_bytes(len.try_into().unwrap()),
        v.len() as u64,
        "{what} length"
    );
    let (data, rest) = rest.split_at(32 * v.len());
    if let Some(i) = v.iter().zip(data.as_chunks::<32>().0).position(|(x, b)| {
        x.0 .0
            .iter()
            .zip(b.as_chunks::<8>().0)
            .any(|(l, b)| l.to_le_bytes() != *b)
    }) {
        panic!("{what}[{i}] differs from gnark's");
    }
    rest
}

#[test]
fn sp1_v6_1_0_solves_as_gnark() {
    let (Some(circuit), Some(cases)) = (
        std::env::var_os("G16_SP1_CIRCUIT"),
        std::env::var("G16_SP1_CASES").ok(),
    ) else {
        resources::skip("sp1: set G16_SP1_CIRCUIT and G16_SP1_CASES");
        return;
    };

    let t = Instant::now();
    let r1cs = GnarkR1cs::read(&PathBuf::from(circuit)).unwrap();
    eprintln!("load {:.3} s", t.elapsed().as_secs_f64());
    assert_eq!(r1cs.nb_constraints(), 15_972_262);
    assert_eq!(r1cs.nb_wires(), 15_747_144);
    assert_eq!((r1cs.nb_public(), r1cs.nb_secret()), (6, 64_900));
    let solver = Solver::sp1();

    for case in cases.split(',').filter(|c| !c.is_empty()) {
        let (witness, solution) = case.split_once('=').expect("WITNESS.json=SOLUTION.bin");
        let json = std::fs::read(witness).unwrap();
        let (public, secret) = sp1::Witness::from_json(&json)
            .unwrap()
            .assignment()
            .unwrap();

        let t = Instant::now();
        let s = solver.solve(&r1cs, &public, &secret).unwrap();
        eprintln!("{witness}: solve {:.3} s", t.elapsed().as_secs_f64());

        let file = std::fs::File::open(solution).unwrap();
        // SAFETY: read-only, and nothing writes the reference solution meanwhile.
        let gold = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        assert_eq!(gold[..8], 0xdeadbeefu64.to_le_bytes());
        let rest = compare("W", &s.w, &gold[8..]);
        let rest = compare("A", &s.a, rest);
        let rest = compare("B", &s.b, rest);
        assert!(rest.is_empty(), "{solution}: bytes after B");
        eprintln!("{witness}: identical to {solution}");
    }
}
