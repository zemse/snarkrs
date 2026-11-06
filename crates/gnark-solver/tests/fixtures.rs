//! The solver against gnark v0.14.0 itself, on the fixtures `testdata/gen` writes: each
//! solution must be the bytes gnark's `Solve` gave, and each failure gnark's error.

use snarkrs_field::Fr;
use snarkrs_gnark_solver::{sp1, GnarkR1cs, ReadError, SolveError, Solver};
use std::path::{Path, PathBuf};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

struct Case {
    r1cs: GnarkR1cs,
    public: Vec<Fr>,
    secret: Vec<Fr>,
    error: Option<String>,
}

fn case(name: &str) -> Case {
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(testdata(&format!("{name}.json"))).unwrap()).unwrap();
    let values = |k: &str| -> Vec<Fr> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| sp1::parse_go_int(v.as_str().unwrap()).unwrap())
            .collect()
    };
    let circuit = json["circuit"].as_str().unwrap();
    Case {
        r1cs: GnarkR1cs::read(&testdata(&format!("{circuit}.r1cs"))).unwrap(),
        public: values("public"),
        secret: values("secret"),
        error: json["error"].as_str().map(str::to_owned),
    }
}

fn dump(name: &str, r1cs: &GnarkR1cs, public: &[Fr], secret: &[Fr]) -> Vec<u8> {
    let s = Solver::sp1().solve(r1cs, public, secret).unwrap();
    let mut out = Vec::new();
    s.write_dump(&mut out).unwrap();
    assert_eq!(s.w.len(), r1cs.nb_wires(), "{name}");
    out
}

fn solves_as_gnark(name: &str) {
    let c = case(name);
    assert!(c.error.is_none());
    let want = std::fs::read(testdata(&format!("{name}.solution.bin"))).unwrap();
    let got = dump(name, &c.r1cs, &c.public, &c.secret);
    assert!(got == want, "{name}: the solution differs from gnark's");
}

fn fails_as_gnark(name: &str) {
    let c = case(name);
    let e = Solver::sp1()
        .solve(&c.r1cs, &c.public, &c.secret)
        .err()
        .unwrap();
    let want = c.error.unwrap();
    match &e {
        SolveError::Unsatisfied { .. } => assert_eq!(e.to_string(), want, "{name}"),
        SolveError::Hint { message, .. } => assert_eq!(message, &want, "{name}"),
        _ => panic!("{name}: {e}"),
    }
}

#[test]
fn lowlevel() {
    let c = case("lowlevel");
    assert_eq!(c.r1cs.gnark_version(), (0, 14, 0));
    assert_eq!((c.r1cs.nb_public(), c.r1cs.nb_secret()), (2, 7));
    assert_eq!(c.r1cs.nb_constraints(), 335);
    assert_eq!(c.r1cs.nb_wires(), 357);
    // Every hint the solver ships, and all four of SP1's.
    assert_eq!(c.r1cs.hint_calls().len(), 8);
    solves_as_gnark("lowlevel");
}

#[test]
fn frontend() {
    let c = case("frontend");
    assert_eq!(c.r1cs.nb_constraints(), 2266);
    assert_eq!(c.r1cs.nb_levels(), 101);
    solves_as_gnark("frontend");
}

#[test]
fn unsatisfied_constraints_are_gnarks() {
    fails_as_gnark("lowlevel_unsatisfied");
    fails_as_gnark("frontend_unsatisfied");
}

#[test]
fn hint_errors_are_gnarks() {
    fails_as_gnark("lowlevel_hint_error");
}

#[test]
fn caller_buffers_get_the_same_solution() {
    let c = case("frontend");
    let s = Solver::new().solve(&c.r1cs, &c.public, &c.secret).unwrap();
    let junk = Fr::from(7u64);
    let mut w = vec![junk; c.r1cs.nb_wires()];
    let mut a = vec![junk; c.r1cs.nb_constraints()];
    let mut b = vec![junk; c.r1cs.nb_constraints()];
    Solver::new()
        .solve_into(&c.r1cs, &c.public, &c.secret, &mut w, &mut a, &mut b)
        .unwrap();
    assert!(w == s.w && a == s.a && b == s.b);
    let e = Solver::new().solve_into(&c.r1cs, &c.public, &c.secret, &mut w[1..], &mut a, &mut b);
    assert!(matches!(e, Err(SolveError::BufferSize { what: "w", .. })));
}

#[test]
fn missing_hints_and_bad_witnesses_are_refused() {
    let c = case("lowlevel");
    let e = Solver::new()
        .solve(&c.r1cs, &c.public, &c.secret)
        .err()
        .unwrap();
    let SolveError::MissingHints(names) = e else {
        panic!("{e}")
    };
    // Registered by ID, so gnark names them by ID.
    assert_eq!(
        names,
        ["3038391954", "3205574893", "342246697", "3795648912"]
    );
    let e = Solver::sp1()
        .solve(&c.r1cs, &c.public, &c.secret[1..])
        .err()
        .unwrap();
    assert!(matches!(e, SolveError::WitnessSize { .. }));
}

/// Every truncation of a valid file is an error, and a byte flipped anywhere is an error or
/// a file that still reads (most of the bytes are coefficients, and most flips of one are
/// another coefficient): never a panic, an out-of-bounds read or an allocation bomb.
#[test]
fn damaged_files_are_errors() {
    let bytes = std::fs::read(testdata("lowlevel.r1cs")).unwrap();
    for n in (0..bytes.len()).step_by(7) {
        assert!(
            GnarkR1cs::from_bytes(&bytes[..n]).is_err(),
            "truncated to {n}"
        );
    }
    for i in 0..bytes.len() {
        let mut b = bytes.clone();
        b[i] ^= 0x5a;
        let _ = GnarkR1cs::from_bytes(&b);
    }

    let mut b = bytes.clone();
    b[16..24].copy_from_slice(&15u64.to_le_bytes());
    assert!(
        matches!(GnarkR1cs::from_bytes(&b), Err(ReadError::UnsupportedVersion(v)) if v == "0.15.0")
    );
}
