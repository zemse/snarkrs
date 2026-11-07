//! `for_each_r1c` against gnark's own solution: every constraint's `L·w` and `R·w` are the
//! A and B gnark's `Solve` gave, and `L·w * R·w == O·w`.

use snarkrs_field::{Fr, Zero};
use snarkrs_gnark_solver::{sp1, GnarkR1cs, Solver, Term};
use std::path::{Path, PathBuf};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn check(name: &str) {
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
    let r1cs = GnarkR1cs::read(&testdata(&format!("{name}.r1cs"))).unwrap();
    let s = Solver::sp1()
        .solve(&r1cs, &values("public"), &values("secret"))
        .unwrap();
    let k = r1cs.coefficients();
    let eval = |terms: &[Term]| -> Fr {
        terms
            .iter()
            .map(|t| k[t.coefficient as usize] * s.w[t.wire as usize])
            .sum()
    };
    let mut seen = vec![false; r1cs.nb_constraints()];
    r1cs.for_each_r1c(0..r1cs.nb_instructions(), |c, l, r, o| {
        assert!(!seen[c], "{name}: constraint {c} twice");
        seen[c] = true;
        let (a, b) = (eval(l), eval(r));
        assert_eq!((a, b), (s.a[c], s.b[c]), "{name}: constraint {c}");
        assert!((a * b - eval(o)).is_zero(), "{name}: constraint {c}");
    });
    assert!(
        seen.iter().all(|&s| s),
        "{name}: a constraint was never visited"
    );
    assert_eq!(
        &k[..4],
        &[Fr::zero(), Fr::from(1u64), Fr::from(2u64), -Fr::from(1u64)]
    );
}

#[test]
fn terms_evaluate_to_gnarks_solution() {
    check("lowlevel");
    check("frontend");
}
