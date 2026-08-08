//! The parallel NTT path against an external oracle.
//!
//! `g16-ntt`'s own `serial_and_parallel_transforms_agree` compares our two paths to each
//! other, and `matches_the_naive_dft` only runs at n = 8 and 16, both far below
//! `PARALLEL_THRESHOLD = 2^13`. Nothing checked the path the prover actually takes at
//! prover sizes against an implementation written by someone else. These vectors come
//! from ffjavascript's `Fr.fft`, the transform snarkjs itself uses, at 2^13, 2^14, 2^16.
//!
//! Lives in g16-core rather than g16-ntt only because this crate already has the JSON and
//! bigint dev dependencies the vectors need.

use g16_field::{Domain, Fr};
use g16_ntt::{CpuNtt, Direction, NttBackend};
use std::path::{Path, PathBuf};

fn vector_dir() -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/fft-vectors")
        .canonicalize()
        .ok()?;
    d.is_dir().then_some(d)
}

fn fr(s: &str) -> Fr {
    let n: num_bigint::BigUint = s.parse().unwrap();
    Fr::from(n)
}

#[test]
fn forward_transform_matches_ffjavascript() {
    let Some(dir) = vector_dir() else {
        eprintln!("SKIPPED: no bench/fft-vectors directory");
        return;
    };
    let ntt = CpuNtt::new();
    let mut ran = 0;
    for log in [13u32, 14, 16] {
        let path = dir.join(format!("fft_{log}.json"));
        if !path.is_file() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let strings = |k: &str| -> Vec<Fr> {
            v[k].as_array()
                .unwrap()
                .iter()
                .map(|x| fr(x.as_str().unwrap()))
                .collect()
        };
        let input = strings("input");
        let want = strings("output");
        assert_eq!(input.len(), 1usize << log);

        let d = Domain::new(1usize << log).unwrap();
        let mut got = input.clone();
        ntt.ntt(&d, &mut got, Direction::Forward);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "n = 2^{log}, index {i}");
        }

        // And back, so the inverse is pinned to the same oracle rather than only to its
        // own forward.
        ntt.ntt(&d, &mut got, Direction::Inverse);
        assert_eq!(got, input, "n = 2^{log} round trip");
        eprintln!(
            "2^{log}: forward transform matches ffjavascript at all {} indices",
            want.len()
        );
        ran += 1;
    }
    assert!(ran > 0, "vector directory exists but holds no fft_*.json");
}
