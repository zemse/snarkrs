//! Per-annotation timing of the proving path.
//!
//! `cargo run --release -p g16-core --features hotpath --example hotpath_prove -- DIR [REPS]`
//!
//! What this gives that the other two instruments do not:
//!
//!   * `StageTimings` reports five stage groups. It cannot separate the three iNTTs from
//!     the three forward NTTs, because both are `ntt_us`.
//!   * A sampling profile reports which function the CPU is in. It cannot separate the
//!     six transforms either, because they are the same code on the same shape of data.
//!
//! The annotations sit at region boundaries in `g16-core` only, so what they measure is
//! the composition of the pipeline, which is exactly the thing `g16-core` owns.
//!
//! Read the report with two caveats. Regions inside `rayon::join` overlap, so the five
//! MSM lines are concurrent windows and sum to more than the enclosing region. And the
//! instrumentation is not free: quote timings from a build without this feature.

use std::path::PathBuf;

use g16_core::{cpu::CpuBackend, prove::prove_with_blinders, Backend, StageTimings};
use g16_field::{BigInteger, Fr, PrimeField};
use g16_zkey::{wtns::Witness, ProvingKey};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "bench/artifacts/js_8x8_d32".to_string()),
    );
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("loading circuit.zkey");
    let witness = Witness::load(&dir.join("circuit.wtns"))
        .expect("loading circuit.wtns")
        .0;
    let circuit = CpuBackend::new().prepare(pk).expect("prepare");

    eprintln!("{} x {reps} reps, warm (key prepared once)", dir.display());

    // Built after loading so the key parse is not in the report: this measures proving.
    // Percentiles rather than a mean, because the interesting question about a stage on a
    // laptop is what it costs on a good run, not what the OS did to the worst one.
    let _guard = hotpath::HotpathGuardBuilder::new("g16-prove")
        .percentiles(&[50.0, 95.0])
        .limit(40)
        .build();

    // Fixed blinders rather than an OS CSPRNG: `prove` requires a `CryptoRng` and
    // `g16-core` does not enable ark-std's `getrandom`, so there is no OS RNG in scope.
    // Nothing this binary produces is a proof anyone should keep.
    //
    // They must be full width. Blinding is six variable-base scalar multiplications, and
    // `mul_bigint` skips the scalar's leading zero bits, so a small `r` makes stage 11
    // look about six times cheaper than it is: `Fr::from(0x5eed)` measured 74 us against
    // the 570 us the real prover reports for the same stage. A profiling harness that
    // feeds an unrepresentative input is worse than one that does not exist.
    let full = |b: u8| Fr::from_le_bytes_mod_order(&[b; 31]);
    let (r, s) = (full(0x9e), full(0x37));
    assert!(
        r.into_bigint().num_bits() > 240 && s.into_bigint().num_bits() > 240,
        "blinders must be full width or stage 11 is mismeasured"
    );
    for _ in 0..reps {
        let mut t = StageTimings::default();
        let proof =
            prove_with_blinders(circuit.as_ref(), &witness, r, s, &mut t).expect("prove");
        std::hint::black_box(&proof);
    }
}
