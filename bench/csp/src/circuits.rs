//! The linked circom witness generators, and the one call that reaches them.
//!
//! `witnesscalc_adapter::witness!(name)` declares an extern `name_witness(&str) ->
//! Result<Vec<u8>>` against the static library `build.rs` compiled from `name.cpp`. The
//! bytes it returns are a `.wtns` file, byte for byte what `snarkjs wtns calculate`
//! writes, which is why [`g16_zkey::wtns::Witness::from_bytes`] can take them unchanged.

use crate::{Target, Variant};
use anyhow::{anyhow, Result};

witnesscalc_adapter::witness!(sha256_128);
witnesscalc_adapter::witness!(sha256_256);
witnesscalc_adapter::witness!(sha256_512);
witnesscalc_adapter::witness!(sha256_1024);
witnesscalc_adapter::witness!(sha256_2048);
witnesscalc_adapter::witness!(keccak_128);
witnesscalc_adapter::witness!(keccak_256);
witnesscalc_adapter::witness!(keccak_512);
witnesscalc_adapter::witness!(keccak_1024);
witnesscalc_adapter::witness!(keccak_2048);
witnesscalc_adapter::witness!(poseidon_2);
witnesscalc_adapter::witness!(poseidon_4);
witnesscalc_adapter::witness!(poseidon_8);
witnesscalc_adapter::witness!(poseidon_12);
witnesscalc_adapter::witness!(poseidon_16);

/// Stack for the witness thread, copied from upstream. A default 2 MiB thread is enough
/// for every circuit here, but the ECDSA generator overruns it, and running the small
/// circuits on a different stack size than the big one is a difference for no reason.
const WITNESS_STACK: usize = 8 * 1024 * 1024;

type WitnessFn = fn(&str) -> Result<Vec<u8>>;

fn witness_fn(v: Variant) -> Result<WitnessFn> {
    let f: WitnessFn = match (v.target, v.input_size) {
        (Target::Sha256, 128) => sha256_128_witness,
        (Target::Sha256, 256) => sha256_256_witness,
        (Target::Sha256, 512) => sha256_512_witness,
        (Target::Sha256, 1024) => sha256_1024_witness,
        (Target::Sha256, 2048) => sha256_2048_witness,
        (Target::Keccak, 128) => keccak_128_witness,
        (Target::Keccak, 256) => keccak_256_witness,
        (Target::Keccak, 512) => keccak_512_witness,
        (Target::Keccak, 1024) => keccak_1024_witness,
        (Target::Keccak, 2048) => keccak_2048_witness,
        (Target::Poseidon, 2) => poseidon_2_witness,
        (Target::Poseidon, 4) => poseidon_4_witness,
        (Target::Poseidon, 8) => poseidon_8_witness,
        (Target::Poseidon, 12) => poseidon_12_witness,
        (Target::Poseidon, 16) => poseidon_16_witness,
        (t, n) => return Err(anyhow!("no {} circuit at input size {n}", t.as_str())),
    };
    Ok(f)
}

/// Compute the `.wtns` bytes for `variant` from `input_json`.
///
/// On its own thread with [`WITNESS_STACK`], because that is where upstream puts it and
/// the thread spawn is part of what the timed region pays for.
pub fn witness(variant: Variant, input_json: String) -> Result<Vec<u8>> {
    let f = witness_fn(variant)?;
    std::thread::Builder::new()
        .stack_size(WITNESS_STACK)
        .spawn(move || f(&input_json))?
        .join()
        .map_err(|_| anyhow!("{}: witness thread panicked", variant.name()))?
}
