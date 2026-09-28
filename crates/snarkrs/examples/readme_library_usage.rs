//! The library snippets from README.md, kept compilable.
//!
//! A README example that does not build is worse than no example: it is read as a promise
//! about the API. This is an example rather than a doctest because it needs the `metal`
//! feature, and it exists so `cargo build --examples --features metal` fails the moment a
//! snippet and the real API disagree. The witness snippet also needs `witness-wasm`.
use snarkrs::{prove, verify, Backend, ProvingKey, StageTimings, Witness};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse once. On a GPU backend `prepare` is also where the key is uploaded, so hold the
    // prepared circuit and prove against it repeatedly: that is exactly the difference
    // between the warm and the cold columns in the README.
    let pk = ProvingKey::load("circuit.zkey".as_ref())?;
    let n_public = pk.n_public;
    let circuit = snarkrs::metal::MetalBackend::new()?.prepare(pk)?;

    let w = Witness::load("circuit.wtns".as_ref())?.0;
    let mut t = StageTimings::default();
    let proof = prove(
        circuit.as_ref(),
        &w,
        &mut snarkrs::rand::thread_rng(),
        &mut t,
    )?;

    // The public signals are the witness prefix, which is what snarkjs publishes.
    let public = &w[1..=n_public];
    verify(&circuit.key().vk, public, &proof)?;
    snarkrs::write_proof("proof.json".as_ref(), &proof)?;
    snarkrs::write_public("public.json".as_ref(), public)?;

    #[cfg(feature = "witness-wasm")]
    witness_from_memory()?;
    Ok(())
}

/// "witness from memory": the witness goes from the calculator to the prover as `Vec<Fr>`,
/// with no `.wtns` in between.
#[cfg(feature = "witness-wasm")]
fn witness_from_memory() -> Result<(), Box<dyn std::error::Error>> {
    use snarkrs::witness::{Input, WitnessCalculator};

    let pk = ProvingKey::load("circuit.zkey".as_ref())?;
    let circuit = snarkrs::metal::MetalBackend::new()?.prepare(pk)?;

    // Compile the wasm once, then one witness per input.
    let calc = WitnessCalculator::from_file("circuit_js/circuit.wasm".as_ref())?;
    let w = calc.calculate(&Input::from_json_str(r#"{"a": "3", "b": "11"}"#)?)?;

    let mut t = StageTimings::default();
    let _proof = prove(
        circuit.as_ref(),
        &w,
        &mut snarkrs::rand::thread_rng(),
        &mut t,
    )?;
    Ok(())
}
