//! The library snippets from README.md, kept compilable.
//!
//! A README example that does not build is worse than no example: it is read as a promise
//! about the API. This is an example rather than a doctest because it needs the `metal`
//! feature, and it exists so `cargo build --examples --features metal` fails the moment a
//! snippet and the real API disagree.
use g16_core::{prove::prove, verify::verify, Backend, StageTimings};
use g16_zkey::{wtns::Witness, ProvingKey};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse once. On a GPU backend `prepare` is also where the key is uploaded, so hold the
    // prepared circuit and prove against it repeatedly: that is exactly the difference
    // between the warm and the cold columns in the README.
    let pk = ProvingKey::load(std::path::Path::new("circuit.zkey"))?;
    let n_public = pk.n_public;
    let circuit = g16_metal::MetalBackend::new()?.prepare(pk)?;

    let w = Witness::load(std::path::Path::new("circuit.wtns"))?.0;
    let mut t = StageTimings::default();
    let proof = prove(
        circuit.as_ref(),
        &w,
        &mut ark_std::rand::thread_rng(),
        &mut t,
    )?;

    // The public signals are the witness prefix, which is what snarkjs publishes.
    let public = &w[1..=n_public];
    verify(&circuit.key().vk, public, &proof)?;
    witness_from_memory()
}

/// "witness from memory": the witness goes from the calculator to the prover as `Vec<Fr>`,
/// with no `.wtns` in between.
fn witness_from_memory() -> Result<(), Box<dyn std::error::Error>> {
    use g16_witness::{Input, WitnessCalculator};

    let pk = ProvingKey::load(std::path::Path::new("circuit.zkey"))?;
    let circuit = g16_metal::MetalBackend::new()?.prepare(pk)?;

    // Compile the wasm once, then one witness per input.
    let calc = WitnessCalculator::from_file(std::path::Path::new("circuit_js/circuit.wasm"))?;
    let w = calc.calculate(&Input::from_json_str(r#"{"a": "3", "b": "11"}"#)?)?;

    let mut t = StageTimings::default();
    let _proof = prove(
        circuit.as_ref(),
        &w,
        &mut ark_std::rand::thread_rng(),
        &mut t,
    )?;
    Ok(())
}
