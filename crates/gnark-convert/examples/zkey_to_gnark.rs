//! A zkey to gnark's `WriteDump` proving key and `WriteTo` verifying key, and optionally a
//! `.wtns` to the witness dump (W, A.w and B.w) snarkrs' gnark path proves from.
//!
//! ```text
//! cargo run --release -p snarkrs-gnark-convert --example zkey_to_gnark -- \
//!     <circuit.zkey> <out dir> [<circuit.wtns>]
//! ```
//!
//! Writes `<out dir>/pk.dump`, `<out dir>/vk.bin` and, with a `.wtns`, `<out dir>/witness.bin`,
//! which `cargo run --release -p snarkrs-gnark --example prove -- <out dir>/pk.dump
//! <out dir>/vk.bin <out dir>/witness.bin <proof dir>` proves. gnark's own prover needs a
//! gnark R1CS of the circuit as well, which this does not write.

use snarkrs_ceremony::CpuGroupFft;
use snarkrs_formats::wtns::Witness;
use snarkrs_formats::ProvingKey;
use snarkrs_gnark_convert::proof::gnark_vk;
use snarkrs_gnark_convert::to_gnark::{gnark_keys, write_dump};
use snarkrs_gnark_convert::zkey_rows;
use snarkrs_gnark_solver::Solution;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let (zkey_path, out, wtns) = match &args[..] {
        [zkey, out] => (zkey, out, None),
        [zkey, out, wtns] => (zkey, out, Some(wtns)),
        _ => return Err("usage: zkey_to_gnark <circuit.zkey> <out dir> [<circuit.wtns>]".into()),
    };

    let t = Instant::now();
    let zkey = ProvingKey::load(zkey_path)?;
    println!(
        "zkey loaded in {:.2} s: {} wires, domain {}",
        secs(t),
        zkey.n_vars,
        zkey.domain_size
    );
    std::fs::create_dir_all(out)?;

    if let Some(wtns) = wtns {
        let w = Witness::load(wtns)?.0;
        let [a, b] = zkey_rows(&zkey, &w);
        let solution = Solution { w, a, b };
        solution.write_dump(BufWriter::new(std::fs::File::create(
            out.join("witness.bin"),
        )?))?;
    }

    let t = Instant::now();
    let (pk, vk) = gnark_keys(&zkey, &CpuGroupFft)?;
    drop(zkey);
    println!("converted in {:.2} s", secs(t));
    let t = Instant::now();
    write_dump(
        &pk,
        BufWriter::new(std::fs::File::create(out.join("pk.dump"))?),
    )?;
    std::fs::write(out.join("vk.bin"), gnark_vk(&vk, true))?;
    println!("wrote {} in {:.2} s", out.display(), secs(t));
    Ok(())
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}
