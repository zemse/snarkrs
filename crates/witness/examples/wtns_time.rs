//! Time witness generation in process, split into compiling the module and running it.
//!
//! `cargo run --release -p snarkrs-witness --example wtns_time -- REPS DIR...`
//!
//! Each DIR holds one `*_js/*.wasm`, `input.json` and the reference `circuit.wtns`. Every
//! run's witness is compared against the reference before anything is printed. One line
//! per DIR: `name compile_ms run_median_ms run_min_ms wires`.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use snarkrs_formats::wtns::Witness;
use snarkrs_witness::{Input, WitnessCalculator};

fn wasm_in(dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir()
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with("_js"))
        {
            for f in std::fs::read_dir(&p)? {
                let f = f?.path();
                if f.extension().is_some_and(|x| x == "wasm") {
                    return Ok(f);
                }
            }
        }
    }
    Err(format!("{}: no *_js/*.wasm", dir.display()).into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let reps: usize = args.next().ok_or("usage: wtns_time REPS DIR...")?.parse()?;
    for dir in args {
        let dir = Path::new(&dir);
        let wasm = std::fs::read(wasm_in(dir)?)?;
        let input = Input::from_json_str(&std::fs::read_to_string(dir.join("input.json"))?)?;
        let want = Witness::load(&dir.join("circuit.wtns"))?.0;

        let t = Instant::now();
        let calc = WitnessCalculator::new(&wasm)?;
        let compile = t.elapsed().as_secs_f64() * 1e3;

        let mut ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            let w = calc.calculate(&input)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            if w != want {
                return Err(format!("{}: witness differs from circuit.wtns", dir.display()).into());
            }
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        println!(
            "{name} {compile:.1} {:.2} {:.2} {}",
            ms[ms.len() / 2],
            ms[0],
            want.len()
        );
    }
    Ok(())
}
