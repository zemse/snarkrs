//! `g16` - prove, verify and benchmark.
//!
//!   g16 prove  --zkey c.zkey --witness c.wtns --proof p.json --public pub.json [--backend cpu|metal]
//!   g16 verify --vkey vkey.json --proof p.json --public pub.json
//!   g16 bench  --artifacts DIR [--reps 15] [--backend ...] [--warm] [--csv out.csv]
//!
//! `--warm` is the whole point of the benchmark: cold pays zkey parse + (on GPU) upload
//! on every proof, warm pays them once and then proves in a loop. Reporting only one of
//! the two is how vendor charts mislead.

fn main() -> anyhow::Result<()> {
    todo!("g16-cli: implement main")
}
