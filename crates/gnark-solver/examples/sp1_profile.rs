//! Profile allocating SP1 solves, checking every result against gnark outside the clock.
//!
//! `cargo run --release -p snarkrs-gnark-solver --example sp1_profile -- CIRCUIT WITNESS REFERENCE`
//!
//! `G16_SP1_PROFILE_REPS` selects 1..=25 repetitions (default 15). No dump is written.

use snarkrs_gnark_solver::{sp1, GnarkR1cs, Solution, Solver};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const CHUNK: usize = 1 << 20;

struct CheckDump<R> {
    reference: R,
    scratch: Vec<u8>,
    offset: u64,
}

impl<R: Read> CheckDump<R> {
    fn new(reference: R) -> Self {
        Self {
            reference,
            scratch: vec![0; CHUNK],
            offset: 0,
        }
    }

    fn finish(mut self) -> io::Result<()> {
        match self.reference.read_exact(&mut [0u8; 1]) {
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(()),
            Err(error) => Err(error),
            Ok(()) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing reference solution data",
            )),
        }
    }
}

impl<R: Read> Write for CheckDump<R> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        for chunk in bytes.chunks(CHUNK) {
            let expected = &mut self.scratch[..chunk.len()];
            self.reference.read_exact(expected)?;
            if let Some(index) = chunk.iter().zip(expected.iter()).position(|(a, b)| a != b) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("solution differs at byte {}", self.offset + index as u64),
                ));
            }
            self.offset += chunk.len() as u64;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn check_solution(solution: &Solution, reference: &Path) -> io::Result<()> {
    let mut check = CheckDump::new(BufReader::with_capacity(CHUNK, File::open(reference)?));
    solution.write_dump(&mut check)?;
    check.finish()
}

fn repetitions(value: Option<&str>) -> Result<usize, Box<dyn Error>> {
    let count = value.map(str::parse).transpose()?.unwrap_or(15);
    if !(1..=25).contains(&count) {
        return Err("G16_SP1_PROFILE_REPS must be between 1 and 25".into());
    }
    Ok(count)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [circuit, witness, reference] = args.as_slice() else {
        return Err("usage: sp1_profile CIRCUIT WITNESS REFERENCE".into());
    };
    let reps = match std::env::var("G16_SP1_PROFILE_REPS") {
        Ok(value) => repetitions(Some(&value))?,
        Err(std::env::VarError::NotPresent) => repetitions(None)?,
        Err(error) => return Err(error.into()),
    };

    let t = Instant::now();
    let r1cs = GnarkR1cs::read(circuit)?;
    let load = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (public, secret) = sp1::Witness::from_json(&std::fs::read(witness)?)?.assignment()?;
    let witness = t.elapsed().as_secs_f64();
    let solver = Solver::sp1();
    let mut samples = Vec::with_capacity(reps);
    for rep in 0..reps {
        let t = Instant::now();
        let solution = solver.solve(&r1cs, &public, &secret)?;
        let elapsed = t.elapsed().as_secs_f64();
        check_solution(&solution, reference)?;
        samples.push(elapsed);
        eprintln!("sample {rep}: solve {elapsed:.6} s verified");
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": 1,
            "scope": "allocating SP1 solves, full reference check outside clocks",
            "load_seconds": load,
            "witness_seconds": witness,
            "solve_seconds": samples,
            "verified_repetitions": reps,
            "constraints": r1cs.nb_constraints(),
            "wires": r1cs.nb_wires(),
            "instructions": r1cs.nb_instructions(),
            "levels": r1cs.nb_levels(),
            "rayon_threads": rayon::current_num_threads(),
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn checks_across_write_and_chunk_boundaries() {
        let expected: Vec<u8> = (0..CHUNK + 13).map(|i| (i % 251) as u8).collect();
        let mut check = CheckDump::new(Cursor::new(&expected));
        check.write_all(&expected[..3]).unwrap();
        check.write_all(&expected[3..]).unwrap();
        check.finish().unwrap();
    }

    #[test]
    fn detects_a_mismatch_after_the_first_chunk() {
        let expected = vec![7u8; CHUNK + 13];
        let mut actual = expected.clone();
        actual[CHUNK + 4] = 8;
        let mut check = CheckDump::new(Cursor::new(expected));
        let error = check.write_all(&actual).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            format!("solution differs at byte {}", CHUNK + 4)
        );
    }

    #[test]
    fn handles_fragmented_and_interrupted_reads() {
        struct Fragmented {
            inner: Cursor<&'static [u8]>,
            interrupt: bool,
        }
        impl Read for Fragmented {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.interrupt = !self.interrupt;
                if self.interrupt {
                    return Err(io::ErrorKind::Interrupted.into());
                }
                let count = bytes.len().min(2);
                self.inner.read(&mut bytes[..count])
            }
        }
        let mut check = CheckDump::new(Fragmented {
            inner: Cursor::new(b"abcdefg"),
            interrupt: false,
        });
        check.write_all(b"abc").unwrap();
        check.write_all(b"defg").unwrap();
        check.finish().unwrap();
    }

    #[test]
    fn refuses_mismatch_truncation_and_trailing_data() {
        let mut wrong = CheckDump::new(Cursor::new(b"abd"));
        assert_eq!(
            wrong.write_all(b"abc").unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let mut short = CheckDump::new(Cursor::new(b"ab"));
        assert_eq!(
            short.write_all(b"abc").unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        let mut trailing = CheckDump::new(Cursor::new(b"abcd"));
        trailing.write_all(b"abc").unwrap();
        assert_eq!(
            trailing.finish().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn repetitions_are_bounded_and_fail_closed() {
        assert_eq!(repetitions(None).unwrap(), 15);
        assert_eq!(repetitions(Some("1")).unwrap(), 1);
        assert_eq!(repetitions(Some("25")).unwrap(), 25);
        for value in ["0", "26", "bad", "", "-1"] {
            assert!(repetitions(Some(value)).is_err());
        }
    }

    #[test]
    fn checks_the_full_gnark_fixture_solutions() {
        for name in ["lowlevel", "frontend"] {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
            let json: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join(format!("{name}.json"))).unwrap())
                    .unwrap();
            let values = |key: &str| {
                json[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| sp1::parse_go_int(v.as_str().unwrap()).unwrap())
                    .collect::<Vec<_>>()
            };
            let r1cs = GnarkR1cs::read(&root.join(format!("{name}.r1cs"))).unwrap();
            let solution = Solver::sp1()
                .solve(&r1cs, &values("public"), &values("secret"))
                .unwrap();
            check_solution(&solution, &root.join(format!("{name}.solution.bin"))).unwrap();
        }
    }
}
