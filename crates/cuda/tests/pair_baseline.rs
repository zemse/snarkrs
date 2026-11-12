//! Unix-only canonical serial proof pairs with ELF/Mach-O oracles; never a skip or fallback.

#![cfg(all(feature = "cuda", unix))]

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ark_std::rand::rngs::OsRng;
use snarkrs_cuda::CudaBackend;
use snarkrs_field::Fr;
use snarkrs_formats::{json_out::JsonWriter, wtns::Witness, ProvingKey, VerifyingKey};
use snarkrs_groth16::json::{proof_to_string, public_from_str, public_to_string};
use snarkrs_groth16::prove::prove_unchecked;
use snarkrs_groth16::verify::verify;
use snarkrs_groth16::{PreparedCircuit, Proof, StageTimings};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const FIXTURES: [&str; 2] = ["tiny_mul", "sha256"];
const SUCCESS: &str = "Result: Valid proof";

struct Config {
    artifacts: PathBuf,
    output: PathBuf,
    oracle: PathBuf,
    warm: usize,
    measured: usize,
    timeout: Duration,
}

impl Config {
    fn parse(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let path = |key| -> Result<PathBuf> {
            let p = PathBuf::from(get(key).ok_or_else(|| format!("set {key}"))?);
            if !p.is_absolute() {
                return Err(format!("{key} must be absolute").into());
            }
            Ok(p)
        };
        let count = |key, default, min, max| -> Result<usize> {
            let n = match get(key) {
                Some(s) => s.parse::<usize>()?,
                None => default,
            };
            if !(min..=max).contains(&n) {
                return Err(format!("{key} must be in {min}..={max}").into());
            }
            Ok(n)
        };
        let config = Self {
            artifacts: path("G16_ARTIFACTS")?,
            output: path("G16_PAIR_OUTPUT")?,
            oracle: path("G16_RAPIDSNARK_VERIFY")?,
            warm: count("G16_PAIR_WARM", 2, 1, 100)?,
            measured: count("G16_PAIR_MEASURED", 15, 15, 1000)?,
            timeout: Duration::from_secs(count("G16_PAIR_ORACLE_TIMEOUT_S", 30, 1, 300)? as u64),
        };
        config.proof_count()?;
        Ok(config)
    }

    fn phases(&self) -> [(&'static str, usize); 2] {
        [("warm", self.warm), ("measured", self.measured)]
    }

    fn proof_count(&self) -> Result<usize> {
        self.warm
            .checked_add(self.measured)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| "proof invocation count overflow".into())
    }

    fn preflight(&self) -> Result<()> {
        if self.output.exists() || !self.output.parent().is_some_and(Path::is_dir) {
            return Err("output must be new with an existing parent directory".into());
        }
        for name in FIXTURES {
            for file in ["circuit.zkey", "circuit.wtns", "vkey.json"] {
                let p = self.artifacts.join(name).join(file);
                if !p.is_file() {
                    return Err(format!("missing input {}", p.display()).into());
                }
            }
        }
        native_executable(&self.oracle)
    }
}

fn native_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let metadata = path.metadata()?;
    if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err("oracle must be an executable native file".into());
    }
    let mut magic = [0; 4];
    File::open(path)?.read_exact(&mut magic)?;
    if !matches!(
        magic,
        [0x7f, b'E', b'L', b'F']
            | [0xcf, 0xfa, 0xed, 0xfe]
            | [0xfe, 0xed, 0xfa, 0xcf]
            | [0xca, 0xfe, 0xba, 0xbe]
            | [0xbe, 0xba, 0xfe, 0xca]
    ) {
        return Err("oracle must be native ELF or Mach-O, not a script adapter".into());
    }
    Ok(())
}

struct Fixture {
    name: &'static str,
    key: ProvingKey,
    witness: Vec<Fr>,
    vk: VerifyingKey,
    public: Vec<Fr>,
    vkey_path: PathBuf,
}

fn load_fixture(root: &Path, name: &'static str) -> Result<Fixture> {
    let dir = root.join(name);
    let key = ProvingKey::load(&dir.join("circuit.zkey"))?;
    let witness = Witness::load(&dir.join("circuit.wtns"))?.0;
    let vkey_path = dir.join("vkey.json");
    let vk = VerifyingKey::from_json(&vkey_path)?;
    if witness.len() != key.n_vars
        || key.n_public >= witness.len()
        || vk.ic.len() != key.n_public + 1
    {
        return Err(format!("{name}: key/witness/vkey dimensions disagree").into());
    }
    if witness[0] != Fr::from(1u64) {
        return Err(format!("{name}: witness constant is not one").into());
    }
    if vk.alpha_g1 != key.vk.alpha_g1
        || vk.beta_g2 != key.vk.beta_g2
        || vk.gamma_g2 != key.vk.gamma_g2
        || vk.delta_g2 != key.vk.delta_g2
        || vk.ic != key.vk.ic
    {
        return Err(format!("{name}: original vkey disagrees with zkey").into());
    }
    let public = witness[1..=key.n_public].to_vec();
    let public_path = dir.join("public.json");
    if public_path.exists() && public_from_str(&std::fs::read_to_string(public_path)?)? != public {
        return Err(format!("{name}: original public.json disagrees with witness").into());
    }
    Ok(Fixture {
        name,
        key,
        witness,
        vk,
        public,
        vkey_path,
    })
}

struct Evidence {
    root: PathBuf,
    files: BTreeSet<PathBuf>,
}

impl Evidence {
    fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir(root)?;
        Ok(Self {
            root: root.to_owned(),
            files: BTreeSet::new(),
        })
    }

    fn create(&mut self, relative: &str) -> io::Result<File> {
        let path = self.root.join(relative);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        self.files.insert(path);
        Ok(file)
    }

    fn bytes(&mut self, relative: &str, data: &[u8]) -> io::Result<()> {
        let mut file = self.create(relative)?;
        file.write_all(data)?;
        file.flush()
    }

    fn json(
        &mut self,
        relative: &str,
        f: impl FnOnce(&mut JsonWriter<File>) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut json = JsonWriter::new(self.create(relative)?);
        f(&mut json)?;
        json.finish()?;
        Ok(())
    }

    fn audit(&self, expected: usize) -> Result<()> {
        let actual: BTreeSet<_> = std::fs::read_dir(&self.root)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<io::Result<_>>()?;
        if self.files.len() != expected || actual != self.files {
            return Err("output file inventory mismatch; no acceptance report".into());
        }
        for path in actual {
            let metadata = path.symlink_metadata()?;
            if !metadata.is_file()
                || (metadata.len() == 0
                    && path
                        .extension()
                        .is_none_or(|e| e != "stdout" && e != "stderr"))
            {
                return Err(format!("invalid evidence file {}", path.display()).into());
            }
        }
        Ok(())
    }
}

/// The clock result is generic so synthetic ordering tests need no fabricated timings.
fn timed_pair<P, C, T>(
    start: impl FnOnce() -> C,
    mut produce: impl FnMut(usize) -> Result<P>,
    stop: impl FnOnce(C) -> T,
    mut retain: impl FnMut(usize, &P) -> Result<()>,
    mut check: impl FnMut(usize, &P) -> Result<()>,
) -> Result<(T, [P; 2])> {
    let clock = start();
    let first = match produce(0) {
        Ok(p) => p,
        Err(e) => {
            stop(clock);
            return Err(e);
        }
    };
    let second = match produce(1) {
        Ok(p) => p,
        Err(e) => {
            stop(clock);
            retain(0, &first)?;
            return Err(e);
        }
    };
    let elapsed = stop(clock);
    let pair = [first, second];
    for (slot, proof) in pair.iter().enumerate() {
        retain(slot, proof)?;
    }
    for (slot, proof) in pair.iter().enumerate() {
        check(slot, proof)?;
    }
    Ok((elapsed, pair))
}

fn oracle_passed(code: Option<i32>, timed_out: bool, stdout: &[u8], stderr: &[u8]) -> bool {
    !timed_out
        && code == Some(0)
        && [stdout, stderr].into_iter().any(|text| {
            String::from_utf8_lossy(text)
                .lines()
                .any(|line| line.trim() == SUCCESS)
        })
}

fn unix_ms() -> Result<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn oracle(
    config: &Config,
    evidence: &mut Evidence,
    prefix: &str,
    vkey: &Path,
    proof_prefix: &str,
) -> Result<()> {
    let argv = [
        config.oracle.clone(),
        vkey.to_owned(),
        evidence.root.join(format!("{proof_prefix}-public.json")),
        evidence.root.join(format!("{proof_prefix}-proof.json")),
    ];
    let stdout_name = format!("{prefix}.stdout");
    let stderr_name = format!("{prefix}.stderr");
    let stdout = evidence.create(&stdout_name)?;
    let stderr = evidence.create(&stderr_name)?;
    let started = unix_ms()?;
    let clock = Instant::now();
    let mut timed_out = false;
    let mut error = None;
    let status = match Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
    {
        Ok(mut child) => loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if clock.elapsed() < config.timeout => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Ok(None) => {
                    timed_out = true;
                    let _ = child.kill();
                    break Some(child.wait()?);
                }
                Err(e) => {
                    error = Some(e.to_string());
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
            }
        },
        Err(e) => {
            error = Some(e.to_string());
            None
        }
    };
    let out = std::fs::read(evidence.root.join(&stdout_name))?;
    let err = std::fs::read(evidence.root.join(&stderr_name))?;
    let passed =
        error.is_none() && oracle_passed(status.and_then(|s| s.code()), timed_out, &out, &err);
    evidence.json(&format!("{prefix}.json"), |json| {
        json.begin_object()?;
        json.field_string("kind", "actual-external-rapidsnark-process")?;
        json.key("argv")?;
        json.begin_array()?;
        for arg in &argv {
            json.string(&arg.to_string_lossy())?;
        }
        json.end()?;
        json.field_raw("started_unix_ms", &started.to_string())?;
        json.field_raw(
            "external_process_wall_ms",
            &clock.elapsed().as_millis().to_string(),
        )?;
        json.field_raw("timeout_ms", &config.timeout.as_millis().to_string())?;
        json.field_raw("timed_out", if timed_out { "true" } else { "false" })?;
        json.field_raw(
            "returncode",
            &status
                .and_then(|s| s.code())
                .map_or("null".into(), |n| n.to_string()),
        )?;
        json.field_string("stdout_file", &stdout_name)?;
        json.field_string("stderr_file", &stderr_name)?;
        json.field_string("required_success_line", SUCCESS)?;
        json.field_raw("passed", if passed { "true" } else { "false" })?;
        json.key("process_error")?;
        if let Some(error) = &error {
            json.string(error)?;
        } else {
            json.raw("null")?;
        }
        json.end()
    })?;
    if !passed {
        return Err(format!("external oracle failed: {prefix}").into());
    }
    Ok(())
}

struct Record {
    proof: Proof,
    timings: StageTimings,
    witness_upload_event_us: u64,
    msm_stream_event_span_us: u64,
    msm_runs: u64,
}

fn record_json(json: &mut JsonWriter<File>, record: &Record) -> io::Result<()> {
    json.begin_object()?;
    json.key("stage_timings_raw_us")?;
    json.begin_object()?;
    let t = record.timings;
    for (key, value) in [
        ("gather_us", t.gather_us),
        ("ntt_us", t.ntt_us),
        ("pointwise_us", t.pointwise_us),
        ("msm_us", t.msm_us),
        ("assemble_us", t.assemble_us),
        ("verify_us", t.verify_us),
    ] {
        json.field_raw(key, &value.to_string())?;
    }
    json.key("msm_split")?;
    if let Some(s) = t.msm_split {
        json.begin_object()?;
        for (key, value) in [
            ("a_us", s.a_us),
            ("b_g2_us", s.b_g2_us),
            ("b_g1_us", s.b_g1_us),
            ("l_us", s.l_us),
            ("h_us", s.h_us),
        ] {
            json.field_raw(key, &value.to_string())?;
        }
        json.end()?;
    } else {
        json.raw("null")?;
    }
    json.end()?;
    json.field_raw(
        "witness_upload_cuda_event_us",
        &record.witness_upload_event_us.to_string(),
    )?;
    json.field_raw(
        "msm_cuda_stream_elapsed_span_us",
        &record.msm_stream_event_span_us.to_string(),
    )?;
    json.field_raw("msm_runs", &record.msm_runs.to_string())?;
    json.end()
}

fn write_config(config: &Config, evidence: &mut Evidence) -> Result<()> {
    evidence.json("config.json", |json| {
        json.begin_object()?;
        json.field_string("mode", "canonical-serial-randomized-proof-pairs")?;
        json.field_string("backend", "cuda")?;
        json.field_raw("cpu_fallback", "false")?;
        json.field_string("source_binding", "requires independent signed-source and executable-hash manifest; no environment source claim")?;
        json.field_string("executable", &std::env::current_exe()?.to_string_lossy())?;
        json.field_string("artifacts_root", &config.artifacts.to_string_lossy())?;
        json.field_string("external_oracle", &config.oracle.to_string_lossy())?;
        json.field_raw("warm_pairs_per_fixture", &config.warm.to_string())?;
        json.field_raw("measured_pairs_per_fixture", &config.measured.to_string())?;
        json.field_raw("oracle_timeout_ms", &config.timeout.as_millis().to_string())?;
        json.field_string("key_residency", "prepare once per fixture outside clocks; one fixture at a time, no two-key fit claim")?;
        json.field_string("pair_wall_clock", "host Instant around two sequential prove_unchecked calls with OsRng, including both diagnostic getter snapshots and internal sync; excludes explicit verification, oracle processes, serialization and preparation")?;
        json.field_string("stage_timing_semantics", "gather_us mixes host pack and CUDA event spans; ntt_us and pointwise_us are stream event spans; fused pointwise zero means included in ntt, not free; msm_us mixes pre-batch host interval, stream event span and host readback/combine; assemble_us is host; verify_us is zero from unchecked calls")?;
        json.field_string("event_warning", "CUDA stream elapsed spans may include host submission gaps/allocator waits; not pure SM busy; do not sum host/device fields or add them to pair wall")?;
        json.field_string("acceptance", "every warm/measured proof verified against original vkey and WTNS publics outside clocks; both slots externally verified at first warm and last measured pair; accepted.json only after complete file audit")?;
        json.field_raw("full_workload_trace", "false")?;
        json.key("traffic_floors_and_two_lane_capacity")?; json.raw("null")?;
        json.key("input_files")?; json.begin_array()?;
        for fixture in FIXTURES {
            for file in ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"] {
                let p = config.artifacts.join(fixture).join(file);
                if p.exists() { json.string(&p.to_string_lossy())?; }
            }
        }
        json.end()?;
        json.key("cuda_environment")?; json.begin_object()?;
        let mut env: Vec<_> = std::env::vars().filter(|(key, _)| key.starts_with("G16_CUDA_") || key == "CUDA_VISIBLE_DEVICES" || key == "RAYON_NUM_THREADS").collect();
        env.sort();
        for (key, value) in env { json.field_string(&key, &value)?; }
        json.end()?;
        json.end()
    })?;
    Ok(())
}

fn run(config: Config) -> Result<()> {
    config.preflight()?;
    let fixtures = FIXTURES
        .into_iter()
        .map(|name| load_fixture(&config.artifacts, name))
        .collect::<Result<Vec<_>>>()?;
    let mut evidence = Evidence::new(&config.output)?;
    write_config(&config, &mut evidence)?;
    let backend = CudaBackend::new()?;
    let mut rng = OsRng;
    let mut raw_pairs = Vec::new();
    let mut own_checks = 0usize;
    let mut external_checks = 0usize;
    for fixture in fixtures {
        let circuit = backend.prepare_circuit(fixture.key)?;
        if circuit.backend_name() != "cuda" {
            return Err("not the CUDA backend".into());
        }
        for (phase, count) in config.phases() {
            for index in 0..count {
                let prefix = format!("{}-{phase}-{index:04}", fixture.name);
                let (elapsed, pair) = timed_pair(
                    Instant::now,
                    |_| {
                        let mut timings = StageTimings::default();
                        let proof =
                            prove_unchecked(&circuit, &fixture.witness, &mut rng, &mut timings)?;
                        Ok(Record {
                            proof,
                            timings,
                            witness_upload_event_us: circuit.last_witness_upload_us(),
                            msm_stream_event_span_us: circuit.last_msm_device_us(),
                            msm_runs: circuit.last_msm_runs(),
                        })
                    },
                    |clock| clock.elapsed(),
                    |slot, record| {
                        evidence.bytes(
                            &format!("{prefix}-slot{slot}-proof.json"),
                            proof_to_string(&record.proof).as_bytes(),
                        )?;
                        evidence.bytes(
                            &format!("{prefix}-slot{slot}-public.json"),
                            public_to_string(&fixture.public).as_bytes(),
                        )?;
                        Ok(())
                    },
                    |_, record| {
                        verify(&fixture.vk, &fixture.public, &record.proof)?;
                        own_checks += 1;
                        Ok(())
                    },
                )?;
                let raw = format!("{prefix}-raw.json");
                evidence.json(&raw, |json| {
                    json.begin_object()?;
                    json.field_string("fixture", fixture.name)?;
                    json.field_string("phase", phase)?;
                    json.field_raw("pair_index", &index.to_string())?;
                    json.field_raw("pair_host_wall_ns", &elapsed.as_nanos().to_string())?;
                    json.field_string(
                        "own_verification",
                        "both passed against original vkey and WTNS publics outside clock",
                    )?;
                    json.field_string("acceptance", "raw only; final accepted.json required")?;
                    json.key("proof_slots")?;
                    json.begin_array()?;
                    for record in &pair {
                        record_json(json, record)?;
                    }
                    json.end()?;
                    json.end()
                })?;
                raw_pairs.push(raw);
                let boundary = if phase == "warm" && index == 0 {
                    Some("first-warm")
                } else if phase == "measured" && index + 1 == count {
                    Some("last-measured")
                } else {
                    None
                };
                if let Some(boundary) = boundary {
                    for slot in 0..2 {
                        oracle(
                            &config,
                            &mut evidence,
                            &format!("{}-{boundary}-slot{slot}-oracle", fixture.name),
                            &fixture.vkey_path,
                            &format!("{prefix}-slot{slot}"),
                        )?;
                        external_checks += 1;
                    }
                }
            }
        }
    }
    let proofs = config.proof_count()?;
    let pairs = proofs / 2;
    let expected_files = proofs * 2 + pairs + 8 * 3 + 1;
    if own_checks != proofs || external_checks != 8 || raw_pairs.len() != pairs {
        return Err("proof/receipt invocation inventory mismatch".into());
    }
    evidence.audit(expected_files)?;
    evidence.json("accepted.json", |json| {
        json.begin_object()?;
        json.field_string("status", "complete-verified-serial-pairs")?;
        json.field_raw("proof_invocations", &proofs.to_string())?;
        json.field_raw("own_verified_proofs", &own_checks.to_string())?;
        json.field_raw("external_endpoint_receipts", &external_checks.to_string())?;
        json.field_raw(
            "audited_files_before_this_report",
            &expected_files.to_string(),
        )?;
        json.key("raw_pair_files")?;
        json.begin_array()?;
        for file in &raw_pairs {
            json.string(file)?;
        }
        json.end()?;
        json.field_raw("full_workload_trace", "false")?;
        json.field_string(
            "limits",
            "not a floor, traffic trace, concurrency or two-lane capacity result",
        )?;
        json.end()
    })?;
    Ok(())
}

#[test]
#[ignore = "requires explicit original fixtures, new output directory, native rapidsnark verifier and NVIDIA; never skips"]
fn serial_proof_pair_baseline() {
    let config = Config::parse(|key| std::env::var(key).ok()).expect("serial pair configuration");
    run(config).expect("every proof, oracle receipt and evidence file must pass");
}

#[test]
fn synthetic_pair_orders_two_calls_before_stop_and_all_io_and_checks() {
    use std::cell::RefCell;
    let log = RefCell::new(Vec::new());
    let config = Config {
        artifacts: PathBuf::from("/synthetic/fixtures"),
        output: PathBuf::from("/synthetic/output"),
        oracle: PathBuf::from("/synthetic/oracle"),
        warm: 2,
        measured: 15,
        timeout: Duration::from_secs(1),
    };
    let pairs = FIXTURES.len()
        * config
            .phases()
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>();
    assert_eq!(pairs * 2, config.proof_count().unwrap());
    for _ in 0..pairs {
        timed_pair(
            || log.borrow_mut().push("start"),
            |slot| {
                log.borrow_mut()
                    .push(if slot == 0 { "call0" } else { "call1" });
                Ok(slot)
            },
            |_| log.borrow_mut().push("stop"),
            |slot, _| {
                log.borrow_mut().push(if slot == 0 { "io0" } else { "io1" });
                Ok(())
            },
            |slot, _| {
                log.borrow_mut()
                    .push(if slot == 0 { "check0" } else { "check1" });
                Ok(())
            },
        )
        .unwrap();
    }
    assert_eq!(
        *log.borrow(),
        ["start", "call0", "call1", "stop", "io0", "io1", "check0", "check1"].repeat(pairs)
    );
}

#[test]
fn synthetic_verification_and_io_errors_prevent_pair_acceptance() {
    for fail_check in 0..2 {
        let mut checks = 0;
        let result = timed_pair(
            || (),
            Ok::<_, Box<dyn std::error::Error>>,
            |_| (),
            |_, _| Ok(()),
            |slot, _| {
                checks += 1;
                if slot == fail_check {
                    Err("synthetic verification failure".into())
                } else {
                    Ok(())
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(checks, fail_check + 1);
    }
    let checks = std::cell::Cell::new(0);
    assert!(timed_pair(
        || (),
        Ok::<_, Box<dyn std::error::Error>>,
        |_| (),
        |_, _| Err("synthetic I/O failure".into()),
        |_, _| {
            checks.set(checks.get() + 1);
            Ok(())
        }
    )
    .is_err());
    assert_eq!(checks.get(), 0);
}

#[test]
fn synthetic_second_call_failure_stops_clock_and_retains_first_output() {
    use std::cell::RefCell;
    let log = RefCell::new(Vec::new());
    let result = timed_pair(
        || log.borrow_mut().push("start"),
        |slot| {
            if slot == 0 {
                Ok("synthetic output, not a proof")
            } else {
                Err("synthetic produce failure".into())
            }
        },
        |_| log.borrow_mut().push("stop"),
        |_, _| {
            log.borrow_mut().push("retain-first");
            Ok(())
        },
        |_, _| panic!("no acceptance checks for a failed pair"),
    );
    assert!(result.is_err());
    assert_eq!(*log.borrow(), ["start", "stop", "retain-first"]);
}

#[test]
fn synthetic_oracle_contract_requires_zero_marker_and_no_timeout() {
    assert!(oracle_passed(Some(0), false, b"Result: Valid proof\n", b""));
    assert!(oracle_passed(Some(0), false, b"", b"Result: Valid proof\n"));
    for code in [None, Some(1), Some(-1)] {
        assert!(!oracle_passed(code, false, SUCCESS.as_bytes(), b""));
    }
    assert!(!oracle_passed(Some(0), true, SUCCESS.as_bytes(), b""));
    for text in [
        "",
        "Result: Invalid proof",
        "prefix Result: Valid proof suffix",
    ] {
        assert!(!oracle_passed(Some(0), false, text.as_bytes(), b""));
    }
}

fn synthetic_temp() -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "pair-runner-synthetic-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&path).unwrap();
    path
}

#[test]
fn synthetic_config_rejects_bad_paths_counts_and_overflow() {
    let base = [
        ("G16_ARTIFACTS", "/synthetic/fixtures"),
        ("G16_PAIR_OUTPUT", "/synthetic/output"),
        ("G16_RAPIDSNARK_VERIFY", "/synthetic/oracle"),
    ];
    let parse = |key: &str, value: Option<&str>| {
        Config::parse(|name| {
            if name == key {
                value.map(str::to_owned)
            } else {
                base.iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.to_string())
            }
        })
    };
    let default = parse("unused", None).unwrap();
    assert_eq!(
        (
            default.warm,
            default.measured,
            default.proof_count().unwrap()
        ),
        (2, 15, 68)
    );
    for key in ["G16_ARTIFACTS", "G16_PAIR_OUTPUT", "G16_RAPIDSNARK_VERIFY"] {
        assert!(parse(key, None).is_err());
        assert!(parse(key, Some("relative")).is_err());
    }
    for (key, values) in [
        ("G16_PAIR_WARM", vec!["0", "101", "bad"]),
        (
            "G16_PAIR_MEASURED",
            vec!["0", "14", "1001", "18446744073709551616"],
        ),
        ("G16_PAIR_ORACLE_TIMEOUT_S", vec!["0", "301", "bad"]),
    ] {
        for value in values {
            assert!(parse(key, Some(value)).is_err());
        }
    }
    let mut huge = default;
    huge.warm = usize::MAX;
    assert!(huge.proof_count().is_err());
}

#[test]
fn synthetic_output_is_exclusive_and_inventory_is_strict() {
    let root = synthetic_temp();
    let output = root.join("exclusive");
    let mut evidence = Evidence::new(&output).unwrap();
    evidence
        .bytes(
            "synthetic.txt",
            b"synthetic evidence, not a proof or timing",
        )
        .unwrap();
    assert!(Evidence::new(&output).is_err());
    assert!(evidence.bytes("synthetic.txt", b"replace").is_err());
    assert_eq!(
        std::fs::read(output.join("synthetic.txt")).unwrap(),
        b"synthetic evidence, not a proof or timing"
    );
    evidence.audit(1).unwrap();
    assert!(evidence.audit(2).is_err());
    std::fs::write(output.join("unexpected"), b"synthetic").unwrap();
    assert!(evidence.audit(1).is_err());
    assert!(!output.join("accepted.json").exists());
}

#[test]
fn synthetic_preflight_refuses_missing_input_and_script_oracle_without_cuda() {
    let root = synthetic_temp();
    let mut config = Config {
        artifacts: root.clone(),
        output: root.join("new-output"),
        oracle: root.join("missing-tool"),
        warm: 2,
        measured: 15,
        timeout: Duration::from_secs(1),
    };
    assert!(config.preflight().is_err());
    for name in FIXTURES {
        std::fs::create_dir(root.join(name)).unwrap();
        for file in ["circuit.zkey", "circuit.wtns", "vkey.json"] {
            std::fs::write(
                root.join(name).join(file),
                b"synthetic invalid input, never proved",
            )
            .unwrap();
        }
    }
    assert!(config.preflight().is_err());
    config.oracle = std::env::current_exe().unwrap();
    config.preflight().unwrap();
    assert!(load_fixture(&root, "tiny_mul").is_err());
    assert!(!config.output.exists());
    let script = root.join("synthetic-script");
    std::fs::write(&script, b"#!/bin/sh\nexit 0\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(native_executable(&script).is_err());
}

#[test]
fn synthetic_json_paths_are_escaped_by_existing_writer() {
    let mut bytes = Vec::new();
    let mut json = JsonWriter::new(&mut bytes);
    json.begin_object().unwrap();
    json.field_string("synthetic_path", "a\"b\\c\n").unwrap();
    json.end().unwrap();
    json.finish().unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "{\n \"synthetic_path\": \"a\\\"b\\\\c\\n\"\n}"
    );
}
