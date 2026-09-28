//! circom's native witness generator: the C++ program `circom --c` emits, run as a
//! subprocess.
//!
//! Its `main.cpp` is the whole interface: `<bin> <input.json> <output.wtns>`, with the
//! circuit's constants read from `<argv[0]>.dat`. It is not a careful program, and three of
//! its habits decide how it is run here:
//!
//! - On a usage error it prints the usage line and exits 0. Success is therefore "exited 0
//!   **and** wrote a witness", and the output path is fresh every time so that an old file
//!   left there cannot pass for a new one.
//! - A missing input, an input that is not in the circuit and a failed `assert` all end in
//!   C `assert(false)`, which is SIGABRT. So is a missing `.dat`, through an uncaught
//!   exception, which is why that one is checked before anything is spawned.
//! - Its messages go to stderr and its `log()` output to stdout. Both are left attached to
//!   ours, so they reach whoever ran the command.
//!
//! What it writes is parsed and checked like any other `.wtns` (header, prime,
//! `w[0] == 1`) before it is used or put where the caller asked.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use snarkrs_field::Fr;
use snarkrs_formats::wtns::Witness;

use crate::WitnessError;

/// What a witness generator path holds, told apart by content and not by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Starts with the wasm magic, `\0asm`.
    Wasm,
    /// Anything else that is an executable file.
    Native,
}

/// Wasm by its magic number, native if it is an executable file, an error otherwise.
pub fn detect(path: &Path) -> Result<Kind, WitnessError> {
    let mut f = std::fs::File::open(path).map_err(|e| WitnessError::Js {
        kind: crate::JsKind::Error,
        message: crate::node_open_error(path, &e),
    })?;
    let mut magic = [0u8; 4];
    let n = f.read(&mut magic).unwrap_or(0);
    if n == 4 && &magic == b"\0asm" {
        return Ok(Kind::Wasm);
    }
    let meta = f
        .metadata()
        .map_err(|e| WitnessError::io(format!("reading {}", path.display()), e))?;
    if meta.is_file() && is_executable(&meta) {
        return Ok(Kind::Native);
    }
    Err(WitnessError::Unsupported(format!(
        "{} is neither a wasm module (it does not start with \\0asm) nor an executable \
         file; pass circom's circuit_js/circuit.wasm or the binary circom --c builds",
        path.display()
    )))
}

#[cfg(unix)]
fn is_executable(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn is_executable(_: &std::fs::Metadata) -> bool {
    true
}

/// Run `bin` on `input` and return the witness, with nothing left on disk: the binary
/// writes to a file only this user can read, in the temp directory, which is zeroed and
/// removed before this returns, success or not.
///
/// `n_vars`, when given, is the length the witness must have: the zkey's wire count, so a
/// binary built from another circuit is refused before anything is proved.
pub fn calculate(bin: &Path, input: &Path, n_vars: Option<usize>) -> Result<Vec<Fr>, WitnessError> {
    let tmp = TempFile::private()?;
    run(bin, input, &tmp.path)?;
    let mut bytes = std::fs::read(&tmp.path)
        .map_err(|e| WitnessError::io(format!("reading {}", tmp.path.display()), e))?;
    if bytes.is_empty() {
        return Err(no_output(bin));
    }
    // `from_bytes` zeroes the buffer whether or not it parses.
    let w = Witness::from_bytes(std::mem::take(&mut bytes))
        .map_err(|e| malformed(bin, e))?
        .0;
    if let Some(n) = n_vars {
        if w.len() != n {
            let mut w = w;
            crate::scrub(&mut w);
            return Err(WitnessError::Malformed(format!(
                "{} wrote a witness of {} wires and the zkey has {n}: the binary and the \
                 zkey are not the same circuit",
                bin.display(),
                w.len()
            )));
        }
    }
    Ok(w)
}

/// Run `bin` on `input` and put the witness at `out`. It is written beside `out` under a
/// fresh name, checked, and renamed into place, so `out` is either the new witness or
/// untouched.
pub fn calculate_to_file(bin: &Path, input: &Path, out: &Path) -> Result<(), WitnessError> {
    let dir = match out.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = out
        .file_name()
        .ok_or_else(|| WitnessError::Native(format!("{} is not a file name", out.display())))?;
    let tmp = TempFile {
        path: dir.join(format!(
            ".{}.{}.tmp",
            name.to_string_lossy(),
            unique_suffix()
        )),
    };
    run(bin, input, &tmp.path)?;
    let bytes = match std::fs::read(&tmp.path) {
        Ok(b) if !b.is_empty() => b,
        Ok(_) => return Err(no_output(bin)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(no_output(bin)),
        Err(e) => {
            return Err(WitnessError::io(
                format!("reading {}", tmp.path.display()),
                e,
            ))
        }
    };
    let mut w = Witness::from_bytes(bytes).map_err(|e| malformed(bin, e))?.0;
    crate::scrub(&mut w);
    std::fs::rename(&tmp.path, out)
        .map_err(|e| WitnessError::io(format!("writing {}", out.display()), e))?;
    std::mem::forget(tmp);
    Ok(())
}

/// Spawn the binary with absolute paths and turn how it ended into an error.
fn run(bin: &Path, input: &Path, out: &Path) -> Result<(), WitnessError> {
    let abs = |p: &Path, what: &str| {
        std::path::absolute(p)
            .map_err(|e| WitnessError::io(format!("resolving the {what} {}", p.display()), e))
    };
    let bin = abs(bin, "witness generator")?;
    let input = abs(input, "input")?;
    let out = abs(out, "output")?;
    // `<argv[0]>.dat`, and argv[0] is the absolute path.
    let mut dat = bin.clone().into_os_string();
    dat.push(".dat");
    let dat = PathBuf::from(dat);
    if !dat.is_file() {
        return Err(WitnessError::Native(format!(
            "{} is missing: the circom witness binary loads its circuit from <binary>.dat, \
             and circom --c writes it beside the binary",
            dat.display()
        )));
    }
    if !input.is_file() {
        return Err(WitnessError::Js {
            kind: crate::JsKind::Error,
            message: crate::node_open_error(
                &input,
                &std::io::Error::from(std::io::ErrorKind::NotFound),
            ),
        });
    }
    let status = Command::new(&bin)
        .arg(&input)
        .arg(&out)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| WitnessError::io(format!("running {}", bin.display()), e))?;
    if status.success() {
        return Ok(());
    }
    const CAUSES: &str = "The stock circom binary fails this way when an input is missing, \
                          an input is not a signal of the circuit, or a circuit assert \
                          fails; its own message is above.";
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return Err(WitnessError::Native(format!(
                "the witness generator {} was killed by {}. {CAUSES}",
                bin.display(),
                signal_name(sig)
            )));
        }
    }
    Err(WitnessError::Native(format!(
        "the witness generator {} exited with {}. {CAUSES}",
        bin.display(),
        match status.code() {
            Some(c) => format!("status {c}"),
            None => "no status".into(),
        }
    )))
}

fn no_output(bin: &Path) -> WitnessError {
    WitnessError::Native(format!(
        "the witness generator {} exited 0 but wrote no witness. The stock circom binary \
         does that on a usage error; it takes <input.json> <output.wtns>",
        bin.display()
    ))
}

fn malformed(bin: &Path, e: snarkrs_formats::ZkeyError) -> WitnessError {
    WitnessError::Malformed(format!(
        "the witness generator {} wrote a file that is not a BN254 witness: {e}",
        bin.display()
    ))
}

#[cfg(unix)]
fn signal_name(sig: i32) -> String {
    match sig {
        1 => "SIGHUP".into(),
        2 => "SIGINT".into(),
        4 => "SIGILL".into(),
        6 => "SIGABRT".into(),
        8 => "SIGFPE".into(),
        9 => "SIGKILL".into(),
        10 if cfg!(target_os = "macos") => "SIGBUS".into(),
        7 if cfg!(target_os = "linux") => "SIGBUS".into(),
        11 => "SIGSEGV".into(),
        13 => "SIGPIPE".into(),
        15 => "SIGTERM".into(),
        n => format!("signal {n}"),
    }
}

fn unique_suffix() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!(
        "{}-{nanos}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// A witness file the binary writes and this module removes. On drop its contents are
/// overwritten with zeros and it is unlinked. Best effort: a copy-on-write filesystem may
/// keep the old blocks until they are reused.
struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// A new, empty file in the temp directory that only this user can open.
    fn private() -> Result<Self, WitnessError> {
        let path = std::env::temp_dir().join(format!("snarkrs-witness-{}.wtns", unique_suffix()));
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
        opts.open(&path)
            .map_err(|e| WitnessError::io(format!("creating {}", path.display()), e))?;
        Ok(Self { path })
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if let Ok(meta) = std::fs::metadata(&self.path) {
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open(&self.path) {
                use std::io::Write;
                let mut f = std::io::BufWriter::new(f);
                let zeros = [0u8; 1 << 16];
                let mut left = meta.len();
                while left > 0 {
                    let n = left.min(zeros.len() as u64) as usize;
                    if f.write_all(&zeros[..n]).is_err() {
                        break;
                    }
                    left -= n as u64;
                }
                let _ = f.flush();
                let _ = f.get_ref().sync_data();
            }
        }
        let _ = std::fs::remove_file(&self.path);
    }
}
