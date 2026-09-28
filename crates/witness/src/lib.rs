//! Witness generation for circom circuits: the wasm calculator snarkjs runs, and the
//! native C++ binary circom can emit instead.
//!
//! - [`WitnessCalculator`] (feature `wasm`, on by default) runs `circuit.wasm` on wasmtime
//!   the way circom_runtime 0.1.28 runs it under snarkjs 0.7.6: the same input reading
//!   ([`Input`]), the same error messages, `log()` printed the same way, and a `.wtns` that
//!   is byte for byte the file `snarkjs wtns calculate` writes.
//! - [`native`] runs a circom `--c` binary as a subprocess and reads back what it wrote.
//!
//! Either way the result is `w = (1, public..., private...)` as `Vec<Fr>`, which is what
//! `snarkrs_groth16::prove::prove` takes. Handing it over in memory skips the `.wtns` file
//! entirely.
//!
//! The wasm is circuit code from whoever compiled it, so it runs inside wasmtime's default
//! sandbox and is given the four host functions a circom 2 module imports and nothing else:
//! no filesystem, no clock, no environment.

// Only the wasm calculator reads input JSON; a native binary reads its own.
#[cfg_attr(not(feature = "wasm"), allow(dead_code))]
mod input;
pub mod native;
#[cfg(feature = "wasm")]
mod wasm;

pub use input::{Input, Json};
#[cfg(feature = "wasm")]
pub use wasm::{Console, StdConsole, WitnessCalculator};

use snarkrs_field::{BigInteger, Fr, PrimeField};

/// The JavaScript error class snarkjs' message would carry. snarkjs prints a thrown error
/// as `<class>: <message>`, so the class is part of the line a script sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsKind {
    Error,
    TypeError,
    RangeError,
    SyntaxError,
}

impl JsKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JsKind::Error => "Error",
            JsKind::TypeError => "TypeError",
            JsKind::RangeError => "RangeError",
            JsKind::SyntaxError => "SyntaxError",
        }
    }
}

/// Everything that can stop a witness. `Display` is snarkjs' message, trailing newlines
/// included where circom_runtime puts them; [`WitnessError::snarkjs_line`] is the whole
/// line snarkjs logs.
#[derive(Debug, thiserror::Error)]
pub enum WitnessError {
    /// What `BigInt()`, `JSON.parse`, `Object.keys` or a file open throws.
    #[error("{message}")]
    Js { kind: JsKind, message: String },
    #[error("Signal {0} not found\n")]
    SignalNotFound(String),
    #[error("Not enough values for input signal {0}\n")]
    NotEnoughValues(String),
    #[error("Too many values for input signal {0}\n")]
    TooManyValues(String),
    #[error("Not all inputs have been set. Only {set} out of {total}")]
    NotAllInputsSet { set: u32, total: u32 },
    /// The circuit raised an exception: an assert, a signal set twice, and so on. The
    /// message names the template and the line for an assert.
    #[error("{message}")]
    Circuit { code: i32, message: String },
    /// The module trapped for a reason of its own, or broke circom's calling convention.
    #[error("{0}")]
    Wasm(String),
    /// A module snarkrs does not run: circom 1, another field, or not circom at all.
    #[error("{0}")]
    Unsupported(String),
    /// A witness the module or the binary produced that is not a witness.
    #[error("{0}")]
    Malformed(String),
    /// A native witness binary that could not run or did not finish.
    #[error("{0}")]
    Native(String),
    #[error("{context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl WitnessError {
    /// The JavaScript class snarkjs would report this under.
    pub fn js_name(&self) -> &'static str {
        match self {
            WitnessError::Js { kind, .. } => kind.as_str(),
            _ => "Error",
        }
    }

    /// `logger.error(err)` in snarkjs' cli.js, less the stack trace.
    pub fn snarkjs_line(&self) -> String {
        format!("{}: {self}", self.js_name())
    }

    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        WitnessError::Io {
            context: context.into(),
            source,
        }
    }
}

/// Node's message for a file that cannot be opened, `ENOENT: no such file or directory,
/// open 'input.json'`, for the errors it has a code for.
pub fn node_open_error(path: &std::path::Path, e: &std::io::Error) -> String {
    use std::io::ErrorKind::*;
    let (code, text) = match e.kind() {
        NotFound => ("ENOENT", "no such file or directory"),
        PermissionDenied => ("EACCES", "permission denied"),
        IsADirectory => ("EISDIR", "illegal operation on a directory"),
        _ => return format!("{e}, open '{}'", path.display()),
    };
    format!("{code}: {text}, open '{}'", path.display())
}

/// Read a file as snarkjs does, failing with node's message.
pub fn read_file(path: &std::path::Path) -> Result<Vec<u8>, WitnessError> {
    std::fs::read(path).map_err(|e| WitnessError::Js {
        kind: JsKind::Error,
        message: node_open_error(path, &e),
    })
}

/// The `.wtns` snarkjs writes for `w`: version 2, two sections, the field's byte size and
/// prime, then every value as a plain little-endian integer. Byte for byte what
/// circom_runtime's `calculateWTNSBin` builds.
pub fn wtns_bytes(w: &[Fr]) -> Vec<u8> {
    const N8: usize = 32;
    let mut b = Vec::with_capacity(4 * 11 + N8 + N8 * w.len());
    b.extend_from_slice(b"wtns");
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&(8 + N8 as u64).to_le_bytes());
    b.extend_from_slice(&(N8 as u32).to_le_bytes());
    b.extend_from_slice(&Fr::MODULUS.to_bytes_le());
    b.extend_from_slice(&(w.len() as u32).to_le_bytes());
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&((N8 * w.len()) as u64).to_le_bytes());
    for x in w {
        b.extend_from_slice(&x.into_bigint().to_bytes_le());
    }
    b
}

/// Zero a witness in place, through volatile writes the optimiser cannot drop.
pub fn scrub(w: &mut [Fr]) {
    for x in w.iter_mut() {
        for limb in x.0 .0.iter_mut() {
            // SAFETY: `limb` is a valid, aligned, exclusive `&mut u64`.
            unsafe { core::ptr::write_volatile(limb, 0) };
        }
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// [`wtns_bytes`] to a file. The byte image is zeroed once it is written.
pub fn write_wtns(path: &std::path::Path, w: &[Fr]) -> Result<(), WitnessError> {
    let mut bytes = wtns_bytes(w);
    let r = std::fs::write(path, &bytes)
        .map_err(|e| WitnessError::io(format!("writing {}", path.display()), e));
    zeroize::Zeroize::zeroize(&mut bytes);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round trip through the repo's own reader, which checks the header, the prime and
    /// that w[0] is 1.
    #[test]
    fn wtns_bytes_parse_back() {
        let w = vec![Fr::from(1u64), Fr::from(21u64), -Fr::from(1u64)];
        let bytes = wtns_bytes(&w);
        assert_eq!(bytes.len(), 44 + 32 + 32 * 3);
        let back = snarkrs_formats::wtns::Witness::from_bytes(bytes).unwrap().0;
        assert_eq!(back, w);
    }

    #[test]
    fn errors_print_as_snarkjs_does() {
        assert_eq!(
            WitnessError::TooManyValues("x".into()).snarkjs_line(),
            "Error: Too many values for input signal x\n"
        );
        let e = read_file(std::path::Path::new("definitely/not/here.json")).unwrap_err();
        assert_eq!(
            e.snarkjs_line(),
            "Error: ENOENT: no such file or directory, open 'definitely/not/here.json'"
        );
    }
}
