//! circom 2's wasm witness calculator on wasmtime, after circom_runtime 0.1.28's
//! `WitnessCalculatorCircom2` (`js/witness_calculator.js`).
//!
//! The protocol is small. The host writes each input value into the module's shared
//! read/write buffer a 32-bit word at a time and calls `setInputSignal` with the FNV hash of
//! the signal's name and the index into it. Setting the last input runs the circuit, inside
//! that call. `getWitness(i)` then puts wire `i` in the buffer. The module calls back into
//! the host to report an exception (`exceptionHandler`), to hand over an error message
//! (`printErrorMessage`) and to print a `log()` (`writeBufferMessage`,
//! `showSharedRWMemory`), and those four are all it may import.

use std::sync::Arc;

use g16_field::{BigInteger, Fr, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use wasmtime::{Caller, Config, Engine, Instance, InstancePre, Linker, Module, Store, TypedFunc};

use crate::input::{flatten, fnv_hash, to_bigint};
use crate::{Input, WitnessError};

/// Where a circuit's `log()` lines and its exception reports go. snarkjs sends the first
/// to `console.log` and the second to `console.error`, which is what [`StdConsole`] does.
pub trait Console: Send + Sync {
    /// One `log()` call, its items joined by spaces.
    fn log(&self, line: &str);
    /// circom_runtime's `ERROR:  <code> <message>` report, printed before the exception is
    /// returned as an error.
    fn error(&self, line: &str);
}

/// stdout and stderr, as snarkjs prints them.
pub struct StdConsole;

impl Console for StdConsole {
    fn log(&self, line: &str) {
        use std::io::Write;
        let _ = writeln!(std::io::stdout().lock(), "{line}");
    }
    fn error(&self, line: &str) {
        use std::io::Write;
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    }
}

/// The only imports a circom 2 module has, all under `runtime`.
const IMPORTS: [&str; 4] = [
    "exceptionHandler",
    "printErrorMessage",
    "writeBufferMessage",
    "showSharedRWMemory",
];

/// A compiled `circuit.wasm`, ready to compute witnesses. Compile once and call
/// [`calculate`](Self::calculate) per input: each call gets a fresh instance, so nothing of
/// one witness is left in the module's memory for the next.
pub struct WitnessCalculator {
    engine: Engine,
    pre: InstancePre<Host>,
    console: Arc<dyn Console>,
}

struct Host {
    console: Arc<dyn Console>,
    /// `msgStr`: the `log()` line being assembled.
    msg: String,
    /// `errStr`: messages the module handed over ahead of an exception.
    err: String,
    /// major, minor, patch. Decides how `showSharedRWMemory` prints.
    version: (i32, i32, i32),
}

/// An exception the circuit raised through `exceptionHandler`, carried out of the wasm
/// call as the error that ends it.
#[derive(Debug)]
struct Thrown {
    code: i32,
    message: String,
}

impl std::fmt::Display for Thrown {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Thrown {}

impl WitnessCalculator {
    /// Compile a circom 2 `circuit.wasm`. A circom 1 module, or a module that imports
    /// anything circom 2 does not, is refused here rather than at the first witness.
    pub fn new(wasm: &[u8]) -> Result<Self, WitnessError> {
        // wasmtime's defaults: bounds-checked linear memory behind guard pages, no fuel, no
        // WASI. The module gets the host functions below and nothing else.
        let engine = Engine::new(&Config::new()).map_err(wasm_err)?;
        let module = Module::new(&engine, wasm)
            .map_err(|e| WitnessError::Wasm(format!("CompileError: {e:#}")))?;

        if module.get_export("getVersion").is_none() {
            let circom1 = module.get_export("getNVars").is_some();
            return Err(WitnessError::Unsupported(if circom1 {
                "this wasm was compiled by circom 1, which snarkrs does not run: recompile the \
                 circuit with circom 2"
                    .into()
            } else {
                "this wasm is not a circom witness calculator: it has no getVersion export".into()
            }));
        }
        for i in module.imports() {
            if i.module() != "runtime" || !IMPORTS.contains(&i.name()) {
                return Err(WitnessError::Unsupported(format!(
                    "this wasm imports {}.{}, which no circom 2 witness calculator does; it \
                     gets circom's own imports and nothing else",
                    i.module(),
                    i.name()
                )));
            }
        }

        let mut linker = Linker::<Host>::new(&engine);
        linker
            .func_wrap(
                "runtime",
                "exceptionHandler",
                |caller: Caller<'_, Host>, code: i32| -> wasmtime::Result<()> {
                    let err = match code {
                        1 => "Signal not found. ",
                        2 => "Too many signals set. ",
                        3 => "Signal already set. ",
                        4 => "Assert Failed. ",
                        5 => "Not enough memory. ",
                        6 => "Input signal array access exceeds the size. ",
                        _ => "Unknown error. ",
                    };
                    let host = caller.data();
                    // `console.error("ERROR: ", code, errStr)`: three items, space separated.
                    host.console.error(&format!("ERROR:  {code} {}", host.err));
                    Err(wasmtime::Error::new(Thrown {
                        code,
                        message: format!("{err}{}", host.err),
                    }))
                },
            )
            .map_err(wasm_err)?;
        linker
            .func_wrap(
                "runtime",
                "printErrorMessage",
                |mut caller: Caller<'_, Host>| -> wasmtime::Result<()> {
                    let m = message(&mut caller)?;
                    caller.data_mut().err.push_str(&m);
                    caller.data_mut().err.push('\n');
                    Ok(())
                },
            )
            .map_err(wasm_err)?;
        linker
            .func_wrap(
                "runtime",
                "writeBufferMessage",
                |mut caller: Caller<'_, Host>| -> wasmtime::Result<()> {
                    let m = message(&mut caller)?;
                    let host = caller.data_mut();
                    // Every `log()` ends with a lone "\n", which is when the line prints.
                    if m == "\n" {
                        let line = std::mem::take(&mut host.msg);
                        host.console.log(&line);
                    } else {
                        if !host.msg.is_empty() {
                            host.msg.push(' ');
                        }
                        host.msg.push_str(&m);
                    }
                    Ok(())
                },
            )
            .map_err(wasm_err)?;
        linker
            .func_wrap(
                "runtime",
                "showSharedRWMemory",
                |mut caller: Caller<'_, Host>| -> wasmtime::Result<()> {
                    let n32 = export::<(), i32>(&mut caller, "getFieldNumLen32")?
                        .call(&mut caller, ())? as u32;
                    let read = export::<i32, i32>(&mut caller, "readSharedRWMemory")?;
                    let mut words = Vec::with_capacity(n32 as usize);
                    for j in 0..n32 {
                        words.push(read.call(&mut caller, j as i32)? as u32);
                    }
                    let value = BigUint::from_slice(&words).to_string();
                    let host = caller.data_mut();
                    // circom 2.0.7 made `log()` take several items; before it, each value
                    // printed on a line of its own.
                    let (major, minor, patch) = host.version;
                    if major >= 2 && (minor >= 1 || patch >= 7) {
                        if !host.msg.is_empty() {
                            host.msg.push(' ');
                        }
                        host.msg.push_str(&value);
                    } else {
                        host.console.log(&value);
                    }
                    Ok(())
                },
            )
            .map_err(wasm_err)?;
        let pre = linker.instantiate_pre(&module).map_err(wasm_err)?;
        Ok(Self {
            engine,
            pre,
            console: Arc::new(StdConsole),
        })
    }

    /// [`new`](Self::new) from a file, failing with node's message if it cannot be read.
    pub fn from_file(path: &std::path::Path) -> Result<Self, WitnessError> {
        Self::new(&crate::read_file(path)?)
    }

    /// Send `log()` output and exception reports somewhere other than stdout and stderr.
    pub fn with_console(mut self, console: Arc<dyn Console>) -> Self {
        self.console = console;
        self
    }

    /// The witness for `input`: `w[0] = 1`, then the public signals, then the rest.
    pub fn calculate(&self, input: &Input) -> Result<Vec<Fr>, WitnessError> {
        self.run(input, false)
    }

    /// [`calculate`](Self::calculate) with the module's own sanity checks on, which is what
    /// `snarkjs wtns debug` asks for (`init(1)`).
    pub fn calculate_sanity_checked(&self, input: &Input) -> Result<Vec<Fr>, WitnessError> {
        self.run(input, true)
    }

    fn run(&self, input: &Input, sanity_check: bool) -> Result<Vec<Fr>, WitnessError> {
        let mut store = Store::new(
            &self.engine,
            Host {
                console: self.console.clone(),
                msg: String::new(),
                err: String::new(),
                version: (1, 0, 0),
            },
        );
        let instance = self.pre.instantiate(&mut store).map_err(wasm_err)?;
        let result = Session::new(&mut store, &instance)
            .and_then(|mut s| s.witness(&mut store, input, sanity_check));
        // The inputs and every wire are in the module's memory. It is unmapped when the
        // store drops, but not before whatever runs next in this process.
        if let Some(m) = instance.get_memory(&mut store, "memory") {
            m.data_mut(&mut store).fill(0);
        }
        result
    }
}

/// One instance's exports, typed.
struct Session {
    n32: usize,
    shared: Option<usize>,
    read: TypedFunc<i32, i32>,
    write: TypedFunc<(i32, i32), ()>,
    init: TypedFunc<i32, ()>,
    set_input: TypedFunc<(i32, i32, i32), ()>,
    input_size_of: Option<TypedFunc<(i32, i32), i32>>,
    input_size: TypedFunc<(), i32>,
    witness_size: TypedFunc<(), i32>,
    get_witness: TypedFunc<i32, ()>,
    memory: Option<wasmtime::Memory>,
}

impl Session {
    fn new(store: &mut Store<Host>, inst: &Instance) -> Result<Self, WitnessError> {
        let version = |store: &mut Store<Host>, name| -> Result<Option<i32>, WitnessError> {
            match inst.get_func(&mut *store, name) {
                None => Ok(None),
                Some(f) => {
                    let f = f.typed::<(), i32>(&*store).map_err(wasm_err)?;
                    f.call(&mut *store, ())
                        .map(Some)
                        .map_err(|e| call_err(e, false))
                }
            }
        };
        let major = version(store, "getVersion")?.unwrap_or(1);
        let minor = version(store, "getMinorVersion")?.unwrap_or(0);
        let patch = version(store, "getPatchVersion")?.unwrap_or(0);
        store.data_mut().version = (major, minor, patch);
        if major != 2 {
            return Err(WitnessError::Unsupported(format!(
                "Unsupported circom version: {major}"
            )));
        }

        let s = Session {
            n32: 0,
            shared: None,
            read: typed(store, inst, "readSharedRWMemory")?,
            write: typed(store, inst, "writeSharedRWMemory")?,
            init: typed(store, inst, "init")?,
            set_input: typed(store, inst, "setInputSignal")?,
            input_size_of: match inst.get_func(&mut *store, "getInputSignalSize") {
                Some(_) => Some(typed(store, inst, "getInputSignalSize")?),
                None => None,
            },
            input_size: typed(store, inst, "getInputSize")?,
            witness_size: typed(store, inst, "getWitnessSize")?,
            get_witness: typed(store, inst, "getWitness")?,
            memory: inst.get_memory(&mut *store, "memory"),
        };
        let call0 = |store: &mut Store<Host>, name| -> Result<i32, WitnessError> {
            typed::<(), i32>(store, inst, name)?
                .call(&mut *store, ())
                .map_err(|e| call_err(e, false))
        };
        let n32 = call0(store, "getFieldNumLen32")? as usize;
        typed::<(), ()>(store, inst, "getRawPrime")?
            .call(&mut *store, ())
            .map_err(|e| call_err(e, false))?;
        let mut prime = Vec::with_capacity(n32);
        for j in 0..n32 {
            prime.push(
                s.read
                    .call(&mut *store, j as i32)
                    .map_err(|e| call_err(e, false))? as u32,
            );
        }
        if BigUint::from_slice(&prime) != BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le()) {
            return Err(WitnessError::Unsupported(format!(
                "this circuit is over the prime {}, not BN254's scalar field; snarkrs is \
                 Groth16 on BN254 only",
                BigUint::from_slice(&prime)
            )));
        }
        // Where `readSharedRWMemory(j)` reads, so a wire can be copied out of memory rather
        // than fetched one call per word.
        let shared = match inst.get_func(&mut *store, "getSharedRWMemoryStart") {
            Some(_) => Some(call0(store, "getSharedRWMemoryStart")? as u32 as usize),
            None => None,
        };
        Ok(Session { n32, shared, ..s })
    }

    fn witness(
        &mut self,
        store: &mut Store<Host>,
        input: &Input,
        sanity_check: bool,
    ) -> Result<Vec<Fr>, WitnessError> {
        let p = BigInt::from_biguint(
            Sign::Plus,
            BigUint::from_bytes_le(&Fr::MODULUS.to_bytes_le()),
        );
        self.init
            .call(&mut *store, sanity_check as i32)
            .map_err(|e| call_err(e, false))?;
        let mut set = 0u32;
        for (name, value) in &input.signals {
            let (msb, lsb) = fnv_hash(name);
            let (msb, lsb) = (msb as i32, lsb as i32);
            let values = flatten(value);
            if let Some(size_of) = &self.input_size_of {
                let size = size_of
                    .call(&mut *store, (msb, lsb))
                    .map_err(|e| call_err(e, false))?;
                if size < 0 {
                    return Err(WitnessError::SignalNotFound(name.clone()));
                }
                if values.len() < size as usize {
                    return Err(WitnessError::NotEnoughValues(name.clone()));
                }
                if values.len() > size as usize {
                    return Err(WitnessError::TooManyValues(name.clone()));
                }
            }
            for (i, v) in values.into_iter().enumerate() {
                // `normalize`: BigInt's `%` keeps the dividend's sign, hence the lift.
                let mut n = to_bigint(v)? % &p;
                if n.sign() == Sign::Minus {
                    n += &p;
                }
                let mut words = n.magnitude().to_u32_digits();
                words.resize(self.n32, 0);
                for (j, w) in words.iter().enumerate() {
                    self.write
                        .call(&mut *store, (j as i32, *w as i32))
                        .map_err(|e| call_err(e, false))?;
                }
                self.set_input
                    .call(&mut *store, (msb, lsb, i as i32))
                    .map_err(|e| call_err(e, true))?;
                set += 1;
            }
        }
        let total = self
            .input_size
            .call(&mut *store, ())
            .map_err(|e| call_err(e, false))? as u32;
        if set < total {
            return Err(WitnessError::NotAllInputsSet { set, total });
        }

        let n = self
            .witness_size
            .call(&mut *store, ())
            .map_err(|e| call_err(e, false))? as u32 as usize;
        let mut w = Vec::with_capacity(n);
        let mut limbs = vec![0u32; self.n32];
        for i in 0..n {
            self.get_witness
                .call(&mut *store, i as i32)
                .map_err(|e| call_err(e, false))?;
            match (self.shared, self.memory) {
                (Some(at), Some(mem)) => {
                    let bytes = mem
                        .data(&*store)
                        .get(at..at + 4 * self.n32)
                        .ok_or_else(|| {
                            WitnessError::Wasm(
                                "the shared buffer lies outside the module's memory".into(),
                            )
                        })?;
                    for (l, b) in limbs.iter_mut().zip(bytes.chunks_exact(4)) {
                        *l = u32::from_le_bytes(b.try_into().unwrap());
                    }
                }
                _ => {
                    for (j, l) in limbs.iter_mut().enumerate() {
                        *l = self
                            .read
                            .call(&mut *store, j as i32)
                            .map_err(|e| call_err(e, false))? as u32;
                    }
                }
            }
            let mut u = [0u64; 4];
            for (k, pair) in limbs.chunks(2).enumerate() {
                u[k] = pair[0] as u64 | (pair.get(1).copied().unwrap_or(0) as u64) << 32;
            }
            let big = <Fr as PrimeField>::BigInt::new(u);
            let x = Fr::from_bigint(big).ok_or_else(|| {
                WitnessError::Malformed(format!(
                    "wire {i} of the witness is not below the field's prime"
                ))
            })?;
            w.push(x);
        }
        limbs.fill(0);
        if w.first() != Some(&Fr::from(1u64)) {
            return Err(WitnessError::Malformed("witness[0] is not 1".into()));
        }
        Ok(w)
    }
}

fn typed<P, R>(
    store: &mut Store<Host>,
    inst: &Instance,
    name: &str,
) -> Result<TypedFunc<P, R>, WitnessError>
where
    P: wasmtime::WasmParams,
    R: wasmtime::WasmResults,
{
    inst.get_typed_func::<P, R>(&mut *store, name)
        .map_err(|e| WitnessError::Unsupported(format!("not a circom 2 module: {name}: {e:#}")))
}

/// An export of the calling instance, from inside a host function.
fn export<P, R>(caller: &mut Caller<'_, Host>, name: &str) -> wasmtime::Result<TypedFunc<P, R>>
where
    P: wasmtime::WasmParams,
    R: wasmtime::WasmResults,
{
    caller
        .get_export(name)
        .and_then(|e| e.into_func())
        .ok_or_else(|| wasmtime::Error::msg(format!("the module does not export {name}")))?
        .typed::<P, R>(&*caller)
}

/// `getMessage`: `getMessageChar` until it returns 0, one UTF-16 unit each.
fn message(caller: &mut Caller<'_, Host>) -> wasmtime::Result<String> {
    let next = export::<(), i32>(caller, "getMessageChar")?;
    let mut units = Vec::new();
    loop {
        let c = next.call(&mut *caller, ())?;
        if c == 0 {
            break;
        }
        units.push(c as u16);
    }
    Ok(String::from_utf16_lossy(&units))
}

fn wasm_err(e: wasmtime::Error) -> WitnessError {
    WitnessError::Wasm(format!("{e:#}"))
}

/// A failed call into the module. An exception thrown inside `setInputSignal` reaches
/// snarkjs through `throw new Error(err)`, which prefixes its text with `Error: `.
fn call_err(e: wasmtime::Error, in_set_input: bool) -> WitnessError {
    let wrap = if in_set_input { "Error: " } else { "" };
    if let Some(t) = e.downcast_ref::<Thrown>() {
        return WitnessError::Circuit {
            code: t.code,
            message: format!("{wrap}{}", t.message),
        };
    }
    match e.downcast_ref::<wasmtime::Trap>() {
        Some(trap) => WitnessError::Wasm(format!("{wrap}RuntimeError: {trap}")),
        None => WitnessError::Wasm(format!("{wrap}{e:#}")),
    }
}
