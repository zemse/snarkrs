//! A load-time rewrite of circom's `Fr_F1m_exp`, the loop behind every `**` a circuit
//! computes.
//!
//! circom's `Fr_pow` hands it a fixed 32-byte exponent, and the loop walks every byte from
//! the top: eight squarings of the accumulator each, whether or not the byte is zero. So
//! `x ** 3` costs 256 squarings, 248 of them of the Montgomery one, and a circuit that
//! raises to a loop counter (rsa2048's `poly_eval`) spends most of its witness there.
//!
//! The rewrite drops the exponent's zero top bytes before the loop starts, keeping at least
//! one. It is exact: `Fr_F1m_square` ends in a full conditional subtraction, so squaring the
//! canonical one writes back the same bytes, and a skipped byte would have left memory as
//! it found it. It applies only to a body byte for byte the one below, calling the
//! functions named below; anything else compiles as it came.

use std::collections::HashMap;

use wasmparser::{BinaryReader, Encoding, KnownCustom, Name, Parser, Payload, TypeRef};

/// `Fr_F1m_exp(base, scalar, scalarLength, r)` with locals `i` and `b`, as circom 2.1.4 and
/// 2.2.3 emit it (every module in bench/artifacts): copy `base` to 1696, `r = 1`,
/// `i = scalarLength`, then `do { i -= 1; b = scalar[i]; 8 x (square r; if bit, r *= base) }
/// while i != 0`.
const EXP_BODY: [u8; 293] = [
    0x01, 0x02, 0x7f, 0x20, 0x00, 0x41, 0xa0, 0x0d, 0x10, 0x04, 0x20, 0x03, 0x10, 0x1f, 0x20, 0x02,
    0x21, 0x04, 0x02, 0x40, 0x03, 0x40, 0x20, 0x04, 0x41, 0x01, 0x6b, 0x21, 0x04, 0x20, 0x01, 0x20,
    0x04, 0x6a, 0x2d, 0x00, 0x00, 0x21, 0x05, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41,
    0x80, 0x01, 0x4f, 0x04, 0x40, 0x20, 0x05, 0x41, 0x80, 0x01, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d,
    0x20, 0x03, 0x20, 0x03, 0x10, 0x18, 0x0b, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41,
    0xc0, 0x00, 0x4f, 0x04, 0x40, 0x20, 0x05, 0x41, 0xc0, 0x00, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d,
    0x20, 0x03, 0x20, 0x03, 0x10, 0x18, 0x0b, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41,
    0x20, 0x4f, 0x04, 0x40, 0x20, 0x05, 0x41, 0x20, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03,
    0x20, 0x03, 0x10, 0x18, 0x0b, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41, 0x10, 0x4f,
    0x04, 0x40, 0x20, 0x05, 0x41, 0x10, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03, 0x20, 0x03,
    0x10, 0x18, 0x0b, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41, 0x08, 0x4f, 0x04, 0x40,
    0x20, 0x05, 0x41, 0x08, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03, 0x20, 0x03, 0x10, 0x18,
    0x0b, 0x20, 0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41, 0x04, 0x4f, 0x04, 0x40, 0x20, 0x05,
    0x41, 0x04, 0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03, 0x20, 0x03, 0x10, 0x18, 0x0b, 0x20,
    0x03, 0x20, 0x03, 0x10, 0x19, 0x20, 0x05, 0x41, 0x02, 0x4f, 0x04, 0x40, 0x20, 0x05, 0x41, 0x02,
    0x6b, 0x21, 0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03, 0x20, 0x03, 0x10, 0x18, 0x0b, 0x20, 0x03, 0x20,
    0x03, 0x10, 0x19, 0x20, 0x05, 0x41, 0x01, 0x4f, 0x04, 0x40, 0x20, 0x05, 0x41, 0x01, 0x6b, 0x21,
    0x05, 0x41, 0xa0, 0x0d, 0x20, 0x03, 0x20, 0x03, 0x10, 0x18, 0x0b, 0x20, 0x04, 0x45, 0x0d, 0x01,
    0x0c, 0x00, 0x0b, 0x0b, 0x0b,
];

/// The functions [`EXP_BODY`] calls, by the index it calls them at.
const EXP_CALLS: [(&str, u32); 4] = [
    ("Fr_int_copy", 4),
    ("Fr_F1m_mul", 24),
    ("Fr_F1m_square", 25),
    ("Fr_F1m_one", 31),
];

/// Where in [`EXP_BODY`] the prologue goes: after `i = scalarLength`, before the loop, so
/// it reads the bytes the loop's first pass would.
const AT: usize = 18;

/// `while i > 1 && scalar[i - 1] == 0 { i -= 1 }` over locals 1 (`scalar`) and 4 (`i`).
#[rustfmt::skip]
const PROLOGUE: [u8; 35] = [
    0x02, 0x40, 0x03, 0x40, // block loop
    0x20, 0x04, 0x41, 0x01, 0x4d, 0x0d, 0x01, // i <= 1: out
    0x20, 0x01, 0x20, 0x04, 0x6a, 0x41, 0x01, 0x6b, // scalar + i - 1
    0x2d, 0x00, 0x00, 0x0d, 0x01, // load8_u != 0: out
    0x20, 0x04, 0x41, 0x01, 0x6b, 0x21, 0x04, // i -= 1
    0x0c, 0x00, 0x0b, 0x0b, // br 0, end end
];

/// `wasm` with the prologue in `Fr_F1m_exp`, or `None` to compile it as it is: not a module
/// wasmparser reads, no name section, no `Fr_F1m_exp`, or a body other than [`EXP_BODY`].
pub(crate) fn trim_exponent(wasm: &[u8]) -> Option<Vec<u8>> {
    let mut imported = 0u32;
    // Sections are back to back, so one starts where the one before it ends.
    let mut prev_end = 0;
    // The code section's first byte, its contents, and where its first entry starts.
    let mut code = None;
    let mut bodies = Vec::new();
    let mut names = HashMap::new();
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.ok()?;
        match &payload {
            Payload::Version {
                encoding, range, ..
            } => {
                if *encoding != Encoding::Module {
                    return None;
                }
                prev_end = range.end;
            }
            Payload::ImportSection(s) => {
                for i in s.clone() {
                    if matches!(i.ok()?.ty, TypeRef::Func(_)) {
                        imported += 1;
                    }
                }
            }
            Payload::CodeSectionStart { range, .. } => {
                let mut r = BinaryReader::new(&wasm[range.clone()], range.start);
                r.read_var_u32().ok()?;
                code = Some((prev_end, range.clone(), r.original_position()));
            }
            Payload::CodeSectionEntry(body) => bodies.push(body.range()),
            Payload::CustomSection(c) => {
                if let KnownCustom::Name(n) = c.as_known() {
                    for sub in n {
                        if let Name::Function(map) = sub.ok()? {
                            for naming in map {
                                let naming = naming.ok()?;
                                names.insert(naming.name, naming.index);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        if let Some((_, range)) = payload.as_section() {
            prev_end = range.end;
        }
    }

    let (header, contents, first) = code?;
    let exp = names.get("Fr_F1m_exp")?.checked_sub(imported)? as usize;
    let body = bodies.get(exp)?.clone();
    // An entry is its size, then its body, so it starts where the one before it ends.
    let start = match exp {
        0 => first,
        _ => bodies[exp - 1].end,
    };
    if wasm[body.clone()] != EXP_BODY || EXP_CALLS.iter().any(|(n, i)| names.get(n) != Some(i)) {
        return None;
    }

    let grow = PROLOGUE.len();
    let mut out = Vec::with_capacity(wasm.len() + grow + 4);
    out.extend_from_slice(&wasm[..=header]);
    leb128(&mut out, u32::try_from(contents.len() + grow).ok()?);
    out.extend_from_slice(&wasm[contents.start..start]);
    leb128(&mut out, u32::try_from(body.len() + grow).ok()?);
    out.extend_from_slice(&wasm[body.start..body.start + AT]);
    out.extend_from_slice(&PROLOGUE);
    out.extend_from_slice(&wasm[body.start + AT..]);
    Some(out)
}

fn leb128(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use wasmtime::{Engine, Module};

    /// Every `bench/artifacts/<circuit>/*_js/*.wasm`.
    fn wasms() -> Vec<PathBuf> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts");
        let mut out = Vec::new();
        for dir in std::fs::read_dir(root).into_iter().flatten().flatten() {
            for js in std::fs::read_dir(dir.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let name = js.file_name().to_string_lossy().into_owned();
                if let Some(stem) = name.strip_suffix("_js") {
                    let wasm = js.path().join(format!("{stem}.wasm"));
                    if wasm.is_file() {
                        out.push(wasm);
                    }
                }
            }
        }
        out.sort();
        out
    }

    fn first_wasm(test: &str) -> Option<Vec<u8>> {
        let w = wasms()
            .into_iter()
            .next()
            .map(|p| std::fs::read(p).unwrap());
        if w.is_none() {
            eprintln!("SKIPPED {test}: no bench/artifacts");
        }
        w
    }

    #[test]
    fn the_prologue_goes_after_i_is_set() {
        // local.get 2, local.set 4: `i = scalarLength`, then the loop's `block`.
        assert_eq!(
            EXP_BODY[AT - 4..AT + 2],
            [0x20, 0x02, 0x21, 0x04, 0x02, 0x40]
        );
    }

    #[test]
    fn every_artifact_is_rewritten_and_validates() {
        let all = wasms();
        if all.is_empty() {
            eprintln!("SKIPPED every_artifact_is_rewritten_and_validates: no bench/artifacts");
            return;
        }
        let engine = Engine::default();
        for p in &all {
            let wasm = std::fs::read(p).unwrap();
            let out =
                trim_exponent(&wasm).unwrap_or_else(|| panic!("{}: not rewritten", p.display()));
            assert_eq!(out.len(), wasm.len() + PROLOGUE.len(), "{}", p.display());
            Module::validate(&engine, &out).unwrap_or_else(|e| panic!("{}: {e:#}", p.display()));
        }
    }

    #[test]
    fn a_body_one_byte_off_is_left_alone() {
        let Some(mut wasm) = first_wasm("a_body_one_byte_off_is_left_alone") else {
            return;
        };
        let at = wasm
            .windows(EXP_BODY.len())
            .position(|w| w == EXP_BODY)
            .unwrap();
        // The low byte of `i32.const 1696`, so the module stays valid.
        wasm[at + 6] ^= 1;
        Module::validate(&Engine::default(), &wasm).unwrap();
        assert!(trim_exponent(&wasm).is_none());
    }

    #[test]
    fn a_module_without_names_is_left_alone() {
        let Some(wasm) = first_wasm("a_module_without_names_is_left_alone") else {
            return;
        };
        let mut stripped = wasm[..8].to_vec();
        let mut prev_end = 8;
        for payload in Parser::new(0).parse_all(&wasm) {
            let payload = payload.unwrap();
            if let Some((_, range)) = payload.as_section() {
                if !matches!(payload, Payload::CustomSection(_)) {
                    stripped.extend_from_slice(&wasm[prev_end..range.end]);
                }
                prev_end = range.end;
            }
        }
        assert!(stripped.len() < wasm.len());
        Module::validate(&Engine::default(), &stripped).unwrap();
        assert!(trim_exponent(&stripped).is_none());
    }
}
