//! `snarkjs file info`: the section table of any iden3 binfile, as `cli.js:1265-1312`
//! prints it.
//!
//! This is deliberately not built on [`crate::binfile::BinFile`]. The point of the command
//! is to describe a file that may be broken, and `BinFile` refuses exactly those: a
//! section running past the end, a chain that stops short. snarkjs' scan
//! (`binfileutils.js:5-35`) refuses nothing past the magic and the version. It records
//! every `(id, length)` it reads, steps over the payload without looking, and leaves the
//! judging to the printer, which flags three things per section id:
//!
//! * **Duplicates**, "has more than one section definition". Only the first entry of a
//!   duplicated id is printed, and when there is a duplicate the zero-size warning is not
//!   raised at all (`cli.js:1290-1295`).
//! * **A zero size**, which is what an interrupted writer leaves behind, since
//!   `startWriteSection` reserves the length as zero and backfills it at the end.
//! * **A payload past the end of the file**.
//!
//! Output details that are easy to get wrong:
//!
//! * Sections are listed by **id ascending, not file order**: `sections` is a sparse array
//!   indexed by id, and `forEach` visits it in index order. A circom `.r1cs` puts section 2
//!   before section 1 on disk and is still listed 1, 2, 3.
//! * The offset printed is the entry header's, `p - 12`, in lowercase hex.
//! * `Version` and `Bin version` always print `undefined`. They read `fd.version` and
//!   `fd.binVersion`, which fastfile never sets; the container version is only checked.
//! * A flagged row colours its marker, `\x1b[31m !!\x1b[0m`, and each reason goes to
//!   **stderr** as its own red line. Every failure to open or scan the file is caught and
//!   its message printed to stderr, and the command still exits 0.
//!
//! A header that straddles the end of the file is read the way fastfile reads it, which
//! is not zero padding. `readToBuffer` counts the bytes that exist and copies them to the
//! **end** of the destination (`osfile.js:304-309`, `buffDst.set(src, offset+len-r)`),
//! leaving zeros in front. So a file cut two bytes into a `u32` version reads that version
//! shifted left by 16 bits, and a cut inside a section length lands the low word in the
//! high word. Both are reproduced; they are why a short file says "Version not supported"
//! rather than "Invalid File format", and why its sizes come out as multiples of 2^32.

use std::io::{self, Write};
use std::path::Path;

use crate::json_out::js_number;

/// `readBinFile(filename, extension, 2, ...)` (`cli.js:1278`): versions above 2 fail.
const FILE_INFO_MAX_VERSION: u32 = 2;

/// The extensions `fileInfo` accepts, which double as the magic it expects.
const KNOWN_TYPES: [&str; 4] = ["zkey", "r1cs", "ptau", "wtns"];

/// `snarkjs file info <file>`: write what it prints to `stdout` and `stderr`. Never fails on
/// the file itself, only on a sink that cannot be written, because snarkjs catches and
/// prints every error and exits 0.
pub fn file_info<O: Write, E: Write>(
    filename: &str,
    stdout: &mut O,
    stderr: &mut E,
) -> io::Result<()> {
    // `filename.split(".").pop()`: the whole name when there is no dot.
    let extension = filename.rsplit('.').next().unwrap_or(filename);
    if !KNOWN_TYPES.contains(&extension) {
        return writeln!(stderr, "Extension {extension} is not allowed.");
    }
    let bytes = match std::fs::read(Path::new(filename)) {
        Ok(b) => b,
        Err(e) => return writeln!(stderr, "{}", node_open_error(&e, filename)),
    };
    let sections = match scan(&bytes, filename, extension) {
        Ok(s) => s,
        Err(msg) => return writeln!(stderr, "{msg}"),
    };

    writeln!(stdout, "File info for    {filename}")?;
    writeln!(stdout)?;
    writeln!(stdout, "File size:       {} bytes", bytes.len())?;
    writeln!(stdout, "File type:       {extension}")?;
    writeln!(stdout, "Version:         undefined")?;
    writeln!(stdout, "Bin version:     undefined")?;
    writeln!(stdout)?;

    let total = bytes.len() as u128;
    for (id, entries) in &sections {
        let first = entries[0];
        let mut errors = Vec::new();
        if entries.len() > 1 {
            errors.push(format!("Section {id} has more than one section definition"));
        } else if first.size == 0 {
            errors.push(format!(
                "Section {id} size is zero. This could cause false errors in other sections."
            ));
        }
        if first.p + first.size as u128 > total {
            errors.push(format!("Section {id} is out of bounds of the file."));
        }
        let marker = if errors.is_empty() {
            "   ".to_string()
        } else {
            "\x1b[31m !!\x1b[0m".to_string()
        };
        writeln!(
            stdout,
            "section {:>5}{marker} size: {}\toffset: 0x{:x}",
            format!("#{id}"),
            js_number(first.size),
            first.p - 12
        )?;
        for e in errors {
            writeln!(stderr, "\x1b[31m                 > {e}\x1b[0m")?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Entry {
    /// Payload offset. Wider than the file because a lying length can carry the scan far
    /// past it, and every later entry is still recorded at the offset it would have had.
    p: u128,
    size: u64,
}

/// `readBinFile`'s scan, keyed by id so iteration is `forEach` order.
fn scan(
    bytes: &[u8],
    filename: &str,
    extension: &str,
) -> Result<std::collections::BTreeMap<u32, Vec<Entry>>, String> {
    // fastfile's short read: the bytes that exist, right-aligned in a zeroed buffer.
    let at = |pos: u128, n: usize| -> Vec<u8> {
        let mut out = vec![0u8; n];
        let avail = (bytes.len() as u128).saturating_sub(pos).min(n as u128) as usize;
        if avail > 0 {
            let start = pos as usize;
            out[n - avail..].copy_from_slice(&bytes[start..start + avail]);
        }
        out
    };
    let u32_at = |pos: u128| u32::from_le_bytes(at(pos, 4).try_into().expect("4 bytes"));
    if at(0, 4) != extension.as_bytes() {
        return Err(format!("{filename}: Invalid File format"));
    }
    if u32_at(4) > FILE_INFO_MAX_VERSION {
        return Err("Version not supported".into());
    }
    let n_sections = u32_at(8);
    let mut sections: std::collections::BTreeMap<u32, Vec<Entry>> = Default::default();
    let mut pos: u128 = 12;
    for i in 0..n_sections {
        if pos >= bytes.len() as u128 {
            // Every entry from here on reads as id 0, size 0, each 12 bytes after the last,
            // and only the first of them and the fact of a second can reach the output. A
            // lying `nSections` of 2^32 - 1 would otherwise spin here for minutes.
            let left = n_sections - i;
            let zeros = sections.entry(0).or_default();
            for k in 0..left.min(2) as u128 {
                zeros.push(Entry {
                    p: pos + 12 * (k + 1),
                    size: 0,
                });
            }
            break;
        }
        let id = u32_at(pos);
        let size = u64::from_le_bytes(at(pos + 4, 8).try_into().expect("8 bytes"));
        pos += 12;
        // `2^32 - 1` is not an array index in JS, so `forEach` would never visit it.
        if id != u32::MAX {
            sections.entry(id).or_default().push(Entry { p: pos, size });
        }
        pos += size as u128;
    }
    Ok(sections)
}

/// The message Node puts on a failed `fs.promises.open`, which fastfile passes through.
fn node_open_error(e: &io::Error, filename: &str) -> String {
    let code = match e.kind() {
        io::ErrorKind::NotFound => "ENOENT: no such file or directory",
        io::ErrorKind::PermissionDenied => "EACCES: permission denied",
        io::ErrorKind::IsADirectory => "EISDIR: illegal operation on a directory",
        _ => return e.to_string(),
    };
    format!("{code}, open '{filename}'")
}
