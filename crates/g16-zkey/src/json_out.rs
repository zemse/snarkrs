//! A streaming JSON writer that lays text out exactly as `bfj.write(name, v, {space: 1})`
//! (bfj 7.1.0) does, which is how every snarkjs `export json` command writes its file (`cli.js:427`,
//! `:476`, `:607`, `:896`).
//!
//! The layout is not `JSON.stringify(v, null, 1)`, and the difference is the empty
//! container: bfj always breaks the line before a closing bracket, so `[]` comes out as
//! `[`, newline, the parent's indent, `]`, and `{}` likewise. Everything else is one space
//! per level, `"key": value`, a comma at the end of every line but the last, and no
//! newline after the final bracket. Strings are escaped the way `JSON.stringify` escapes
//! them, which is also serde_json's escaping: the two short-escape `\b \f \n \r \t`, the
//! rest of the C0 range as lowercase `\u00xx`, and nothing above it.
//!
//! Streaming matters because the inputs do not fit in a `String` comfortably. snarkjs
//! materialises the whole object and still streams the text; a 2^20 zkey is two million
//! G1 triples, around 400 MB of JSON, which this writes through a `BufWriter` without
//! ever holding more than one value.

use std::io::{self, Write};

/// One open container and whether it has emitted a member yet.
struct Frame {
    close: u8,
    empty: bool,
}

/// A bfj-layout JSON writer over any [`Write`]. Values are emitted in call order; the
/// caller is responsible for pairing every `begin_*` with an [`JsonWriter::end`], and a
/// value written after a [`JsonWriter::key`] becomes that key's value.
pub struct JsonWriter<W: Write> {
    out: W,
    stack: Vec<Frame>,
    /// A key has been written and its value has not, so the next value takes no separator.
    after_key: bool,
}

impl<W: Write> JsonWriter<W> {
    pub fn new(out: W) -> Self {
        Self {
            out,
            stack: Vec::new(),
            after_key: false,
        }
    }

    /// The comma, newline and indent that precede a value or a key.
    fn separate(&mut self) -> io::Result<()> {
        if std::mem::take(&mut self.after_key) {
            return Ok(());
        }
        let depth = self.stack.len();
        if let Some(top) = self.stack.last_mut() {
            if !std::mem::take(&mut top.empty) {
                self.out.write_all(b",")?;
            }
            self.out.write_all(b"\n")?;
            indent(&mut self.out, depth)?;
        }
        Ok(())
    }

    fn begin(&mut self, open: u8, close: u8) -> io::Result<()> {
        self.separate()?;
        self.out.write_all(&[open])?;
        self.stack.push(Frame { close, empty: true });
        Ok(())
    }

    pub fn begin_object(&mut self) -> io::Result<()> {
        self.begin(b'{', b'}')
    }

    pub fn begin_array(&mut self) -> io::Result<()> {
        self.begin(b'[', b']')
    }

    /// Close the innermost container. The line break comes before the bracket whether or
    /// not the container held anything, which is the one place bfj and `JSON.stringify`
    /// part ways.
    pub fn end(&mut self) -> io::Result<()> {
        let frame = self
            .stack
            .pop()
            .ok_or_else(|| io::Error::other("json: end with no open container"))?;
        self.out.write_all(b"\n")?;
        indent(&mut self.out, self.stack.len())?;
        self.out.write_all(&[frame.close])
    }

    /// An object member's key. The next value written is its value.
    pub fn key(&mut self, k: &str) -> io::Result<()> {
        self.separate()?;
        write_json_string(&mut self.out, k)?;
        self.out.write_all(b": ")?;
        self.after_key = true;
        Ok(())
    }

    /// A string value, escaped.
    pub fn string(&mut self, s: &str) -> io::Result<()> {
        self.separate()?;
        write_json_string(&mut self.out, s)
    }

    /// A value written verbatim: a number, `true`, `false` or `null`.
    pub fn raw(&mut self, s: &str) -> io::Result<()> {
        self.separate()?;
        self.out.write_all(s.as_bytes())
    }

    /// `key` then a string value.
    pub fn field_string(&mut self, k: &str, v: &str) -> io::Result<()> {
        self.key(k)?;
        self.string(v)
    }

    /// `key` then a verbatim value.
    pub fn field_raw(&mut self, k: &str, v: &str) -> io::Result<()> {
        self.key(k)?;
        self.raw(v)
    }

    /// Flush and hand the sink back. Refuses a document with a container still open,
    /// which would otherwise be a truncated file that looks written.
    pub fn finish(mut self) -> io::Result<W> {
        if !self.stack.is_empty() {
            return Err(io::Error::other(format!(
                "json: {} containers still open",
                self.stack.len()
            )));
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

fn indent<W: Write>(out: &mut W, depth: usize) -> io::Result<()> {
    const SPACES: [u8; 64] = [b' '; 64];
    let mut left = depth;
    while left > 0 {
        let n = left.min(SPACES.len());
        out.write_all(&SPACES[..n])?;
        left -= n;
    }
    Ok(())
}

fn write_json_string<W: Write>(out: &mut W, s: &str) -> io::Result<()> {
    // serde_json's escaping is `JSON.stringify`'s for every string a Rust `str` can hold;
    // the one case they differ on, a lone surrogate, is not representable here.
    serde_json::to_writer(&mut *out, s).map_err(io::Error::other)
}

/// A JS `Number` holding a `u64`, printed as `Number.prototype.toString` would. Exact up
/// to 2^53; past it JS prints the nearest double, and so does Rust's shortest round-trip
/// `f64` formatting, which never switches to an exponent below 1e21.
pub fn js_number(v: u64) -> String {
    if v < (1u64 << 53) {
        v.to_string()
    } else {
        format!("{}", v as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(f: impl FnOnce(&mut JsonWriter<&mut Vec<u8>>) -> io::Result<()>) -> String {
        let mut buf = Vec::new();
        let mut w = JsonWriter::new(&mut buf);
        f(&mut w).unwrap();
        w.finish().unwrap();
        String::from_utf8(buf).unwrap()
    }

    /// The four shapes probed against bfj 7.1.0, the copy snarkjs 0.7.6 resolves: empty
    /// containers at the top and nested, and a mixed object.
    #[test]
    fn matches_bfj_space_one() {
        assert_eq!(
            render(|w| {
                w.begin_array()?;
                w.end()
            }),
            "[\n]"
        );
        assert_eq!(
            render(|w| {
                w.begin_object()?;
                w.end()
            }),
            "{\n}"
        );
        assert_eq!(
            render(|w| {
                w.begin_array()?;
                w.begin_array()?;
                w.end()?;
                w.end()
            }),
            "[\n [\n ]\n]"
        );
        assert_eq!(
            render(|w| {
                w.begin_object()?;
                w.key("x")?;
                w.begin_array()?;
                w.end()?;
                w.key("y")?;
                w.begin_object()?;
                w.end()?;
                w.key("z")?;
                w.begin_array()?;
                w.raw("null")?;
                w.raw("1")?;
                w.string("s\u{1}\"é")?;
                w.end()?;
                w.field_raw("w", "true")?;
                w.end()
            }),
            "{\n \"x\": [\n ],\n \"y\": {\n },\n \"z\": [\n  null,\n  1,\n  \"s\\u0001\\\"é\"\n ],\n \"w\": true\n}"
        );
    }

    #[test]
    fn js_number_rounds_like_a_double() {
        assert_eq!(js_number(0), "0");
        assert_eq!(js_number((1 << 53) - 1), "9007199254740991");
        assert_eq!(js_number(u64::MAX), "18446744073709552000");
    }
}
