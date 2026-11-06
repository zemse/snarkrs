//! Just enough CBOR (RFC 8949) to read the body gnark writes with fxamacker/cbor v2.9.0's
//! core deterministic encoding: a map from field name to value. The reader takes the few
//! fields the solver needs and skips the rest, which on SP1's circuit is 894 MB of debug
//! information, without building it.

/// A CBOR data item's head: its major type and its argument. `arg` is `None` for an
/// indefinite length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Head {
    pub major: u8,
    pub arg: Option<u64>,
}

pub const UINT: u8 = 0;
pub const NINT: u8 = 1;
pub const BYTES: u8 = 2;
pub const TEXT: u8 = 3;
pub const ARRAY: u8 = 4;
pub const MAP: u8 = 5;
pub const TAG: u8 = 6;
pub const SIMPLE: u8 = 7;

/// `null`, the simple value Go's encoder writes for a nil slice, map or pointer.
pub const NULL: Head = Head {
    major: SIMPLE,
    arg: Some(22),
};

/// How deep an indefinite-length item may nest. Definite items are skipped by count and
/// never recurse, so this bounds the stack only for input gnark does not write.
const MAX_DEPTH: usize = 64;

pub struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn byte(&mut self) -> Result<u8, String> {
        let b = *self.bytes.get(self.pos).ok_or("CBOR item past the end")?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8], String> {
        let n = usize::try_from(n).map_err(|_| "CBOR length overflows")?;
        if n > self.bytes.len() - self.pos {
            return Err(format!("CBOR string of {n} bytes past the end"));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn peek(&mut self) -> Result<Head, String> {
        let at = self.pos;
        let h = self.head();
        self.pos = at;
        h
    }

    pub fn head(&mut self) -> Result<Head, String> {
        let b = self.byte()?;
        let (major, info) = (b >> 5, b & 0x1f);
        let arg = match info {
            0..=23 => Some(u64::from(info)),
            24..=27 => {
                let n = 1usize << (info - 24);
                let s = self.take(n as u64)?;
                Some(s.iter().fold(0u64, |v, &b| v << 8 | u64::from(b)))
            }
            31 if matches!(major, BYTES | TEXT | ARRAY | MAP) => None,
            31 if major == SIMPLE => None,
            _ => return Err(format!("CBOR head 0x{b:02x} is not well formed")),
        };
        Ok(Head { major, arg })
    }

    /// The definite length of an item of `major`, or an error naming `what`.
    fn expect(&mut self, major: u8, what: &str) -> Result<u64, String> {
        match self.head()? {
            Head {
                major: m,
                arg: Some(n),
            } if m == major => Ok(n),
            h => Err(format!("{what}: unexpected CBOR item {h:?}")),
        }
    }

    pub fn uint(&mut self, what: &str) -> Result<u64, String> {
        self.expect(UINT, what)
    }

    /// A Go `int`: a CBOR integer of either sign.
    pub fn int(&mut self, what: &str) -> Result<i64, String> {
        match self.head()? {
            Head {
                major: UINT,
                arg: Some(n),
            } => i64::try_from(n).map_err(|_| format!("{what}: {n} overflows an int")),
            Head {
                major: NINT,
                arg: Some(n),
            } => i64::try_from(n)
                .map(|n| -1 - n)
                .map_err(|_| format!("{what}: overflows an int")),
            h => Err(format!("{what}: unexpected CBOR item {h:?}")),
        }
    }

    pub fn text(&mut self, what: &str) -> Result<&'a str, String> {
        let n = self.expect(TEXT, what)?;
        std::str::from_utf8(self.take(n)?).map_err(|_| format!("{what}: not UTF-8"))
    }

    /// The length of an array, or `None` for `null`.
    pub fn array_or_null(&mut self, what: &str) -> Result<Option<u64>, String> {
        if self.peek()? == NULL {
            self.head()?;
            return Ok(None);
        }
        self.expect(ARRAY, what).map(Some)
    }

    /// The number of pairs in a map, or `None` for `null`.
    pub fn map_or_null(&mut self, what: &str) -> Result<Option<u64>, String> {
        if self.peek()? == NULL {
            self.head()?;
            return Ok(None);
        }
        self.expect(MAP, what).map(Some)
    }

    pub fn tag(&mut self, what: &str) -> Result<u64, String> {
        self.expect(TAG, what)
    }

    /// A `[]string`, which Go writes as `null` when it is nil. The count is bounded by the
    /// bytes left, one per string at least, before anything is allocated.
    pub fn strings(&mut self, what: &str) -> Result<Vec<String>, String> {
        let n = self.array_or_null(what)?.unwrap_or(0);
        if n > (self.bytes.len() - self.pos) as u64 {
            return Err(format!("{what}: {n} strings in fewer bytes"));
        }
        (0..n).map(|_| self.text(what).map(str::to_owned)).collect()
    }

    /// Skip one data item, however large. Definite-length containers only add to a count of
    /// items still to skip, so a deep or wide body costs no stack and no allocation.
    pub fn skip(&mut self) -> Result<(), String> {
        self.skip_at(0)
    }

    fn skip_at(&mut self, depth: usize) -> Result<(), String> {
        let mut pending = 1u64;
        while pending > 0 {
            pending -= 1;
            let h = self.head()?;
            match (h.major, h.arg) {
                (UINT | NINT, _) | (SIMPLE, Some(_)) => {}
                (BYTES | TEXT, Some(n)) => {
                    self.take(n)?;
                }
                (ARRAY, Some(n)) => pending = add(pending, n)?,
                (MAP, Some(n)) => {
                    pending = add(pending, n.checked_mul(2).ok_or("CBOR map overflows")?)?
                }
                (TAG, Some(_)) => pending = add(pending, 1)?,
                (BYTES | TEXT | ARRAY | MAP, None) => {
                    if depth == MAX_DEPTH {
                        return Err("CBOR nests too deep".into());
                    }
                    // Items until the break byte; a map's come in pairs, which skipping one
                    // at a time handles the same way.
                    while self.bytes.get(self.pos) != Some(&0xff) {
                        self.skip_at(depth + 1)?;
                    }
                    self.pos += 1;
                }
                (SIMPLE, None) => return Err("CBOR break outside an indefinite item".into()),
                _ => unreachable!("head() only returns these"),
            }
            // Every pending item takes a byte at least, so a count past the bytes left is
            // a lie, and refusing it keeps a hostile count from spinning.
            if pending > (self.bytes.len() - self.pos) as u64 {
                return Err("CBOR container holds more items than bytes".into());
            }
        }
        Ok(())
    }
}

fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "CBOR count overflows".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_skips() {
        // {"a": [1, -2, "x"], "bb": {_ 1: h'00'}, "c": tag(7, null)}
        let bytes = [
            0xa3, 0x61, b'a', 0x83, 0x01, 0x21, 0x61, b'x', 0x62, b'b', b'b', 0xbf, 0x01, 0x41,
            0x00, 0xff, 0x61, b'c', 0xc7, 0xf6,
        ];
        let mut r = Reader::new(&bytes);
        assert_eq!(r.map_or_null("m").unwrap(), Some(3));
        assert_eq!(r.text("k").unwrap(), "a");
        assert_eq!(r.array_or_null("a").unwrap(), Some(3));
        assert_eq!(r.int("i").unwrap(), 1);
        assert_eq!(r.int("i").unwrap(), -2);
        assert_eq!(r.text("s").unwrap(), "x");
        assert_eq!(r.text("k").unwrap(), "bb");
        r.skip().unwrap();
        assert_eq!(r.text("k").unwrap(), "c");
        assert_eq!(r.tag("t").unwrap(), 7);
        assert_eq!(r.array_or_null("n").unwrap(), None);
        assert!(r.is_empty());

        let mut r = Reader::new(&bytes);
        r.skip().unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn refuses_counts_the_bytes_cannot_back() {
        // An array claiming 2^32 items in four bytes.
        assert!(Reader::new(&[0x9a, 0xff, 0xff, 0xff, 0xff]).skip().is_err());
        assert!(Reader::new(&[0x7a, 0, 0, 0, 9, b'a']).skip().is_err());
        assert!(Reader::new(&[0x9a, 0, 1, 0, 0]).strings("s").is_err());
        assert!(Reader::new(&[0xff]).skip().is_err());
    }
}
