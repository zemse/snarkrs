//! The decoder for `github.com/ronanh/intcomp` v1.1.1, which gnark uses for its levels and
//! its four instruction columns (`internal/backend/ioutils/intcomp.go`).
//!
//! A stream is a run of blocks then one trailing word, the last block's compressed length,
//! which the decoder ignores. A block whose first word (its value count) is below the block
//! size is delta variable-byte coded; any other block is delta bit-packed in groups of 32
//! (u32) or 64 (u64) values, a group at a bit width of its own and zigzagged when a delta in
//! it is negative. A group at the full width holds its values verbatim, not their deltas.
//! Everything wraps, as Go's unsigned arithmetic does.
//!
//! Every count is checked against the words that are actually there before anything is
//! allocated, so a corrupt stream is an error and never an allocation the file cannot back.

/// A u32 stream, as `intcomp.UncompressUint32` decodes it, refusing one of more than `max`
/// values before it allocates them.
pub fn uncompress_u32(input: &[u32], max: usize) -> Result<Vec<u32>, String> {
    let Some((_, input)) = input.split_last() else {
        return Ok(Vec::new());
    };
    let out_len = total_len(input, |w| (w[0] as usize, w[1] as usize), 2, BLOCK_32)?;
    if out_len > max {
        return Err(format!(
            "intcomp stream of {out_len} values, at most {max} expected"
        ));
    }
    let mut out = Vec::with_capacity(out_len);
    let mut rest = input;
    while !rest.is_empty() {
        let (n, words) = (rest[0] as usize, rest[1] as usize);
        let (block, tail) = rest.split_at(words);
        if n < BLOCK_32 {
            var_byte(&block[2..], n, &mut out, 4, |w, i| {
                (w[i / 4] >> (24 - 8 * (i % 4))) as u8
            })?;
        } else {
            bit_packed_32(block, n, &mut out)?;
        }
        rest = tail;
    }
    Ok(out)
}

/// A u64 stream, as `intcomp.UncompressUint64` decodes it, with the same `max`. A block
/// header packs the value count into the low 32 bits (read as an int32) and the word count
/// into the high 32.
pub fn uncompress_u64(input: &[u64], max: usize) -> Result<Vec<u64>, String> {
    let Some((_, input)) = input.split_last() else {
        return Ok(Vec::new());
    };
    let header = |w: &[u64]| (w[0] as u32 as i32, (w[0] >> 32) as usize);
    let out_len = total_len(
        input,
        |w| {
            let (n, words) = header(w);
            (usize::try_from(n).unwrap_or(usize::MAX), words)
        },
        1,
        BLOCK_64,
    )?;
    if out_len > max {
        return Err(format!(
            "intcomp stream of {out_len} values, at most {max} expected"
        ));
    }
    let mut out = Vec::with_capacity(out_len);
    let mut rest = input;
    while !rest.is_empty() {
        let (n, words) = header(rest);
        let n = n as usize;
        let (block, tail) = rest.split_at(words);
        if n < BLOCK_64 {
            var_byte(&block[1..], n, &mut out, 8, |w, i| {
                (w[i / 8] >> (56 - 8 * (i % 8))) as u8
            })?;
        } else {
            bit_packed_64(block, n, &mut out)?;
        }
        rest = tail;
    }
    Ok(out)
}

const BLOCK_32: usize = 128;
const BLOCK_64: usize = 256;

/// Walk the block headers: every block must lie inside `input`, and the values it claims
/// must fit the words it has, so the sum bounds an allocation by the input's own size.
/// `header_words` is the shorter of the two headers; a word past it holds a byte of a
/// variable-byte value at least, or opens a bit-packed block of `block` values.
fn total_len<T>(
    input: &[T],
    header: impl Fn(&[T]) -> (usize, usize),
    header_words: usize,
    block: usize,
) -> Result<usize, String> {
    let mut at = 0;
    let mut total = 0usize;
    while at < input.len() {
        let rest = &input[at..];
        if rest.len() < header_words {
            return Err("intcomp block header past the end".into());
        }
        let (n, words) = header(rest);
        if words <= header_words || words > rest.len() {
            return Err(format!("intcomp block of {words} words at word {at}"));
        }
        if n / block >= words - header_words {
            return Err(format!("intcomp block claims {n} values in {words} words"));
        }
        total += n;
        at += words;
    }
    Ok(total)
}

/// One delta variable-byte block of `n` values: 7 bits a byte, least significant group
/// first, a clear top bit ending a value. Bytes come out of each word most significant
/// first, and the block ends in 0x80 padding that never terminates a value.
fn var_byte<W, T: Copy + Default + VarByteValue>(
    words: &[W],
    n: usize,
    out: &mut Vec<T>,
    bytes_per_word: usize,
    byte: impl Fn(&[W], usize) -> u8,
) -> Result<(), String> {
    let start = out.len();
    let mut prev = T::default();
    let mut delta = 0u64;
    let mut shift = 0u32;
    for i in 0..words.len() * bytes_per_word {
        let c = byte(words, i);
        if shift < 64 {
            delta |= u64::from(c & 0x7f) << shift;
        }
        shift += 7;
        if c & 0x80 == 0 {
            if out.len() - start == n {
                return Err("intcomp variable-byte block holds more values than it claims".into());
            }
            prev = prev.wrapping_add_u64(delta);
            out.push(prev);
            delta = 0;
            shift = 0;
        }
    }
    if out.len() - start != n {
        return Err("intcomp variable-byte block holds fewer values than it claims".into());
    }
    Ok(())
}

trait VarByteValue {
    fn wrapping_add_u64(self, delta: u64) -> Self;
}

impl VarByteValue for u32 {
    fn wrapping_add_u64(self, delta: u64) -> Self {
        // Go accumulates the delta in a uint32, so the bits past 32 never existed.
        self.wrapping_add(delta as u32)
    }
}

impl VarByteValue for u64 {
    fn wrapping_add_u64(self, delta: u64) -> Self {
        self.wrapping_add(delta)
    }
}

/// The four groups' widths and zigzag flags out of a block header word.
fn group_headers(word: u32) -> [(usize, bool); 4] {
    [24, 16, 8, 0].map(|s| (((word >> s) & 0x7f) as usize, (word >> (s + 7)) & 1 == 1))
}

fn bit_packed_32(block: &[u32], n: usize, out: &mut Vec<u32>) -> Result<(), String> {
    if !n.is_multiple_of(BLOCK_32) || block.len() < 3 {
        return Err(format!("intcomp bit-packed block of {n} values"));
    }
    let mut prev = block[2];
    let mut at = 3;
    for _ in 0..n / BLOCK_32 {
        let word = *block
            .get(at)
            .ok_or("intcomp bit-packed block past its end")?;
        at += 1;
        for (bits, zigzag) in group_headers(word) {
            if bits > 32 || at + bits > block.len() {
                return Err(format!("intcomp group of width {bits}"));
            }
            let packed = &block[at..at + bits];
            at += bits;
            for i in 0..32 {
                let v = if bits == 32 {
                    packed[i]
                } else {
                    let raw = extract(packed, i * bits, bits) as u32;
                    let d = if zigzag {
                        (raw >> 1) ^ (raw & 1).wrapping_neg()
                    } else {
                        raw
                    };
                    prev.wrapping_add(d)
                };
                out.push(v);
                prev = v;
            }
        }
    }
    if at != block.len() {
        return Err("intcomp bit-packed block has words left over".into());
    }
    Ok(())
}

fn bit_packed_64(block: &[u64], n: usize, out: &mut Vec<u64>) -> Result<(), String> {
    if !n.is_multiple_of(BLOCK_64) || block.len() < 2 {
        return Err(format!("intcomp bit-packed block of {n} values"));
    }
    let mut prev = block[1];
    let mut at = 2;
    for _ in 0..n / BLOCK_64 {
        let word = *block
            .get(at)
            .ok_or("intcomp bit-packed block past its end")?;
        at += 1;
        for (bits, zigzag) in group_headers(word as u32) {
            if bits > 64 || at + bits > block.len() {
                return Err(format!("intcomp group of width {bits}"));
            }
            let packed = &block[at..at + bits];
            at += bits;
            for i in 0..64 {
                let v = if bits == 64 {
                    packed[i]
                } else {
                    let raw = extract(packed, i * bits, bits);
                    let d = if zigzag {
                        (raw >> 1) ^ (raw & 1).wrapping_neg()
                    } else {
                        raw
                    };
                    prev.wrapping_add(d)
                };
                out.push(v);
                prev = v;
            }
        }
    }
    if at != block.len() {
        return Err("intcomp bit-packed block has words left over".into());
    }
    Ok(())
}

/// `bits` bits (below the word width) starting at bit `pos` of a little-endian run of words.
fn extract<W: Copy + Into<u64>>(words: &[W], pos: usize, bits: usize) -> u64 {
    if bits == 0 {
        return 0;
    }
    let width = std::mem::size_of::<W>() * 8;
    let (w, off) = (pos / width, pos % width);
    let mut v = words[w].into() >> off;
    if off + bits > width {
        v |= words[w + 1].into() << (width - off);
    }
    v & ((1u64 << bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoder side of both block kinds, written from intcomp's compressor, so the
    /// decoder is tested on streams it did not produce itself.
    fn compress_u32(input: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        if input.is_empty() {
            return out;
        }
        let full = input.len() / BLOCK_32 * BLOCK_32;
        let mut last_header = 0;
        if full > 0 {
            last_header = out.len();
            out.extend([full as u32, 0, input[0]]);
            let mut prev = input[0];
            for block in input[..full].chunks(BLOCK_32) {
                let mut header = 0u32;
                let mut groups = Vec::new();
                for (g, group) in block.chunks(32).enumerate() {
                    let deltas: Vec<i32> = group
                        .iter()
                        .scan(prev, |p, &v| {
                            let d = v.wrapping_sub(*p) as i32;
                            *p = v;
                            Some(d)
                        })
                        .collect();
                    prev = group[31];
                    let mask = deltas
                        .iter()
                        .fold(0u32, |m, &d| m | ((d << 1) ^ (d >> 31)) as u32);
                    let sign = mask & 1;
                    let bits = (32 - (mask >> 1).leading_zeros()) + sign;
                    header |= ((sign << 7) | bits) << (24 - 8 * g);
                    let vals: Vec<u32> = if bits == 32 {
                        group.to_vec()
                    } else if sign == 1 {
                        deltas
                            .iter()
                            .map(|&d| ((d << 1) ^ (d >> 31)) as u32)
                            .collect()
                    } else {
                        deltas.iter().map(|&d| d as u32).collect()
                    };
                    let mut packed = vec![0u32; bits as usize];
                    for (i, &v) in vals.iter().enumerate() {
                        let pos = i * bits as usize;
                        if bits == 32 {
                            packed[i] = v;
                            continue;
                        }
                        if bits == 0 {
                            continue;
                        }
                        packed[pos / 32] |= v << (pos % 32);
                        if pos % 32 + bits as usize > 32 {
                            packed[pos / 32 + 1] |= v >> (32 - pos % 32);
                        }
                    }
                    groups.extend(packed);
                }
                out.push(header);
                out.extend(groups);
            }
            out[last_header + 1] = (out.len() - last_header) as u32;
        }
        let rest = &input[full..];
        if !rest.is_empty() {
            last_header = out.len();
            out.extend([rest.len() as u32, 0]);
            let mut bytes = Vec::new();
            let mut prev = 0u32;
            for &v in rest {
                let mut d = v.wrapping_sub(prev);
                prev = v;
                while d >= 0x80 {
                    bytes.push((d as u8) | 0x80);
                    d >>= 7;
                }
                bytes.push(d as u8);
            }
            while bytes.len() % 4 != 0 {
                bytes.push(0x80);
            }
            for c in bytes.chunks(4) {
                out.push(u32::from_be_bytes([c[0], c[1], c[2], c[3]]));
            }
            out[last_header + 1] = (out.len() - last_header) as u32;
        }
        let last = (out.len() - last_header) as u32;
        out.push(last);
        out
    }

    #[test]
    fn round_trips_both_block_kinds() {
        let mut x = 0x9e3779b9u32;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let cases: Vec<Vec<u32>> = vec![
            vec![],
            vec![7],
            (0..127).collect(),
            (0..128).collect(),
            (0..1000).map(|i| i * 3).collect(),
            (0..700).map(|_| next()).collect(),
            (0..515).map(|i| 1_000_000 - i * 7).collect(),
            (0..256)
                .map(|i| if i % 2 == 0 { 5 } else { u32::MAX })
                .collect(),
        ];
        for case in cases {
            assert_eq!(
                uncompress_u32(&compress_u32(&case), usize::MAX).unwrap(),
                case
            );
        }
    }

    #[test]
    fn rejects_blocks_past_the_end() {
        let mut s = compress_u32(&(0..300).collect::<Vec<_>>());
        s[1] = 10_000;
        assert!(uncompress_u32(&s, usize::MAX).is_err());
        let mut s = compress_u32(&(0..50).collect::<Vec<_>>());
        s[0] = 1_000_000;
        assert!(uncompress_u32(&s, usize::MAX).is_err());
        assert!(uncompress_u64(&[1u64 << 32 | 5, 0], usize::MAX).is_err());
        let s = compress_u32(&(0..300).collect::<Vec<_>>());
        assert!(uncompress_u32(&s, 299).is_err());
    }
}
