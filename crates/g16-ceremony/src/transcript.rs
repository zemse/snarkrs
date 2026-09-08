//! The hash-and-RNG layer both ceremony phases share, and the one place the big-endian
//! point encoding lives.
//!
//! Every hash in the ceremony is **unkeyed BLAKE2b-512**: no key, no salt, no
//! personalisation, at all fourteen call sites. The RNG is **RFC 8439 ChaCha20** with a
//! zero nonce and a zero counter, keyed from a digest read **big-endian per 32-bit word**.
//! Miss any one of those and the output parses fine and verifies nowhere.
//!
//! Four things here are not what a fresh implementation would choose, and all four are
//! required:
//!
//! * **Points enter a hash uncompressed, non-Montgomery, big-endian, and G2 writes `c1`
//!   before `c0`.** `_LEMtoU` reverses each coordinate blob whole, and for G2 the blob is
//!   the entire 64-byte `c0 || c1` pair (`build_curve_jacobian_a0.js:1203-1229`, with
//!   `n8` being the `F2` size per `build_bn128.js:49`). That is the opposite of the
//!   on-disk order in [`crate::write::g2_lem`], in the same command, for the same point.
//! * **`Fr::fromRng` installs the sampled bits as Montgomery limbs.** It masks to 254
//!   bits, rejection-samples against the modulus (about 24% rejection per attempt, so
//!   multi-attempt draws are the common case and the word accounting must model them),
//!   and then writes the raw value into a buffer the wasm field reads as Montgomery. The
//!   mathematical element is therefore `v * R^-1 mod p`, which `f1field.js:304-315` makes
//!   explicit. Build these with `new_unchecked`, never `from`.
//! * **`nextU64` takes the high word first**, the opposite of `rand_core`'s. `nextBool`
//!   consumes a whole 32-bit word. So the RNG must be driven with `next_u32` only:
//!   calling `rand_chacha`'s own `next_u64` or `fill_bytes` silently reorders the stream.
//! * **`partialHash` is a serialised mid-stream BLAKE2b state**, 216 bytes of block
//!   buffer, chaining words and counters (`misc.js:89-127`), stored verbatim in every ptau
//!   contribution record. `blake2::Blake2b512` does not expose any of that, so
//!   [`Transcript::partial_hash`] cannot be built on top of the ordinary streaming API and
//!   the compression loop has to be driven directly. This is the one primitive where
//!   "just use the crate" does not work.
//!
//! `G::fromRng` is try-and-increment on a fresh random `x`, with a `greatest` bool drawn
//! after `x` and before the square test, so a failed attempt still consumes it. Fq's
//! `sqrt` normalises its sign and Fp2's does not, but the two conventions collapse to one
//! rule, proved in [`select_root`], so no Tonelli-Shanks port is needed. G2 then clears
//! the cofactor with a literal scalar multiplication, not a psi-based map.
//!
//! Every function here is pinned to a byte vector printed by snarkjs 0.7.6 itself; the
//! script that produced them is quoted at the top of `tests/transcript.rs`.

use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ff::{AdditiveGroup, BigInt};
use blake2::{Blake2b512, Digest as _};
use g16_field::{
    g1, g2, AffineRepr, BigInteger, Bn254, CurveGroup, Field, Fq, Fq2, Fr, G1Affine, G2Affine,
    G2Projective, Pairing, PrimeField, Zero,
};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::Sha256;

use crate::{CeremonyError, N8, SCG1, SCG2, SG1, SG2};

/// A 64-byte BLAKE2b-512 digest. Every challenge, response, transcript and `csHash` is
/// one of these.
pub type Digest = [u8; 64];

/// Bytes of the serialised BLAKE2b state snarkjs calls `partialHash`
/// (`powersoftau_utils.js:172`).
pub const PARTIAL_HASH_BYTES: usize = 216;

/// The phase-1 pubkey blob: six G1 points then three G2 points
/// (`powersoftau_utils.js:126-134`). The module doc-comment at `powersoftau_new.js:54-62`
/// lists a different order and invents a field that does not exist; `toPtauPubKeyRpr` is
/// the format.
pub const PTAU_PUBKEY_BYTES: usize = 6 * SG1 + 3 * SG2;

/// Personalisation byte for `getG2sp`: 0 for tau, 1 for alpha, 2 for beta
/// (`keypair.js:70-72`). Phase 2 has no equivalent; its `g2_sp` comes straight from the
/// transcript with no domain separation.
pub const PERSONALIZATION_TAU: u8 = 0;
pub const PERSONALIZATION_ALPHA: u8 = 1;
pub const PERSONALIZATION_BETA: u8 = 2;

/// BLAKE2b's block, and therefore the length of the buffer [`Transcript::partial_hash`]
/// dumps.
const BLOCK_BYTES: usize = 128;

/// SHA-512's IV, which BLAKE2b reuses unchanged (RFC 7693 §2.6).
const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// RFC 7693 §2.7. BLAKE2b runs twelve rounds, so rows 10 and 11 repeat rows 0 and 1.
#[rustfmt::skip]
const SIGMA: [[usize; 16]; 12] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],
    [ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8],
    [ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13],
    [ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9],
    [12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11],
    [13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10],
    [ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5],
    [10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0],
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
];

#[inline(always)]
fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// An unkeyed BLAKE2b-512 in progress.
///
/// `Clone` is not a convenience: `zkey verify` clones the accumulated hasher per
/// contribution to check one transcript without disturbing the chain
/// (`zkey_verify_frominit.js:51-59`).
///
/// The three fields are exactly what [`Transcript::partial_hash`] has to serialise, and
/// they are named after the noble-hashes fields `misc.js:111-127` reaches into: `h` is
/// `v0l..v7h`, `pos` is the fill of the block buffer, and `length` counts every byte
/// absorbed, buffered ones included.
#[derive(Clone)]
pub struct Transcript {
    h: [u64; 8],
    buffer: [u8; BLOCK_BYTES],
    pos: usize,
    length: u64,
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    pub fn new() -> Self {
        let mut h = IV;
        // Parameter block word 0: `0x01010000 ^ (keylen << 8) ^ outlen` with keylen 0 and
        // outlen 64 (RFC 7693 §2.5). Nothing else in the parameter block is ever set,
        // because no ceremony call site passes a key, salt or personalization.
        h[0] ^= 0x0101_0000 ^ 64;
        Self {
            h,
            buffer: [0u8; BLOCK_BYTES],
            pos: 0,
            length: 0,
        }
    }

    fn compress(&mut self, block: &[u8; BLOCK_BYTES], last: bool) {
        let mut m = [0u64; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..].copy_from_slice(&IV);
        // The counter is 128 bits; the high half stays zero because `length` is a u64 and
        // no ceremony file is 2^64 bytes.
        v[12] ^= self.length;
        if last {
            v[14] = !v[14];
        }
        for s in SIGMA.iter() {
            mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }

    /// Absorb bytes.
    ///
    /// The loop mirrors `blake2.js:125-163` rather than being written from the RFC, and
    /// the difference is visible in the file. BLAKE2b cannot compress a block until it
    /// knows another one follows, so a full buffer sits uncompressed until more input
    /// arrives; and when a whole aligned block can be taken straight from the caller's
    /// slice, noble compresses it in place and **leaves the buffer holding stale bytes
    /// from an earlier block**. Those stale bytes never reach a digest, because
    /// `digestInto` zeroes everything past `pos` first, but they are copied verbatim into
    /// [`Transcript::partial_hash`] and from there into the `.ptau` file, so the
    /// zero-copy branch is reproduced here and not simplified away.
    ///
    /// One caveat that only phase 1 can settle: noble takes that branch only when the
    /// input view is 4-byte aligned (`blake2.js:148`), and alignment is a property of the
    /// JavaScript `Uint8Array` a caller happened to pass. This models `byteOffset == 0`,
    /// which holds for every freshly allocated buffer snarkjs hashes. If a stored
    /// `partialHash` ever disagrees with ours in the region past `pos`, that is the
    /// reason, and only the stale tail can differ: the digest is unaffected.
    pub fn update(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        let mut pos = 0usize;
        while pos < len {
            if self.pos == BLOCK_BYTES {
                let block = self.buffer;
                self.compress(&block, false);
                self.pos = 0;
            }
            let take = core::cmp::min(BLOCK_BYTES - self.pos, len - pos);
            if take == BLOCK_BYTES && pos & 3 == 0 && pos + take < len {
                while pos + BLOCK_BYTES < len {
                    self.length += BLOCK_BYTES as u64;
                    let block: [u8; BLOCK_BYTES] =
                        bytes[pos..pos + BLOCK_BYTES].try_into().unwrap();
                    self.compress(&block, false);
                    pos += BLOCK_BYTES;
                }
                continue;
            }
            self.buffer[self.pos..self.pos + take].copy_from_slice(&bytes[pos..pos + take]);
            self.pos += take;
            self.length += take as u64;
            pos += take;
        }
    }

    /// `hashU32`, four **big-endian** bytes (`zkey_new.js:579-584`, `setUint32(0, n,
    /// false)`). Every integer that reaches a file is little-endian; this one does not
    /// reach a file.
    pub fn update_u32_be(&mut self, v: u32) {
        self.update(&v.to_be_bytes());
    }

    /// 64 bytes, [`g1_uncompressed`]. snarkjs' `hashG1` (`zkey_utils.js:546-550`).
    pub fn update_g1(&mut self, p: &G1Affine) {
        self.update(&g1_uncompressed(p));
    }

    /// 128 bytes, [`g2_uncompressed`], `c1` before `c0`. snarkjs' `hashG2`
    /// (`zkey_utils.js:552-556`).
    pub fn update_g2(&mut self, p: &G2Affine) {
        self.update(&g2_uncompressed(p));
    }

    pub fn finalize(mut self) -> Digest {
        // Padding: everything past `pos` is zeroed before the final compression, which is
        // what makes the stale bytes `update` preserves invisible to the digest.
        self.buffer[self.pos..].fill(0);
        let block = self.buffer;
        self.compress(&block, true);
        let mut out = [0u8; 64];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// The 216-byte resumable snapshot. Layout is 128 bytes of block buffer (stale bytes
    /// past `pos` included), then the eight chaining words as sixteen little-endian u32
    /// lo/hi pairs, then `length - pos`, a zero, `pos`, and eight bytes of padding
    /// (`misc.js:111-127`).
    ///
    /// The lo/hi pairs are just each chaining word little-endian, so this writes u64s.
    ///
    /// snarkjs truncates `length - pos` to 32 bits, so a response hash that has absorbed
    /// more than 4 GiB (roughly power 26 and up) resumes wrong. That is their bug, not
    /// ours; match it if we ever have to interoperate at that size, and say so first.
    pub fn partial_hash(&self) -> [u8; PARTIAL_HASH_BYTES] {
        let mut out = [0u8; PARTIAL_HASH_BYTES];
        out[..BLOCK_BYTES].copy_from_slice(&self.buffer);
        for (i, word) in self.h.iter().enumerate() {
            let at = BLOCK_BYTES + i * 8;
            out[at..at + 8].copy_from_slice(&word.to_le_bytes());
        }
        let compressed = self.length - self.pos as u64;
        out[192..196].copy_from_slice(&(compressed as u32).to_le_bytes());
        out[200..204].copy_from_slice(&(self.pos as u32).to_le_bytes());
        out
    }

    /// Resume from a stored snapshot.
    ///
    /// `fromPartialHash` reads the byte count as `w16 + w17*2^32` and the buffer fill as
    /// `w18 + w19*2^32`, then sets `length = count + pos` (`misc.js:100-106`). Words 17
    /// and 19 are never written, so both high halves are always zero; they are read here
    /// anyway, because a file written by something other than snarkjs may set them.
    pub fn from_partial_hash(blob: &[u8; PARTIAL_HASH_BYTES]) -> Result<Self, CeremonyError> {
        let mut h = [0u64; 8];
        for (i, word) in h.iter_mut().enumerate() {
            let at = BLOCK_BYTES + i * 8;
            *word = u64::from_le_bytes(blob[at..at + 8].try_into().unwrap());
        }
        let compressed = u64::from_le_bytes(blob[192..200].try_into().unwrap());
        let pos = u64::from_le_bytes(blob[200..208].try_into().unwrap());
        if pos > BLOCK_BYTES as u64 {
            return Err(CeremonyError::malformed(
                7,
                format!("partial hash claims {pos} buffered bytes, block is {BLOCK_BYTES}"),
            ));
        }
        let length = compressed.checked_add(pos).ok_or_else(|| {
            CeremonyError::malformed(7, "partial hash byte count overflows a u64")
        })?;
        Ok(Self {
            h,
            buffer: blob[..BLOCK_BYTES].try_into().unwrap(),
            pos: pos as usize,
            length,
        })
    }
}

/// One-shot BLAKE2b-512. `blake2b512(b"")` is the "Blank Contribution Hash" the first
/// challenge starts from (`powersoftau_utils.js:322`).
///
/// This is the `blake2` crate rather than [`Transcript`], on purpose: the two are held
/// against each other by a test, so the hand-driven compression loop that
/// [`Transcript::partial_hash`] forces on us has an independent implementation to fail
/// against rather than only a set of pinned vectors.
pub fn blake2b512(bytes: &[u8]) -> Digest {
    let mut h = Blake2b512::new();
    h.update(bytes);
    h.finalize().into()
}

/// ffjavascript's `ChaCha`, which is RFC 8439 ChaCha20 with a zero nonce and words 12 to
/// 15 acting as one 128-bit little-endian block counter (`chacha.js:41-95`).
///
/// Only [`CeremonyRng::next_u32`] reads the stream directly. The other two are built on
/// top of it and do not match `rand_core`'s versions.
///
/// `rand_chacha` serves unmodified. `chacha.js` installs the seed as eight state words
/// where RFC 8439 installs the key as eight little-endian words, so the seed bytes are
/// swapped per word on the way in ([`CeremonyRng::from_seed_words`]); after that the
/// keystreams are identical, and ffjavascript's four-word counter agrees with
/// `rand_chacha`'s 64-bit counter plus 64-bit stream id until block 2^64, which is
/// 2^70 bytes away. The zero-seed vector in `tests/transcript.rs` is what settles the
/// word order, and it reads past the first block so a future change to `BlockRng`'s
/// batching cannot slip through.
pub struct CeremonyRng {
    inner: ChaCha20Rng,
}

impl CeremonyRng {
    /// Seed from eight words that have already been read big-endian out of a digest.
    pub fn from_seed_words(words: [u32; 8]) -> Self {
        let mut key = [0u8; 32];
        for (i, w) in words.iter().enumerate() {
            key[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Self {
            inner: ChaCha20Rng::from_seed(key),
        }
    }

    /// Seed from the **first 32 bytes** of a digest, read big-endian per word. Bytes 32
    /// onwards of a 64-byte digest are discarded, in every caller
    /// (`misc.js:194-196`, `keypair.js:29`).
    pub fn from_digest(digest: &[u8]) -> Self {
        let mut words = [0u32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            *w = u32::from_be_bytes(digest[i * 4..i * 4 + 4].try_into().unwrap());
        }
        Self::from_seed_words(words)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// `nextU64`: two words, **high half first** (`chacha.js:68-70`).
    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// `nextBool`: the low bit of a whole consumed word (`chacha.js:72-74`).
    pub fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 1
    }
}

/// `getRandomRng` (`misc.js:182-199`): BLAKE2b-512 over 64 OS-random bytes **then** the
/// UTF-8 entropy string, with no NUL and no length prefix, seeded from the first 32 bytes
/// of the digest.
///
/// Deliberately not reproducible. Both `contribute` commands are non-deterministic by
/// design, and a Rust port that made them deterministic would be a weaker ceremony, not a
/// more testable one. [`rng_from_entropy_with`] is the test hook.
///
/// The 64 bytes come from `/dev/urandom` rather than a crate: `rand_core::OsRng` needs
/// its `getrandom` feature, which nothing in this crate's dependency graph turns on and
/// which we would then be relying on a sibling crate to keep turning on. Panicking is the
/// right failure here, because a ceremony that silently proceeds on degraded entropy is
/// worse than one that stops.
pub fn rng_from_entropy(entropy: &str) -> CeremonyRng {
    use std::io::Read;
    let mut os_bytes = [0u8; 64];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut os_bytes))
        .expect("/dev/urandom: no operating system entropy for the contribution");
    rng_from_entropy_with(&os_bytes, entropy)
}

/// [`rng_from_entropy`] with the 64 OS-random bytes injected, so a run can be compared
/// against snarkjs byte for byte. Test-only in spirit; the CLI never calls it.
pub fn rng_from_entropy_with(os_bytes: &[u8; 64], entropy: &str) -> CeremonyRng {
    let mut h = Transcript::new();
    h.update(os_bytes);
    h.update(entropy.as_bytes());
    CeremonyRng::from_digest(&h.finalize())
}

/// `rngFromBeaconParams` (`misc.js:201-228`): SHA-256 iterated exactly
/// `2^num_iterations_exp` times, the first over `beacon_hash` itself and every later one
/// over the previous 32-byte digest, then eight big-endian words of that digest.
///
/// This is the deterministic half of the ceremony and where byte-comparison against
/// snarkjs belongs. Callers bound `num_iterations_exp` to `10..=63`.
pub fn rng_from_beacon_params(beacon_hash: &[u8], num_iterations_exp: u8) -> CeremonyRng {
    // snarkjs splits the count into an inner and an outer loop only to keep each bound
    // inside a 32-bit int (`misc.js:205-211`); the product is 2^num_iterations_exp.
    let iterations = 1u128
        .checked_shl(num_iterations_exp.into())
        .expect("numIterationsExp is bounded to 10..=63 by parse_beacon_args");
    let mut cur: Vec<u8> = beacon_hash.to_vec();
    for _ in 0..iterations {
        cur = Sha256::digest(&cur).to_vec();
    }
    CeremonyRng::from_digest(&cur)
}

/// Clear every bit at or above `bits`, snarkjs' `Scalar.band(v, this.mask)` with
/// `mask = 2^bitLength(p) - 1` (`wasm_field1.js:21`). Both BN254 primes are 254 bits, so
/// this drops bits 254 and 255 of the sampled 256-bit value, both of which live in the
/// top limb.
fn mask_bits(limbs: &mut [u64; 4], bits: u32) {
    for (i, limb) in limbs.iter_mut().enumerate() {
        let low = i as u32 * 64;
        if low >= bits {
            *limb = 0;
        } else if low + 64 > bits {
            *limb &= (1u64 << (bits - low)) - 1;
        }
    }
}

/// The shared body of `Fr.fromRng` and `Fq.fromRng` (`wasm_field1.js:195-207`): four
/// `nextU64`s into little-endian limbs, mask, and redraw the whole thing while the result
/// is at or above the modulus.
///
/// The returned limbs are the value snarkjs writes into the buffer, which the wasm field
/// then reads as Montgomery limbs. Callers must install them with `new_unchecked`.
fn sample_limbs(rng: &mut CeremonyRng, bits: u32, modulus: BigInt<4>) -> BigInt<4> {
    loop {
        let mut limbs = [0u64; 4];
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        mask_bits(&mut limbs, bits);
        let v = BigInt::new(limbs);
        if v < modulus {
            return v;
        }
    }
}

/// `Fr.fromRng` (`wasm_field1.js:195-207`). Eight `next_u32` per attempt, assembled as
/// four little-endian u64 limbs each of which takes its **high** word first, masked to 254
/// bits, redrawn while at or above the modulus, and installed as Montgomery limbs.
pub fn fr_from_rng(rng: &mut CeremonyRng) -> Fr {
    Fr::new_unchecked(sample_limbs(rng, Fr::MODULUS_BIT_SIZE, Fr::MODULUS))
}

/// [`fr_from_rng`] over the base field. Same 254-bit mask, since BN254's two primes have
/// the same bit length.
pub fn fq_from_rng(rng: &mut CeremonyRng) -> Fq {
    Fq::new_unchecked(sample_limbs(rng, Fq::MODULUS_BIT_SIZE, Fq::MODULUS))
}

/// `c0` then `c1`, each an independent rejection loop (`wasm_field2.js:149-156`). The two
/// locals there are named `c1` and `c2`, but they are written at offsets 0 and `n8`, which
/// are `c0` and `c1`.
pub fn fq2_from_rng(rng: &mut CeremonyRng) -> Fq2 {
    let c0 = fq_from_rng(rng);
    let c1 = fq_from_rng(rng);
    Fq2::new(c0, c1)
}

/// `f1m_isNegative` (`build_f1m.js:118-131`): the **canonical** value is at or above
/// `(q-1)/2 + 1`. `f1m_sign` returns -1 on exactly this condition and 0 for zero, so the
/// compressed-point sign bit and this predicate agree everywhere.
fn fq_is_negative(x: &Fq) -> bool {
    x.into_bigint() > Fq::MODULUS_MINUS_ONE_DIV_TWO
}

/// `f2m_isNegative` (`build_f2m.js:134-150`): `c1`'s sign, falling back to `c0`'s when
/// `c1` is zero.
fn fq2_is_negative(x: &Fq2) -> bool {
    if x.c1.is_zero() {
        fq_is_negative(&x.c0)
    } else {
        fq_is_negative(&x.c1)
    }
}

/// Pick the square root `wasm_curve.js:320-323` would have picked, given either root.
///
/// snarkjs computes `s = isNegative(sqrt(x3b))` and negates when `greatest ^ s`. That
/// looks like it depends on which root the sqrt algorithm returns, and it is why the spec
/// warns that Fq's Tonelli-Shanks normalises its sign (`build_f1m.js:873-877`) while
/// Fp2's Algorithm 9adj does not. It does not actually depend on it. For a nonzero root
/// `r`, exactly one of `r` and `-r` is negative, so replacing `r` by `-r` flips both `s`
/// and the negation decision and lands on the same point; and `r == 0` is fixed by
/// negation anyway. The rule that survives is the one below: the chosen root's sign
/// **is** `greatest`. So arkworks' sqrt can be used directly and no root-selection
/// convention has to be ported.
fn select_root<F: Field + Copy>(root: F, greatest: bool, is_negative: fn(&F) -> bool) -> F {
    if is_negative(&root) == greatest {
        root
    } else {
        -root
    }
}

/// `G1.fromRng` (`wasm_curve.js:309-334`): draw `x`, draw the `greatest` bool, retry the
/// whole attempt when `x^3 + b` is not square. G1 has no cofactor on BN254
/// (`bn128.js:38-44` never sets `cofactorG1`), so the point is returned as it is.
pub fn g1_from_rng(rng: &mut CeremonyRng) -> G1Affine {
    loop {
        let x = fq_from_rng(rng);
        // Drawn after x and before the square test, so a failed attempt still burns it.
        let greatest = rng.next_bool();
        let x3b = x * x * x + g1::Config::COEFF_B;
        if let Some(root) = x3b.sqrt() {
            let y = select_root(root, greatest, fq_is_negative);
            return G1Affine::new_unchecked(x, y);
        }
    }
}

/// `G2.fromRng`, then a literal scalar multiplication by BN254's G2 cofactor
/// `0x30644e72e131a029b85045b68181585e06ceecda572a2489345f2299c0f9fa8d`
/// (`bn128.js:43`). Not a psi-based cofactor clearing: prove the two agree before
/// substituting one.
pub fn g2_from_rng(rng: &mut CeremonyRng) -> G2Affine {
    loop {
        let x = fq2_from_rng(rng);
        let greatest = rng.next_bool();
        let x3b = x * x * x + g2::Config::COEFF_B;
        if let Some(root) = x3b.sqrt() {
            let y = select_root(root, greatest, fq2_is_negative);
            return mul_by_g2_cofactor(&G2Affine::new_unchecked(x, y)).into_affine();
        }
    }
}

/// BN254's G2 cofactor, `Scalar.toLEBuff(cofactorG2)` (`wasm_curve.js:24-26`), as u64
/// limbs.
const G2_COFACTOR: [u64; 4] = [
    0x345f_2299_c0f9_fa8d,
    0x06ce_ecda_572a_2489,
    0xb850_45b6_8181_585e,
    0x3064_4e72_e131_a029,
];

/// Square-and-multiply by [`G2_COFACTOR`], written out rather than handed to
/// `mul_bigint`.
///
/// The input is a point on `E'(Fq2)` that is **not** in the r-torsion subgroup yet, which
/// is the whole reason the multiplication is happening. arkworks' `mul_bigint` dispatches
/// through `SWCurveConfig::mul_projective`, and ark-bn254 overrides that with a GLV
/// decomposition for G1 (`curves/g1.rs:43-49`). G2 does not override it today, so this
/// would be correct by accident; a GLV endomorphism is only valid inside the subgroup, so
/// the day it does override it, `mul_bigint` here would start returning a different point.
fn mul_by_g2_cofactor(p: &G2Affine) -> G2Projective {
    let mut acc = G2Projective::zero();
    let mut started = false;
    for limb in G2_COFACTOR.iter().rev() {
        for bit in (0..64).rev() {
            if started {
                acc.double_in_place();
            }
            if (limb >> bit) & 1 == 1 {
                if started {
                    acc += *p;
                } else {
                    acc = (*p).into();
                    started = true;
                }
            }
        }
    }
    acc
}

/// `hashToG2` (`keypair.js:24-36`): seed a ChaCha from the first 32 bytes of the digest,
/// big-endian per word, and take one `G2.fromRng`. No SWU, no domain separation string.
pub fn hash_to_g2(digest: &Digest) -> G2Affine {
    let mut rng = CeremonyRng::from_digest(digest);
    g2_from_rng(&mut rng)
}

/// `getG2sp` (`keypair.js:38-51`), the phase-1 wrapper: BLAKE2b-512 over the 1-byte
/// personalisation, the 64-byte previous challenge, and the two G1 points uncompressed,
/// 193 bytes in total, then [`hash_to_g2`].
pub fn get_g2_sp(
    personalization: u8,
    challenge: &Digest,
    g1_s: &G1Affine,
    g1_sx: &G1Affine,
) -> G2Affine {
    let mut h = Transcript::new();
    h.update(&[personalization]);
    h.update(challenge);
    h.update_g1(g1_s);
    h.update_g1(g1_sx);
    hash_to_g2(&h.finalize())
}

/// The public half of one phase-1 key. `g2_sp` is **not** stored in the file; it is
/// recomputed from the previous challenge, which is what ties a contribution to its
/// predecessor.
#[derive(Clone, Copy, Debug)]
pub struct PtauPubKey {
    pub g1_s: G1Affine,
    pub g1_sx: G1Affine,
    pub g2_sp: G2Affine,
    pub g2_spx: G2Affine,
}

/// The three public keys of one phase-1 contribution, in the order they are drawn and
/// stored.
#[derive(Clone, Copy, Debug)]
pub struct PtauPubKeys {
    pub tau: PtauPubKey,
    pub alpha: PtauPubKey,
    pub beta: PtauPubKey,
}

/// One phase-1 key, private half included. Only a key we just generated has one.
#[derive(Clone, Copy, Debug)]
pub struct PtauKeyPair {
    pub prv_key: Fr,
    pub pubkey: PtauPubKey,
}

/// A whole phase-1 contribution key.
#[derive(Clone, Copy, Debug)]
pub struct PtauKey {
    pub tau: PtauKeyPair,
    pub alpha: PtauKeyPair,
    pub beta: PtauKeyPair,
}

impl PtauKey {
    pub fn pubkeys(&self) -> PtauPubKeys {
        PtauPubKeys {
            tau: self.tau.pubkey,
            alpha: self.alpha.pubkey,
            beta: self.beta.pubkey,
        }
    }
}

/// `calculatePubKey` (`keypair.js:53-59`). Only `g1_s` comes from the RNG; the other
/// three points are derived.
fn calculate_pubkey(
    prv_key: &Fr,
    personalization: u8,
    challenge: &Digest,
    rng: &mut CeremonyRng,
) -> PtauKeyPair {
    let g1_s = g1_from_rng(rng);
    // `timesFr` de-Montgomeries the scalar before multiplying (`build_bn128.js:54-72`),
    // so the multiplier is the element's mathematical value, which is what arkworks'
    // `Mul<Fr>` uses too.
    let g1_sx = (g1_s * prv_key).into_affine();
    let g2_sp = get_g2_sp(personalization, challenge, &g1_s, &g1_sx);
    let g2_spx = (g2_sp * prv_key).into_affine();
    PtauKeyPair {
        prv_key: *prv_key,
        pubkey: PtauPubKey {
            g1_s,
            g1_sx,
            g2_sp,
            g2_spx,
        },
    }
}

/// `createPTauKey` (`keypair.js:61-74`). The RNG is consumed as `Fr, Fr, Fr, G1, G1, G1`:
/// the three private scalars up front in tau/alpha/beta order, then one `g1_s` per key in
/// the same order. Nothing else touches it, and the order is load-bearing for a beacon.
pub fn create_ptau_key(rng: &mut CeremonyRng, challenge: &Digest) -> PtauKey {
    let tau_prv = fr_from_rng(rng);
    let alpha_prv = fr_from_rng(rng);
    let beta_prv = fr_from_rng(rng);
    PtauKey {
        tau: calculate_pubkey(&tau_prv, PERSONALIZATION_TAU, challenge, rng),
        alpha: calculate_pubkey(&alpha_prv, PERSONALIZATION_ALPHA, challenge, rng),
        beta: calculate_pubkey(&beta_prv, PERSONALIZATION_BETA, challenge, rng),
    }
}

/// The 768-byte pubkey blob. `montgomery` picks the encoding, and both are used on the
/// same nine points in the same command: `true` is the LEM form stored in section 7
/// (`powersoftau_utils.js:253`), `false` is the uncompressed form that is hashed
/// (`:176-180`). Getting it backwards yields a wrong response hash and nothing else.
pub fn write_ptau_pubkey(keys: &PtauPubKeys, montgomery: bool) -> [u8; PTAU_PUBKEY_BYTES] {
    let mut out = [0u8; PTAU_PUBKEY_BYTES];
    let mut at = 0usize;
    for p in [
        &keys.tau.g1_s,
        &keys.tau.g1_sx,
        &keys.alpha.g1_s,
        &keys.alpha.g1_sx,
        &keys.beta.g1_s,
        &keys.beta.g1_sx,
    ] {
        let bytes = if montgomery {
            crate::write::g1_lem(p)
        } else {
            g1_uncompressed(p)
        };
        out[at..at + SG1].copy_from_slice(&bytes);
        at += SG1;
    }
    for p in [&keys.tau.g2_spx, &keys.alpha.g2_spx, &keys.beta.g2_spx] {
        let bytes = if montgomery {
            crate::write::g2_lem(p)
        } else {
            g2_uncompressed(p)
        };
        out[at..at + SG2].copy_from_slice(&bytes);
        at += SG2;
    }
    out
}

/// Read the nine stored points and recompute the three `g2_sp` from `challenge`.
///
/// Always the Montgomery form: the only reader in snarkjs' file path is
/// `readPtauPubKey(fd, curve, true)` (`powersoftau_utils.js:172`). The uncompressed form
/// is written to be hashed and is never read back.
pub fn read_ptau_pubkey(bytes: &[u8], challenge: &Digest) -> Result<PtauPubKeys, CeremonyError> {
    if bytes.len() < PTAU_PUBKEY_BYTES {
        return Err(CeremonyError::malformed(
            7,
            format!(
                "pubkey needs {PTAU_PUBKEY_BYTES} bytes, {} present",
                bytes.len()
            ),
        ));
    }
    let g1 = |i: usize| g16_zkey::binfile::g1(&bytes[i * SG1..(i + 1) * SG1]);
    let g2 = |i: usize| {
        let at = 6 * SG1 + i * SG2;
        g16_zkey::binfile::g2(&bytes[at..at + SG2])
    };
    let key = |g1_s: G1Affine, g1_sx: G1Affine, g2_spx: G2Affine, personalization: u8| PtauPubKey {
        g1_s,
        g1_sx,
        g2_sp: get_g2_sp(personalization, challenge, &g1_s, &g1_sx),
        g2_spx,
    };
    Ok(PtauPubKeys {
        tau: key(g1(0), g1(1), g2(0), PERSONALIZATION_TAU),
        alpha: key(g1(2), g1(3), g2(1), PERSONALIZATION_ALPHA),
        beta: key(g1(4), g1(5), g2(2), PERSONALIZATION_BETA),
    })
}

/// The phase-2 delta key (`zkey_contribute.js:54-61`). Unlike phase 1 there is no
/// personalisation byte: `g2_sp` is `hashToG2(transcript)` directly.
#[derive(Clone, Copy, Debug)]
pub struct DeltaKey {
    pub prv_key: Fr,
    pub g1_s: G1Affine,
    pub g1_sx: G1Affine,
    pub g2_sp: G2Affine,
    pub g2_spx: G2Affine,
}

/// Draw a delta key and close the transcript that names it.
///
/// `prior` must already have absorbed `csHash` and every earlier contribution's pubkey.
/// This function draws `prv_key` and `g1_s` from `rng`, feeds `g1_s` and `g1_sx` into the
/// transcript, digests it, and derives `g2_sp` from that digest. The returned [`Digest`]
/// is the value stored in the contribution record, and `zkey_verify_frominit.js:46-63`
/// reconstructs it exactly this way, so any divergence is caught by snarkjs' own verifier.
///
/// `keypair.js:76-83` has a `createDeltaKey` that looks like this and is dead code:
/// nothing calls it, and it uses `timesScalar` where both real call sites use `timesFr`.
/// This follows `zkey_contribute.js`, not that function.
pub fn create_delta_key(rng: &mut CeremonyRng, prior: Transcript) -> (DeltaKey, Digest) {
    let prv_key = fr_from_rng(rng);
    let g1_s = g1_from_rng(rng);
    let g1_sx = (g1_s * prv_key).into_affine();
    let mut hasher = prior;
    hasher.update_g1(&g1_s);
    hasher.update_g1(&g1_sx);
    let transcript = hasher.finalize();
    let g2_sp = hash_to_g2(&transcript);
    let g2_spx = (g2_sp * prv_key).into_affine();
    (
        DeltaKey {
            prv_key,
            g1_s,
            g1_sx,
            g2_sp,
            g2_spx,
        },
        transcript,
    )
}

/// A base-field coordinate as it enters a hash: the canonical value, big-endian.
fn fq_be(x: &Fq) -> [u8; N8] {
    let mut out = [0u8; N8];
    out.copy_from_slice(&x.into_bigint().to_bytes_be());
    out
}

/// The inverse of [`fq_be`]. Rejects a value at or above the modulus rather than reducing
/// it, matching [`g16_zkey::binfile::fr_normal`]: snarkjs' `_UtoLEM` would happily
/// Montgomery-convert such a blob into a point that then fails every ratio check, and a
/// named error beats that.
fn fq_from_be(b: &[u8], what: &str) -> Result<Fq, CeremonyError> {
    let mut le = [0u8; N8];
    le.copy_from_slice(b);
    le.reverse();
    Fq::from_bigint(g16_zkey::binfile::bigint(&le)).ok_or_else(|| {
        CeremonyError::malformed(0, format!("{what} is not below the base field modulus"))
    })
}

/// Uncompressed G1: `x` then `y`, each 32 bytes, non-Montgomery **big-endian**. Infinity
/// is all zero bytes, and it stays all zero bytes: `_UtoLEM` detects infinity from bit
/// `0x40` instead, so the round trip is asymmetric. Do not "fix" that.
pub fn g1_uncompressed(p: &G1Affine) -> [u8; SG1] {
    let mut out = [0u8; SG1];
    if p.is_zero() {
        return out;
    }
    out[..N8].copy_from_slice(&fq_be(&p.x));
    out[N8..].copy_from_slice(&fq_be(&p.y));
    out
}

/// Uncompressed G2: `x.c1, x.c0, y.c1, y.c0`, each 32 bytes big-endian. The swap is what
/// the byte reversal over the whole 64-byte `F2` blob produces, and it is the same
/// ordering Ethereum's pairing precompiles use.
pub fn g2_uncompressed(p: &G2Affine) -> [u8; SG2] {
    let mut out = [0u8; SG2];
    if p.is_zero() {
        return out;
    }
    out[..N8].copy_from_slice(&fq_be(&p.x.c1));
    out[N8..2 * N8].copy_from_slice(&fq_be(&p.x.c0));
    out[2 * N8..3 * N8].copy_from_slice(&fq_be(&p.y.c1));
    out[3 * N8..].copy_from_slice(&fq_be(&p.y.c0));
    out
}

/// The `0x40` infinity flag `_UtoLEM` and `_CtoLEM` test on byte 0
/// (`build_curve_jacobian_a0.js:1245`, `:1273`).
const FLAG_INFINITY: u8 = 0x40;
/// The `0x80` sign flag `_LEMtoC` sets when `sign(y) == -1` (`:1187-1199`).
const FLAG_NEGATIVE: u8 = 0x80;

pub fn g1_from_uncompressed(bytes: &[u8]) -> Result<G1Affine, CeremonyError> {
    if bytes.len() < SG1 {
        return Err(CeremonyError::malformed(
            0,
            format!("uncompressed G1 needs {SG1} bytes, {} present", bytes.len()),
        ));
    }
    if bytes[0] & FLAG_INFINITY != 0 {
        return Ok(G1Affine::identity());
    }
    let x = fq_from_be(&bytes[..N8], "uncompressed G1 x")?;
    let y = fq_from_be(&bytes[N8..SG1], "uncompressed G1 y")?;
    if x.is_zero() && y.is_zero() {
        // What `_LEMtoU` actually emits for infinity. It decodes through the ordinary
        // path to the affine pair (0, 0), which is the same thing one layer down, so the
        // asymmetric flag never bites; keep the mapping explicit anyway, as
        // `g16_zkey::binfile::g1` does.
        return Ok(G1Affine::identity());
    }
    Ok(G1Affine::new_unchecked(x, y))
}

pub fn g2_from_uncompressed(bytes: &[u8]) -> Result<G2Affine, CeremonyError> {
    if bytes.len() < SG2 {
        return Err(CeremonyError::malformed(
            0,
            format!("uncompressed G2 needs {SG2} bytes, {} present", bytes.len()),
        ));
    }
    if bytes[0] & FLAG_INFINITY != 0 {
        return Ok(G2Affine::identity());
    }
    let x = Fq2::new(
        fq_from_be(&bytes[N8..2 * N8], "uncompressed G2 x.c0")?,
        fq_from_be(&bytes[..N8], "uncompressed G2 x.c1")?,
    );
    let y = Fq2::new(
        fq_from_be(&bytes[3 * N8..], "uncompressed G2 y.c0")?,
        fq_from_be(&bytes[2 * N8..3 * N8], "uncompressed G2 y.c1")?,
    );
    if x.is_zero() && y.is_zero() {
        return Ok(G2Affine::identity());
    }
    Ok(G2Affine::new_unchecked(x, y))
}

/// Compressed G1, 32 bytes: big-endian normal-form `x`, with `0x80` OR'd into byte 0 when
/// `sign(y) == -1`, and byte 0 = `0x40` with the rest zero for infinity
/// (`build_curve_jacobian_a0.js:1164-1201`). `f1m_sign` is 0 for zero, -1 when the
/// canonical value is at or above `(q-1)/2 + 1`, else 1.
///
/// Used only by phase 1's response hash and the response file. Nothing in a `.ptau` body
/// is compressed.
pub fn g1_compressed(p: &G1Affine) -> [u8; SCG1] {
    let mut out = [0u8; SCG1];
    if p.is_zero() {
        out[0] = FLAG_INFINITY;
        return out;
    }
    out.copy_from_slice(&fq_be(&p.x));
    if fq_is_negative(&p.y) {
        out[0] |= FLAG_NEGATIVE;
    }
    out
}

/// Compressed G2, 64 bytes. `f2m_sign` is `sign(c1)`, falling back to `sign(c0)` when
/// `c1` is zero (`build_f2m.js:412-429`). `x` is the whole reversed `F2` blob, so `c1`
/// comes first here as well.
pub fn g2_compressed(p: &G2Affine) -> [u8; SCG2] {
    let mut out = [0u8; SCG2];
    if p.is_zero() {
        out[0] = FLAG_INFINITY;
        return out;
    }
    out[..N8].copy_from_slice(&fq_be(&p.x.c1));
    out[N8..].copy_from_slice(&fq_be(&p.x.c0));
    if fq2_is_negative(&p.y) {
        out[0] |= FLAG_NEGATIVE;
    }
    out
}

/// `_CtoLEM` (`build_curve_jacobian_a0.js:1257-1320`). Its four-way branch on
/// `sign(sqrt)` and `greatest` reduces to the same rule as [`select_root`]: the recovered
/// `y` is the root whose sign is `greatest`.
pub fn g1_from_compressed(bytes: &[u8]) -> Result<G1Affine, CeremonyError> {
    if bytes.len() < SCG1 {
        return Err(CeremonyError::malformed(
            0,
            format!("compressed G1 needs {SCG1} bytes, {} present", bytes.len()),
        ));
    }
    if bytes[0] & FLAG_INFINITY != 0 {
        return Ok(G1Affine::identity());
    }
    let greatest = bytes[0] & FLAG_NEGATIVE != 0;
    let mut x_be = [0u8; N8];
    x_be.copy_from_slice(&bytes[..N8]);
    x_be[0] &= 0x3f;
    let x = fq_from_be(&x_be, "compressed G1 x")?;
    let root = (x * x * x + g1::Config::COEFF_B)
        .sqrt()
        .ok_or_else(|| CeremonyError::malformed(0, "compressed G1 x is not on the curve"))?;
    Ok(G1Affine::new_unchecked(
        x,
        select_root(root, greatest, fq_is_negative),
    ))
}

pub fn g2_from_compressed(bytes: &[u8]) -> Result<G2Affine, CeremonyError> {
    if bytes.len() < SCG2 {
        return Err(CeremonyError::malformed(
            0,
            format!("compressed G2 needs {SCG2} bytes, {} present", bytes.len()),
        ));
    }
    if bytes[0] & FLAG_INFINITY != 0 {
        return Ok(G2Affine::identity());
    }
    let greatest = bytes[0] & FLAG_NEGATIVE != 0;
    let mut c1_be = [0u8; N8];
    c1_be.copy_from_slice(&bytes[..N8]);
    c1_be[0] &= 0x3f;
    let x = Fq2::new(
        fq_from_be(&bytes[N8..], "compressed G2 x.c0")?,
        fq_from_be(&c1_be, "compressed G2 x.c1")?,
    );
    let root = (x * x * x + g2::Config::COEFF_B)
        .sqrt()
        .ok_or_else(|| CeremonyError::malformed(0, "compressed G2 x is not on the curve"))?;
    Ok(G2Affine::new_unchecked(
        x,
        select_root(root, greatest, fq2_is_negative),
    ))
}

/// `sameRatio` (`misc.js:129-137`): reject any zero input, then check
/// `e(a1, b2) * e(-b1, a2) == 1`. Every phase-1 and phase-2 verification step is one of
/// these, and the zero rejection is not decoration: without it the identity passes every
/// ratio.
pub fn same_ratio(a1: &G1Affine, b1: &G1Affine, a2: &G2Affine, b2: &G2Affine) -> bool {
    if a1.is_zero() || b1.is_zero() || a2.is_zero() || b2.is_zero() {
        return false;
    }
    // `PairingOutput` is written additively, so the target-group identity is its zero.
    Bn254::multi_pairing([*a1, -*b1], [*b2, *a2]).is_zero()
}
