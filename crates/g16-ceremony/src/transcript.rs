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
//! `sqrt` normalises its sign and Fp2's does not, which is why `greatest` alone picks the
//! root on G1 while both `greatest` and the root's own sign matter on G2. G2 then clears
//! the cofactor with a literal scalar multiplication, not a psi-based map.
//!
//! Test vectors for every function here are in the transcript spec; write the zero-seed
//! ChaCha assertion first, because it is what settles `rand_chacha`'s word order.

use g16_field::{Fq, Fq2, Fr, G1Affine, G2Affine};

use crate::{CeremonyError, SCG1, SCG2, SG1, SG2};

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

/// An unkeyed BLAKE2b-512 in progress.
///
/// `Clone` is not a convenience: `zkey verify` clones the accumulated hasher per
/// contribution to check one transcript without disturbing the chain
/// (`zkey_verify_frominit.js:51-59`).
#[derive(Clone)]
pub struct Transcript {
    /// Scaffold placeholder. The real state is the eight chaining words, the 128-byte
    /// block buffer and the byte counter, because `partial_hash` has to serialise all
    /// three.
    #[allow(dead_code)]
    state: (),
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    pub fn new() -> Self {
        todo!("Transcript::new")
    }

    pub fn update(&mut self, bytes: &[u8]) {
        let _ = bytes;
        todo!("Transcript::update")
    }

    /// `hashU32`, four **big-endian** bytes (`zkey_new.js:579-584`, `setUint32(0, n,
    /// false)`). Every integer that reaches a file is little-endian; this one does not
    /// reach a file.
    pub fn update_u32_be(&mut self, v: u32) {
        let _ = v;
        todo!("Transcript::update_u32_be")
    }

    /// 64 bytes, [`g1_uncompressed`].
    pub fn update_g1(&mut self, p: &G1Affine) {
        let _ = p;
        todo!("Transcript::update_g1")
    }

    /// 128 bytes, [`g2_uncompressed`], `c1` before `c0`.
    pub fn update_g2(&mut self, p: &G2Affine) {
        let _ = p;
        todo!("Transcript::update_g2")
    }

    pub fn finalize(self) -> Digest {
        todo!("Transcript::finalize")
    }

    /// The 216-byte resumable snapshot. Layout is 128 bytes of block buffer (stale bytes
    /// past `pos` included), then the eight chaining words as sixteen little-endian u32
    /// lo/hi pairs, then `length - pos`, a zero, `pos`, and eight bytes of padding
    /// (`misc.js:111-127`).
    ///
    /// snarkjs truncates `length - pos` to 32 bits, so a response hash that has absorbed
    /// more than 4 GiB (roughly power 26 and up) resumes wrong. That is their bug, not
    /// ours; match it if we ever have to interoperate at that size, and say so first.
    pub fn partial_hash(&self) -> [u8; PARTIAL_HASH_BYTES] {
        todo!("Transcript::partial_hash")
    }

    /// Resume from a stored snapshot.
    pub fn from_partial_hash(blob: &[u8; PARTIAL_HASH_BYTES]) -> Result<Self, CeremonyError> {
        let _ = blob;
        todo!("Transcript::from_partial_hash")
    }
}

/// One-shot BLAKE2b-512. `blake2b512(b"")` is the "Blank Contribution Hash" the first
/// challenge starts from (`powersoftau_utils.js:322`).
pub fn blake2b512(bytes: &[u8]) -> Digest {
    let _ = bytes;
    todo!("blake2b512")
}

/// ffjavascript's `ChaCha`, which is RFC 8439 ChaCha20 with a zero nonce and words 12 to
/// 15 acting as one 128-bit little-endian block counter (`chacha.js:41-95`).
///
/// Only [`CeremonyRng::next_u32`] reads the stream directly. The other two are built on
/// top of it and do not match `rand_core`'s versions.
pub struct CeremonyRng {
    /// Scaffold placeholder. The real state is the keystream source plus the index of the
    /// next word, since the stream is consumed a word at a time.
    #[allow(dead_code)]
    state: (),
}

impl CeremonyRng {
    /// Seed from eight words that have already been read big-endian out of a digest.
    pub fn from_seed_words(words: [u32; 8]) -> Self {
        let _ = words;
        todo!("CeremonyRng::from_seed_words")
    }

    /// Seed from the **first 32 bytes** of a digest, read big-endian per word. Bytes 32
    /// onwards of a 64-byte digest are discarded, in every caller
    /// (`misc.js:194-196`, `keypair.js:29`).
    pub fn from_digest(digest: &[u8]) -> Self {
        let _ = digest;
        todo!("CeremonyRng::from_digest")
    }

    pub fn next_u32(&mut self) -> u32 {
        todo!("CeremonyRng::next_u32")
    }

    /// `nextU64`: two words, **high half first** (`chacha.js:68-70`).
    pub fn next_u64(&mut self) -> u64 {
        todo!("CeremonyRng::next_u64")
    }

    /// `nextBool`: the low bit of a whole consumed word (`chacha.js:72-74`).
    pub fn next_bool(&mut self) -> bool {
        todo!("CeremonyRng::next_bool")
    }
}

/// `getRandomRng` (`misc.js:182-199`): BLAKE2b-512 over 64 OS-random bytes **then** the
/// UTF-8 entropy string, with no NUL and no length prefix, seeded from the first 32 bytes
/// of the digest.
///
/// Deliberately not reproducible. Both `contribute` commands are non-deterministic by
/// design, and a Rust port that made them deterministic would be a weaker ceremony, not a
/// more testable one. [`rng_from_entropy_with`] is the test hook.
pub fn rng_from_entropy(entropy: &str) -> CeremonyRng {
    let _ = entropy;
    todo!("rng_from_entropy")
}

/// [`rng_from_entropy`] with the 64 OS-random bytes injected, so a run can be compared
/// against snarkjs byte for byte. Test-only in spirit; the CLI never calls it.
pub fn rng_from_entropy_with(os_bytes: &[u8; 64], entropy: &str) -> CeremonyRng {
    let _ = (os_bytes, entropy);
    todo!("rng_from_entropy_with")
}

/// `rngFromBeaconParams` (`misc.js:201-228`): SHA-256 iterated exactly
/// `2^num_iterations_exp` times, the first over `beacon_hash` itself and every later one
/// over the previous 32-byte digest, then eight big-endian words of that digest.
///
/// This is the deterministic half of the ceremony and where byte-comparison against
/// snarkjs belongs. Callers bound `num_iterations_exp` to `10..=63`.
pub fn rng_from_beacon_params(beacon_hash: &[u8], num_iterations_exp: u8) -> CeremonyRng {
    let _ = (beacon_hash, num_iterations_exp);
    todo!("rng_from_beacon_params")
}

/// `Fr.fromRng` (`wasm_field1.js:195-207`). Eight `next_u32` per attempt, assembled as
/// four little-endian u64 limbs each of which takes its **high** word first, masked to 254
/// bits, redrawn while at or above the modulus, and installed as Montgomery limbs.
pub fn fr_from_rng(rng: &mut CeremonyRng) -> Fr {
    let _ = rng;
    todo!("fr_from_rng")
}

/// [`fr_from_rng`] over the base field. Same 254-bit mask, since BN254's two primes have
/// the same bit length.
pub fn fq_from_rng(rng: &mut CeremonyRng) -> Fq {
    let _ = rng;
    todo!("fq_from_rng")
}

/// `c0` then `c1`, each an independent rejection loop (`wasm_field2.js:149-156`).
pub fn fq2_from_rng(rng: &mut CeremonyRng) -> Fq2 {
    let _ = rng;
    todo!("fq2_from_rng")
}

/// `G1.fromRng` (`wasm_curve.js:309-334`): draw `x`, draw the `greatest` bool, retry the
/// whole attempt when `x^3 + b` is not square. G1 has no cofactor on BN254.
pub fn g1_from_rng(rng: &mut CeremonyRng) -> G1Affine {
    let _ = rng;
    todo!("g1_from_rng")
}

/// `G2.fromRng`, then a literal scalar multiplication by BN254's G2 cofactor
/// `0x30644e72e131a029b85045b68181585e06ceecda572a2489345f2299c0f9fa8d`
/// (`bn128.js:43`). Not a psi-based cofactor clearing: prove the two agree before
/// substituting one.
pub fn g2_from_rng(rng: &mut CeremonyRng) -> G2Affine {
    let _ = rng;
    todo!("g2_from_rng")
}

/// `hashToG2` (`keypair.js:24-36`): seed a ChaCha from the first 32 bytes of the digest,
/// big-endian per word, and take one `G2.fromRng`. No SWU, no domain separation string.
pub fn hash_to_g2(digest: &Digest) -> G2Affine {
    let _ = digest;
    todo!("hash_to_g2")
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
    let _ = (personalization, challenge, g1_s, g1_sx);
    todo!("get_g2_sp")
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

/// `createPTauKey` (`keypair.js:61-74`). The RNG is consumed as `Fr, Fr, Fr, G1, G1, G1`:
/// the three private scalars up front in tau/alpha/beta order, then one `g1_s` per key in
/// the same order. Nothing else touches it, and the order is load-bearing for a beacon.
pub fn create_ptau_key(rng: &mut CeremonyRng, challenge: &Digest) -> PtauKey {
    let _ = (rng, challenge);
    todo!("create_ptau_key")
}

/// The 768-byte pubkey blob. `montgomery` picks the encoding, and both are used on the
/// same nine points in the same command: `true` is the LEM form stored in section 7
/// (`powersoftau_utils.js:253`), `false` is the uncompressed form that is hashed
/// (`:176-180`). Getting it backwards yields a wrong response hash and nothing else.
pub fn write_ptau_pubkey(keys: &PtauPubKeys, montgomery: bool) -> [u8; PTAU_PUBKEY_BYTES] {
    let _ = (keys, montgomery);
    todo!("write_ptau_pubkey")
}

/// Read the nine stored points and recompute the three `g2_sp` from `challenge`.
pub fn read_ptau_pubkey(bytes: &[u8], challenge: &Digest) -> Result<PtauPubKeys, CeremonyError> {
    let _ = (bytes, challenge);
    todo!("read_ptau_pubkey")
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
pub fn create_delta_key(rng: &mut CeremonyRng, prior: Transcript) -> (DeltaKey, Digest) {
    let _ = (rng, prior);
    todo!("create_delta_key")
}

/// Uncompressed G1: `x` then `y`, each 32 bytes, non-Montgomery **big-endian**. Infinity
/// is all zero bytes, and it stays all zero bytes: `_UtoLEM` detects infinity from bit
/// `0x40` instead, so the round trip is asymmetric. Do not "fix" that.
pub fn g1_uncompressed(p: &G1Affine) -> [u8; SG1] {
    let _ = p;
    todo!("g1_uncompressed")
}

/// Uncompressed G2: `x.c1, x.c0, y.c1, y.c0`, each 32 bytes big-endian. The swap is what
/// the byte reversal over the whole 64-byte `F2` blob produces, and it is the same
/// ordering Ethereum's pairing precompiles use.
pub fn g2_uncompressed(p: &G2Affine) -> [u8; SG2] {
    let _ = p;
    todo!("g2_uncompressed")
}

pub fn g1_from_uncompressed(bytes: &[u8]) -> Result<G1Affine, CeremonyError> {
    let _ = bytes;
    todo!("g1_from_uncompressed")
}

pub fn g2_from_uncompressed(bytes: &[u8]) -> Result<G2Affine, CeremonyError> {
    let _ = bytes;
    todo!("g2_from_uncompressed")
}

/// Compressed G1, 32 bytes: big-endian normal-form `x`, with `0x80` OR'd into byte 0 when
/// `sign(y) == -1`, and byte 0 = `0x40` with the rest zero for infinity
/// (`build_curve_jacobian_a0.js:1164-1201`). `f1m_sign` is 0 for zero, -1 when the
/// canonical value is at or above `(q-1)/2 + 1`, else 1.
///
/// Used only by phase 1's response hash and the response file. Nothing in a `.ptau` body
/// is compressed.
pub fn g1_compressed(p: &G1Affine) -> [u8; SCG1] {
    let _ = p;
    todo!("g1_compressed")
}

/// Compressed G2, 64 bytes. `f2m_sign` is `sign(c1)`, falling back to `sign(c0)` when
/// `c1` is zero (`build_f2m.js:412-429`).
pub fn g2_compressed(p: &G2Affine) -> [u8; SCG2] {
    let _ = p;
    todo!("g2_compressed")
}

pub fn g1_from_compressed(bytes: &[u8]) -> Result<G1Affine, CeremonyError> {
    let _ = bytes;
    todo!("g1_from_compressed")
}

pub fn g2_from_compressed(bytes: &[u8]) -> Result<G2Affine, CeremonyError> {
    let _ = bytes;
    todo!("g2_from_compressed")
}

/// `sameRatio` (`misc.js:129-137`): reject any zero input, then check
/// `e(a1, b2) * e(-b1, a2) == 1`. Every phase-1 and phase-2 verification step is one of
/// these, and the zero rejection is not decoration: without it the identity passes every
/// ratio.
pub fn same_ratio(a1: &G1Affine, b1: &G1Affine, a2: &G2Affine, b2: &G2Affine) -> bool {
    let _ = (a1, b1, a2, b2);
    todo!("same_ratio")
}
