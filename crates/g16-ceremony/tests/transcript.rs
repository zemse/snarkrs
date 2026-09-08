//! Byte vectors for [`g16_ceremony::transcript`], every one of them printed by snarkjs
//! 0.7.6 and its own dependencies rather than derived here.
//!
//! This module is the only thing standing between us and a beacon that is well-formed,
//! parses, and reproduces on nobody else's machine. Everything else in the crate can be
//! checked by "does snarkjs read it back"; the RNG and the transcript cannot, because a
//! wrong draw produces a different but equally valid-looking contribution.
//!
//! The vectors come from `tv-transcript.mjs`, run against snarkjs 0.7.6 with
//! ffjavascript 0.3.1, wasmcurves 0.2.2 and @noble/hashes 1.8.0. It is reproduced here in
//! outline so it can be regenerated rather than trusted:
//!
//! ```js
//! import { ChaCha, buildBn128, Scalar } from "ffjavascript";
//! import { blake2b } from "@noble/hashes/blake2b";
//! import { sha256 } from "@noble/hashes/sha256";
//! import { createPTauKey, getG2sp, hashToG2 } from "snarkjs/src/keypair.js";
//! import { toPtauPubKeyRpr } from "snarkjs/src/powersoftau_utils.js";
//! import { toPartialHash } from "snarkjs/src/misc.js";
//! const curve = await buildBn128();
//! const seedFromDigest = (d) => { const v = new DataView(d.buffer, d.byteOffset, d.byteLength);
//!     const s = []; for (let i=0;i<8;i++) s[i] = v.getUint32(i*4, false); return s; };
//! // e.g. the beacon draw:
//! let cur = new Uint8Array([0x0a,0x0b,0x0c,0x0d]);
//! for (let i=0;i<1024;i++) cur = sha256(cur);
//! const rng = new ChaCha(seedFromDigest(cur));
//! const prv = curve.Fr.fromRng(rng);
//! const g1s = curve.G1.toAffine(curve.G1.fromRng(rng));
//! ```
//!
//! Word counts are pinned indirectly: after a draw, the next word off the RNG must equal
//! word `n` of a fresh stream on the same seed. That pins both the count and the stream.

use g16_ceremony::transcript::*;
use g16_ceremony::write::{g1_lem, g2_lem};
use g16_field::{AffineRepr, BigInteger, Fq, Fq2, Fr, G1Affine, G2Affine};
use std::str::FromStr;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
        .collect()
}

/// The raw little-endian buffer `Fr.fromRng` returns, which is the Montgomery limbs.
fn fr_raw(v: &Fr) -> String {
    hex(&v.0.to_bytes_le())
}

fn fq_raw(v: &Fq) -> String {
    hex(&v.0.to_bytes_le())
}

/// Word `index` of the stream on `seed`, so a draw's cost can be asserted by reading the
/// word that must come next.
fn stream_word(seed: [u32; 8], index: usize) -> u32 {
    let mut rng = CeremonyRng::from_seed_words(seed);
    for _ in 0..index {
        rng.next_u32();
    }
    rng.next_u32()
}

const SEED_1_8: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

// ---------------------------------------------------------------------------- ChaCha

/// `new ChaCha([0;8])`, first forty `nextU32()`. This is the test that settles
/// `rand_chacha`'s word order and its block batching: words 16 onwards come from later
/// keystream blocks, so a crate that reordered or re-batched them would fail here rather
/// than in a ceremony six months from now.
#[test]
fn chacha_zero_seed_matches_ffjavascript() {
    let expected = [
        0xade0b876u32,
        0x903df1a0,
        0xe56a5d40,
        0x28bd8653,
        0xb819d2bd,
        0x1aed8da0,
        0xccef36a8,
        0xc70d778b,
        0x7c5941da,
        0x8d485751,
        0x3fe02477,
        0x374ad8b8,
        0xf4b8436a,
        0x1ca11815,
        0x69b687c3,
        0x8665eeb2,
        0xbee7079f,
        0x7a385155,
        0x7c97ba98,
        0x0d082d73,
    ];
    let mut rng = CeremonyRng::from_seed_words([0; 8]);
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(rng.next_u32(), *want, "word {i}");
    }
    // Words 32..40, two blocks in.
    let block2 = [
        0xe6a0092du32,
        0xe16c2663,
        0x08d17eae,
        0x75a06819,
        0x998e718e,
        0xc662d37b,
        0x3446c3b0,
        0x5db3a0a9,
    ];
    for _ in 20..32 {
        rng.next_u32();
    }
    for (i, want) in block2.iter().enumerate() {
        assert_eq!(rng.next_u32(), *want, "word {}", 32 + i);
    }
}

/// `nextU64` takes the high word first and `nextBool` burns a whole word, both the
/// opposite of what `rand_core` would do (`chacha.js:68-74`).
#[test]
fn chacha_u64_and_bool_word_order() {
    let words = [
        0x8891120du32,
        0xa1d7fd60,
        0x546a15d9,
        0xf896cad8,
        0x12976914,
        0x7cf5a49d,
        0x9ec2376e,
        0xafb55bc1,
    ];
    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    for (i, want) in words.iter().enumerate() {
        assert_eq!(rng.next_u32(), *want, "word {i}");
    }

    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    assert_eq!(rng.next_u64(), 0x8891_120d_a1d7_fd60);
    // Two words consumed, not one: the third word is next.
    assert_eq!(rng.next_u32(), 0x546a15d9);

    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    assert!(rng.next_bool());
    assert_eq!(rng.next_u32(), 0xa1d7fd60);
}

/// `keypair.js:29` and `misc.js:194-196` read the seed **big-endian per word** out of the
/// first 32 bytes of the digest and drop the rest.
#[test]
fn seed_from_digest_is_big_endian_and_ignores_the_tail() {
    let digest = blake2b512(b"test");
    assert_eq!(
        hex(&digest),
        concat!(
            "a71079d42853dea26e453004338670a53814b78137ffbed07603a41d76a483aa",
            "9bc33b582f77d30a65e6f29a896c0411f38312e1d66e0bf16386c86a89bea572",
        )
    );
    let words = [
        0xa71079d4u32,
        0x2853dea2,
        0x6e453004,
        0x338670a5,
        0x3814b781,
        0x37ffbed0,
        0x7603a41d,
        0x76a483aa,
    ];
    let mut a = CeremonyRng::from_digest(&digest);
    let mut b = CeremonyRng::from_seed_words(words);
    for _ in 0..8 {
        assert_eq!(a.next_u32(), b.next_u32());
    }
    let mut a = CeremonyRng::from_digest(&digest);
    let first_eight = [
        0x0fb56f52u32,
        0xaec47054,
        0x72bc3fff,
        0x17468528,
        0xddc605af,
        0xa4a7d27b,
        0x058fdcfb,
        0xc123baa5,
    ];
    for (i, want) in first_eight.iter().enumerate() {
        assert_eq!(a.next_u32(), *want, "word {i}");
    }
}

// --------------------------------------------------------------------------- BLAKE2b

#[test]
fn blake2b512_known_digests() {
    assert_eq!(
        hex(&blake2b512(b"")),
        concat!(
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419",
            "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
        )
    );
    assert_eq!(
        hex(&blake2b512(b"test")),
        concat!(
            "a71079d42853dea26e453004338670a53814b78137ffbed07603a41d76a483aa",
            "9bc33b582f77d30a65e6f29a896c0411f38312e1d66e0bf16386c86a89bea572",
        )
    );
}

/// [`Transcript`] drives its own compression loop because `partial_hash` needs the state,
/// so it has to be held against an implementation that did not come from the same head.
/// Lengths straddle the 128-byte block on both sides and the chunking varies, since
/// chunk boundaries are exactly what the hand-rolled buffer logic could get wrong.
#[test]
fn transcript_agrees_with_the_blake2_crate() {
    let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 + 7) as u8).collect();
    for len in [0usize, 1, 63, 127, 128, 129, 200, 255, 256, 383, 384, 1000] {
        let msg = &data[..len];
        let mut t = Transcript::new();
        t.update(msg);
        assert_eq!(t.finalize(), blake2b512(msg), "one-shot, len {len}");

        for chunk in [1usize, 7, 64, 128, 129] {
            let mut t = Transcript::new();
            for part in msg.chunks(chunk) {
                t.update(part);
            }
            assert_eq!(
                t.finalize(),
                blake2b512(msg),
                "len {len} in {chunk}-byte chunks"
            );
        }
    }
}

/// `toPartialHash` / `fromPartialHash` (`misc.js:89-127`), on 200 bytes so one block is
/// compressed and 72 stay buffered. The 128-byte prefix is the block buffer, the next 64
/// are the chaining words little-endian, then `length - pos`, a zero, and `pos`.
#[test]
fn partial_hash_matches_snarkjs_and_resumes() {
    let msg: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
    let mut t = Transcript::new();
    t.update(&msg);

    let expected = concat!(
        "838a91989fa6adb4bbc2c9d0d7dee5ecf3fa01080f161d242b323940474e555c",
        "636a71787f868d949ba2a9b0b7bec5ccd3dae1e8eff6fd040b121920272e353c",
        "434a51585f666d74000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "e849e71feec63152d0f9bb18e7e6e2e8694718aaef9cc31ece01327328ca8d7c",
        "901c5c6d97182f65ad5bffaaa524c8b51d9bcc8a6cf02c1a546952072afa337e",
        "800000000000000048000000000000000000000000000000",
    );
    let blob = t.partial_hash();
    assert_eq!(hex(&blob), expected);

    let tail = [9u8, 8, 7, 6, 5];
    let mut resumed = Transcript::from_partial_hash(&blob).unwrap();
    resumed.update(&tail);
    assert_eq!(
        hex(&resumed.finalize()),
        concat!(
            "8be5e41187a2f96581705e50f1a053fdfe14721dbcdd5c72989356c45f8daee7",
            "0d4745a60cbafaf3df2d1eb5751f733cb6404f55a58f844de3f2cd6a112b8b2c",
        )
    );
}

#[test]
fn partial_hash_rejects_an_impossible_buffer_fill() {
    let mut blob = [0u8; PARTIAL_HASH_BYTES];
    blob[200] = 200;
    assert!(Transcript::from_partial_hash(&blob).is_err());
}

/// `hashU32` is the one integer in the whole ceremony that is big-endian
/// (`zkey_new.js:579-584`).
#[test]
fn update_u32_is_big_endian() {
    let mut t = Transcript::new();
    t.update_u32_be(0x0102_0304);
    assert_eq!(t.finalize(), blake2b512(&[1, 2, 3, 4]));
}

// -------------------------------------------------------------------- point encodings

const G1_GEN_U: &str = "0000000000000000000000000000000000000000000000000000000000000001\
                        0000000000000000000000000000000000000000000000000000000000000002";
const G2_GEN_U: &str = "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2\
                        1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed\
                        090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b\
                        12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa";

fn strip(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The hashing encoding is big-endian with `c1` first, and it is not the on-disk one. Both
/// orderings are produced for the same G2 generator in the same command, so pin them side
/// by side.
#[test]
fn uncompressed_is_big_endian_with_c1_first() {
    assert_eq!(
        hex(&g1_uncompressed(&G1Affine::generator())),
        strip(G1_GEN_U)
    );
    assert_eq!(
        hex(&g2_uncompressed(&G2Affine::generator())),
        strip(G2_GEN_U)
    );
    // The on-disk codec for the same point is little-endian Montgomery, so it shares no
    // bytes with the hashing form.
    assert_ne!(
        hex(&g1_lem(&G1Affine::generator())),
        hex(&g1_uncompressed(&G1Affine::generator()))
    );
}

#[test]
fn g2_lem_and_g2_uncompressed_disagree_on_purpose() {
    let lem = g2_lem(&G2Affine::generator());
    let unc = g2_uncompressed(&G2Affine::generator());
    assert_eq!(
        hex(&lem),
        strip(
            "2620bc02d1b5838e72017b493519ebdcdf1a81974726b8fb3b5096af41385719\
             40614ca87d73b4afc4d802585add4360862fa052fc50e9096b7bea3a83f0fe14\
             f6e96b889dfa9d61789b9ef597d27ffefe7d1b23621a9eff06429eaeeb7efd28\
             ee5618c7565b0964bb3c7d3222f957dc76103533be35f9558264fd93e6a0a40d"
        )
    );
    assert_ne!(hex(&lem), hex(&unc));
}

#[test]
fn negated_and_infinite_points_encode_as_snarkjs_writes_them() {
    let n1 = -G1Affine::generator();
    let n2 = -G2Affine::generator();
    assert_eq!(
        hex(&g1_uncompressed(&n1)),
        strip(
            "0000000000000000000000000000000000000000000000000000000000000001\
             30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45"
        )
    );
    assert_eq!(
        hex(&g2_uncompressed(&n2)),
        strip(
            "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2\
             1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed\
             275dc4a288d1afb3cbb1ac09187524c7db36395df7be3b99e673b13a075a65ec\
             1d9befcd05a5323e6da4d435f3b617cdb3af83285c2df711ef39c01571827f9d"
        )
    );
    // `_LEMtoU` writes all zeros for infinity, no flag byte
    // (`build_curve_jacobian_a0.js:1216-1221`), and that blob decodes back to infinity
    // through the ordinary path because (0, 0) is how the affine zero is spelled.
    assert_eq!(
        hex(&g1_uncompressed(&G1Affine::identity())),
        "00".repeat(64)
    );
    assert!(g1_from_uncompressed(&[0u8; 64]).unwrap().is_zero());
    assert!(g2_from_uncompressed(&[0u8; 128]).unwrap().is_zero());
}

#[test]
fn compressed_matches_snarkjs() {
    let cases: [(&str, G1Affine); 3] = [
        (
            "0000000000000000000000000000000000000000000000000000000000000001",
            G1Affine::generator(),
        ),
        (
            "8000000000000000000000000000000000000000000000000000000000000001",
            -G1Affine::generator(),
        ),
        (
            "4000000000000000000000000000000000000000000000000000000000000000",
            G1Affine::identity(),
        ),
    ];
    for (want, p) in cases {
        assert_eq!(hex(&g1_compressed(&p)), want);
        assert_eq!(g1_from_compressed(&unhex(want)).unwrap(), p);
    }

    let g2_cases: [(&str, G2Affine); 3] = [
        (
            "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2\
             1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
            G2Affine::generator(),
        ),
        (
            "998e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2\
             1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed",
            -G2Affine::generator(),
        ),
        (
            "4000000000000000000000000000000000000000000000000000000000000000\
             0000000000000000000000000000000000000000000000000000000000000000",
            G2Affine::identity(),
        ),
    ];
    for (want, p) in g2_cases {
        let want = strip(want);
        assert_eq!(hex(&g2_compressed(&p)), want);
        assert_eq!(g2_from_compressed(&unhex(&want)).unwrap(), p);
    }
}

#[test]
fn point_codecs_round_trip_on_random_points() {
    let mut seed = CeremonyRng::from_seed_words([9, 9, 9, 9, 9, 9, 9, 9]);
    for _ in 0..8 {
        let p1 = g1_from_rng(&mut seed);
        let p2 = g2_from_rng(&mut seed);
        assert_eq!(g1_from_uncompressed(&g1_uncompressed(&p1)).unwrap(), p1);
        assert_eq!(g2_from_uncompressed(&g2_uncompressed(&p2)).unwrap(), p2);
        assert_eq!(g1_from_compressed(&g1_compressed(&p1)).unwrap(), p1);
        assert_eq!(g2_from_compressed(&g2_compressed(&p2)).unwrap(), p2);
    }
}

#[test]
fn a_coordinate_at_or_above_the_modulus_is_rejected() {
    let mut bytes = [0xffu8; 64];
    bytes[0] = 0x30;
    assert!(g1_from_uncompressed(&bytes).is_err());
}

// ---------------------------------------------------------------------- field sampling

/// `Fr.fromRng` draws four `nextU64`s (eight words), masks to 254 bits, and installs the
/// result as **Montgomery** limbs, so the element's value is `v * R^-1`
/// (`wasm_field1.js:195-207`, `f1field.js:304-315`).
#[test]
fn fr_from_rng_matches_ffjavascript() {
    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    let v = fr_from_rng(&mut rng);
    assert_eq!(
        fr_raw(&v),
        "60fdd7a10d129188d8ca96f8d9156a549da4f57c14699712c15bb5af6e37c21e"
    );
    assert_eq!(
        v,
        Fr::from_str(
            "11421152797359801513699257512876826534900512878648310510227136947029091328776"
        )
        .unwrap()
    );
    // Exactly eight words, no more.
    assert_eq!(rng.next_u32(), stream_word(SEED_1_8, 8));
}

#[test]
fn fq_and_fq2_from_rng_match_ffjavascript() {
    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    let v = fq_from_rng(&mut rng);
    assert_eq!(
        fq_raw(&v),
        "60fdd7a10d129188d8ca96f8d9156a549da4f57c14699712c15bb5af6e37c21e"
    );
    assert_eq!(
        v,
        Fq::from_str(
            "2550808096805763633707842382229829953173348836788035625838433109551305386664"
        )
        .unwrap()
    );
    assert_eq!(rng.next_u32(), stream_word(SEED_1_8, 8));

    // `wasm_field2.js:149-156` writes the first draw at offset 0, which is c0.
    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    let v: Fq2 = fq2_from_rng(&mut rng);
    assert_eq!(
        format!("{}{}", fq_raw(&v.c0), fq_raw(&v.c1)),
        concat!(
            "60fdd7a10d129188d8ca96f8d9156a549da4f57c14699712c15bb5af6e37c21e",
            "06b162f2df27ce4597bc554f50fe4b5f9c341f8fad15d5fe3b86576c0e540e2d",
        )
    );
    assert_eq!(rng.next_u32(), stream_word(SEED_1_8, 16));
}

/// `G.fromRng` is try-and-increment with the `greatest` bool drawn between `x` and the
/// square test, so a rejected attempt costs the bool too. On this seed G1 takes two
/// attempts (18 words) and G2 takes two attempts with an extra Fq rejection inside one of
/// them (50 words). Both numbers are what ffjavascript actually consumed.
#[test]
fn g1_and_g2_from_rng_match_ffjavascript() {
    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    let p = g1_from_rng(&mut rng);
    assert_eq!(
        hex(&g1_uncompressed(&p)),
        strip(
            "0a7ff1694a79046066b7967c3351100f2674fee9758c26c0622faac5e4469698\
             0744637bbe02b7d14e2158c7872ba1c6eb6bafa0db7cff8d0409d1bc376b134f"
        )
    );
    assert!(p.is_on_curve());
    assert_eq!(rng.next_u32(), stream_word(SEED_1_8, 18));

    let mut rng = CeremonyRng::from_seed_words(SEED_1_8);
    let p = g2_from_rng(&mut rng);
    assert_eq!(
        hex(&g2_uncompressed(&p)),
        strip(
            "2e32b9a8c1ab7b1a54e9a9c83480cc2cba00e452dd14ab791004b5d73a27691e\
             046a3f44c06efa3c450185453d0e37cd0363bdb676f108e104ce452123cb4628\
             14a8e9ffd499baddfbdbac21789623fdba132131150ade912d8576b455ef966d\
             1a08e99e008b8db296dbfc3380e6ebff7565be7420e27dda9f6937341f2e54ef"
        )
    );
    assert!(p.is_on_curve());
    // The cofactor multiplication is the point of `g2_from_rng`: without it the result
    // is on the curve but outside the pairing group.
    assert!(p.is_in_correct_subgroup_assuming_on_curve());
    assert_eq!(rng.next_u32(), stream_word(SEED_1_8, 50));
}

// ------------------------------------------------------------------------ beacon path

/// The deterministic half of the ceremony, and the only place a Rust beacon can be proved
/// equal to a snarkjs one. `rngFromBeaconParams(0a0b0c0d, 10)` iterates SHA-256 exactly
/// 1024 times (`misc.js:201-228`), then the first three draws are `zkey_beacon.js:56-58`.
#[test]
fn beacon_rng_matches_snarkjs() {
    let mut rng = rng_from_beacon_params(&[0x0a, 0x0b, 0x0c, 0x0d], 10);
    let expected_seed = unhex("38a809c6c8540db7555f3dcfeceac2a11d126cfc7d98a8f0f7fc53967cb9ecac");
    let mut reference = CeremonyRng::from_digest(&expected_seed);
    for _ in 0..4 {
        assert_eq!(rng.next_u32(), reference.next_u32());
    }

    let mut rng = rng_from_beacon_params(&[0x0a, 0x0b, 0x0c, 0x0d], 10);
    let prv = fr_from_rng(&mut rng);
    assert_eq!(
        fr_raw(&prv),
        "42ef4b6b13fed6bbada810fce61ec502cb63c159875cd510741f9b21930fcd05"
    );
    assert_eq!(
        prv,
        Fr::from_str("171432668985670609303570112350389340347008608898847443863786640552402138124")
            .unwrap()
    );
    let g1_s = g1_from_rng(&mut rng);
    assert_eq!(
        hex(&g1_uncompressed(&g1_s)),
        strip(
            "0e3bb2ff6d8e225e61c3ef8abc2c7aa638d07556de9f6d035d2b9175c79bafbd\
             26208b1de070ac9b07f3b109c0f308e6646829b244787f50d6bd82d2771ced9e"
        )
    );
    // `timesFr` de-Montgomeries the scalar first, so this is a multiplication by the
    // element's value and not by its limbs (`build_bn128.js:54-72`).
    let g1_sx: G1Affine = (g1_s * prv).into();
    assert_eq!(
        hex(&g1_uncompressed(&g1_sx)),
        strip(
            "1d4848cedc231d31a12c875a1ca477246b3ab550e6b18eb6c90b11469a6d621c\
             2d92c7373e9fe1e6271c5b42d62a0625cd77eb02954ed07b13a98e443c720596"
        )
    );
}

/// `getRandomRng` is not reproducible in production, but its shape is: BLAKE2b-512 over
/// the 64 OS bytes **then** the UTF-8 entropy, no NUL and no length prefix
/// (`misc.js:187-191`). The entropy here is non-ASCII so the encoding is pinned too.
#[test]
fn entropy_rng_hashes_os_bytes_then_utf8_entropy() {
    let mut os = [0u8; 64];
    for (i, b) in os.iter_mut().enumerate() {
        *b = i as u8;
    }
    let mut rng = rng_from_entropy_with(&os, "cafébabe");
    let expected = [0x9f445299u32, 0xd0a03ad0, 0x5c7f5e5a, 0x03611228];
    for (i, want) in expected.iter().enumerate() {
        assert_eq!(rng.next_u32(), *want, "word {i}");
    }
    let mut rng = rng_from_entropy_with(&os, "cafébabe");
    assert_eq!(
        fr_raw(&fr_from_rng(&mut rng)),
        "d03aa0d09952449f281261035a5e7f5c8f3937dca347eebb9fc2182f8b9b6710"
    );
}

// --------------------------------------------------------------- hashToG2 and the keys

/// `hashToG2` (`keypair.js:24-36`): the first 32 bytes of the digest seed a ChaCha and the
/// result is one `G2.fromRng`. No SWU, no domain separation string.
#[test]
fn hash_to_g2_matches_snarkjs() {
    let mut digest = [0u8; 64];
    for (i, b) in digest.iter_mut().enumerate() {
        *b = i as u8;
    }
    assert_eq!(
        hex(&g2_uncompressed(&hash_to_g2(&digest))),
        strip(
            "097197b18b86f218d43cf232ae11cb0318f9ff79c8315ba8cc3494ece86bd08c\
             0993c1f526542d4ea0f7960fe68c321263da35704524158f53a5ec1f6ae3cc0c\
             07e5f5e692f511f13c16f270392204a3e2ebf321f8b80f73d87faaea048b7bf0\
             1f619cecd6aebc1a35e6397979d76d2caa56e4cacbc10cd20d93cf715a0dd968"
        )
    );
}

/// `getG2sp` (`keypair.js:38-51`): 193 bytes in, one personalisation byte first. Both
/// personalisations are pinned because a swapped byte still produces a valid point.
#[test]
fn get_g2_sp_matches_snarkjs() {
    let challenge = [0xAAu8; 64];
    let g = G1Affine::generator();
    let two_g: G1Affine = (g + g).into();
    assert_eq!(
        hex(&g2_uncompressed(&get_g2_sp(
            PERSONALIZATION_TAU,
            &challenge,
            &g,
            &two_g
        ))),
        strip(
            "24b5b7a7cc785f36b1462ddc7dce8c3da75da4d61230865590586e27f332ea7c\
             0e61379abf8480537e27830efb5df865f9c4a8b79696f91f0af98f49627974f4\
             2dc5b1cf63facf1796140bda0517290a7b8326a81153837e645e6f56eed10a40\
             22f4850a2a1418e268559fc92f8f30babddcea55d230fcb92475362a3387d97f"
        )
    );
    assert_eq!(
        hex(&g2_uncompressed(&get_g2_sp(
            PERSONALIZATION_ALPHA,
            &challenge,
            &g,
            &two_g
        ))),
        strip(
            "15634f7ed11ac7ba991635c171936e339f6b895cfc32cd02ec8de843b9770f45\
             29e8150d95162ab22fa794605bfaf1a5a3eb34f376108ea75dc6a513eaeb21d4\
             2f9c37ee2d53a8cb695d0ace306f29d459f4d6cb7a9f81f01cc6bff891777eb7\
             2e9e2e18817e5e93d1c2ce59b8c5608d79dfeb723583fc9dc8c8d3e456f5128c"
        )
    );
}

/// The whole phase-1 key, drawn `Fr, Fr, Fr, G1, G1, G1` from one beacon RNG
/// (`keypair.js:61-74`). This pins the draw order, the personalisation assignment and both
/// pubkey serialisations at once; a single reordered draw changes every byte after it.
#[test]
fn create_ptau_key_matches_snarkjs() {
    let mut rng = rng_from_beacon_params(&[0x0a, 0x0b, 0x0c, 0x0d], 10);
    let challenge = blake2b512(b"");
    let key = create_ptau_key(&mut rng, &challenge);

    assert_eq!(
        fr_raw(&key.tau.prv_key),
        "42ef4b6b13fed6bbada810fce61ec502cb63c159875cd510741f9b21930fcd05"
    );
    assert_eq!(
        fr_raw(&key.alpha.prv_key),
        "6e217c68838683cd552195489e616112af3ebd5217f37ab08863669cfc25090e"
    );
    assert_eq!(
        fr_raw(&key.beta.prv_key),
        "f5ec0833ff058baecdf8c2a46cf1eba10d38ee14d0e3a0750167782c7d5f8e2f"
    );

    // tau's g2_sp is the one point in the record that is never stored: it has to be
    // rebuilt from the challenge, which is what chains a contribution to its predecessor.
    assert_eq!(
        hex(&g2_uncompressed(&key.tau.pubkey.g2_sp)),
        strip(
            "0f7295757b6b1a5b1dfa4598ebe3007f61418bff11de31b3b709ba6c2c402151\
             2078be990ca856748258566c2370ce481431846904eb414a6862aa107f36c136\
             10dbb564ee026a163f5fc65be730bb2d5d2add0a96b725474a0a132654fa9254\
             1cc3d2c8d712888c6502f81c270276e0b6a944fadf22c1b88fc48e1a19db1c32"
        )
    );

    let pubkeys = key.pubkeys();
    assert_eq!(hex(&write_ptau_pubkey(&pubkeys, false)), strip(PUBKEY_U));
    assert_eq!(hex(&write_ptau_pubkey(&pubkeys, true)), strip(PUBKEY_LEM));

    // The stored form is the Montgomery one, and reading it back must recover the three
    // `g2_sp` from the challenge alone (`powersoftau_utils.js:172-181`).
    let read = read_ptau_pubkey(&write_ptau_pubkey(&pubkeys, true), &challenge).unwrap();
    assert_eq!(read.tau.g1_s, pubkeys.tau.g1_s);
    assert_eq!(read.tau.g2_sp, pubkeys.tau.g2_sp);
    assert_eq!(read.alpha.g2_sp, pubkeys.alpha.g2_sp);
    assert_eq!(read.beta.g2_spx, pubkeys.beta.g2_spx);
}

/// `zkey_contribute.js:54-61`: no personalisation byte, and `g2_sp` comes from the
/// transcript this call closes. `zkey_verify_frominit.js:46-63` reconstructs it the same
/// way, so a divergence here is caught by snarkjs' own verifier.
#[test]
fn create_delta_key_closes_the_transcript_it_names() {
    let mut rng = rng_from_beacon_params(&[0x0a, 0x0b, 0x0c, 0x0d], 10);
    let mut prior = Transcript::new();
    prior.update(&blake2b512(b"cs"));
    let (delta, transcript) = create_delta_key(&mut rng, prior.clone());

    assert_eq!(
        fr_raw(&delta.prv_key),
        "42ef4b6b13fed6bbada810fce61ec502cb63c159875cd510741f9b21930fcd05"
    );
    assert_eq!(
        hex(&g1_uncompressed(&delta.g1_s)),
        strip(
            "0e3bb2ff6d8e225e61c3ef8abc2c7aa638d07556de9f6d035d2b9175c79bafbd\
             26208b1de070ac9b07f3b109c0f308e6646829b244787f50d6bd82d2771ced9e"
        )
    );

    let mut expected = prior;
    expected.update_g1(&delta.g1_s);
    expected.update_g1(&delta.g1_sx);
    assert_eq!(transcript, expected.finalize());
    assert_eq!(delta.g2_sp, hash_to_g2(&transcript));
    assert!(delta.g2_sp.is_in_correct_subgroup_assuming_on_curve());
    let spx: G2Affine = (delta.g2_sp * delta.prv_key).into();
    assert_eq!(delta.g2_spx, spx);
}

// ------------------------------------------------------------------------- same_ratio

#[test]
fn same_ratio_accepts_a_matching_pair_and_rejects_the_rest() {
    let mut rng = CeremonyRng::from_seed_words([4, 8, 15, 16, 23, 42, 0, 1]);
    let x = fr_from_rng(&mut rng);
    let y = fr_from_rng(&mut rng);
    let a1 = G1Affine::generator();
    let b1: G1Affine = (a1 * x).into();
    let a2 = G2Affine::generator();
    let b2: G2Affine = (a2 * x).into();
    assert!(same_ratio(&a1, &b1, &a2, &b2));

    let wrong: G2Affine = (a2 * y).into();
    assert!(!same_ratio(&a1, &b1, &a2, &wrong));

    // `misc.js:130-133` rejects a zero input before pairing, and it has to: the identity
    // satisfies every ratio.
    assert!(!same_ratio(&G1Affine::identity(), &b1, &a2, &b2));
    assert!(!same_ratio(&a1, &b1, &G2Affine::identity(), &b2));
}

const PUBKEY_U: &str = "\
277251382a30cae869bf70bf6ac8a1c3226f043866be95e93303cc1a8e3ac146\
21c8d894be043d16cd936102dc1e1ef39cd98b5807d45f606dbe7b055be56f4e\
1227ab6094b04f8c499e25eb8afb7d8522d7f0fed2132ca99c8a6d2f7cea707a\
2d2e85b102fff82cf5bc60d76b9d0524da862e3d3f58143aa4f33b49714ac6bc\
07db69fcd5e49e31af221d1f999607e63beb5834ed43561974d21ff429b48fda\
2e8578b5c28be4300f8a0908965711076137b2406f976335beb735f9c772d322\
09354b292150e795e7427a0865040ed8cdb2a186dd5394c6a0481af32068e70a\
1db16e2e169a2b9ab89e4dcde86935cd4d3af5d87d7c4c27c23af6ffba124a10\
14bfe3dd1bbe01964d43cdcac5e395ef3da4b2334a9449d79be8fe21fb5b48b4\
06f18b00e53f1404ee3c994df87c8eb25ab9574322ecee1380c8db00654b13ad\
14f2504a6e9518db6bc090d7d9a0e964b1365ac60b2c2b3406c3cf4b74881212\
1b561553fb9ba490742dc3ec2b83aef616d4c78f3984337409b127c30b611397\
2a7474efdbc618fdb111368efd55673be5739bdd4bea7e728eafc5d5d7147322\
1e6ae9e3be64f4b319fa0d72e01fc00a71a462929d39d78db7b2b40f651abb7f\
0f1458eda14aff2428e2d0c8a1091004287c56d7776aae2040234b3a5c703c89\
1c53296a39d6baf559f6c33777bd2309e441365b3bbc71ac2a207973198ba865\
2a00730bf5914a5e783d6fa2b8c3efd989f1d951a43f0cd1b18623614fe5c73f\
067f0b3319dc64b195c75ffcd57f4420323236657e61b54cea6db7267179c622\
1415f5616a8d5d8d54b392b7bda278262615199382f89e6093abbe0679cfc743\
23a7edb77ac0cc73b9e6fb32ea8005102acf1dc2d711b3896a473df7f88dde82\
26fd43cb150c4f04831e3144d750e9cf30e7797a2ff316be94d1b5e5869d3fd5\
1a566ff2e55083ae11d9cb9cef59544b08dcbf9edbf96898a3f7e4cccb5766b8\
15036e4c476fd1347d416c8efc969be74714591738047e58d7ad01456e4db188\
26ae496824fe710119d52e68b69943341c9d8eba37b8feb9550b081e1a3d49c9";

const PUBKEY_LEM: &str = "\
4389c9a49b28cd028893ef67766c3a67abea9ad19353db433e5b1be1a8b63527\
810ec3dc2360017562912815172d9db95d580c97b889f0de3966d6ec802b931f\
a240c90c850628e148021e0dfeac33ab3955daa3acf31aa72376bcc97ae4db02\
859776d88061227dafdbf250717149f732ae1b56d54bd60b0c2cd6fec7947f18\
534b8ab3afefb6241b1c8f819884ddb325d39d6ca4abb70582f00fd99dc09621\
d60119f29fc90c2a7aa44f4067290712eb2b3badea06fedb1bb7bfdacd182322\
adf84b25a71b8b83c8155c244361965b0e43db441d45894e1eb67606802d7229\
64a824de86b2068a80d044d6f7011b74857d23f105b1f30ddc68cd62f9a05e2f\
69da047cc0f64b78aab7a22f8417283379a9d8e72feab5fc0fd0e24187b02412\
4425e9eb7312a9d5c2ea70a36738590b4660a38529d57afa840c665e35785103\
a3f973d850ca6d9022a49657c15f07494d096439ce25d833aae447682e0a5c1d\
f2e5664962e40ce8399af96e8300ac987f4c9b1fb035b29a8777fe7770cfad13\
482c33dd8936a9b277444d3816c80442db33e1517f51ddcfe7912f4732ef9302\
fadfb7afa1f390fa65d8ed98a399205bfa78d651b0cdf21eca9d36a09bcafb02\
4c0470a86596f298ca767db29bc5837c3a9422a737f67d68a3a5fff44bce6a08\
1076f2b48836aeb01821d98ecb3ad086603e9c787fcb85e3b21b4b38b8bd682c\
1045eeb1471db0e43d557739445714b8eca812ceae664cd4dd43f11b10c0fc12\
0ddc07ad5d0c77ceb6eceda6729a58d5970e4f3061be51346b1d7a543e85fb08\
f12f516133a781bcf5f57e7143b9bb055c98db34873ec035161bd96eae22ce2f\
6bef6656071532e99ac33b4fdd2d26adc2bc5fab50248925f567758f9c9aee09\
b0d5324fdfda0acd43696037db916f1fb2116a3a07b20c940f58cb4b5715fd25\
493f77b6f88d27841d42aaff46f8eea116d098bc3507da9a6f65f2f645be9b2d\
03b98fbde61f4ae8c85e86f0d63ccaabbeef6143319378d809ad2e9d6a637f05\
9d49c67c587da250465be7c9aa44253a229b27202752d06e7fa59b5c7b1a3c27";
