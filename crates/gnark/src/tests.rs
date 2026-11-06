//! Parsing the committed toy fixture, and refusing every way of breaking it. The fixture
//! comes from `test-support/gnark` (`go run . gen toy`).

use super::*;
use crate::pk::DumpHeader;
use crate::witness::fr_slice;
use std::path::PathBuf;

fn toy(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-support/gnark/toy")
        .join(name)
}

fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(toy(name)).unwrap()
}

/// Byte offsets into the toy dump, from the layout in `pk.rs`.
const DOMAIN: usize = 8;
const GENERATOR: usize = DOMAIN + 8 + 32;
const ALPHA: usize = DOMAIN + 169;
const WIRES: usize = ALPHA + 3 * 64 + 2 * 128;
const FLAGS: usize = WIRES + 24;

fn wires(pk: &[u8]) -> usize {
    u64::from_be_bytes(pk[WIRES..WIRES + 8].try_into().unwrap()) as usize
}

/// Where the `u32` commitment count and then the first dumped slice's length sit.
fn after_flags(pk: &[u8]) -> usize {
    FLAGS + 2 * wires(pk)
}

#[test]
fn reads_the_toy_dump() {
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
    let h = &pk.header;
    assert_eq!(h.domain_size, 4096);
    assert_eq!(h.n_wires, 3072);
    // X, Y, Sum, Low and the constant wire.
    assert_eq!(h.n_public_wires(), 5);
    assert_eq!(h.g1_z, 4095);
    // The 64 bits of S[0] appear in no A or B row, so both drop points.
    assert!(h.nb_infinity_a > 0 && h.nb_infinity_b > 0, "{h:?}");
    assert_eq!(h.g1_a + h.nb_infinity_a, h.n_wires);
    assert_eq!(h.g2_b + h.nb_infinity_b, h.n_wires);
    for v in [&pk.a, &pk.b_g1, &pk.k, &pk.z] {
        assert!(v.iter().all(|p| p.is_on_curve()));
    }
    assert_eq!(pk.a.len(), h.n_wires);
    assert_eq!(pk.b_g2.len(), h.n_wires);
    assert_eq!(pk.k.len(), h.n_wires - 5);
    assert_eq!(
        pk.a.iter().filter(|p| p.infinity).count(),
        h.nb_infinity_a,
        "only the flagged wires are infinity in A"
    );
    assert_eq!(DumpHeader::read(&toy("pk.dump")).unwrap(), pk.header);
}

#[test]
fn the_verifying_key_matches_the_proving_key() {
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    assert_eq!(vk.alpha_g1, pk.alpha_g1);
    assert_eq!(vk.beta_g1, pk.beta_g1);
    assert_eq!(vk.delta_g1, pk.delta_g1);
    assert_eq!(vk.beta_g2, pk.beta_g2);
    assert_eq!(vk.delta_g2, pk.delta_g2);
    assert_eq!(vk.k.len(), pk.header.n_public_wires());
    assert_eq!(vk.n_public(), 4);
}

#[test]
fn compressed_and_raw_points_decode_alike() {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    for p in vk.k.iter().chain([&vk.alpha_g1, &vk.beta_g1, &vk.delta_g1]) {
        assert_eq!(encoded_g1(&raw_g1(p)).unwrap(), *p);
    }
    for p in [&vk.beta_g2, &vk.gamma_g2, &vk.delta_g2] {
        assert_eq!(encoded_g2(&raw_g2(p)).unwrap(), *p);
    }
    assert!(encoded_g1(&raw_g1(&G1Affine::identity())).unwrap().infinity);
    assert!(encoded_g2(&raw_g2(&G2Affine::identity())).unwrap().infinity);
    // Both signs of Y, through the compressed form gnark writes into vk.bin.
    let g = G1Affine::generator();
    let mut c = [0u8; 32];
    c.copy_from_slice(&raw_g1(&g)[..32]);
    c[0] |= COMPRESSED_SMALLEST;
    let small = encoded_g1(&c).unwrap();
    c[0] |= COMPRESSED_LARGEST;
    let large = encoded_g1(&c).unwrap();
    assert_eq!(small, -large);
    assert!(small == g || large == g);
}

#[test]
fn reads_the_toy_witness() {
    let w = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    assert_eq!(w.w().len(), 3072);
    assert_eq!(w.n_constraints(), 2071);
    assert_eq!(w.bw().len(), 2071);
    let public: Vec<Fr> = bytes("public.bin")
        .chunks(32)
        .map(|b| fr_be(b).unwrap())
        .collect();
    assert_eq!(w.public(4).unwrap(), &public[..]);
    // X = 5 is the first public input.
    assert_eq!(public[0], Fr::from(5u64));
}

/// Every prefix of each file is refused, and none panics.
#[test]
fn every_truncation_is_refused() {
    let pk = bytes("pk.dump");
    let step = (pk.len() / 997).max(1);
    for cut in (0..pk.len()).step_by(step).chain([pk.len() - 1]) {
        assert!(
            GnarkProvingKey::from_bytes(&pk[..cut]).is_err(),
            "pk cut at {cut}"
        );
    }
    let vk = bytes("vk.bin");
    for cut in 0..vk.len() {
        assert!(
            GnarkVerifyingKey::from_bytes(&vk[..cut]).is_err(),
            "vk cut at {cut}"
        );
    }
    let w = bytes("witness.bin");
    for cut in (0..w.len()).step_by(97).chain([w.len() - 1]) {
        assert!(
            GnarkWitness::from_bytes(&w[..cut]).is_err(),
            "witness cut at {cut}"
        );
    }
}

#[test]
fn trailing_bytes_are_refused() {
    for name in ["pk.dump", "vk.bin", "witness.bin"] {
        let mut b = bytes(name);
        b.push(0);
        let err = match name {
            "pk.dump" => GnarkProvingKey::from_bytes(&b).err(),
            "vk.bin" => GnarkVerifyingKey::from_bytes(&b).err(),
            _ => GnarkWitness::from_bytes(&b).err(),
        };
        assert!(err.unwrap().to_string().contains("trailing"), "{name}");
    }
}

/// A length that claims more than the file holds is refused before it is allocated. Were
/// it trusted, `u64::MAX` points would be an abort and 2^40 would be 64 TB.
#[test]
fn oversized_lengths_are_refused_without_allocating() {
    let pk = bytes("pk.dump");
    for huge in [u64::MAX, 1 << 40, (pk.len() as u64) / 64 + 1] {
        let mut b = pk.clone();
        b[WIRES..WIRES + 8].copy_from_slice(&huge.to_be_bytes());
        assert!(GnarkProvingKey::from_bytes(&b).is_err(), "nbWires {huge}");

        let mut b = pk.clone();
        let at = after_flags(&pk) + 4;
        b[at..at + 8].copy_from_slice(&huge.to_le_bytes());
        let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
        assert!(err.contains("G1.A"), "{err}");
    }

    let w = bytes("witness.bin");
    for huge in [u64::MAX, 1 << 40] {
        let mut b = w.clone();
        b[8..16].copy_from_slice(&huge.to_le_bytes());
        assert!(GnarkWitness::from_bytes(&b).is_err(), "W length {huge}");
    }

    let vk = bytes("vk.bin");
    // G1.K's count follows three compressed G1 and three compressed G2 points.
    let at = 3 * 32 + 3 * 64;
    for huge in [u32::MAX, 1 << 24] {
        let mut b = vk.clone();
        b[at..at + 4].copy_from_slice(&huge.to_be_bytes());
        let err = GnarkVerifyingKey::from_bytes(&b).err().unwrap().to_string();
        assert!(err.contains("G1.K"), "{err}");
    }
}

#[test]
fn a_slice_length_that_disagrees_with_the_flags_is_refused() {
    // Moving one wire's flag in InfinityA leaves G1.A one point short of the header.
    let mut b = bytes("pk.dump");
    let n = wires(&b);
    let i = (0..n).find(|&i| b[FLAGS + i] == 0).unwrap();
    b[FLAGS + i] = 1;
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("InfinityA"), "{err}");
    // And a flag that is not a bool.
    let mut b = bytes("pk.dump");
    b[FLAGS] = 2;
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("not a bool"), "{err}");
}

#[test]
fn commitments_are_refused() {
    let mut b = bytes("pk.dump");
    let at = after_flags(&b);
    b[at..at + 4].copy_from_slice(&1u32.to_be_bytes());
    assert!(matches!(
        GnarkProvingKey::from_bytes(&b),
        Err(GnarkError::Commitments { count: 1, .. })
    ));
    let mut b = bytes("vk.bin");
    let at = b.len() - 4;
    b[at..].copy_from_slice(&1u32.to_be_bytes());
    assert!(matches!(
        GnarkVerifyingKey::from_bytes(&b),
        Err(GnarkError::Commitments { count: 1, .. })
    ));
}

#[test]
fn a_foreign_domain_is_refused() {
    let mut b = bytes("pk.dump");
    b[GENERATOR + 31] ^= 1;
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("Generator"), "{err}");
    let mut b = bytes("pk.dump");
    b[DOMAIN..DOMAIN + 8].copy_from_slice(&3000u64.to_be_bytes());
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("power of two"), "{err}");
}

#[test]
fn a_bad_marker_is_refused() {
    for name in ["pk.dump", "witness.bin"] {
        let mut b = bytes(name);
        b[..8].copy_from_slice(&0xdead_beef_u64.to_be_bytes());
        let err = match name {
            "pk.dump" => GnarkProvingKey::from_bytes(&b).err(),
            _ => GnarkWitness::from_bytes(&b).err(),
        };
        assert!(matches!(err, Some(GnarkError::BadMarker(_))), "{name}");
    }
}

#[test]
fn a_zeroed_header_point_is_refused() {
    let mut b = bytes("pk.dump");
    // G1.Beta, the second raw header point.
    b[ALPHA + 64..ALPHA + 128].fill(0);
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("G1.Beta is the point at infinity"), "{err}");
}

#[test]
fn an_off_curve_query_point_is_refused() {
    let mut b = bytes("pk.dump");
    // The first point of G1.A: its x limb 0, one past the slice length.
    let at = after_flags(&b) + 4 + 8;
    b[at] ^= 1;
    let err = GnarkProvingKey::from_bytes(&b).err().unwrap().to_string();
    assert!(
        err.contains("G1.A") && err.contains("not on the curve"),
        "{err}"
    );
}

#[test]
fn an_off_curve_vk_point_is_refused() {
    let mut b = bytes("vk.bin");
    // Alpha's x, compressed: most x are not the abscissa of a point, so walk the low byte
    // until one is refused.
    let refused = (0..=255u8).any(|v| {
        b[31] = v;
        GnarkVerifyingKey::from_bytes(&b).is_err()
    });
    assert!(refused);
}

#[test]
fn witness_shape_is_checked() {
    let w = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let (mut ww, aw, bw) = (w.w().to_vec(), w.aw().to_vec(), w.bw().to_vec());
    assert!(GnarkWitness::from_parts(&ww, &aw, &bw[1..]).is_err());
    ww[0] = Fr::from(2u64);
    let err = GnarkWitness::from_parts(&ww, &aw, &bw)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("W[0]"), "{err}");

    // An element at the modulus, from the file and from memory.
    let mut b = bytes("witness.bin");
    b[16..48].copy_from_slice(&Fr::MODULUS.to_bytes_le());
    let err = GnarkWitness::from_bytes(&b).err().unwrap().to_string();
    assert!(err.contains("not below the scalar modulus"), "{err}");
    assert!(fr_slice(&[Fr::MODULUS.0]).is_none());
}

/// What a caller handing in gnark's solver buffers relies on: the dump's raw bytes, read as
/// limbs and reinterpreted in place, are the same `Fr` values the decoder produces.
#[test]
fn dumped_bytes_reinterpret_as_fr() {
    let w = GnarkWitness::read_dump(&toy("witness.bin")).unwrap();
    let b = bytes("witness.bin");
    let limbs = |off: usize, n: usize| -> Vec<[u64; 4]> {
        b[off..off + 32 * n]
            .as_chunks::<32>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|i| u64::from_le_bytes(c[8 * i..8 * i + 8].try_into().unwrap()))
            })
            .collect()
    };
    let n_w = w.w().len();
    let n_c = w.n_constraints();
    let w_limbs = limbs(16, n_w);
    let aw_limbs = limbs(16 + 32 * n_w + 8, n_c);
    let bw_limbs = limbs(16 + 32 * n_w + 8 + 32 * n_c + 8, n_c);
    let (wv, awv, bwv) = (
        fr_slice(&w_limbs).unwrap(),
        fr_slice(&aw_limbs).unwrap(),
        fr_slice(&bw_limbs).unwrap(),
    );
    assert_eq!(wv, w.w());
    assert_eq!(awv, w.aw());
    assert_eq!(bwv, w.bw());
    assert_eq!(wv[0], Fr::ONE);
    // The borrowed witness points at the caller's memory, not a copy of it.
    let borrowed = GnarkWitness::from_parts(wv, awv, bwv).unwrap();
    assert_eq!(borrowed.w().as_ptr(), w_limbs.as_ptr().cast());
    // And the values are the field elements, not just the same bits: the public X is 5.
    assert_eq!(wv[1] * wv[1], Fr::from(25u64));
}

/// Every query streamed through `GnarkDump::read_points` in runs that start and end
/// anywhere, across an `EXPAND_BLOCK` boundary too, decodes to exactly what `read_dump`
/// expands, with the identity as zero bytes in the flagged slots.
#[test]
fn a_streamed_dump_reads_what_read_dump_decodes() {
    let pk = GnarkProvingKey::read_dump(&toy("pk.dump")).unwrap();
    // The toy key is smaller than one `EXPAND_BLOCK`, so a small block is what makes the
    // runs cross block boundaries, partway into a block and exactly at one.
    for block in [1 << 14, 1000, 1] {
        streamed_matches(
            &pk,
            &GnarkDump::open_blocked(&toy("pk.dump"), block).unwrap(),
        );
    }
}

fn streamed_matches(pk: &GnarkProvingKey, dump: &GnarkDump) {
    use snarkrs_formats::binfile;
    assert_eq!(dump.header(), &pk.header);
    for q in Query::ALL {
        let stride = q.stride();
        let n = dump.len(q);
        let len = match q {
            Query::G1A => pk.a.len(),
            Query::G1B => pk.b_g1.len(),
            Query::G1Z => pk.z.len(),
            Query::G1K => pk.k.len(),
            Query::G2B => pk.b_g2.len(),
        };
        assert_eq!(len, n, "{q:?}");
        for run in [1, 7, 1000, n] {
            let mut bytes = Vec::new();
            let mut live = 0;
            let mut at = 0;
            while at < n {
                let m = run.min(n - at);
                let mut out = vec![0xa5u8; m * stride];
                live += dump.read_points(q, at, &mut out).unwrap();
                bytes.extend_from_slice(&out);
                at += m;
            }
            let mut want_live = 0;
            for (i, b) in bytes.chunks_exact(stride).enumerate() {
                let infinity = if q == Query::G2B {
                    let p = binfile::g2(b).unwrap();
                    assert_eq!(p, pk.b_g2[i], "{q:?} point {i}");
                    p.infinity
                } else {
                    let want = match q {
                        Query::G1A => pk.a[i],
                        Query::G1B => pk.b_g1[i],
                        Query::G1Z => pk.z[i],
                        _ => pk.k[i],
                    };
                    assert_eq!(binfile::g1(b).unwrap(), want, "{q:?} point {i}");
                    want.infinity
                };
                if infinity {
                    assert!(b.iter().all(|&v| v == 0), "{q:?} point {i}");
                } else {
                    want_live += 1;
                }
            }
            assert_eq!(live, want_live, "{q:?} run {run}");
            dump.check_not_all_infinity(q, live).unwrap();
        }
    }
    assert!(dump.check_not_all_infinity(Query::G1A, 0).is_err());
    dump.check_not_all_infinity(Query::G1K, 0).unwrap();
}

/// A run past the end, or one that is not whole points, is refused rather than read short.
#[test]
fn a_streamed_run_out_of_range_is_refused() {
    let dump = GnarkDump::open(&toy("pk.dump")).unwrap();
    let n = dump.len(Query::G1Z);
    assert!(dump.read_points(Query::G1Z, n, &mut [0u8; 64]).is_err());
    assert!(dump.read_points(Query::G1Z, 0, &mut [0u8; 65]).is_err());
    assert_eq!(dump.read_points(Query::G1Z, n, &mut []).unwrap(), 0);
}

/// The header-only key is `into_proving_key`'s with the queries left out.
#[test]
fn a_streamed_dump_gives_the_key_without_its_queries() {
    let vk = GnarkVerifyingKey::read(&toy("vk.bin")).unwrap();
    let full = GnarkProvingKey::read_dump(&toy("pk.dump"))
        .unwrap()
        .into_proving_key(&vk)
        .unwrap();
    let key = GnarkDump::open(&toy("pk.dump"))
        .unwrap()
        .proving_key(&vk)
        .unwrap();
    assert_eq!(
        (key.n_vars, key.n_public, key.domain_size, key.h_basis),
        (full.n_vars, full.n_public, full.domain_size, full.h_basis)
    );
    assert_eq!(
        (key.alpha_g1, key.beta_g1, key.delta_g1),
        (full.alpha_g1, full.beta_g1, full.delta_g1)
    );
    assert_eq!((key.beta_g2, key.delta_g2), (full.beta_g2, full.delta_g2));
    assert_eq!(key.vk.ic, full.vk.ic);
    assert!(key.a_query.is_empty() && key.h_query.is_empty() && key.l_query.is_empty());
}

/// A query point off the curve is refused by the streamed read too, by the read that
/// covers it and no other.
#[test]
fn a_streamed_off_curve_point_is_refused() {
    let mut b = bytes("pk.dump");
    let at = after_flags(&b) + 4 + 8;
    b[at] ^= 1;
    let dir = std::env::temp_dir().join(format!("snarkrs-gnark-dump-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pk.dump");
    std::fs::write(&path, &b).unwrap();
    let dump = GnarkDump::open(&path).unwrap();
    let mut out = vec![0u8; dump.len(Query::G1A) * 64];
    let err = dump
        .read_points(Query::G1A, 0, &mut out)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("G1.A") && err.contains("not on the curve"),
        "{err}"
    );
    let mut out = vec![0u8; dump.len(Query::G1B) * 64];
    dump.read_points(Query::G1B, 0, &mut out).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}
