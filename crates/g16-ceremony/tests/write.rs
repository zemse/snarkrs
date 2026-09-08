//! The writer against the reader that has to accept what it produces.
//!
//! Three levels, because each catches something the others cannot:
//!
//! 1. A synthetic file exercises every typed writer and is read back with
//!    [`g16_zkey::binfile`]. That is the only test that covers the encoders on values we
//!    chose, including the point at infinity and a section long enough to cross the
//!    staging buffer.
//! 2. Real `.zkey` and `.ptau` sections are decoded and re-encoded in place, byte against
//!    byte. Round-tripping our own output cannot catch a writer that is wrong in exactly
//!    the way the reader is wrong; snarkjs' bytes can. This is what pins the double
//!    Montgomery of zkey section 4, where an `R` instead of an `R^2` still parses.
//! 3. Whole real files are rewritten section by section through the writer and compared
//!    to the input byte for byte. That covers the framing the other two do not touch: the
//!    preamble, the entry headers, the length backfill, and the fact that snarkjs orders
//!    a zkey's sections `1, 2, 4, 3, 9, 8, 5, 6, 7, 10` rather than by id.
//!
//! Everything runs against the checked-in artifacts and skips with a message when they
//! are absent, so a fresh clone without `gen-artifacts.sh` does not report false green.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use g16_ceremony::write::{fq_lem, fr_double_montgomery, fr_plain, g1_lem, g2_lem, BinFileWriter};
use g16_ceremony::{q_le, r_le, Groth16Header, N8, SG1, SG2};
use g16_field::{AffineRepr, CurveGroup, Fq, Fr, G1Affine, G2Affine};
use g16_zkey::binfile::{self, BinFile, Cursor};

const ZKEY_MAGIC: &[u8; 4] = b"zkey";
const ZKEY_MAX_VERSION: u32 = 2;
const PTAU_MAGIC: &[u8; 4] = b"ptau";
const PTAU_MAX_VERSION: u32 = 1;

/// Bytes of one zkey section-4 record: `u32 matrix, u32 constraint, u32 signal`, then the
/// coefficient (`zkey_new.js:210-216`).
const COEF_RECORD: usize = 12 + N8;

/// Points per staging buffer inside the writer's slice and repeat helpers. Kept here so
/// the synthetic file can deliberately straddle it; it is private over there.
const POINT_BATCH: usize = 8192;

fn tmp_dir(test: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("g16-ceremony-write-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn artifacts() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/artifacts")
        .canonicalize();
    let Ok(root) = root else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file())
        .map(|d| (d.file_name().unwrap().to_string_lossy().into_owned(), d))
        .collect();
    out.sort();
    out
}

fn ptaus() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/ptau")
        .canonicalize();
    let Ok(root) = root else { return Vec::new() };
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "ptau"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p))
        .collect();
    out.sort();
    out
}

fn g1(k: u64) -> G1Affine {
    (G1Affine::generator() * Fr::from(k)).into_affine()
}

fn g2(k: u64) -> G2Affine {
    (G2Affine::generator() * Fr::from(k)).into_affine()
}

/// Rewrite every section of `src` verbatim into `dst`, keeping the declared count, the
/// version and the chain order the input had.
fn rewrite_verbatim(src: &Path, magic: &[u8; 4], max_version: u32, dst: &Path) {
    let inp = BinFile::open(src, magic, max_version).unwrap();
    let version = u32::from_le_bytes(inp.bytes()[4..8].try_into().unwrap());
    let mut w = BinFileWriter::create(dst, magic, version, inp.declared_sections() as u32).unwrap();
    for s in inp.sections() {
        w.write_section_verbatim(s.id, &inp.bytes()[s.start..s.start + s.len])
            .unwrap();
    }
    w.finish().unwrap();
}

/// Compare two files a megabyte at a time. The largest artifact is 602 MB, so holding
/// both in memory to compare them costs more than the test does.
fn assert_same_bytes(a: &Path, b: &Path, what: &str) {
    let la = std::fs::metadata(a).unwrap().len();
    let lb = std::fs::metadata(b).unwrap().len();
    assert_eq!(la, lb, "{what}: file length");

    let mut fa = BufReader::new(File::open(a).unwrap());
    let mut fb = BufReader::new(File::open(b).unwrap());
    let mut ba = vec![0u8; 1 << 20];
    let mut bb = vec![0u8; 1 << 20];
    let mut off = 0u64;
    loop {
        let n = fill(&mut fa, &mut ba);
        let m = fill(&mut fb, &mut bb);
        assert_eq!(n, m, "{what}: read {n} vs {m} bytes at offset {off}");
        if n == 0 {
            break;
        }
        if ba[..n] != bb[..n] {
            let i = ba[..n]
                .iter()
                .zip(&bb[..n])
                .position(|(x, y)| x != y)
                .expect("slices differ");
            panic!(
                "{what}: first difference at byte {}, {:#04x} vs {:#04x}",
                off + i as u64,
                ba[i],
                bb[i]
            );
        }
        off += n as u64;
    }
}

/// `Read::read` may return short of the buffer at any point, so a chunk comparison has to
/// fill deliberately rather than trust one call.
fn fill(r: &mut impl Read, buf: &mut [u8]) -> usize {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]).unwrap() {
            0 => break,
            n => got += n,
        }
    }
    got
}

#[test]
fn typed_writers_are_the_inverse_of_the_binfile_decoders() {
    let dir = tmp_dir("inverse");
    let path = dir.join("synthetic.zkey");

    // k = 0 gives the point at infinity, which is stored as all-zero bytes and is the one
    // value both point encoders special-case (`build_curve_jacobian_a0.js:50-71`).
    let g1s: Vec<G1Affine> = (0..5).map(g1).collect();
    let g2s: Vec<G2Affine> = (0..3).map(g2).collect();
    let fqs: Vec<Fq> = [0u64, 1, 7, u64::MAX]
        .iter()
        .map(|&k| Fq::from(k))
        .collect();
    let frs: Vec<Fr> = [0u64, 1, 2, 1 << 40].iter().map(|&k| Fr::from(k)).collect();
    // One past the staging buffer, so the repeat helper has to loop and the backfilled
    // length has to survive a flush that happened mid-section.
    let repeat = POINT_BATCH + 1;

    // Ids ascend nowhere and 2 appears twice: the chain is a list, not a map
    // (`binfileutils.js:26-30`), and a zkey's own order is `1, 2, 4, 3, 9, ...`.
    let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 5).unwrap();

    w.start_section(1).unwrap();
    w.write_prime(&q_le()).unwrap();
    w.write_prime(&r_le()).unwrap();
    w.write_u32(0xdead_beef).unwrap();
    w.write_u64(0x0123_4567_89ab_cdef).unwrap();
    w.write_u8(255).unwrap();
    w.end_section().unwrap();

    w.start_section(4).unwrap();
    for v in &frs {
        w.write_fr_double_montgomery(v).unwrap();
    }
    w.end_section().unwrap();

    w.start_section(3).unwrap();
    for v in &frs {
        w.write_fr_plain(v).unwrap();
    }
    for v in &fqs {
        w.write_fq(v).unwrap();
    }
    w.end_section().unwrap();

    w.start_section(2).unwrap();
    w.write_g1_slice(&g1s).unwrap();
    w.write_g2_slice(&g2s).unwrap();
    w.write_g1(&G1Affine::identity()).unwrap();
    w.write_g2(&G2Affine::identity()).unwrap();
    w.end_section().unwrap();

    w.start_section(2).unwrap();
    w.write_g1_repeated(&g1(3), repeat).unwrap();
    w.write_g2_repeated(&g2(4), 5).unwrap();
    w.end_section().unwrap();

    w.finish().unwrap();

    let f = BinFile::open(&path, ZKEY_MAGIC, ZKEY_MAX_VERSION).unwrap();
    assert_eq!(f.declared_sections(), 5);
    assert!(f.truncation().is_none(), "chain reached the end");
    let ids: Vec<u32> = f.sections().iter().map(|s| s.id).collect();
    assert_eq!(ids, vec![1, 4, 3, 2, 2], "sections keep write order");

    // The chain has to end exactly at EOF, or a section length is wrong somewhere.
    let last = f.sections().last().unwrap();
    assert_eq!(last.start + last.len, f.total_len(), "chain ends at EOF");

    let s = f.sections();

    let body = &f.bytes()[s[0].start..s[0].start + s[0].len];
    let mut cur = Cursor::new(body, 1);
    g16_ceremony::check_modulus(&mut cur, &q_le()).unwrap();
    g16_ceremony::check_modulus(&mut cur, &r_le()).unwrap();
    assert_eq!(cur.u32().unwrap(), 0xdead_beef);
    assert_eq!(cur.u64().unwrap(), 0x0123_4567_89ab_cdef);
    assert_eq!(cur.u8().unwrap(), 255);
    assert_eq!(cur.remaining(), 0, "section 1 read to the end");

    let body = &f.bytes()[s[1].start..s[1].start + s[1].len];
    assert_eq!(body.len(), frs.len() * N8);
    let r_inv = binfile::r_inv();
    for (i, want) in frs.iter().enumerate() {
        let got = binfile::fr_double_montgomery(&body[i * N8..(i + 1) * N8], &r_inv);
        assert_eq!(got, *want, "double Montgomery coefficient {i}");
    }

    let body = &f.bytes()[s[2].start..s[2].start + s[2].len];
    assert_eq!(body.len(), (frs.len() + fqs.len()) * N8);
    for (i, want) in frs.iter().enumerate() {
        let got = binfile::fr_normal(&body[i * N8..(i + 1) * N8], 3).unwrap();
        assert_eq!(got, *want, "plain scalar {i}");
    }
    let off = frs.len() * N8;
    for (i, want) in fqs.iter().enumerate() {
        let got = binfile::fq(&body[off + i * N8..off + (i + 1) * N8]);
        assert_eq!(got, *want, "base field coordinate {i}");
    }

    let body = &f.bytes()[s[3].start..s[3].start + s[3].len];
    assert_eq!(
        body.len(),
        6 * SG1 + 4 * SG2,
        "five G1 plus one, three G2 plus one"
    );
    let mut at = 0;
    for (i, want) in g1s.iter().enumerate() {
        assert_eq!(binfile::g1(&body[at..at + SG1]), *want, "g1 slice {i}");
        at += SG1;
    }
    for (i, want) in g2s.iter().enumerate() {
        assert_eq!(binfile::g2(&body[at..at + SG2]), *want, "g2 slice {i}");
        at += SG2;
    }
    assert!(
        body[at..at + SG1].iter().all(|&b| b == 0),
        "G1 infinity is all zero bytes"
    );
    assert_eq!(binfile::g1(&body[at..at + SG1]), G1Affine::identity());
    at += SG1;
    assert!(
        body[at..at + SG2].iter().all(|&b| b == 0),
        "G2 infinity is all zero bytes"
    );
    assert_eq!(binfile::g2(&body[at..at + SG2]), G2Affine::identity());
    at += SG2;
    assert_eq!(at, body.len());

    let body = &f.bytes()[s[4].start..s[4].start + s[4].len];
    assert_eq!(
        body.len(),
        repeat * SG1 + 5 * SG2,
        "a section that crossed the staging buffer got the right length backfilled"
    );
    for i in [0usize, 1, POINT_BATCH - 1, POINT_BATCH, repeat - 1] {
        assert_eq!(
            binfile::g1(&body[i * SG1..(i + 1) * SG1]),
            g1(3),
            "repeat {i}"
        );
    }
    let off = repeat * SG1;
    for i in 0..5 {
        assert_eq!(
            binfile::g2(&body[off + i * SG2..off + (i + 1) * SG2]),
            g2(4),
            "g2 repeat {i}"
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_writer_that_miscounts_or_misnests_its_sections_is_refused() {
    let dir = tmp_dir("misuse");

    let path = dir.join("short.zkey");
    let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 3).unwrap();
    w.write_section_verbatim(1, b"hello").unwrap();
    let err = w.finish().unwrap_err().to_string();
    assert!(err.contains("declared 3 sections, wrote 1"), "{err}");

    let path = dir.join("open.zkey");
    let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 1).unwrap();
    w.start_section(1).unwrap();
    assert!(w.start_section(2).is_err(), "two sections open at once");
    w.end_section().unwrap();
    assert!(w.end_section().is_err(), "end without start");
    // A payload byte outside a section would be read back as the next entry header.
    assert!(w.write_u32(7).is_err(), "write outside a section");
    w.finish().unwrap();

    let path = dir.join("unclosed.zkey");
    let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 1).unwrap();
    w.start_section(1).unwrap();
    w.write_u32(7).unwrap();
    assert!(w.finish().is_err(), "finish with a section still open");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn section_len_counts_the_open_payload_only() {
    let dir = tmp_dir("seclen");
    let path = dir.join("len.zkey");

    let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 2).unwrap();
    w.start_section(1).unwrap();
    assert_eq!(w.section_len(), 0);
    w.write_g1(&g1(1)).unwrap();
    assert_eq!(w.section_len(), SG1 as u64);
    w.write_u32(0).unwrap();
    assert_eq!(w.section_len(), SG1 as u64 + 4);
    w.end_section().unwrap();
    // `paramLength` is emitted from this counter mid-section, so it must restart at zero
    // rather than accumulate across the file.
    w.start_section(2).unwrap();
    assert_eq!(w.section_len(), 0);
    w.end_section().unwrap();
    w.finish().unwrap();

    let f = BinFile::open(&path, ZKEY_MAGIC, ZKEY_MAX_VERSION).unwrap();
    assert_eq!(f.sections()[0].len, SG1 + 4);
    assert_eq!(f.sections()[1].len, 0);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_encoders_reproduce_snarkjs_bytes_on_a_real_zkey() {
    let found = artifacts();
    if found.is_empty() {
        eprintln!("SKIPPED the_encoders_reproduce_snarkjs_bytes_on_a_real_zkey: no artifacts");
        return;
    }
    for (name, dir) in found {
        let f = BinFile::open(&dir.join("circuit.zkey"), ZKEY_MAGIC, ZKEY_MAX_VERSION).unwrap();

        // Sections 3, 5, 6, 8 and 9 are flat runs of G1; 7 is a flat run of G2
        // (`zkey_new.js:76-168`). Decoding and re-encoding each point in place is the
        // check a round trip of our own output cannot make: these bytes are snarkjs'.
        let mut n_g1 = 0usize;
        for id in [3u32, 5, 6, 8, 9] {
            let body = f.unique_section(id).unwrap();
            assert_eq!(
                body.len() % SG1,
                0,
                "{name}: section {id} is not whole points"
            );
            for (i, chunk) in body.chunks(SG1).enumerate() {
                let p = binfile::g1(chunk);
                assert_eq!(&g1_lem(&p)[..], chunk, "{name}: section {id} point {i}");
            }
            n_g1 += body.len() / SG1;
        }
        let body = f.unique_section(7).unwrap();
        assert_eq!(body.len() % SG2, 0, "{name}: section 7 is not whole points");
        for (i, chunk) in body.chunks(SG2).enumerate() {
            let p = binfile::g2(chunk);
            assert_eq!(&g2_lem(&p)[..], chunk, "{name}: section 7 point {i}");
        }
        let n_g2 = body.len() / SG2;

        // Section 4: `u32 nCoefs` then 44-byte records whose tail is the coefficient in
        // DOUBLE Montgomery. Writing `v*R` instead of `v*R^2` here produces a file that
        // parses and proves nothing, so this is the assertion that matters most.
        let body = f.unique_section(4).unwrap();
        let n_coefs = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
        assert_eq!(
            body.len(),
            4 + n_coefs * COEF_RECORD,
            "{name}: section 4 size"
        );
        let r_inv = binfile::r_inv();
        for i in 0..n_coefs {
            let at = 4 + i * COEF_RECORD + 12;
            let raw = &body[at..at + N8];
            let v = binfile::fr_double_montgomery(raw, &r_inv);
            assert_eq!(
                &fr_double_montgomery(&v)[..],
                raw,
                "{name}: coefficient {i}"
            );
        }

        // Section 1 stores the protocol id as a plain u32, and section 2 is the one place
        // `write_prime` and the point writers are driven by `Groth16Header::write`.
        let body = f.unique_section(2).unwrap();
        let hdr = Groth16Header::read(body).unwrap();
        assert_eq!(body.len(), Groth16Header::BYTES, "{name}: header size");
        let out = tmp_dir(&format!("hdr-{name}"));
        let path = out.join("header.zkey");
        let mut w = BinFileWriter::create(&path, ZKEY_MAGIC, 1, 1).unwrap();
        w.start_section(2).unwrap();
        hdr.write(&mut w).unwrap();
        w.end_section().unwrap();
        w.finish().unwrap();
        let re = BinFile::open(&path, ZKEY_MAGIC, ZKEY_MAX_VERSION).unwrap();
        assert_eq!(
            re.unique_section(2).unwrap(),
            body,
            "{name}: groth16 header re-encodes to the same bytes"
        );
        std::fs::remove_dir_all(&out).unwrap();

        // Every coordinate in the file went through `fq_lem` above; spot-check the scalar
        // path once more on a value that is not a coordinate.
        assert_eq!(fq_lem(&hdr.alpha_g1.x)[..], g1_lem(&hdr.alpha_g1)[..N8]);
        assert_eq!(
            fr_plain(&Fr::from(1u64))[0],
            1,
            "plain one is a literal one"
        );

        eprintln!(
            "{name}: {n_g1} G1, {n_g2} G2 and {n_coefs} coefficients re-encode byte-identically"
        );
    }
}

#[test]
fn a_real_zkey_rewritten_through_the_writer_is_byte_identical() {
    let found = artifacts();
    if found.is_empty() {
        eprintln!(
            "SKIPPED a_real_zkey_rewritten_through_the_writer_is_byte_identical: no artifacts"
        );
        return;
    }
    let dir = tmp_dir("zkey-verbatim");
    for (name, src) in found {
        let src = src.join("circuit.zkey");
        let dst = dir.join("out.zkey");
        rewrite_verbatim(&src, ZKEY_MAGIC, ZKEY_MAX_VERSION, &dst);
        assert_same_bytes(&src, &dst, &name);
        let bytes = std::fs::metadata(&dst).unwrap().len();
        std::fs::remove_file(&dst).unwrap();
        eprintln!("{name}: {bytes} bytes rewritten byte-identically");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_real_ptau_rewritten_through_the_writer_is_byte_identical() {
    let found = ptaus();
    if found.is_empty() {
        eprintln!(
            "SKIPPED a_real_ptau_rewritten_through_the_writer_is_byte_identical: no bench/ptau"
        );
        return;
    }
    let dir = tmp_dir("ptau-verbatim");
    // Only the smallest, because these run to gigabytes and the framing under test does
    // not vary with `power`; the encoders are covered on the zkeys.
    let (name, src) = found
        .iter()
        .min_by_key(|(_, p)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX))
        .unwrap();
    let dst = dir.join("out.ptau");
    rewrite_verbatim(src, PTAU_MAGIC, PTAU_MAX_VERSION, &dst);
    assert_same_bytes(src, &dst, name);
    eprintln!(
        "{name}: {} bytes rewritten byte-identically",
        std::fs::metadata(&dst).unwrap().len()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
