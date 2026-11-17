//! `groth16 setup`: an `.r1cs` and a prepared `.ptau` in, an initial `.zkey` out.
//!
//! `zkey_new.js:36-171`. Four things about this command surprise people who have only
//! read the paper:
//!
//! * **The initial key has `gamma = delta = 1` and setup performs no division.**
//!   `zkey_new.js:127-129` writes the plain generators as `gamma2`, `delta1` and `delta2`,
//!   so sections 3 and 8 are undivided. `zkey contribute` is what introduces a delta at
//!   all (`zkey_contribute.js:63-64, :90-92`).
//! * **This is not one big MSM, and it is not even MSM-bound.** `composeAndWritePoints`
//!   produces one output point per signal, dispatching a single scalar multiplication when
//!   a signal appears once and a multiexp when it appears more than once, then one
//!   batch-to-affine over the chunk (`zkey_new.js:459-491`). The shape is per-signal and
//!   the slots are tiny, so most of the arithmetic never reaches a multiexp at all. See
//!   [`setup`] for the measured split and what it means for a device.
//! * **Section 9 involves no arithmetic at all.** `writeHs` is a strided gather: read the
//!   `2*domainSize` Lagrange block of ptau section 12 and take every odd element
//!   (`zkey_new.js:186-191`). The `cirPower == Fr::TWO_ADICITY` case is different because
//!   that block came out of the coset branch and is laid out as two contiguous halves
//!   rather than interleaved, so it reads the second half in one go (`:192-194`).
//! * **The section write order is `1, 2, 4, 3, 9, 8, 5, 6, 7, 10`**, not id order
//!   (`zkey_new.js:76-168`). It changes both the file layout and `csHash`, so getting it
//!   wrong is not cosmetic.
//!
//! The one shipped bug that has to be reproduced rather than fixed lives in
//! [`hash_h_points`].

use std::ops::Range;
use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ark_ec::short_weierstrass::{Affine, Projective};
use rayon::prelude::*;
use snarkrs_field::raw::RawField;
use snarkrs_field::{
    AffineRepr, CurveGroup, FftField, Fq, Fr, G1Affine, G1Projective, G2Affine, One, PrimeField,
    Zero,
};
use snarkrs_formats::binfile;
use snarkrs_msm::glv::{self, GlvField};
use snarkrs_msm::xyzz::{to_projective, RawCurve, Xyzz};
use snarkrs_msm::{AccelError, MsmBackend};

use crate::ptau::{self, Ptau};
use crate::r1cs::{Matrix, R1cs, S_CONSTRAINTS, TERM_BYTES};
use crate::transcript::{self, Digest, Transcript};
use crate::write::BinFileWriter;
use crate::{CeremonyError, Groth16Header, N8, SG1, SG2};

/// zkey section ids, in id order rather than write order.
pub const S_PROTOCOL: u32 = 1;
pub const S_HEADER: u32 = 2;
pub const S_IC: u32 = 3;
pub const S_COEFFS: u32 = 4;
pub const S_A: u32 = 5;
pub const S_B1: u32 = 6;
pub const S_B2: u32 = 7;
pub const S_C: u32 = 8;
pub const S_H: u32 = 9;
pub const S_MPC_PARAMS: u32 = 10;

/// Sections a fresh zkey holds, and the number `createBinFile` declares
/// (`zkey_new.js:49`).
pub const ZKEY_SECTIONS: u32 = 10;

/// `.zkey` magic, and the version this crate writes.
pub const ZKEY_MAGIC: &[u8; 4] = b"zkey";
pub const ZKEY_VERSION: u32 = 1;
/// Both readers in `zkey_verify_frominit.js` pass a max version of 2 while every writer
/// emits 1, so 1 and 2 are both acceptable on read.
pub const ZKEY_MAX_VERSION: u32 = 2;

/// Section 4 record stride on BN254: three u32 indices then one 32-byte scalar.
pub const COEF_RECORD_BYTES: usize = 12 + crate::N8;

/// Section 4 records encoded per `write_bytes` call. Only a buffering choice: the bytes
/// are identical whatever the split, and 65536 records is 2.9 MB.
const COEF_BATCH: usize = 1 << 16;

/// Points per `normalize_batch` call in [`to_affine`]: one field inversion each, so the
/// split only trades inversions against scratch, and the affine output is the same.
const NORMALIZE_CHUNK: usize = 4096;

/// Points per `hashHPoints` chunk, snarkjs' `CHUNK_SIZE` at `zkey_new.js:505`. Unlike the
/// coefficient batch this one IS load bearing, because the missing `- i` at `:511` makes
/// the number of points hashed a function of it.
const H_CHUNK: usize = 1 << 14;

/// Terms in one accumulator slot below which a Pippenger multiexp is not worth its bucket
/// array, and the slot is summed by batched ladders instead (see [`compose`]).
///
/// snarkjs splits at 2 (`zkey_new.js:467-481`), but only because its alternative is a
/// round trip into a wasm worker either way. The sum is the same however it is computed,
/// and slots here are tiny: on `anon-aadhaar` the mean A slot holds three terms, so
/// dispatching a full MSM per slot would be several million bucket allocations.
///
/// 256 is tuned for [`snarkrs_msm::CpuMsm`] against the batched ladders, where a term costs
/// its additions and a share of doublings, about what a Pippenger over a slot of a few
/// hundred spends per term. On `js_16x16_d32`, whose C section has 21,897 slots of 32 or
/// more (55 terms on average), `compose_c` measured 2.14 s at 32, 1.81 at 128, 1.78 at 256
/// and 1.77 at 512. A backend whose dispatch costs more than a few hundred terms of
/// arithmetic wants a higher threshold still, and the trait has no way to say so. See
/// [`setup`].
const MULTIEXP_MIN_TERMS: usize = 256;

/// What a setup produced, for the CLI to print and a test to assert on.
#[derive(Clone, Debug)]
pub struct SetupReport {
    pub n_vars: usize,
    /// `nOutputs + nPubInputs`, the ONE signal excluded.
    pub n_public: usize,
    pub n_constraints: usize,
    pub domain_size: usize,
    pub cir_power: u32,
    /// Section 4's record count.
    pub n_coefs: usize,
    /// The circuit hash written into section 10, and the value every later contribution
    /// transcript is seeded with.
    pub cs_hash: Digest,
    /// Wall time per stage, in run order.
    pub phases: Vec<(&'static str, Duration)>,
}

/// `zkey_new.js:59` with the floor-log2 of `misc.js:53`:
/// `floor(log2(nConstraints + nPubInputs + nOutputs)) + 1`.
///
/// Note the consequence, which is not an off-by-one: for a total that is already a power
/// of two the domain still doubles, so `2^cirPower` is always strictly greater than
/// `nConstraints + nPublic`. That is precisely what makes the synthetic `nConstraints + s`
/// row indices land inside the `domainSize`-element Lagrange window.
pub fn circuit_power(n_constraints: usize, n_public: usize) -> u32 {
    crate::floor_log2((n_constraints + n_public) as u64) + 1
}

/// Run the whole setup.
///
/// # `msm` accelerates a minority of this command, and on a device it costs more than it saves
///
/// The five [`compose_points_g1`] and [`compose_points_g2`] calls are 97.5% to 98.7% of the
/// run: the r1cs parse, the ptau read, the section-4 encode, the H gather and every write
/// together are under 2.5%. So the command is compute bound. What it is *not* is multiexp
/// bound, because [`MULTIEXP_MIN_TERMS`] sends almost everything to the loop below it:
///
/// | circuit | domain | terms in slots >= 32 | share of compose the MSM path holds |
/// |---|---|---:|---:|
/// | `prod_large_O1` | 2^20 | 4.8% | 0.50% |
/// | `prod_small_O1` | 2^17 | 3.8% | 0.25% |
/// | `prod_small_O2` | 2^16 | 75.8% | 45.1% |
/// | `prod_large_O2` | 2^18 | 74.2% | 42.9% |
///
/// The split inverts with circom's `-O` level, and `--O1` is the default, so on the output
/// circom produces with no flags an MSM that ran in zero time takes the 2^20 circuit from
/// 15.12 s to 15.04 s. What holds the rest is millions of scalar multiplications of one to
/// thirty terms each, which is a different kernel and would need a different seam: N terms
/// in, N products out, reduced per slot, rather than one multiexp per output point. (The
/// table and timings here are from when the threshold was 32 and the loop multiplied one
/// term at a time; [`compose`] is now that seam, and the threshold is 256.)
///
/// A device behind *this* parameter loses, and the reason is the per-slot dispatch. A
/// `--backend metal` run measures 1.15 ms per `msm_g1` call against 0.149 ms for a bare
/// Metal commit-and-wait, so the 97,648 slots of `prod_large_O2` are 112 s
/// of round trip on a command the CPU finishes in 46.8 s. Measured end to end on an M2 Max,
/// `--backend metal` against `--backend cpu`: 16.09 s vs 16.67 s at 2^20 `--O1`, 2.94 s vs
/// 3.00 s at 2^17 `--O1`, 28.2 s vs 9.5 s at 2^16 `--O2`, 144.5 s vs 46.8 s at 2^18 `--O2`.
/// Every one of those pairs is byte-identical, which is the property that matters here; the
/// numbers are why the backend that produced them is not the default.
pub fn setup(
    r1cs_path: &Path,
    ptau_path: &Path,
    out_path: &Path,
    msm: &dyn MsmBackend,
) -> Result<SetupReport, CeremonyError> {
    crate::write::staged(out_path, |stage| {
        let setup = Prepared::open(r1cs_path, ptau_path)?;
        let mut file = FileSink::create(stage)?;
        setup.run(&mut file, msm)
    })
}

/// A setup whose inputs are open and checked, and whose header is known, before any
/// section is produced.
pub(crate) struct Prepared {
    ptau: Ptau,
    r1cs: R1cs,
    header: Groth16Header,
}

impl Prepared {
    pub(crate) fn open(r1cs_path: &Path, ptau_path: &Path) -> Result<Self, CeremonyError> {
        let ptau = Ptau::open(ptau_path)?;
        let r1cs = R1cs::open(r1cs_path)?;
        let cir_power = r1cs.header().cir_power();

        // Order matters only for the message a caller sees: snarkjs tests the power first
        // (`zkey_new.js:61`) and only then section 12 (`:66`).
        if cir_power > ptau.power() {
            return Err(CeremonyError::PtauTooSmall {
                needed: cir_power,
                have: ptau.power(),
            });
        }
        if !ptau.is_prepared() {
            return Err(CeremonyError::NotPrepared);
        }

        // `vk_alpha_1` and `vk_beta_1` are element 0 of the alpha and beta ptau sections,
        // that is `alpha*tau^0*G1` and `beta*tau^0*G1` (`zkey_new.js:101,107,113`). Reading
        // them from the Lagrange sections instead would give `alpha*L_0(tau)*G1`, which for
        // a one-element domain is the same point and for every real domain is not.
        let alpha_g1 = ptau.g1_points(ptau::S_ALPHA_TAU_G1, 0, 1)?[0];
        let beta_g1 = ptau.g1_points(ptau::S_BETA_TAU_G1, 0, 1)?[0];
        let beta_g2 = ptau.g2_points(ptau::S_BETA_G2, 0, 1)?[0];

        let header = Groth16Header {
            n_vars: r1cs.header().n_vars,
            n_public: r1cs.header().n_public() as u32,
            domain_size: r1cs.header().domain_size() as u32,
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2: G2Affine::generator(),
            delta_g1: G1Affine::generator(),
            delta_g2: G2Affine::generator(),
        };
        Ok(Self { ptau, r1cs, header })
    }

    /// Section 2 of the key [`Prepared::run`] produces.
    pub(crate) fn header(&self) -> &Groth16Header {
        &self.header
    }

    /// Produce every section into `sink`, in write order.
    pub(crate) fn run<S: SetupSink>(
        &self,
        sink: &mut S,
        msm: &dyn MsmBackend,
    ) -> Result<SetupReport, CeremonyError> {
        let Self { ptau, r1cs, header } = self;
        let n_vars = r1cs.header().n_vars as usize;
        let n_public = r1cs.header().n_public();
        let n_constraints = r1cs.header().n_constraints as usize;
        let cir_power = r1cs.header().cir_power();
        let domain_size = r1cs.header().domain_size();
        let mut phases = Phases::start();

        sink.header(header)?;
        // The six header points, uncompressed, in write order (`zkey_new.js:104,110,116,
        // :130-132`). The scalars beside them on disk are never hashed.
        let mut header_hash = Transcript::new();
        header_hash.update_g1(&header.alpha_g1);
        header_hash.update_g1(&header.beta_g1);
        header_hash.update_g2(&header.beta_g2);
        header_hash.update_g2(&header.gamma_g2);
        header_hash.update_g1(&header.delta_g1);
        header_hash.update_g2(&header.delta_g2);
        let cs = CsHash::spawn(header_hash)?;

        // The r1cs walks are serial and the Lagrange check is not, so they run side by
        // side; an r1cs error still wins, as it would have run first.
        let (acc, bases) = rayon::join(
            || -> Result<Accumulators, CeremonyError> {
                let counts = TermCounts::walk(r1cs, domain_size)?;
                let n = coef_count(counts.n_coefs)?;
                sink.start(S_COEFFS)?;
                sink.bytes(&n.to_le_bytes())?;
                let acc =
                    counts.fill(r1cs, |batch| sink.bytes(&encode_coefficients(r1cs, batch)?))?;
                sink.end()?;
                Ok(acc)
            },
            || LagrangeBlocks::read(ptau, domain_size),
        );
        let acc = acc?;
        let bases = bases?;
        phases.lap("constraints");

        let ic = acc.ic();
        point_section(
            sink,
            &cs,
            S_IC,
            ic.len(),
            |r| compose_points_g1(ic.sub(r), &bases, r1cs, msm),
            emit_g1,
        )?;
        phases.lap("IC");

        if sink.wants(S_H) {
            sink.start(S_H)?;
            h_points(ptau, domain_size, |points| sink.g1(points))?;
            sink.end()?;
            phases.lap("H");
        }
        hash_h_points_into(ptau, domain_size, |b| cs.send(b))?;
        phases.lap("H hash");

        for (id, slots, name) in [
            (S_C, acc.c(), "C"),
            (S_A, acc.a.all(), "A"),
            (S_B1, acc.b.all(), "B1"),
        ] {
            point_section(
                sink,
                &cs,
                id,
                slots.len(),
                |r| compose_points_g1(slots.sub(r), &bases, r1cs, msm),
                emit_g1,
            )?;
            phases.lap(name);
        }
        let b2 = acc.b.all();
        point_section(
            sink,
            &cs,
            S_B2,
            b2.len(),
            |r| compose_points_g2(b2.sub(r), &bases, r1cs, msm),
            emit_g2,
        )?;
        phases.lap("B2");

        let cs_hash = cs.finish().finalize();
        phases.lap("csHash");
        sink.finish(&cs_hash)?;

        Ok(SetupReport {
            n_vars,
            n_public,
            n_constraints,
            domain_size,
            cir_power,
            n_coefs: acc.n_coefs,
            cs_hash,
            phases: phases.laps,
        })
    }
}

/// Wall time per stage of a run, for `-v`.
struct Phases {
    at: Instant,
    laps: Vec<(&'static str, Duration)>,
}

impl Phases {
    fn start() -> Self {
        Self {
            at: Instant::now(),
            laps: Vec::new(),
        }
    }

    fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        self.laps.push((name, now - self.at));
        self.at = now;
    }
}

/// Where [`Prepared::run`]'s sections go: a file for `groth16 setup`, or a comparison
/// against a contributed key for `zkey verify` (`crate::contribute`).
///
/// Sections arrive in write order, each as [`SetupSink::start`], its body in any number of
/// pieces, then [`SetupSink::end`]. A point section arrives [`SECTION_CHUNK`] points at a
/// time, so neither side ever holds a whole one.
pub(crate) trait SetupSink: Send {
    /// False for a section whose body the sink ignores, so it is never produced. Only
    /// section 9 is optional, because it alone is not hashed.
    fn wants(&self, _id: u32) -> bool {
        true
    }
    /// Sections 1 and 2.
    fn header(&mut self, header: &Groth16Header) -> Result<(), CeremonyError>;
    fn start(&mut self, id: u32) -> Result<(), CeremonyError>;
    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CeremonyError>;
    fn g1(&mut self, points: &[G1Affine]) -> Result<(), CeremonyError>;
    fn g2(&mut self, points: &[G2Affine]) -> Result<(), CeremonyError>;
    fn end(&mut self) -> Result<(), CeremonyError>;
    /// Section 10, once the hash is known.
    fn finish(&mut self, cs_hash: &Digest) -> Result<(), CeremonyError>;
}

/// The sink `groth16 setup` writes through.
struct FileSink(Option<BinFileWriter>);

impl FileSink {
    fn create(path: &Path) -> Result<Self, CeremonyError> {
        Ok(Self(Some(BinFileWriter::create(
            path,
            ZKEY_MAGIC,
            ZKEY_VERSION,
            ZKEY_SECTIONS,
        )?)))
    }

    fn w(&mut self) -> Result<&mut BinFileWriter, CeremonyError> {
        self.0
            .as_mut()
            .ok_or_else(|| CeremonyError::malformed(S_MPC_PARAMS, "written after section 10"))
    }
}

impl SetupSink for FileSink {
    fn header(&mut self, header: &Groth16Header) -> Result<(), CeremonyError> {
        let w = self.w()?;
        w.start_section(S_PROTOCOL)?;
        w.write_u32(crate::PROTOCOL_GROTH16)?;
        w.end_section()?;
        w.start_section(S_HEADER)?;
        header.write(w)?;
        w.end_section()
    }

    fn start(&mut self, id: u32) -> Result<(), CeremonyError> {
        self.w()?.start_section(id)
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), CeremonyError> {
        self.w()?.write_bytes(bytes)
    }

    fn g1(&mut self, points: &[G1Affine]) -> Result<(), CeremonyError> {
        self.w()?.write_g1_slice(points)
    }

    fn g2(&mut self, points: &[G2Affine]) -> Result<(), CeremonyError> {
        self.w()?.write_g2_slice(points)
    }

    fn end(&mut self) -> Result<(), CeremonyError> {
        self.w()?.end_section()
    }

    fn finish(&mut self, cs_hash: &Digest) -> Result<(), CeremonyError> {
        let mut w = self
            .0
            .take()
            .ok_or_else(|| CeremonyError::malformed(S_MPC_PARAMS, "section 10 written twice"))?;
        w.start_section(S_MPC_PARAMS)?;
        w.write_bytes(cs_hash)?;
        w.write_u32(0)?;
        w.end_section()?;
        w.finish()
    }
}

/// Output points per chunk of a point section.
///
/// Each chunk is composed while the one before it goes to the sink and the hasher, so a
/// section is never whole in memory: the working set is a few chunks, about 170 MB each in
/// G1 and 330 MB in G2, wherever the circuit's size. Only a buffering choice: the bytes and
/// the hash are the same whatever the split.
const SECTION_CHUNK: usize = 1 << 20;

/// One point section: hash the slot count, then compose, emit and hash each chunk.
///
/// snarkjs interleaves the writing and the hashing chunk by chunk (`zkey_new.js:369-377`)
/// but nothing else touches the hasher in between, so the byte stream is the same.
fn point_section<T, S, F>(
    sink: &mut S,
    cs: &CsHash,
    id: u32,
    n: usize,
    compose: F,
    emit: fn(&mut S, &CsHash, Vec<T>) -> Result<(), CeremonyError>,
) -> Result<(), CeremonyError>
where
    T: Send,
    S: SetupSink,
    F: Fn(Range<usize>) -> Result<Vec<T>, CeremonyError> + Sync,
{
    cs.send((n as u32).to_be_bytes().to_vec());
    sink.start(id)?;
    let mut done: Option<Vec<T>> = None;
    let mut from = 0;
    while from < n {
        let to = (from + SECTION_CHUNK).min(n);
        let (next, emitted) = rayon::join(
            || compose(from..to),
            || done.take().map_or(Ok(()), |points| emit(sink, cs, points)),
        );
        // A serial run would have emitted this chunk before composing the next one.
        emitted?;
        done = Some(next?);
        from = to;
    }
    if let Some(points) = done {
        emit(sink, cs, points)?;
    }
    sink.end()
}

/// A G1 chunk to the sink, then to the hasher.
fn emit_g1<S: SetupSink>(
    sink: &mut S,
    cs: &CsHash,
    points: Vec<G1Affine>,
) -> Result<(), CeremonyError> {
    sink.g1(&points)?;
    cs.send(transcript::g1_uncompressed_all(&points));
    Ok(())
}

/// [`emit_g1`] for section 7.
fn emit_g2<S: SetupSink>(
    sink: &mut S,
    cs: &CsHash,
    points: Vec<G2Affine>,
) -> Result<(), CeremonyError> {
    sink.g2(&points)?;
    cs.send(transcript::g2_uncompressed_all(&points));
    Ok(())
}

/// Which of the four ptau Lagrange sections a term's base comes from
/// (`zkey_new.js:38-41`, dispatched through `sBuffs` at `:415-420`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BaseTag {
    /// Section 12, `L_j(tau) * G1`.
    TauG1,
    /// Section 13, `L_j(tau) * G2`.
    TauG2,
    /// Section 14, `alpha * L_j(tau) * G1`.
    AlphaTauG1,
    /// Section 15, `beta * L_j(tau) * G1`.
    BetaTauG1,
}

/// One term of one output point: which Lagrange buffer, which element of it, and where
/// the scalar lives.
///
/// `coef_ptr` of `None` is snarkjs' `-1` sentinel and means "the scalar is 1"
/// (`zkey_new.js:436-446`). It appears only in the public-signal tail, which appends a
/// synthetic constraint row per public signal with `a = 1, b = 0, c = 0` so that `A_s(x)`
/// is nonzero for every public signal.
///
/// Packed into 12 bytes, because a 2^25 circuit holds a quarter of a billion of these at
/// once: the tag rides in the top two bits of the element index, which
/// [`process_constraints`] keeps below `2^30`, and `None` is `u64::MAX`, which no offset
/// into a section can be.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PointTerm {
    word: u32,
    coef: [u32; 2],
}

/// Bits of [`PointTerm`]'s element index; the tag takes the rest of the word.
const INDEX_BITS: u32 = 30;
const NO_COEF: u64 = u64::MAX;

impl PointTerm {
    const EMPTY: Self = Self {
        word: 0,
        coef: [0; 2],
    };

    pub fn new(tag: BaseTag, index: u32, coef_ptr: Option<u64>) -> Self {
        debug_assert!(index < 1 << INDEX_BITS, "element {index} does not pack");
        let tag = match tag {
            BaseTag::TauG1 => 0,
            BaseTag::TauG2 => 1,
            BaseTag::AlphaTauG1 => 2,
            BaseTag::BetaTauG1 => 3,
        };
        let ptr = coef_ptr.unwrap_or(NO_COEF);
        Self {
            word: tag << INDEX_BITS | index,
            coef: [ptr as u32, (ptr >> 32) as u32],
        }
    }

    pub fn tag(&self) -> BaseTag {
        match self.word >> INDEX_BITS {
            0 => BaseTag::TauG1,
            1 => BaseTag::TauG2,
            2 => BaseTag::AlphaTauG1,
            _ => BaseTag::BetaTauG1,
        }
    }

    /// Element index into the tagged Lagrange block, which is the constraint index.
    pub fn index(&self) -> u32 {
        self.word & ((1 << INDEX_BITS) - 1)
    }

    pub fn coef_ptr(&self) -> Option<u64> {
        let ptr = u64::from(self.coef[0]) | u64::from(self.coef[1]) << 32;
        (ptr != NO_COEF).then_some(ptr)
    }
}

impl std::fmt::Debug for PointTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointTerm")
            .field("tag", &self.tag())
            .field("index", &self.index())
            .field("coef_ptr", &self.coef_ptr())
            .finish()
    }
}

/// One list of terms per output point, flattened: slot `i` is
/// `terms[offsets[i] .. offsets[i + 1]]`.
///
/// Built by a counting walk and a filling walk over the r1cs rather than by pushing into a
/// vector per slot, which on a 2^25 circuit was 74 million vector headers and their
/// doubling slack. Offsets count terms, so a u32 holds every term of an r1cs up to about
/// 144 GiB, and [`process_constraints`] refuses a larger one before counting.
pub struct Slots {
    offsets: Vec<u32>,
    terms: Vec<PointTerm>,
}

impl Slots {
    /// One slot per list, in order.
    pub fn from_lists(lists: &[Vec<PointTerm>]) -> Self {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        offsets.push(0u32);
        let mut terms = Vec::new();
        for list in lists {
            terms.extend_from_slice(list);
            offsets.push(u32::try_from(terms.len()).expect("terms fit a u32 offset"));
        }
        Self { offsets, terms }
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slot(&self, i: usize) -> &[PointTerm] {
        &self.terms[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    pub fn all(&self) -> SlotRange<'_> {
        self.range(0..self.len())
    }

    /// # Panics
    ///
    /// Panics unless `range` lies within `0..len()`.
    pub fn range(&self, range: Range<usize>) -> SlotRange<'_> {
        assert!(range.start <= range.end && range.end <= self.len());
        SlotRange {
            slots: self,
            start: range.start,
            end: range.end,
        }
    }
}

/// A run of consecutive slots of one [`Slots`], numbered from zero.
#[derive(Clone, Copy)]
pub struct SlotRange<'a> {
    slots: &'a Slots,
    start: usize,
    end: usize,
}

impl<'a> SlotRange<'a> {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn slot(&self, k: usize) -> &'a [PointTerm] {
        assert!(k < self.len());
        self.slots.slot(self.start + k)
    }

    /// The slots `range` of this run, numbered from zero again.
    pub fn sub(&self, range: Range<usize>) -> SlotRange<'a> {
        assert!(range.start <= range.end && range.end <= self.len());
        SlotRange {
            slots: self.slots,
            start: self.start + range.start,
            end: self.start + range.end,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a [PointTerm]> + 'a {
        let (slots, range) = (self.slots, self.start..self.end);
        range.map(move |i| slots.slot(i))
    }
}

/// The accumulator lists `processConstraints` builds, before any point is computed.
///
/// Lengths are fixed: `A`, `B1` and `B2` are `nVars` long, `C` is `nVars - nPublic - 1`,
/// and `IC` is `nPublic + 1`. That matters beyond bookkeeping because each length is
/// hashed into `csHash` as a big-endian u32 before its points. A signal that appears in no
/// constraint leaves an empty slot, and an empty slot writes the group identity, which
/// serialises as all-zero bytes.
///
/// Section 4's records are not kept: they are handed out in push order while the lists
/// fill (see [`TermCounts::fill`]), and only their count stays.
pub struct Accumulators {
    /// Section 5, one slot per signal.
    pub a: Slots,
    /// Sections 6 and 7, one slot per signal. They differ only in the group each term's
    /// base comes from, `L_j(tau)` in G1 for B1 and in G2 for B2, so the list is held once
    /// and tagged for B1.
    pub b: Slots,
    /// Sections 3 and 8, one slot per signal: [`Accumulators::ic`] is signals
    /// `0 ..= nPublic` and [`Accumulators::c`] the rest. They partition the signals, which
    /// is why one list covers both.
    pub ic_c: Slots,
    /// Section 4's record count, `nnz(A) + nnz(B) + nPublic + 1`.
    pub n_coefs: usize,
    n_public: usize,
}

impl Accumulators {
    pub fn ic(&self) -> SlotRange<'_> {
        self.ic_c.range(0..self.n_public + 1)
    }

    pub fn c(&self) -> SlotRange<'_> {
        self.ic_c.range(self.n_public + 1..self.ic_c.len())
    }
}

/// One section-4 record.
#[derive(Clone, Copy, Debug)]
pub struct CoefRecord {
    /// 0 for A, 1 for B. Never 2.
    pub matrix: u32,
    pub constraint: u32,
    pub signal: u32,
    /// `None` is the scalar 1, as in [`PointTerm`].
    pub coef_ptr: Option<u64>,
}

/// Walk the r1cs and build every accumulator, in the three-pass order
/// `zkey_new.js:222-300` uses: A terms, then B terms, then C terms per constraint, then
/// the public-signal tail.
pub fn process_constraints(r1cs: &R1cs, domain_size: usize) -> Result<Accumulators, CeremonyError> {
    TermCounts::walk(r1cs, domain_size)?.fill(r1cs, |_| Ok(()))
}

/// The first of [`process_constraints`]' two walks: how many terms each slot will hold.
///
/// Every structural error in the r1cs surfaces here, so the second walk, which reads the
/// same bytes, cannot fail.
struct TermCounts {
    a: Vec<u32>,
    b: Vec<u32>,
    ic_c: Vec<u32>,
    n_coefs: usize,
    n_public: usize,
    n_constraints: usize,
}

impl TermCounts {
    fn walk(r1cs: &R1cs, domain_size: usize) -> Result<Self, CeremonyError> {
        let n_vars = r1cs.header().n_vars as usize;
        let n_public = r1cs.header().n_public();
        let n_constraints = r1cs.header().n_constraints as usize;

        // Signal 0 is the constant ONE and signals `1 ..= nPublic` are the public ones, so
        // a circuit with no private signal at all still needs `nVars > nPublic`. snarkjs
        // would instead build a `BigArray` of negative length and only fail later, in the
        // C write.
        if n_vars <= n_public {
            return Err(CeremonyError::BadParams(format!(
                "{n_vars} signals cannot hold the ONE signal plus {n_public} public ones"
            )));
        }
        // The tail rows sit at Lagrange element `nConstraints + s`, and the largest of them
        // has to stay inside the `domainSize`-element block. `circuit_power` guarantees it;
        // this catches a caller that computed `domain_size` some other way, before the bound
        // turns into an out-of-range base index a few million points later.
        if n_constraints + n_public >= domain_size {
            return Err(CeremonyError::BadParams(format!(
                "domain size {domain_size} does not cover {n_constraints} constraints plus \
                 {n_public} public signals"
            )));
        }
        // Every element index is below `domain_size`, and no ptau goes past 2^28.
        if domain_size > 1 << INDEX_BITS {
            return Err(CeremonyError::CircuitTooBig(crate::floor_log2(
                domain_size as u64,
            )));
        }
        // The IC and C list holds every term plus the tail, the most any list does.
        let section = r1cs.constraint_bytes()?.len();
        if section / TERM_BYTES + n_public + 1 > u32::MAX as usize {
            return Err(CeremonyError::malformed(
                S_CONSTRAINTS,
                format!("{section} bytes can hold more terms than a u32 offset counts"),
            ));
        }

        let mut a = vec![0u32; n_vars + 1];
        let mut b = vec![0u32; n_vars + 1];
        let mut ic_c = vec![0u32; n_vars + 1];
        let (mut n_a, mut n_b) = (0usize, 0usize);
        r1cs.walk(|matrix, _, term| {
            let s = term.signal as usize;
            ic_c[s] += 1;
            match matrix {
                Matrix::A => {
                    a[s] += 1;
                    n_a += 1;
                }
                Matrix::B => {
                    b[s] += 1;
                    n_b += 1;
                }
                Matrix::C => {}
            }
        })?;
        for s in 0..=n_public {
            a[s] += 1;
            ic_c[s] += 1;
        }
        Ok(Self {
            a,
            b,
            ic_c,
            n_coefs: n_a + n_b + n_public + 1,
            n_public,
            n_constraints,
        })
    }

    /// The second walk: place every term, and hand section 4's records to `records` in push
    /// order, at most [`COEF_BATCH`] at a time. The first error `records` returns is the
    /// one this returns, once the walk is done.
    fn fill(
        self,
        r1cs: &R1cs,
        mut records: impl FnMut(&[CoefRecord]) -> Result<(), CeremonyError>,
    ) -> Result<Accumulators, CeremonyError> {
        let Self {
            a: mut a_at,
            b: mut b_at,
            ic_c: mut ic_c_at,
            n_coefs,
            n_public,
            n_constraints,
        } = self;
        let mut a = vec![PointTerm::EMPTY; counts_to_starts(&mut a_at)];
        let mut b = vec![PointTerm::EMPTY; counts_to_starts(&mut b_at)];
        let mut ic_c = vec![PointTerm::EMPTY; counts_to_starts(&mut ic_c_at)];

        let mut batch = Vec::with_capacity(COEF_BATCH);
        let mut failed = None;
        let mut emit = |batch: &mut Vec<CoefRecord>| {
            if failed.is_none() {
                if let Err(e) = records(batch) {
                    failed = Some(e);
                }
            }
            batch.clear();
        };
        // `IC` and `C` share a list, so one placement covers both destinations. The tag
        // differs per pass, which is the only thing that distinguishes the three.
        r1cs.walk(|matrix, constraint, term| {
            let s = term.signal as usize;
            let coef_ptr = Some(term.coef_ptr);
            match matrix {
                Matrix::A => {
                    put(&mut a, &mut a_at, s, BaseTag::TauG1, constraint, coef_ptr);
                    put(
                        &mut ic_c,
                        &mut ic_c_at,
                        s,
                        BaseTag::BetaTauG1,
                        constraint,
                        coef_ptr,
                    );
                }
                Matrix::B => {
                    put(&mut b, &mut b_at, s, BaseTag::TauG1, constraint, coef_ptr);
                    put(
                        &mut ic_c,
                        &mut ic_c_at,
                        s,
                        BaseTag::AlphaTauG1,
                        constraint,
                        coef_ptr,
                    );
                }
                // The C pass touches no coefficient record and no A/B accumulator: the C
                // matrix reaches the key only through `IC` and `C` (`zkey_new.js:272-287`).
                Matrix::C => {
                    put(
                        &mut ic_c,
                        &mut ic_c_at,
                        s,
                        BaseTag::TauG1,
                        constraint,
                        coef_ptr,
                    );
                    return;
                }
            }
            batch.push(CoefRecord {
                matrix: matrix.as_u32(),
                constraint,
                signal: term.signal,
                coef_ptr,
            });
            if batch.len() == COEF_BATCH {
                emit(&mut batch);
            }
        })?;

        // One synthetic row per public signal, `a = 1` and nothing else, so `A_s(x)` is
        // nonzero for every public signal (`zkey_new.js:290-300`).
        for s in 0..=n_public {
            let index = (n_constraints + s) as u32;
            put(&mut a, &mut a_at, s, BaseTag::TauG1, index, None);
            put(&mut ic_c, &mut ic_c_at, s, BaseTag::BetaTauG1, index, None);
            batch.push(CoefRecord {
                matrix: Matrix::A.as_u32(),
                constraint: index,
                signal: s as u32,
                coef_ptr: None,
            });
        }
        emit(&mut batch);
        if let Some(e) = failed {
            return Err(e);
        }

        Ok(Accumulators {
            a: slots(a_at, a),
            b: slots(b_at, b),
            ic_c: slots(ic_c_at, ic_c),
            n_coefs,
            n_public,
        })
    }
}

/// Per-slot counts to start offsets, in place, returning the total. The last entry, a
/// count of zero, becomes the total.
fn counts_to_starts(counts: &mut [u32]) -> usize {
    let mut at = 0u32;
    for c in counts.iter_mut() {
        (*c, at) = (at, at + *c);
    }
    at as usize
}

/// Place one term at its slot's cursor and advance the cursor.
fn put(
    terms: &mut [PointTerm],
    cursor: &mut [u32],
    slot: usize,
    tag: BaseTag,
    index: u32,
    coef_ptr: Option<u64>,
) {
    let at = &mut cursor[slot];
    terms[*at as usize] = PointTerm::new(tag, index, coef_ptr);
    *at += 1;
}

/// After a fill each cursor sits at its slot's end, which is the next slot's start, so
/// shifting them up one gives the offsets.
fn slots(mut cursor: Vec<u32>, terms: Vec<PointTerm>) -> Slots {
    let n = cursor.len() - 1;
    cursor.copy_within(0..n, 1);
    cursor[0] = 0;
    Slots {
        offsets: cursor,
        terms,
    }
}

/// Section 4's count is a u32 field, which a section 2 under about 144 GiB cannot
/// overflow.
fn coef_count(n: usize) -> Result<u32, CeremonyError> {
    u32::try_from(n)
        .map_err(|_| CeremonyError::BadParams(format!("{n} coefficients do not fit a u32 count")))
}

/// Section 4's records, 44 bytes each, the coefficient re-encoded from the r1cs' plain
/// little-endian into double Montgomery (`zkey_new.js:303-334`).
fn encode_coefficients(r1cs: &R1cs, batch: &[CoefRecord]) -> Result<Vec<u8>, CeremonyError> {
    let mut out = vec![0u8; batch.len() * COEF_RECORD_BYTES];
    out.par_chunks_mut(COEF_RECORD_BYTES)
        .zip(batch.par_iter())
        .try_for_each(|(out, rec)| {
            let value = coefficient(r1cs, rec.coef_ptr)?;
            out[..4].copy_from_slice(&rec.matrix.to_le_bytes());
            out[4..8].copy_from_slice(&rec.constraint.to_le_bytes());
            out[8..12].copy_from_slice(&rec.signal.to_le_bytes());
            out[12..].copy_from_slice(&crate::write::fr_double_montgomery(&value));
            Ok::<_, CeremonyError>(())
        })?;
    Ok(out)
}

/// The scalar a term contributes: the raw r1cs coefficient, or 1 for the `-1` sentinel.
fn coefficient(r1cs: &R1cs, coef_ptr: Option<u64>) -> Result<Fr, CeremonyError> {
    match coef_ptr {
        Some(ptr) => r1cs.coef(ptr),
        None => Ok(Fr::one()),
    }
}

/// Messages [`CsHash`] holds before a sender waits. Each is at most one chunk's encoding.
const HASH_QUEUE: usize = 2;

/// `csHash` on a thread of its own, fed bytes in send order through one channel.
///
/// BLAKE2b is serial, and on a 2^25 circuit each point section is 1.2 to 2.4 GB of it, so
/// absorbing a section in line held the whole pool for 10 to 15 s. Here the next chunk
/// composes while the last one is hashed. The digest depends only on the byte stream, not
/// on how it is split.
///
/// The worker never waits on the rayon pool, which is why senders encode the points and
/// not it: a sender can be a pool thread blocked on a full queue, and a worker queued
/// behind that thread's stack would never be served.
struct CsHash {
    tx: Option<SyncSender<Vec<u8>>>,
    worker: Option<JoinHandle<Transcript>>,
}

impl CsHash {
    fn spawn(mut transcript: Transcript) -> Result<Self, CeremonyError> {
        let (tx, rx) = sync_channel::<Vec<u8>>(HASH_QUEUE);
        let worker = std::thread::Builder::new()
            .name("cs-hash".into())
            .spawn(move || {
                for bytes in rx {
                    transcript.update(&bytes);
                }
                transcript
            })?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    fn send(&self, message: Vec<u8>) {
        // A send fails only once the worker has panicked, and `finish` re-raises that.
        if let Some(tx) = &self.tx {
            let _ = tx.send(message);
        }
    }

    /// Wait for everything sent to be absorbed.
    fn finish(mut self) -> Transcript {
        drop(self.tx.take());
        match self.worker.take().map(JoinHandle::join) {
            Some(Ok(transcript)) => transcript,
            Some(Err(panic)) => std::panic::resume_unwind(panic),
            None => unreachable!("finish consumes the hasher"),
        }
    }
}

impl Drop for CsHash {
    /// An early return still waits for the worker, so no thread outlives the setup.
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// One output point per accumulator slot: identity for an empty slot, and otherwise the
/// slot's terms summed, through a multiexp only once the slot reaches
/// `MULTIEXP_MIN_TERMS`.
pub fn compose_points_g1(
    slots: SlotRange<'_>,
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<G1Affine>, CeremonyError> {
    compose(
        slots,
        |term| bases.g1(term),
        |term| coefficient(r1cs, term.coef_ptr()),
        |b, s| msm.try_msm_g1(b, s),
    )
}

/// [`compose_points_g1`] for section 7, the only G2 output. Each term's base is its
/// element of `L_j(tau) * G2` whatever its tag, because B2 shares B1's list
/// ([`Accumulators::b`]).
pub fn compose_points_g2(
    slots: SlotRange<'_>,
    bases: &LagrangeBlocks,
    r1cs: &R1cs,
    msm: &dyn MsmBackend,
) -> Result<Vec<G2Affine>, CeremonyError> {
    compose(
        slots,
        |term| bases.tau_g2(term.index()),
        |term| coefficient(r1cs, term.coef_ptr()),
        |b, s| msm.try_msm_g2(b, s),
    )
}

/// Terms per rayon task in [`compose`], counting a slot that goes to the multiexp as
/// none. The task's slots are one [`glv::mul_sums_xyzz_in`] batch, which shares an
/// inversion per ladder step across all of them and holds a table per term.
const TERMS_PER_TASK: usize = 4096;

/// Most terms under one ladder in [`compose`]; a larger slot is several sums. The terms of
/// a sum share its doublings, but each step of the batch takes as many inversion rounds as
/// its busiest sum has nonzero digits at that step, so a sum of hundreds would pay a round
/// per term where its neighbours pay two or three.
const TERMS_PER_SUM: usize = 16;

/// [`compose_points_g1`] over any group: `base` and `scalar` give a term's point and
/// coefficient, `msm` takes the slots of at least `MULTIEXP_MIN_TERMS` terms.
///
/// The smaller slots go through batched ladders rather than one multiplication per term
/// ([`glv::mul_sums_xyzz_in`]): the terms of a slot, up to [`TERMS_PER_SUM`] at a time,
/// share their doublings, and the batch shares each step's inversion across the task. A
/// coefficient of 1 or -1, which is most of them on some circuits, is an addition and no
/// ladder. An error is the one of the lowest failing slot, as a serial walk would report
/// it.
fn compose<P, B, S, M>(
    slots: SlotRange<'_>,
    base: B,
    scalar: S,
    msm: M,
) -> Result<Vec<Affine<P>>, CeremonyError>
where
    P: RawCurve<RF: GlvField>,
    B: Fn(&PointTerm) -> Result<Affine<P>, CeremonyError> + Sync,
    S: Fn(&PointTerm) -> Result<Fr, CeremonyError> + Sync,
    M: Fn(&[Affine<P>], &[Fr]) -> Result<Projective<P>, AccelError> + Sync,
{
    let first_err = std::sync::Mutex::new(None::<(usize, CeremonyError)>);
    let fail = |i: usize, e: CeremonyError| {
        let mut slot = first_err.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_none_or(|(j, _)| i < *j) {
            *slot = Some((i, e));
        }
    };
    // Contiguous slot ranges of about `per_task` terms each.
    let mut tasks = Vec::new();
    let (mut from, mut terms) = (0, 0);
    for (i, slot) in slots.iter().enumerate() {
        if slot.len() < MULTIEXP_MIN_TERMS {
            terms += slot.len();
        }
        if terms >= TERMS_PER_TASK {
            tasks.push(from..i + 1);
            (from, terms) = (i + 1, 0);
        }
    }
    if from < slots.len() {
        tasks.push(from..slots.len());
    }
    let mut acc = vec![Projective::<P>::zero(); slots.len()];
    let mut rest = acc.as_mut_slice();
    let mut parts = Vec::with_capacity(tasks.len());
    for t in &tasks {
        let (head, tail) = rest.split_at_mut(t.len());
        parts.push((t.clone(), head));
        rest = tail;
    }
    parts
        .into_par_iter()
        .for_each_init(glv::BatchState::default, |st, (range, acc)| {
            let (mut points, mut scalars, mut ends, mut which) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            // Per slot: its terms of coefficient 1 or -1 (most of them on some circuits),
            // added straight in, and then its ladders' sums.
            let mut sum = vec![Xyzz::<P::RF>::ZERO; range.len()];
            let mut ladder = vec![false; range.len()];
            let minus_one = -Fr::one();
            for (k, slot) in slots.sub(range.clone()).iter().enumerate() {
                let i = range.start + k;
                let (p, s) = match gather(slot, &base, &scalar) {
                    Ok(t) => t,
                    Err(e) => {
                        fail(i, e);
                        continue;
                    }
                };
                if p.len() >= MULTIEXP_MIN_TERMS {
                    match combine(&p, &s, &msm) {
                        Ok(r) => acc[k] = r,
                        Err(e) => fail(i, e.into()),
                    }
                    continue;
                }
                ladder[k] = true;
                let start = points.len();
                for (q, s) in p.iter().zip(s) {
                    if q.infinity || s.is_zero() {
                        continue;
                    }
                    let (x, y) = P::raw_xy(q);
                    if s.is_one() {
                        sum[k].madd(x, y);
                    } else if s == minus_one {
                        sum[k].madd(x, y.neg());
                    } else {
                        points.push(Xyzz {
                            x,
                            y,
                            zz: P::RF::ONE,
                            zzz: P::RF::ONE,
                        });
                        scalars.push(s);
                    }
                }
                let mut end = start;
                while end < points.len() {
                    end = (end + TERMS_PER_SUM).min(points.len());
                    ends.push(end);
                    which.push(k);
                }
            }
            let mut out = vec![Xyzz::ZERO; ends.len()];
            glv::mul_sums_xyzz_in(st, &points, &scalars, &ends, &mut out);
            for (o, &k) in out.iter().zip(&which) {
                if !o.is_zero() {
                    sum[k].madd(o.x, o.y);
                }
            }
            for (k, s) in sum.iter().enumerate() {
                if ladder[k] {
                    acc[k] = to_projective(s);
                }
            }
        });
    if let Some((_, e)) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(e);
    }
    Ok(to_affine(&acc))
}

/// [`CurveGroup::normalize_batch`] a chunk at a time, into the one vector returned.
///
/// Over the whole slice it would hold a second vector of every `z` and a third of their
/// running products, one field element per signal each, while the projective input is
/// still alive.
pub(crate) fn to_affine<P: CurveGroup>(acc: &[P]) -> Vec<P::Affine> {
    let mut out = vec![P::Affine::zero(); acc.len()];
    out.par_chunks_mut(NORMALIZE_CHUNK)
        .zip(acc.par_chunks(NORMALIZE_CHUNK))
        .for_each(|(out, acc)| out.copy_from_slice(&P::normalize_batch(acc)));
    out
}

/// The bases and scalars of one slot, in term order.
fn gather<P, B, S>(
    slot: &[PointTerm],
    base: &B,
    scalar: &S,
) -> Result<(Vec<P>, Vec<Fr>), CeremonyError>
where
    B: Fn(&PointTerm) -> Result<P, CeremonyError>,
    S: Fn(&PointTerm) -> Result<Fr, CeremonyError>,
{
    let mut points = Vec::with_capacity(slot.len());
    let mut scalars = Vec::with_capacity(slot.len());
    for term in slot {
        points.push(base(term)?);
        scalars.push(scalar(term)?);
    }
    Ok((points, scalars))
}

/// snarkjs' multiexp branch (`zkey_new.js:476-481`) for a slot of at least
/// `MULTIEXP_MIN_TERMS` terms, with the shape checked first; the smaller slots, which
/// snarkjs splits again at one term, are [`compose`]'s batched ladders.
fn combine<P, F>(points: &[Affine<P>], scalars: &[Fr], msm: F) -> Result<Projective<P>, AccelError>
where
    P: RawCurve,
    F: Fn(&[Affine<P>], &[Fr]) -> Result<Projective<P>, AccelError>,
{
    if points.len() != scalars.len() {
        return Err(AccelError::Shape {
            op: "setup combine",
            reason: format!("{} bases but {} scalars", points.len(), scalars.len()),
        });
    }
    msm(points, scalars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_checks_shape_before_either_branch() {
        for n in [0, 1, MULTIEXP_MIN_TERMS - 1, MULTIEXP_MIN_TERMS + 1] {
            let points = vec![G1Affine::generator(); n];
            let scalars = vec![Fr::one(); n + 1];
            assert!(matches!(
                combine(&points, &scalars, |_, _| panic!("must reject before MSM")),
                Err(AccelError::Shape { .. })
            ));
            assert!(matches!(
                combine(
                    &vec![G1Affine::generator(); n + 1],
                    &scalars[..n],
                    |_, _| panic!("must reject before MSM")
                ),
                Err(AccelError::Shape { .. })
            ));
        }
    }

    #[test]
    fn small_slots_stay_on_cpu() {
        // Slots of 0, 1 and 31 terms, an empty one between, and one term a base the
        // lookup refuses, which must surface as the error of its slot and no other's.
        let len = [0, 1, MULTIEXP_MIN_TERMS - 1, 0, 3];
        let lists: Vec<Vec<PointTerm>> = len
            .iter()
            .map(|&n| {
                (0..n)
                    .map(|i| PointTerm::new(BaseTag::TauG1, i as u32, None))
                    .collect()
            })
            .collect();
        let slots = Slots::from_lists(&lists);
        let slots = slots.all();
        let g = G1Affine::generator();
        let three = |_: &PointTerm| Ok(Fr::from(3u64));
        let msm = |_: &[G1Affine], _: &[Fr]| -> Result<G1Projective, AccelError> {
            panic!("small slot must stay on CPU")
        };
        let got = compose(slots, |_| Ok(g), three, msm).unwrap();
        for (p, &n) in got.iter().zip(&len) {
            assert_eq!(*p, (g * Fr::from(3 * n as u64)).into_affine());
        }
        let bad = |t: &PointTerm| {
            if t.index() == 2 {
                Err(CeremonyError::BadParams("no such base".into()))
            } else {
                Ok(g)
            }
        };
        assert!(matches!(
            compose(slots, bad, three, msm),
            Err(CeremonyError::BadParams(_))
        ));
    }
}

/// The four `domainSize`-point Lagrange blocks setup reads out of the ptau, one per
/// [`BaseTag`], each from element `domainSize - 1` of its section.
///
/// Held as the ptau's own bytes and decoded per term, not decoded up front: on a 2^25
/// circuit the decoded blocks were 11.7 GB of anonymous memory, where the mapped file's
/// pages are ones the kernel can drop and read back. A decode is two range checks and a
/// copy, because a ptau stores coordinates in the Montgomery form arkworks keeps.
pub struct LagrangeBlocks<'a> {
    tau_g1: &'a [u8],
    tau_g2: &'a [u8],
    alpha_tau_g1: &'a [u8],
    beta_tau_g1: &'a [u8],
}

impl<'a> LagrangeBlocks<'a> {
    /// Locate the four blocks and check every coordinate in them is below `q`, which is
    /// all the decode they replace ever checked, so a block a term never reads still fails
    /// the same way it did.
    pub fn read(ptau: &'a Ptau, domain_size: usize) -> Result<Self, CeremonyError> {
        let block = |id: u32, stride: usize| -> Result<&'a [u8], CeremonyError> {
            let bytes = ptau.lagrange_block_bytes(id, stride, domain_size)?;
            bytes
                .par_chunks(N8)
                .with_min_len(1 << 14)
                .try_for_each(|f| binfile::fq(f).map(drop))?;
            Ok(bytes)
        };
        Ok(Self {
            tau_g1: block(ptau::S_LAGRANGE_TAU_G1, SG1)?,
            tau_g2: block(ptau::S_LAGRANGE_TAU_G2, SG2)?,
            alpha_tau_g1: block(ptau::S_LAGRANGE_ALPHA_TAU_G1, SG1)?,
            beta_tau_g1: block(ptau::S_LAGRANGE_BETA_TAU_G1, SG1)?,
        })
    }

    fn g1(&self, term: &PointTerm) -> Result<G1Affine, CeremonyError> {
        let block = match term.tag() {
            BaseTag::TauG1 => self.tau_g1,
            BaseTag::AlphaTauG1 => self.alpha_tau_g1,
            BaseTag::BetaTauG1 => self.beta_tau_g1,
            BaseTag::TauG2 => return Err(wrong_group(term)),
        };
        Ok(binfile::g1(element(block, SG1, term.index())?)?)
    }

    fn tau_g2(&self, index: u32) -> Result<G2Affine, CeremonyError> {
        Ok(binfile::g2(element(self.tau_g2, SG2, index)?)?)
    }
}

fn element(block: &[u8], stride: usize, index: u32) -> Result<&[u8], CeremonyError> {
    let at = index as usize * stride;
    block.get(at..at + stride).ok_or_else(|| {
        CeremonyError::BadParams(format!(
            "Lagrange element {index} is past the {}-point block",
            block.len() / stride
        ))
    })
}

fn wrong_group(term: &PointTerm) -> CeremonyError {
    CeremonyError::BadParams(format!("{:?} is not a base of this group", term.tag()))
}

/// Section 9: every odd element of the `2*domainSize` Lagrange block of ptau section 12.
/// No arithmetic. Only section 12 carries the `power+1` block, which is why this read is
/// possible at all when `cirPower == power`.
pub fn write_hs(ptau: &Ptau, domain_size: usize) -> Result<Vec<G1Affine>, CeremonyError> {
    let mut out = Vec::with_capacity(domain_size);
    h_points(ptau, domain_size, |points| {
        out.extend_from_slice(points);
        Ok(())
    })?;
    Ok(out)
}

/// [`write_hs`] handed to `emit` [`SECTION_CHUNK`] points at a time, in order.
fn h_points(
    ptau: &Ptau,
    domain_size: usize,
    mut emit: impl FnMut(&[G1Affine]) -> Result<(), CeremonyError>,
) -> Result<(), CeremonyError> {
    if !domain_size.is_power_of_two() {
        return Err(CeremonyError::BadParams(format!(
            "domain size {domain_size} is not a power of two"
        )));
    }
    let cir_power = domain_size.trailing_zeros();
    // `2*domainSize` points from the block that starts at element `2*domainSize - 1`.
    let block = 2 * domain_size;
    if cir_power < Fr::TWO_ADICITY {
        let data = ptau.section_elements(ptau::S_LAGRANGE_TAU_G1, SG1, block - 1, block)?;
        // Each record is an (even, odd) pair, so its second half is the odd element.
        for chunk in data.chunks(SECTION_CHUNK * 2 * SG1) {
            emit(&binfile::decode_records(
                chunk,
                2 * SG1,
                G1Affine::identity(),
                |_, b| binfile::g1(&b[SG1..]),
            )?)?;
        }
        Ok(())
    } else if cir_power == Fr::TWO_ADICITY {
        // At the two-adicity there is no `2*domainSize`-th root of unity, so
        // `lagrangeEvaluations` took its coset branch and wrote the even and odd halves
        // contiguously instead of interleaved. Every odd element is then just the second
        // half (`zkey_new.js:192-194`).
        let data = ptau.section_elements(
            ptau::S_LAGRANGE_TAU_G1,
            SG1,
            block - 1 + domain_size,
            domain_size,
        )?;
        for chunk in data.chunks(SECTION_CHUNK * SG1) {
            emit(&binfile::decode_records(
                chunk,
                SG1,
                G1Affine::identity(),
                |_, b| binfile::g1(b),
            )?)?;
        }
        Ok(())
    } else {
        Err(CeremonyError::CircuitTooBig(cir_power))
    }
}

/// Fold the H difference points into `csHash`, bug included.
///
/// The points are `tauG1[i + domainSize] - tauG1[i]` from ptau **section 2**, the raw
/// monomial powers, not the Lagrange section 12 (`zkey_new.js:517-518`).
///
/// `zkey_new.js:511` reads `Math.min(domainSize-1, CHUNK_SIZE)` and forgot the `- i`, so
/// `n` does not depend on the chunk offset. With `CHUNK_SIZE = 1<<14`, a `domainSize` of
/// 2^15 or more hashes a full 16384 points per chunk over `2^(k-14)` chunks, that is
/// `domainSize` points after announcing `domainSize - 1`. When additionally
/// `cirPower == power` the last chunk reads 64 bytes past the end of section 2, through a
/// raw `fd.read` that bounds-checks nothing, and folds section 3's header and the first 52
/// bytes of `tauG2[0]` into the digest.
///
/// This is the shipped behaviour and `csHash` is defined by it, so reproduce it. Which is
/// why the reads below go through the whole-file buffer and the section's start offset
/// rather than [`Ptau::section_elements`]: a bounds-checked read would refuse the last
/// chunk of exactly the case that matters, and why the subtraction goes through
/// [`h_difference`] rather than straight to arkworks.
///
/// The overrun affects only that one point, which is worth knowing because it is not
/// obvious: `batchToAffine` chains all 16384 z values through a single inversion, so a
/// non-congruent one could have polluted the whole chunk, and snarkjs splits that chunk
/// across `os.cpus().length` threads, which would then have made `csHash` depend on the
/// machine. Checked directly against ffjavascript: the whole-chunk and the eight-way split
/// results are identical, so the chain stays congruent and only the last point differs.
///
/// The verifier is immune because `sameRatioH` multiplies index `domainSize - 1` by zero.
pub fn hash_h_points(
    ptau: &Ptau,
    domain_size: usize,
    transcript: &mut Transcript,
) -> Result<(), CeremonyError> {
    let cs = CsHash::spawn(transcript.clone())?;
    let hashed = hash_h_points_into(ptau, domain_size, |b| cs.send(b));
    *transcript = cs.finish();
    hashed
}

/// [`hash_h_points`] with each chunk's bytes handed to `absorb` in order, so the hash can
/// run beside the next chunk's arithmetic.
fn hash_h_points_into(
    ptau: &Ptau,
    domain_size: usize,
    mut absorb: impl FnMut(Vec<u8>),
) -> Result<(), CeremonyError> {
    absorb(((domain_size - 1) as u32).to_be_bytes().to_vec());

    let file = ptau.file();
    let start = file
        .sections()
        .iter()
        .find(|s| s.id == ptau::S_TAU_G1)
        .ok_or(CeremonyError::MissingSection(ptau::S_TAU_G1))?
        .start;
    let bytes = file.bytes();

    let n = std::cmp::min(domain_size - 1, H_CHUNK);
    let read = |element: usize| -> Result<&[u8], CeremonyError> {
        let from = start + element * SG1;
        bytes.get(from..from + n * SG1).ok_or_else(|| {
            CeremonyError::malformed(
                ptau::S_TAU_G1,
                format!("H hash reads {n} points at element {element}, past the end of the file"),
            )
        })
    };
    let chunk = |i: usize| -> Result<Vec<u8>, CeremonyError> {
        let hi = read(i + domain_size)?;
        let lo = read(i)?;
        let diff: Vec<G1Projective> = hi
            .par_chunks_exact(SG1)
            .zip(lo.par_chunks_exact(SG1))
            .map(|(a, b)| h_difference(a, b))
            .collect();
        // Normalised a few thousand at a time rather than the chunk under one inversion;
        // each point's `x / z^2` and `y / z^3` are exact either way, the overrun point's
        // included (see above).
        Ok(transcript::g1_uncompressed_all(&to_affine(&diff)))
    };
    let mut i = 0usize;
    while i < domain_size - 1 {
        absorb(chunk(i)?);
        i += H_CHUNK;
    }
    Ok(())
}

/// One `g1m_subAffine`, on whichever implementation the input calls for.
///
/// Every real ptau point takes the arkworks path. The bytes read past section 2 do not: a
/// coordinate at or above `q` makes wasmcurves' wrapping arithmetic observable, and the
/// digest is defined by it, so those go through [`noncanonical`].
fn h_difference(hi: &[u8], lo: &[u8]) -> G1Projective {
    let canonical = [&hi[..N8], &hi[N8..], &lo[..N8], &lo[N8..]]
        .iter()
        .all(|b| noncanonical::in_range(&noncanonical::load(b)));
    if canonical {
        // `in_range` above is the same compare `binfile::g1` makes, so neither decode
        // can fail on this branch.
        let hi = binfile::g1(hi).expect("coordinates checked in range");
        let lo = binfile::g1(lo).expect("coordinates checked in range");
        return G1Projective::from(hi) - lo;
    }
    let (x, y, z) = noncanonical::sub_affine(hi, lo);
    if noncanonical::is_zero(&z) {
        return G1Projective::zero();
    }
    G1Projective::new_unchecked(
        montgomery_limbs(&x),
        montgomery_limbs(&y),
        montgomery_limbs(&z),
    )
}

/// Limbs wasmcurves left above `q` become the arkworks element with the same value: the
/// remaining steps, a batch inversion and two multiplications, are congruence preserving on
/// both sides, so only the representative had to be settled here.
fn montgomery_limbs(limbs: &noncanonical::U256) -> Fq {
    let mut le = [0u8; N8];
    for (i, limb) in limbs.iter().enumerate() {
        le[i * 4..i * 4 + 4].copy_from_slice(&limb.to_le_bytes());
    }
    Fq::new_unchecked(Fq::from_le_bytes_mod_order(&le).into_bigint())
}

/// wasmcurves' `Fq` and its Jacobian point addition, transcribed limb for limb.
///
/// It exists for one input in the whole pipeline: the 64 bytes [`hash_h_points`]
/// deliberately reads past ptau section 2, which are a section header and part of the next
/// point rather than a curve point. Both coordinates come out at or above `q`, and there
/// arkworks and wasmcurves stop agreeing. `f1m_add` (`build_f1m.js:69-85`) reacts to a
/// 256-bit carry by subtracting `q` once from the *wrapped* sum, which is short by exactly
/// `2^256`, so the result is not even congruent mod `q`. Reducing the inputs first does not
/// recover it, and neither does any canonical implementation: the shipped digest is defined
/// by these exact wrapping semantics.
///
/// Nothing else may use this. Every in-range input goes through arkworks, which is both
/// faster and, on those, identical.
mod noncanonical {
    /// Eight little-endian 32-bit limbs, the layout wasm linear memory holds and the width
    /// `f1m_mul`'s inner loop works in (`build_f1m.js:236-243`).
    pub type U256 = [u32; 8];

    pub const ZERO: U256 = [0; 8];
    const Q: U256 = [
        0xd87c_fd47,
        0x3c20_8c16,
        0x6871_ca8d,
        0x9781_6a91,
        0x8181_585d,
        0xb850_45b6,
        0xe131_a029,
        0x3064_4e72,
    ];
    /// `R mod q`, which is what `f1m_one` writes (`build_f1m.js:41`).
    const ONE: U256 = [
        0xc58f_0d9d,
        0xd35d_438d,
        0xf5c7_0b3d,
        0x0a78_eb28,
        0x7879_462c,
        0x666e_a36f,
        0x9a07_df2f,
        0x0e0a_77c1,
    ];
    /// `2^32 - (q^-1 mod 2^32)`, the Montgomery constant the 32-bit CIOS loop multiplies by.
    const NP32: u64 = 0xe486_6389;

    pub fn load(bytes: &[u8]) -> U256 {
        let mut out = ZERO;
        for (i, limb) in out.iter_mut().enumerate() {
            *limb = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
        }
        out
    }

    /// True when the limbs are a canonical field element, so arkworks and wasmcurves agree
    /// on every operation that follows and this module is not needed.
    pub fn in_range(x: &U256) -> bool {
        !int_gte(x, &Q)
    }

    pub fn is_zero(x: &U256) -> bool {
        x == &ZERO
    }

    /// `int_add`: the wrapped sum and the carry out (`build_int.js:186-230`).
    fn int_add(x: &U256, y: &U256) -> (U256, bool) {
        let mut out = ZERO;
        let mut carry = 0u64;
        for i in 0..8 {
            let acc = u64::from(x[i]) + u64::from(y[i]) + carry;
            out[i] = acc as u32;
            carry = acc >> 32;
        }
        (out, carry != 0)
    }

    /// `int_sub`: the wrapped difference and the borrow out (`build_int.js:232-278`).
    fn int_sub(x: &U256, y: &U256) -> (U256, bool) {
        let mut out = ZERO;
        let mut borrow = 0u64;
        for i in 0..8 {
            let acc = u64::from(x[i])
                .wrapping_sub(u64::from(y[i]))
                .wrapping_sub(borrow);
            out[i] = acc as u32;
            borrow = (acc >> 32) & 1;
        }
        (out, borrow != 0)
    }

    /// `int_gte`, an unsigned compare from the top limb down (`build_int.js:148-180`).
    fn int_gte(x: &U256, y: &U256) -> bool {
        for i in (0..8).rev() {
            if x[i] != y[i] {
                return x[i] > y[i];
            }
        }
        true
    }

    /// `f1m_add` (`build_f1m.js:69-85`). The carry branch subtracts `q` from a sum that has
    /// already lost `2^256`; that is the whole reason this module exists.
    pub fn add(x: &U256, y: &U256) -> U256 {
        let (r, carry) = int_add(x, y);
        if carry || int_gte(&r, &Q) {
            int_sub(&r, &Q).0
        } else {
            r
        }
    }

    /// `f1m_sub` (`build_f1m.js:87-102`).
    pub fn sub(x: &U256, y: &U256) -> U256 {
        let (r, borrow) = int_sub(x, y);
        if borrow {
            int_add(&r, &Q).0
        } else {
            r
        }
    }

    pub fn neg(x: &U256) -> U256 {
        sub(&ZERO, x)
    }

    /// `f1m_mul` (`build_f1m.js:235-435`): CIOS over 32-bit limbs with a two-word
    /// accumulator, then the same carry-or-gte reduction `add` uses.
    ///
    /// The `[c0, c1] = [c1, c0]` at `:407` is a rename in the code generator, so at run time
    /// it is a value swap followed by `c1 = c0 >> 32`, in that order.
    pub fn mul(x: &U256, y: &U256) -> U256 {
        let mut r = ZERO;
        let mut m = [0u64; 8];
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for k in 0..15usize {
            for i in k.saturating_sub(7)..=k.min(7) {
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(x[i]) * u64::from(y[k - i]);
                c1 += c0 >> 32;
            }
            for i in k.saturating_sub(7).max(1)..=k.min(7) {
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(Q[i]) * m[k - i];
                c1 += c0 >> 32;
            }
            if k < 8 {
                m[k] = ((c0 & 0xFFFF_FFFF) * NP32) & 0xFFFF_FFFF;
                c0 = (c0 & 0xFFFF_FFFF) + u64::from(Q[0]) * m[k];
                c1 += c0 >> 32;
            }
            if k >= 8 {
                r[k - 8] = c0 as u32;
            }
            std::mem::swap(&mut c0, &mut c1);
            c1 = c0 >> 32;
        }
        r[7] = c0 as u32;
        if c1 as u32 != 0 || int_gte(&r, &Q) {
            int_sub(&r, &Q).0
        } else {
            r
        }
    }

    /// `g1m_subAffine` (`build_curve_jacobian_a0.js:919-932`): negate the second point, then
    /// `addAffine`, which is mmadd-2007-bl with both `Z` implicitly one (`:760-843`).
    ///
    /// Returns the Jacobian triple `batchToAffine` would be handed.
    pub fn sub_affine(p1: &[u8], p2: &[u8]) -> (U256, U256, U256) {
        let (x1, y1) = (load(&p1[..32]), load(&p1[32..64]));
        let (x2, y2) = (load(&p2[..32]), neg(&load(&p2[32..64])));

        // `isZeroAffine` is an integer test on both coordinates, so it fires on the raw
        // limbs and not on the value.
        if is_zero(&x1) && is_zero(&y1) {
            return (x2, y2, ONE);
        }
        if is_zero(&x2) && is_zero(&y2) {
            return (x1, y1, ONE);
        }
        // Equality is on limbs too, so two representatives of the same value would miss the
        // doubling branch. That is the shipped behaviour.
        if x1 == x2 && y1 == y2 {
            return double_affine(&x2, &y2);
        }

        let h = sub(&x2, &x1);
        let y2_minus_y1 = sub(&y2, &y1);
        let hh = square(&h);
        let i = add(&hh, &hh);
        let i = add(&i, &i);
        let j = mul(&h, &i);
        let r = add(&y2_minus_y1, &y2_minus_y1);
        let v = mul(&x1, &i);
        let r2 = square(&r);
        let v2 = add(&v, &v);

        let x3 = sub(&r2, &j);
        let x3 = sub(&x3, &v2);

        let y1_j2 = mul(&y1, &j);
        let y1_j2 = add(&y1_j2, &y1_j2);

        let y3 = sub(&v, &x3);
        let y3 = mul(&y3, &r);
        let y3 = sub(&y3, &y1_j2);

        (x3, y3, add(&h, &h))
    }

    /// `doubleAffine`, dbl-2009-l (`build_curve_jacobian_a0.js:358-424`). Reachable from
    /// [`sub_affine`] only when the two limb patterns coincide.
    fn double_affine(x: &U256, y: &U256) -> (U256, U256, U256) {
        if is_zero(x) && is_zero(y) {
            return (ZERO, ZERO, ZERO);
        }
        let xx = square(x);
        let yy = square(y);
        let yyyy = square(&yy);
        let s = add(x, &yy);
        let s = square(&s);
        let s = sub(&s, &xx);
        let s = sub(&s, &yyyy);
        let s = add(&s, &s);
        let m = add(&xx, &xx);
        let m = add(&m, &xx);
        let z3 = add(y, y);
        let x3 = square(&m);
        let x3 = sub(&x3, &s);
        let x3 = sub(&x3, &s);
        let eight_yyyy = add(&yyyy, &yyyy);
        let eight_yyyy = add(&eight_yyyy, &eight_yyyy);
        let eight_yyyy = add(&eight_yyyy, &eight_yyyy);
        let y3 = sub(&s, &x3);
        let y3 = mul(&y3, &m);
        let y3 = sub(&y3, &eight_yyyy);
        (x3, y3, z3)
    }

    /// `f1m_square` is a separate function from `f1m_mul` (`build_f1m.js:437`), but only as
    /// a schedule: it accumulates the same partial products and ends in the same
    /// carry-or-gte reduction.
    ///
    /// Treating it as `mul(x, x)` is therefore an assumption, and it is the one assumption
    /// in this module. It holds on the inputs that reach here: [`sub_affine`] squares `H`
    /// and `r`, and the six exact-fit circuits in `tests/setup.rs` all come out
    /// byte-identical to snarkjs, which they could not if a square had diverged.
    fn square(x: &U256) -> U256 {
        mul(x, x)
    }
}
