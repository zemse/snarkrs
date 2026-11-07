//! Reading a gnark v0.14 BN254 R1CS, and the one pass over it that fixes, before any
//! witness exists, everything the solver would otherwise find out per term.
//!
//! The file is what `cs.WriteTo` writes (`constraint/bn254/marshal.go`): a u64 length, the
//! gnark version as three u64s, then `System.ToBytes` (`constraint/marshal.go`) and the
//! coefficient table. `System.ToBytes` is a header of four section lengths, then the levels
//! and the four instruction columns (both intcomp streams, see [`crate::intcomp`]), the
//! calldata (a u64 count and that many uvarints) and a CBOR body holding everything else.
//! The table is a u64 count and each coefficient's four Montgomery limbs, little-endian,
//! which is the in-memory layout of an arkworks `Fr`.
//!
//! The calldata stays as the file holds it, uvarints, and in the file: [`GnarkR1cs::read`]
//! maps just that section, so the 592 million values of SP1's circuit are 1.5 GB of clean
//! file pages rather than 2.4 GB of heap. The solver decodes them as it goes.

use crate::cbor;
use crate::intcomp;
use crate::ReadError;
use rayon::prelude::*;
use snarkrs_field::{BigInteger, Fr, PrimeField, Zero};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// gnark marks a constant term (no wire) with this wire ID (`Term.MarkConstant`).
pub(crate) const CONSTANT: u32 = u32::MAX;

/// The CBOR tags gnark's `getTagSet` assigns, in order, from 5309735.
const TAGS: [&str; 14] = [
    "BlueprintGenericHint",
    "BlueprintGenericR1C",
    "Groth16Commitments",
    "PlonkCommitments",
    "BlueprintGenericSparseR1C[U32]",
    "BlueprintSparseR1CAdd[U32]",
    "BlueprintSparseR1CMul[U32]",
    "BlueprintSparseR1CBool[U32]",
    "BlueprintLookupHint[U32]",
    "BlueprintGenericSparseR1C[U64]",
    "BlueprintSparseR1CAdd[U64]",
    "BlueprintSparseR1CMul[U64]",
    "BlueprintSparseR1CBool[U64]",
    "BlueprintLookupHint[U64]",
];
const TAG_BASE: u64 = 5309735;
const TAG_HINT: u64 = TAG_BASE;
const TAG_R1C: u64 = TAG_BASE + 1;
const TAG_GROTH16_COMMITMENTS: u64 = TAG_BASE + 2;

/// `ScalarField` as gnark writes it: BN254's r in lowercase hex.
const BN254_R_HEX: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

/// `constraint.SystemR1CS`.
const SYSTEM_R1CS: u64 = 1;

/// A uvarint pass hands each task at least this many bytes.
const VARINT_CHUNK: usize = 1 << 20;

/// A level shorter than this is checked on the calling thread.
const PAR_LEVEL: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Blueprint {
    Hint,
    R1c,
}

/// One instruction, in level order, as the solver runs it. `start` is the byte offset of
/// its calldata in [`Calldata`]. `solve` is what the pass found: for an R1C, 0 when every
/// wire in it is already known and the constraint is only checked, or else the index in
/// the instruction's calldata of the one unknown term's coefficient ID; [`Op::HINT`] for a
/// hint.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Op {
    pub start: u64,
    pub constraint: u32,
    pub solve: u32,
}

impl Op {
    pub const HINT: u32 = u32::MAX;
}

/// A gnark v0.14 BN254 R1CS, checked and ready to solve.
///
/// Reading it proves what the solver's hot loop then takes for granted: every instruction's
/// calldata lies inside the calldata and parses as its blueprint's layout, every coefficient
/// and wire ID is in range, every instruction is in exactly one level, every R1C has at most
/// one unknown wire and a constraint index of its own, no instruction reads a wire solved in
/// its own level or later, and every wire is solved exactly once. That is also what makes
/// it sound to solve a level's instructions in parallel over one wire vector.
pub struct GnarkR1cs {
    version: (u64, u64, u64),
    public_names: Vec<String>,
    secret_names: Vec<String>,
    n_internal: usize,
    n_constraints: usize,
    hint_dependencies: BTreeMap<u32, String>,
    pub(crate) coefficients: Vec<Fr>,
    /// `1 / coefficients[i]`, 0 for 0, which is what gnark's `Div` gives.
    pub(crate) coefficient_inverses: Vec<Fr>,
    pub(crate) calldata: Calldata,
    pub(crate) ops: Vec<Op>,
    /// Level `l` is `ops[level_ptr[l]..level_ptr[l + 1]]`.
    pub(crate) level_ptr: Vec<usize>,
    /// Every hint ID an instruction calls, sorted.
    pub(crate) hint_ids: Vec<u32>,
}

impl GnarkR1cs {
    /// Read a file `cs.WriteTo` wrote. The file is mapped to parse it, and only the
    /// calldata section stays mapped once it is read.
    #[cfg(not(target_family = "wasm"))]
    pub fn read(path: &std::path::Path) -> Result<Self, ReadError> {
        let file = std::fs::File::open(path).map_err(|e| ReadError::io(path, e))?;
        // SAFETY (both maps): read-only. Another process truncating or rewriting the file
        // while it is mapped is outside what any reader of a file can defend against; the
        // file is the one gnark wrote, and the parser bounds-checks every read of it.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| ReadError::io(path, e))?;
        let (parsed, range) = parse(&map)?;
        drop(map);
        let calldata = if range.is_empty() {
            Backing::Owned(Vec::new())
        } else {
            Backing::Mapped(
                unsafe {
                    memmap2::MmapOptions::new()
                        .offset(range.start as u64)
                        .len(range.len())
                        .map(&file)
                }
                .map_err(|e| ReadError::io(path, e))?,
            )
        };
        Ok(parsed.with(Calldata(calldata)))
    }

    /// Parse the bytes of a file `cs.WriteTo` wrote. The calldata is copied out of them.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReadError> {
        let (parsed, range) = parse(bytes)?;
        Ok(parsed.with(Calldata(Backing::Owned(bytes[range].to_vec()))))
    }

    /// The gnark version the file was written by.
    pub fn gnark_version(&self) -> (u64, u64, u64) {
        self.version
    }

    /// Public wires, the constant one wire (always wire 0) included.
    pub fn nb_public(&self) -> usize {
        self.public_names.len()
    }

    pub fn nb_secret(&self) -> usize {
        self.secret_names.len()
    }

    pub fn nb_internal(&self) -> usize {
        self.n_internal
    }

    /// Every wire: public, then secret, then internal, as gnark numbers them.
    pub fn nb_wires(&self) -> usize {
        self.public_names.len() + self.secret_names.len() + self.n_internal
    }

    pub fn nb_constraints(&self) -> usize {
        self.n_constraints
    }

    pub fn nb_instructions(&self) -> usize {
        self.ops.len()
    }

    pub fn nb_levels(&self) -> usize {
        self.level_ptr.len() - 1
    }

    pub fn nb_coefficients(&self) -> usize {
        self.coefficients.len()
    }

    /// The public inputs' names in wire order; the first is gnark's `"1"`.
    pub fn public_names(&self) -> &[String] {
        &self.public_names
    }

    pub fn secret_names(&self) -> &[String] {
        &self.secret_names
    }

    /// The hints the circuit was compiled against (`MHintsDependencies`): ID to the name
    /// gnark derived the ID from, or the ID itself for a hint registered by ID.
    pub fn hint_dependencies(&self) -> &BTreeMap<u32, String> {
        &self.hint_dependencies
    }

    /// How many instructions call each hint, by hint ID.
    pub fn hint_calls(&self) -> BTreeMap<u32, usize> {
        let mut calls = BTreeMap::new();
        for op in self.ops.iter().filter(|op| op.solve == Op::HINT) {
            *calls.entry(self.calldata.hint_id(op)).or_default() += 1;
        }
        calls
    }
}

/// The calldata section's uvarints, every one a u32, as the reader checked.
pub(crate) struct Calldata(Backing);

enum Backing {
    #[cfg(not(target_family = "wasm"))]
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Calldata {
    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.0 {
            #[cfg(not(target_family = "wasm"))]
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }

    /// A hint instruction's hint ID, its second value.
    pub(crate) fn hint_id(&self, op: &Op) -> u32 {
        let mut pos = op.start as usize;
        // SAFETY: an op's offset starts a hint's calldata, which holds five values at least.
        unsafe {
            uvarint(self.bytes(), &mut pos);
            uvarint(self.bytes(), &mut pos)
        }
    }
}

/// The uvarint at `pos`, advancing past it.
///
/// # Safety
/// `pos` starts a uvarint of the checked calldata: one of at most five bytes, ending inside
/// `bytes`, whose value is a u32.
#[inline(always)]
pub(crate) unsafe fn uvarint(bytes: &[u8], pos: &mut usize) -> u32 {
    let b = *bytes.get_unchecked(*pos);
    *pos += 1;
    if b < 0x80 {
        return u32::from(b);
    }
    let mut v = u32::from(b & 0x7f);
    let mut shift = 7;
    loop {
        let b = *bytes.get_unchecked(*pos);
        *pos += 1;
        v |= u32::from(b & 0x7f) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// Everything [`parse`] reads but the calldata, which the caller keeps its own way.
struct Parsed {
    version: (u64, u64, u64),
    public_names: Vec<String>,
    secret_names: Vec<String>,
    n_internal: usize,
    n_constraints: usize,
    hint_dependencies: BTreeMap<u32, String>,
    coefficients: Vec<Fr>,
    coefficient_inverses: Vec<Fr>,
    ops: Vec<Op>,
    level_ptr: Vec<usize>,
    hint_ids: Vec<u32>,
}

impl Parsed {
    fn with(self, calldata: Calldata) -> GnarkR1cs {
        GnarkR1cs {
            version: self.version,
            public_names: self.public_names,
            secret_names: self.secret_names,
            n_internal: self.n_internal,
            n_constraints: self.n_constraints,
            hint_dependencies: self.hint_dependencies,
            coefficients: self.coefficients,
            coefficient_inverses: self.coefficient_inverses,
            calldata,
            ops: self.ops,
            level_ptr: self.level_ptr,
            hint_ids: self.hint_ids,
        }
    }
}

/// Parse and check a file: it, and where in `bytes` its calldata uvarints are.
fn parse(bytes: &[u8]) -> Result<(Parsed, std::ops::Range<usize>), ReadError> {
    let mut c = Bytes::new(bytes, "file header");
    let total = c.u64()?;
    let version = (c.u64()?, c.u64()?, c.u64()?);
    if version.0 != 0 || version.1 != 14 {
        return Err(ReadError::UnsupportedVersion(format!(
            "{}.{}.{}",
            version.0, version.1, version.2
        )));
    }
    if total != c.rest().len() as u64 {
        return Err(ReadError::malformed(
            "file header",
            format!("says {total} bytes follow, {} do", c.rest().len()),
        ));
    }

    let mut s = Bytes::new(c.rest(), "system header");
    let lens = [s.u64()?, s.u64()?, s.u64()?, s.u64()?];
    let levels = s.take(lens[0])?;
    let instructions = s.take(lens[1])?;
    let calldata = s.take(lens[2])?;
    let body = s.take(lens[3])?;
    let coefficients = read_coefficients(s.rest())?;

    // The calldata count bounds the other sections: an instruction takes one value of
    // calldata at least, and a level entry is an instruction.
    let (count, calldata) = split_count(calldata, "calldata")?;
    let max_instructions = usize::try_from(count).unwrap_or(usize::MAX);
    // The body is the longest of these by far, and the calldata index wants the
    // instructions, so the index runs after them and beside the body.
    let (system, (levels, indexed)) = rayon::join(
        || read_body(body),
        || {
            rayon::join(
                || read_levels(levels, max_instructions),
                || {
                    let instructions = read_instructions(instructions, max_instructions)?;
                    let offsets = index_uvarints(calldata, count, &instructions.start)?;
                    Ok::<_, ReadError>((instructions, offsets))
                },
            )
        },
    );
    let (system, levels, (instructions, offsets)) = (system?, levels?, indexed?);

    let header_version = format!("{}.{}.{}", version.0, version.1, version.2);
    if system.gnark_version.trim_start_matches('v') != header_version {
        return Err(ReadError::malformed(
            "body",
            format!(
                "GnarkVersion {} disagrees with the header's {header_version}",
                system.gnark_version
            ),
        ));
    }

    let start = calldata.as_ptr() as usize - bytes.as_ptr() as usize;
    let indexed = Indexed {
        bytes: calldata,
        count,
        starts: &instructions.start,
        offsets,
    };
    let parsed = compile(
        version,
        system,
        coefficients,
        &indexed,
        levels,
        &instructions,
    )?;
    Ok((parsed, start..start + calldata.len()))
}

/// The CBOR body's fields the solver uses.
struct System {
    gnark_version: String,
    blueprints: Vec<Blueprint>,
    n_constraints: u64,
    n_internal: u64,
    public: Vec<String>,
    secret: Vec<String>,
    hint_dependencies: BTreeMap<u32, String>,
}

fn read_body(body: &[u8]) -> Result<System, ReadError> {
    read_body_inner(body).map_err(|e| ReadError::malformed("body", e))
}

fn read_body_inner(body: &[u8]) -> Result<System, String> {
    let mut r = cbor::Reader::new(body);
    let n = r.map_or_null("body")?.ok_or("body is null")?;
    let mut gnark_version = None;
    let mut scalar_field = None;
    let mut system_type = None;
    let mut blueprints = None;
    let mut n_constraints = None;
    let mut n_internal = None;
    let mut public = None;
    let mut secret = None;
    let mut hint_dependencies = BTreeMap::new();
    for _ in 0..n {
        let key = r.text("body key")?;
        match key {
            "GnarkVersion" => gnark_version = Some(r.text(key)?.to_owned()),
            "ScalarField" => scalar_field = Some(r.text(key)?.to_owned()),
            "Type" => system_type = Some(r.uint(key)?),
            "NbConstraints" => n_constraints = Some(non_negative(r.int(key)?, key)?),
            "NbInternalVariables" => n_internal = Some(non_negative(r.int(key)?, key)?),
            "Public" => public = Some(r.strings(key)?),
            "Secret" => secret = Some(r.strings(key)?),
            "Blueprints" => {
                let n = r.array_or_null(key)?.unwrap_or(0);
                if n > body.len() as u64 {
                    return Err(format!("{n} blueprints"));
                }
                let mut b = Vec::new();
                for _ in 0..n {
                    let tag = r.tag("blueprint")?;
                    let kind = match tag {
                        TAG_HINT => Blueprint::Hint,
                        TAG_R1C => Blueprint::R1c,
                        _ => {
                            let name = tag
                                .checked_sub(TAG_BASE)
                                .and_then(|i| TAGS.get(i as usize))
                                .copied()
                                .unwrap_or("not a gnark blueprint");
                            return Err(format!("unsupported blueprint: CBOR tag {tag} ({name})"));
                        }
                    };
                    // Both blueprints are field-less structs, an empty map.
                    if r.map_or_null("blueprint")? != Some(0) {
                        return Err("a blueprint with fields".into());
                    }
                    b.push(kind);
                }
                blueprints = Some(b);
            }
            "MHintsDependencies" => {
                let n = r.map_or_null(key)?.unwrap_or(0);
                for _ in 0..n {
                    let id = r.uint("hint ID")?;
                    let id = u32::try_from(id).map_err(|_| format!("hint ID {id}"))?;
                    hint_dependencies.insert(id, r.text("hint name")?.to_owned());
                }
            }
            "CommitmentInfo" => {
                if r.peek()? == cbor::NULL {
                    r.skip()?;
                    continue;
                }
                let tag = r.tag(key)?;
                if tag != TAG_GROTH16_COMMITMENTS || r.array_or_null(key)?.unwrap_or(0) != 0 {
                    return Err("commitments are not supported".into());
                }
            }
            "GkrInfo" => {
                let n = r.map_or_null(key)?.unwrap_or(0);
                for _ in 0..n {
                    if r.text("GkrInfo key")? == "Circuit" && r.peek()? != cbor::NULL {
                        return Err("GKR is not supported".into());
                    }
                    r.skip()?;
                }
            }
            _ => r.skip()?,
        }
    }
    if !r.is_empty() {
        return Err(format!("{} bytes after the body", body.len() - r.pos()));
    }
    let missing = |f: &str| format!("no {f}");
    let scalar_field = scalar_field.ok_or_else(|| missing("ScalarField"))?;
    if scalar_field != BN254_R_HEX {
        return Err(format!("scalar field {scalar_field} is not BN254's"));
    }
    if system_type.ok_or_else(|| missing("Type"))? != SYSTEM_R1CS {
        return Err("not an R1CS (a PLONK constraint system?)".into());
    }
    Ok(System {
        gnark_version: gnark_version.ok_or_else(|| missing("GnarkVersion"))?,
        blueprints: blueprints.ok_or_else(|| missing("Blueprints"))?,
        n_constraints: n_constraints.ok_or_else(|| missing("NbConstraints"))?,
        n_internal: n_internal.ok_or_else(|| missing("NbInternalVariables"))?,
        public: public.ok_or_else(|| missing("Public"))?,
        secret: secret.ok_or_else(|| missing("Secret"))?,
        hint_dependencies,
    })
}

fn non_negative(v: i64, what: &str) -> Result<u64, String> {
    u64::try_from(v).map_err(|_| format!("{what} is {v}"))
}

/// A cursor over little-endian fields of one section, naming it in errors.
struct Bytes<'a> {
    bytes: &'a [u8],
    what: &'static str,
}

impl<'a> Bytes<'a> {
    fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Self { bytes, what }
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8], ReadError> {
        if n > self.bytes.len() as u64 {
            return Err(ReadError::malformed(
                self.what,
                format!("{n} bytes wanted, {} left", self.bytes.len()),
            ));
        }
        let (head, rest) = self.bytes.split_at(n as usize);
        self.bytes = rest;
        Ok(head)
    }

    fn u64(&mut self) -> Result<u64, ReadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn rest(&self) -> &'a [u8] {
        self.bytes
    }
}

fn split_count<'a>(section: &'a [u8], what: &'static str) -> Result<(u64, &'a [u8]), ReadError> {
    let mut c = Bytes::new(section, what);
    Ok((c.u64()?, c.rest()))
}

fn read_coefficients(bytes: &[u8]) -> Result<Vec<Fr>, ReadError> {
    let (n, table) = split_count(bytes, "coefficient table")?;
    if table.len() as u64 != n.saturating_mul(32) {
        return Err(ReadError::malformed(
            "coefficient table",
            format!("{n} coefficients in {} bytes", table.len()),
        ));
    }
    let coefficients = table
        .as_chunks::<32>()
        .0
        .iter()
        .map(|b| {
            let limbs = std::array::from_fn(|i| {
                u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())
            });
            let big = ark_ff::BigInt::new(limbs);
            if big >= Fr::MODULUS {
                return Err(ReadError::malformed(
                    "coefficient table",
                    "a coefficient is not below r",
                ));
            }
            Ok(Fr::new_unchecked(big))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // gnark's table always opens with these five (`newCoeffTable`), and its solver treats
    // IDs 0 to 3 as these values whatever the table holds. Requiring them makes the two
    // readings agree, so the solver may compute a term either way.
    let fixed = [
        Fr::zero(),
        Fr::from(1u64),
        Fr::from(2u64),
        -Fr::from(1u64),
        -Fr::from(2u64),
    ];
    if coefficients.len() < 5 || coefficients[..5] != fixed {
        return Err(ReadError::malformed(
            "coefficient table",
            "does not open with 0, 1, 2, -1, -2",
        ));
    }
    Ok(coefficients)
}

/// An intcomp u32 stream as `ioutils.ReadAndDecompressUints32` frames it: a u64 count of
/// 32-bit words, then the words. Returns the values and the bytes after the stream.
fn read_u32_stream<'a>(
    bytes: &'a [u8],
    max: usize,
    what: &'static str,
) -> Result<(Vec<u32>, &'a [u8]), ReadError> {
    let mut c = Bytes::new(bytes, what);
    let n = c.u64()?;
    let words: Vec<u32> = c
        .take(n.saturating_mul(4))?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect();
    let values = intcomp::uncompress_u32(&words, max).map_err(|e| ReadError::malformed(what, e))?;
    Ok((values, c.rest()))
}

/// `levelsFromBytes`: a u64 level count, then one intcomp stream per level.
fn read_levels(bytes: &[u8], max_instructions: usize) -> Result<Vec<Vec<u32>>, ReadError> {
    let (n, mut rest) = split_count(bytes, "levels")?;
    // Every stream has an 8-byte length at least.
    if n > (rest.len() / 8) as u64 {
        return Err(ReadError::malformed(
            "levels",
            format!("{n} levels in {} bytes", rest.len()),
        ));
    }
    // Find each level's bytes first, so they can be decoded in parallel.
    let mut spans = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut c = Bytes::new(rest, "levels");
        let words = c.u64()?;
        let len = 8 + words.saturating_mul(4);
        let span = Bytes::new(rest, "levels").take(len)?;
        spans.push(span);
        rest = &rest[span.len()..];
    }
    if !rest.is_empty() {
        return Err(ReadError::malformed("levels", "bytes after the last level"));
    }
    let levels: Vec<Vec<u32>> = spans
        .into_par_iter()
        .map(|span| read_u32_stream(span, max_instructions, "levels").map(|(v, _)| v))
        .collect::<Result<_, _>>()?;
    let total: usize = levels.iter().map(Vec::len).sum();
    if total > max_instructions {
        return Err(ReadError::malformed(
            "levels",
            "more entries than calldata words",
        ));
    }
    Ok(levels)
}

/// gnark's `PackedInstruction`, which the solver keeps only until [`compile`] is done.
struct Instructions {
    blueprint: Vec<u32>,
    constraint: Vec<u32>,
    start: Vec<u64>,
}

/// `instructionsFromBytes`: blueprint IDs, constraint offsets and wire offsets as u32
/// streams, then calldata starts as a u64 stream. The solver has no use for wire offsets.
fn read_instructions(bytes: &[u8], max: usize) -> Result<Instructions, ReadError> {
    const WHAT: &str = "instructions";
    let (blueprint, rest) = read_u32_stream(bytes, max, WHAT)?;
    let (constraint, rest) = read_u32_stream(rest, max, WHAT)?;
    let (wires, rest) = read_u32_stream(rest, max, WHAT)?;
    let mut c = Bytes::new(rest, WHAT);
    let n = c.u64()?;
    let words: Vec<u64> = c
        .take(n.saturating_mul(8))?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&b| u64::from_le_bytes(b))
        .collect();
    if !c.rest().is_empty() {
        return Err(ReadError::malformed(WHAT, "bytes after the last column"));
    }
    let start = intcomp::uncompress_u64(&words, max).map_err(|e| ReadError::malformed(WHAT, e))?;
    if constraint.len() != blueprint.len()
        || wires.len() != blueprint.len()
        || start.len() != blueprint.len()
    {
        return Err(ReadError::malformed(WHAT, "columns of different lengths"));
    }
    Ok(Instructions {
        blueprint,
        constraint,
        start,
    })
}

/// `calldataFromBytes`'s `count` uvarints, each a u32, checked without being kept, and the
/// byte offset of each of `starts`, the value at which an instruction's calldata starts
/// (`u64::MAX` for one past the end). In parallel, as every byte below 0x80 ends a value: a
/// count of those per chunk places each chunk's values, and a chunk boundary moved to just
/// past such a byte starts a value.
fn index_uvarints(bytes: &[u8], count: u64, starts: &[u64]) -> Result<Vec<u64>, ReadError> {
    const WHAT: &str = "calldata";
    // One byte a value at least.
    if count > bytes.len() as u64 {
        return Err(ReadError::malformed(
            WHAT,
            format!("{count} values in {} bytes", bytes.len()),
        ));
    }
    let mut bounds = vec![0];
    let mut at = VARINT_CHUNK;
    while at < bytes.len() {
        // A u32 takes five bytes at most, so a boundary moves four at most.
        let mut b = at;
        while bytes[b - 1] >= 0x80 {
            b += 1;
            if b == bytes.len() || b - at > 4 {
                return Err(ReadError::malformed(WHAT, "a uvarint longer than a u32"));
            }
        }
        bounds.push(b);
        at = b + VARINT_CHUNK;
    }
    bounds.push(bytes.len());
    let chunks: Vec<(usize, &[u8])> = bounds
        .windows(2)
        .map(|w| (w[0], &bytes[w[0]..w[1]]))
        .collect();
    let counts: Vec<u64> = chunks
        .par_iter()
        .map(|(_, c)| c.iter().filter(|&&b| b < 0x80).count() as u64)
        .collect();
    if counts.iter().sum::<u64>() != count {
        return Err(ReadError::malformed(
            WHAT,
            "the uvarints do not match their count",
        ));
    }

    // gnark appends each instruction's calldata after the last, so the starts ascend and
    // each chunk finds its own with a binary search. Any other order is sorted first.
    let order: Option<Vec<u32>> = (!starts.is_sorted()).then(|| {
        let mut o: Vec<u32> = (0..starts.len() as u32).collect();
        o.par_sort_unstable_by_key(|&i| starts[i as usize]);
        o
    });
    let sorted: std::borrow::Cow<[u64]> = match &order {
        None => starts.into(),
        Some(o) => o
            .iter()
            .map(|&i| starts[i as usize])
            .collect::<Vec<_>>()
            .into(),
    };

    let mut offsets = vec![u64::MAX; starts.len()];
    let mut slots = &mut offsets[..];
    let mut parts = Vec::with_capacity(chunks.len());
    let (mut first, mut done) = (0u64, 0usize);
    for (&(offset, chunk), &n) in chunks.iter().zip(&counts) {
        // The starts whose values fall in this chunk.
        let end = done + sorted[done..].partition_point(|&s| s < first + n);
        let (head, tail) = slots.split_at_mut(end - done);
        parts.push((offset, chunk, first, &sorted[done..end], head));
        slots = tail;
        (first, done) = (first + n, end);
    }
    parts
        .into_par_iter()
        .try_for_each(|(offset, chunk, first, starts, slots)| {
            let mut index = first;
            let mut next = 0;
            let mut v = 0u64;
            let mut shift = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                while shift == 0 && next < starts.len() && starts[next] == index {
                    slots[next] = (offset + i) as u64;
                    next += 1;
                }
                v |= u64::from(b & 0x7f) << shift;
                if b < 0x80 {
                    if v > u64::from(u32::MAX) {
                        return Err(());
                    }
                    index += 1;
                    v = 0;
                    shift = 0;
                } else {
                    shift += 7;
                    if shift > 28 {
                        return Err(());
                    }
                }
            }
            // The last chunk ends on a value's last byte, as the boundaries do.
            if shift != 0 {
                return Err(());
            }
            Ok(())
        })
        .map_err(|()| ReadError::malformed(WHAT, "a uvarint is not a u32"))?;
    if let Some(order) = order {
        let mut by_instruction = vec![0; starts.len()];
        for (&i, &o) in order.iter().zip(&offsets) {
            by_instruction[i as usize] = o;
        }
        offsets = by_instruction;
    }
    Ok(offsets)
}

/// The checked calldata, and where each instruction's starts.
struct Indexed<'a> {
    bytes: &'a [u8],
    count: u64,
    starts: &'a [u64],
    offsets: Vec<u64>,
}

impl Indexed<'_> {
    /// Instruction `i`'s byte offset and a reader of its values after the first, its length,
    /// which must fit in what is left of the calldata.
    fn instruction(&self, i: usize) -> Option<(u64, usize, Values<'_>)> {
        let offset = self.offsets[i];
        if offset == u64::MAX {
            return None;
        }
        let mut pos = offset as usize;
        // SAFETY: `offset` starts a value of the checked calldata.
        let len = unsafe { uvarint(self.bytes, &mut pos) } as u64;
        if len == 0 || len > self.count - self.starts[i] {
            return None;
        }
        Some((
            offset,
            len as usize,
            Values {
                bytes: self.bytes,
                pos,
                left: len as usize - 1,
            },
        ))
    }
}

/// The values of one instruction, which [`Indexed::instruction`] checked are all there.
struct Values<'a> {
    bytes: &'a [u8],
    pos: usize,
    left: usize,
}

impl Iterator for Values<'_> {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        // SAFETY: one of the values the instruction's length promised, inside the calldata.
        Some(unsafe { uvarint(self.bytes, &mut self.pos) })
    }
}

/// The static solve: run every level once without values, tracking only which wires are
/// known, and keep what that decides. `stamp[w]` is 0 for an input, `l + 1` for a wire
/// solved at level `l`, and `UNSOLVED` until then.
fn compile(
    version: (u64, u64, u64),
    system: System,
    coefficients: Vec<Fr>,
    calldata: &Indexed,
    levels: Vec<Vec<u32>>,
    instructions: &Instructions,
) -> Result<Parsed, ReadError> {
    const UNSOLVED: u32 = u32::MAX;
    let malformed = |reason: String| ReadError::malformed("instructions", reason);

    let n_inputs = system.public.len() + system.secret.len();
    let n_wires = (n_inputs as u64).saturating_add(system.n_internal);
    // u32::MAX is the constant marker. And an internal wire is one an instruction solves,
    // so the instructions bound how many there can be before a stamp is allocated for each.
    let solvable = instructions
        .blueprint
        .par_iter()
        .enumerate()
        .map_init(Vec::new, |cd, (i, &b)| {
            match system.blueprints.get(b as usize) {
                Some(Blueprint::R1c) => 1,
                Some(Blueprint::Hint) => calldata
                    .instruction(i)
                    .and_then(|(_, len, values)| {
                        cd.clear();
                        cd.push(len as u32);
                        cd.extend(values);
                        hint_layout(cd)
                    })
                    .map_or(0, |(_, from, to)| u64::from(to - from)),
                None => 0,
            }
        })
        .reduce(|| 0, u64::saturating_add);
    if n_wires >= u64::from(CONSTANT) || system.n_internal > solvable {
        return Err(malformed(format!("{n_wires} wires")));
    }
    if system.public.first().map(String::as_str) != Some("1") {
        return Err(malformed("wire 0 is not the constant one wire".into()));
    }
    let n_wires = n_wires as usize;
    let n_constraints = usize::try_from(system.n_constraints).unwrap_or(usize::MAX);
    let n_instructions = instructions.blueprint.len();
    if n_constraints > n_instructions {
        return Err(malformed(format!(
            "{n_constraints} constraints in {n_instructions} instructions"
        )));
    }
    let n_coefficients = coefficients.len() as u32;

    let stamp: Vec<AtomicU32> = (0..n_wires)
        .into_par_iter()
        .map(|w| AtomicU32::new(if w < n_inputs { 0 } else { UNSOLVED }))
        .collect();
    let placed: Vec<AtomicBool> = (0..n_instructions)
        .into_par_iter()
        .map(|_| AtomicBool::new(false))
        .collect();
    let constrained: Vec<AtomicBool> = (0..n_constraints)
        .into_par_iter()
        .map(|_| AtomicBool::new(false))
        .collect();

    let check = |cd: &mut Vec<u32>, level: u32, i: u32| -> Result<Op, String> {
        let i = i as usize;
        if i >= n_instructions || placed[i].swap(true, Ordering::Relaxed) {
            return Err(format!("instruction {i} is out of range or in two levels"));
        }
        let blueprint = *system
            .blueprints
            .get(instructions.blueprint[i] as usize)
            .ok_or_else(|| format!("instruction {i}: blueprint {}", instructions.blueprint[i]))?;
        let (start, len, mut values) = calldata
            .instruction(i)
            .ok_or_else(|| format!("instruction {i}: calldata out of range"))?;
        let known = |w: u32| stamp[w as usize].load(Ordering::Relaxed) <= level;
        let claim = |w: u32| {
            stamp[w as usize]
                .compare_exchange(UNSOLVED, level + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        };
        match blueprint {
            Blueprint::R1c => {
                if len < 4 {
                    return Err(format!("instruction {i}: short R1C"));
                }
                let mut next = || values.next().unwrap();
                let terms = next() as u64 + next() as u64 + next() as u64;
                if 4 + 2 * terms != len as u64 {
                    return Err(format!("instruction {i}: R1C length"));
                }
                let mut unknown = 0u32;
                let mut unknown_wire = 0;
                for k in (4..len).step_by(2) {
                    let (c, w) = (next(), next());
                    if c >= n_coefficients || w as usize >= n_wires {
                        return Err(format!("instruction {i}: term out of range"));
                    }
                    if !known(w) {
                        // gnark panics on a second unknown and divides by the coefficient,
                        // so a zero one (ID 0) panics too.
                        if unknown != 0 || c == 0 {
                            return Err(format!("instruction {i}: R1C cannot be solved"));
                        }
                        unknown = k as u32;
                        unknown_wire = w;
                    }
                }
                if unknown != 0 && !claim(unknown_wire) {
                    return Err(format!("instruction {i}: wire solved twice"));
                }
                let constraint = instructions.constraint[i];
                if constraint as usize >= n_constraints
                    || constrained[constraint as usize].swap(true, Ordering::Relaxed)
                {
                    return Err(format!(
                        "instruction {i}: constraint {constraint} out of range or reused"
                    ));
                }
                Ok(Op {
                    start,
                    constraint,
                    solve: unknown,
                })
            }
            Blueprint::Hint => {
                cd.clear();
                cd.push(len as u32);
                cd.extend(values);
                let (inputs_end, from, to) =
                    hint_layout(cd).ok_or_else(|| format!("instruction {i}: hint calldata"))?;
                let mut j = 3;
                while j < inputs_end {
                    let n = cd[j] as usize;
                    for &[c, w] in cd[j + 1..j + 1 + 2 * n].as_chunks::<2>().0 {
                        if c >= n_coefficients || (w != CONSTANT && w as usize >= n_wires) {
                            return Err(format!("instruction {i}: term out of range"));
                        }
                        // gnark adds a zero-coefficient term without reading its wire.
                        if w != CONSTANT && c != 0 && !known(w) {
                            return Err(format!("instruction {i}: hint reads an unsolved wire"));
                        }
                    }
                    j += 1 + 2 * n;
                }
                if to as usize > n_wires {
                    return Err(format!("instruction {i}: hint outputs {from}..{to}"));
                }
                if !(from..to).all(claim) {
                    return Err(format!("instruction {i}: wire solved twice"));
                }
                Ok(Op {
                    start,
                    constraint: 0,
                    solve: Op::HINT,
                })
            }
        }
    };

    let total: usize = levels.iter().map(Vec::len).sum();
    if total != n_instructions {
        return Err(malformed(format!(
            "{total} level entries for {n_instructions} instructions"
        )));
    }
    let mut ops = Vec::with_capacity(n_instructions);
    let mut level_ptr = Vec::with_capacity(levels.len() + 1);
    level_ptr.push(0);
    let mut cd = Vec::new();
    for (l, level) in levels.iter().enumerate() {
        let l = u32::try_from(l).map_err(|_| malformed("too many levels".into()))?;
        if level.len() < PAR_LEVEL {
            for &i in level {
                ops.push(check(&mut cd, l, i).map_err(malformed)?);
            }
        } else {
            let at = ops.len();
            ops.resize(at + level.len(), Op::default());
            let task = 8.max(level.len() / (32 * rayon::current_num_threads()));
            ops[at..]
                .par_chunks_mut(task)
                .zip(level.par_chunks(task))
                .try_for_each(|(ops, level)| {
                    let mut cd = Vec::new();
                    for (op, &i) in ops.iter_mut().zip(level) {
                        *op = check(&mut cd, l, i)?;
                    }
                    Ok::<_, String>(())
                })
                .map_err(malformed)?;
        }
        level_ptr.push(ops.len());
    }
    if let Some(w) = stamp
        .par_iter()
        .position_any(|s| s.load(Ordering::Relaxed) == UNSOLVED)
    {
        return Err(malformed(format!("wire {w} is never solved")));
    }
    if constrained.par_iter().any(|c| !c.load(Ordering::Relaxed)) {
        return Err(malformed("a constraint index no R1C has".into()));
    }
    drop((stamp, placed, constrained, levels));

    let mut hint_ids: Vec<u32> = ops
        .par_iter()
        .filter(|op| op.solve == Op::HINT)
        .map(|op| {
            let mut pos = op.start as usize;
            // SAFETY: the op's offset starts a checked hint instruction.
            unsafe {
                uvarint(calldata.bytes, &mut pos);
                uvarint(calldata.bytes, &mut pos)
            }
        })
        .collect();
    hint_ids.sort_unstable();
    hint_ids.dedup();

    // batch_inversion leaves a zero as zero.
    let mut coefficient_inverses = coefficients.clone();
    ark_ff::batch_inversion(&mut coefficient_inverses);

    Ok(Parsed {
        version,
        public_names: system.public,
        secret_names: system.secret,
        n_internal: n_wires - n_inputs,
        n_constraints,
        hint_dependencies: system.hint_dependencies,
        coefficients,
        coefficient_inverses,
        ops,
        level_ptr,
        hint_ids,
    })
}

/// Where a hint's inputs end and its output range, if its calldata parses:
/// `[size, hint ID, input count, (term count, (coefficient ID, wire ID)...)..., from, to]`.
pub(crate) fn hint_layout(cd: &[u32]) -> Option<(usize, u32, u32)> {
    let n_inputs = *cd.get(2)?;
    let mut j = 3usize;
    for _ in 0..n_inputs {
        let n = *cd.get(j)? as usize;
        j = j.checked_add(1 + n.checked_mul(2)?)?;
    }
    if j.checked_add(2)? != cd.len() || cd[j] > cd[j + 1] {
        return None;
    }
    Some((j, cd[j], cd[j + 1]))
}

/// Decimal, as gnark's `fr.Element.Text(10)` prints it: an element whose negation is at
/// most 65535 prints as that negation with a minus sign.
pub(crate) fn gnark_text(x: &Fr) -> String {
    let neg = (-*x).into_bigint();
    if !x.is_zero() && neg.0[1..].iter().all(|&l| l == 0) && neg.0[0] <= 65535 {
        return format!("-{}", neg.0[0]);
    }
    let v = x.into_bigint();
    if v.0[1..].iter().all(|&l| l == 0) {
        return v.0[0].to_string();
    }
    num_bigint::BigUint::from_bytes_le(&v.to_bytes_le()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uvarints(values: &[u32]) -> (Vec<u8>, Vec<usize>) {
        let (mut bytes, mut at) = (Vec::new(), Vec::new());
        for &v in values {
            at.push(bytes.len());
            let mut v = v;
            while v >= 0x80 {
                bytes.push(v as u8 | 0x80);
                v >>= 7;
            }
            bytes.push(v as u8);
        }
        (bytes, at)
    }

    /// Starts in any order, repeated, or past the end, each find their own byte offset,
    /// across more than one parallel chunk.
    #[test]
    fn indexes_starts_in_any_order() {
        let values: Vec<u32> = (0..800_000u32)
            .map(|i| i.wrapping_mul(2654435761) >> (i % 32))
            .collect();
        let (bytes, at) = uvarints(&values);
        assert!(bytes.len() > 2 * VARINT_CHUNK);
        let n = values.len() as u64;
        for starts in [
            vec![0, 1, 5, 5, 700_000, n - 1],
            vec![n - 1, 3, 799_000, 3, 0, 123_456, n, n + 7],
        ] {
            let offsets = index_uvarints(&bytes, n, &starts).unwrap();
            for (&s, &o) in starts.iter().zip(&offsets) {
                let want = if s < n {
                    at[s as usize] as u64
                } else {
                    u64::MAX
                };
                assert_eq!(o, want, "start {s}");
            }
        }
        assert!(index_uvarints(&bytes, n + 1, &[]).is_err());
        assert!(index_uvarints(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x01], 1, &[]).is_err());
        assert!(index_uvarints(&[0xff, 0xff, 0xff, 0xff, 0x1f], 1, &[]).is_err());
        assert!(index_uvarints(&[0xff, 0xff, 0xff, 0xff, 0x0f], 1, &[]).is_ok());
    }

    #[test]
    fn text_is_gnarks() {
        assert_eq!(gnark_text(&Fr::zero()), "0");
        assert_eq!(gnark_text(&-Fr::from(65535u64)), "-65535");
        assert_eq!(gnark_text(&-Fr::from(65536u64)).len(), 77);
        assert_eq!(gnark_text(&Fr::from(u64::MAX)), u64::MAX.to_string());
    }
}
