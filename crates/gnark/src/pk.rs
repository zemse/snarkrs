//! gnark's `ProvingKey.WriteDump` (backend/groth16/bn254/marshal.go), read the way
//! `ReadDump` reads it:
//!
//! ```text
//!   u64 marker 0xdeadbeef (native, so little-endian)
//!   Domain.WriteTo: u64 BE cardinality, 32-byte BE CardinalityInv, Generator,
//!                   GeneratorInv, FrMultiplicativeGen, FrMultiplicativeGenInv, u8 precompute
//!   raw encoder points: G1.Alpha, G1.Beta, G1.Delta, G2.Beta, G2.Delta
//!   u64 BE nbWires, NbInfinityA, NbInfinityB
//!   InfinityA, InfinityB: nbWires bytes each, no length prefix
//!   u32 BE len(CommitmentKeys)
//!   unsafe.WriteSlice: G1.A, G1.B, G1.Z, G1.K, G2.B
//!   per commitment key: unsafe.WriteSlice Basis, BasisExpSigma
//! ```
//!
//! `G1.A`, `G1.B` and `G2.B` leave out every wire whose `InfinityA` or `InfinityB` flag is
//! set, and are put back to `nbWires` entries here with the identity in those slots.
//! `G1.Z` is `domain - 1` points `[tau^bitrev(j) * t(tau) / delta]_1`: setup bit-reverses
//! the `n` powers and drops the last, which is `bitrev(n - 1) = n - 1`, the one coefficient
//! of H that is always zero. `G1.K` is the private wires, `nbPublic..nbWires`.

use crate::{check_header_g1, check_header_g2, GnarkError, GnarkVerifyingKey, Reader, DUMP_MARKER};
use rayon::prelude::*;
use snarkrs_field::*;
use snarkrs_formats::binfile::{self, G1_BYTES, G2_BYTES};
use snarkrs_formats::{Coefficients, HBasis, ProvingKey};

/// The largest `log2(domain)` this crate accepts. The CPU prover evaluates on the odd half
/// of the `2n`-th roots of unity, so it needs a `2n`-th root, and Fr has 2-adicity 28.
const MAX_LOG_DOMAIN: u32 = Fr::TWO_ADICITY - 1;

/// Wires one task of the re-expansion handles. Small enough to spread a toy key over the
/// pool, large enough that the per-block count is noise next to the decode.
const EXPAND_BLOCK: usize = 1 << 14;

/// The fixed-size part of a dump, and where each dumped slice sits, without decoding a
/// single point. Cheap on any size of key, so it is also how a 6 GB dump is sized up
/// before committing the memory to load it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpHeader {
    pub domain_size: usize,
    pub n_wires: usize,
    pub nb_infinity_a: usize,
    pub nb_infinity_b: usize,
    /// Points in each dumped slice, in file order.
    pub g1_a: usize,
    pub g1_b: usize,
    pub g1_z: usize,
    pub g1_k: usize,
    pub g2_b: usize,
}

impl DumpHeader {
    /// Public wires, the constant wire included: what `G1.K` leaves out.
    pub fn n_public_wires(&self) -> usize {
        self.n_wires - self.g1_k
    }
}

/// A gnark proving key, every query re-expanded to snarkrs' one-base-per-wire layout.
pub struct GnarkProvingKey {
    pub header: DumpHeader,
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub delta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub delta_g2: G2Affine,
    /// `G1.A`, one base per wire.
    pub a: Vec<G1Affine>,
    /// `G1.B`, one base per wire.
    pub b_g1: Vec<G1Affine>,
    /// `G2.B`, one base per wire.
    pub b_g2: Vec<G2Affine>,
    /// `G1.Z`, `domain_size - 1` bases in gnark's bit-reversed order.
    pub z: Vec<G1Affine>,
    /// `G1.K`, one base per private wire.
    pub k: Vec<G1Affine>,
}

/// Where the five dumped slices are, once the header is through.
struct Layout<'a> {
    header: DumpHeader,
    points: [G1Affine; 3],
    g2_points: [G2Affine; 2],
    infinity_a: &'a [u8],
    infinity_b: &'a [u8],
    g1_a: &'a [u8],
    g1_b: &'a [u8],
    g1_z: &'a [u8],
    g1_k: &'a [u8],
    g2_b: &'a [u8],
}

impl GnarkProvingKey {
    /// Map a dump and decode it, every check below included.
    ///
    /// Every query point is checked on the curve, which for G1 is the subgroup too. The
    /// G2 query is not subgroup checked, as in the zkey loader: that is 70 to 90 us a point,
    /// and the one G2 point that leaves the prover is checked on the proof instead. The five
    /// header points are checked in full and refused at infinity, and so is a query that is
    /// entirely infinity. The domain's generator must be the one snarkrs builds for that
    /// size, or nothing the prover computes would line up with the key's `Z`.
    pub fn read_dump(path: &std::path::Path) -> Result<Self, GnarkError> {
        Self::from_bytes(&crate::map(path)?)
    }

    /// [`GnarkProvingKey::read_dump`] on a dump already in memory.
    pub fn from_bytes(data: &[u8]) -> Result<Self, GnarkError> {
        let l = layout(data)?;
        let flags_a = l.infinity_a;
        let flags_b = l.infinity_b;
        let g1 = |part: &'static str| {
            move |i: usize, b: &[u8]| -> Result<G1Affine, GnarkError> {
                let p = binfile::g1(b).map_err(|e| bad(part, format!("point {i}: {e}")))?;
                if !(p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()) {
                    return Err(bad(part, format!("point {i} is not on the curve")));
                }
                Ok(p)
            }
        };
        let g2 = |i: usize, b: &[u8]| -> Result<G2Affine, GnarkError> {
            let p = binfile::g2(b).map_err(|e| bad("G2.B", format!("point {i}: {e}")))?;
            if !p.is_on_curve() {
                return Err(bad("G2.B", format!("point {i} is not on the curve")));
            }
            Ok(p)
        };
        let ((a, b_g1), (b_g2, (z, k))) = rayon::join(
            || {
                rayon::join(
                    || expand(l.g1_a, G1_BYTES, flags_a, G1Affine::identity(), g1("G1.A")),
                    || expand(l.g1_b, G1_BYTES, flags_b, G1Affine::identity(), g1("G1.B")),
                )
            },
            || {
                rayon::join(
                    || expand(l.g2_b, G2_BYTES, flags_b, G2Affine::identity(), g2),
                    || {
                        rayon::join(
                            || decode(l.g1_z, G1_BYTES, G1Affine::identity(), g1("G1.Z")),
                            || decode(l.g1_k, G1_BYTES, G1Affine::identity(), g1("G1.K")),
                        )
                    },
                )
            },
        );
        let (a, b_g1, b_g2, z, k) = (a?, b_g1?, b_g2?, z?, k?);

        // A whole query of identities is the other half of the attack the header checks
        // stop, as in the zkey loader. An ALL check and not an ANY check: real keys carry
        // infinity points, and K is only ever all infinity on a key with no private wires.
        for (part, all_inf, n) in [
            (
                "G1.A",
                l.header.g1_a == 0 || a.iter().all(|p| p.infinity),
                a.len(),
            ),
            (
                "G1.B",
                l.header.g1_b == 0 || b_g1.iter().all(|p| p.infinity),
                b_g1.len(),
            ),
            (
                "G2.B",
                l.header.g2_b == 0 || b_g2.iter().all(|p| p.infinity),
                b_g2.len(),
            ),
            ("G1.Z", z.iter().all(|p| p.infinity), z.len()),
        ] {
            if n > 0 && all_inf {
                return Err(bad(
                    part,
                    format!(
                        "every one of the {n} points is the point at infinity; the query \
                         has been zeroed"
                    ),
                ));
            }
        }

        let [alpha_g1, beta_g1, delta_g1] = l.points;
        let [beta_g2, delta_g2] = l.g2_points;
        Ok(Self {
            header: l.header,
            alpha_g1,
            beta_g1,
            delta_g1,
            beta_g2,
            delta_g2,
            a,
            b_g1,
            b_g2,
            z,
            k,
        })
    }
}

impl GnarkProvingKey {
    /// snarkrs' proving key, with `vk` supplying what the dump does not hold: `gamma` and
    /// the public wires' `K`, which is snarkjs' IC. The five points both files hold must
    /// agree, and `vk` must have one `K` point per public wire.
    ///
    /// The mapping onto zkey terms: `G1.A`, `G1.B` and `G2.B` are the A, B1 and B2 queries;
    /// `G1.K` is the C query over the private wires, `witness[n_public + 1..]`, because
    /// gnark's `nbPublic` counts the constant wire and snarkjs' `n_public` does not; and
    /// `G1.Z` goes in the H query under [`HBasis::GnarkBitReversed`]. The key carries no
    /// constraint matrices, so it proves only from A.w and B.w rows.
    pub fn into_proving_key(self, vk: &GnarkVerifyingKey) -> Result<ProvingKey, GnarkError> {
        let mismatch = |what: &str| GnarkError::Mismatch {
            part: "verifying key",
            reason: format!("{what} differs from the proving key's"),
        };
        if vk.alpha_g1 != self.alpha_g1 {
            return Err(mismatch("G1.Alpha"));
        }
        if vk.beta_g1 != self.beta_g1 {
            return Err(mismatch("G1.Beta"));
        }
        if vk.delta_g1 != self.delta_g1 {
            return Err(mismatch("G1.Delta"));
        }
        if vk.beta_g2 != self.beta_g2 {
            return Err(mismatch("G2.Beta"));
        }
        if vk.delta_g2 != self.delta_g2 {
            return Err(mismatch("G2.Delta"));
        }
        let public_wires = self.header.n_public_wires();
        if vk.k.len() != public_wires {
            return Err(GnarkError::Mismatch {
                part: "verifying key",
                reason: format!(
                    "{} K points, the proving key leaves {public_wires} public wires",
                    vk.k.len()
                ),
            });
        }
        Ok(ProvingKey {
            n_vars: self.header.n_wires,
            n_public: public_wires - 1,
            domain_size: self.header.domain_size,
            alpha_g1: self.alpha_g1,
            beta_g1: self.beta_g1,
            beta_g2: self.beta_g2,
            delta_g1: self.delta_g1,
            delta_g2: self.delta_g2,
            a_query: self.a,
            b_g1_query: self.b_g1,
            b_g2_query: self.b_g2,
            l_query: self.k,
            h_query: self.z,
            h_basis: HBasis::GnarkBitReversed,
            coeffs: Coefficients {
                row_ptr: Default::default(),
                signal: Default::default(),
                value_id: Default::default(),
                values: Vec::new(),
            },
            vk: vk.to_verifying_key(),
        })
    }
}

impl DumpHeader {
    /// The header of a dump, with every length checked against the file and no point
    /// decoded. A few milliseconds on any key: it reads the two `nbWires` flag vectors and
    /// skips the rest.
    pub fn read(path: &std::path::Path) -> Result<Self, GnarkError> {
        Self::from_bytes(&crate::map(path)?)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, GnarkError> {
        Ok(layout(data)?.header)
    }
}

fn bad(part: &'static str, reason: String) -> GnarkError {
    GnarkError::Malformed { part, reason }
}

/// Everything up to the slices' contents: the header parsed and checked, every slice length
/// checked against the bytes present and against the header.
fn layout(data: &[u8]) -> Result<Layout<'_>, GnarkError> {
    let mut r = Reader::new(data, "marker");
    let marker: [u8; 8] = r.take(8)?.try_into().expect("8 bytes");
    if marker != DUMP_MARKER {
        return Err(GnarkError::BadMarker(marker));
    }

    r.part("domain");
    let domain_size = read_domain(&mut r)?;

    r.part("header points");
    let alpha_g1 = r.g1("G1.Alpha")?;
    let beta_g1 = r.g1("G1.Beta")?;
    let delta_g1 = r.g1("G1.Delta")?;
    let beta_g2 = r.g2("G2.Beta")?;
    let delta_g2 = r.g2("G2.Delta")?;
    for (p, what) in [
        (&alpha_g1, "G1.Alpha"),
        (&beta_g1, "G1.Beta"),
        (&delta_g1, "G1.Delta"),
    ] {
        check_header_g1(p, "header points", what)?;
    }
    for (p, what) in [(&beta_g2, "G2.Beta"), (&delta_g2, "G2.Delta")] {
        check_header_g2(p, "header points", what)?;
    }

    r.part("wire counts");
    let n_wires = r.u64_be()?;
    let nb_infinity_a = r.u64_be()?;
    let nb_infinity_b = r.u64_be()?;
    // Two flag bytes a wire follow, so the file bounds `nbWires` before anything sized from
    // it exists. Nothing is allocated for the flags at all: they are read in place.
    r.part("InfinityA");
    let infinity_a = r.records(n_wires, 1)?;
    r.part("InfinityB");
    let infinity_b = r.records(n_wires, 1)?;
    let n_wires = n_wires as usize;
    let count_a = count_flags(infinity_a, "InfinityA")?;
    let count_b = count_flags(infinity_b, "InfinityB")?;
    for (part, count, claimed) in [
        ("InfinityA", count_a, nb_infinity_a),
        ("InfinityB", count_b, nb_infinity_b),
    ] {
        if count as u64 != claimed {
            return Err(bad(
                part,
                format!("{count} flags are set, the header says {claimed}"),
            ));
        }
    }

    r.part("commitment count");
    let commitments = r.u32_be()?;
    if commitments != 0 {
        return Err(GnarkError::Commitments {
            part: "proving key",
            count: commitments.into(),
        });
    }

    r.part("G1.A");
    let g1_a = r.dumped_slice(G1_BYTES)?;
    r.part("G1.B");
    let g1_b = r.dumped_slice(G1_BYTES)?;
    r.part("G1.Z");
    let g1_z = r.dumped_slice(G1_BYTES)?;
    r.part("G1.K");
    let g1_k = r.dumped_slice(G1_BYTES)?;
    r.part("G2.B");
    let g2_b = r.dumped_slice(G2_BYTES)?;
    r.part("trailer");
    r.finish()?;

    let header = DumpHeader {
        domain_size,
        n_wires,
        nb_infinity_a: count_a,
        nb_infinity_b: count_b,
        g1_a: g1_a.len() / G1_BYTES,
        g1_b: g1_b.len() / G1_BYTES,
        g1_z: g1_z.len() / G1_BYTES,
        g1_k: g1_k.len() / G1_BYTES,
        g2_b: g2_b.len() / G2_BYTES,
    };
    for (part, got, want) in [
        ("G1.A", header.g1_a, n_wires - count_a),
        ("G1.B", header.g1_b, n_wires - count_b),
        ("G2.B", header.g2_b, n_wires - count_b),
        ("G1.Z", header.g1_z, domain_size - 1),
    ] {
        if got != want {
            return Err(bad(part, format!("{got} points, the header needs {want}")));
        }
    }
    // `G1.K` is the private wires, so it leaves at least the constant wire out.
    if header.g1_k >= n_wires {
        return Err(bad(
            "G1.K",
            format!(
                "{} points for {n_wires} wires leaves no room for the constant wire",
                header.g1_k
            ),
        ));
    }

    Ok(Layout {
        header,
        points: [alpha_g1, beta_g1, delta_g1],
        g2_points: [beta_g2, delta_g2],
        infinity_a,
        infinity_b,
        g1_a,
        g1_b,
        g1_z,
        g1_k,
        g2_b,
    })
}

/// `Domain.WriteTo`. Returns the cardinality once it is a power of two this prover can use
/// and the three elements derived from it are the ones snarkrs derives.
fn read_domain(r: &mut Reader<'_>) -> Result<usize, GnarkError> {
    let cardinality = r.u64_be()?;
    let mut elems = [Fr::zero(); 5];
    for (e, what) in elems.iter_mut().zip([
        "CardinalityInv",
        "Generator",
        "GeneratorInv",
        "FrMultiplicativeGen",
        "FrMultiplicativeGenInv",
    ]) {
        let b = r.take(32)?;
        *e = crate::fr_be(b)
            .ok_or_else(|| r.bad(format!("{what} is not below the scalar modulus")))?;
    }
    let precompute = r.u8()?;
    if precompute > 1 {
        return Err(r.bad(format!("withPrecompute byte is {precompute}, not a bool")));
    }

    let log = cardinality.trailing_zeros();
    if cardinality == 0 || !cardinality.is_power_of_two() || log > MAX_LOG_DOMAIN {
        return Err(r.bad(format!(
            "cardinality {cardinality} is not a power of two at most 2^{MAX_LOG_DOMAIN}"
        )));
    }
    let n = cardinality as usize;
    let domain = Domain::new(n).map_err(|e| r.bad(e.to_string()))?;
    let [card_inv, generator, generator_inv, _, _] = elems;
    // Both libraries take the 2^28-th root of unity from the same constant and square it
    // down, so a mismatch here is a key built for another field or another convention, and
    // the H the prover computes would not pair with this key's `Z`.
    for (got, want, what) in [
        (card_inv, domain.size_inv(), "CardinalityInv"),
        (generator, domain.group_gen(), "Generator"),
        (generator_inv, domain.group_gen_inv(), "GeneratorInv"),
    ] {
        if got != want {
            return Err(r.bad(format!(
                "{what} is not the one snarkrs derives for a domain of {n}"
            )));
        }
    }
    Ok(n)
}

/// How many of `flags` are set, refusing a byte that is not a bool. Go's `binary.Read`
/// takes any nonzero byte as true; a dump never writes one.
fn count_flags(flags: &[u8], part: &'static str) -> Result<usize, GnarkError> {
    if let Some(i) = flags.par_iter().position_first(|&f| f > 1) {
        return Err(bad(part, format!("flag {i} is {}, not a bool", flags[i])));
    }
    Ok(flags.par_iter().filter(|&&f| f == 1).count())
}

/// Decode a dumped slice in parallel into the one vector returned.
fn decode<T, F>(data: &[u8], stride: usize, fallback: T, f: F) -> Result<Vec<T>, GnarkError>
where
    T: Copy + Send + Sync,
    F: Fn(usize, &[u8]) -> Result<T, GnarkError> + Sync,
{
    binfile::decode_records(data, stride, fallback, f)
}

/// A dumped slice that left out every wire whose flag is set, decoded straight into
/// `flags.len()` entries with `identity` in the flagged slots. One pass and no compact
/// copy: blocks of wires count their unflagged wires, a prefix sum turns the counts into
/// each block's start in `data`, and each block then decodes its own run.
fn expand<T, F>(
    data: &[u8],
    stride: usize,
    flags: &[u8],
    identity: T,
    f: F,
) -> Result<Vec<T>, GnarkError>
where
    T: Copy + Send + Sync,
    F: Fn(usize, &[u8]) -> Result<T, GnarkError> + Sync,
{
    let mut starts: Vec<usize> = flags
        .par_chunks(EXPAND_BLOCK)
        .map(|block| block.iter().filter(|&&f| f == 0).count())
        .collect();
    let mut acc = 0;
    for s in &mut starts {
        let n = *s;
        *s = acc;
        acc += n;
    }
    // `layout` matched the slice to the flags; this is the invariant the blocks index by.
    assert_eq!(
        acc * stride,
        data.len(),
        "unflagged wires match the dumped slice"
    );

    let mut out = vec![identity; flags.len()];
    let first = std::sync::Mutex::new(None::<(usize, GnarkError)>);
    out.par_chunks_mut(EXPAND_BLOCK)
        .zip(flags.par_chunks(EXPAND_BLOCK))
        .zip(starts.par_iter())
        .for_each(|((out, flags), &start)| {
            let mut j = start;
            for (slot, &flag) in out.iter_mut().zip(flags) {
                if flag == 1 {
                    continue;
                }
                match f(j, &data[j * stride..(j + 1) * stride]) {
                    Ok(p) => *slot = p,
                    Err(e) => {
                        let mut first = first.lock().unwrap_or_else(|p| p.into_inner());
                        if first.as_ref().is_none_or(|(k, _)| j < *k) {
                            *first = Some((j, e));
                        }
                        return;
                    }
                }
                j += 1;
            }
        });
    match first.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some((_, e)) => Err(e),
        None => Ok(out),
    }
}
