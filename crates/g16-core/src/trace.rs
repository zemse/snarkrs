//! A deterministic execution trace, for finding where two machines disagree.
//!
//! **Debugging only. Never a proving path.** Stage 10's blinders are fixed constants here,
//! and a reused `(r, s)` across two proofs of different witnesses leaks the witness. Nothing
//! in this module is reachable from [`crate::prove::prove`]; the entry points that use it say
//! `trace` in their name, return a `Trace` rather than a `Proof`, and the CLI's `trace`
//! subcommand refuses to write a `proof.json`.
//!
//! # What it is for
//!
//! An iPhone runs the WebGPU backend to completion and produces a proof snarkjs rejects,
//! deterministically, while the public signals match. The failure is therefore somewhere
//! between the witness and the five MSM outputs, and a wrong 256-bit number carries no
//! information about which stage wrote it. Given the same zkey and the same witness, stages 0
//! to 9 are a pure function: two correct machines must agree on every intermediate value.
//! Pin the blinders and stage 11 becomes pure too, so the whole proof is comparable and a
//! diff between a laptop's trace and a phone's names the first stage that disagrees.
//!
//! # Two things would make a naive version of this useless
//!
//! **Points have no canonical form.** The MSMs accumulate in projective coordinates, where
//! one curve point has many representations depending on the order the additions happened in,
//! and the scatter that feeds them is built with atomics, so that order is not even stable
//! between two runs on the same device. Every point here is normalised to affine before it is
//! written, which is the only representation two machines can be held to.
//!
//! **A single digest of `H` says "different" and stops.** The coefficients are digested in
//! [`H_CHUNKS`] equal pieces instead, so a divergence is localised to one chunk of the domain
//! without the trace carrying a megabyte of field elements. The chunk index is a real lead:
//! stage 4 is elementwise, so a wrong chunk is a wrong range of dispatches.
//!
//! # The digest
//!
//! FNV-1a over the canonical little-endian bytes. Not a cryptographic hash and it does not
//! need to be: the two sides of this comparison are two runs of our own code, not an
//! adversary, and the alternative was a dependency this workspace does not otherwise have.

use crate::{MsmOutputs, Proof};
use g16_field::*;

/// Stage 10's `r`, pinned. An arbitrary constant, and arbitrary is the point: a value with
/// structure (0, 1, a power of two) would let a bug that drops a whole term still produce
/// matching traces.
pub const TRACE_R: u64 = 0x5772_6f6e_6720_5231;
/// Stage 10's `s`, pinned. See [`TRACE_R`].
pub const TRACE_S: u64 = 0x5772_6f6e_6720_5332;

/// How many pieces the `H` coefficients are digested in. 64 puts a 2^18 domain into 4,096
/// coefficients per chunk, which is narrow enough to point at a range of stage 4 dispatches
/// and short enough that the trace stays a page of text.
pub const H_CHUNKS: usize = 64;

/// The fixed blinders. See the module docs for why this is not an argument someone can reach
/// for by accident.
pub fn blinders() -> (Fr, Fr) {
    (Fr::from(TRACE_R), Fr::from(TRACE_S))
}

/// FNV-1a 64 over the canonical little-endian bytes of a run of field elements.
///
/// `into_bigint` rather than the in-memory representation, because the CPU backend keeps `Fr`
/// in Montgomery form and a GPU backend that read its buffer back in standard form would
/// disagree with it over values that are equal.
fn digest_fr(xs: &[Fr]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in xs {
        for b in x.into_bigint().to_bytes_le() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// One named intermediate value, rendered.
struct Row {
    name: &'static str,
    value: String,
}

/// The comparable state of one proof, from the witness to the assembled proof.
///
/// Held as rendered strings rather than as points and field elements, because the only thing
/// anybody does with it is diff it against another one. Building it is where the
/// normalisation and the digesting happen, so a `Trace` cannot be constructed carrying a
/// projective coordinate that would compare unequal to an equal point.
pub struct Trace {
    backend: &'static str,
    shape: Vec<Row>,
    rows: Vec<Row>,
    h_chunks: Option<Vec<u64>>,
}

impl Trace {
    /// `h` is the stage 0-4 output, if the backend can hand it over. A GPU backend leaves it
    /// in device memory and reading it down is a deliberate extra round trip, so this takes
    /// an `Option` rather than forcing every caller to pay for it: without it the trace still
    /// carries `h_g1`, which is a digest of the same data through the MSM.
    pub fn build(
        backend: &'static str,
        n_vars: usize,
        n_public: usize,
        domain_size: usize,
        witness: &[Fr],
        h: Option<&[Fr]>,
        m: &MsmOutputs,
        proof: &Proof,
    ) -> Self {
        let public = &witness[1..(n_public + 1).min(witness.len())];
        let (r, s) = blinders();

        let shape = vec![
            Row {
                name: "backend",
                value: backend.to_string(),
            },
            Row {
                name: "n_vars",
                value: n_vars.to_string(),
            },
            Row {
                name: "n_public",
                value: n_public.to_string(),
            },
            Row {
                name: "domain_size",
                value: domain_size.to_string(),
            },
            Row {
                name: "witness_len",
                value: witness.len().to_string(),
            },
            Row {
                name: "witness_digest",
                value: hex64(digest_fr(witness)),
            },
            Row {
                name: "public_digest",
                value: hex64(digest_fr(public)),
            },
            Row {
                name: "r",
                value: crate::json::dec(r),
            },
            Row {
                name: "s",
                value: crate::json::dec(s),
            },
        ];

        let rows = vec![
            Row {
                name: "h_len",
                value: h.map_or_else(|| "-".into(), |v| v.len().to_string()),
            },
            Row {
                name: "h_digest",
                value: h.map_or_else(|| "-".into(), |v| hex64(digest_fr(v))),
            },
            Row {
                name: "msm_a_g1",
                value: g1(m.a_g1),
            },
            Row {
                name: "msm_b_g2",
                value: g2(m.b_g2),
            },
            Row {
                name: "msm_b_g1",
                value: g1(m.b_g1),
            },
            Row {
                name: "msm_l_g1",
                value: g1(m.l_g1),
            },
            Row {
                name: "msm_h_g1",
                value: g1(m.h_g1),
            },
            Row {
                name: "proof_a",
                value: affine1(&proof.a),
            },
            Row {
                name: "proof_b",
                value: affine2(&proof.b),
            },
            Row {
                name: "proof_c",
                value: affine1(&proof.c),
            },
        ];

        Trace {
            backend,
            shape,
            rows,
            h_chunks: h.map(chunk_digests),
        }
    }

    /// The rows that must match between two correct machines, in stage order. The first one
    /// that differs is the lead.
    pub fn rows(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.rows.iter().map(|r| (r.name, r.value.as_str()))
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Plain text, one `name value` per line, which is what `diff` wants. The shape block
    /// comes first and is separated by a blank line: a mismatch there means the two runs were
    /// not given the same problem and nothing below it means anything.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for r in &self.shape {
            out.push_str(&format!("{:<16} {}\n", r.name, r.value));
        }
        out.push('\n');
        for r in &self.rows {
            out.push_str(&format!("{:<16} {}\n", r.name, r.value));
        }
        if let Some(chunks) = &self.h_chunks {
            out.push('\n');
            for (i, d) in chunks.iter().enumerate() {
                out.push_str(&format!("h_chunk[{i:02}]      {}\n", hex64(*d)));
            }
        }
        out
    }

    /// The same thing as JSON, for the browser, which posts it rather than printing it.
    /// Hand-written in the order above for the reason in [`crate::json`]: a sorted map makes
    /// the diff unreadable.
    pub fn to_json(&self) -> String {
        let member = |r: &Row| format!("\"{}\":{}", r.name, quote(&r.value));
        let shape: Vec<String> = self.shape.iter().map(member).collect();
        let rows: Vec<String> = self.rows.iter().map(member).collect();
        let chunks = match &self.h_chunks {
            Some(c) => {
                let items: Vec<String> = c.iter().map(|d| format!("\"{}\"", hex64(*d))).collect();
                format!("[{}]", items.join(","))
            }
            None => "null".into(),
        };
        format!(
            "{{\"shape\":{{{}}},\"rows\":{{{}}},\"h_chunks\":{}}}",
            shape.join(","),
            rows.join(","),
            chunks
        )
    }
}

/// The rows where two traces disagree, as `(name, left, right)`. Empty means the two runs
/// computed the same proof.
pub fn diff<'a>(a: &'a Trace, b: &'a Trace) -> Vec<(&'static str, &'a str, &'a str)> {
    a.rows()
        .zip(b.rows())
        .filter(|((_, x), (_, y))| x != y)
        .map(|((name, x), (_, y))| (name, x, y))
        .collect()
}

fn chunk_digests(h: &[Fr]) -> Vec<u64> {
    if h.is_empty() {
        return Vec::new();
    }
    let per = h.len().div_ceil(H_CHUNKS);
    h.chunks(per).map(digest_fr).collect()
}

fn hex64(x: u64) -> String {
    format!("{x:016x}")
}

/// JSON string escaping is not needed: every value here is decimal digits, hex, or one of a
/// handful of fixed words. Asserting that rather than pulling in an escaper keeps the writer
/// as simple as the one in [`crate::json`].
fn quote(v: &str) -> String {
    debug_assert!(
        v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b" ,.-_\"".contains(&b)),
        "trace value {v:?} needs escaping"
    );
    format!("\"{}\"", v.replace('"', "'"))
}

fn g1(p: G1Projective) -> String {
    affine1(&p.into_affine())
}

fn g2(p: G2Projective) -> String {
    affine2(&p.into_affine())
}

fn affine1(p: &G1Affine) -> String {
    match p.xy() {
        Some((x, y)) => format!("{} {}", crate::json::dec(x), crate::json::dec(y)),
        None => "infinity".into(),
    }
}

fn affine2(p: &G2Affine) -> String {
    match p.xy() {
        Some((x, y)) => format!(
            "{} {} {} {}",
            crate::json::dec(x.c0),
            crate::json::dec(x.c1),
            crate::json::dec(y.c0),
            crate::json::dec(y.c1)
        ),
        None => "infinity".into(),
    }
}
