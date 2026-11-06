//! gnark's `VerifyingKey.WriteTo` and `WriteRawTo` (backend/groth16/bn254/marshal.go):
//!
//! ```text
//!   encoder points: G1.Alpha, G1.Beta, G2.Beta, G2.Gamma, G1.Delta, G2.Delta
//!   []G1Affine G1.K: u32 BE count, then the points
//!   [][]uint64 PublicAndCommitmentCommitted: u32 BE count, each a u32 BE count of u64 BE
//!   u32 BE len(CommitmentKeys), then the commitment keys
//! ```
//!
//! `WriteTo` compresses every point and `WriteRawTo` none; each point carries its own flag,
//! so one reader takes both. `G1.K` is one base per public wire, the constant wire first,
//! which is snarkjs' IC.

use crate::{check_header_g1, check_header_g2, GnarkError, Reader, G1_COMPRESSED};
use snarkrs_field::*;
use snarkrs_formats::VerifyingKey;

/// A gnark verifying key without commitments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GnarkVerifyingKey {
    pub alpha_g1: G1Affine,
    /// Unused by the verifier; gnark keeps it "for compatibility purposes". Checked against
    /// the proving key's.
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    /// Unused by the verifier, like `beta_g1`.
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,
    /// `G1.K`: `[(beta u_i + alpha v_i + w_i) / gamma]_1` for each public wire.
    pub k: Vec<G1Affine>,
}

impl GnarkVerifyingKey {
    /// Read a `WriteTo` or `WriteRawTo` file, every point checked on the curve and in the
    /// subgroup as gnark's own decoder checks them.
    pub fn read(path: &std::path::Path) -> Result<Self, GnarkError> {
        Self::from_bytes(&crate::map(path)?)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, GnarkError> {
        let mut r = Reader::new(data, "verifying key");
        let alpha_g1 = r.g1("G1.Alpha")?;
        let beta_g1 = r.g1("G1.Beta")?;
        let beta_g2 = r.g2("G2.Beta")?;
        let gamma_g2 = r.g2("G2.Gamma")?;
        let delta_g1 = r.g1("G1.Delta")?;
        let delta_g2 = r.g2("G2.Delta")?;
        for (p, what) in [
            (&alpha_g1, "G1.Alpha"),
            (&beta_g1, "G1.Beta"),
            (&delta_g1, "G1.Delta"),
        ] {
            check_header_g1(p, "verifying key", what)?;
        }
        for (p, what) in [
            (&beta_g2, "G2.Beta"),
            (&gamma_g2, "G2.Gamma"),
            (&delta_g2, "G2.Delta"),
        ] {
            check_header_g2(p, "verifying key", what)?;
        }

        r.part("G1.K");
        let n_k = r.u32_be()?;
        // The smallest a point can be encoded is 32 bytes, so the file bounds the count
        // before the vector is reserved.
        if (n_k as usize).saturating_mul(G1_COMPRESSED) > r.remaining() {
            return Err(r.bad(format!(
                "{n_k} points cannot fit in the {} bytes left",
                r.remaining()
            )));
        }
        let mut k = Vec::with_capacity(n_k as usize);
        for i in 0..n_k {
            let p = r.g1(&format!("point {i}"))?;
            k.push(p);
        }

        r.part("PublicAndCommitmentCommitted");
        let n_committed = r.u32_be()?;
        if n_committed != 0 {
            return Err(GnarkError::Commitments {
                part: "verifying key",
                count: n_committed.into(),
            });
        }
        r.part("commitment count");
        let commitments = r.u32_be()?;
        if commitments != 0 {
            return Err(GnarkError::Commitments {
                part: "verifying key",
                count: commitments.into(),
            });
        }
        r.part("trailer");
        r.finish()?;

        let vk = Self {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            k,
        };
        // The same structural refusals the zkey path makes: an empty or infinity-laden IC,
        // and `gamma == delta`, which is a key anyone can forge against.
        vk.to_verifying_key()
            .check_structure()
            .map_err(|e| GnarkError::Malformed {
                part: "verifying key",
                reason: e.to_string(),
            })?;
        Ok(vk)
    }

    /// Public inputs this key takes, the constant wire not counted.
    pub fn n_public(&self) -> usize {
        self.k.len().saturating_sub(1)
    }

    /// snarkrs' verifying key: gnark's `G1.K` is snarkjs' IC, constant wire first.
    pub fn to_verifying_key(&self) -> VerifyingKey {
        VerifyingKey {
            alpha_g1: self.alpha_g1,
            beta_g2: self.beta_g2,
            gamma_g2: self.gamma_g2,
            delta_g2: self.delta_g2,
            ic: self.k.clone(),
        }
    }
}
