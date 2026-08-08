//! Number-theoretic transforms over `Fr`, and the coset machinery for Jordi's trick.
//!
//! The prover needs `6 * NTT_n` per proof: three iNTTs to take the A/B/C evaluations on
//! the domain into coefficient form, three forward NTTs to re-evaluate them on a disjoint
//! coset. The coset shift between them is a pure elementwise map and is exposed
//! separately so a backend can fuse it into an NTT epilogue.

use g16_field::{Domain, Fr};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Forward,
    Inverse,
}

/// The transform primitives a backend must provide. Deliberately slice-shaped: the CPU
/// backend operates in place on host memory, and the Metal backend implements this trait
/// only for testing kernels in isolation. The *production* GPU path does not go through
/// this trait, because round-tripping every transform across the boundary would defeat
/// the point; it implements [`g16_core::PreparedCircuit::compute_h`] directly and keeps
/// the three vectors device-resident across all six transforms.
pub trait NttBackend: Send + Sync {
    fn name(&self) -> &'static str;
    /// In-place radix-2 NTT. `a.len()` must equal `domain.size`.
    fn ntt(&self, domain: &Domain, a: &mut [Fr], dir: Direction);
    /// In-place `a[i] *= shift^i`.
    fn distribute_powers(&self, a: &mut [Fr], shift: Fr);
}

/// Multi-threaded CPU NTT.
pub struct CpuNtt {
    pub threads: usize,
}

impl CpuNtt {
    pub fn new() -> Self {
        todo!("g16-ntt: implement CpuNtt::new")
    }
}

impl NttBackend for CpuNtt {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn ntt(&self, _domain: &Domain, _a: &mut [Fr], _dir: Direction) {
        todo!("g16-ntt: implement CpuNtt::ntt")
    }
    fn distribute_powers(&self, _a: &mut [Fr], _shift: Fr) {
        todo!("g16-ntt: implement CpuNtt::distribute_powers")
    }
}
