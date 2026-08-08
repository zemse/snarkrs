//! The CPU backend: composes `g16-ntt` and `g16-msm` into stages 0-9.

use crate::{Backend, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::*;
use g16_zkey::ProvingKey;

pub struct CpuBackend {
    pub threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        todo!("g16-core::cpu: implement CpuBackend::new")
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn prepare(&self, _pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        todo!("g16-core::cpu: implement CpuBackend::prepare")
    }
}

pub struct CpuCircuit {
    // pk, domain, twiddles, thread pool
}
