//! Metal implementation of `g16_core::Backend`.

use g16_core::{Backend, MsmOutputs, PreparedCircuit, ProveError, StageTimings};
use g16_field::Fr;
use g16_zkey::ProvingKey;

pub struct MetalBackend {
    // device, command queue, compiled pipeline states
}

impl MetalBackend {
    /// Picks the system default device and compiles every kernel from source.
    /// Fails on a machine with no Metal device rather than silently falling back to CPU:
    /// a benchmark that quietly measures the wrong backend is worse than an error.
    pub fn new() -> Result<Self, ProveError> {
        todo!("g16-metal: implement MetalBackend::new")
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }
    fn prepare(&self, _pk: ProvingKey) -> Result<Box<dyn PreparedCircuit>, ProveError> {
        todo!("g16-metal: implement MetalBackend::prepare")
    }
}

pub struct MetalCircuit {
    // resident buffers: bases, CSR coefficients, twiddles; scratch; pipeline states
}
