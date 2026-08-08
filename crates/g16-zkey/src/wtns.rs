//! `.wtns` witness file parsing. Little-endian, Montgomery-form field elements.

use g16_field::Fr;

/// The full witness vector `w = (1, public..., private...)`, length `n_vars`.
pub struct Witness(pub Vec<Fr>);

impl Witness {
    pub fn load(path: &std::path::Path) -> Result<Self, super::ZkeyError> {
        todo!("g16-zkey::wtns: implement Witness::load")
    }
}
