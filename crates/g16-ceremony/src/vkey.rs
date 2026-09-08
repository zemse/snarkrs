//! `zkey export verificationkey`: the four verifier points, the precomputed pairing, and
//! IC, as snarkjs' `verification_key.json`.
//!
//! `zkey_export_verificationkey.js:55-89` builds the object and `cli.js:589-601` writes it
//! with one space of indentation per level. Key order is JS object insertion order, so
//! `IC` is last because `vKey.IC` is assigned after the literal. `vk_beta_1` and
//! `vk_delta_1` are **not** exported: only the four points a verifier needs, plus IC.
//!
//! Three encoding details, none of them the same as anything on disk:
//!
//! * Every coordinate is a **decimal string** in normal, non-Montgomery form, and every
//!   point is in projective triple form, so G1 is `[x, y, "1"]` and G2 is
//!   `[[x0, x1], [y0, y1], ["1", "0"]]` (`wasm_curve.js:338-356`).
//! * G2 nests `c0` first, the opposite of the `c1`-first order the same points take when
//!   they enter a transcript hash.
//! * `nPublic` stays a JSON **number**. `stringifyBigInts` returns non-bigint values
//!   unchanged, so it is the one field that is not a string.
//!
//! `vk_alphabeta_12` is `e(alpha_1, beta_2)` in `F12`, nested 2 x 3 x 2 for twelve decimal
//! strings (`engine.js:30-33`). It is redundant, a precomputed pairing verifiers may use.
//!
//! This crate already reads a `verification_key.json` on the proving side
//! ([`g16_zkey::VerifyingKey::from_json`]), so the two must agree on field names. If we
//! ever want a byte-identical file, diff against a real snarkjs run rather than guessing
//! `bfj`'s newline and trailing-byte behaviour: nothing downstream cares about the
//! whitespace, so it is not worth guessing at.

use std::path::Path;

use crate::CeremonyError;

/// Build the JSON object from a zkey's section 2 and section 3.
pub fn verification_key_json(zkey: &Path) -> Result<serde_json::Value, CeremonyError> {
    let _ = zkey;
    todo!("verification_key_json")
}

/// [`verification_key_json`] written to `out`.
pub fn export_verification_key(zkey: &Path, out: &Path) -> Result<(), CeremonyError> {
    let _ = (zkey, out);
    todo!("export_verification_key")
}
