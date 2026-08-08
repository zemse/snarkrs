//! Discovery of benchmark artifacts: one subdirectory per circuit variant.
//!
//! A variant directory is only usable if it has all four of `circuit.zkey`,
//! `circuit.wtns`, `vkey.json` and `public.json`. Half-generated directories are skipped
//! rather than reported as failures, because `gen-artifacts.sh` leaves them behind when a
//! circuit fails to compile and a benchmark that dies on one of those is useless.

use std::path::{Path, PathBuf};

pub struct Variant {
    pub name: String,
    pub dir: PathBuf,
}

pub const REQUIRED: [&str; 4] = ["circuit.zkey", "circuit.wtns", "vkey.json", "public.json"];

impl Variant {
    pub fn zkey(&self) -> PathBuf {
        self.dir.join("circuit.zkey")
    }
    pub fn wtns(&self) -> PathBuf {
        self.dir.join("circuit.wtns")
    }
    pub fn vkey(&self) -> PathBuf {
        self.dir.join("vkey.json")
    }

    /// Constraint count from snarkjs' `r1cs-info.txt`, or -1 when it is missing.
    ///
    /// That file is snarkjs' terminal output verbatim, ANSI colour escapes included, so
    /// the count is taken as the last whitespace-separated token of the matching line
    /// rather than by parsing the line's structure. -1 is the same "unknown" marker
    /// `run-comparison.py` writes, so the two agree in a merged CSV.
    pub fn constraints(&self) -> i64 {
        let Ok(text) = std::fs::read_to_string(self.dir.join("r1cs-info.txt")) else {
            return -1;
        };
        text.lines()
            .find(|l| l.contains("# of Constraints"))
            .and_then(|l| l.split_whitespace().last())
            .and_then(|t| t.parse().ok())
            .unwrap_or(-1)
    }
}

/// Every complete variant under `root`, sorted by name. A missing or unreadable `root`
/// yields an empty list; callers decide whether that is an error.
pub fn variants(root: impl AsRef<Path>) -> Vec<Variant> {
    let Ok(entries) = std::fs::read_dir(root.as_ref()) else {
        return Vec::new();
    };
    let mut out: Vec<Variant> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.is_dir() && REQUIRED.iter().all(|f| d.join(f).is_file()))
        .map(|dir| Variant {
            name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// `variants`, filtered to `wanted` (all of them when `wanted` is empty). Naming a
/// variant that does not exist is an error: silently benchmarking nothing is how a
/// typo'd `--variant` turns into an empty CSV nobody notices.
pub fn selected(root: impl AsRef<Path>, wanted: &[String]) -> anyhow::Result<Vec<Variant>> {
    let found = variants(root.as_ref());
    if wanted.is_empty() {
        if found.is_empty() {
            anyhow::bail!(
                "no complete artifact directories under {} (each needs {})",
                root.as_ref().display(),
                REQUIRED.join(", ")
            );
        }
        return Ok(found);
    }
    let mut out = Vec::new();
    for name in wanted {
        match found.iter().position(|v| &v.name == name) {
            Some(i) => out.push(found[i].dir.clone()),
            None => anyhow::bail!(
                "variant {name:?} not found under {} (have: {})",
                root.as_ref().display(),
                found
                    .iter()
                    .map(|v| v.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    Ok(out
        .into_iter()
        .map(|dir| Variant {
            name: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            dir,
        })
        .collect())
}
