//! Peak resident set, sampled the way the upstream `measure_mem_avg.sh` samples it.
//!
//! Not `getrusage` from inside the prover: `ru_maxrss` is a high-water mark for the whole
//! process, so a reading taken in-process includes whatever the benchmark driver was
//! holding, and on this driver that is every earlier variant's parsed key. The number has
//! to come from a process that does one proof and exits, which is `g16-csp-mem`.
//!
//! Upstream averages ten samples rather than taking the maximum. That is theirs to
//! justify; we copy it so the two columns mean the same thing.

use crate::{Backend, Variant};
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::process::Command;

/// Mean peak RSS in bytes over `reps` runs of `bin`.
pub fn sample(
    bin: &Path,
    variant: Variant,
    backend: Backend,
    artifacts: &Path,
    reps: usize,
) -> Result<usize> {
    anyhow::ensure!(reps > 0, "--mem-reps must be at least 1");
    let mut total = 0usize;
    for _ in 0..reps {
        total += one(bin, variant, backend, artifacts)?;
    }
    Ok(total / reps)
}

#[cfg(target_os = "macos")]
const TIME_ARGS: [&str; 1] = ["-l"];
#[cfg(not(target_os = "macos"))]
const TIME_ARGS: [&str; 1] = ["-v"];

/// GNU time reports `ru_maxrss` in kibibytes; the BSD one macOS ships reports bytes.
#[cfg(target_os = "macos")]
const RSS_SCALE: usize = 1;
#[cfg(not(target_os = "macos"))]
const RSS_SCALE: usize = 1024;

fn one(bin: &Path, variant: Variant, backend: Backend, artifacts: &Path) -> Result<usize> {
    let out = Command::new("/usr/bin/time")
        .args(TIME_ARGS)
        .arg(bin)
        .args(["--target", variant.target.as_str()])
        .args(["--input-size", &variant.input_size.to_string()])
        .args(["--backend", backend.as_str()])
        .arg("--artifacts")
        .arg(artifacts)
        .output()
        .with_context(|| format!("running /usr/bin/time on {}", bin.display()))?;

    // /usr/bin/time writes its report to stderr, and so does anything the child logged.
    let report = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        return Err(anyhow!(
            "{} exited with {}: {report}",
            variant.name(),
            out.status
        ));
    }
    parse_rss(&report).ok_or_else(|| {
        anyhow!(
            "{}: no maximum resident set size in:\n{report}",
            variant.name()
        )
    })
}

/// The line reads `<bytes>  maximum resident set size` on macOS and `Maximum resident set
/// size (kbytes): <n>` on Linux, so take the one number on whichever line names it.
fn parse_rss(report: &str) -> Option<usize> {
    report
        .lines()
        .find(|l| l.to_ascii_lowercase().contains("maximum resident set size"))
        .and_then(|l| l.split_whitespace().find_map(|t| t.parse::<usize>().ok()))
        .map(|n| n * RSS_SCALE)
}

#[cfg(test)]
mod tests {
    use super::parse_rss;

    #[test]
    fn reads_both_time_dialects() {
        let bsd = "        0.53 real         0.41 user\n    268435456  maximum resident set size\n      1234  peak memory footprint\n";
        assert_eq!(parse_rss(bsd), Some(268435456 * super::RSS_SCALE));
        let gnu = "\tMaximum resident set size (kbytes): 262144\n\tExit status: 0\n";
        assert_eq!(parse_rss(gnu), Some(262144 * super::RSS_SCALE));
    }

    /// The child's own stdout/stderr lands in the same buffer, so a line that merely
    /// mentions the phrase must not be mistaken for the report.
    #[test]
    fn ignores_a_line_with_no_number() {
        assert_eq!(parse_rss("maximum resident set size\n"), None);
    }
}
