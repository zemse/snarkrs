//! The log lines snarkjs' informational commands print, and the exact bytes its logger
//! wraps them in.
//!
//! `r1cs info`, `r1cs print` and `wtns check` produce nothing but log lines: their whole
//! output is `logger.info(...)` and `logger.warn(...)` calls on the one logplease logger
//! `cli.js:50` creates, `Logger.create("snarkJS", {showTimestamp: false})`. The commands
//! here therefore take a [`SnarkjsLog`] and hand it `(level, message)`, so the message
//! text is this crate's to get right and the framing is the caller's to choose.
//!
//! [`Logplease`] is that framing, byte for byte (`logplease/src/index.js:107-218`, 1.2.15).
//! Under Node every line is coloured whether or not stdout is a terminal, goes to
//! **stdout** through `console.log` even at `WARN` and `ERROR`, and reads
//!
//! ```text
//! ESC[3{c};22m[LEVEL]{pad} ESC[39;1msnarkJS ESC[0m: message
//! ```
//!
//! with no space before the second escape, `c` the level's colour (cyan, green, yellow,
//! red for debug, info, warn, error) and `pad` one extra space for the two four-letter
//! levels, so the category lines up. The global level defaults to `INFO` (`cli.js:51`) and
//! `-v` lowers it to `DEBUG`; the `LOG` environment variable overrides both, which is left
//! to the caller.

use std::io::{self, Write};

/// logplease's levels, lowest first. Ordering is the filter: a line is printed when its
/// level is at or above the logger's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    /// `loglevelColors` (`index.js:56`), indexed by level.
    fn colour(self) -> u8 {
        match self {
            Self::Debug => 6,
            Self::Info => 2,
            Self::Warn => 3,
            Self::Error => 1,
        }
    }
}

/// Where a command's log lines go.
pub trait SnarkjsLog {
    fn log(&mut self, level: Level, msg: &str) -> io::Result<()>;

    fn info(&mut self, msg: &str) -> io::Result<()> {
        self.log(Level::Info, msg)
    }

    fn warn(&mut self, msg: &str) -> io::Result<()> {
        self.log(Level::Warn, msg)
    }

    fn debug(&mut self, msg: &str) -> io::Result<()> {
        self.log(Level::Debug, msg)
    }
}

/// Collects `(level, message)` pairs, for a caller that wants the lines rather than text.
impl SnarkjsLog for Vec<(Level, String)> {
    fn log(&mut self, level: Level, msg: &str) -> io::Result<()> {
        self.push((level, msg.to_string()));
        Ok(())
    }
}

/// snarkjs' own logger: coloured lines at or above `level`, onto one writer.
pub struct Logplease<W: Write> {
    pub out: W,
    pub level: Level,
}

impl<W: Write> Logplease<W> {
    /// The CLI's logger at its default `INFO` level.
    pub fn new(out: W) -> Self {
        Self {
            out,
            level: Level::Info,
        }
    }
}

impl<W: Write> SnarkjsLog for Logplease<W> {
    fn log(&mut self, level: Level, msg: &str) -> io::Result<()> {
        if level < self.level {
            return Ok(());
        }
        let pad = if matches!(level, Level::Info | Level::Warn) {
            " "
        } else {
            ""
        };
        writeln!(
            self.out,
            "\x1b[3{};22m[{}]{pad} \x1b[39;1msnarkJS\x1b[0m: {msg}",
            level.colour(),
            level.name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from `snarkjs r1cs info` and `wtns check` output.
    #[test]
    fn frames_like_logplease() {
        let mut log = Logplease::new(Vec::new());
        log.info("Curve: bn-128").unwrap();
        log.warn("WITNESS IS NOT CORRECT").unwrap();
        log.debug("hidden at INFO").unwrap();
        assert_eq!(
            String::from_utf8(log.out).unwrap(),
            "\x1b[32;22m[INFO]  \x1b[39;1msnarkJS\x1b[0m: Curve: bn-128\n\
             \x1b[33;22m[WARN]  \x1b[39;1msnarkJS\x1b[0m: WITNESS IS NOT CORRECT\n"
        );
    }
}
