//! snarkjs' log lines: logplease's format under the `snarkJS` category, all on stdout.
//!
//! The category is snarkjs' own spelling rather than ours, so a script that greps
//! `snarkJS: OK!` or `[ERROR] snarkJS: Invalid proof` out of a snarkjs run reads the same
//! line from this one. logplease writes every level with `console.log`, errors included,
//! and so does this.
//!
//! One deliberate difference: logplease colours unconditionally under Node, which puts
//! escape codes between `[INFO]` and `snarkJS` in a piped log and breaks exactly those
//! greps. Here the colours follow the terminal, and `NO_COLOR` turns them off.

use std::fmt::Display;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// snarkjs' `-v`: `Logger.setLogLevel("DEBUG")`. INFO and above print regardless.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    fn name(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    /// logplease's `loglevelColors`: cyan, green, yellow, red.
    fn colour(self) -> u8 {
        match self {
            Level::Debug => 6,
            Level::Info => 2,
            Level::Warn => 3,
            Level::Error => 1,
        }
    }
}

/// One line as logplease renders it. INFO and WARN are padded by a space so the category
/// starts in the same column as after `[DEBUG]` and `[ERROR]`.
pub fn line(level: Level, msg: &str, colour: bool) -> String {
    let pad = if matches!(level, Level::Info | Level::Warn) {
        " "
    } else {
        ""
    };
    if colour {
        format!(
            "\x1b[3{};22m[{}]{pad} \x1b[39;1msnarkJS\x1b[0m: {msg}",
            level.colour(),
            level.name()
        )
    } else {
        format!("[{}]{pad} snarkJS: {msg}", level.name())
    }
}

fn write(level: Level, msg: &str) {
    if level == Level::Debug && !verbose() {
        return;
    }
    let out = std::io::stdout();
    let colour = out.is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let mut out = out.lock();
    // A closed pipe is not worth a panic: the verdict is in the exit code as well.
    let _ = writeln!(out, "{}", line(level, msg, colour));
}

pub fn debug(msg: impl Display) {
    write(Level::Debug, &msg.to_string());
}

pub fn info(msg: impl Display) {
    write(Level::Info, &msg.to_string());
}

pub fn warn(msg: impl Display) {
    write(Level::Warn, &msg.to_string());
}

pub fn error(msg: impl Display) {
    write(Level::Error, &msg.to_string());
}

/// `misc.formatHash`: the title, then four tab-indented rows of four big-endian 32-bit
/// words. Every ceremony hash snarkjs prints goes through it, and a contributor copies the
/// result into a public attestation, so the layout is part of the interface.
pub fn format_hash(hash: &[u8], title: &str) -> String {
    let mut s = String::new();
    for (i, row) in hash.chunks(16).take(4).enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str("\t\t");
        for (j, word) in row.chunks(4).enumerate() {
            if j > 0 {
                s.push(' ');
            }
            s.push_str(&hex::encode(word));
        }
    }
    if title.is_empty() {
        s
    } else {
        format!("{title}\n{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_lines_match_logplease_without_colour() {
        assert_eq!(line(Level::Info, "OK!", false), "[INFO]  snarkJS: OK!");
        assert_eq!(
            line(Level::Error, "Invalid proof", false),
            "[ERROR] snarkJS: Invalid proof"
        );
        assert_eq!(line(Level::Warn, "w", false), "[WARN]  snarkJS: w");
        assert_eq!(line(Level::Debug, "d", false), "[DEBUG] snarkJS: d");
    }

    /// Byte for byte what `snarkjs groth16 verify` 0.7.6 wrote to a pipe.
    #[test]
    fn coloured_lines_match_logplease_byte_for_byte() {
        assert_eq!(
            line(Level::Info, "OK!", true),
            "\x1b[32;22m[INFO]  \x1b[39;1msnarkJS\x1b[0m: OK!"
        );
        assert_eq!(
            line(Level::Error, "Invalid proof", true),
            "\x1b[31;22m[ERROR] \x1b[39;1msnarkJS\x1b[0m: Invalid proof"
        );
    }

    /// `snarkjs ptn bn128 4`'s first challenge hash, as it printed it.
    #[test]
    fn hashes_print_as_four_rows_of_four_words() {
        let h = hex::decode(
            "2054432085403180e1678602c83562f1f4ddefafb4b9e7171b53070455a4cc6d\
             b11b2e5bfe5e89c0cb9ab4a7b3b9fd5a17bad62ad5ba013c34e7dd2fbb4f143b",
        )
        .unwrap();
        assert_eq!(
            format_hash(&h, "First Contribution Hash:"),
            "First Contribution Hash:\n\
             \t\t20544320 85403180 e1678602 c83562f1\n\
             \t\tf4ddefaf b4b9e717 1b530704 55a4cc6d\n\
             \t\tb11b2e5b fe5e89c0 cb9ab4a7 b3b9fd5a\n\
             \t\t17bad62a d5ba013c 34e7dd2f bb4f143b"
        );
    }
}
