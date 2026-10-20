//! The CLI's log lines, in the shape of a Rust command line tool.
//!
//! An INFO line is the bare message on stdout, so a script that wants a verdict reads
//! `OK!` and nothing else. Warnings, errors and `-v` debug lines go to stderr behind a
//! `warning:`, `error:` or `debug:` label, which is coloured when stderr is a terminal and
//! `NO_COLOR` is unset.

use std::fmt::Display;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// `-v`: debug lines print too. INFO and above print regardless.
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
    /// The label and its SGR colour: cyan, yellow and red, bold. INFO has none.
    fn label(self) -> Option<(&'static str, &'static str)> {
        match self {
            Level::Debug => Some(("debug", "36")),
            Level::Info => None,
            Level::Warn => Some(("warning", "33")),
            Level::Error => Some(("error", "31")),
        }
    }
}

/// One line as printed, without the newline.
pub fn line(level: Level, msg: &str, colour: bool) -> String {
    match level.label() {
        None => msg.to_string(),
        Some((name, sgr)) if colour => format!("\x1b[{sgr};1m{name}:\x1b[0m {msg}"),
        Some((name, _)) => format!("{name}: {msg}"),
    }
}

fn write(level: Level, msg: &str) {
    if level == Level::Debug && !verbose() {
        return;
    }
    // A closed pipe is not worth a panic: the verdict is in the exit code as well.
    if level == Level::Info {
        let _ = writeln!(std::io::stdout().lock(), "{msg}");
        return;
    }
    let err = std::io::stderr();
    let colour = err.is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let _ = writeln!(err.lock(), "{}", line(level, msg, colour));
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

/// Printed after a witness is computed: the file round trip is what a Rust caller skips.
pub const WITNESS_TIP: &str = "for faster witness generation, rewrite it in optimised Rust and \
use snarkrs as a library:\n     pass the witness from memory to snarkrs_groth16::prove::prove, which \
takes &[Fr], and skip writing\n     and re-reading the .wtns. See \"witness from memory\" in \
README.md. SNARKRS_NO_TIPS=1 hides this.";

/// A tip for whoever is at the terminal. On stderr, so stdout keeps only the result, and
/// off when `SNARKRS_NO_TIPS` is set to anything but `0` or nothing.
pub fn tip(msg: &str) {
    if std::env::var_os("SNARKRS_NO_TIPS").is_some_and(|v| !v.is_empty() && v != "0") {
        return;
    }
    let err = std::io::stderr();
    let colour = err.is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let label = if colour {
        "\x1b[36;1mtip:\x1b[0m"
    } else {
        "tip:"
    };
    let _ = writeln!(err.lock(), "{label} {msg}");
}

pub fn tip_witness() {
    tip(WITNESS_TIP);
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
    fn plain_lines_are_bare_info_and_labelled_otherwise() {
        assert_eq!(line(Level::Info, "OK!", false), "OK!");
        assert_eq!(
            line(Level::Error, "Invalid proof", false),
            "error: Invalid proof"
        );
        assert_eq!(line(Level::Warn, "w", false), "warning: w");
        assert_eq!(line(Level::Debug, "d", false), "debug: d");
    }

    #[test]
    fn colour_is_on_the_label_only() {
        assert_eq!(line(Level::Info, "OK!", true), "OK!");
        assert_eq!(
            line(Level::Error, "Invalid proof", true),
            "\x1b[31;1merror:\x1b[0m Invalid proof"
        );
        assert_eq!(line(Level::Warn, "w", true), "\x1b[33;1mwarning:\x1b[0m w");
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
