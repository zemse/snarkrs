//! circom's `.sym`, the signal-name table `r1cs print` labels constraints with.
//!
//! One line per signal, `labelIdx,varIdx,componentIdx,name`. `varIdx` is the witness
//! position and is `-1` for a signal the optimiser removed. Several labels can share one
//! wire, which is how an alias looks after simplification.
//!
//! Loaded the way `loadsyms.js:22-54` loads it, which is looser than the format:
//!
//! * The text is split on `\n` alone, so a CRLF file keeps its `\r` on every name, and
//!   decoded with `TextDecoder("utf-8")`, which substitutes invalid sequences and drops a
//!   leading byte-order mark.
//! * A line that does not split into **exactly four** fields on `,` is skipped, so a name
//!   containing a comma silently disappears.
//! * `varIdx` is used as a **string** key into a JS array. `"5"` is index 5, but `"-1"`
//!   and `"05"` are plain properties that `r1cs print`, which looks wires up by their
//!   canonical decimal, never finds. So this keeps the text as written.
//! * A wire that already has a non-empty name gets `"|" + name` appended; an empty first
//!   name is falsy and is replaced instead.
//! * Wire 0 is preset to `"one"`, and a later line for wire 0 appends to it.
//!
//! Only the wire map is kept. `labelIdx2Name` and `componentIdx2Name` are built by snarkjs
//! and read by nothing in the commands this crate backs.

use std::collections::HashMap;
use std::path::Path;

use crate::CeremonyError;

/// `sym.varIdx2Name`, keyed by the `varIdx` field exactly as written.
#[derive(Clone, Debug)]
pub struct Syms {
    var_names: HashMap<String, String>,
}

impl Syms {
    /// `loadSyms(path)`.
    pub fn load(path: &Path) -> Result<Self, CeremonyError> {
        Ok(Self::parse(&String::from_utf8_lossy(&std::fs::read(path)?)))
    }

    /// [`Syms::load`] on text already decoded.
    pub fn parse(text: &str) -> Self {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let mut var_names = HashMap::new();
        var_names.insert("0".to_string(), "one".to_string());
        for line in text.split('\n') {
            let fields: Vec<&str> = line.split(',').collect();
            let [_label, var, _component, name] = fields[..] else {
                continue;
            };
            match var_names.get_mut(var) {
                Some(existing) if !existing.is_empty() => {
                    existing.push('|');
                    existing.push_str(name);
                }
                _ => {
                    var_names.insert(var.to_string(), name.to_string());
                }
            }
        }
        Self { var_names }
    }

    /// The name of witness wire `var`, or `None` where snarkjs would read `undefined`.
    pub fn var_name(&self, var: u32) -> Option<&str> {
        self.var_names.get(&var.to_string()).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_loadsyms() {
        let s = Syms::parse(
            "\u{feff}1,1,0,main.out\n2,2,0,main.a\r\n3,2,0,main.alias\n4,-1,0,main.gone\n\
             5,05,0,main.padded\n6,3,0,has,comma\n7,4,0,\n8,4,0,main.second\n9,0,0,main.one",
        );
        assert_eq!(s.var_name(0), Some("one|main.one"));
        assert_eq!(s.var_name(1), Some("main.out"));
        assert_eq!(s.var_name(2), Some("main.a\r|main.alias"));
        assert_eq!(s.var_name(3), None);
        assert_eq!(s.var_name(4), Some("main.second"));
        assert_eq!(s.var_name(5), None);
    }
}
