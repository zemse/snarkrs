//! snarkjs' command line, rewritten into one clap can parse.
//!
//! snarkjs 0.7.6 does not parse its command line the way clap does (`src/clprocessor.js`),
//! and a drop-in has to accept every line snarkjs accepts:
//!
//! - Every token that starts with `-` is an option, wherever it is on the line. Leading
//!   dashes are stripped however many there are, and `key=value` splits at the first `=`.
//!   Anything else is a word.
//! - The leading words name the command, compared case-insensitively against the full
//!   command and each of its aliases, and the words after them are positional parameters.
//!   The first command in table order with a matching alias wins.
//! - Options have short forms (`-v`, `-e`, `-n`) that the long names stand for.
//!
//! [`normalise`] does that part and hands clap a line in its own dialect: the canonical
//! lower-case command words, the positionals, then every option as `--key` or
//! `--key=value`. Defaults, help and the count of positionals are clap's from there on.
//!
//! Our own options (`--backend`, `--self-verify`, ...) predate this and take their value
//! as the next token, which snarkjs' tokenizer would read as a positional. [`VALUE_OPTIONS`]
//! is the list of those, and `-backend=metal` works as well.

use std::ffi::OsString;

/// Whether snarkrs runs a snarkjs command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Yes,
    /// Not yet. Groth16 on BN254, so it belongs here eventually.
    Later,
    /// A different proving system. snarkrs is Groth16 only.
    Never,
}

/// One entry of snarkjs' `commands` table (`cli.js:55-345`).
#[derive(Debug, PartialEq, Eq)]
pub struct Command {
    /// The usage line exactly as snarkjs writes it: command words, then `<required>` and
    /// `[optional]` parameters.
    pub cmd: &'static str,
    pub alias: &'static [&'static str],
    pub description: &'static str,
    pub support: Support,
}

impl Command {
    /// The command words, the part of `cmd` before the first parameter.
    pub fn words(&self) -> Vec<&'static str> {
        command_words(self.cmd)
    }

    /// The parameter half of `cmd`, as the usage line prints it.
    pub fn params(&self) -> String {
        let n = self.words().len();
        self.cmd
            .split_whitespace()
            .skip(n)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Every word sequence that selects this command, its own first.
    fn spellings(&self) -> Vec<Vec<&'static str>> {
        std::iter::once(self.cmd)
            .chain(self.alias.iter().copied())
            .map(command_words)
            .collect()
    }
}

/// `parseLine` in `clprocessor.js`: words up to the first one that opens with `<`, `[` or
/// `-`. The legacy aliases carry option specs such as `verify -vk|verificationkey`, and this
/// is what drops them: snarkjs never reads those options, only the words in front.
fn command_words(line: &'static str) -> Vec<&'static str> {
    line.split_whitespace()
        .take_while(|w| !w.starts_with(['<', '[', '-']))
        .collect()
}

use Support::*;

/// snarkjs 0.7.6's table, in its order, with its aliases verbatim.
///
/// The one place this does not reproduce snarkjs is `zkey verify`, where 0.7.6 rejects both
/// of its own documented spellings with exit 99. `zkey verify` is an alias of the r1cs
/// form, snarkjs matches all of a command's aliases in lockstep and takes the first to run
/// out, and that is the two-word alias: `zkey verify r1cs a b c` becomes four parameters
/// with "r1cs" first, and `zkey verify init a b c` is matched by the r1cs command before
/// the init one is tried. Only `zkv`, `zkvr`, `zkvi` and the bare `zkey verify` work there.
/// Here [`find`] takes the longest matching spelling and the init form comes first, so
/// both documented lines work and every line that worked in snarkjs means the same thing.
pub const COMMANDS: &[Command] = &[
    Command {
        cmd: "powersoftau new <curve> <power> [powersoftau_0000.ptau]",
        alias: &["ptn"],
        description: "Starts a powers of tau ceremony",
        support: Yes,
    },
    Command {
        cmd: "powersoftau contribute <powersoftau.ptau> <new_powersoftau.ptau>",
        alias: &["ptc"],
        description: "creates a ptau file with a new contribution",
        support: Yes,
    },
    Command {
        cmd: "powersoftau export challenge <powersoftau_0000.ptau> [challenge]",
        alias: &["ptec"],
        description: "Creates a challenge",
        support: Yes,
    },
    Command {
        cmd: "powersoftau challenge contribute <curve> <challenge> [response]",
        alias: &["ptcc"],
        description: "Contribute to a challenge",
        support: Yes,
    },
    Command {
        cmd: "powersoftau import response <powersoftau_old.ptau> <response> <<powersoftau_new.ptau>",
        alias: &["ptir"],
        description: "import a response to a ptau file",
        support: Yes,
    },
    Command {
        cmd: "powersoftau beacon <old_powersoftau.ptau> <new_powersoftau.ptau> <beaconHash(Hex)> <numIterationsExp>",
        alias: &["ptb"],
        description: "adds a beacon",
        support: Yes,
    },
    Command {
        cmd: "powersoftau prepare phase2 <powersoftau.ptau> <new_powersoftau.ptau>",
        alias: &["pt2"],
        description: "Prepares phase 2. ",
        support: Yes,
    },
    Command {
        cmd: "powersoftau convert <old_powersoftau.ptau> <new_powersoftau.ptau>",
        alias: &["ptcv"],
        description: "Convert ptau",
        support: Later,
    },
    Command {
        cmd: "powersoftau truncate <powersoftau.ptau>",
        alias: &["ptt"],
        description: "Generate different powers of tau with smaller sizes ",
        support: Later,
    },
    Command {
        cmd: "powersoftau verify <powersoftau.ptau>",
        alias: &["ptv"],
        description: "verifies a powers of tau file",
        support: Yes,
    },
    Command {
        cmd: "powersoftau export json <powersoftau_0000.ptau> <powersoftau_0000.json>",
        alias: &["ptej"],
        description: "Exports a power of tau file to a JSON",
        support: Later,
    },
    Command {
        cmd: "r1cs info [circuit.r1cs]",
        alias: &["ri", "info -r|r1cs:circuit.r1cs"],
        description: "Print statistiscs of a circuit",
        support: Later,
    },
    Command {
        cmd: "r1cs print [circuit.r1cs] [circuit.sym]",
        alias: &["rp", "print -r|r1cs:circuit.r1cs -s|sym"],
        description: "Print the constraints of a circuit",
        support: Later,
    },
    Command {
        cmd: "r1cs export json [circuit.r1cs] [circuit.json]",
        alias: &["rej"],
        description: "Export r1cs to JSON file",
        support: Later,
    },
    Command {
        cmd: "wtns calculate [circuit.wasm] [input.json] [witness.wtns]",
        alias: &[
            "wc",
            "calculatewitness -ws|wasm:circuit.wasm -i|input:input.json -wt|witness:witness.wtns",
        ],
        description: "Caclculate specific witness of a circuit given an input",
        support: Later,
    },
    Command {
        cmd: "wtns debug [circuit.wasm] [input.json] [witness.wtns] [circuit.sym]",
        alias: &["wd"],
        description: "Calculate the witness with debug info.",
        support: Later,
    },
    Command {
        cmd: "wtns export json [witness.wtns] [witnes.json]",
        alias: &["wej"],
        description: "Calculate the witness with debug info.",
        support: Later,
    },
    Command {
        cmd: "wtns check [circuit.r1cs] [[witness.wtns]",
        alias: &["wchk"],
        description: "Check if a specific witness of a circuit fulfills the r1cs constraints",
        support: Later,
    },
    Command {
        cmd: "zkey contribute <circuit_old.zkey> <circuit_new.zkey>",
        alias: &["zkc"],
        description: "creates a zkey file with a new contribution",
        support: Yes,
    },
    Command {
        cmd: "zkey export bellman <circuit_xxxx.zkey> [circuit.mpcparams]",
        alias: &["zkeb"],
        description: "Export a zKey to a MPCParameters file compatible with kobi/phase2 (Bellman)",
        support: Yes,
    },
    Command {
        cmd: "zkey bellman contribute <curve> <circuit.mpcparams> <circuit_response.mpcparams>",
        alias: &["zkbc"],
        description: "contributes to a challenge file in bellman format",
        support: Yes,
    },
    Command {
        cmd: "zkey import bellman <circuit_old.zkey> <circuit.mpcparams> <circuit_new.zkey>",
        alias: &["zkib"],
        description: "Export a zKey to a MPCParameters file compatible with kobi/phase2 (Bellman) ",
        support: Yes,
    },
    Command {
        cmd: "zkey beacon <circuit_old.zkey> <circuit_new.zkey> <beaconHash(Hex)> <numIterationsExp>",
        alias: &["zkb"],
        description: "adds a beacon",
        support: Yes,
    },
    Command {
        cmd: "zkey verify init [circuit_0000.zkey] [powersoftau.ptau] [circuit_final.zkey]",
        alias: &["zkvi"],
        description: "Verify zkey file contributions and verify that matches with the original circuit.r1cs and ptau",
        support: Yes,
    },
    Command {
        cmd: "zkey verify r1cs [circuit.r1cs] [powersoftau.ptau] [circuit_final.zkey]",
        alias: &["zkv", "zkvr", "zkey verify"],
        description: "Verify zkey file contributions and verify that matches with the original circuit.r1cs and ptau",
        support: Yes,
    },
    Command {
        cmd: "zkey export verificationkey [circuit_final.zkey] [verification_key.json]",
        alias: &["zkev"],
        description: "Exports a verification key",
        support: Yes,
    },
    Command {
        cmd: "zkey export json [circuit_final.zkey] [circuit_final.zkey.json]",
        alias: &["zkej"],
        description: "Exports a circuit key to a JSON file",
        support: Later,
    },
    Command {
        cmd: "zkey export solidityverifier [circuit_final.zkey] [verifier.sol]",
        alias: &["zkesv", "generateverifier -vk|verificationkey -v|verifier"],
        description: "Creates a verifier in solidity",
        support: Yes,
    },
    Command {
        cmd: "zkey export soliditycalldata [public.json] [proof.json]",
        alias: &["zkesc", "generatecall -pub|public -p|proof"],
        description: "Generates call parameters ready to be called.",
        support: Later,
    },
    Command {
        cmd: "groth16 setup [circuit.r1cs] [powersoftau.ptau] [circuit_0000.zkey]",
        alias: &["g16s", "zkn", "zkey new"],
        description: "Creates an initial groth16 pkey file with zero contributions",
        support: Yes,
    },
    Command {
        cmd: "groth16 prove [circuit_final.zkey] [witness.wtns] [proof.json] [public.json]",
        alias: &[
            "g16p",
            "zpw",
            "zksnark proof",
            "proof -pk|provingkey -wt|witness -p|proof -pub|public",
        ],
        description: "Generates a zk Proof from witness",
        support: Yes,
    },
    Command {
        cmd: "groth16 fullprove [input.json] [circuit_final.wasm] [circuit_final.zkey] [proof.json] [public.json]",
        alias: &["g16f", "g16i"],
        description: "Generates a zk Proof from input",
        support: Later,
    },
    Command {
        cmd: "groth16 verify [verification_key.json] [public.json] [proof.json]",
        alias: &["g16v", "verify -vk|verificationkey -pub|public -p|proof"],
        description: "Verify a zk Proof",
        support: Yes,
    },
    Command {
        cmd: "plonk setup [circuit.r1cs] [powersoftau.ptau] [circuit.zkey]",
        alias: &["pks"],
        description: "Creates an initial PLONK pkey ",
        support: Never,
    },
    Command {
        cmd: "plonk prove [circuit.zkey] [witness.wtns] [proof.json] [public.json]",
        alias: &["pkp"],
        description: "Generates a PLONK Proof from witness",
        support: Never,
    },
    Command {
        cmd: "plonk fullprove [input.json] [circuit.wasm] [circuit.zkey] [proof.json] [public.json]",
        alias: &["pkf"],
        description: "Generates a PLONK Proof from input",
        support: Never,
    },
    Command {
        cmd: "plonk verify [verification_key.json] [public.json] [proof.json]",
        alias: &["pkv"],
        description: "Verify a PLONK Proof",
        support: Never,
    },
    Command {
        cmd: "fflonk setup [circuit.r1cs] [powersoftau.ptau] [circuit.zkey]",
        alias: &["ffs"],
        description: "BETA version. Creates a FFLONK zkey from a circuit",
        support: Never,
    },
    Command {
        cmd: "fflonk prove [circuit.zkey] [witness.wtns] [proof.json] [public.json]",
        alias: &["ffp"],
        description: "BETA version. Generates a FFLONK Proof from witness",
        support: Never,
    },
    Command {
        cmd: "fflonk fullprove [witness.json] [circuit.wasm] [circuit.zkey] [proof.json] [public.json]",
        alias: &["fff"],
        description: "BETA version. Generates a witness and the FFLONK Proof in the same command",
        support: Never,
    },
    Command {
        cmd: "fflonk verify [verification_key.json] [public.json] [proof.json]",
        alias: &["ffv"],
        description: "BETA version. Verify a FFLONK Proof",
        support: Never,
    },
    Command {
        cmd: "file info [binary.file]",
        alias: &["fi"],
        description: "Check info of a binary file",
        support: Yes,
    },
];

/// Commands of ours that snarkjs does not have. They keep clap's own syntax and are passed
/// through untouched.
pub const EXTRAS: &[(&str, &str)] = &[
    (
        "bench",
        "Benchmark proving, cold and warm, verifying every proof",
    ),
    (
        "trace",
        "Print a deterministic execution trace, for diffing",
    ),
    #[cfg(feature = "cuda")]
    ("fft-bench", "Sweep the CUDA FFT kernel variants"),
];

/// Our options that take their value as the next token. Everything else is a flag or
/// `key=value`, as in snarkjs.
pub const VALUE_OPTIONS: &[&str] = &["backend", "self-verify", "fallback", "vkey"];

/// snarkjs' short option names (`-verbose|v`, `-entropy|e`, `-name|n`) and `-h`.
fn long_name(key: &str) -> &str {
    match key {
        "v" => "verbose",
        "e" => "entropy",
        "n" => "name",
        "h" => "help",
        "V" => "version",
        other => other,
    }
}

/// What [`normalise`] made of a command line.
#[derive(Debug, PartialEq, Eq)]
pub enum Parsed {
    /// A line for clap.
    Clap(Vec<OsString>),
    /// No command words and no `--help` or `--version`: snarkjs prints its command list
    /// and exits 99.
    NoCommand,
    /// `--help` and no command words: the command list, exit 0.
    Help,
    /// Words that name no command: "Invalid command", the list, exit 99.
    Unknown(Vec<String>),
    /// A snarkjs command snarkrs does not run.
    Unsupported(&'static Command),
}

/// The command a line of words selects, and the words after it. The first command in table
/// order with any matching spelling, and of its spellings the longest; see [`COMMANDS`].
pub fn find(words: &[String]) -> Option<(&'static Command, &[String])> {
    COMMANDS.iter().find_map(|c| {
        c.spellings()
            .into_iter()
            .filter(|s| {
                s.len() <= words.len()
                    && s.iter().zip(words).all(|(a, b)| a.eq_ignore_ascii_case(b))
            })
            .map(|s| s.len())
            .max()
            .map(|n| (c, &words[n..]))
    })
}

/// snarkjs' command line (`argv[0]` included) into clap's. See the module docs.
pub fn normalise(args: impl IntoIterator<Item = OsString>) -> Parsed {
    let args: Vec<OsString> = args.into_iter().collect();
    let Some((prog, rest)) = args.split_first() else {
        return Parsed::NoCommand;
    };
    let rest: Vec<String> = rest
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();

    let mut words = Vec::new();
    let mut options = Vec::new();
    let mut tokens = rest.iter();
    while let Some(tok) = tokens.next() {
        let Some(stripped) = tok.strip_prefix('-') else {
            words.push(tok.clone());
            continue;
        };
        let stripped = stripped.trim_start_matches('-');
        // A bare `-` or `--` has no key. snarkjs files it under "" and never reads it.
        if stripped.is_empty() {
            continue;
        }
        let (key, value) = match stripped.split_once('=') {
            Some((k, v)) => (k, Some(v.to_string())),
            None => (stripped, None),
        };
        let key = long_name(key);
        let value = match value {
            None if VALUE_OPTIONS.contains(&key) => tokens.next().cloned(),
            v => v,
        };
        options.push(match value {
            Some(v) => format!("--{key}={v}"),
            None => format!("--{key}"),
        });
    }

    if let Some(first) = words.first() {
        if EXTRAS.iter().any(|(e, _)| first.eq_ignore_ascii_case(e)) {
            return Parsed::Clap(args);
        }
    }
    let Some((cmd, positionals)) = find(&words) else {
        if words.is_empty() {
            if options.iter().any(|o| o == "--help") {
                return Parsed::Help;
            }
            if options.iter().any(|o| o == "--version") {
                return Parsed::Clap(vec![prog.clone(), "--version".into()]);
            }
            return Parsed::NoCommand;
        }
        return Parsed::Unknown(words);
    };
    if cmd.support != Yes {
        return Parsed::Unsupported(cmd);
    }
    let mut line = vec![prog.clone()];
    line.extend(cmd.words().into_iter().map(OsString::from));
    line.extend(positionals.iter().map(OsString::from));
    line.extend(options.into_iter().map(OsString::from));
    Parsed::Clap(line)
}

/// `helpAll`, over the commands snarkrs runs and then our own.
pub fn help_all() -> String {
    let mut s = format!(
        "snarkrs@{}\n\
         \x20       A drop-in for the snarkjs 0.7.6 command line: Groth16 on BN254.\n\
         \n\
         Usage:\n\
         \x20       snarkrs <full command> ...  <options>\n\
         \x20  or   snarkrs <shortcut> ...  <options>\n\
         \n\
         Type snarkrs <command> --help to get more information for that command\n\
         \n\
         Full Command                  Description\n\
         ============                  =================\n",
        env!("CARGO_PKG_VERSION")
    );
    for c in COMMANDS.iter().filter(|c| c.support == Yes) {
        s.push_str(&format!("{:<30}{}\n", c.words().join(" "), c.description));
        let short = c
            .alias
            .first()
            .map_or(c.words().join(" "), |a| command_words(a).join(" "));
        s.push_str(&format!("     Usage:  snarkrs {short} {}\n", c.params()));
    }
    for (name, description) in EXTRAS {
        s.push_str(&format!("{name:<30}{description}\n"));
        s.push_str(&format!("     Usage:  snarkrs {name} --help\n"));
    }
    s
}

/// `curves.getCurveFromName`: case and punctuation fold away, so `bn128`, `BN-254` and
/// `alt_bn128` all name the one curve snarkrs has.
pub fn check_curve(name: &str) -> anyhow::Result<()> {
    let norm: String = name
        .to_ascii_uppercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    match norm.as_str() {
        "BN128" | "BN254" | "ALTBN128" => Ok(()),
        "BLS12381" => {
            anyhow::bail!("Curve not supported: {name}. snarkrs is Groth16 on BN254 (bn128) only")
        }
        _ => anyhow::bail!("Curve not supported: {name}"),
    }
}

/// `changeExt` in cli.js, the default response name of the two challenge contributes: the
/// text up to and including the last `.` anywhere in the name, then `ext`, or the whole name
/// and `.ext` when there is no dot. The dot need not be in the file name, so `./challenge`
/// becomes `.response`, as it does in snarkjs.
pub fn change_ext(name: &str, ext: &str) -> String {
    match name.rfind('.') {
        Some(i) => format!("{}{ext}", &name[..=i]),
        None => format!("{name}.{ext}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(line: &str) -> Parsed {
        normalise(
            std::iter::once("snarkrs")
                .chain(line.split_whitespace())
                .map(OsString::from),
        )
    }

    fn clap(line: &str) -> Vec<String> {
        match norm(line) {
            Parsed::Clap(v) => v[1..]
                .iter()
                .map(|s| s.to_string_lossy().into_owned())
                .collect(),
            other => panic!("`{line}` did not normalise: {other:?}"),
        }
    }

    #[test]
    fn every_alias_of_every_supported_command_reaches_its_words() {
        for c in COMMANDS.iter().filter(|c| c.support == Yes) {
            for s in c.spellings() {
                let got = clap(&s.join(" "));
                assert_eq!(got, c.words(), "{s:?}");
            }
        }
    }

    #[test]
    fn short_aliases_expand() {
        for (alias, words) in [
            ("ptn bn128 12", "powersoftau new bn128 12"),
            ("pt2 a b", "powersoftau prepare phase2 a b"),
            ("g16p", "groth16 prove"),
            ("g16s", "groth16 setup"),
            ("zkn", "groth16 setup"),
            ("zkey new", "groth16 setup"),
            ("zpw", "groth16 prove"),
            ("zksnark proof", "groth16 prove"),
            ("proof", "groth16 prove"),
            ("verify v p q", "groth16 verify v p q"),
            ("g16v", "groth16 verify"),
            ("zkv", "zkey verify r1cs"),
            ("zkvr", "zkey verify r1cs"),
            ("zkey verify a b c", "zkey verify r1cs a b c"),
            ("zkvi", "zkey verify init"),
            ("zkey verify init a b c", "zkey verify init a b c"),
            ("zkev", "zkey export verificationkey"),
            ("fi x.ptau", "file info x.ptau"),
            ("ptec a", "powersoftau export challenge a"),
            ("ptcc bn128 c", "powersoftau challenge contribute bn128 c"),
            ("ptir a r b", "powersoftau import response a r b"),
            ("zkeb a", "zkey export bellman a"),
            ("zkbc bn128 a b", "zkey bellman contribute bn128 a b"),
            ("zkib a m b", "zkey import bellman a m b"),
            ("zkesv", "zkey export solidityverifier"),
            ("generateverifier", "zkey export solidityverifier"),
        ] {
            assert_eq!(clap(alias).join(" "), words, "{alias}");
        }
    }

    #[test]
    fn command_words_fold_case_and_positionals_do_not() {
        assert_eq!(
            clap("powersOfTau Prepare PHASE2 In.ptau Out.ptau").join(" "),
            "powersoftau prepare phase2 In.ptau Out.ptau"
        );
        assert_eq!(clap("G16V").join(" "), "groth16 verify");
    }

    #[test]
    fn options_go_anywhere_with_any_number_of_dashes() {
        assert_eq!(
            clap("-v zkc -e=some entropy a.zkey --name=me b.zkey").join(" "),
            "zkey contribute entropy a.zkey b.zkey --verbose --entropy=some --name=me"
        );
        assert_eq!(
            clap("ptc a b ---entropy=x=y -n=first").join(" "),
            "powersoftau contribute a b --entropy=x=y --name=first"
        );
        assert_eq!(clap("g16p -h"), ["groth16", "prove", "--help"]);
    }

    #[test]
    fn our_value_options_take_the_next_token() {
        assert_eq!(
            clap("g16p --backend metal c.zkey w.wtns --self-verify false -vkey v.json").join(" "),
            "groth16 prove c.zkey w.wtns --backend=metal --self-verify=false --vkey=v.json"
        );
        assert_eq!(
            clap("g16p -backend=cpu").join(" "),
            "groth16 prove --backend=cpu"
        );
    }

    #[test]
    fn extras_pass_through_untouched() {
        assert_eq!(
            clap("bench --artifacts d --reps 2"),
            ["bench", "--artifacts", "d", "--reps", "2"]
        );
    }

    #[test]
    fn unknown_empty_and_unsupported_lines() {
        assert_eq!(norm(""), Parsed::NoCommand);
        assert_eq!(norm("-v"), Parsed::NoCommand);
        assert_eq!(norm("--help"), Parsed::Help);
        assert_eq!(norm("-h"), Parsed::Help);
        assert_eq!(
            norm("groth17 prove"),
            Parsed::Unknown(vec!["groth17".into(), "prove".into()])
        );
        // One word short of a command is not a prefix match.
        assert!(matches!(norm("powersoftau"), Parsed::Unknown(_)));
        match norm("pks") {
            Parsed::Unsupported(c) => assert_eq!(c.support, Never),
            other => panic!("{other:?}"),
        }
        match norm("zkey export json") {
            Parsed::Unsupported(c) => assert_eq!(c.support, Later),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn curve_names_fold_like_snarkjs() {
        for ok in ["bn128", "BN128", "bn254", "alt_bn128", "altbn128", "Bn-254"] {
            check_curve(ok).unwrap();
        }
        let e = check_curve("bls12381").unwrap_err().to_string();
        assert!(e.contains("not supported") && e.contains("BN254"), "{e}");
        let e = check_curve("secp256k1").unwrap_err().to_string();
        assert!(e.contains("Curve not supported"), "{e}");
    }

    #[test]
    fn change_ext_is_snarkjs_text_surgery() {
        assert_eq!(change_ext("challenge", "response"), "challenge.response");
        assert_eq!(change_ext("ch_0003.bin", "response"), "ch_0003.response");
        assert_eq!(change_ext("a.b.c", "response"), "a.b.response");
        assert_eq!(change_ext("./challenge", "response"), ".response");
        assert_eq!(change_ext("dir.d/challenge", "response"), "dir.response");
    }

    #[test]
    fn help_lists_every_supported_command_with_its_shortcut() {
        let h = help_all();
        for c in COMMANDS.iter().filter(|c| c.support == Yes) {
            assert!(h.contains(&c.words().join(" ")), "{}", c.cmd);
        }
        assert!(h.contains("Usage:  snarkrs ptn <curve> <power> [powersoftau_0000.ptau]"));
        assert!(!h.contains("plonk"));
    }
}
