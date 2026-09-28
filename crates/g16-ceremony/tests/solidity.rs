//! The generated verifier contract, run on a real EVM.
//!
//! Each case compiles [`g16_ceremony::solidity::solidity_verifier`]'s output with `solc`,
//! deploys it to a throwaway `anvil`, and calls `verifyProof` through `cast` with the
//! artifact's checked-in proof: once as is, and once per tampering the contract has to
//! refuse. When snarkjs is installed its own generated verifier is deployed next to ours
//! and has to give the same answer on every call, fed with snarkjs' own
//! `zkey export soliditycalldata` output, so the ABI is exercised the way a snarkjs user
//! calls it.
//!
//! Skips with a message when `solc`, `anvil`, `cast` or the artifacts are absent.
//! `SNARKJS` names the snarkjs binary, as in `phase2.rs`.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::str::FromStr;

use g16_field::{BigInteger, Fq, Fr, PrimeField};

/// anvil's first well-known dev account.
const DEV_KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

fn artifacts() -> Vec<(String, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/artifacts");
    ["tiny_mul", "js_1x1_d8"]
        .iter()
        .map(|n| ((*n).to_owned(), root.join(n)))
        .filter(|(_, d)| {
            ["circuit.zkey", "proof.json", "public.json"]
                .iter()
                .all(|f| d.join(f).is_file())
        })
        .collect()
}

fn have(bin: &str) -> bool {
    Command::new(bin).arg("--version").output().is_ok()
}

/// The binary and whether it is 0.7.6, the version this crate matches. Earlier releases
/// generate a verifier that accepts a public signal `s + r` (0.7.2 does), so only 0.7.6
/// is held to [`cases`]' answer on that tampering.
fn snarkjs_bin() -> Option<(String, bool)> {
    let bin = std::env::var("SNARKJS").unwrap_or_else(|_| "snarkjs".to_owned());
    let out = Command::new(&bin).arg("--help").output().ok()?;
    let is_076 = String::from_utf8_lossy(&out.stdout).contains("snarkjs@0.7.6");
    Some((bin, is_076))
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "g16-ceremony-solidity-{test}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(cmd: &mut Command) -> String {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("running {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// An anvil on its own port, killed on drop so a failed assertion does not leak it.
struct Anvil {
    child: Child,
    url: String,
}

impl Anvil {
    fn start() -> Self {
        let port = 20000 + (std::process::id() % 20000) as u16;
        let child = Command::new("anvil")
            .args(["--port", &port.to_string(), "--silent"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning anvil");
        let anvil = Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        };
        for _ in 0..100 {
            let up = Command::new("cast")
                .args(["chain-id", "--rpc-url", &anvil.url])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if up {
                return anvil;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("anvil did not come up on {}", anvil.url);
    }

    /// Compile `source` and deploy the one contract in it, returning its address.
    fn deploy(&self, source: &Path) -> String {
        let out = run(Command::new("solc")
            .args(["--optimize", "--bin"])
            .arg(source));
        // `solc --bin` prints a banner per contract, then "Binary:", then the hex.
        let bin = out
            .lines()
            .skip_while(|l| !l.starts_with("Binary"))
            .nth(1)
            .expect("solc printed no binary")
            .trim()
            .to_owned();
        let json = run(Command::new("cast").args([
            "send",
            "--rpc-url",
            &self.url,
            "--private-key",
            DEV_KEY,
            "--json",
            "--create",
            &format!("0x{bin}"),
        ]));
        let receipt: serde_json::Value = serde_json::from_str(&json).unwrap();
        receipt["contractAddress"].as_str().unwrap().to_owned()
    }

    fn call(&self, addr: &str, c: &Calldata) -> bool {
        let out = run(Command::new("cast")
            .args(["call", "--rpc-url", &self.url, addr, &c.signature()])
            .args(c.args()));
        match out.trim() {
            "true" => true,
            "false" => false,
            other => panic!("verifyProof returned {other:?}"),
        }
    }

    /// Gas used by a transaction calling `verifyProof` with a generous limit.
    ///
    /// Not `cast estimate`: a precompile that runs out of gas makes `staticcall` return 0
    /// and the verifier return `false` without reverting, so the smallest limit that does
    /// not revert is one where the pairing never ran.
    fn gas_used(&self, addr: &str, c: &Calldata) -> u64 {
        let json = run(Command::new("cast")
            .args([
                "send",
                "--rpc-url",
                &self.url,
                "--private-key",
                DEV_KEY,
                "--gas-limit",
                "2000000",
                "--json",
                addr,
                &c.signature(),
            ])
            .args(c.args()));
        let receipt: serde_json::Value = serde_json::from_str(&json).unwrap();
        let used = receipt["gasUsed"]
            .as_str()
            .unwrap()
            .trim_start_matches("0x");
        u64::from_str_radix(used, 16).unwrap()
    }
}

impl Drop for Anvil {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `verifyProof`'s four arguments as decimal strings, `_pB` already in EIP-197 order.
#[derive(Clone)]
struct Calldata {
    a: [String; 2],
    b: [[String; 2]; 2],
    c: [String; 2],
    public: Vec<String>,
}

impl Calldata {
    /// From snarkjs' `proof.json` and `public.json`, swapping each G2 coordinate's halves
    /// the way `zkey export soliditycalldata` does.
    fn from_json(proof: &Path, public: &Path) -> Self {
        let p: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(proof).unwrap()).unwrap();
        let s = |v: &serde_json::Value| v.as_str().unwrap().to_owned();
        let public: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(public).unwrap()).unwrap();
        Self {
            a: [s(&p["pi_a"][0]), s(&p["pi_a"][1])],
            b: [
                [s(&p["pi_b"][0][1]), s(&p["pi_b"][0][0])],
                [s(&p["pi_b"][1][1]), s(&p["pi_b"][1][0])],
            ],
            c: [s(&p["pi_c"][0]), s(&p["pi_c"][1])],
            public: public.iter().map(s).collect(),
        }
    }

    /// snarkjs' `soliditycalldata` output, which is the four arrays joined by commas.
    fn from_snarkjs(text: &str) -> Self {
        let v: Vec<serde_json::Value> = serde_json::from_str(&format!("[{text}]")).unwrap();
        // Hex strings; normalise to decimal so the two sources compare equal.
        let d = |v: &serde_json::Value| {
            let h = v.as_str().unwrap().trim_start_matches("0x");
            let mut limbs = [0u64; 4];
            let h = format!("{h:0>64}");
            for (i, l) in limbs.iter_mut().enumerate() {
                *l = u64::from_str_radix(&h[64 - 16 * (i + 1)..64 - 16 * i], 16).unwrap();
            }
            ark_ff::BigInt::<4>(limbs).to_string()
        };
        Self {
            a: [d(&v[0][0]), d(&v[0][1])],
            b: [
                [d(&v[1][0][0]), d(&v[1][0][1])],
                [d(&v[1][1][0]), d(&v[1][1][1])],
            ],
            c: [d(&v[2][0]), d(&v[2][1])],
            public: v[3].as_array().unwrap().iter().map(d).collect(),
        }
    }

    fn signature(&self) -> String {
        format!(
            "verifyProof(uint256[2],uint256[2][2],uint256[2],uint256[{}])(bool)",
            self.public.len()
        )
    }

    fn args(&self) -> [String; 4] {
        let arr = |xs: &[String]| format!("[{}]", xs.join(","));
        [
            arr(&self.a),
            format!("[{},{}]", arr(&self.b[0]), arr(&self.b[1])),
            arr(&self.c),
            arr(&self.public),
        ]
    }
}

/// `x + m` for a decimal `x < m`, which stays under 2^256 for both BN254 moduli.
fn plus_modulus<F: PrimeField<BigInt = ark_ff::BigInt<4>>>(x: &str) -> String {
    let mut v = F::from_str(x).ok().unwrap().into_bigint();
    assert!(!v.add_with_carry(&F::MODULUS));
    v.to_string()
}

/// `x + 1` for a decimal `x < r`, reduced.
fn plus_one(x: &str) -> String {
    (Fr::from_str(x).ok().unwrap() + Fr::from(1u64)).to_string()
}

/// Every call a verifier must answer: the valid proof, and the tamperings with the
/// answer each must get.
fn cases(valid: &Calldata) -> Vec<(&'static str, Calldata, bool)> {
    let mut out = vec![("valid", valid.clone(), true)];

    let mut t = valid.clone();
    t.public[0] = plus_one(&t.public[0]);
    out.push(("public[0] + 1", t, false));

    // The same value mod r: 0x07 would accept it, so only the range check refuses it.
    let mut t = valid.clone();
    t.public[0] = plus_modulus::<Fr>(&t.public[0]);
    out.push(("public[0] + r", t, false));

    let mut t = valid.clone();
    t.a = valid.c.clone();
    out.push(("A replaced by C", t, false));

    let mut t = valid.clone();
    t.b = [
        [valid.b[0][1].clone(), valid.b[0][0].clone()],
        [valid.b[1][1].clone(), valid.b[1][0].clone()],
    ];
    out.push(("B with unswapped halves", t, false));

    let mut t = valid.clone();
    t.a[1] = plus_modulus::<Fq>(&t.a[1]);
    out.push(("A.y + q", t, false));
    out
}

#[test]
fn generated_verifier_accepts_the_proof_and_refuses_tampering_like_snarkjs() {
    if !(have("solc") && have("anvil") && have("cast")) {
        eprintln!("SKIPPED: solc, anvil or cast is not installed");
        return;
    }
    let found = artifacts();
    if found.is_empty() {
        eprintln!("SKIPPED: no artifacts under bench/artifacts");
        return;
    }
    let snarkjs = snarkjs_bin();
    if snarkjs.is_none() {
        eprintln!("snarkjs not installed: checking our verifier alone");
    }
    let dir = tmp_dir("verify");
    let anvil = Anvil::start();

    for (name, art) in &found {
        let ours_sol = dir.join(format!("{name}_ours.sol"));
        g16_ceremony::solidity::export_solidity_verifier(&art.join("circuit.zkey"), &ours_sol)
            .unwrap();
        let ours = anvil.deploy(&ours_sol);
        let valid = Calldata::from_json(&art.join("proof.json"), &art.join("public.json"));

        let theirs = snarkjs.as_ref().map(|(bin, _)| {
            let sol = dir.join(format!("{name}_snarkjs.sol"));
            run(Command::new(bin)
                .args(["zkey", "export", "solidityverifier"])
                .arg(art.join("circuit.zkey"))
                .arg(&sol));
            let cd = run(Command::new(bin)
                .args(["zkey", "export", "soliditycalldata"])
                .arg(art.join("public.json"))
                .arg(art.join("proof.json")));
            let cd = Calldata::from_snarkjs(cd.trim());
            // snarkjs' calldata is the argument list we built ourselves.
            assert_eq!(cd.args(), valid.args(), "{name}: soliditycalldata differs");
            anvil.deploy(&sol)
        });

        for (what, c, want) in cases(&valid) {
            assert_eq!(anvil.call(&ours, &c), want, "{name}: ours on {what}");
            if let (Some(theirs), Some((_, is_076))) = (&theirs, &snarkjs) {
                let got = anvil.call(theirs, &c);
                if *is_076 || what != "public[0] + r" {
                    assert_eq!(got, want, "{name}: snarkjs' on {what}");
                } else if got != want {
                    eprintln!("{name}: this snarkjs' verifier answers {got} on {what}");
                }
            }
        }
        let gas = anvil.gas_used(&ours, &valid);
        match &theirs {
            Some(t) => eprintln!(
                "{name}: verifyProof transaction gas {gas} ours, {} snarkjs'",
                anvil.gas_used(t, &valid)
            ),
            None => eprintln!("{name}: verifyProof transaction gas {gas} ours"),
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
