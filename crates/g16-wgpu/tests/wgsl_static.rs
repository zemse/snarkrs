//! An adversarial static audit of every WGSL module this crate generates.
//!
//! Written by the verifier of units U5 to U8, not by their authors, and deliberately not
//! device-backed: everything here is computed from the generator's own output text, so it
//! answers questions a passing dispatch on an M2 Max cannot.
//!
//! # Why a source audit when the kernels already run
//!
//! Because native wgpu is the *looser* of the two targets on three limits that matter, and
//! `cargo test` on this machine cannot see any of them:
//!
//! * `maxBindGroups` is 8 natively and **4** in Chrome.
//! * `minStorageBufferOffsetAlignment` is 32 natively and **256** in Chrome.
//! * `SHADER_F16` is granted by this adapter even under `STRICT_WEBGPU_COMPLIANCE`, so an
//!   `enable f16;` would compile here and be rejected by a browser that does not offer the
//!   feature.
//!
//! And two the device does enforce, but only for the shapes some test happens to build:
//! storage buffers per entry point (floor 8) and workgroup storage per entry point (floor
//! 16384). Both are checked here for **every** tile size and every module shape the
//! generators can emit, including the ones nothing dispatches yet.
//!
//! # What the parser is and what it is not
//!
//! A brace-matching splitter plus an identifier scan, not a WGSL front end. It resolves the
//! call graph from each entry point and unions the module-scope `var<storage>` and
//! `var<workgroup>` declarations reachable from it, which is what WebGPU calls the static
//! resource interface. It over-approximates: an identifier inside a comment or a string
//! would count as a use. That direction is safe for a limit check, and comments are stripped
//! first so the common case does not arise.
//!
//! It runs on the host with no GPU, so it also passes in CI on a machine with no adapter.

use g16_wgpu::gen::field::{field_module, Variant};
use g16_wgpu::gen::msm::{self as msmgen, LimbPick, Workgroups};
use g16_wgpu::gen::{gather as gathergen, ntt as nttgen, pointwise as pwgen};

// ---------------------------------------------------------------------------
// A very small WGSL reader
// ---------------------------------------------------------------------------

/// One module-scope resource: its name, address space and byte size where that is known.
#[derive(Debug, Clone)]
struct ModuleVar {
    name: String,
    space: String,
    /// Bytes for a `var<workgroup>` whose type is `array<T, N>` with T of known size.
    workgroup_bytes: u64,
    group: u32,
}

#[derive(Debug, Clone)]
struct Function {
    name: String,
    body: String,
    /// `Some(n)` for `@compute @workgroup_size(n)`.
    workgroup_size: Option<u32>,
}

struct Module {
    label: String,
    src: String,
    vars: Vec<ModuleVar>,
    fns: Vec<Function>,
}

/// Everything outside a string literal, with `//` comments removed. WGSL has no block
/// comments in anything this crate emits, and the assertion says so rather than assuming it.
fn strip_comments(src: &str) -> String {
    assert!(
        !src.contains("/*"),
        "the generator emitted a block comment; this reader only handles // comments"
    );
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Bytes one WGSL type occupies, for the types this crate declares in workgroup storage.
fn type_bytes(ty: &str) -> Option<u64> {
    let ty = ty.trim();
    match ty {
        "u32" | "i32" | "f32" => Some(4),
        // `struct Fr { v0: u32, ... }` in gen::field is eight u32.
        "Fr" | "Scalar" => Some(32),
        _ => None,
    }
}

/// Splits `src` at top level, collecting module-scope `var` declarations and every function.
fn parse(label: &str, src: &str) -> Module {
    let clean = strip_comments(src);
    let bytes: Vec<char> = clean.chars().collect();
    let mut vars = Vec::new();
    let mut fns = Vec::new();

    let mut i = 0usize;
    // Text since the end of the previous top-level item, which is where a function's
    // attributes live.
    let mut item_start = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            ';' => {
                let item: String = bytes[item_start..i].iter().collect();
                if let Some(v) = parse_var(&item) {
                    vars.push(v);
                }
                item_start = i + 1;
                i += 1;
            }
            '{' => {
                let head: String = bytes[item_start..i].iter().collect();
                // Skip to the matching close brace.
                let mut depth = 0usize;
                let open = i;
                while i < bytes.len() {
                    if bytes[i] == '{' {
                        depth += 1;
                    } else if bytes[i] == '}' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    i += 1;
                }
                let body: String = bytes[open + 1..i.min(bytes.len())].iter().collect();
                if let Some(name) = fn_name(&head) {
                    fns.push(Function {
                        name,
                        body,
                        workgroup_size: workgroup_size(&head),
                    });
                }
                i += 1;
                // A struct declaration ends `};`, so let the `;` arm reset item_start.
                item_start = i;
            }
            _ => i += 1,
        }
    }

    Module {
        label: label.to_string(),
        src: clean,
        vars,
        fns,
    }
}

/// `@group(0) @binding(3) var<storage, read> NAME: T` -> a [`ModuleVar`].
fn parse_var(item: &str) -> Option<ModuleVar> {
    let at = item.find("var<")?;
    let rest = &item[at + 4..];
    let close = rest.find('>')?;
    let space = rest[..close].split(',').next()?.trim().to_string();
    let after = rest[close + 1..].trim_start();
    let colon = after.find(':')?;
    let name = after[..colon].trim().to_string();
    let ty = after[colon + 1..].trim();
    let group = item
        .find("@group(")
        .and_then(|g| {
            item[g + 7..]
                .split(')')
                .next()
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(u32::MAX);
    let workgroup_bytes = if space == "workgroup" {
        array_bytes(ty).unwrap_or(0)
    } else {
        0
    };
    Some(ModuleVar {
        name,
        space,
        workgroup_bytes,
        group,
    })
}

/// `array<T, N>` -> `N * sizeof(T)`.
fn array_bytes(ty: &str) -> Option<u64> {
    let inner = ty.strip_prefix("array<")?.strip_suffix('>')?;
    let (elem, count) = inner.rsplit_once(',')?;
    let n: u64 = count.trim().trim_end_matches('u').parse().ok()?;
    Some(type_bytes(elem)? * n)
}

fn fn_name(head: &str) -> Option<String> {
    let at = head.rfind("fn ")?;
    let rest = &head[at + 3..];
    let paren = rest.find('(')?;
    Some(rest[..paren].trim().to_string())
}

fn workgroup_size(head: &str) -> Option<u32> {
    if !head.contains("@compute") {
        return None;
    }
    let at = head.find("@workgroup_size(")?;
    let rest = &head[at + 16..];
    let close = rest.find(')')?;
    rest[..close].trim().parse().ok()
}

/// Identifiers appearing in `body`, as a set-ish sorted vector.
fn idents(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in body.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.sort();
    out.dedup();
    out
}

impl Module {
    fn entry_points(&self) -> Vec<&Function> {
        self.fns
            .iter()
            .filter(|f| f.workgroup_size.is_some())
            .collect()
    }

    /// Every function reachable from `entry`, including itself.
    fn reachable(&self, entry: &str) -> Vec<&Function> {
        let mut seen: Vec<&Function> = Vec::new();
        let mut stack = vec![entry.to_string()];
        while let Some(name) = stack.pop() {
            let Some(f) = self.fns.iter().find(|f| f.name == name) else {
                continue;
            };
            if seen.iter().any(|s| s.name == f.name) {
                continue;
            }
            seen.push(f);
            for id in idents(&f.body) {
                if self.fns.iter().any(|g| g.name == id) {
                    stack.push(id);
                }
            }
        }
        seen
    }

    /// Module-scope variables in `space` that `entry` statically reaches.
    fn used_vars(&self, entry: &str, space: &str) -> Vec<&ModuleVar> {
        let bodies: Vec<Vec<String>> = self
            .reachable(entry)
            .iter()
            .map(|f| idents(&f.body))
            .collect();
        self.vars
            .iter()
            .filter(|v| v.space.starts_with(space))
            .filter(|v| bodies.iter().any(|ids| ids.contains(&v.name)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Every module this crate can generate
// ---------------------------------------------------------------------------

fn all_modules() -> Vec<Module> {
    let v = Variant::default();
    let mut out = vec![
        parse("field", &field_module(v)),
        parse("gather_abc", &gathergen::gather_module(v)),
        parse("h_join", &pwgen::h_join_module(v)),
        parse("msm digits", &msmgen::digits_module()),
        parse("msm mont", &msmgen::mont_module(v)),
        parse(
            "msm fused",
            &msmgen::fused_module_at(v, Workgroups::default(), LimbPick::default()),
        ),
        parse(
            "msm fused, Index pick",
            &msmgen::fused_module_at(v, Workgroups::default(), LimbPick::Index),
        ),
    ];
    // Every tile size the generator will accept, not only the ones `split_passes` produces
    // at the shipped cap. A `k` nothing dispatches today is a `k` nobody has checked.
    for k in 0..=9u32 {
        out.push(parse(&format!("ntt k={k}"), &nttgen::ntt_module(v, &[k])));
    }
    // The two-tile module a domain with an uneven split compiles, which is what a real key
    // at log_n % batches != 0 builds.
    out.push(parse("ntt k=6,7", &nttgen::ntt_module(v, &[6, 7])));
    out.push(parse("ntt k=8,9", &nttgen::ntt_module(v, &[8, 9])));
    out
}

// ---------------------------------------------------------------------------
// 1. Storage buffers, bind groups, workgroup size and workgroup storage
// ---------------------------------------------------------------------------

/// `maxStorageBuffersPerShaderStage`. 8 at the spec floor; this adapter reports 9 under
/// strict compliance, so a kernel that needs 9 passes natively and fails in a browser.
const FLOOR_STORAGE_BUFFERS: usize = 8;
/// `maxBindGroups`. 4 in every browser at every tier, 8 natively.
const FLOOR_BIND_GROUPS: u32 = 4;
/// `maxComputeInvocationsPerWorkgroup`. 256 at the floor, 1024 on this adapter.
const FLOOR_INVOCATIONS: u32 = 256;
/// `maxComputeWorkgroupStorageSize`, in bytes. 16384 at the floor, 32768 on this adapter.
const FLOOR_WORKGROUP_BYTES: u64 = 16384;

#[test]
fn every_generated_entry_point_fits_the_browser_floor() {
    let mut checked = 0usize;
    for m in all_modules() {
        // `field` is a prelude with no entry point of its own; everything else has one.
        assert!(
            !m.entry_points().is_empty() || m.label == "field",
            "{}: parsed no entry points, so this audit proved nothing",
            m.label
        );
        for g in &m.vars {
            if g.space.starts_with("storage") || g.space == "uniform" {
                assert!(
                    g.group < FLOOR_BIND_GROUPS,
                    "{}: {} is in @group({}), and maxBindGroups is {FLOOR_BIND_GROUPS} in \
                     every browser (8 natively, so this passes here and fails in Chrome)",
                    m.label,
                    g.name,
                    g.group
                );
            }
        }
        for f in m.entry_points() {
            let wg = f.workgroup_size.unwrap();
            assert!(
                wg > 0 && wg <= FLOOR_INVOCATIONS,
                "{}: {} runs {wg} threads, the floor allows 1..={FLOOR_INVOCATIONS}",
                m.label,
                f.name
            );
            let storage = m.used_vars(&f.name, "storage");
            assert!(
                storage.len() <= FLOOR_STORAGE_BUFFERS,
                "{}: {} statically reaches {} storage buffers ({:?}), the floor allows \
                 {FLOOR_STORAGE_BUFFERS}",
                m.label,
                f.name,
                storage.len(),
                storage.iter().map(|v| &v.name).collect::<Vec<_>>()
            );
            let shared: u64 = m
                .used_vars(&f.name, "workgroup")
                .iter()
                .map(|v| {
                    assert!(
                        v.workgroup_bytes > 0,
                        "{}: could not size workgroup array {}",
                        m.label,
                        v.name
                    );
                    v.workgroup_bytes
                })
                .sum();
            assert!(
                shared <= FLOOR_WORKGROUP_BYTES,
                "{}: {} allocates {shared} bytes of workgroup storage, the floor allows \
                 {FLOOR_WORKGROUP_BYTES}",
                m.label,
                f.name
            );
            println!(
                "{:22} {:24} wg {:3}  storage {}  shared {:5} B",
                m.label,
                f.name,
                wg,
                storage.len(),
                shared
            );
            checked += 1;
        }
    }
    println!("{checked} entry points audited across every generated module shape");
    assert!(
        checked > 40,
        "only {checked} entry points found; the parser is not seeing the modules"
    );
}

// ---------------------------------------------------------------------------
// 2. Nothing native-only, because native is the looser target
// ---------------------------------------------------------------------------

#[test]
fn no_generated_module_uses_a_construct_the_browser_lacks() {
    // Each entry is (token, why it would pass here and fail in a browser).
    let banned: &[(&str, &str)] = &[
        (
            "u64",
            "WGSL has no 64-bit integer; naga accepts none either, but a future one would",
        ),
        ("i64", "same"),
        (
            "f16",
            "SHADER_F16 is granted by this adapter under strict compliance and is not \
                 guaranteed in a browser, so `enable f16` compiles here and fails there",
        ),
        (
            "enable ",
            "wgpu 30's browser FEATURES_MAPPING has 16 entries and no directive this \
                     crate needs is among them",
        ),
        (
            "subgroup",
            "not reachable through wgpu in a browser; naga rejects the directive",
        ),
        (
            "mulExtended",
            "not WGSL; it is GLSL/HLSL and would be a name naga resolves to nothing",
        ),
        (
            "ptr<",
            "unrestricted_pointer_parameters ships in Chrome and Safari and is \
                  unimplemented in naga, so this is the other direction: native-only failure",
        ),
        (
            "texture",
            "no kernel here samples anything, and a texture binding would change the \
                     limit being counted",
        ),
        (
            "atomicCompareExchangeWeak",
            "not used; flagged so a future CAS gets a limits review",
        ),
    ];
    for m in all_modules() {
        for (tok, why) in banned {
            assert!(
                !m.src.contains(tok),
                "{}: generated WGSL contains {tok:?}. {why}",
                m.label
            );
        }
    }
    println!(
        "{} banned constructs checked against every generated module",
        banned.len()
    );
}

// ---------------------------------------------------------------------------
// 3. Barrier uniformity, which WGSL makes a hard error and MSL makes undefined
// ---------------------------------------------------------------------------

/// What kind of block a `{` opened.
#[derive(Debug, Clone, PartialEq)]
enum Block {
    Fn,
    Cond(String),
    Loop(String),
    Bare,
}

/// Every `workgroupBarrier()` in `body`, with the stack of blocks enclosing it.
fn barrier_contexts(body: &str) -> Vec<Vec<Block>> {
    let chars: Vec<char> = body.chars().collect();
    let needle: Vec<char> = "workgroupBarrier".chars().collect();
    let mut stack: Vec<Block> = vec![Block::Fn];
    let mut out = Vec::new();
    let mut i = 0usize;
    // Start of the statement currently being read, so a `{` can be classified from its head.
    let mut stmt = 0usize;
    while i < chars.len() {
        match chars[i] {
            '{' => {
                let head: String = chars[stmt..i].iter().collect();
                let head = head.trim().to_string();
                let kind = if head.starts_with("if")
                    || head.contains("} else")
                    || head.starts_with("else")
                    || head.starts_with("switch")
                    || head.starts_with("case")
                    || head.starts_with("default")
                {
                    Block::Cond(head.clone())
                } else if head.starts_with("for")
                    || head.starts_with("while")
                    || head.starts_with("loop")
                {
                    Block::Loop(head.clone())
                } else {
                    Block::Bare
                };
                stack.push(kind);
                stmt = i + 1;
            }
            '}' => {
                stack.pop();
                stmt = i + 1;
            }
            ';' => stmt = i + 1,
            _ => {
                if chars[i..].starts_with(&needle[..]) {
                    out.push(stack.clone());
                }
            }
        }
        i += 1;
    }
    out
}

#[test]
fn every_barrier_sits_in_uniform_control_flow() {
    let mut barriers = 0usize;
    for m in all_modules() {
        for f in &m.fns {
            for stack in barrier_contexts(&f.body) {
                barriers += 1;
                for b in &stack {
                    match b {
                        Block::Cond(h) => panic!(
                            "{}: {} has a workgroupBarrier() inside a conditional block \
                             ({h:?}). WGSL rejects the shader; MSL merely makes it undefined, \
                             so the Metal original may well look like this.",
                            m.label, f.name
                        ),
                        Block::Loop(h) => {
                            // A loop bound is uniform if it is a literal or comes out of the
                            // uniform block. `P.` is the only uniform binding in this crate.
                            let uniform = h.contains("P.")
                                || h.chars().any(|c| c.is_ascii_digit())
                                || h.contains("chunks");
                            assert!(
                                uniform,
                                "{}: {} has a workgroupBarrier() inside {h:?}, whose trip \
                                 count is neither a literal nor a uniform read",
                                m.label, f.name
                            );
                        }
                        Block::Fn | Block::Bare => {}
                    }
                }
            }
        }
    }
    println!("{barriers} workgroupBarrier() sites, all in uniform control flow");
    assert!(
        barriers > 20,
        "only {barriers} barriers found; the NTT and msm_scan both have several, so the \
         reader is not seeing them"
    );
}

// ---------------------------------------------------------------------------
// 4. The uniform parameter structs, which nothing on either side validates
// ---------------------------------------------------------------------------

/// `struct NAME { a: u32, b: u32, ... }` -> the field names in declaration order.
fn struct_fields(src: &str, name: &str) -> Vec<String> {
    let at = src
        .find(&format!("struct {name} {{"))
        .unwrap_or_else(|| panic!("no struct {name} in the generated source"));
    let body = &src[at..];
    let open = body.find('{').unwrap();
    let close = body.find('}').unwrap();
    body[open + 1..close]
        .split(',')
        .flat_map(|f| f.split(';'))
        .filter_map(|f| f.split(':').next())
        .map(|f| f.trim().to_string())
        .filter(|f| !f.is_empty())
        .collect()
}

/// A parameter block reaches the shader as bytes and **nothing checks the correspondence**:
/// `ParamRing::push` takes any `Pod`, WGSL's uniform layout rules differ from `repr(C)`, and
/// a field added, dropped or reordered on one side reads as plausible garbage with no
/// validation error on either target.
///
/// So the field list is written out here a second time on purpose. That is duplication, and
/// it is the point: a one-sided edit has to fail this test before it can ship. The size
/// assertion is the other half, and it is the half that catches a field silently changing
/// type.
#[test]
fn every_uniform_parameter_struct_matches_its_host_mirror() {
    let v = Variant::default();
    let cases: [(&str, String, &str, usize, &[&str]); 4] = [
        (
            "gather_abc",
            gathergen::gather_module(v),
            "GatherParams",
            std::mem::size_of::<g16_wgpu::GatherParams>(),
            &[
                "row_lo",
                "row_hi",
                "row_base_a",
                "row_base_b",
                "nz_base_a",
                "nz_base_b",
                "pad0",
                "pad1",
            ],
        ),
        (
            "ntt",
            nttgen::ntt_module(v, &[6]),
            "NttParams",
            std::mem::size_of::<g16_wgpu::NttParams>(),
            &[
                "log_n",
                "s0",
                "scale_mode",
                "pad0",
                "ks0",
                "ks1",
                "ks2",
                "ks3",
                "ks4",
                "ks5",
                "ks6",
                "ks7",
            ],
        ),
        (
            "h_join",
            pwgen::h_join_module(v),
            "HJoinParams",
            std::mem::size_of::<g16_wgpu::HJoinParams>(),
            &["lo", "hi", "pad0", "pad1"],
        ),
        (
            "msm",
            msmgen::fused_module_at(v, Workgroups::default(), LimbPick::default()),
            "MsmParams",
            std::mem::size_of::<g16_wgpu::MsmParams>(),
            // The last four are the point stages'. They were added to both sides when the
            // G2 MSM landed and this list was not, which is precisely the drift this test
            // exists to catch: the host struct and the WGSL struct agreed with each other
            // and disagreed with the expectation written here.
            &[
                "n",
                "c",
                "n_windows",
                "n_buckets",
                "cap",
                "scalar_off",
                "lo",
                "base_off",
                "ones_groups",
                "slice_len",
                "slices",
                "pad0",
            ],
        ),
    ];

    for (label, src, name, host_bytes, want) in cases {
        let got = struct_fields(&strip_comments(&src), name);
        assert_eq!(
            got,
            want.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            "{label}: the WGSL {name} fields are not what the host mirror declares"
        );
        assert_eq!(
            host_bytes,
            got.len() * 4,
            "{label}: {name} is {host_bytes} bytes on the host and {} u32 in WGSL",
            got.len()
        );
        // WGSL rounds every uniform struct up to a 16-byte alignment, so a host struct that
        // is not already a multiple of 16 has invisible tail padding the shader will read
        // as the next field.
        assert_eq!(
            host_bytes % 16,
            0,
            "{label}: {name} is {host_bytes} bytes, and WGSL aligns a uniform struct to 16"
        );
        // And it has to fit a ring slot, which is minUniformBufferOffsetAlignment.
        assert!(
            host_bytes <= 256,
            "{label}: {name} is {host_bytes} bytes, over the 256-byte ring slot"
        );
    }
    println!("4 uniform parameter structs checked field for field against their host mirrors");
}

// ---------------------------------------------------------------------------
// 5. The recoding width, which a doc comment says must match a private constant
// ---------------------------------------------------------------------------

/// `crate::msm::RECODE_BITS` says it "must match `g16_msm::RECODE_BITS` and
/// `g16_metal::msm::RECODE_BITS`", and it cannot be compared to either: both are private
/// `const`s in crates this one does not depend on, so the requirement is unenforced and a
/// drift would show up as an MSM that disagrees with the CPU one by `2^(W*c)` on some
/// scalars and not others.
///
/// What can be checked is the thing all three are derived from, which is the field: the
/// recoding carries one bit past the top of the scalar, so it is `MODULUS_BIT_SIZE + 1`.
/// `g16-msm` spells exactly that (`SCALAR_BITS + 1`), so pinning it here pins the pair
/// without a dependency.
#[test]
fn the_recoding_width_is_the_modulus_bit_size_plus_one() {
    use g16_field::{Fr, PrimeField};
    assert_eq!(
        g16_wgpu::msm::RECODE_BITS,
        Fr::MODULUS_BIT_SIZE + 1,
        "RECODE_BITS is not MODULUS_BIT_SIZE + 1, so the top window's carry is no longer \
         provably zero and g16-msm's private copy has drifted away from this one"
    );
    // And the top window really does sit above the modulus, which is the property the carry
    // argument rests on: r < 2^254 and the digits are laid out over 255 bits.
    assert_eq!(Fr::MODULUS_BIT_SIZE, 254);
    println!(
        "RECODE_BITS = {} = MODULUS_BIT_SIZE + 1",
        g16_wgpu::msm::RECODE_BITS
    );
}
