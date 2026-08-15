//! Heap profile of one cold proof, phase by phase.
//!
//! `cargo run --release -p g16-cli --features dhat-heap --bin g16-dhat -- ARTIFACT_DIR [OUT.json]`
//!
//! This is a separate binary rather than a flag on `g16` because it installs a global
//! allocator. `dhat::Alloc` wraps every allocation with a backtrace capture, which is
//! both slow and a permanent property of the process, so it must not be reachable from
//! the binary anyone benchmarks. The `dhat-heap` feature defaults off and the `dhat`
//! dependency is optional, so a plain `cargo build --release` does not link it at all.
//!
//! **The timings this prints are meaningless.** Under `dhat::Alloc` everything is many
//! times slower. Bytes and counts are what this measures; wall clock comes from
//! `bench/scripts/profile-wallclock.sh`.
//!
//! Phase boundaries are snapshotted so allocation can be attributed to load, prepare and
//! prove separately. A single total would be dominated by the zkey parse and would say
//! nothing about the proving path, which is the part that runs per proof on a warm server.

use std::path::PathBuf;

use g16_core::{cpu::CpuBackend, prove::prove, Backend, StageTimings};
use g16_zkey::{wtns::Witness, ProvingKey};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// The four numbers that answer different questions: `total_*` is allocation churn (how
/// hard the allocator is being worked), `max_*` is peak RSS pressure (how much memory the
/// process actually needs), and neither can be derived from the other.
#[derive(Clone, Copy)]
struct Snap {
    total_blocks: u64,
    total_bytes: u64,
    max_blocks: usize,
    max_bytes: usize,
    curr_bytes: usize,
}

impl Snap {
    fn now() -> Self {
        let s = dhat::HeapStats::get();
        Snap {
            total_blocks: s.total_blocks,
            total_bytes: s.total_bytes,
            max_blocks: s.max_blocks,
            max_bytes: s.max_bytes,
            curr_bytes: s.curr_bytes,
        }
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn phase(label: &str, before: Snap, after: Snap) {
    println!(
        "{label:<26} {:>12} allocs  {:>12.2} MiB allocated  live now {:>10.2} MiB",
        after.total_blocks - before.total_blocks,
        mib(after.total_bytes - before.total_bytes),
        mib(after.curr_bytes as u64),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "bench/artifacts/js_8x8_d32".to_string()),
    );
    let out = args
        .next()
        .unwrap_or_else(|| "bench/results/profiling/dhat-heap.json".to_string());
    if let Some(parent) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Built before the profiler so the profiler's own setup is not in the numbers, and
    // dropped at the end of main so the JSON is written even though the release profile
    // uses `panic = "abort"` (nothing here is expected to unwind, but a panic would lose
    // the file, which is worth knowing rather than being surprised by).
    let _profiler = dhat::Profiler::builder().file_name(&out).build();

    let start = Snap::now();
    let pk = ProvingKey::load(&dir.join("circuit.zkey")).expect("loading circuit.zkey");
    let after_zkey = Snap::now();
    let witness = Witness::load(&dir.join("circuit.wtns"))
        .expect("loading circuit.wtns")
        .0;
    let after_wtns = Snap::now();

    let n_vars = pk.n_vars;
    let domain = pk.domain_size;
    let circuit = CpuBackend::new().prepare(pk).expect("prepare");
    let after_prepare = Snap::now();

    let mut t = StageTimings::default();
    let mut rng = ark_std::rand::thread_rng();
    let proof = prove(circuit.as_ref(), &witness, &mut rng, &mut t).expect("prove");
    let after_prove = Snap::now();

    // A second proof on the same prepared circuit: this is the per-proof cost a resident
    // service pays, with every one-off allocation already made. The difference between it
    // and the first proof is warm-up, and if the two differ much that is itself a finding.
    let before_prove2 = Snap::now();
    let proof2 = prove(circuit.as_ref(), &witness, &mut rng, &mut t).expect("prove");
    let after_prove2 = Snap::now();

    // Both proofs are consumed so nothing above can be optimised away.
    std::hint::black_box((&proof, &proof2));

    println!("artifact  {}", dir.display());
    println!("n_vars {n_vars}  domain {domain}");
    println!();
    phase("zkey load", start, after_zkey);
    phase("witness load", after_zkey, after_wtns);
    phase("prepare (backend)", after_wtns, after_prepare);
    phase("prove #1 (cold caches)", after_prepare, after_prove);
    phase("prove #2 (warm)", before_prove2, after_prove2);
    println!();
    println!(
        "process total     {:>12} allocs  {:>12.2} MiB allocated",
        after_prove2.total_blocks,
        mib(after_prove2.total_bytes)
    );
    println!(
        "peak heap         {:>12} blocks  {:>12.2} MiB live at peak",
        after_prove2.max_blocks,
        mib(after_prove2.max_bytes as u64)
    );
    println!();
    println!("wrote {out} (open with https://nnethercote.github.io/dh_view/dh_view.html)");
}
