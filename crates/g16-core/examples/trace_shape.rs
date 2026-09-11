//! Every dimension stages 0-9 are made of, for one artifact, as plain counts.
//!
//! `msm_shape` and `ntt_shape` each answer one question with timings attached. This
//! answers "how much work is there", with no timings at all, because a roofline needs the
//! operation counts and the byte traffic and nothing else. Everything printed here is a
//! property of the key and the witness, so it is identical on every machine.
//!
//! `cargo run --release -p g16-core --example trace_shape -- bench/artifacts/csp`

use g16_field::{AffineRepr, Fr, One, Zero};
use g16_msm::window_size;
use g16_zkey::{wtns::Witness, ProvingKey};

/// Bytes on the wire for one field element and one point, in the packed device layout
/// (`g16_gpu_layout`), not in ark's padded host representation.
const FR: usize = 32;
const G1A: usize = 64;
const G2A: usize = 128;

fn general_count(s: &[Fr]) -> usize {
    s.iter().filter(|x| !x.is_zero() && !x.is_one()).count()
}

fn ones_count(s: &[Fr]) -> usize {
    s.iter().filter(|x| x.is_one()).count()
}

fn inf_count<A: AffineRepr>(b: &[A]) -> usize {
    b.iter().filter(|p| p.is_zero()).count()
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/artifacts/csp".to_string());

    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("reading {root}: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|d| d.join("circuit.zkey").is_file() && d.join("circuit.wtns").is_file())
        .collect();
    dirs.sort();

    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let pk = ProvingKey::load(&dir.join("circuit.zkey")).unwrap();
        let w = Witness::load(&dir.join("circuit.wtns")).unwrap().0;
        let n = pk.domain_size;
        let log_n = n.trailing_zeros();

        println!("\n=== {name} ===");
        println!("n_vars        {}", pk.n_vars);
        println!("n_public      {}", pk.n_public);
        println!("domain_size   {n}  (2^{log_n})");

        // ---- stage 0: the CSR gather ----
        let nnz_a = pk.coeffs.signal[0].len();
        let nnz_b = pk.coeffs.signal[1].len();
        println!("\n-- stage 0, gather --");
        println!("nnz(A)        {nnz_a}");
        println!("nnz(B)        {nnz_b}");
        println!("nnz total     {}", nnz_a + nnz_b);
        // One mul + one add per nonzero, plus the C = A*B pointwise over the domain.
        println!(
            "Fr muls       {}  (gather {} + C {n})",
            nnz_a + nnz_b + n,
            nnz_a + nnz_b
        );
        // Each nonzero reads a u32 signal, an Fr coefficient and an Fr witness entry.
        let gather_bytes = (nnz_a + nnz_b) * (4 + FR + FR) + 3 * n * FR;
        println!(
            "bytes moved   {gather_bytes}  ({:.1} MB)",
            gather_bytes as f64 / 1e6
        );

        // ---- stages 1-3: six transforms ----
        // Radix-2: n/2 butterflies per pass, log_n passes, 1 mul + 1 add + 1 sub each.
        let bf = (n / 2) * log_n as usize;
        println!("\n-- stages 1-3, six transforms --");
        println!("passes each   {log_n}");
        println!("butterflies   {bf} per transform, {} over six", bf * 6);
        println!("Fr muls       {}  (six transforms)", bf * 6);
        println!("coset shift   {} muls (3 vectors x {n})", 3 * n);
        // Every pass reads and writes the whole vector once, plus a twiddle read.
        let ntt_bytes = 6 * log_n as usize * n * (2 * FR) + 6 * (n / 2) * FR;
        println!(
            "bytes moved   {ntt_bytes}  ({:.1} MB, {} read+write passes)",
            ntt_bytes as f64 / 1e6,
            6 * log_n
        );

        // ---- stage 4 ----
        println!("\n-- stage 4, H = A*B - C --");
        println!("Fr muls       {n}");
        println!(
            "bytes moved   {}  ({:.1} MB)",
            4 * n * FR,
            4.0 * n as f64 * FR as f64 / 1e6
        );

        // ---- stages 5-9 ----
        println!("\n-- stages 5-9, five MSMs --");
        let l = &w[pk.n_public + 1..];
        let mut h_ones = 0usize;
        println!(
            "{:<8} {:>9} {:>9} {:>9} {:>6} {:>5} {:>8} {:>9} {:>12}",
            "msm", "len", "general", "ones", "inf", "c", "windows", "buckets", "base MB"
        );
        let mut row = |tag: &str,
                       len: usize,
                       general: usize,
                       ones: usize,
                       inf: usize,
                       pt: usize| {
            let c = window_size(general);
            let nw = 255usize.div_ceil(c as usize);
            let nb = 1usize << (c - 1);
            println!(
                "{tag:<8} {len:>9} {general:>9} {ones:>9} {inf:>6} {c:>5} {nw:>8} {nb:>9} {:>12.1}",
                (len * pt) as f64 / 1e6
            );
            (c, nw, nb)
        };
        let gen_w = general_count(&w);
        let ones_w = ones_count(&w);
        row(
            "A_g1",
            pk.a_query.len(),
            gen_w,
            ones_w,
            inf_count(&pk.a_query),
            G1A,
        );
        row(
            "B_g2",
            pk.b_g2_query.len(),
            gen_w,
            ones_w,
            inf_count(&pk.b_g2_query),
            G2A,
        );
        row(
            "B_g1",
            pk.b_g1_query.len(),
            gen_w,
            ones_w,
            inf_count(&pk.b_g1_query),
            G1A,
        );
        row(
            "L_g1",
            pk.l_query.len(),
            general_count(l),
            ones_count(l),
            inf_count(&pk.l_query),
            G1A,
        );
        h_ones += 0;
        let (hc, hnw, _) = row(
            "H_g1",
            pk.h_query.len(),
            n,
            h_ones,
            inf_count(&pk.h_query),
            G1A,
        );
        let _ = hc;

        // The dense MSM is the one worth costing out: every scalar reaches the bucket
        // loop, so its adds are exactly one per window per scalar plus the bucket merge.
        println!(
            "\nH MSM adds    {} mixed (n x windows) + {} bucket-merge",
            n * hnw,
            hnw * (1usize << (window_size(n) - 1))
        );
        println!("H base bytes  {:.1} MB", (n * G1A) as f64 / 1e6);
    }
}
