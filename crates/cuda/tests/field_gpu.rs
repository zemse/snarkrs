//! The CUDA field layers, checked element by element against arkworks on a real device:
//! `Fr` (what the FFT stages compute in) and `Fq` / `Fq2` (what the MSM computes in).
//!
//! Every test here skips (rather than fails) when no NVIDIA device is present, because
//! this suite is expected to be run on the M2 Max where the rest of the workspace is
//! developed. A skip prints why, so a silent pass on a machine that *does* have a GPU is
//! not possible to mistake for a real one.

#![cfg(feature = "cuda")]

mod resources {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-support/resources.rs"
    ));
}

use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field, One, PrimeField, Zero};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use snarkrs_cuda::{as_words, from_words, Cuda};
use snarkrs_field::{Fq, Fq2, Fr};
use snarkrs_gpu_layout::testrng::SplitMix64;
use snarkrs_gpu_layout::{Packed, PackedFq, PackedFq2, PackedFr};

/// `None` plus a printed reason when there is no device, so the suite is honest about
/// what it did and did not check.
fn device() -> Option<Cuda> {
    match Cuda::new(0) {
        Ok(c) => {
            let (major, minor) = c.compute_capability();
            eprintln!(
                "cuda device: {} (sm_{major}{minor}, {} SMs)",
                c.device_name(),
                c.sm_count()
            );
            Some(c)
        }
        Err(e) => {
            resources::skip(format_args!("cuda field tests, no usable device: {e}"));
            None
        }
    }
}

const N: usize = 4096;

#[test]
fn fr_arithmetic_matches_arkworks_on_device() {
    let Some(cuda) = device() else { return };
    let stream = cuda.stream().clone();

    let src = snarkrs_cuda::kernels::unit_field_probe();
    let funcs = cuda
        .functions("field_probe", &src, &["fr_probe"])
        .expect("NVRTC compile of the field probe");
    let probe = &funcs[0];

    // Inputs worth having alongside random ones: the identities, the boundary at r-1, and
    // pairs whose sum and difference cross zero and the modulus.
    let mut rng = SplitMix64(0xF1E1D0);
    let mut a: Vec<Fr> = vec![
        Fr::zero(),
        Fr::one(),
        -Fr::one(),
        Fr::zero(),
        -Fr::one(),
        Fr::from(2u64),
    ];
    let mut b: Vec<Fr> = vec![
        Fr::zero(),
        Fr::one(),
        Fr::one(),
        -Fr::one(),
        -Fr::one(),
        -Fr::one(),
    ];
    while a.len() < N {
        a.push(rng.next_fr());
        b.push(rng.next_fr());
    }

    let pa = PackedFr::pack_slice(&a);
    let pb = PackedFr::pack_slice(&b);
    let d_a = stream.clone_htod(as_words(&pa)).unwrap();
    let d_b = stream.clone_htod(as_words(&pb)).unwrap();

    let words = N * 8;
    let mut outs: Vec<_> = (0..5)
        .map(|_| stream.alloc_zeros::<u32>(words).unwrap())
        .collect();

    let cfg = LaunchConfig {
        grid_dim: (N.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = N as u32;
    {
        let (o0, rest) = outs.split_at_mut(1);
        let (o1, rest) = rest.split_at_mut(1);
        let (o2, rest) = rest.split_at_mut(1);
        let (o3, o4) = rest.split_at_mut(1);
        let mut lb = stream.launch_builder(probe);
        lb.arg(&d_a)
            .arg(&d_b)
            .arg(&mut o0[0])
            .arg(&mut o1[0])
            .arg(&mut o2[0])
            .arg(&mut o3[0])
            .arg(&mut o4[0])
            .arg(&n_u32);
        unsafe { lb.launch(cfg) }.expect("fr_probe launch");
    }
    stream.synchronize().unwrap();

    let got: Vec<Vec<Fr>> = outs
        .iter()
        .map(|d| {
            let w = stream.clone_dtoh(d).unwrap();
            let packed: Vec<PackedFr> =
                from_words(&w).expect("output length is a whole number of Fr");
            PackedFr::unpack_slice(&packed)
        })
        .collect();

    for i in 0..N {
        assert_eq!(got[0][i], a[i] + b[i], "fr_add mismatch at {i}");
        assert_eq!(got[1][i], a[i] - b[i], "fr_sub mismatch at {i}");
        assert_eq!(got[2][i], a[i] * b[i], "fr_mul mismatch at {i}");
        assert_eq!(got[3][i], -a[i], "fr_neg mismatch at {i}");
        assert_eq!(got[4][i], a[i].square(), "fr_sqr mismatch at {i}");
    }
    eprintln!("{N} elements x 5 operations agree with arkworks");
}

#[test]
fn montgomery_constants_are_right_on_device() {
    let Some(cuda) = device() else { return };
    let stream = cuda.stream().clone();
    let src = snarkrs_cuda::kernels::unit_field_probe();
    let funcs = cuda
        .functions("field_probe", &src, &["fr_constants"])
        .expect("NVRTC compile");

    let mut out = stream.alloc_zeros::<u32>(4 * 8).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let mut lb = stream.launch_builder(&funcs[0]);
        lb.arg(&mut out);
        unsafe { lb.launch(cfg) }.expect("fr_constants launch");
    }
    stream.synchronize().unwrap();

    let w = stream.clone_dtoh(&out).unwrap();
    let p: Vec<PackedFr> = from_words(&w).unwrap();

    // fr_one() is the Montgomery representative of 1, so it must decode to 1.
    assert_eq!(p[0].to_fr(), Fr::one(), "fr_one is not 1");
    assert_eq!(p[1].to_fr(), Fr::zero(), "fr_zero is not 0");
    // fr_from_mont(fr_one()) is the *integer* 1, which as raw limbs is 1 and as a
    // Montgomery representative decodes to R^-1. Checking the limbs is the direct test.
    assert_eq!(p[2].v, [1, 0, 0, 0, 0, 0, 0, 0], "fr_from_mont(one) != 1");
    // And the round trip has to land back exactly where it started, which is what pins
    // FR_R2 down. A wrong R^2 passes every test above and fails only here.
    assert_eq!(p[3], p[0], "fr_to_mont(fr_from_mont(x)) != x");
}

// ---------------------------------------------------------------------------------------
// Fq and Fq2: the MSM's fields. The kernels are a separate copy of the Fr layout in
// bn254_curve.cuh, so the Fr probe above says nothing about them, and they have the most
// to lose from a limb-level mistake: a wrong `fq_mul` on one witness in a million shows
// up as a whole MSM disagreeing, with nothing to say which multiply it was.
// ---------------------------------------------------------------------------------------

/// Enough random pairs that a mistake confined to a rare carry pattern has a real chance
/// of being hit, on top of the edge cross products below.
const N_Q: usize = 16384;

/// Flag bits written by `fq_probe` / `fq2_probe`, mirroring the `PROBE_FLAG_*` defines.
const FLAG_IS_ZERO: u32 = 1;
const FLAG_EQ: u32 = 2;

/// An element from the raw Montgomery limbs it will carry on the device. The carry chain
/// and the conditional subtraction see limbs, not decoded values, so the edge cases worth
/// having are limb patterns: every pattern below q is some element's representative, and
/// arkworks' arithmetic on that element is the right oracle for the device's arithmetic
/// on those limbs.
fn fq_from_limbs(limbs: BigInt<4>) -> Fq {
    assert!(limbs < Fq::MODULUS, "not a reduced representative");
    Fq::new_unchecked(limbs)
}

/// Uniform on `[0, q)` up to the usual negligible bias.
fn next_fq(rng: &mut SplitMix64) -> Fq {
    let mut b = [0u8; 32];
    for c in b.chunks_mut(8) {
        c.copy_from_slice(&rng.next_u64().to_le_bytes());
    }
    Fq::from_le_bytes_mod_order(&b)
}

/// Limb patterns at the edges of the carry chain, plus the decoded-value identities.
/// Everything that can force or just miss the conditional subtraction after an add, a
/// borrow through every limb in a sub, and a top limb at or near q's in a mul.
fn fq_edges() -> Vec<Fq> {
    let q = Fq::MODULUS;
    let minus = |k: u64| {
        let mut x = q;
        x.sub_with_borrow(&BigInt::from(k));
        x
    };
    let mut half_q_minus_1 = minus(1); // (q-1)/2
    half_q_minus_1.div2();
    let mut half_q_plus_1 = half_q_minus_1; // (q+1)/2
    half_q_plus_1.add_with_carry(&BigInt::from(1u64));
    let mut two_254_mod_q = BigInt::<4>([0, 0, 0, 1 << 62]); // 2^254 - q
    two_254_mod_q.sub_with_borrow(&q);
    let two = Fq::from(2u64);
    vec![
        Fq::zero(),
        Fq::one(),      // limbs R
        -Fq::one(),     // limbs q - R
        two.pow([256]), // the element R: limbs R^2 mod q
        fq_from_limbs(BigInt::from(1u64)),
        fq_from_limbs(BigInt::from(u32::MAX as u64)),
        fq_from_limbs(minus(1)),
        fq_from_limbs(minus(2)),
        fq_from_limbs(half_q_minus_1),
        fq_from_limbs(half_q_plus_1),
        fq_from_limbs(BigInt([0, 0, 0, 1 << 61])), // 2^253
        fq_from_limbs(BigInt([1, 0, 0, 1 << 61])), // 2^253 + 1
        fq_from_limbs(two_254_mod_q),
        // q's top limb is 0x30644e72 and the ones below it are not all set, so this is the
        // largest all-ones tail that is still below q.
        fq_from_limbs(BigInt([
            u64::MAX,
            u64::MAX,
            u64::MAX,
            0x3064_4e71_ffff_ffff,
        ])),
        fq_from_limbs(BigInt([
            u64::MAX,
            u64::MAX,
            u64::MAX,
            0x0000_0000_ffff_ffff,
        ])), // 2^224 - 1
        fq_from_limbs(BigInt([
            0xaaaa_aaaa_aaaa_aaaa,
            0xaaaa_aaaa_aaaa_aaaa,
            0xaaaa_aaaa_aaaa_aaaa,
            0x2aaa_aaaa_aaaa_aaaa,
        ])),
    ]
}

/// Every ordered pair of edges (so the diagonal covers `a == b`, and both orders cover
/// `a - b` crossing zero each way), then random pairs up to `n`.
fn pairs<T: Clone>(edges: &[T], n: usize, mut random: impl FnMut() -> T) -> (Vec<T>, Vec<T>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for x in edges {
        for y in edges {
            a.push(x.clone());
            b.push(y.clone());
        }
    }
    assert!(a.len() < n, "edge cross product alone exceeds n");
    while a.len() < n {
        a.push(random());
        b.push(random());
    }
    (a, b)
}

/// Compiles the probe unit, launches `kernel` with `(a, b, out[0..ops], flags, n)` and
/// returns the outputs decoded plus the flags. The same shape serves `fq_probe` and
/// `fq2_probe`; a new operation is one more output here and one more assert in the test.
fn probe<T: Packed + Default>(
    cuda: &Cuda,
    kernel: &str,
    a: &[T],
    b: &[T],
    ops: usize,
) -> (Vec<Vec<T>>, Vec<u32>) {
    let stream = cuda.stream().clone();
    let src = snarkrs_cuda::kernels::unit_field_probe();
    let funcs = cuda
        .functions("field_probe", &src, &[kernel])
        .expect("NVRTC compile of the field probe");

    let n = a.len();
    assert_eq!(b.len(), n);
    let d_a = stream.clone_htod(as_words(a)).unwrap();
    let d_b = stream.clone_htod(as_words(b)).unwrap();
    let words = std::mem::size_of_val(a) / 4;
    let mut outs: Vec<CudaSlice<u32>> = (0..ops)
        .map(|_| stream.alloc_zeros::<u32>(words).unwrap())
        .collect();
    let mut flags = stream.alloc_zeros::<u32>(n).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    {
        let mut lb = stream.launch_builder(&funcs[0]);
        lb.arg(&d_a).arg(&d_b);
        for o in outs.iter_mut() {
            lb.arg(o);
        }
        lb.arg(&mut flags).arg(&n_u32);
        unsafe { lb.launch(cfg) }.unwrap_or_else(|e| panic!("{kernel} launch: {e}"));
    }
    stream.synchronize().unwrap();

    let got = outs
        .iter()
        .map(|d| {
            let w = stream.clone_dtoh(d).unwrap();
            from_words(&w).expect("output length is a whole number of elements")
        })
        .collect();
    (got, stream.clone_dtoh(&flags).unwrap())
}

#[test]
fn fq_arithmetic_matches_arkworks_on_device() {
    let Some(cuda) = device() else { return };

    let mut rng = SplitMix64(0xF0_0D15);
    let (a, b) = pairs(&fq_edges(), N_Q, || next_fq(&mut rng));
    let pa: Vec<PackedFq> = a.iter().map(PackedFq::from_fq).collect();
    let pb: Vec<PackedFq> = b.iter().map(PackedFq::from_fq).collect();

    let (got, flags) = probe(&cuda, "fq_probe", &pa, &pb, 6);

    for i in 0..N_Q {
        let (x, y) = (a[i], b[i]);
        let at = |op: &str| format!("{op} mismatch at {i}: a={:?} b={:?}", pa[i], pb[i]);
        assert_eq!(got[0][i].to_fq(), x + y, "{}", at("fq_add"));
        assert_eq!(got[1][i].to_fq(), x - y, "{}", at("fq_sub"));
        assert_eq!(got[2][i].to_fq(), x * y, "{}", at("fq_mul"));
        assert_eq!(got[3][i].to_fq(), -x, "{}", at("fq_neg"));
        assert_eq!(got[4][i].to_fq(), x.square(), "{}", at("fq_sqr"));
        assert_eq!(got[5][i].to_fq(), x.double(), "{}", at("fq_add(x, x)"));
        assert_eq!(
            (flags[i] & FLAG_IS_ZERO) != 0,
            x.is_zero(),
            "{}",
            at("fq_is_zero")
        );
        assert_eq!((flags[i] & FLAG_EQ) != 0, x == y, "{}", at("fq_eq"));
    }
    eprintln!(
        "{N_Q} Fq pairs ({} edge pairs) x 6 operations + 2 predicates agree with arkworks",
        fq_edges().len().pow(2)
    );
}

#[test]
fn fq2_arithmetic_matches_arkworks_on_device() {
    let Some(cuda) = device() else { return };

    // Fq2 edges are every (c0, c1) pair from a shorter Fq list: the cross product of the
    // full list would exceed N_Q on its own, and the Fq test already walks that list.
    let fq = fq_edges();
    let short: Vec<Fq> = [0usize, 1, 2, 4, 6, 10, 13]
        .iter()
        .map(|&k| fq[k])
        .collect();
    let mut edges = Vec::new();
    for &c0 in &short {
        for &c1 in &short {
            edges.push(Fq2::new(c0, c1));
        }
    }
    let mut rng = SplitMix64(0xF0_0D25);
    let (a, b) = pairs(&edges, N_Q, || {
        Fq2::new(next_fq(&mut rng), next_fq(&mut rng))
    });
    let pa: Vec<PackedFq2> = a.iter().map(PackedFq2::from_fq2).collect();
    let pb: Vec<PackedFq2> = b.iter().map(PackedFq2::from_fq2).collect();

    let (got, flags) = probe(&cuda, "fq2_probe", &pa, &pb, 6);

    for i in 0..N_Q {
        let (x, y) = (a[i], b[i]);
        let at = |op: &str| format!("{op} mismatch at {i}: a={:?} b={:?}", pa[i], pb[i]);
        assert_eq!(got[0][i].to_fq2(), x + y, "{}", at("fq2_add"));
        assert_eq!(got[1][i].to_fq2(), x - y, "{}", at("fq2_sub"));
        assert_eq!(got[2][i].to_fq2(), x * y, "{}", at("fq2_mul"));
        assert_eq!(got[3][i].to_fq2(), -x, "{}", at("fq2_neg"));
        assert_eq!(got[4][i].to_fq2(), x.square(), "{}", at("fq2_sqr"));
        assert_eq!(got[5][i].to_fq2(), x.double(), "{}", at("fq2_add(x, x)"));
        assert_eq!(
            (flags[i] & FLAG_IS_ZERO) != 0,
            x.is_zero(),
            "{}",
            at("fq2_is_zero")
        );
        assert_eq!((flags[i] & FLAG_EQ) != 0, x == y, "{}", at("fq2_eq"));
    }
    eprintln!(
        "{N_Q} Fq2 pairs ({} edge pairs) x 6 operations + 2 predicates agree with arkworks",
        edges.len().pow(2)
    );
}

#[test]
fn fq_constants_are_right_on_device() {
    let Some(cuda) = device() else { return };
    let stream = cuda.stream().clone();
    let src = snarkrs_cuda::kernels::unit_field_probe();
    let funcs = cuda
        .functions("field_probe", &src, &["fq_constants"])
        .expect("NVRTC compile");

    let mut out = stream.alloc_zeros::<u32>(4 * 8).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let mut lb = stream.launch_builder(&funcs[0]);
        lb.arg(&mut out);
        unsafe { lb.launch(cfg) }.expect("fq_constants launch");
    }
    stream.synchronize().unwrap();

    let w = stream.clone_dtoh(&out).unwrap();
    let p: Vec<PackedFq> = from_words(&w).unwrap();

    // FQ_R, limb for limb against arkworks' Montgomery one. None of the element-wise
    // probes reads FQ_R, so this is the only place a wrong value can show.
    assert_eq!(p[0], PackedFq::from_fq(&Fq::one()), "fq_one is not R mod q");
    assert_eq!(p[1], PackedFq::ZERO, "fq_zero is not 0");
    assert_eq!(p[2], p[0], "fq2_one().c0 is not fq_one()");
    assert_eq!(p[3], PackedFq::ZERO, "fq2_one().c1 is not 0");
}
