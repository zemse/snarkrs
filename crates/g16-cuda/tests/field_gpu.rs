//! The CUDA Fr layer, checked element by element against arkworks on a real device.
//!
//! Every test here skips (rather than fails) when no NVIDIA device is present, because
//! this suite is expected to be run on the M2 Max where the rest of the workspace is
//! developed. A skip prints why, so a silent pass on a machine that *does* have a GPU is
//! not possible to mistake for a real one.

#![cfg(feature = "cuda")]

use ark_ff::{Field, One, Zero};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use g16_cuda::{as_words, from_words, Cuda};
use g16_field::Fr;
use g16_gpu_layout::testrng::SplitMix64;
use g16_gpu_layout::PackedFr;

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
            eprintln!("SKIPPING cuda field tests, no usable device: {e}");
            None
        }
    }
}

const N: usize = 4096;

#[test]
fn fr_arithmetic_matches_arkworks_on_device() {
    let Some(cuda) = device() else { return };
    let stream = cuda.stream().clone();

    let src = g16_cuda::kernels::unit_field_probe();
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
    let src = g16_cuda::kernels::unit_field_probe();
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
