//! `raw.rs` add, sub and mul in three regimes, next to arkworks' own, single thread.
//!
//! - `lat`: one dependent chain, `x = op(x, v[i % 1024])`.
//! - `tput`: eight interleaved chains of the same.
//! - `stream`: `out[i] = op(a[i], b[i])` over 2^17 random pairs (8 MB in, 4 MB out), no
//!   repeats, so a data-dependent branch cannot be learned.
//!
//! Every kernel is checked against arkworks on its own inputs before it is timed.
//! Prints `field op mode ns_per_op` lines, the median of `REPS` (default 21).
//!
//! `cargo run --release -p snarkrs-field --example raw_ops`

use std::hint::black_box;
use std::time::Instant;

use ark_ff::UniformRand;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use snarkrs_field::raw::{RawFq, RawFr};
use snarkrs_field::{Fq, Fr};

const TABLE: usize = 1024;
const CHAIN: usize = 1 << 20;
const STREAM: usize = 1 << 17;

trait Ops: Copy {
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
}

macro_rules! raw_ops {
    ($t:ty) => {
        impl Ops for $t {
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                <$t>::add(self, o)
            }
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                <$t>::sub(self, o)
            }
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                <$t>::mul(self, o)
            }
        }
    };
}
raw_ops!(RawFq);
raw_ops!(RawFr);

macro_rules! ark_ops {
    ($t:ty) => {
        impl Ops for $t {
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                self + o
            }
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                self - o
            }
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                self * o
            }
        }
    };
}
ark_ops!(Fq);
ark_ops!(Fr);

const ADD: u8 = 0;
const SUB: u8 = 1;
const MUL: u8 = 2;

/// The op as a const parameter, so each timed loop is compiled for exactly one op.
#[inline(always)]
fn apply<const OP: u8, T: Ops>(a: T, b: T) -> T {
    match OP {
        ADD => a.add(b),
        SUB => a.sub(b),
        _ => a.mul(b),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[inline(never)]
fn lat<const OP: u8, T: Ops>(seed: T, v: &[T]) -> T {
    let mut x = seed;
    for i in 0..CHAIN {
        x = apply::<OP, T>(x, v[i % TABLE]);
    }
    x
}

#[inline(never)]
fn tput<const OP: u8, T: Ops>(seeds: [T; 8], v: &[T]) -> [T; 8] {
    let mut x = seeds;
    for i in 0..CHAIN / 8 {
        let y = v[i % TABLE];
        for l in &mut x {
            *l = apply::<OP, T>(*l, y);
        }
    }
    x
}

#[inline(never)]
fn stream<const OP: u8, T: Ops>(a: &[T], b: &[T], out: &mut [T]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = apply::<OP, T>(*x, *y);
    }
}

/// Times one op of one field in three regimes, after checking it against arkworks on
/// the same inputs. `to` maps the field to the arkworks element it should equal.
fn one<const OP: u8, T: Ops + PartialEq, A: Ops + PartialEq + std::fmt::Debug>(
    name: &str,
    data: &[A],
    from: &impl Fn(&A) -> T,
    to: &impl Fn(T) -> A,
    reps: usize,
) {
    let label = ["add", "sub", "mul"][OP as usize];
    let v: Vec<T> = data[..TABLE].iter().map(from).collect();
    let a: Vec<T> = data[..STREAM].iter().map(from).collect();
    let b: Vec<T> = data[STREAM..2 * STREAM].iter().map(from).collect();
    let mut out = a.clone();
    stream::<OP, T>(&a, &b, &mut out);
    for i in 0..STREAM {
        assert_eq!(
            to(out[i]),
            apply::<OP, A>(data[i], data[STREAM + i]),
            "{name} {label} stream {i}"
        );
    }
    let mut x = from(&data[0]);
    let mut y = data[0];
    for i in 0..4096 {
        x = apply::<OP, T>(x, v[i % TABLE]);
        y = apply::<OP, A>(y, data[i % TABLE]);
        assert_eq!(to(x), y, "{name} {label} chain {i}");
    }

    let seeds: [T; 8] = std::array::from_fn(|i| v[i]);
    let (mut l, mut t, mut s) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..reps {
        let t0 = Instant::now();
        black_box(lat::<OP, T>(black_box(v[0]), black_box(&v)));
        l.push(t0.elapsed().as_secs_f64() * 1e9 / CHAIN as f64);
        let t0 = Instant::now();
        black_box(tput::<OP, T>(black_box(seeds), black_box(&v)));
        t.push(t0.elapsed().as_secs_f64() * 1e9 / CHAIN as f64);
        let t0 = Instant::now();
        stream::<OP, T>(black_box(&a), black_box(&b), black_box(&mut out));
        s.push(t0.elapsed().as_secs_f64() * 1e9 / STREAM as f64);
    }
    println!("{name} {label} lat {:.3}", median(l));
    println!("{name} {label} tput {:.3}", median(t));
    println!("{name} {label} stream {:.3}", median(s));
}

fn run<T: Ops + PartialEq, A: Ops + PartialEq + std::fmt::Debug>(
    name: &str,
    data: &[A],
    from: impl Fn(&A) -> T,
    to: impl Fn(T) -> A,
    reps: usize,
) {
    one::<ADD, T, A>(name, data, &from, &to, reps);
    one::<SUB, T, A>(name, data, &from, &to, reps);
    one::<MUL, T, A>(name, data, &from, &to, reps);
}

fn main() {
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);
    let mut rng = StdRng::seed_from_u64(0x7261_775f_6f70);
    let fq: Vec<Fq> = (0..2 * STREAM).map(|_| Fq::rand(&mut rng)).collect();
    let fr: Vec<Fr> = (0..2 * STREAM).map(|_| Fr::rand(&mut rng)).collect();
    run("RawFq", &fq, RawFq::from_fq, RawFq::to_fq, reps);
    run("RawFr", &fr, RawFr::from_fr, RawFr::to_fr, reps);
    run("ArkFq", &fq, |x| *x, |x| x, reps);
    run("ArkFr", &fr, |x| *x, |x| x, reps);
}
