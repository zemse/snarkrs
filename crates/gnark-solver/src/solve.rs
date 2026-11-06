//! The solver: gnark's `constraint/bn254/solver.go` over a [`GnarkR1cs`], level by level.
//!
//! What gnark decides per term at run time (is this wire solved yet?) the reader already
//! decided once, so a level here is straight-line arithmetic: accumulate L, R and O into
//! A, B and C skipping the one unknown term, solve for it the way `solveR1C` does, and
//! record A and B. Every value is exact field arithmetic, so a different evaluation order
//! gives the same bits.

use crate::hints::Hint;
use crate::r1cs::{gnark_text, Op, CONSTANT};
use crate::{GnarkR1cs, Solution, SolveError};
use rayon::prelude::*;
use snarkrs_field::raw::RawFr;
use snarkrs_field::{Field, Fr, Zero};
use std::collections::BTreeMap;

/// A level shorter than this runs on the calling thread. Measured on SP1's circuit, where
/// 123,576 of the 135,582 levels are under 16 instructions: from 128 up, a level is worth
/// waking the pool for.
const PAR_LEVEL: usize = 128;

/// The fewest instructions one task takes from a parallel level. Tasks are small, 32 per
/// thread, so that work stealing evens out cores of different speeds (the M2 Max's four
/// efficiency cores) before the level's barrier rather than leaving it waiting on them.
const MIN_TASK: usize = 8;
const TASKS_PER_THREAD: usize = 32;

/// Solves gnark R1CS files, with a registry of hints by gnark's hint ID.
pub struct Solver {
    hints: BTreeMap<u32, (String, Box<Hint>)>,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// A solver with gnark's standard hints registered: `bits.nBits`, `bits.ithBit`,
    /// `bits.nTrits` and `solver.InvZeroHint`.
    pub fn new() -> Self {
        let mut s = Self::empty();
        crate::hints::register_std(&mut s);
        s
    }

    /// A solver with no hints.
    pub fn empty() -> Self {
        Self {
            hints: BTreeMap::new(),
        }
    }

    /// Register `f` under gnark's ID for the hint function `name`, the fully qualified Go
    /// name gnark hashes (`github.com/consensys/gnark/std/math/bits.nBits`).
    pub fn register(
        &mut self,
        name: &str,
        f: impl Fn(&[Fr], &mut [Fr]) -> Result<(), String> + Send + Sync + 'static,
    ) -> &mut Self {
        self.register_id(crate::hint_id(name), name, f)
    }

    /// Register `f` under an explicit hint ID, as `solver.OverrideHint` does. `name` is
    /// only for messages. A later registration of an ID replaces the earlier one.
    pub fn register_id(
        &mut self,
        id: u32,
        name: &str,
        f: impl Fn(&[Fr], &mut [Fr]) -> Result<(), String> + Send + Sync + 'static,
    ) -> &mut Self {
        self.hints.insert(id, (name.to_owned(), Box::new(f)));
        self
    }

    pub fn has_hint(&self, id: u32) -> bool {
        self.hints.contains_key(&id)
    }

    /// Solve `r1cs` for an assignment: `public` without the constant one wire, then
    /// `secret`, in wire order, as gnark's `witness.Vector()` holds them.
    pub fn solve(
        &self,
        r1cs: &GnarkR1cs,
        public: &[Fr],
        secret: &[Fr],
    ) -> Result<Solution, SolveError> {
        let mut s = Solution {
            w: zeroed(r1cs.nb_wires()),
            a: zeroed(r1cs.nb_constraints()),
            b: zeroed(r1cs.nb_constraints()),
        };
        self.solve_into(r1cs, public, secret, &mut s.w, &mut s.a, &mut s.b)?;
        Ok(s)
    }

    /// [`Solver::solve`] into a caller's buffers: `w` of [`GnarkR1cs::nb_wires`], `a` and
    /// `b` of [`GnarkR1cs::nb_constraints`]. Every slot is written, so their contents going
    /// in do not matter. On an error they hold a partial solve.
    pub fn solve_into(
        &self,
        r1cs: &GnarkR1cs,
        public: &[Fr],
        secret: &[Fr],
        w: &mut [Fr],
        a: &mut [Fr],
        b: &mut [Fr],
    ) -> Result<(), SolveError> {
        let (n_public, n_secret) = (r1cs.nb_public() - 1, r1cs.nb_secret());
        if public.len() != n_public || secret.len() != n_secret {
            return Err(SolveError::WitnessSize {
                public: public.len(),
                secret: secret.len(),
                expected_public: n_public,
                expected_secret: n_secret,
            });
        }
        for (what, len, want) in [
            ("w", w.len(), r1cs.nb_wires()),
            ("a", a.len(), r1cs.nb_constraints()),
            ("b", b.len(), r1cs.nb_constraints()),
        ] {
            if len != want {
                return Err(SolveError::BufferSize { what, len, want });
            }
        }
        // gnark refuses before solving when a hint the circuit depends on is missing, and
        // the reader listed every hint an instruction calls as well.
        let deps = r1cs.hint_dependencies();
        let missing: Vec<String> = deps
            .keys()
            .chain(&r1cs.hint_ids)
            .filter(|id| !self.hints.contains_key(id))
            .map(|id| deps.get(id).cloned().unwrap_or_else(|| id.to_string()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if !missing.is_empty() {
            return Err(SolveError::MissingHints(missing));
        }

        w[0] = Fr::from(1u64);
        w[1..1 + n_public].copy_from_slice(public);
        w[1 + n_public..1 + n_public + n_secret].copy_from_slice(secret);

        let run = Run {
            r1cs,
            hints: &self.hints,
            w: Shared::new(w),
            a: Shared::new(a),
            b: Shared::new(b),
        };
        for level in r1cs.level_ptr.windows(2) {
            let ops = &r1cs.ops[level[0]..level[1]];
            if ops.len() < PAR_LEVEL {
                run.ops(ops)?;
            } else {
                let task =
                    MIN_TASK.max(ops.len() / (TASKS_PER_THREAD * rayon::current_num_threads()));
                // Each task stops at its first error and the earliest one wins, so the
                // error is the same whichever thread finds it.
                let errors: Vec<SolveError> = ops
                    .par_chunks(task)
                    .filter_map(|chunk| run.ops(chunk).err())
                    .collect();
                if let Some(e) = errors.into_iter().next() {
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

/// A zeroed vector of field elements straight from the allocator, which hands back pages
/// the OS has already zeroed rather than writing every byte again.
fn zeroed(n: usize) -> Vec<Fr> {
    if n == 0 {
        return Vec::new();
    }
    let layout = std::alloc::Layout::array::<Fr>(n).expect("solution size overflows");
    // SAFETY: `layout` has a non-zero size. All-zero bytes are a valid `Fr`: its only
    // field is a `[u64; 4]` of Montgomery limbs (plus a zero-sized marker), and zero limbs
    // are the zero element. The pointer comes from the global allocator with the layout
    // `Vec<Fr>` itself uses for capacity `n`, so the `Vec` may own and free it.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut Fr;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
}

/// A slice the tasks of one level share. Each slot is written by one instruction only,
/// which is no other instruction of that level's to read, so no two threads ever touch one
/// slot unless both only read it: [`GnarkR1cs`]' reader proved that of the file.
#[derive(Clone, Copy)]
struct Shared {
    ptr: *mut Fr,
    #[cfg(debug_assertions)]
    len: usize,
}

// SAFETY: see the type's documentation; the accesses are disjoint by construction.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn new(s: &mut [Fr]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            #[cfg(debug_assertions)]
            len: s.len(),
        }
    }

    /// # Safety
    /// `i` is in range and not being written by another thread.
    #[inline(always)]
    unsafe fn get(&self, i: usize) -> Fr {
        #[cfg(debug_assertions)]
        assert!(i < self.len);
        *self.ptr.add(i)
    }

    /// # Safety
    /// `i` is in range and no other thread reads or writes it meanwhile.
    #[inline(always)]
    unsafe fn set(&self, i: usize, v: Fr) {
        #[cfg(debug_assertions)]
        assert!(i < self.len);
        *self.ptr.add(i) = v;
    }
}

struct Run<'a> {
    r1cs: &'a GnarkR1cs,
    hints: &'a BTreeMap<u32, (String, Box<Hint>)>,
    w: Shared,
    a: Shared,
    b: Shared,
}

impl Run<'_> {
    fn ops(&self, ops: &[Op]) -> Result<(), SolveError> {
        for op in ops {
            if op.solve == Op::HINT {
                self.hint(op)?;
            } else {
                self.r1c(op)?;
            }
        }
        Ok(())
    }

    /// `coefficient * w[wire]` added to `acc`, with gnark's shortcuts for IDs 0 to 3.
    ///
    /// # Safety
    /// `c` and `wire` are in range and `wire` is solved, as the reader proved of every term
    /// it handed the solver.
    #[inline(always)]
    unsafe fn accumulate(&self, acc: &mut RawFr, c: u32, wire: u32) {
        let x = || RawFr::from_fr(&self.w.get(wire as usize));
        match c {
            0 => {}
            1 => *acc = acc.add(x()),
            2 => {
                let x = x();
                *acc = acc.add(x.add(x))
            }
            3 => *acc = acc.sub(x()),
            _ => {
                let k = RawFr::from_fr(self.r1cs.coefficients.get_unchecked(c as usize));
                *acc = acc.add(k.mul(x()))
            }
        }
    }

    fn r1c(&self, op: &Op) -> Result<(), SolveError> {
        let cd = &self.r1cs.calldata[op.start as usize..];
        let (nl, nr, no) = (cd[1] as usize, cd[2] as usize, cd[3] as usize);
        let (end_l, end_r) = (4 + 2 * nl, 4 + 2 * (nl + nr));
        let end = end_r + 2 * no;
        let skip = op.solve as usize;
        let mut v = [RawFr::ZERO; 3];
        let mut k = 4;
        for (side, until) in [(0, end_l), (1, end_r), (2, end)] {
            while k < until {
                if k != skip {
                    // SAFETY: the reader checked this instruction's calldata and terms.
                    unsafe {
                        self.accumulate(
                            &mut v[side],
                            *cd.get_unchecked(k),
                            *cd.get_unchecked(k + 1),
                        )
                    };
                }
                k += 2;
            }
        }
        let [mut a, mut b, mut c] = v.map(RawFr::to_fr);
        let unsatisfied = |a: &Fr, b: &Fr, c: &Fr| SolveError::Unsatisfied {
            constraint: op.constraint as usize,
            message: format!("{} ⋅ {} != {}", gnark_text(a), gnark_text(b), gnark_text(c)),
        };
        if skip == 0 {
            if a * b != c {
                return Err(unsatisfied(&a, &b, &c));
            }
        } else {
            // The unknown term's value, `coefficient * wire`, as `solveR1C` finds it.
            let term = if skip < end_l {
                if b.is_zero() {
                    if !c.is_zero() {
                        return Err(unsatisfied(&a, &b, &c));
                    }
                    Fr::zero()
                } else {
                    let t = c * b.inverse().unwrap() - a;
                    a += t;
                    t
                }
            } else if skip < end_r {
                if a.is_zero() {
                    if !c.is_zero() {
                        return Err(unsatisfied(&a, &b, &c));
                    }
                    Fr::zero()
                } else {
                    let t = c * a.inverse().unwrap() - b;
                    b += t;
                    t
                }
            } else {
                let t = a * b - c;
                c += t;
                t
            };
            let _ = c;
            let wire = match cd[skip] {
                1 => term,
                3 => -term,
                id => term * self.r1cs.coefficient_inverses[id as usize],
            };
            // SAFETY: the reader gave this wire to this instruction alone.
            unsafe { self.w.set(cd[skip + 1] as usize, wire) };
        }
        // SAFETY: the reader gave this constraint index to this instruction alone.
        unsafe {
            self.a.set(op.constraint as usize, a);
            self.b.set(op.constraint as usize, b);
        }
        Ok(())
    }

    fn hint(&self, op: &Op) -> Result<(), SolveError> {
        let cd = &self.r1cs.calldata[op.start as usize..];
        let cd = &cd[..cd[0] as usize];
        let id = cd[1];
        let n_inputs = cd[2] as usize;
        let mut inputs = Vec::with_capacity(n_inputs);
        let mut j = 3;
        for _ in 0..n_inputs {
            let n = cd[j] as usize;
            let mut v = RawFr::ZERO;
            for &[c, wire] in cd[j + 1..j + 1 + 2 * n].as_chunks::<2>().0 {
                if wire == CONSTANT {
                    v = v.add(RawFr::from_fr(&self.r1cs.coefficients[c as usize]));
                } else {
                    // SAFETY: as for an R1C term; a zero coefficient's wire is not read.
                    unsafe { self.accumulate(&mut v, c, wire) };
                }
            }
            inputs.push(v.to_fr());
            j += 1 + 2 * n;
        }
        let (from, to) = (cd[j] as usize, cd[j + 1] as usize);
        let mut outputs = vec![Fr::zero(); to - from];
        let (name, f) = &self.hints[&id];
        f(&inputs, &mut outputs).map_err(|message| SolveError::Hint {
            id,
            name: name.clone(),
            message,
        })?;
        for (i, v) in outputs.into_iter().enumerate() {
            // SAFETY: the reader gave these wires to this instruction alone.
            unsafe { self.w.set(from + i, v) };
        }
        Ok(())
    }
}
