//! The constraint matrices themselves, for a caller that needs `L`, `R` and `O` rather than a
//! solution: what converting a gnark key to a zkey writes into section 4.

use crate::r1cs::{uvarint, Op};
use crate::GnarkR1cs;
use snarkrs_field::Fr;

/// One term of a linear expression, `coefficients()[coefficient] * w[wire]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub coefficient: u32,
    pub wire: u32,
}

impl GnarkR1cs {
    /// The coefficient table a [`Term`] indexes. gnark puts 0, 1, 2 and -1 at IDs 0 to 3.
    pub fn coefficients(&self) -> &[Fr] {
        &self.coefficients
    }

    /// `f(constraint, L, R, O)` for each R1C among instructions `ops`, in level order, hints
    /// skipped. Every term is in range, as the reader checked; a zero-coefficient term (ID 0)
    /// is passed on as the file holds it.
    ///
    /// # Panics
    /// If `ops` reaches past [`GnarkR1cs::nb_instructions`].
    pub fn for_each_r1c(
        &self,
        ops: std::ops::Range<usize>,
        mut f: impl FnMut(usize, &[Term], &[Term], &[Term]),
    ) {
        let bytes = self.calldata.bytes();
        let mut terms = Vec::new();
        for op in &self.ops[ops] {
            if op.solve == Op::HINT {
                continue;
            }
            let mut pos = op.start as usize;
            // SAFETY: the op's offset starts an R1C's calldata, which the reader checked
            // holds 4 + 2 * (nl + nr + no) values, every one a u32.
            let mut next = || unsafe { uvarint(bytes, &mut pos) };
            let _len = next();
            let (nl, nr, no) = (next() as usize, next() as usize, next() as usize);
            terms.clear();
            for _ in 0..nl + nr + no {
                let (coefficient, wire) = (next(), next());
                terms.push(Term { coefficient, wire });
            }
            let (l, rest) = terms.split_at(nl);
            let (r, o) = rest.split_at(nr);
            f(op.constraint as usize, l, r, o);
        }
    }
}
