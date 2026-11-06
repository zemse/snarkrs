// Writes the snarkrs-gnark-solver test fixtures with gnark v0.14.0 itself: each R1CS as
// cs.WriteTo writes it, the assignment it is solved with, and what cs.Solve returns for it.
//
//	go run . ..
//
// Each circuit is written once as CIRCUIT.r1cs. Each case NAME solved over one gets NAME.json
// and, when gnark solves it, NAME.solution.bin: the 0xdeadbeef marker, then W, A and B, each
// as a u64 LE length and the fr.Element limbs in memory order. NAME.json names the circuit
// and holds the public values (without the one wire), the secret values, and gnark's error
// when it does not solve.
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"os"
	"path/filepath"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/std/math/bits"
)

// The IDs gnark derives for SP1's koalabear hints: FNV-1a of the qualified function name.
const (
	invEHint       solver.HintID = 3205574893
	invFHint       solver.HintID = 3795648912
	splitLimbsHint solver.HintID = 3038391954
	reduceHint     solver.HintID = 342246697
)

var koalabear = big.NewInt(2130706433)

func main() {
	out := ".."
	if len(os.Args) > 1 {
		out = os.Args[1]
	}

	low := lowLevel()
	writeR1cs(out, "lowlevel", low)
	solve(out, "lowlevel", "lowlevel", low, lowLevelAssignment(false))
	solve(out, "lowlevel_unsatisfied", "lowlevel", low, lowLevelAssignment(true))
	bad := lowLevelAssignment(false)
	// f = p fails SplitLimbsHint: it is not a KoalaBear element.
	bad.secret[2] = new(big.Int).Set(koalabear)
	solve(out, "lowlevel_hint_error", "lowlevel", low, bad)

	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &Front{Z: make([]frontend.Variable, 64)})
	check(err)
	front := ccs.(*cs.R1CS)
	writeR1cs(out, "frontend", front)
	solve(out, "frontend", "frontend", front, frontAssignment(false))
	solve(out, "frontend_unsatisfied", "frontend", front, frontAssignment(true))
}

type assignment struct {
	public, secret []*big.Int
}

func writeR1cs(dir, name string, r *cs.R1CS) {
	f, err := os.Create(filepath.Join(dir, name+".r1cs"))
	check(err)
	_, err = r.WriteTo(f)
	check(err)
	check(f.Close())
	fmt.Printf("%s: %d constraints, %d wires, %d levels\n", name, r.GetNbConstraints(),
		len(r.Public)+len(r.Secret)+r.NbInternalVariables, len(r.Levels))
}

// solve writes NAME.json and, when gnark solves the assignment, NAME.solution.bin.
func solve(dir, name, circuit string, r *cs.R1CS, a assignment) {
	w, err := witness.New(ecc.BN254.ScalarField())
	check(err)
	values := make(chan any)
	go func() {
		for _, v := range a.public {
			values <- v
		}
		for _, v := range a.secret {
			values <- v
		}
		close(values)
	}()
	check(w.Fill(len(a.public), len(a.secret), values))

	meta := map[string]any{"circuit": circuit, "public": strs(a.public), "secret": strs(a.secret)}
	sol, err := r.Solve(w, solver.OverrideHint(invEHint, invE), solver.OverrideHint(invFHint, invF),
		solver.OverrideHint(splitLimbsHint, splitLimbs), solver.OverrideHint(reduceHint, reduce))
	if err != nil {
		meta["error"] = err.Error()
	} else {
		s := sol.(*cs.R1CSSolution)
		var buf []byte
		buf = binary.LittleEndian.AppendUint64(buf, 0xdeadbeef)
		for _, v := range []fr.Vector{s.W, s.A, s.B} {
			buf = binary.LittleEndian.AppendUint64(buf, uint64(len(v)))
			for _, e := range v {
				for _, l := range e {
					buf = binary.LittleEndian.AppendUint64(buf, l)
				}
			}
		}
		check(os.WriteFile(filepath.Join(dir, name+".solution.bin"), buf, 0o644))
	}
	j, err := json.MarshalIndent(meta, "", " ")
	check(err)
	check(os.WriteFile(filepath.Join(dir, name+".json"), append(j, '\n'), 0o644))
	fmt.Printf("%s: error=%v\n", name, meta["error"])
}

func strs(v []*big.Int) []string {
	s := make([]string, len(v))
	for i := range v {
		s[i] = v[i].String()
	}
	return s
}

// lowLevel builds an R1CS term by term, so each case the solver distinguishes is there on
// purpose: the unknown in L, in R and in O, under coefficients 1, -1 and others, an L
// unknown beside a zero B, assertions, constant terms in hint inputs, every hint the
// solver ships, and one level wide enough to be split across workers.
func lowLevel() *cs.R1CS {
	r := cs.NewR1CS(0)
	one := r.AddPublicVariable("1")
	x := r.AddPublicVariable("x")
	y := r.AddSecretVariable("y")
	z := r.AddSecretVariable("z")
	f := r.AddSecretVariable("f")
	var e [4]int
	for i := range e {
		e[i] = r.AddSecretVariable(fmt.Sprintf("e%d", i))
	}
	bp := r.AddBlueprint(&constraint.BlueprintGenericR1C{})

	t := func(c int64, v int) constraint.Term { return r.MakeTerm(r.FromInterface(c), v) }
	tb := func(c *big.Int, v int) constraint.Term { return r.MakeTerm(r.FromInterface(c), v) }
	constant := func(c int64) constraint.Term {
		term := r.MakeTerm(r.FromInterface(c), 0)
		term.MarkConstant()
		return term
	}
	le := func(ts ...constraint.Term) constraint.LinearExpression { return ts }
	add := func(l, rr, o constraint.LinearExpression) {
		r.AddR1C(constraint.R1C{L: l, R: rr, O: o}, bp)
	}
	internal := func() int { return r.AddInternalVariable() }
	hint := func(id solver.HintID, n int, in ...constraint.LinearExpression) []int {
		w, err := r.AddSolverHint(nil, id, in, n)
		check(err)
		return w
	}

	// O unknown: t1 = x*y.
	t1 := internal()
	add(le(t(1, x)), le(t(1, y)), le(t(1, t1)))
	// L unknown: t2*y = x.
	t2 := internal()
	add(le(t(1, t2)), le(t(1, y)), le(t(1, x)))
	// R unknown beside a known term: z*(t3 + 2x) = x + 3.
	t3 := internal()
	add(le(t(1, z)), le(t(1, t3), t(2, x)), le(t(1, x), t(3, one)))
	// O unknown under -1: x*x = -t4.
	t4 := internal()
	add(le(t(1, x)), le(t(1, x)), le(t(-1, t4)))
	// O unknown under 5, beside a constant: y*z = 5*t5 + 3.
	t5 := internal()
	add(le(t(1, y)), le(t(1, z)), le(t(5, t5), t(3, one)))
	// L unknown under -2 with B = 0: gnark checks A*B == C and sets the wire to 0.
	t6 := internal()
	add(le(t(-2, t6), t(1, x)), le(t(1, z), t(-1, z)), le())
	// O unknown from 2 and -2 coefficients: 2*t1 - 2*y = t7.
	t7 := internal()
	add(le(t(2, t1), t(-2, y)), le(t(1, one)), le(t(1, t7)))
	// L unknown under 7 (divided out by the coefficient's inverse): (7*t8)*z = t1.
	t8 := internal()
	add(le(t(7, t8)), le(t(1, z)), le(t(1, t1)))
	// R unknown under -1: y*(-t9) = x.
	t9 := internal()
	add(le(t(1, y)), le(t(-1, t9)), le(t(1, x)))
	// Assertion: t2*y = x again, nothing left to solve.
	add(le(t(1, t2)), le(t(1, y)), le(t(1, x)))

	// InvZeroHint over x + 2y + 7, the 7 a constant term, then inv*(x+2y+7) = 1.
	inv := hint(solver.GetHintID(solver.InvZeroHint), 1, le(t(1, x), t(2, y), constant(7)))
	add(le(t(1, inv[0])), le(t(1, x), t(2, y), t(7, one)), le(t(1, one)))
	// InvZeroHint over x - x is 0.
	zero := hint(solver.GetHintID(solver.InvZeroHint), 1, le(t(1, x), t(-1, x)))
	add(le(t(1, zero[0])), le(t(1, one)), le())

	// nBits(y, 12), recomposed: sum 2^i b_i = y.
	bitsID := solver.GetHintID(nBitsFn())
	b := hint(bitsID, 12, le(t(1, y)))
	var sum constraint.LinearExpression
	for i, w := range b {
		sum = append(sum, t(1<<i, w))
		add(le(t(1, w)), le(t(1, w)), le(t(1, w)))
	}
	add(sum, le(t(1, one)), le(t(1, y)))
	// ithBit(z, 3) and nTrits(z, 9), constant inputs as constant terms.
	ith := hint(solver.GetHintID(ithBitFn()), 1, le(t(1, z)), le(constant(3)))
	add(le(t(1, ith[0])), le(t(1, ith[0])), le(t(1, ith[0])))
	trits := hint(solver.GetHintID(nTritsFn()), 9, le(t(1, z)))
	var tsum constraint.LinearExpression
	p := int64(1)
	for _, w := range trits {
		tsum = append(tsum, t(p, w))
		p *= 3
	}
	add(tsum, le(t(1, one)), le(t(1, z)))

	// SP1's hints by their IDs. ReduceHint(t1) = (q, r) with t1 = q*p + r.
	qr := hint(reduceHint, 2, le(t(1, t1)))
	tq := internal()
	add(le(t(1, qr[0])), le(tb(koalabear, one)), le(t(1, tq)))
	add(le(t(1, tq), t(1, qr[1])), le(t(1, one)), le(t(1, t1)))
	// SplitLimbsHint(f) = (lo, hi), f = hi*2^24 + lo.
	lh := hint(splitLimbsHint, 2, le(t(1, f)))
	add(le(t(1<<24, lh[1]), t(1, lh[0])), le(t(1, one)), le(t(1, f)))
	// InvFHint(f), and the BN254 product f*finv as a wire of its own.
	finv := hint(invFHint, 1, le(t(1, f)))
	tf := internal()
	add(le(t(1, f)), le(t(1, finv[0])), le(t(1, tf)))
	// InvEHint(e0..e3), and its outputs feed one more R unknown each.
	ei := hint(invEHint, 4, le(t(1, e[0])), le(t(1, e[1])), le(t(1, e[2])), le(t(1, e[3])))
	for _, w := range ei {
		u := internal()
		add(le(t(1, e[0]), t(1, w)), le(t(1, u)), le(t(1, x)))
	}

	// 300 independent O unknowns in one level: x*(y + i) = w_i.
	for i := 0; i < 300; i++ {
		w := internal()
		add(le(t(1, x)), le(t(1, y), t(int64(i), one)), le(t(1, w)))
	}
	return r
}

func lowLevelAssignment(unsatisfied bool) assignment {
	n := func(s string) *big.Int {
		v, ok := new(big.Int).SetString(s, 0)
		if !ok {
			panic(s)
		}
		return v
	}
	x := n("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd")
	y := big.NewInt(3001)
	if unsatisfied {
		// x + 2y + 7 = 0, so inv*(x+2y+7) = 1 is the one constraint that fails.
		x = new(big.Int).Neg(big.NewInt(2*3001 + 7))
		x.Mod(x, fr.Modulus())
	}
	return assignment{
		public: []*big.Int{x},
		secret: []*big.Int{y, big.NewInt(12345), big.NewInt(1999999999),
			big.NewInt(7), big.NewInt(2130706432), big.NewInt(0), big.NewInt(123456789)},
	}
}

func nBitsFn() solver.Hint  { return hintByName("github.com/consensys/gnark/std/math/bits.nBits") }
func ithBitFn() solver.Hint { return hintByName("github.com/consensys/gnark/std/math/bits.ithBit") }
func nTritsFn() solver.Hint { return hintByName("github.com/consensys/gnark/std/math/bits.nTrits") }

func hintByName(name string) solver.Hint {
	for _, h := range bits.GetHints() {
		if solver.GetHintName(h) == name {
			return h
		}
	}
	panic("hint not registered: " + name)
}

// Front is a circuit the real frontend compiles, so the fixture carries the shapes gnark's
// own builder emits: Div and Inverse as L unknowns, IsZero, ToBinary, Select, ternary
// decomposition and a range check.
type Front struct {
	X frontend.Variable `gnark:",public"`
	Y frontend.Variable
	P frontend.Variable
	Z []frontend.Variable
}

func (c *Front) Define(api frontend.API) error {
	acc := c.X
	for i, z := range c.Z {
		m := api.Mul(z, c.X)
		d := api.Div(m, api.Add(z, 1))
		inv := api.Inverse(api.Add(z, 2))
		isz := api.IsZero(api.Sub(z, i))
		b := api.ToBinary(z, 16)
		s := api.Select(isz, d, inv)
		acc = api.Add(acc, api.Mul(s, b[0]), b[15])
	}
	api.AssertIsDifferent(acc, 0)
	api.AssertIsLessOrEqual(c.Y, 1000)
	t := bits.ToTernary(api, c.Y, bits.WithNbDigits(8))
	api.AssertIsEqual(bits.FromTernary(api, t), c.Y)
	api.AssertIsEqual(api.Mul(c.X, c.Y), c.P)
	return nil
}

func frontAssignment(unsatisfied bool) assignment {
	x := big.NewInt(987654321)
	y := big.NewInt(777)
	p := new(big.Int).Mul(x, y)
	if unsatisfied {
		p.Add(p, big.NewInt(1))
	}
	a := assignment{public: []*big.Int{x}, secret: []*big.Int{y, p}}
	for i := 0; i < 64; i++ {
		// z_i = i for every third i, so IsZero takes both branches.
		v := int64(i*977 + 5)
		if i%3 == 0 {
			v = int64(i)
		}
		a.secret = append(a.secret, big.NewInt(v))
	}
	return a
}

// The four koalabear hints with SP1's semantics. reduce and splitLimbs are SP1's code;
// invF uses big.Int's inverse and invE solves the 4x4 multiplication matrix by Gaussian
// elimination, so neither shares a method with the Rust side's exponentiation.

func reduce(_ *big.Int, in []*big.Int, out []*big.Int) error {
	out[0].Div(in[0], koalabear)
	out[1].Rem(in[0], koalabear)
	return nil
}

func splitLimbs(_ *big.Int, in []*big.Int, out []*big.Int) error {
	if in[0].Cmp(koalabear) >= 0 {
		return fmt.Errorf("input is not in the field")
	}
	two24 := big.NewInt(int64(math.Pow(2, 24)))
	out[0].Rem(in[0], two24)
	out[1].Quo(in[0], two24)
	return nil
}

func invF(_ *big.Int, in []*big.Int, out []*big.Int) error {
	a := new(big.Int).Mod(in[0], koalabear)
	if a.Sign() == 0 {
		return fmt.Errorf("inverse of zero")
	}
	out[0].ModInverse(a, koalabear)
	return nil
}

func invE(_ *big.Int, in []*big.Int, out []*big.Int) error {
	p := koalabear.Uint64()
	var a [4]uint64
	for i := range a {
		a[i] = new(big.Int).Mod(in[i], koalabear).Uint64()
	}
	// m[k][j] = coefficient of X^k in a*X^j, with X^4 = 3; solve m*v = (1, 0, 0, 0).
	var m [4][5]uint64
	for j := 0; j < 4; j++ {
		for k := 0; k < 4; k++ {
			if k >= j {
				m[k][j] = a[k-j]
			} else {
				m[k][j] = 3 * a[k-j+4] % p
			}
		}
	}
	m[0][4] = 1
	for col := 0; col < 4; col++ {
		piv := -1
		for r := col; r < 4; r++ {
			if m[r][col] != 0 {
				piv = r
				break
			}
		}
		if piv < 0 {
			return fmt.Errorf("inverse of zero")
		}
		m[col], m[piv] = m[piv], m[col]
		inv := new(big.Int).ModInverse(new(big.Int).SetUint64(m[col][col]), koalabear).Uint64()
		for j := col; j < 5; j++ {
			m[col][j] = m[col][j] * inv % p
		}
		for r := 0; r < 4; r++ {
			if r != col && m[r][col] != 0 {
				f := m[r][col]
				for j := col; j < 5; j++ {
					m[r][j] = (m[r][j] + (p-f)*m[col][j]) % p
				}
			}
		}
	}
	for i := 0; i < 4; i++ {
		out[i].SetUint64(m[i][4])
	}
	return nil
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
