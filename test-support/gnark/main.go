// Command gnark-fixture writes a toy gnark v0.14.0 Groth16 BN254 fixture for snarkrs-gnark,
// and verifies a proof against a gnark verifying key with gnark's own groth16.Verify.
//
//	go run . gen <dir>
//	go run . r1cs <circuit.r1cs>
//	go run . verify <vk.bin> <public.bin> <proof.bin>
//
// gen writes, into <dir>:
//
//	pk.dump     ProvingKey.WriteDump
//	vk.bin      VerifyingKey.WriteTo (compressed points)
//	witness.bin unsafe.WriteMarker, then unsafe.WriteSlice of W (every wire, W[0] = 1),
//	            Aw and Bw (A.w and B.w per constraint), as []fr.Element
//	public.bin  the public inputs W[1..nbPublic], 32 bytes each, big-endian, canonical
//	proof.bin   a gnark proof of the same witness, Proof.WriteRawTo
//	circuit.r1cs the compiled circuit, cs.WriteTo
//
// r1cs writes just circuit.r1cs, which compiling makes the same for every setup.
//
// verify reads a verifying key written by VerifyingKey.WriteTo or WriteRawTo, a proof
// written by Proof.WriteRawTo or WriteTo, the public inputs in the public.bin layout above
// or as gnark's own public witness (witness.MarshalBinary, told apart by its 12-byte
// header), and exits 0 only when groth16.Verify accepts.
package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/utils/unsafe"
	"github.com/consensys/gnark/backend/groth16"
	groth16bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// rounds is the length of the squaring chain. Two constraints a round plus the 64-bit
// decomposition puts the circuit at 2,071 constraints, so the domain is 4096 and the
// prover has to pad A.w and B.w.
const rounds = 1000

// toy has four public inputs and no api.Commit. The bit decomposition gives wires that
// appear only in C, so pk.G1.A and pk.G1.B both drop points at infinity.
type toy struct {
	X   frontend.Variable `gnark:",public"`
	Y   frontend.Variable `gnark:",public"`
	Sum frontend.Variable `gnark:",public"`
	Low frontend.Variable `gnark:",public"`
	S   [rounds]frontend.Variable
}

func (c *toy) Define(api frontend.API) error {
	acc := c.X
	var sum frontend.Variable = 0
	for i := range c.S {
		acc = api.Mul(acc, acc)
		acc = api.Add(acc, api.Mul(c.S[i], c.X))
		sum = api.Add(sum, c.S[i])
	}
	api.AssertIsEqual(acc, c.Y)
	api.AssertIsEqual(sum, c.Sum)
	bits := api.ToBinary(c.S[0], 64)
	api.AssertIsEqual(api.FromBinary(bits[:16]...), c.Low)
	return nil
}

// assignment is the witness: S[i] = 3i + 7, X = 5, and Y, Sum and Low computed here.
func assignment() *toy {
	var a toy
	var x, acc, sum, s fr.Element
	x.SetUint64(5)
	acc.Set(&x)
	for i := range a.S {
		s.SetUint64(uint64(3*i + 7))
		a.S[i] = s
		var t fr.Element
		t.Mul(&s, &x)
		acc.Square(&acc).Add(&acc, &t)
		sum.Add(&sum, &s)
	}
	a.X = x
	a.Y = acc
	a.Sum = sum
	a.Low = 7 & 0xffff
	return &a
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "gnark-fixture:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	switch {
	case len(args) == 2 && args[0] == "gen":
		return gen(args[1])
	case len(args) == 2 && args[0] == "r1cs":
		ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &toy{})
		if err != nil {
			return fmt.Errorf("compile: %w", err)
		}
		var buf bytes.Buffer
		if _, err := ccs.WriteTo(&buf); err != nil {
			return err
		}
		return os.WriteFile(args[1], buf.Bytes(), 0o644)
	case len(args) == 4 && args[0] == "verify":
		return verify(args[1], args[2], args[3])
	default:
		return errors.New("usage: gen <dir> | r1cs <circuit.r1cs> | verify <vk.bin> <public.bin> <proof.bin>")
	}
}

func gen(dir string) error {
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &toy{})
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	r1, ok := ccs.(*cs.R1CS)
	if !ok {
		return errors.New("not a BN254 R1CS")
	}
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		return fmt.Errorf("setup: %w", err)
	}
	full, err := frontend.NewWitness(assignment(), ecc.BN254.ScalarField())
	if err != nil {
		return fmt.Errorf("witness: %w", err)
	}
	public, err := full.Public()
	if err != nil {
		return err
	}
	raw, err := r1.Solve(full)
	if err != nil {
		return fmt.Errorf("solve: %w", err)
	}
	sol := raw.(*cs.R1CSSolution)
	if len(sol.A) != r1.GetNbConstraints() || len(sol.B) != r1.GetNbConstraints() {
		return fmt.Errorf("solution has %d and %d rows for %d constraints", len(sol.A), len(sol.B), r1.GetNbConstraints())
	}
	proof, err := groth16.Prove(ccs, pk, full)
	if err != nil {
		return fmt.Errorf("prove: %w", err)
	}
	if err := groth16.Verify(proof, vk, public); err != nil {
		return fmt.Errorf("gnark does not verify its own proof: %w", err)
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := pk.(*groth16bn254.ProvingKey).WriteDump(&buf); err != nil {
		return err
	}
	if err := write(dir, "pk.dump", buf.Bytes()); err != nil {
		return err
	}
	buf.Reset()
	if _, err := vk.WriteTo(&buf); err != nil {
		return err
	}
	if err := write(dir, "vk.bin", buf.Bytes()); err != nil {
		return err
	}
	buf.Reset()
	if err := unsafe.WriteMarker(&buf); err != nil {
		return err
	}
	for _, v := range []fr.Vector{sol.W, sol.A, sol.B} {
		if err := unsafe.WriteSlice(&buf, []fr.Element(v)); err != nil {
			return err
		}
	}
	if err := write(dir, "witness.bin", buf.Bytes()); err != nil {
		return err
	}
	buf.Reset()
	nbPublic := r1.GetNbPublicVariables()
	for i := 1; i < nbPublic; i++ {
		b := sol.W[i].Bytes()
		buf.Write(b[:])
	}
	if err := write(dir, "public.bin", buf.Bytes()); err != nil {
		return err
	}
	buf.Reset()
	if _, err := proof.(*groth16bn254.Proof).WriteRawTo(&buf); err != nil {
		return err
	}
	if err := write(dir, "proof.bin", buf.Bytes()); err != nil {
		return err
	}
	buf.Reset()
	if _, err := ccs.WriteTo(&buf); err != nil {
		return err
	}
	if err := write(dir, "circuit.r1cs", buf.Bytes()); err != nil {
		return err
	}
	fmt.Printf("constraints %d, wires %d, public %d (with the constant wire), domain %d\n",
		r1.GetNbConstraints(), len(sol.W), nbPublic, pk.(*groth16bn254.ProvingKey).Domain.Cardinality)
	return nil
}

func verify(vkPath, publicPath, proofPath string) error {
	vk := groth16.NewVerifyingKey(ecc.BN254)
	if err := readFrom(vkPath, vk); err != nil {
		return fmt.Errorf("read vk: %w", err)
	}
	proof := groth16.NewProof(ecc.BN254)
	if err := readFrom(proofPath, proof); err != nil {
		return fmt.Errorf("read proof: %w", err)
	}
	raw, err := os.ReadFile(publicPath)
	if err != nil {
		return err
	}
	// A public witness is a 12-byte header and whole scalars, so never a multiple of 32.
	if len(raw)%fr.Bytes == 12 {
		public, err := witness.New(ecc.BN254.ScalarField())
		if err != nil {
			return err
		}
		if err := public.UnmarshalBinary(raw); err != nil {
			return fmt.Errorf("public witness: %w", err)
		}
		if err := groth16.Verify(proof, vk, public); err != nil {
			return fmt.Errorf("groth16.Verify: %w", err)
		}
		fmt.Println("ok")
		return nil
	}
	if len(raw)%fr.Bytes != 0 {
		return fmt.Errorf("public inputs are %d bytes, not a multiple of %d", len(raw), fr.Bytes)
	}
	n := len(raw) / fr.Bytes
	values := make(chan any, n)
	for i := 0; i < n; i++ {
		var e fr.Element
		if err := e.SetBytesCanonical(raw[i*fr.Bytes : (i+1)*fr.Bytes]); err != nil {
			return fmt.Errorf("public input %d: %w", i, err)
		}
		values <- e
	}
	close(values)
	public, err := witness.New(ecc.BN254.ScalarField())
	if err != nil {
		return err
	}
	if err := public.Fill(n, 0, values); err != nil {
		return err
	}
	if err := groth16.Verify(proof, vk, public); err != nil {
		return fmt.Errorf("groth16.Verify: %w", err)
	}
	fmt.Println("ok")
	return nil
}

func readFrom(path string, into io.ReaderFrom) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = into.ReadFrom(bufio.NewReader(f))
	return err
}

func write(dir, name string, b []byte) error {
	return os.WriteFile(filepath.Join(dir, name), b, 0o644)
}
