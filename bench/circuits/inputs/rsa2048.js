// Emit input.json for rsa2048.circom (RSAVerifier65537(121, 17)).
//
// The circuit does not just check a pairing-friendly encoding: it recomputes
// sig^65537 mod n and asserts it equals the PKCS#1 v1.5 block it builds
// internally, DER prefix and all. So the signature has to be genuinely valid
// or witness generation fails outright, well before proving. That is why this
// generates a real key and a real signature rather than random limbs.
//
// Limb encoding, from zk-email's own bigIntToChunkedBytes: 121 bits per limb,
// 17 limbs, LEAST significant limb first, each limb a decimal string.
const crypto = require("crypto");
const fs = require("fs");

const N = 121n;
const K = 17;

// Deterministic message, random key. The key must be fresh (node has no
// fixed-seed RSA keygen) but nothing about proving time depends on which key
// it is, only on the limb count, which is fixed.
const message = Buffer.from("gpu-snark rsa2048 benchmark");

const { privateKey, publicKey } = crypto.generateKeyPairSync("rsa", {
  modulusLength: 2048,
  publicExponent: 65537,
});

const sig = crypto.sign("RSA-SHA256", message, {
  key: privateKey,
  padding: crypto.constants.RSA_PKCS1_PADDING,
});

// Sanity: if node itself will not verify this, nothing downstream will.
if (!crypto.verify("RSA-SHA256", message, { key: publicKey, padding: crypto.constants.RSA_PKCS1_PADDING }, sig)) {
  throw new Error("generated signature does not verify");
}

const bytesToBig = (buf) => BigInt("0x" + buf.toString("hex"));

// jwk.n is the modulus, base64url, big-endian.
const jwk = publicKey.export({ format: "jwk" });
const modulus = bytesToBig(Buffer.from(jwk.n, "base64url"));

const digest = crypto.createHash("sha256").update(message).digest();

function chunk(x) {
  const mask = (1n << N) - 1n;
  const out = [];
  for (let i = 0; i < K; i++) out.push(((x >> (BigInt(i) * N)) & mask).toString());
  return out;
}

const input = {
  signature: chunk(bytesToBig(sig)),
  modulus: chunk(modulus),
  // The bare digest as an integer. RSAPad adds the DER prefix and the 0xff
  // padding inside the circuit, and asserts every bit above 256 is zero, so
  // handing it a pre-padded block here would fail the witness.
  message: chunk(bytesToBig(digest)),
};

if (modulus >> 2047n !== 1n) throw new Error("modulus is not a full 2048 bits");
fs.writeFileSync(process.argv[2] || "input.json", JSON.stringify(input));
console.log(`modulus ${modulus.toString(2).length} bits, signature ${sig.length} bytes, ${K} limbs of ${N} bits`);
