pragma circom 2.1.6;

include "rsa.circom";

// RSA-2048 PKCS#1 v1.5 signature verification with the near-universal public
// exponent 65537. This is the shape that zkemail and every "prove I received
// this email" circuit actually proves, so it is the one worth benchmarking.
//
// (121, 17) is the parameterisation zk-email itself uses: 121 bits per limb,
// 17 limbs. Two constraints drive that choice. n * k must exceed 2048 so the
// modulus fits (121 * 17 = 2057), and n must stay under 127 so a limb times a
// limb stays inside the BN254 scalar field without overflow.
//
// Only the modulus is public. The signature and the message digest are witness.
// Note that the PKCS#1 v1.5 DER padding is applied INSIDE the circuit by
// RSAPad, so `message` is the bare SHA-256 digest as an integer, not a padded
// block.
component main { public [modulus] } = RSAVerifier65537(121, 17);
