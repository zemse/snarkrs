import { buildBn128 } from "ffjavascript";
import fs from "fs";
const curve = await buildBn128(); const Fr = curve.Fr;
const n8 = Fr.n8;
for (const log of [13,14,16]) {
  const n = 1<<log;
  // deterministic pseudo-random inputs, same xorshift the rust test uses
  let s = BigInt(0xC0FFEE + log) | 1n;
  const M = (1n<<64n)-1n;
  const vals = [];
  const buf = new Uint8Array(n*n8);
  for (let i=0;i<n;i++){
    s ^= (s<<13n)&M; s &= M; s ^= s>>7n; s ^= (s<<17n)&M; s &= M;
    vals.push(s);
    buf.set(Fr.e(s), i*n8);
  }
  const out = await Fr.fft(buf);
  const res = [];
  for (let i=0;i<n;i++) res.push(Fr.toString(out.slice(i*n8,(i+1)*n8)));
  fs.writeFileSync(process.argv[2]+"/fft_"+log+".json", JSON.stringify({log, input: vals.map(String), output: res}));
  console.log("log",log,"in0",vals[0].toString(),"out0",res[0]);
}
await curve.terminate();
