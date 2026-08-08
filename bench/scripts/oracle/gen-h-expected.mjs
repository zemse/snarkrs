import * as binFileUtils from "@iden3/binfileutils";
import { buildBn128 } from "ffjavascript";
import { Scalar, BigBuffer } from "ffjavascript";
import fs from "fs";

const dir = process.argv[2];
const curve = await buildBn128();
const Fr = curve.Fr;

const {fd: fdZKey, sections: sZ} = await binFileUtils.readBinFile(dir+"/circuit.zkey","zkey",2,1<<25,1<<23);
const {fd: fdW, sections: sW} = await binFileUtils.readBinFile(dir+"/circuit.wtns","wtns",2,1<<25,1<<23);

// read groth16 header manually
const s2 = await binFileUtils.readSection(fdZKey, sZ, 2);
let o=0;
const n8q = new DataView(s2.buffer,s2.byteOffset).getUint32(0,true); o=4+n8q;
const n8r = new DataView(s2.buffer,s2.byteOffset).getUint32(o,true); o+=4+n8r;
const dv = new DataView(s2.buffer,s2.byteOffset);
const nVars = dv.getUint32(o,true), nPublic=dv.getUint32(o+4,true), domainSize=dv.getUint32(o+8,true);
const power = Math.round(Math.log2(domainSize));

const buffWitness = await binFileUtils.readSection(fdW, sW, 2);
const buffCoeffs  = await binFileUtils.readSection(fdZKey, sZ, 4);

const n8 = Fr.n8;
const sCoef = 4*3 + n8r;
const nCoef = (buffCoeffs.byteLength-4)/sCoef;
const A = new Uint8Array(domainSize*n8), B = new Uint8Array(domainSize*n8), C = new Uint8Array(domainSize*n8);
const outBuf=[A,B];
for (let i=0;i<nCoef;i++){
  const bc = buffCoeffs.slice(4+i*sCoef, 4+i*sCoef+sCoef);
  const v = new DataView(bc.buffer, bc.byteOffset);
  const m=v.getUint32(0,true), c=v.getUint32(4,true), s=v.getUint32(8,true);
  const coef = bc.slice(12,12+n8);
  outBuf[m].set(Fr.add(outBuf[m].slice(c*n8,c*n8+n8), Fr.mul(coef, buffWitness.slice(s*n8,s*n8+n8))), c*n8);
}
for (let i=0;i<domainSize;i++){
  C.set(Fr.mul(A.slice(i*n8,i*n8+n8), B.slice(i*n8,i*n8+n8)), i*n8);
}
const inc = power == Fr.s ? Fr.shift : Fr.w[power+1];
const tr = async (buf) => {
  let x = await Fr.ifft(buf);
  x = await Fr.batchApplyKey(x, Fr.e(1), inc);
  return await Fr.fft(x);
};
const At = await tr(A), Bt = await tr(B), Ct = await tr(C);
const out=[];
for (let i=0;i<domainSize;i++){
  const p = Fr.sub(Fr.mul(At.slice(i*n8,i*n8+n8), Bt.slice(i*n8,i*n8+n8)), Ct.slice(i*n8,i*n8+n8));
  out.push(Fr.toString(p));
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
console.log("domainSize",domainSize,"nVars",nVars,"nPublic",nPublic,"first",out[0],"last",out[out.length-1]);
await fdZKey.close(); await fdW.close(); await curve.terminate();
