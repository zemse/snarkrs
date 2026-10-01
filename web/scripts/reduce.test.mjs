import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../static/dbg/reduce.html', import.meta.url), 'utf8');
const script = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];

function probe(options = {}) {
  const out = { textContent: '' };
  const reports = [];
  let destroyed = false, maps = 0;
  const scopes = [];
  const dev = {
    lost: options.lost ? Promise.resolve({ reason: 'unknown', message: 'test loss' }) : new Promise(() => {}),
    destroy() { destroyed = true; },
    createShaderModule() {
      return { getCompilationInfo: async () => ({ messages: options.compileError ? [{ type: 'error', message: 'bad shader', lineNum: 1 }] : [] }) };
    },
    pushErrorScope(kind) { scopes.push(kind); },
    async popErrorScope() {
      return scopes.pop() === 'validation' && options.validationError ? { message: 'invalid binding' } : null;
    },
    async createComputePipelineAsync() {
      if (options.pipelineError) throw new Error('pipeline rejected');
      return { getBindGroupLayout() { return {}; } };
    },
    createBuffer({ size }) {
      return {
        async mapAsync() { maps++; },
        getMappedRange() {
          const words = new Uint32Array(size / 4);
          words[0] = options.differ && maps === 2 ? 2 : 1;
          return words.buffer;
        },
        unmap() {},
      };
    },
    createBindGroup() { return {}; },
    createCommandEncoder() {
      return {
        beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; },
        copyBufferToBuffer() {},
        finish() { return {}; },
      };
    },
    queue: { writeBuffer() {}, submit() {} },
  };
  const navigator = { userAgent: 'Node mock, not a phone' };
  if (!options.noGPU) navigator.gpu = { async requestAdapter() {
    if (options.noAdapter) return null;
    return { async requestDevice() {
      if (options.deviceError) throw new Error('device rejected');
      return dev;
    } };
  } };
  const context = vm.createContext({
    document: { getElementById: (id) => id === 'out' ? out : {} },
    navigator, isSecureContext: true, location: { search: '?report=1' }, URLSearchParams,
    Uint32Array, performance, setTimeout: (fn) => { fn(); },
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, COPY_SRC: 8, MAP_READ: 16 },
    GPUMapMode: { READ: 1 },
    async fetch(url, init) {
      if (url === '/__report') { reports.push(init.body); return { ok: true }; }
      return { ok: !options.fetchError, status: 404, text: async () => 'mock shader' };
    },
  });
  vm.runInContext(script, context);
  return { run: context.__run, out, reports, scopes, destroyed: () => destroyed };
}

for (const [name, options, verdict, detail] of [
  ['missing WebGPU', { noGPU: true }, 'UNAVAILABLE', 'no kernels tested'],
  ['null adapter', { noAdapter: true }, 'UNAVAILABLE', 'adapter unavailable'],
  ['device rejection', { deviceError: true }, 'ERROR', 'device rejected'],
  ['missing shader', { fetchError: true }, 'ERROR', 'HTTP 404'],
  ['compile error', { compileError: true }, 'ERROR', 'WGSL compilation failed'],
  ['pipeline rejection', { pipelineError: true }, 'ERROR', 'pipeline rejected'],
  ['validation error with equal readbacks', { validationError: true }, 'ERROR', 'invalid binding'],
  ['actual device loss', { lost: true }, 'LOST', 'test loss'],
  ['equal readbacks', {}, 'MATCH', 'outputs MATCH'],
  ['unequal readbacks', { differ: true }, 'DIFFER', 'outputs DIFFER'],
]) {
  test(name, async () => {
    const p = probe(options);
    assert.equal(await p.run(1, 1), verdict);
    assert.ok(p.out.textContent.includes(detail));
    assert.ok(p.out.textContent.includes('Node mock, not a phone'));
    assert.equal(p.reports.length, 1);
    assert.ok(p.reports[0].includes('verdict: ' + verdict));
    assert.deepEqual(p.scopes, []);
    assert.equal(p.destroyed(), !options.noGPU && !options.noAdapter && !options.deviceError);
    if (verdict !== 'MATCH') assert.ok(!p.out.textContent.includes('outputs MATCH'));
  });
}

test('invalid dimensions do not test a zero-sized or oversized dispatch', async () => {
  for (const [windows, buckets] of [[0, 1], [1, 0], [65, 1], [1, 257], [1.5, 1]]) {
    const p = probe();
    assert.equal(await p.run(windows, buckets), 'ERROR');
    assert.equal(p.destroyed(), false);
    assert.equal(p.reports.length, 1);
  }
});
