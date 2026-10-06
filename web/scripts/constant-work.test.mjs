import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';
import { compile, compileModule } from 'svelte/compiler';
import { render } from 'svelte/server';

// Execute the actual modules, with worker/WASM and browser I/O replaced at their boundaries.
// Server compilation lowers runes without a DOM; no adapter, shader or proof runs here.
function modules(globals = {}, imports = {}, wasm = {}) {
  const context = vm.createContext({ performance, URLSearchParams, console, ...globals });
  const cache = new Map();
  function stub(name, values) {
    if (!cache.has(name)) {
      cache.set(name, new vm.SyntheticModule(Object.keys(values), function () {
        for (const [key, value] of Object.entries(values)) this.setExport(key, value);
      }, { context, identifier: name }));
    }
    return cache.get(name);
  }
  function source(url) {
    if (cache.has(url.href)) return cache.get(url.href);
    const text = readFileSync(url, 'utf8');
    let code;
    if (url.pathname.endsWith('.svelte')) {
      code = compile(text, { filename: url.pathname, generate: 'server' }).js.code;
    } else {
      code = ts.transpileModule(text, {
        compilerOptions: { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext }
      }).outputText;
      if (url.pathname.endsWith('.svelte.ts')) {
        code = compileModule(code, { filename: url.pathname, generate: 'server' }).js.code;
      }
    }
    const mod = new vm.SourceTextModule(code, {
      context,
      identifier: url.href,
      initializeImportMeta(meta) { meta.env = { DEV: true }; },
      async importModuleDynamically() {
        const mod = stub('wasm', wasm);
        if (mod.status === 'unlinked') await mod.link(() => {});
        if (mod.status === 'linked') await mod.evaluate();
        return mod;
      }
    });
    cache.set(url.href, mod);
    return mod;
  }
  async function link(name, parent) {
    if (name in imports) return stub(name, imports[name]);
    if (name.startsWith('svelte')) return stub(name, await import(name));
    return source(new URL(`${name}.ts`, parent.identifier));
  }
  return async (path) => {
    const mod = source(new URL(`../src/${path}`, import.meta.url));
    await mod.link(link);
    await mod.evaluate();
    return mod.namespace;
  };
}

async function worker() {
  const calls = [], replies = [];
  let bridge;
  const scope = {
    postMessage(data) {
      replies.push(data);
      bridge?.onmessage?.({ data });
    }
  };
  class Worker {
    constructor() { bridge = this; }
    postMessage(data) { void scope.onmessage({ data }); }
    terminate() {}
  }
  const load = modules({ self: scope }, {
    '../pkg-version': { PKG_VERSION: 'test' },
    './worker/prover?worker': { default: Worker }
  }, {
    default: async () => {},
    start() {},
    create_prover: async () => {},
    caps: () => '{}',
    wasm_memory_bytes: () => 0,
    prove: async (constantWork) => {
      calls.push(constantWork);
      return JSON.stringify({ proof: { test: true }, publicSignals: ['1'], timings: {} });
    }
  });
  await load('lib/worker/prover.ts');
  const { Prover } = await load('lib/prover.ts');
  const prover = new Prover();
  await prover.init('/pkg', 'floor');
  return {
    prover, calls,
    async message(type, args = {}) {
      await scope.onmessage({ data: { id: 100, type, ...args } });
      return replies.at(-1);
    }
  };
}

for (const option of [undefined, false, true]) {
  test(`warm facade and worker forward ${option} to WASM`, async () => {
    const w = await worker();
    const result = await w.prover.prove(option);
    assert.deepEqual(w.calls, [option ?? false]);
    assert.equal(result.constantWork, option ?? false);
    assert.equal(result.proof.test, true);
    assert.ok(result.wallMs >= 0);
  });
}

test('worker defaults omitted options to variable-work and rejects non-booleans', async () => {
  const w = await worker();
  const result = await w.message('prove');
  assert.equal(result.ok, true);
  assert.equal(result.value.constantWork, false);
  for (const constantWork of ['true', 'false', 1, null]) {
    const result = await w.message('prove', { constantWork });
    assert.equal(result.ok, false);
    assert.match(result.error, /constantWork must be a boolean/);
  }
  assert.deepEqual(w.calls, [false]);
});

test('cold proving is not exposed by the worker without a retained-input loader', async () => {
  const w = await worker();
  const result = await w.message('prove_cold', { constantWork: true });
  assert.equal(result.ok, false);
  assert.match(result.error, /unknown message type "prove_cold"/);
  assert.deepEqual(w.calls, []);
});

async function runner(options = {}) {
  const calls = [], reports = [];
  const document = { visibilityState: 'visible' };
  const proof = { proof: {}, publicSignals: ['1'], timings: {}, wallMs: 2 };
  class Prover {
    async init() {
      await options.init?.();
      return { caps: {}, moduleMs: 0, deviceMs: 0 };
    }
    async load() { return { downloadedBytes: 1, downloadMs: 1, vkey: {} }; }
    async prepare() { return { wallMs: 1 }; }
    async prove(constantWork) {
      calls.push(constantWork);
      await options.prove?.(calls.length, document);
      return proof;
    }
    async verify() { return { verified: true }; }
    async trace() { return { text: 'GPU trace' }; }
    async cpuTrace() { return { text: 'CPU trace' }; }
    async proveHOnly() { return { wallMs: 1 }; }
    async proveMsmProbe() { return { wallMs: 1 }; }
    async unload() {}
    terminate() {}
  }
  const search = '?circuits=railgun-01x01,tornado&warmup=1&reps=2&report=1' + (options.query ?? '');
  const load = modules({
    location: { search, origin: 'https://test.invalid', href: `https://test.invalid/${search}` },
    navigator: { userAgent: 'Node mock, no GPU', hardwareConcurrency: 1 },
    self: {}, document,
    setInterval: () => 1, clearInterval() {},
    async fetch(url, init) {
      assert.equal(url, '/__report');
      reports.push(JSON.parse(init.body));
      await options.report?.();
    }
  }, {
    './prover': { Prover },
    './snarkjs': {
      snarkjsProve: async () => ({ ...proof, ms: [3, 3] }),
      snarkjsVerify: async () => true
    },
    ...(options.imports ?? {})
  });
  const { Run } = await load('lib/runner.svelte.ts');
  return { run: new Run(), calls, reports };
}

for (const constantWork of [false, true]) {
  test(`run snapshots ${constantWork} across warm-ups, reps, rows and reporting`, async () => {
    let expected = constantWork;
    const reportStates = [];
    let release;
    const opening = new Promise((resolve) => { release = resolve; });
    const f = await runner({
      init: () => opening,
      prove() {
        assert.equal(f.run.busy, true);
        f.run.constantWork = !expected;
        assert.equal(f.run.constantWork, expected);
      },
      async report() {
        reportStates.push([f.run.phase, f.run.busy, f.run.constantWork]);
        f.run.constantWork = !expected;
        await f.run.start();
        reportStates.push([f.run.phase, f.run.busy, f.run.constantWork]);
      }
    });
    assert.equal(f.run.constantWork, false);
    f.run.constantWork = constantWork;
    const running = f.run.start();
    assert.equal(f.run.phase, 'starting');
    f.run.constantWork = !constantWork;
    assert.equal(f.run.constantWork, constantWork);
    release();
    await running;
    assert.equal(f.run.phase, 'done');
    assert.equal(f.run.busy, false);
    assert.deepEqual(f.calls, Array(6).fill(constantWork));
    assert.equal(f.reports.length, 1);
    assert.equal(f.reports[0].constantWork, constantWork);
    for (const row of f.run.rows) {
      assert.equal(row.status, 'done');
      assert.equal(row.crossVerified, true);
      assert.equal(row.constantWork, constantWork);
      assert.equal(row.webgpuReps.length, 2);
    }
    assert.ok(f.reports[0].rows.every((r) => r.constantWork === constantWork));
    f.run.constantWork = !constantWork;
    assert.equal(f.run.constantWork, !constantWork);
    assert.equal(f.run.resultConstantWork, constantWork);
    assert.ok(f.run.diagnostics.includes(`work       ${constantWork ? 'constant-work' : 'variable-work'}`));
    expected = !constantWork;
    await f.run.start();
    assert.deepEqual(f.calls.slice(6), Array(6).fill(!constantWork));
    assert.equal(f.run.resultConstantWork, !constantWork);
    assert.deepEqual(reportStates, [
      ['done', true, constantWork], ['done', true, constantWork],
      ['done', true, !constantWork], ['done', true, !constantWork]
    ]);
    assert.equal(f.reports.length, 2);
    assert.equal(f.reports[1].constantWork, !constantWork);
  });
}

test('hidden-tab retries keep the selected mode', async () => {
  const f = await runner({
    prove(n, document) { document.visibilityState = n === 2 ? 'hidden' : 'visible'; },
    imports: {
      './visibility': {
        visible: async () => {},
        Discards: class { count() {} }
      }
    }
  });
  f.run.constantWork = true;
  await f.run.start();
  assert.deepEqual(f.calls, Array(7).fill(true));
  assert.ok(f.run.rows.every((r) => r.status === 'done' && r.webgpuReps.length === 2));
});

for (const query of ['&trace=1', '&stages=h', '&stages=msm:b-g2']) {
  test(`diagnostic mode ${query} cannot claim constant-work proofs`, async () => {
    const f = await runner({ query });
    assert.equal(f.run.supportsConstantWork, false);
    f.run.constantWork = true;
    assert.equal(f.run.constantWork, false);
    await f.run.start();
    assert.deepEqual(f.calls, []);
    assert.ok(f.run.rows.every((r) => r.status === 'done' && r.constantWork === undefined));
    assert.equal(f.reports[0].constantWork, false);
  });
}

test('startup failure unlocks selection but keeps the failed run mode', async () => {
  const f = await runner({ init() { throw new Error('mock startup failure'); } });
  f.run.constantWork = true;
  await f.run.start();
  assert.equal(f.run.phase, 'error');
  assert.equal(f.run.busy, false);
  f.run.constantWork = false;
  assert.equal(f.run.resultConstantWork, true);
  assert.equal(f.reports[0].constantWork, true);
  assert.deepEqual(f.calls, []);
});

test('page renders the selection, lock, warning and actual result mode', async () => {
  const f = await runner();
  f.run.constantWork = true;
  await f.run.start();
  f.run.constantWork = false;
  const load = modules({}, {
    '$lib/runner.svelte': { Run: class { constructor() { return f.run; } } },
    '$lib/support': { checkSupport: async () => ({ ok: false }) },
    '$lib/format': await importFormat()
  });
  const { default: Page } = await load('routes/+page.svelte');
  const html = render(Page).body;
  assert.match(html, /Constant-work WebGPU proofs \(opt-in\)/);
  assert.match(html, /Constant-work is not constant-time/);
  const input = html.match(/<input\b[^>]*>/)[0];
  assert.doesNotMatch(input, /\bchecked\b/);
  assert.doesNotMatch(input, /\bdisabled\b/);
  assert.match(html, /<span class="fine[^>]*>constant-work<\/span>/);
  const running = f.run.start();
  const locked = render(Page).body.match(/<input\b[^>]*>/)[0];
  assert.match(locked, /\bdisabled\b/);
  await running;
});

async function importFormat() {
  return modules()('lib/format.ts');
}
