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
  const calls = [], reports = [], instances = [], deadlines = [];
  const document = { visibilityState: 'visible' };
  const proof = { proof: {}, publicSignals: ['1'], timings: {}, wallMs: 2 };
  class Prover {
    constructor() { this.index = instances.length; this.terminated = 0; instances.push(this); }
    async init() {
      await options.init?.(this.index);
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
    async selftest() { return await options.selftest?.(this.index) ?? { ok: true }; }
    async unload() { await options.unload?.(); }
    terminate() { this.terminated++; }
  }
  const search = '?circuits=railgun-01x01,tornado&warmup=1&reps=2&report=1' + (options.query ?? '');
  const load = modules({
    location: { search, origin: 'https://test.invalid', href: `https://test.invalid/${search}` },
    navigator: { userAgent: 'Node mock, no GPU', hardwareConcurrency: 1 },
    self: {}, document,
    setInterval: () => 1, clearInterval() {}, AbortController,
    setTimeout(fn, ms) { deadlines.push(ms); return setTimeout(fn, options.fastDeadlines ? 5 : ms); },
    clearTimeout,
    async fetch(url, init) {
      assert.equal(url, '/__report');
      reports.push(JSON.parse(init.body));
      await options.report?.(init.signal);
      return { ok: true, status: 200 };
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
  return { run: new Run(), calls, reports, instances, deadlines };
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

async function transport() {
  let bridge;
  class Worker {
    constructor() { bridge = this; this.sent = []; this.terminations = 0; }
    postMessage(data) {
      if (this.sendError) throw new Error('send failed');
      this.sent.push(data);
    }
    terminate() { this.terminations++; }
  }
  const { Prover } = await modules({}, {
    './worker/prover?worker': { default: Worker }
  })('lib/prover.ts');
  return { prover: new Prover(), get bridge() { return bridge; } };
}

for (const fault of ['error', 'messageerror', 'send', 'terminate']) {
  test(`terminal worker ${fault} rejects pending and future calls once`, async () => {
    const { prover, bridge } = await transport();
    const first = prover.prove(true), second = prover.prepare();
    const checked = [assert.rejects(first, /prover worker:/), assert.rejects(second, /prover worker:/)];
    if (fault === 'error') bridge.onerror({ message: 'module failed' });
    if (fault === 'messageerror') bridge.onmessageerror({});
    if (fault === 'send') {
      bridge.sendError = true;
      checked.push(assert.rejects(prover.unload(), /send failed/));
    }
    if (fault === 'terminate') prover.terminate();
    await Promise.all(checked);
    const sent = bridge.sent.length;
    await assert.rejects(prover.selftest(), /prover worker:/);
    await assert.rejects(prover.unload(), /prover worker:/);
    prover.terminate();
    bridge.onerror({ message: 'late failure' });
    assert.equal(bridge.sent.length, sent);
    assert.equal(bridge.terminations, 1);
  });
}

test('ordinary worker operation errors leave diagnostics and cleanup usable', async () => {
  const { prover, bridge } = await transport();
  const failed = prover.prove(true);
  bridge.onmessage({ data: { id: bridge.sent.at(-1).id, ok: false, error: 'bad proof' } });
  await assert.rejects(failed, /bad proof/);
  for (const operation of ['selftest', 'unload']) {
    const pending = prover[operation]();
    bridge.onmessage({ data: { id: bridge.sent.at(-1).id, ok: true, value: 'reported' } });
    assert.equal(await pending, 'reported');
  }
  assert.equal(bridge.terminations, 0);
  prover.terminate();
});

const stalled = () => new Promise(() => {});
for (const constantWork of [false, true]) {
  for (const fault of ['diagnostic-init', 'diagnostic-selftest', 'report']) {
    test(`${fault} deadline preserves startup error and unlocks mode ${constantWork}`, async () => {
      let signal;
      const f = await runner({
        fastDeadlines: true,
        init(index) {
          if (index === 0) throw new Error('primary startup failure');
          if (fault === 'diagnostic-init') return stalled();
        },
        selftest: fault === 'diagnostic-selftest' ? stalled : undefined,
        report(s) { signal = s; if (fault === 'report') return stalled(); }
      });
      f.run.constantWork = constantWork;
      await f.run.start();
      assert.equal(f.run.fatal, 'primary startup failure');
      assert.equal(f.run.phase, 'error');
      assert.equal(f.run.busy, false);
      assert.equal(f.run.resultConstantWork, constantWork);
      assert.equal(f.reports[0].fatal, 'primary startup failure');
      assert.equal(f.reports[0].constantWork, constantWork);
      assert.ok(f.instances.every((p) => p.terminated > 0));
      if (fault === 'report') {
        assert.equal(signal.aborted, true);
        assert.match(f.run.env.reportError, /timed out after 10000/);
      } else {
        assert.match(f.run.env.selftest.error, /timed out after 30000/);
      }
      f.run.constantWork = !constantWork;
      assert.equal(f.run.constantWork, !constantWork);
    });
  }
  test(`unload deadline preserves proving errors and unlocks mode ${constantWork}`, async () => {
    const f = await runner({
      fastDeadlines: true,
      prove() { throw new Error('primary proving failure'); },
      unload: stalled
    });
    f.run.constantWork = constantWork;
    await f.run.start();
    assert.equal(f.run.busy, false);
    assert.ok(f.run.rows.every((r) => r.error === 'primary proving failure'));
    assert.equal(f.run.env.cleanupErrors.length, 2);
    assert.ok(f.run.env.cleanupErrors.every((e) => /timed out after 10000/.test(e.error)));
    assert.deepEqual(f.calls, [constantWork, constantWork]);
    assert.ok(f.reports[0].rows.every((r) => r.error === 'primary proving failure' && r.constantWork === constantWork));
  });
  test(`slow legitimate proof has no deadline in mode ${constantWork}`, async () => {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const f = await runner({ fastDeadlines: true, prove: () => gate });
    f.run.constantWork = constantWork;
    const running = f.run.start();
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(f.run.busy, true);
    assert.deepEqual(f.deadlines, []);
    release();
    await running;
    assert.ok(f.run.rows.every((r) => r.status === 'done'));
    assert.deepEqual(f.calls, Array(6).fill(constantWork));
  });
}

async function importFormat() {
  return modules()('lib/format.ts');
}
