import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import { createWriteStream, readFileSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { once } from 'node:events';
import { finished } from 'node:stream/promises';
import { createServer as createHttpServer } from 'node:http';
import { createServer, mergeConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';

// Local correctness smoke, not a benchmark. Only the circuit catalogue is replaced with a
// tiny fixture; the actual page, runner, worker, WASM and snarkjs execute without mocks.
// The fixture config uses SvelteKit directly, not private operational/deployed Vite config.
// Asset hashes identify the supplied files; --wasm-revision is caller-declared provenance.
// node scripts/browser-parity.mjs --assets /absolute/pkg --fixtures /absolute/tiny_mul
//   --wasm-revision <Rust build revision> [--out /absolute/ignored/output]
const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = resolve(web, '..');
const { values } = parseArgs({
  options: {
    assets: { type: 'string' }, fixtures: { type: 'string' },
    'wasm-revision': { type: 'string' }, out: { type: 'string' },
    'test-server-config': { type: 'boolean', default: false },
    'test-lifecycle': { type: 'boolean', default: false },
    chrome: { type: 'string', default: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' }
  }
});
// Ignore checkout-local TLS settings: loopback HTTP is a secure context for WebGPU.
const ownedServer = { host: '127.0.0.1', port: 0, open: false, https: false };
function fixtureConfig(options = {}) {
  return mergeConfig(options, {
    root: web, configFile: false, server: ownedServer, plugins: [sveltekit()]
  });
}
if (values['test-lifecycle']) {
  await lifecycleTests();
  process.exit(0);
}
if (values['test-server-config']) {
  for (const https of [undefined, { cert: 'unused test certificate', key: 'unused test key' }]) {
    const server = await deadline(createServer(fixtureConfig({
      configFile: join(web, 'must-not-load-this-config.mjs'), server: { https },
      plugins: [{
        name: 'http-readiness',
        configureServer(server) {
          server.middlewares.use('/__parity-ready', (_req, res) => res.end('ready'));
        }
      }]
    })), 30000, 'fixture config');
    try {
      assert.ok(!server.config.configFile, 'no operational Vite config may be loaded');
      assert.equal(server.config.server.https, false, 'inherited TLS must be disabled');
      await deadline(server.listen(), 10000, 'HTTP listen');
      const response = await fetch(`http://127.0.0.1:${server.httpServer.address().port}/__parity-ready`, {
        signal: AbortSignal.timeout(10000), redirect: 'error'
      });
      assert.equal(response.status, 200);
      assert.equal(await response.text(), 'ready');
      console.log(`server config ${https ? 'with inherited TLS' : 'without TLS'}: HTTP readiness passed`);
    } finally {
      await deadline(server.close(), 5000, 'HTTP close');
    }
  }
  process.exit(0);
}
for (const key of ['assets', 'fixtures']) {
  assert.ok(values[key] && isAbsolute(values[key]), `--${key} must be an absolute path`);
}
assert.ok(values['wasm-revision'], '--wasm-revision identifies the separately built Rust source');
const out = values.out ?? join(repo, 'target', 'browser-parity', String(Date.now()));
assert.ok(isAbsolute(out), '--out must be an absolute path');
execFileSync('git', ['check-ignore', out], { cwd: repo });
await mkdir(out, { recursive: true });
// Refuse to attach to a previous run's profile or browser process.
await mkdir(join(out, 'chrome-profile'));
const sha256 = (data) => createHash('sha256').update(data).digest('hex');
const assets = new Map();
for (const name of ['snarkrs_web.js', 'snarkrs_web_bg.wasm']) {
  assets.set(`/pkg/${name}`, await readFile(join(values.assets, name)));
}
assert.ok(assets.get('/pkg/snarkrs_web.js').includes('export function prove(constant_work)'));
const version = sha256(assets.get('/pkg/snarkrs_web_bg.wasm'));
const fixtures = new Map();
for (const name of ['circuit.zkey', 'circuit.wtns', 'vkey.json', 'public.json']) {
  fixtures.set(`/__fixtures/tiny_mul/${name}`, await readFile(join(values.fixtures, name)));
}
const publicSignals = JSON.parse(fixtures.get('/__fixtures/tiny_mul/public.json'));
const vkey = JSON.parse(fixtures.get('/__fixtures/tiny_mul/vkey.json'));
const circuit = {
  name: 'tiny_mul', label: 'tiny_mul correctness fixture', blurb: 'Not a performance measurement.',
  constraints: 2,
  wires: JSON.parse(await readFile(join(values.fixtures, 'wtns.json'), 'utf8')).length,
  zkeyBytes: fixtures.get('/__fixtures/tiny_mul/circuit.zkey').length,
  wtnsBytes: fixtures.get('/__fixtures/tiny_mul/circuit.wtns').length
};
const evidence = {
  revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
  workingTree: execFileSync('git', ['status', '--short'], { cwd: repo, encoding: 'utf8' }).trim(),
  wasmRevision: values['wasm-revision'],
  limitations: {
    wasmRevision: 'Caller-declared, not independently attested.',
    assets: 'Supplied Rust release assets before wasm-opt, not production asset validation.',
    config: 'Owned HTTP fixture config with SvelteKit, not operational/deployed Vite config.',
    adapter: 'Separately requested window adapter; worker device identity is not attested.'
  },
  harnessSha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
  assets: Object.fromEntries([...assets].map(([name, data]) => [name, { bytes: data.length, sha256: sha256(data) }])),
  fixtures: Object.fromEntries([...fixtures].map(([name, data]) => [name, { bytes: data.length, sha256: sha256(data) }])),
  profile: 'floor', fixture: circuit, runs: [], requests: [], errors: [],
  phones: 'Not tested: no iPhone or Android device available in this local Chrome run.',
  passed: false
};
const reports = [], pendingReports = new Set();
let server, chrome, browser, page;
const chromeLog = createWriteStream(join(out, 'chrome.log'));
chromeLog.on('error', (e) => evidence.errors.push(`Chrome log: ${e.message}`));

async function deadline(work, ms, label) {
  let timer;
  try {
    return await Promise.race([
      work,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`timed out: ${label}`)), ms);
      })
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function connectSocket(url, timeout = 10000, Socket = WebSocket) {
  const ws = new Socket(url);
  let open, error, close;
  try {
    await deadline(new Promise((resolve, reject) => {
      open = resolve;
      error = () => reject(new Error('WebSocket connection failed'));
      close = () => reject(new Error('WebSocket closed before connecting'));
      ws.addEventListener('open', open, { once: true });
      ws.addEventListener('error', error, { once: true });
      ws.addEventListener('close', close, { once: true });
    }), timeout, 'WebSocket handshake');
    return ws;
  } catch (e) {
    ws.close();
    throw e;
  } finally {
    ws.removeEventListener('open', open);
    ws.removeEventListener('error', error);
    ws.removeEventListener('close', close);
  }
}

async function closeSocket(ws) {
  if (ws.readyState === 3) return;
  let close;
  try {
    await deadline(new Promise((resolve) => {
      close = resolve;
      ws.addEventListener('close', close, { once: true });
      ws.close();
    }), 2000, 'WebSocket close');
  } finally {
    ws.removeEventListener('close', close);
  }
}

async function fetchJson(url, options = {}, timeout = 10000) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeout), redirect: 'error' });
  assert.equal(response.status, 200, `HTTP status for ${url}`);
  return response.json();
}

function childExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}
async function waitForExit(child, timeout) {
  if (childExited(child)) return true;
  let timer, exit;
  try {
    return await new Promise((resolve) => {
      exit = () => resolve(true);
      child.once('exit', exit);
      timer = setTimeout(() => resolve(false), timeout);
    });
  } finally {
    clearTimeout(timer);
    child.removeListener('exit', exit);
  }
}

async function stopOwnedChild(child, { graceMs = 0, termMs = 5000, killMs = 2000 } = {}) {
  const result = { pid: child?.pid, termSent: false, killSent: false };
  try {
    if (!child?.pid) return { ...result, notSpawned: true };
    if (!childExited(child) && graceMs) await waitForExit(child, graceMs);
    if (!childExited(child)) {
      result.termSent = child.kill('SIGTERM');
      await waitForExit(child, termMs);
    }
    if (!childExited(child)) {
      result.killSent = child.kill('SIGKILL');
      if (!await waitForExit(child, killMs)) throw new Error(`owned child ${child.pid} did not exit after SIGKILL`);
    }
    return { ...result, exitCode: child.exitCode, signalCode: child.signalCode };
  } finally {
    for (const stream of child?.stdio ?? []) stream?.destroy();
    if (child?.connected) child.disconnect();
  }
}

async function cleanupStep(steps, name, action, timeout = 5000) {
  try {
    const result = await deadline(Promise.resolve().then(action), timeout, name);
    steps.push({ name, ok: true, result });
  } catch (e) {
    steps.push({ name, ok: false, error: e.message ?? String(e) });
  }
}

async function lifecycleTests() {
  class StalledSocket extends EventTarget {
    static last;
    constructor() { super(); StalledSocket.last = this; }
    close() { this.closed = true; }
  }
  await assert.rejects(connectSocket('ws://unused', 20, StalledSocket), /timed out: WebSocket handshake/);
  assert.equal(StalledSocket.last.closed, true);
  console.log('stalled WebSocket: deadline and close passed');
  await assert.rejects(waitFor(() => new Promise(() => {}), 'stalled readiness', 20), /timed out: stalled readiness/);
  console.log('stalled readiness callback: deadline passed');
  const steps = [];
  await cleanupStep(steps, 'stalled teardown', () => new Promise(() => {}), 20);
  assert.equal(steps[0].ok, false);
  assert.match(steps[0].error, /timed out/);
  console.log('stalled teardown: recorded as a failed cleanup step');
  const http = createHttpServer((_req, res) => { res.writeHead(200); res.write('{'); });
  try {
    http.listen(0, '127.0.0.1');
    await deadline(once(http, 'listening'), 1000, 'test HTTP listen');
    await assert.rejects(fetchJson(`http://127.0.0.1:${http.address().port}`, {}, 30), /abort|timeout/i);
    console.log('stalled HTTP body: abort deadline passed');
  } finally {
    http.closeAllConnections();
    await deadline(new Promise((resolve) => http.close(resolve)), 1000, 'test HTTP close');
  }
  for (const refuseTerm of [false, true]) {
    const child = spawn(process.execPath, ['-e',
      `${refuseTerm ? "process.on('SIGTERM', () => {});" : ''} process.send('ready'); setInterval(() => {}, 1000);`
    ], { stdio: ['ignore', 'pipe', 'pipe', 'ipc'] });
    try {
      await deadline(once(child, 'message'), 2000, 'owned dummy readiness');
      const stopped = await stopOwnedChild(child, { termMs: 100, killMs: 2000 });
      assert.equal(stopped.termSent, true);
      assert.equal(stopped.killSent, refuseTerm);
      assert.equal(stopped.signalCode, refuseTerm ? 'SIGKILL' : 'SIGTERM');
      assert.ok(child.stdout.destroyed && child.stderr.destroyed);
      console.log(`owned child ${refuseTerm ? 'refusing TERM' : 'accepting TERM'}: exit and pipe cleanup passed`);
    } finally {
      await stopOwnedChild(child, { termMs: 100, killMs: 2000 });
    }
  }
}

async function waitFor(fn, label, timeout = 120000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const value = await deadline(Promise.resolve().then(fn), Math.max(1, end - Date.now()), label);
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`timed out: ${label}`);
}

async function cdp(url) {
  const ws = await connectSocket(url);
  let next = 0;
  const pending = new Map();
  ws.addEventListener('message', ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.method === 'Runtime.exceptionThrown') evidence.errors.push(msg.params);
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    clearTimeout(p.timer);
    msg.error ? p.reject(new Error(JSON.stringify(msg.error))) : p.resolve(msg.result);
  });
  ws.addEventListener('close', () => {
    for (const p of pending.values()) {
      clearTimeout(p.timer);
      p.reject(new Error('CDP connection closed'));
    }
    pending.clear();
  });
  return {
    send(method, params = {}, timeout = 120000) {
      const id = ++next;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }, timeout);
        pending.set(id, { resolve, reject, timer });
        try {
          ws.send(JSON.stringify({ id, method, params }));
        } catch (e) {
          clearTimeout(timer);
          pending.delete(id);
          reject(e);
        }
      });
    },
    close() { return closeSocket(ws); }
  };
}

async function evaluate(expression, timeout = 120000) {
  const result = await page.send('Runtime.evaluate', {
    expression, awaitPromise: true, returnByValue: true
  }, timeout);
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}

try {
  server = await deadline(createServer(fixtureConfig({
    define: { 'import.meta.env.VITE_ARTIFACT_BASE': JSON.stringify('/__fixtures') },
    plugins: [{
      name: 'tiny-parity-fixture', enforce: 'pre',
      resolveId(id) { if (id === '../pkg-version') return '\0parity-version'; },
      load(id) { if (id === '\0parity-version') return `export const PKG_VERSION = ${JSON.stringify(version)};`; },
      transform(code, id) {
        if (id.split('?')[0] === join(web, 'src/lib/circuits.ts')) {
          return `${code}\nCIRCUITS.splice(0, CIRCUITS.length, ${JSON.stringify(circuit)});\n`;
        }
      },
      configureServer(vite) {
        vite.middlewares.use((req, res, next) => {
          const url = new URL(req.url, 'http://localhost');
          if (url.pathname === '/__report' && req.method === 'POST') {
            const chunks = [];
            req.on('data', (chunk) => chunks.push(chunk));
            req.on('end', () => {
              try {
                const report = JSON.parse(Buffer.concat(chunks));
                pendingReports.add(res);
                reports.push({ report, release() { res.writeHead(204).end(); pendingReports.delete(res); } });
              } catch (e) {
                evidence.errors.push(String(e));
                res.writeHead(400).end();
              }
            });
            return;
          }
          const body = assets.get(url.pathname) ?? fixtures.get(url.pathname);
          if (!body) return next();
          evidence.requests.push(req.url);
          res.writeHead(200, {
            'Content-Type': url.pathname.endsWith('.js') ? 'text/javascript' :
              url.pathname.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream',
            'Content-Length': body.length, 'Cache-Control': 'no-store'
          });
          res.end(body);
        });
      }
    }]
  })), 30000, 'fixture server creation');
  assert.ok(!server.config.configFile, 'operational Vite config must not be loaded');
  assert.equal(server.config.server.https, false, 'the smoke server must override inherited HTTPS');
  await deadline(server.listen(), 10000, 'fixture server listen');
  const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
  evidence.origin = origin;
  const readinessUrl = `${origin}/pkg/snarkrs_web.js?v=${version}`;
  const readiness = await fetch(readinessUrl, { signal: AbortSignal.timeout(10000), redirect: 'error' });
  assert.equal(readiness.status, 200, 'the owned server must answer over HTTP before Chrome starts');
  const servedHash = sha256(Buffer.from(await readiness.arrayBuffer()));
  assert.equal(servedHash, sha256(assets.get('/pkg/snarkrs_web.js')), 'HTTP must serve the fresh glue');
  evidence.server = { https: server.config.server.https, readinessUrl, status: readiness.status, servedHash };
  chrome = spawn(values.chrome, [
    '--remote-debugging-port=0', `--user-data-dir=${join(out, 'chrome-profile')}`,
    '--no-first-run', '--no-default-browser-check', 'about:blank'
  ], { stdio: ['ignore', 'pipe', 'pipe'] });
  chrome.stdout.pipe(chromeLog, { end: false });
  chrome.stderr.pipe(chromeLog, { end: false });
  let launchError;
  chrome.on('error', (e) => { launchError = e; });
  const port = await waitFor(async () => {
    if (launchError) throw launchError;
    if (chrome.exitCode !== null) throw new Error(`Chrome exited: ${chrome.exitCode}`);
    try { return Number((await readFile(join(out, 'chrome-profile/DevToolsActivePort'), 'utf8')).split('\n')[0]); }
    catch { return false; }
  }, 'Chrome debugger readiness', 30000);
  const endpoint = `http://127.0.0.1:${port}`;
  const info = await fetchJson(`${endpoint}/json/version`);
  evidence.browser = info;
  browser = await cdp(info.webSocketDebuggerUrl);
  evidence.gpu = await browser.send('SystemInfo.getInfo');
  const tab = await fetchJson(`${endpoint}/json/new?about:blank`, { method: 'PUT' });
  page = await cdp(tab.webSocketDebuggerUrl);
  await page.send('Page.enable');
  await page.send('Runtime.enable');
  // Observe real worker replies without replacing any handler, request or proof.
  await page.send('Page.addScriptToEvaluateOnNewDocument', {
    source: `
    window.__parityProofs = [];
    const watched = new WeakSet();
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function(...args) {
      if (args[0]?.type === 'init' && typeof args[0].pkgBase === 'string' && !watched.has(this)) {
        watched.add(this);
        this.addEventListener('message', ({ data }) => {
          if (data?.ok && data.value?.proof) window.__parityProofs.push(data.value);
        });
      }
      return post.apply(this, args);
    };
  ` });
  await page.send('Page.navigate', { url: `${origin}/?circuits=tiny_mul&profile=floor&warmup=1&reps=1&selftest=1&report=1&diag=1` });
  await page.send('Page.bringToFront');
  await waitFor(async () => {
    const state = await evaluate(`({ text: document.body?.innerText ?? '', ready: document.querySelector('.go .word')?.textContent === 'GO' })`);
    if (state.text.includes('unsupported')) throw new Error(`WebGPU unavailable: ${state.text}`);
    return state.ready;
  }, 'supported, hydrated page');
  assert.equal(await evaluate(`document.querySelector('.work-mode input').checked`), false);
  evidence.adapter = await evaluate(`(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no WebGPU adapter');
    const i = adapter.info;
    return { userAgent: navigator.userAgent, vendor: i.vendor, architecture: i.architecture,
      device: i.device, description: i.description, isFallbackAdapter: i.isFallbackAdapter,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
  })()`);
  assert.equal(evidence.adapter.isFallbackAdapter, false, 'require a confirmed native GPU adapter');
  for (const constantWork of [false, true]) {
    const index = evidence.runs.length;
    const start = await evaluate(`(async () => {
      const input = document.querySelector('.work-mode input');
      if (input.checked !== ${constantWork}) input.click();
      document.querySelector('.go').click();
      await new Promise(requestAnimationFrame);
      const run = window.__g16run;
      input.click();
      run.constantWork = ${!constantWork};
      return { busy: run.busy, disabled: input.disabled, selected: run.constantWork };
    })()`);
    assert.deepEqual(start, { busy: true, disabled: true, selected: constantWork });
    await waitFor(() => reports[index], `mode ${constantWork}: proof and report`);
    const locked = await evaluate(`({ phase: __g16run.phase, busy: __g16run.busy,
      disabled: document.querySelector('.work-mode input').disabled, selected: __g16run.constantWork })`);
    assert.deepEqual(locked, { phase: 'done', busy: true, disabled: true, selected: constantWork });
    const { report } = reports[index];
    assert.equal(report.fatal, null);
    assert.equal(report.constantWork, constantWork);
    assert.equal(report.rows.length, 1);
    assert.equal(report.rows[0].circuit, 'tiny_mul');
    assert.equal(report.rows[0].status, 'done');
    assert.equal(report.rows[0].constantWork, constantWork);
    assert.equal(report.rows[0].crossVerified, true);
    assert.equal(report.env.selftest.checks.length, 7);
    assert.ok(report.env.selftest.checks.every((check) => check.ok), JSON.stringify(report.env.selftest));
    reports[index].release();
    await waitFor(() => evaluate('!__g16run.busy'), 'report completion unlock');
    const checked = await evaluate(`(async () => {
      const proofs = __parityProofs.splice(0);
      const results = [];
      for (const p of proofs) results.push({ constantWork: p.constantWork,
        publicSignals: p.publicSignals, proof: p.proof,
        verified: await snarkjs.groth16.verify(${JSON.stringify(vkey)}, p.publicSignals, p.proof) });
      return results;
    })()`);
    assert.equal(checked.length, 2, 'one warm-up and one checked warm proof');
    for (const proof of checked) {
      assert.equal(proof.constantWork, constantWork);
      assert.deepEqual(proof.publicSignals, publicSignals);
      assert.equal(proof.verified, true, 'independent snarkjs verification');
    }
    const labels = await evaluate(`(() => {
      document.querySelector('.work-mode input').click();
      return { selected: __g16run.constantWork, result: __g16run.resultConstantWork,
        row: document.querySelector('.gpunum .fine')?.textContent,
        summary: document.querySelector('.side.gpu .fine')?.textContent,
        diagnostics: __g16run.diagnostics };
    })()`);
    assert.equal(labels.selected, !constantWork);
    assert.equal(labels.result, constantWork);
    assert.equal(labels.row, constantWork ? 'constant-work' : 'variable-work');
    assert.equal(labels.summary, labels.row);
    assert.ok(labels.diagnostics.includes(`work       ${labels.row}`));
    evidence.runs.push({ constantWork, startupGuard: 'create_prover completed', start, locked, report, checked, labels });
    const screenshot = await page.send('Page.captureScreenshot', { format: 'png' });
    await writeFile(join(out, `${constantWork ? 'constant' : 'variable'}.png`), Buffer.from(screenshot.data, 'base64'));
    console.log(`${labels.row}: startup guard, 7 selftests, 2 snarkjs-verified GPU proofs, reference signals, UI/report lock and labels passed`);
  }
  for (const name of assets.keys()) {
    assert.ok(evidence.requests.some((url) => url === `${name}?v=${version}`), `fresh asset not requested: ${name}`);
  }
  assert.equal(evidence.errors.length, 0, JSON.stringify(evidence.errors));
  evidence.checksPassed = true;
} catch (e) {
  evidence.failure = e.stack ?? String(e);
  console.error(evidence.failure);
} finally {
  const started = performance.now();
  const steps = [];
  evidence.cleanup = { state: 'in-progress', steps };
  evidence.reports = reports.map(({ report }) => report);
  // A killed or stuck teardown must leave a record that has not declared the gate passed.
  await cleanupStep(steps, 'preliminary evidence', () =>
    writeFile(join(out, 'evidence.json'), JSON.stringify(evidence, null, 2)));
  if (page && !evidence.checksPassed) {
    try { evidence.pageAtFailure = await evaluate(`document.body?.innerText ?? ''`, 2000); }
    catch (e) { evidence.pageDiagnosticError = e.message; }
  }
  await cleanupStep(steps, 'release reports', () => {
    for (const res of pendingReports) res.writeHead(503).end();
  });
  if (browser) {
    await cleanupStep(steps, 'Browser.close', async () => {
      try { await browser.send('Browser.close', {}, 2000); }
      catch (e) {
        if (e.message !== 'CDP connection closed') throw e;
        return { closedBeforeReply: true };
      }
    }, 3000);
  }
  await cleanupStep(steps, 'owned Chrome exit and pipes', () =>
    stopOwnedChild(chrome, { graceMs: 2000 }), 10000);
  await cleanupStep(steps, 'page socket close', () => page?.close(), 3000);
  await cleanupStep(steps, 'browser socket close', () => browser?.close(), 3000);
  await cleanupStep(steps, 'fixture server close', () => server?.close());
  server?.httpServer?.closeAllConnections();
  await cleanupStep(steps, 'Chrome log close', async () => {
    chromeLog.end();
    await finished(chromeLog);
  }, 2000);
  chromeLog.destroy();
  evidence.cleanup.state = steps.every((step) => step.ok) ? 'complete' : 'failed';
  evidence.cleanup.elapsedMs = performance.now() - started;
  evidence.passed = evidence.checksPassed === true && evidence.cleanup.state === 'complete' && evidence.errors.length === 0;
  try {
    await deadline(writeFile(join(out, 'evidence.json'), JSON.stringify(evidence, null, 2)), 5000, 'final evidence');
  } catch (e) {
    evidence.passed = false;
    console.error(e);
  }
  console.log(`Evidence: ${join(out, 'evidence.json')}`);
  // Bound process lifetime even when a failed close left a socket or watcher referenced.
  process.exit(evidence.passed ? 0 : 1);
}
