import assert from 'node:assert/strict';
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), 'deploy-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const root = join(dir, 'repo');
  const bin = join(dir, 'bin');
  const events = join(dir, 'events');
  const args = join(dir, 'args');
  mkdirSync(join(root, 'web/scripts'), { recursive: true });
  mkdirSync(bin);
  const env = { ...process.env, PATH: `${bin}:${process.env.PATH}`, EVENTS: events, ARGS: args };
  for (const key of Object.keys(env)) {
    if (key.startsWith('GIT_')) delete env[key];
  }
  env.GIT_CONFIG_NOSYSTEM = '1';
  env.GIT_CONFIG_GLOBAL = '/dev/null';
  env.GIT_CEILING_DIRECTORIES = dir;
  function git(...argv) {
    const result = spawnSync('git', argv, { cwd: root, env, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
  }
  function write(path, content) { writeFileSync(join(root, path), content); }
  copyFileSync(new URL('./deploy.sh', import.meta.url), join(root, 'web/scripts/deploy.sh'));
  write('.gitignore', 'web/static/pkg/\n');
  write('tracked.txt', 'original\n');
  write('web/scripts/build-wasm.sh', `#!/usr/bin/env bash
set -eu
printf 'build\\n' >> "$EVENTS"
mkdir -p static/pkg
printf 'wasm' > static/pkg/snarkrs_web_bg.wasm
if [ "\${DIRTY_DURING:-}" = build ]; then
  printf 'changed\\n' >> ../tracked.txt
fi
if [ "\${FAIL_GIT_AFTER_BUILD:-}" = 1 ]; then
  printf '#!/usr/bin/env bash\\nexit 42\\n' > "$STUB_BIN/git"
  chmod +x "$STUB_BIN/git"
fi
`);
  chmodSync(join(root, 'web/scripts/build-wasm.sh'), 0o755);
  writeFileSync(join(bin, 'vercel'), `#!/usr/bin/env bash
set -eu
printf '%s\\n' "$1" >> "$EVENTS"
if [ "$1" = whoami ] && [ "\${DIRTY_DURING:-}" = auth ]; then
  printf 'changed\\n' >> ../tracked.txt
fi
if [ "$1" = deploy ]; then
  printf '%s\\0' "$@" > "$ARGS"
fi
`, { mode: 0o755 });
  env.STUB_BIN = bin;
  git('init', '--quiet');
  git('config', 'user.name', 'Deploy test');
  git('config', 'user.email', 'deploy-test@example.invalid');
  git('add', '.');
  git('-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture');
  return {
    root, bin, env, git, write,
    run: (...argv) => spawnSync('bash', [join(root, 'web/scripts/deploy.sh'), ...argv], {
      cwd: dir, env, encoding: 'utf8', timeout: 10000,
    }),
    events: () => existsSync(events) ? readFileSync(events, 'utf8').trim().split('\n') : [],
    args: () => readFileSync(args, 'utf8').split('\0').slice(0, -1),
  };
}

for (const [name, change] of [
  ['unstaged root modification', (f) => f.write('tracked.txt', 'changed\n')],
  ['staged root modification', (f) => { f.write('tracked.txt', 'changed\n'); f.git('add', 'tracked.txt'); }],
  ['tracked deletion', (f) => rmSync(join(f.root, 'tracked.txt'))],
  ['staged deletion', (f) => f.git('rm', 'tracked.txt')],
  ['nonignored root untracked file', (f) => f.write('new.txt', 'new\n')],
  ['nonignored web untracked file', (f) => f.write('web/new.txt', 'new\n')],
]) {
  test(name + ' fails before auth or build', (t) => {
    const f = fixture(t);
    change(f);
    const result = f.run();
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /repository is dirty/);
    assert.deepEqual(f.events(), []);
  });
}

for (const ignored of [false, true]) {
  test(ignored ? 'ignored wasm output is allowed' : 'clean repository deploys with arguments intact', (t) => {
    const f = fixture(t);
    if (ignored) {
      mkdirSync(join(f.root, 'web/static/pkg'), { recursive: true });
      f.write('web/static/pkg/old.wasm', 'old');
    }
    const result = f.run('--force', '--meta', 'label=two words', '');
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(f.events(), ['whoami', 'build', 'deploy']);
    assert.deepEqual(f.args(), ['deploy', '--prod', '--force', '--meta', 'label=two words', '']);
  });
}

for (const phase of ['build', 'auth']) {
  test('tracked change during ' + phase + ' blocks deploy', (t) => {
    const f = fixture(t);
    f.env.DIRTY_DURING = phase;
    const result = f.run();
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /repository is dirty/);
    assert.deepEqual(f.events(), ['whoami', 'build']);
  });
}

for (const afterBuild of [false, true]) {
  test('git status failure ' + (afterBuild ? 'after build' : 'before auth') + ' fails closed', (t) => {
    const f = fixture(t);
    if (afterBuild) f.env.FAIL_GIT_AFTER_BUILD = '1';
    else writeFileSync(join(f.bin, 'git'), '#!/usr/bin/env bash\nexit 42\n', { mode: 0o755 });
    const result = f.run();
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /cannot check repository status/);
    assert.deepEqual(f.events(), afterBuild ? ['whoami', 'build'] : []);
  });
}

test('outside a repository fails before auth or build', (t) => {
  const f = fixture(t);
  rmSync(join(f.root, '.git'), { recursive: true, force: true });
  const result = f.run();
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /cannot check repository status/);
  assert.deepEqual(f.events(), []);
});
