<script lang="ts">
  import { onMount } from 'svelte';
  import { Run } from '$lib/runner.svelte';
  import { bytes, count, duration, ms, ratioLabel } from '$lib/format';
  import { checkSupport, type Support } from '$lib/support';

  const run = new Run();

  // null while the check is in flight. See $lib/support for what it actually tests and why
  // checking `navigator.gpu` alone is not enough.
  let support = $state<Support | null>(null);
  const supported = $derived(support === null ? null : support.ok);

  onMount(async () => {
    support = await checkSupport();
    // Size the ladder to this adapter before the button is enabled, so the download total on
    // it is what this machine will actually fetch. The device may still grant less than the
    // adapter claims, and `start()` re-plans when it finds out.
    if (support.ok) run.plan(support.maxStorageBinding);
    // `?auto=1` starts without a click. This exists so the page can be run on a browser
    // that cannot be driven from here: Safari refuses WebDriver until "Allow remote
    // automation" is switched on by hand, and a Safari-only failure is exactly the kind that
    // has to be reproduced rather than reasoned about.
    if (support.ok && new URLSearchParams(location.search).get('auto') === '1') run.start();
  });

  const busy = $derived(run.phase === 'running' || run.phase === 'starting');
  // Capped just short of full until the run says it is done. The ring is priced from
  // estimates, so a run that outlasts them would otherwise show a full ring with circuits
  // still to prove, which is the one thing a progress indicator must never say.
  const progress = $derived(run.phase === 'done' ? 1 : Math.min(0.99, run.progress));
  const headline = $derived(run.headline);
  const verdict = $derived(headline == null ? null : ratioLabel(headline));
  const workers = $derived(run.rows.find((r) => r.snarkjsWorkers)?.snarkjsWorkers ?? null);

  // A 270-degree gauge, open at the bottom, which is the speedtest dial shape. The arc is
  // drawn as a dashed circle rather than as a path, so the sweep is one number to animate.
  const R = 124;
  const SWEEP = 0.75; // three quarters of the circle
  const CIRC = 2 * Math.PI * R;
  const TICKS = Array.from({ length: 41 }, (_, i) => 135 + (i * 270) / 40);
</script>

<main>
  <header>
    <h1>Groth16 Speed Test</h1>
    <p class="sub">
      How fast does your GPU prove a zkSNARK? This runs real deployed circuits through a
      WebGPU prover and through
      <a href="https://github.com/iden3/snarkjs" target="_blank" rel="noreferrer">snarkjs</a>,
      in this browser, on the same bytes, and makes each one verify the other's proof.
    </p>
  </header>

  <section class="dial">
    <div class="gauge">
      <svg viewBox="0 0 320 320" aria-hidden="true">
        <!-- tick marks, the dial's face -->
        <g class="ticks">
          {#each TICKS as deg, i}
            <line
              x1="160"
              y1="9"
              x2="160"
              y2={i % 5 === 0 ? 26 : 21}
              transform="rotate({deg - 90} 160 160)"
              class:major={i % 5 === 0}
            />
          {/each}
        </g>
        <!-- the sweep -->
        <circle
          class="track"
          cx="160"
          cy="160"
          r={R}
          stroke-dasharray="{CIRC * SWEEP} {CIRC}"
          transform="rotate(135 160 160)"
        />
        <circle
          class="fill"
          class:done={run.phase === 'done'}
          class:empty={progress <= 0}
          cx="160"
          cy="160"
          r={R}
          stroke-dasharray="{CIRC * SWEEP * progress} {CIRC}"
          transform="rotate(135 160 160)"
        />
      </svg>

      <button
        class="go"
        class:busy
        disabled={supported === false || busy}
        onclick={() => run.start()}
      >
        {#if supported === null}
          <span class="word">···</span>
        {:else if supported === false}
          <span class="word small">unsupported</span>
        {:else if run.phase === 'idle'}
          <span class="word">GO</span>
        {:else if run.phase === 'done' && verdict}
          <span class="num">{verdict.x.split('x')[0]}<em>x</em></span>
          <span class="cap" class:bad={!verdict.faster}>{verdict.faster ? 'faster' : 'slower'}</span>
        {:else if run.phase === 'done'}
          <span class="word small">no result</span>
        {:else if run.etaMs != null}
          <span class="num">{duration(run.etaMs)}</span>
          <span class="cap">left</span>
        {:else}
          <span class="word small">starting</span>
        {/if}
      </button>
    </div>

    <div class="activity">
      {#if run.fatal}
        <p class="bad">{run.fatal}</p>
      {:else if support && !support.ok}
        <p class="bad">{support.reason}</p>
        <p class="fine">{support.detail}</p>
      {:else if run.activity}
        <p class="what">{run.activity}</p>
        <p class="fine">
          {#if run.mbits > 0}{run.mbits.toFixed(0)} Mbit/s ·{/if}
          {run.calibrated
            ? 'estimate calibrated on this machine'
            : 'estimate from a reference machine until the first measurement lands'}
        </p>
        <button class="link" onclick={() => run.stop()}>stop after this circuit</button>
      {:else if run.phase === 'idle'}
        <p class="fine">
          {run.willRun} real circuits, smallest first · {bytes(run.totalDownload)} to
          download · keep this tab in front, a hidden tab gets a throttled clock and its
          timings are thrown away
        </p>
      {:else if run.phase === 'done'}
        <p class="fine">
          {run.totals.n} circuits, every proof checked by the other prover's verifier.
          <button class="link" onclick={() => run.start()}>run it again</button>
        </p>
      {/if}
    </div>
  </section>

  {#if run.phase === 'done' && verdict}
    {@const t = run.totals}
    <section class="verdict">
      <div class="side sj">
        <h2>snarkjs</h2>
        <p class="big">{ms(t.snarkjs)}</p>
        <p class="fine">wasm on {workers ?? '?'} workers</p>
      </div>
      <div class="mid">
        <span class="x" class:bad={!verdict.faster}>{verdict.x}</span>
        <span class="fine">median across circuits</span>
      </div>
      <div class="side gpu">
        <h2>WebGPU</h2>
        <p class="big">{ms(t.webgpu)}</p>
        <p class="fine">{[run.env?.adapter?.vendor, run.env?.adapter?.architecture].filter(Boolean).join(' ') || 'this GPU'}</p>
      </div>
    </section>
  {/if}

  {#if run.rows.length}
    <section class="results">
      <table>
        <thead>
          <tr>
            <th>circuit</th>
            <th class="n">constraints</th>
            <th class="n">snarkjs</th>
            <th class="n">WebGPU</th>
            <th class="n">GPU upload</th>
            <th class="n">result</th>
          </tr>
        </thead>
        <tbody>
          {#each run.rows as row (row.circuit.name)}
            <tr class={row.status}>
              <td>
                <strong>{row.circuit.label}</strong>
                <span class="fine">{row.circuit.blurb}</span>
                {#if row.error}<span class="fine bad">{row.error}</span>{/if}
                {#if row.note}<span class="fine warn">{row.note}</span>{/if}
              </td>
              <td class="n">{count(row.circuit.constraints)}</td>
              <td class="n sjnum">{ms(row.snarkjsMs)}</td>
              <td class="n gpunum">{ms(row.webgpuMs)}</td>
              <td class="n dim">{ms(row.prepareMs)}</td>
              <td class="n">
                {#if row.status === 'done' && row.snarkjsMs && row.webgpuMs}
                  {@const l = ratioLabel(row.snarkjsMs / row.webgpuMs)}
                  <span class="pill" class:bad={!l.faster}>{l.x}</span>
                  {#if row.crossVerified}<span class="fine ok">both verified</span>{/if}
                {:else if row.status === 'downloading'}
                  <span class="fine">{bytes(row.downloaded)} of {bytes(row.circuit.zkeyBytes + row.circuit.wtnsBytes)}</span>
                {:else if row.status === 'webgpu'}
                  <span class="fine">on the GPU</span>
                {:else if row.status === 'snarkjs'}
                  <span class="fine">on snarkjs</span>
                {:else if row.status === 'skipped'}
                  <span class="fine">not run</span>
                {:else if row.status === 'error'}
                  <span class="fine bad">failed</span>
                {:else}
                  <span class="fine dim">queued · {bytes(row.circuit.zkeyBytes)}</span>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}

  <footer>
    <p>
      Both provers get the identical proving key and witness, already in memory, so no fetch
      happens inside a timed call. Every number is the median of the timed proofs after a
      warm-up, never the mean. snarkjs has no <code>prepare()</code> — it re-reads the key on
      every call — so its number is both its cold and its warm one. Ours is warm, and the
      upload it warms up with is the separate <em>GPU upload</em> column rather than being
      hidden inside the proof.
    </p>
    {#if run.aboveFloor.length}
      <p class="fine">
        {run.aboveFloor.map((r) => r.circuit.label).join(', ')}
        {run.aboveFloor.length === 1 ? 'needs' : 'need'} a storage binding larger than the 128
        MiB WebGPU guarantees, and this adapter grants
        {(run.bindingLimit / 1024 ** 3).toFixed(2)} GiB. A device offering only the floor would
        skip {run.aboveFloor.length === 1 ? 'that row' : 'those rows'}, so its ladder is shorter
        than this one.
      </p>
    {/if}
    {#if run.env}
      <p class="fine">
        {[run.env.adapter?.vendor, run.env.adapter?.architecture, run.env.adapter?.device]
          .filter(Boolean)
          .join(' ') || 'adapter did not identify itself'} · {run.env.hardwareConcurrency} cores ·
        wasm module {ms(run.env.moduleMs)}, adapter {ms(run.env.deviceMs)}
      </p>
    {/if}
  </footer>
</main>

<style>
  :global(:root) {
    color-scheme: dark;
    --bg: #0a0e17;
    --panel: #111725;
    --line: #1d2637;
    --fg: #eaeef6;
    --dim: #8593ab;
    --gpu: #34d399;
    --sj: #fbbf24;
    --bad: #f87171;
  }
  :global(body) {
    margin: 0;
    background: radial-gradient(1200px 760px at 50% -14%, #16233a 0%, var(--bg) 60%);
    color: var(--fg);
    font: 15px/1.55 ui-sans-serif, -apple-system, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
  }
  main {
    max-width: 920px;
    margin: 0 auto;
    padding: 3rem 1.25rem 5rem;
  }
  header {
    text-align: center;
  }
  h1 {
    font-size: 1.55rem;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
    font-weight: 650;
  }
  .sub {
    color: var(--dim);
    max-width: 36rem;
    margin: 0 auto;
  }
  a {
    color: var(--gpu);
  }

  .dial {
    display: grid;
    justify-items: center;
    gap: 1rem;
    margin: 2rem 0 1.5rem;
  }
  .gauge {
    position: relative;
    width: 320px;
    height: 320px;
  }
  .gauge svg {
    width: 100%;
    height: 100%;
    display: block;
  }
  /* The dial's face, and only that. These used to light up as the run advanced, which made
     them a second progress readout of the same number the arc already draws. */
  .ticks line {
    stroke: #202a3c;
    stroke-width: 2;
  }
  .ticks line.major {
    stroke-width: 3;
  }
  .track {
    fill: none;
    stroke: #18202f;
    stroke-width: 10;
    stroke-linecap: round;
  }
  .fill.empty {
    /* A round cap draws a dot even at dash length zero, which reads as a dial already an
       instant into its sweep before anything has started. */
    visibility: hidden;
  }
  .fill {
    fill: none;
    stroke: var(--gpu);
    stroke-width: 10;
    stroke-linecap: round;
    /* Tweened, because the ETA is recomputed five times a second and an untweened arc
       ratchets. Just under the tick, so each step has landed before the next arrives and
       the sweep reads as continuous rather than as five jumps a second. */
    transition: stroke-dasharray 0.18s linear;
  }

  /* The button is the inner disc, so the dial around it stays visible and unclickable. */
  .go {
    position: absolute;
    inset: 58px;
    border-radius: 50%;
    border: 1px solid #24304a;
    background: linear-gradient(180deg, #131b2b, #0d1421);
    color: inherit;
    cursor: pointer;
    display: grid;
    align-content: center;
    justify-items: center;
    gap: 0.15rem;
    font: inherit;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.1s;
  }
  .go:not(:disabled):hover {
    border-color: var(--gpu);
    box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.25), 0 0 44px rgba(52, 211, 153, 0.16);
  }
  .go:not(:disabled):active {
    transform: scale(0.985);
  }
  .go:disabled {
    cursor: default;
  }
  .go.busy {
    /* Compositor-driven, so it keeps breathing even while snarkjs holds the main thread.
       Anything driven from JS would stall exactly when the page most needs to look alive. */
    animation: breathe 2.6s ease-in-out infinite;
  }
  @keyframes breathe {
    0%,
    100% {
      box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.1), 0 0 30px rgba(52, 211, 153, 0.06);
    }
    50% {
      box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.3), 0 0 54px rgba(52, 211, 153, 0.2);
    }
  }
  .word {
    font-size: 3.6rem;
    font-weight: 300;
    letter-spacing: 0.16em;
    text-indent: 0.16em;
  }
  .word.small {
    font-size: 1.25rem;
    letter-spacing: 0.02em;
    text-indent: 0;
    color: var(--dim);
  }
  .num {
    font-size: 3.5rem;
    font-weight: 300;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .num em {
    font-size: 1.4rem;
    font-style: normal;
    color: var(--dim);
    margin-left: 0.08em;
  }
  .cap {
    color: var(--dim);
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .cap.bad {
    color: var(--bad);
  }

  .activity {
    text-align: center;
    min-height: 4.5rem;
    max-width: 34rem;
  }
  .activity p {
    margin: 0.25rem 0;
  }
  .what {
    font-variant-numeric: tabular-nums;
  }
  .link {
    background: none;
    border: 0;
    color: var(--dim);
    text-decoration: underline;
    cursor: pointer;
    font: inherit;
    font-size: 0.82rem;
    padding: 0.3rem;
  }
  .link:hover {
    color: var(--fg);
  }

  .verdict {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 1rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
  }
  .side {
    text-align: center;
  }
  .side h2 {
    margin: 0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--dim);
  }
  .side.sj h2 {
    color: var(--sj);
  }
  .side.gpu h2 {
    color: var(--gpu);
  }
  .big {
    font-size: 1.9rem;
    font-weight: 600;
    margin: 0.3rem 0 0.15rem;
    font-variant-numeric: tabular-nums;
  }
  .mid {
    display: grid;
    justify-items: center;
    padding: 0 0.75rem;
  }
  .x {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--gpu);
    white-space: nowrap;
  }
  .x.bad {
    color: var(--bad);
  }

  .results {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
  }
  th,
  td {
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
  }
  th {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--dim);
    font-weight: 600;
  }
  tbody tr:last-child td {
    border-bottom: 0;
  }
  .n {
    text-align: right;
    white-space: nowrap;
  }
  .sjnum {
    color: var(--sj);
  }
  .gpunum {
    color: var(--gpu);
  }
  td .fine {
    display: block;
  }
  tr.pending td,
  tr.skipped td {
    opacity: 0.45;
  }
  tr.downloading,
  tr.webgpu,
  tr.snarkjs {
    background: #141c2c;
  }
  .pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 99px;
    background: rgba(52, 211, 153, 0.14);
    color: var(--gpu);
    font-weight: 600;
    font-size: 0.84rem;
  }
  .pill.bad {
    background: rgba(248, 113, 113, 0.14);
    color: var(--bad);
  }

  .fine {
    font-size: 0.77rem;
    color: var(--dim);
    font-weight: 400;
  }
  .dim {
    color: var(--dim);
  }
  .ok {
    color: var(--gpu);
  }
  .warn {
    color: var(--sj);
  }
  .bad {
    color: var(--bad);
  }
  footer {
    margin-top: 1.75rem;
    color: var(--dim);
    font-size: 0.84rem;
  }
  code {
    background: var(--panel);
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }

  @media (max-width: 640px) {
    .gauge {
      width: 270px;
      height: 270px;
    }
    .go {
      inset: 50px;
    }
    .word {
      font-size: 3rem;
    }
    .verdict {
      grid-template-columns: 1fr;
    }
    th:nth-child(2),
    td:nth-child(2),
    th:nth-child(5),
    td:nth-child(5) {
      display: none;
    }
  }
</style>
