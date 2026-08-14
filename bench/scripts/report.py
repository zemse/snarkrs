#!/usr/bin/env python3
"""Render the sweep into a standalone HTML report.

Numbers come from report_data.py, which shares its cost model with cost.py, so the page
and COST.md cannot drift apart. Nothing here recomputes a timing.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_data import load  # noqa: E402

HEADLINE = "js_16x16_d32"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def build(d, headline=HEADLINE):
    machines, cells = d["machines"], d["cells"]
    by = {(c["machine"], c["backend"], c["mode"], c["variant"]): c for c in cells}
    variants = sorted({(c["constraints"], c["variant"]) for c in cells})

    def rows(mode, variant):
        out = []
        for (m, b, md, v), c in by.items():
            if md == mode and v == variant:
                out.append(c)
        return out

    warm = rows("warm", headline)
    cheapest = min(warm, key=lambda c: c["usd_per_k"]) if warm else None
    fastest = min(warm, key=lambda c: c["ms"]) if warm else None
    nc = next((n for n, v in variants if v == headline), 0)

    # Pareto frontier: a machine is on it when nothing is both cheaper and faster.
    def pareto(rs):
        keep = []
        for a in rs:
            if not any(b is not a and b["usd_per_k"] <= a["usd_per_k"] and b["ms"] <= a["ms"]
                       and (b["usd_per_k"] < a["usd_per_k"] or b["ms"] < a["ms"]) for b in rs):
                keep.append(a)
        return sorted(keep, key=lambda c: c["ms"])

    front = pareto(warm)
    frontset = {(c["machine"], c["backend"]) for c in front}

    chart = [dict(m=c["machine"], b=c["backend"], ms=c["ms"], cost=c["usd_per_k"],
                  gpu=machines[c["machine"]].get("gpu", ""),
                  cpu=machines[c["machine"]].get("cpu_model", ""),
                  price=machines[c["machine"]]["usd_per_hour"],
                  front=(c["machine"], c["backend"]) in frontset) for c in warm]

    def money(x):
        return f"${x:,.3f}"

    # ---- ranked table ------------------------------------------------------------------
    rank_rows = []
    for i, c in enumerate(sorted(warm, key=lambda x: x["usd_per_k"]), 1):
        meta = machines[c["machine"]]
        acc = meta.get("gpu") or meta.get("cpu_model", "")
        rank_rows.append(
            f'<tr{" class=\'front\'" if (c["machine"], c["backend"]) in frontset else ""}>'
            f'<td class="num">{i}</td>'
            f'<td><span class="mono mach">{esc(c["machine"])}</span>'
            f'<span class="acc">{esc(acc)}</span></td>'
            f'<td><span class="chip chip-{c["backend"]}">{esc(c["backend"])}</span></td>'
            f'<td class="num">{c["ms"]:,.1f}</td>'
            f'<td class="num strong">{money(c["usd_per_k"])}</td>'
            f'<td class="num dim">{money(c["usd_per_k_spot"])}</td>'
            f'<td class="num dim">{1/c["usd_per_k"]*1000:,.0f}</td></tr>')

    # ---- per-size winners --------------------------------------------------------------
    size_rows = []
    for n, v in variants:
        rs = rows("warm", v)
        if not rs:
            continue
        ch = min(rs, key=lambda c: c["usd_per_k"])
        fa = min(rs, key=lambda c: c["ms"])
        size_rows.append(
            f'<tr><td class="mono">{esc(v)}</td><td class="num">{n:,}</td>'
            f'<td><span class="mono">{esc(ch["machine"])}</span> '
            f'<span class="chip chip-{ch["backend"]}">{esc(ch["backend"])}</span></td>'
            f'<td class="num strong cost">{money(ch["usd_per_k"])}</td>'
            f'<td><span class="mono">{esc(fa["machine"])}</span> '
            f'<span class="chip chip-{fa["backend"]}">{esc(fa["backend"])}</span></td>'
            f'<td class="num strong speed">{fa["ms"]:,.1f} ms</td></tr>')

    # ---- machine inventory -------------------------------------------------------------
    inv = []
    for m in sorted(machines, key=lambda k: machines[k]["usd_per_hour"]):
        meta = machines[m]
        inv.append(
            f'<tr><td class="mono">{esc(m)}</td>'
            f'<td>{esc(meta.get("cpu_model",""))}</td>'
            f'<td>{esc(meta.get("gpu") or "—")}</td>'
            f'<td class="num">{meta.get("vcpu","?")} / {meta.get("cores","?")}</td>'
            f'<td class="num">${meta["usd_per_hour"]:.4f}</td>'
            f'<td class="num gate">{esc(meta.get("tests","?"))} '
            f'<span class="dim">{meta.get("tests_passed","")}</span></td></tr>')

    return dict(cheapest=cheapest, fastest=fastest, nc=nc, chart=chart,
                rank_rows="\n".join(rank_rows), size_rows="\n".join(size_rows),
                inv_rows="\n".join(inv), machines=machines, front=front)


# The page uses {{TOKEN}} placeholders rather than an f-string: the CSS is full of braces
# and escaping every one of them would make the stylesheet unreadable and easy to break.
TEMPLATE = r"""<title>Cost of a Groth16 Proof</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
/* Light is the base palette. Dark redefines only the tokens, twice: once for the
   un-stamped "system" state via prefers-color-scheme, once for an explicit toggle. Every
   component reads tokens, never a literal, so all three states resolve as a set. */
:root{
  --ground:#F4F6F3; --surface:#FFFFFF; --sunk:#EDF0EC;
  --ink:#131817; --ink-2:#3D4A47; --muted:#677772; --hair:#D9DFD8;
  --cost:#8A6D12;      /* ochre: the money axis */
  --speed:#1F6E7A;     /* slate cyan: the time axis */
  --cost-soft:#F0E9D2; --speed-soft:#DCEDEF;
  --warn:#8A3D1A;
  --shadow:0 1px 2px rgba(19,24,23,.06),0 8px 24px -12px rgba(19,24,23,.18);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1413; --surface:#151D1B; --sunk:#111917;
    --ink:#E7EDEA; --ink-2:#B9C6C1; --muted:#83958F; --hair:#24312E;
    --cost:#D9B45A; --speed:#5FC3CE;
    --cost-soft:#2A2416; --speed-soft:#122A2D;
    --warn:#E08A5C;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"]{
  --ground:#0E1413; --surface:#151D1B; --sunk:#111917;
  --ink:#E7EDEA; --ink-2:#B9C6C1; --muted:#83958F; --hair:#24312E;
  --cost:#D9B45A; --speed:#5FC3CE;
  --cost-soft:#2A2416; --speed-soft:#122A2D;
  --warn:#E08A5C;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:clamp(24px,5vw,64px) clamp(18px,4vw,40px) 96px;}
h1,h2,h3,.label,.mono,th,.num,.chip,.big{font-family:Archivo,"Helvetica Neue",Arial,sans-serif}
h1{font-size:clamp(30px,5.2vw,50px); line-height:1.04; letter-spacing:-.025em; font-weight:700;
   margin:0 0 14px; text-wrap:balance;}
h2{font-size:clamp(20px,2.6vw,26px); letter-spacing:-.015em; font-weight:600; margin:0 0 6px; text-wrap:balance}
h3{font-size:15px; letter-spacing:-.005em; font-weight:600; margin:0 0 8px}
p{margin:0 0 14px; max-width:66ch}
a{color:var(--speed)}
.label{font-size:11px; text-transform:uppercase; letter-spacing:.13em; font-weight:600; color:var(--muted)}
.mono,.num{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums}
.lede{font-size:clamp(17px,2vw,20px); color:var(--ink-2); max-width:62ch; margin-bottom:22px}
section{margin-top:56px}
.rule{height:1px; background:var(--hair); border:0; margin:0}

/* provenance strip: small, factual, always visible */
.prov{display:flex; flex-wrap:wrap; gap:6px 18px; padding:12px 0 0; border-top:1px solid var(--hair);
      font-size:12px; color:var(--muted); font-family:"IBM Plex Mono",monospace}
.prov b{color:var(--ink-2); font-weight:500}

/* the two answers */
.verdicts{display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin:28px 0 0}
.card{background:var(--surface); border:1px solid var(--hair); border-radius:3px; padding:20px 22px 22px;
      box-shadow:var(--shadow); position:relative; overflow:hidden}
.card::before{content:""; position:absolute; inset:0 auto 0 0; width:3px}
.card.cost::before{background:var(--cost)} .card.speed::before{background:var(--speed)}
.card .label{display:block; margin-bottom:10px}
.big{font-size:clamp(28px,4vw,38px); font-weight:700; letter-spacing:-.03em; line-height:1;
     font-family:"IBM Plex Mono",monospace}
.card.cost .big{color:var(--cost)} .card.speed .big{color:var(--speed)}
.unit{font-size:14px; font-weight:500; color:var(--muted); letter-spacing:0}
.card .mach{display:block; margin-top:12px; font-size:16px; font-weight:600}
.card .sub{font-size:13px; color:var(--muted); font-family:"Source Serif 4",serif; margin-top:4px}

/* chart */
.chartbox{background:var(--surface); border:1px solid var(--hair); border-radius:3px; padding:18px;
          box-shadow:var(--shadow); margin-top:18px}
canvas{display:block; width:100%; height:420px}
.legend{display:flex; flex-wrap:wrap; gap:14px; margin-top:12px; font-size:12px; color:var(--muted);
        font-family:"IBM Plex Mono",monospace}
.key{display:inline-flex; align-items:center; gap:6px}
.dot{width:9px; height:9px; border-radius:50%; display:inline-block}

/* tables */
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--hair);
        border-radius:3px; background:var(--surface); box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; font-size:14px; min-width:640px}
th{font-size:11px; text-transform:uppercase; letter-spacing:.09em; font-weight:600; color:var(--muted);
   text-align:left; padding:11px 14px; border-bottom:1px solid var(--hair); white-space:nowrap; background:var(--sunk)}
td{padding:10px 14px; border-bottom:1px solid var(--hair); vertical-align:baseline}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right; font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
.strong{font-weight:600}
.dim{color:var(--muted)}
.cost{color:var(--cost)} .speed{color:var(--speed)}
.mach{font-weight:600; font-size:13.5px}
.acc{display:block; font-size:11.5px; color:var(--muted); font-family:"Source Serif 4",serif; margin-top:1px}
tr.front td{background:linear-gradient(90deg,var(--speed-soft),transparent 60%)}
.chip{display:inline-block; font-size:10.5px; text-transform:uppercase; letter-spacing:.07em;
      font-weight:600; padding:2px 7px; border-radius:2px; border:1px solid var(--hair); color:var(--ink-2)}
.chip-cuda{background:var(--speed-soft); border-color:transparent; color:var(--speed)}
.chip-metal{background:var(--speed-soft); border-color:transparent; color:var(--speed)}
.chip-cpu{background:var(--sunk)}
.gate{font-size:12px}

/* notes */
.notes{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:0 32px}
.note{padding:16px 0; border-top:1px solid var(--hair)}
.note h3{color:var(--ink)}
.note p{font-size:15px; color:var(--ink-2); margin:0}
.callout{border-left:3px solid var(--warn); background:var(--surface); padding:14px 18px; margin:18px 0;
         border-radius:0 3px 3px 0}
.callout p{margin:0; font-size:15px}
.callout .label{color:var(--warn); display:block; margin-bottom:5px}
@media (prefers-reduced-motion:no-preference){.card{transition:transform .18s ease}}
:focus-visible{outline:2px solid var(--speed); outline-offset:2px}
</style>

<div class="wrap">
  <header>
    <h1>What a zero-knowledge proof costs</h1>
    <p class="lede">{{LEDE}}</p>
    <div class="prov">
      <span><b>circuit</b> {{HEADLINE}} · {{NC}} constraints</span>
      <span><b>region</b> us-east-1</span>
      <span><b>mode</b> warm</span>
      <span><b>reps</b> 10, median</span>
      <span><b>gate</b> every box passed its full test suite</span>
    </div>
  </header>

  <div class="verdicts">
    <div class="card cost">
      <span class="label">Cheapest per proof</span>
      <span class="big">{{CHEAP_COST}}<span class="unit"> / 1k proofs</span></span>
      <span class="mach mono">{{CHEAP_MACHINE}}</span>
      <span class="sub">{{CHEAP_SUB}}</span>
    </div>
    <div class="card speed">
      <span class="label">Fastest per proof</span>
      <span class="big">{{FAST_MS}}<span class="unit"> ms</span></span>
      <span class="mach mono">{{FAST_MACHINE}}</span>
      <span class="sub">{{FAST_SUB}}</span>
    </div>
  </div>

  <section>
    <h2>The tradeoff</h2>
    <p>Every machine in the sweep, warm, on the {{NC}}-constraint circuit. Down is cheaper,
    left is faster. {{FRONTIER_NOTE}}</p>
    <div class="chartbox">
      <canvas id="c" aria-label="Cost per thousand proofs against milliseconds per proof"></canvas>
      <div class="legend">
        <span class="key"><span class="dot" style="background:var(--speed)"></span>GPU (cuda)</span>
        <span class="key"><span class="dot" style="background:var(--cost)"></span>CPU</span>
        <span class="key">▬ Pareto frontier</span>
        <span class="key">bubble area ∝ $/hour</span>
      </div>
    </div>
  </section>

  <section>
    <h2>Ranked by cost</h2>
    <p>Cost assumes the machine proves continuously. Idle time is not modelled and would
    raise every row. Shaded rows are on the frontier.</p>
    <div class="scroll"><table>
      <thead><tr><th class="num">#</th><th>machine</th><th>backend</th>
      <th class="num">ms / proof</th><th class="num">$ / 1k</th><th class="num">$ / 1k spot</th>
      <th class="num">proofs / $</th></tr></thead>
      <tbody>{{RANK_ROWS}}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Does the answer change with circuit size?</h2>
    <p>A GPU has a fixed cost per proof that a small circuit cannot amortise, so the winner
    at three thousand constraints need not be the winner at a hundred and forty thousand.</p>
    <div class="scroll"><table>
      <thead><tr><th>circuit</th><th class="num">constraints</th><th>cheapest</th>
      <th class="num">$ / 1k</th><th>fastest</th><th class="num">ms</th></tr></thead>
      <tbody>{{SIZE_ROWS}}</tbody>
    </table></div>
  </section>

  <section>
    <h2>The machines</h2>
    <p>Prices are AWS's published on-demand rate for us-east-1 Linux, cross-checked against
    an independent source. A box only appears here if it passed its full test suite first.</p>
    <div class="scroll"><table>
      <thead><tr><th>instance</th><th>cpu</th><th>gpu</th><th class="num">vcpu / cores</th>
      <th class="num">$ / hr</th><th class="num">gate</th></tr></thead>
      <tbody>{{INV_ROWS}}</tbody>
    </table></div>
  </section>

  <section>
    <h2>What this does not say</h2>
    <div class="notes">{{NOTES}}</div>
  </section>
</div>

<script>
const DATA = {{CHART_JSON}};
const cv = document.getElementById('c');
function tok(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function draw(){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  const ink=tok('--ink'), muted=tok('--muted'), hair=tok('--hair'),
        cost=tok('--cost'), speed=tok('--speed');
  const P={l:66,r:18,t:16,b:42};
  const xs=DATA.map(d=>d.ms), ys=DATA.map(d=>d.cost);
  // log x: the spread runs from tens of ms to thousands, and a linear axis would pile
  // every GPU into one pixel column.
  const x0=Math.log10(Math.min(...xs)*0.82), x1=Math.log10(Math.max(...xs)*1.18);
  const y1=Math.max(...ys)*1.10, y0=0;
  const X=v=>P.l+(Math.log10(v)-x0)/(x1-x0)*(w-P.l-P.r);
  const Y=v=>h-P.b-(v-y0)/(y1-y0)*(h-P.t-P.b);
  g.font='11px "IBM Plex Mono", monospace'; g.textBaseline='middle';
  // horizontal grid + $ axis
  const steps=5;
  for(let i=0;i<=steps;i++){
    const v=y0+(y1-y0)*i/steps, y=Y(v);
    g.strokeStyle=hair; g.lineWidth=1; g.beginPath(); g.moveTo(P.l,y+.5); g.lineTo(w-P.r,y+.5); g.stroke();
    g.fillStyle=muted; g.textAlign='right'; g.fillText('$'+v.toFixed(3), P.l-10, y);
  }
  // decade ticks on ms
  g.textAlign='center';
  for(let e=Math.ceil(x0); e<=Math.floor(x1); e++){
    for(const m of [1,2,5]){
      const v=m*Math.pow(10,e); if(Math.log10(v)<x0||Math.log10(v)>x1) continue;
      const x=X(v);
      g.strokeStyle=hair; g.beginPath(); g.moveTo(x+.5,P.t); g.lineTo(x+.5,h-P.b); g.stroke();
      g.fillStyle=muted; g.fillText(v>=1000?(v/1000)+'s':v+'ms', x, h-P.b+15);
    }
  }
  g.fillStyle=muted; g.textAlign='left';
  g.fillText('$ per 1000 proofs', P.l-56, P.t+2);
  // frontier
  const f=DATA.filter(d=>d.front).sort((a,b)=>a.ms-b.ms);
  if(f.length>1){
    g.strokeStyle=speed; g.lineWidth=1.5; g.setLineDash([5,4]); g.beginPath();
    f.forEach((d,i)=>i?g.lineTo(X(d.ms),Y(d.cost)):g.moveTo(X(d.ms),Y(d.cost)));
    g.stroke(); g.setLineDash([]);
  }
  // points, area proportional to hourly price
  DATA.forEach(d=>{
    const r=Math.max(5,Math.sqrt(d.price)*7.5);
    const col=(d.b==='cpu')?cost:speed;
    g.beginPath(); g.arc(X(d.ms),Y(d.cost),r,0,7);
    g.globalAlpha=d.front?0.9:0.42; g.fillStyle=col; g.fill();
    g.globalAlpha=1; g.lineWidth=d.front?2:1; g.strokeStyle=col; g.stroke();
  });
  // label only the frontier, so the plot stays readable
  g.font='600 11px Archivo, sans-serif'; g.fillStyle=ink;
  f.forEach(d=>{
    const r=Math.max(5,Math.sqrt(d.price)*7.5);
    g.textAlign='left'; g.fillText(d.m, X(d.ms)+r+6, Y(d.cost));
  });
}
const ro=new ResizeObserver(draw); ro.observe(cv);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
draw();
</script>
"""


NOTES = [
    ("Cost assumes the box never idles",
     "Every cost here is the hourly rate divided into continuous proving. A prover that "
     "sits idle between requests pays the same rent for fewer proofs, and a fleet that "
     "scales to zero pays for boot as well. Treat these as a floor, not a forecast."),
    ("A GPU box is paying for an idle CPU",
     "On the GPU machines the host cores do almost nothing while the card works, and they "
     "are on the bill. Co-scheduling CPU proofs alongside GPU proofs would raise throughput "
     "per dollar at no extra rent. That is not measured here, so every GPU row is "
     "pessimistic by an amount this sweep does not know."),
    ("Spot prices move, and not uniformly",
     "The spot column is a single sample. The discount ran 60 to 66 percent on these CPU "
     "families but only 22 percent on the Ada GPUs, which is enough to reorder the ranking "
     "on its own. Re-sample before quoting it."),
    ("Warm is not the only number that matters",
     "Warm pays setup once and then proves in a loop, which is what a resident service "
     "does. A one-shot invocation pays key parse and, on a GPU, module load and upload "
     "every time. Both are in the repository; only warm is on this page."),
    ("A fresh GPU box cannot prove anything for minutes",
     "The first CUDA run on a new instance compiles the kernels: NVRTC to PTX, then the "
     "driver's JIT to machine code. On a T4 that was measured at 113 s plus roughly 175 s. "
     "It is per machine, not per proof, so it is outside every timing here, but an "
     "autoscaled fleet pays it on every new instance."),
    ("One machine failed and published nothing",
     "The arm64 GPU box failed its correctness gate: its AMI shipped an NVRTC newer than "
     "its own driver, and every MSM module load was rejected. Its timings were discarded "
     "rather than reported, which is the point of gating on the suite."),
]


def main():
    d = load()
    b = build(d)
    if not b["cheapest"]:
        raise SystemExit("no warm data for the headline circuit")
    ch, fa = b["cheapest"], b["fastest"]
    chm, fam = b["machines"][ch["machine"]], b["machines"][fa["machine"]]

    notes = "\n".join(
        f'<div class="note"><h3>{esc(t)}</h3><p>{esc(p)}</p></div>' for t, p in NOTES)

    same = ch["machine"] == fa["machine"] and ch["backend"] == fa["backend"]
    n_mach = len(b["machines"])
    cpu_best = min((c for c in b["chart"] if c["b"] == "cpu"),
                   key=lambda c: c["cost"], default=None)
    if same:
        lede = (f"{n_mach} EC2 machines, one Groth16 prover, the same commit compiled on "
                f"every one of them. One machine is both the cheapest and the fastest, "
                f"which is not the tradeoff this was expected to find")
        if cpu_best:
            lede += (f": the best CPU box costs "
                     f"{cpu_best['cost']/ch['usd_per_k']:.1f}x more per proof than "
                     f"{esc(ch['machine'])} and takes "
                     f"{cpu_best['ms']/fa['ms']:.1f}x as long")
        lede += "."
    else:
        lede = (f"{n_mach} EC2 machines, one Groth16 prover, the same commit compiled on "
                f"every one of them. The cheapest machine and the fastest machine are not "
                f"the same machine: {esc(ch['machine'])} costs "
                f"{fa['usd_per_k']/ch['usd_per_k']:.1f}x less per proof, "
                f"{esc(fa['machine'])} is {ch['ms']/fa['ms']:.1f}x faster.")
    nfront = len(b["front"])
    if nfront <= 1:
        frontier = ("Only one machine is on the Pareto frontier, which means it is not a "
                    "frontier at all: that machine is cheaper and faster than every other "
                    "box here, so nothing else in this sweep is worth buying for this "
                    "circuit.")
    else:
        frontier = (f"The dashed line is the Pareto frontier, the {nfront} machines where "
                    f"buying more speed costs more money. Everything above and right of it "
                    f"is strictly worse than something on it.")

    html = TEMPLATE
    for k, v in {
        "{{LEDE}}": lede,
        "{{FRONTIER_NOTE}}": frontier,
        "{{HEADLINE}}": HEADLINE,
        "{{NC}}": f"{b['nc']:,}",
        "{{CHEAP_COST}}": f"${ch['usd_per_k']:,.3f}",
        "{{CHEAP_MACHINE}}": ch["machine"],
        "{{CHEAP_SUB}}": (f"{chm.get('gpu') or chm.get('cpu_model','')} · "
                          f"${chm['usd_per_hour']:.4f}/hr · {ch['ms']:,.0f} ms per proof · "
                          f"{1000/ch['usd_per_k']:,.0f} proofs per dollar"),
        "{{FAST_MS}}": f"{fa['ms']:,.0f}",
        "{{FAST_MACHINE}}": fa["machine"],
        "{{FAST_SUB}}": (f"{fam.get('gpu') or fam.get('cpu_model','')} · "
                         f"${fam['usd_per_hour']:.4f}/hr · ${fa['usd_per_k']:,.3f} per 1000 "
                         f"proofs, {fa['usd_per_k']/ch['usd_per_k']:.1f}x the cheapest"),
        "{{RANK_ROWS}}": b["rank_rows"],
        "{{SIZE_ROWS}}": b["size_rows"],
        "{{INV_ROWS}}": b["inv_rows"],
        "{{NOTES}}": notes,
        "{{CHART_JSON}}": json.dumps(b["chart"]),
    }.items():
        html = html.replace(k, v)

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "cost-report.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html):,} bytes, {len(b['chart'])} points, "
          f"{len(b['front'])} on the frontier)")
    print(f"  cheapest: {ch['machine']}/{ch['backend']} ${ch['usd_per_k']:.4f}/1k")
    print(f"  fastest:  {fa['machine']}/{fa['backend']} {fa['ms']:.1f} ms")


if __name__ == "__main__":
    main()
