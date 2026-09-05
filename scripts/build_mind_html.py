import sys

from assets import ARTIFACTS, ROOT

DATA_PATH = ARTIFACTS / "refusal_mind.json"
OUT_PATH = ARTIFACTS / "refusal_mind.html"
PLACEHOLDER = "__JLENS_DATA__"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jacobian lens slice: refusal steering on harmless prompts</title>
<style>
:root {
  --surface: #fcfcfb;
  --surface-2: #f4f3f0;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #6f6e6a;
  --rule: #e0dfda;
  --accent: #2a78d6;
  --warn: #eb6834;
  --cell: #ffffff;
  --shade: 11, 11, 11;
  --ramp: 42, 120, 214;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19;
    --surface-2: #232322;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --ink-3: #9a998f;
    --rule: #383835;
    --accent: #3987e5;
    --warn: #d95926;
    --cell: #1f1f1e;
    --shade: 255, 255, 255;
    --ramp: 57, 135, 229;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 28px 56px;
  background: var(--surface); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1500px; margin: 0 auto; }
h1 { font-size: 21px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 15px; font-weight: 600; margin: 34px 0 10px; }
p { margin: 0 0 12px; color: var(--ink-2); max-width: 86ch; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.sub { color: var(--ink-3); font-size: 13px; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 18px 0 14px; }
button {
  font: inherit; font-size: 13px; padding: 5px 11px; cursor: pointer;
  background: var(--cell); color: var(--ink);
  border: 1px solid var(--rule); border-radius: 6px;
}
button:hover { border-color: var(--accent); }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
button.pin { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
button.pin[data-kind="suppressed"]:hover { border-color: var(--warn); }
button.pin[aria-pressed="true"][data-kind="suppressed"] {
  background: var(--warn); border-color: var(--warn);
}
.label { font-size: 12px; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.07em; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; }
@media (max-width: 1120px) { .panels { grid-template-columns: 1fr; } }
.panel { min-width: 0; }
.panel h3 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
.completion {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12.5px; line-height: 1.5; color: var(--ink);
  background: var(--surface-2); border: 1px solid var(--rule); border-left: 3px solid var(--rule);
  border-radius: 5px; padding: 9px 11px; margin: 0 0 12px; white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.panel[data-condition="steered"] .completion { border-left-color: var(--warn); }
.panel[data-condition="base"] .completion { border-left-color: var(--accent); }
.scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; font-size: 11px; width: 100%; }
th, td {
  border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 0; text-align: center;
}
th.pos {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 500; color: var(--ink-2); background: var(--surface-2);
  padding: 5px 3px; white-space: nowrap; font-size: 10.5px;
  position: sticky; top: 0; z-index: 2;
}
td.layer, th.corner {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--ink-3); background: var(--surface-2);
  padding: 0 7px; white-space: nowrap; font-size: 10px;
  position: sticky; left: 0; z-index: 1;
}
th.corner { z-index: 3; }
td.cell {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  cursor: default; padding: 3px 4px; min-width: 54px; max-width: 92px;
  overflow: hidden; white-space: nowrap; background: var(--cell);
}
td.cell .rk { font-size: 8px; color: var(--ink-3); vertical-align: super; margin-left: 1px; }
td.cell.diff { outline: 2px solid var(--warn); outline-offset: -2px; }
td.cell.faded { opacity: 0.28; }
#tip {
  position: fixed; pointer-events: none; opacity: 0; transition: opacity 90ms;
  background: var(--ink); color: var(--surface); border-radius: 6px;
  padding: 8px 10px; font-size: 11.5px; z-index: 40; max-width: 280px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  box-shadow: 0 4px 14px rgba(0,0,0,0.24);
}
#tip .hd { font-weight: 600; margin-bottom: 4px; }
#tip .row { display: flex; justify-content: space-between; gap: 14px; }
#tip .row span:last-child { color: var(--ink-3); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--ink-3); margin-top: 10px; }
.swatches { display: flex; align-items: center; gap: 3px; }
.sw { width: 15px; height: 11px; border: 1px solid var(--rule); }
.tokens { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 900px) { .tokens { grid-template-columns: 1fr; } }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px;
  padding: 3px 7px; border-radius: 4px; border: 1px solid var(--rule); background: var(--cell);
}
.chip b { font-weight: 500; color: var(--ink-3); margin-left: 5px; font-size: 10px; }
.up { border-left: 3px solid var(--accent); }
.down { border-left: 3px solid var(--warn); }
</style>
</head>
<body>
<div class="wrap">
<h1>Jacobian lens slice: what a harmless prompt looks like inside a refusing model</h1>
<p class="sub mono" id="meta"></p>

<p>Each grid is one forward pass. Rows are layers, from the embedding at the top to the
final layer at the bottom; the bottom row is what the model is about to say. Columns are token
positions, starting at the last prompt token (<span class="mono">&rarr;</span>) and running through
the generated completion. Every cell is the Jacobian lens read of that activation: the top-1
token the residual stream is currently disposed to emit, with its rank in the model's real
next-token distribution as a superscript. Cell shading is the lens probability. Hover a cell for
the ranked list; click a highlighted token below to trace its rank through the whole grid.</p>

<p>The left grid is the unmodified model. The right grid has the refusal direction added to the
residual stream at layer <span class="mono" id="layer-note"></span>. Same prompt, same weights.</p>

<div class="bar">
  <span class="label">Prompt</span>
  <span id="prompt-tabs"></span>
  <span style="flex:1"></span>
  <span class="label">View</span>
  <button id="mode-top" aria-pressed="true">Top-1 token</button>
  <button id="mode-diff" aria-pressed="false">Highlight disagreements</button>
  <button id="mode-clear" hidden>Clear pin</button>
</div>

<div class="bar">
  <span class="label">Pin a token</span>
  <span id="pins" class="chips"></span>
</div>

<div class="panels" id="panels"></div>

<div class="legend">
  <span class="swatches"><span class="label" style="margin-right:6px">Lens probability</span>
    <span class="sw" style="background:var(--cell)"></span>
    <span class="sw" style="background:rgba(var(--shade),0.10)"></span>
    <span class="sw" style="background:rgba(var(--shade),0.22)"></span>
    <span class="sw" style="background:rgba(var(--shade),0.38)"></span>
    <span class="sw" style="background:rgba(var(--shade),0.55)"></span>
  </span>
  <span class="swatches"><span class="label" style="margin-right:6px">Pinned rank</span>
    <span class="sw" style="background:rgba(var(--ramp),0.85)"></span>
    <span class="sw" style="background:rgba(var(--ramp),0.55)"></span>
    <span class="sw" style="background:rgba(var(--ramp),0.30)"></span>
    <span class="sw" style="background:rgba(var(--ramp),0.12)"></span>
    <span class="sw" style="background:var(--cell)"></span>
    <span style="margin-left:6px">rank 1 &rarr; unranked</span>
  </span>
</div>

<h2>Which tokens the steering moves</h2>
<p>Mean lens log-probability shift at the last prompt token, averaged over layers
<span class="mono" id="rank-layers"></span> and over all ten harmless prompts. This is the ranking
that the derived refusal vector was built from.</p>
<div class="tokens">
  <div>
    <p class="label">Promoted by steering</p>
    <div class="chips" id="promoted"></div>
  </div>
  <div>
    <p class="label">Suppressed by steering</p>
    <div class="chips" id="suppressed"></div>
  </div>
</div>
</div>
<div id="tip"></div>
<script id="payload" type="application/json">__JLENS_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const CONDITIONS = [["base", "Unmodified"], ["steered", "Refusal steered"]];
const state = { prompt: 0, mode: "top", pin: null };

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const show = (s) => esc(s.replace(/\\n/g, "\\\\n").replace(/\\t/g, "\\\\t")) || "&nbsp;";
const clip = (s, n) => (s.length > n ? s.slice(0, n) + "\\u2026" : s);

function shade(p) {
  return p <= 0.002 ? "var(--cell)" : `rgba(var(--shade), ${Math.min(0.55, 0.06 + 0.6 * Math.sqrt(p)).toFixed(3)})`;
}
function rankShade(rank) {
  const a = Math.max(0, 1 - Math.log10(Math.max(1, rank)) / 4.2);
  return a < 0.02 ? "var(--cell)" : `rgba(var(--ramp), ${(0.85 * a).toFixed(3)})`;
}

function buildTable(view, other) {
  const positions = view.position_tokens;
  const head = positions.map((t) => `<th class="pos">${show(clip(t, 8))}</th>`).join("");
  const rows = view.layers
    .map((layer, li) => {
      const cells = view.cells[li]
        .map((cell, pi) => {
          const differs = other && other.cells[li][pi].t !== cell.t;
          return `<td class="cell" data-l="${li}" data-p="${pi}"${differs ? ' data-differs="1"' : ""}>` +
            `${show(clip(cell.t, 9))}<span class="rk">${cell.r}</span></td>`;
        })
        .join("");
      return `<tr><td class="layer">L${String(layer).padStart(2, "0")}</td>${cells}</tr>`;
    })
    .join("");
  return `<div class="scroll"><table><thead><tr><th class="corner">layer</th>${head}</tr></thead>` +
    `<tbody>${rows}</tbody></table></div>`;
}

function render() {
  const view = DATA.views[state.prompt];
  document.getElementById("panels").innerHTML = CONDITIONS.map(([key, title]) => {
    const self = view[key];
    const other = view[key === "base" ? "steered" : "base"];
    const note = key === "steered" ? `coefficient ${DATA.steer_coefficient}` : "no steering";
    return `<div class="panel" data-condition="${key}"><h3>${title}` +
      ` <span class="sub">&middot; ${note}</span></h3>` +
      `<div class="completion">${show(self.completion)}</div>${buildTable(self, other)}</div>`;
  }).join("");
  paint();
  attach();
}

function paint() {
  const view = DATA.views[state.prompt];
  document.querySelectorAll(".panel").forEach((panel) => {
    const self = view[panel.dataset.condition];
    panel.querySelectorAll("td.cell").forEach((td) => {
      const li = +td.dataset.l, pi = +td.dataset.p;
      const cell = self.cells[li][pi];
      td.classList.toggle("diff", state.mode === "diff" && td.dataset.differs === "1");
      td.classList.toggle("faded", state.mode === "diff" && td.dataset.differs !== "1");
      if (state.pin !== null) {
        td.style.background = rankShade(self.pinned_ranks[li][state.pin][pi]);
        td.innerHTML = `${show(clip(cell.t, 9))}<span class="rk">${self.pinned_ranks[li][state.pin][pi]}</span>`;
      } else {
        td.style.background = shade(cell.p);
        td.innerHTML = `${show(clip(cell.t, 9))}<span class="rk">${cell.r}</span>`;
      }
    });
  });
}

const tip = document.getElementById("tip");
function attach() {
  const view = DATA.views[state.prompt];
  document.querySelectorAll(".panel").forEach((panel) => {
    const self = view[panel.dataset.condition];
    panel.querySelectorAll("td.cell").forEach((td) => {
      td.addEventListener("mousemove", (event) => {
        const cell = self.cells[+td.dataset.l][+td.dataset.p];
        const pinLine = state.pin === null ? "" :
          `<div class="row"><span>${esc(DATA.pinnable[state.pin].token)}</span>` +
          `<span>rank ${self.pinned_ranks[+td.dataset.l][state.pin][+td.dataset.p]}</span></div>`;
        tip.innerHTML = `<div class="hd">L${self.layers[+td.dataset.l]} &middot; pos ` +
          `${+td.dataset.p} &middot; real rank ${cell.r}</div>` +
          cell.top.map(([t, p]) =>
            `<div class="row"><span>${show(clip(t, 16))}</span><span>${(100 * p).toFixed(1)}%</span></div>`
          ).join("") + pinLine;
        tip.style.opacity = 1;
        const box = tip.getBoundingClientRect();
        tip.style.left = Math.min(event.clientX + 14, innerWidth - box.width - 12) + "px";
        tip.style.top = Math.min(event.clientY + 14, innerHeight - box.height - 12) + "px";
      });
      td.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
    });
  });
}

function setMode(mode) {
  state.mode = mode;
  document.getElementById("mode-top").setAttribute("aria-pressed", mode === "top");
  document.getElementById("mode-diff").setAttribute("aria-pressed", mode === "diff");
  paint();
}

function setPin(index) {
  state.pin = state.pin === index ? null : index;
  document.querySelectorAll("#pins .pin").forEach((b, i) =>
    b.setAttribute("aria-pressed", String(i === state.pin)));
  document.getElementById("mode-clear").hidden = state.pin === null;
  paint();
}

document.getElementById("meta").textContent =
  `${DATA.model}  ·  steering at layer ${DATA.steer_layer}, coefficient ${DATA.steer_coefficient}` +
  `  ·  lens: neuronpedia/jacobian-lens`;
document.getElementById("layer-note").textContent = DATA.steer_layer;
document.getElementById("rank-layers").textContent =
  `${DATA.rank_layers[0]}\\u2013${DATA.rank_layers[DATA.rank_layers.length - 1]}`;

document.getElementById("prompt-tabs").innerHTML = DATA.views
  .map((v, i) => `<button data-i="${i}" aria-pressed="${i === 0}">${esc(clip(v.base.prompt, 44))}</button>`)
  .join(" ");
document.querySelectorAll("#prompt-tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    state.prompt = +b.dataset.i;
    document.querySelectorAll("#prompt-tabs button").forEach((o) =>
      o.setAttribute("aria-pressed", String(o === b)));
    render();
  }));

document.getElementById("pins").innerHTML = DATA.pinnable
  .map((entry, i) => {
    const kind = entry.delta_logprob >= 0 ? "promoted" : "suppressed";
    return `<button class="pin" data-kind="${kind}" data-i="${i}" aria-pressed="false">` +
      `${show(clip(entry.token, 10))}</button>`;
  })
  .join("");
document.querySelectorAll("#pins .pin").forEach((b) =>
  b.addEventListener("click", () => setPin(+b.dataset.i)));

document.getElementById("mode-top").addEventListener("click", () => setMode("top"));
document.getElementById("mode-diff").addEventListener("click", () => setMode("diff"));
document.getElementById("mode-clear").addEventListener("click", () => setPin(state.pin));

const chips = (entries, cls) => entries
  .map((e) => `<span class="chip ${cls}">${show(clip(e.token, 14))}` +
    `<b>${e.delta_logprob >= 0 ? "+" : ""}${e.delta_logprob.toFixed(2)}</b></span>`)
  .join("");
document.getElementById("promoted").innerHTML = chips(DATA.ranking.promoted.slice(0, 36), "up");
document.getElementById("suppressed").innerHTML = chips(DATA.ranking.suppressed.slice(0, 36), "down");

render();
</script>
</body>
</html>
"""


def main() -> int:
    data = DATA_PATH.read_text()
    OUT_PATH.write_text(TEMPLATE.replace(PLACEHOLDER, data))
    size = OUT_PATH.stat().st_size / 1024
    print(f"saved {OUT_PATH.relative_to(ROOT)} ({size:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
