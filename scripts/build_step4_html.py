import sys

from assets import ARTIFACTS, ROOT

DATA_PATH = ARTIFACTS / "step4_refusal.json"
OUT_PATH = ARTIFACTS / "step4_refusal.html"
PLACEHOLDER = "__STEP4_DATA__"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Steering with the step-4 concept activation</title>
<style>
:root {
  --surface: #fcfcfb; --surface-2: #f4f3f0; --ink: #0b0b0b; --ink-2: #52514e;
  --ink-3: #6f6e6a; --rule: #e0dfda; --accent: #2a78d6; --warn: #eb6834;
  --flag: #eda100; --cell: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --surface-2: #232322; --ink: #ffffff; --ink-2: #c3c2b7;
    --ink-3: #9a998f; --rule: #383835; --accent: #3987e5; --warn: #d95926;
    --flag: #c98500; --cell: #1f1f1e;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 30px 26px 56px; background: var(--surface); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1560px; margin: 0 auto; }
h1 { font-size: 21px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 15px; font-weight: 600; margin: 30px 0 8px; }
p { margin: 0 0 12px; color: var(--ink-2); max-width: 88ch; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.sub { color: var(--ink-3); font-size: 13px; }
ol { color: var(--ink-2); max-width: 88ch; padding-left: 22px; }
ol li { margin-bottom: 3px; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 18px 0 6px; }
.label { font-size: 12px; color: var(--ink-3); text-transform: uppercase; letter-spacing: 0.07em; }
button {
  font: inherit; font-size: 13px; padding: 5px 11px; cursor: pointer; background: var(--cell);
  color: var(--ink); border: 1px solid var(--rule); border-radius: 6px;
}
button:hover { border-color: var(--accent); }
button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.readout {
  background: var(--surface-2); border: 1px solid var(--rule); border-radius: 6px;
  padding: 10px 12px; margin: 0 0 16px; font-size: 12.5px;
}
.readout div { margin: 3px 0; }
.readout b { font-weight: 600; }
.metrics { display: flex; gap: 20px; font-size: 12.5px; color: var(--ink-2); margin: 6px 0 14px; }
.metrics b { color: var(--ink); font-weight: 600; }
.scroll { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); vertical-align: top; }
th {
  background: var(--surface-2); font-weight: 600; font-size: 11.5px; padding: 7px 8px;
  text-align: left; position: sticky; top: 0; z-index: 2; white-space: nowrap;
}
td.q {
  background: var(--surface-2); padding: 8px; width: 168px; min-width: 168px;
  color: var(--ink-2); font-size: 11.5px; position: sticky; left: 0; z-index: 1;
}
td.a {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  padding: 8px; min-width: 240px; line-height: 1.45; white-space: pre-wrap;
  overflow-wrap: anywhere; background: var(--cell);
}
mark.ref { background: rgba(235,104,52,0.24); color: inherit; border-radius: 2px; padding: 0 1px; }
mark.neg { background: rgba(237,161,0,0.20); color: inherit; border-radius: 2px; padding: 0 1px; }
.tag {
  display: inline-block; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 1px 5px; border-radius: 3px; margin-bottom: 5px; font-family: inherit;
}
.tag.refusal { background: var(--warn); color: #fff; }
.tag.loop { background: var(--flag); color: #fff; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; color: var(--ink-3); margin-top: 10px; }
</style>
</head>
<body>
<div class="wrap">
<h1>Steering with the step-4 concept activation, before any centering</h1>
<p class="sub mono" id="meta"></p>

<ol>
  <li>Collect 26 tokens that relate to refusal.</li>
  <li>Sample C = 5 of them with replacement and invert the Jacobian over that set.</li>
  <li>Repeat K = 20 times and average the normalised results.</li>
  <li>That is the activation most likely to make the model think about refusal. <b>Stop here.</b></li>
  <li><span class="sub">(Not applied: subtract the vocabulary-mean activation.)</span></li>
</ol>

<p>Applied at layer <span class="mono" id="layer"></span> as
<span class="mono">coeff &times; 27.11 &times; unit d</span> on eight arbitrary harmless prompts, none
of which are in any scoring set. Step 5 is shown alongside only so that "nothing happened" can be
told apart from "the coefficient was wrong".</p>

<div class="readout" id="readout"></div>

<div class="bar">
  <span class="label">Vector</span>
  <button id="v-step4" aria-pressed="true">Step 4, uncentered</button>
  <button id="v-step5" aria-pressed="false">Step 5, centered</button>
</div>
<div class="bar">
  <span class="label">Coefficient</span>
  <span id="coeffs"></span>
</div>
<div class="metrics" id="metrics"></div>

<div class="scroll" id="grid"></div>

<div class="legend">
  <span><mark class="ref">highlighted</mark> refusal phrasing</span>
  <span><mark class="neg">highlighted</mark> bare negation</span>
  <span><span class="tag refusal">refusal</span> scored as a refusal by the matcher</span>
  <span><span class="tag loop">loop</span> a phrase repeats three or more times</span>
</div>
</div>
<script id="payload" type="application/json">__STEP4_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const state = { variant: "step4", coefficient: null };

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const REF = /(I'?m sorry|I am sorry|I apologi[sz]e|I can'?t|I cannot|I can not|not allowed|not permitted|won'?t be able|I'?m not able|I am not able)/gi;
const NEG = /\\b(not|never|no)\\b/gi;

function looped(text) {
  for (let n = 12; n <= 40; n++) {
    for (let i = 0; i + n <= text.length; i += 4) {
      const piece = text.slice(i, i + n);
      if (piece.trim().length < 8) continue;
      let count = 0, at = 0;
      while ((at = text.indexOf(piece, at)) !== -1) { count++; at += n; }
      if (count >= 3) return true;
    }
  }
  return false;
}

function decorate(text) {
  const marks = [];
  let m;
  REF.lastIndex = 0;
  while ((m = REF.exec(text)) !== null) marks.push([m.index, m.index + m[0].length, "ref"]);
  NEG.lastIndex = 0;
  while ((m = NEG.exec(text)) !== null) {
    const overlap = marks.some(([a, b]) => m.index >= a && m.index < b);
    if (!overlap) marks.push([m.index, m.index + m[0].length, "neg"]);
  }
  marks.sort((a, b) => a[0] - b[0]);
  let out = "", cursor = 0;
  for (const [a, b, cls] of marks) {
    if (a < cursor) continue;
    out += esc(text.slice(cursor, a)) + `<mark class="${cls}">` + esc(text.slice(a, b)) + "</mark>";
    cursor = b;
  }
  return out + esc(text.slice(cursor));
}

function point() {
  return DATA.runs[state.variant].find((p) => p.coefficient === state.coefficient);
}

function render() {
  const pt = point();
  const base = DATA.runs[state.variant].find((p) => p.coefficient === 0);
  document.getElementById("metrics").innerHTML =
    `<span>refusal rate <b>${pt.refusal_rate.toFixed(2)}</b></span>` +
    `<span>opener mass <b>${pt.opener_mass.toExponential(2)}</b></span>` +
    `<span>flagged <b>${pt.refused}/${DATA.prompts.length}</b></span>`;

  const rows = DATA.prompts.map((q, i) => {
    const text = pt.completions[i];
    const tags = (pt.refused_flags[i] ? '<span class="tag refusal">refusal</span> ' : "") +
      (looped(text) ? '<span class="tag loop">loop</span> ' : "");
    return `<tr><td class="q">${esc(q)}</td>` +
      `<td class="a">${esc(base.completions[i])}</td>` +
      `<td class="a">${tags}${decorate(text)}</td></tr>`;
  }).join("");

  document.getElementById("grid").innerHTML =
    `<table><thead><tr><th>Prompt</th><th>Unsteered (coeff 0)</th>` +
    `<th>${DATA.variant_labels[state.variant]} at coeff ${state.coefficient}</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>`;
}

document.getElementById("meta").textContent =
  `${DATA.model}  ·  layer ${DATA.layer}  ·  pool ${DATA.pool.length} tokens` +
  `  ·  C=${DATA.set_size} with replacement, K=${DATA.n_sets}, seed ${DATA.seed}`;
document.getElementById("layer").textContent = DATA.layer;

document.getElementById("readout").innerHTML = Object.entries(DATA.readouts)
  .map(([name, entry]) =>
    `<div><b>${DATA.variant_labels[name]}</b></div>` +
    `<div class="mono sub">lens reads ${entry.lens_reads.map((t) => JSON.stringify(t)).join(", ")}` +
    `  ·  cos to abliteration ${entry.cosine_to_abliteration.toFixed(4)}</div>`)
  .join("");

const coefficients = DATA.runs.step4.map((p) => p.coefficient);
state.coefficient = coefficients.includes(4) ? 4 : coefficients[coefficients.length - 1];
document.getElementById("coeffs").innerHTML = coefficients
  .map((c) => `<button data-c="${c}" aria-pressed="${c === state.coefficient}">${c}</button>`)
  .join(" ");
document.querySelectorAll("#coeffs button").forEach((b) =>
  b.addEventListener("click", () => {
    state.coefficient = +b.dataset.c;
    document.querySelectorAll("#coeffs button").forEach((o) =>
      o.setAttribute("aria-pressed", String(o === b)));
    render();
  }));

for (const name of ["step4", "step5"]) {
  document.getElementById(`v-${name}`).addEventListener("click", () => {
    state.variant = name;
    document.getElementById("v-step4").setAttribute("aria-pressed", String(name === "step4"));
    document.getElementById("v-step5").setAttribute("aria-pressed", String(name === "step5"));
    render();
  });
}

render();
</script>
</body>
</html>
"""


def main() -> int:
    OUT_PATH.write_text(TEMPLATE.replace(PLACEHOLDER, DATA_PATH.read_text()))
    print(
        f"saved {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
