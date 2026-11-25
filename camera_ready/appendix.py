"""Camera-ready Appendix F: re-implementation results across models, with intervals, seeds, mixtures and depth.

Reads results/<tag>/{summary,meta,judge_haiku_summary}.json (run analyze.py and JUDGE=haiku judge.py first) and
writes paper/appendix_f/*.tex and *.pdf.

uv run python appendix.py
"""
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from jspace.behaviours import ORDER

OUT = pathlib.Path("paper/appendix_f")
OUT.mkdir(parents=True, exist_ok=True)

# smallest to largest
MODELS = [("g270", "Gemma-3 270M", 0.27), ("q35a", "Qwen3.5 0.8B", 0.8), ("g1", "Gemma-3 1B", 1.0),
          ("q25", "Qwen2.5 1.5B", 1.5), ("q17", "Qwen3 1.7B", 1.7), ("q35b", "Qwen3.5 2B", 2.0),
          ("q4", "Qwen3 4B", 4.0), ("q8", "Qwen3 8B", 8.0)]
BNAME = {"caps": "ALL CAPS", "french": "answer in French", "emoji": "use emoji", "pirate": "pirate speak",
         "hedge": "hedged / uncertain", "syco": "sycophantic agreement", "refusal": "refuse harmless requests",
         "brief": "extreme brevity"}
SHORT = {"caps": "caps", "french": "French", "emoji": "emoji", "pirate": "pirate", "hedge": "hedging",
         "syco": "sycophancy", "refusal": "refusal", "brief": "brevity"}
COMPETITIVE = 2 / 3     # a direction "works" if its lift reaches 2/3 of the fitted lift
FITTED_MIN = 0.30       # fitted lift over unsteered below this: the fitted vector fails, row not classified


def load(tag):
    d = pathlib.Path("results") / tag
    if not (d / "summary.json").exists():
        return None
    S = json.load(open(d / "summary.json"))
    M = json.load(open(d / "meta.json"))
    J = json.load(open(d / "judge_haiku_summary.json")) if (d / "judge_haiku_summary.json").exists() else {}
    return S, M, J


DATA = {t: load(t) for t, _, _ in MODELS}
MODELS = [m for m in MODELS if DATA[m[0]] is not None]


def case(R):
    """Lift over the unsteered model decides: the fitted vector must lift the rate by FITTED_MIN, and a direction
    works if its lift is at least COMPETITIVE times the fitted lift."""
    u = R["unsteered"]
    f = R["fitted"]["mean"] - u
    if f < FITTED_MIN:
        return "-"
    works = lambda k: R[k]["mean"] - u >= COMPETITIVE * f
    if works("ours"):
        return "U" if works("eye") else "J"
    return "E" if works("oracle") else "S"


# ---------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0, "xtick.major.size": 2.5, "axes.labelcolor": "#52514e",
    "pdf.fonttype": 42, "legend.frameon": False})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
METH = [  # key, label, colour, marker, filled
    ("fitted", "fitted (diff-in-means)", "#6f6e69", "D", True),
    ("ours", "ours (J-lens, Eq. 2)", "#2a78d6", "o", True),
    ("eye", r"$J{=}I$ control", "#eb6834", "^", True),
    ("oracle", "oracle projection", "#1baf7a", "s", False),
]


def fig_models():
    n = len(MODELS)
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 0.45 + 0.2 * n * 2 + 0.5), sharex=True, sharey=True)
    off = np.linspace(-0.27, 0.27, len(METH))
    for ax, b in zip(axes.flat, ORDER):
        for y, (t, name, _) in enumerate(MODELS):
            R = DATA[t][0].get(b)
            if R is None:
                continue
            yy = n - 1 - y
            if y % 2 == 0:
                ax.axhspan(yy - 0.5, yy + 0.5, color="#f4f3ef", lw=0, zorder=0)
            ax.plot([R["unsteered"]] * 2, [yy - 0.42, yy + 0.42], color=INK, lw=0.9, zorder=2.5,
                    solid_capstyle="butt")
            for o, (k, lab, c, mk, filled) in zip(off, METH):
                if k not in R:
                    continue
                m, (lo, hi) = R[k]["mean"], R[k]["ci"]
                ax.plot([lo, hi], [yy + o] * 2, color=c, lw=1.0, solid_capstyle="round", zorder=2)
                ax.plot(m, yy + o, marker=mk, ms=3.6 if mk != "^" else 4.0, color=c,
                        mfc=c if filled else "white", mew=0.9, ls="none", zorder=3)
        ax.set_title(SHORT[b], color=INK, pad=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(range(n))
        ax.set_yticklabels([m[1] for m in MODELS][::-1])
    for ax in axes[1]:
        ax.set_xlabel("behaviour rate", labelpad=2)
    handles = [plt.Line2D([], [], color=c, marker=mk, mfc=c if f else "white", mew=0.9, ms=4, lw=1.0, label=l)
               for k, l, c, mk, f in METH]
    handles.append(plt.Line2D([], [], color=INK, marker="|", ms=7, mew=0.9, lw=0, label="unsteered"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.54, 1.0), handlelength=1.4,
               columnspacing=1.0, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=0.6, h_pad=0.9)
    fig.savefig(OUT / "fig_models.pdf")
    plt.close(fig)


def fig_depth():
    """Rate against relative depth for fitted / ours / J=I, one thin line per model and the mean."""
    with_sweep = [m for m in MODELS if DATA[m[0]][1].get("sweep")]
    if not with_sweep:
        return []
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.6), sharex=True, sharey=True)
    grid = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
    for ax, b in zip(axes.flat, ORDER):
        for k, lab, c, mk, filled in METH[:3]:
            curves = []
            for t, name, _ in with_sweep:
                S, M, _ = DATA[t]
                if b not in S:
                    continue
                R, L, nl = S[b], M["layer"], M["n_layers"]
                pts = {}
                for l in [L] + M["sweep"]:
                    key = k if l == L else (f"{k}@{l}" if k == "fitted" else f"{k}_s0@{l}")
                    if l == L:
                        v = R[k]["seed_means"][0] if k != "fitted" else R[k]["mean"]
                    elif key in R:
                        v = R[key]["mean"]
                    else:
                        continue
                    pts[grid[np.argmin(abs(grid - l / nl))]] = v
                if len(pts) == len(grid):
                    curves.append([pts[g] for g in grid])
            if not curves:
                continue
            C = np.array(curves)
            for row in C:
                ax.plot(grid, row, color=c, lw=0.5, alpha=0.22, zorder=1)
            ax.plot(grid, C.mean(0), color=c, lw=1.6, marker=mk, ms=3.2, mfc=c if filled else "white", mew=0.8,
                    zorder=3, label=lab)
        ax.axvline(0.55, color=GRID, lw=0.8, zorder=0)
        ax.set_title(SHORT[b], color=INK, pad=3)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.35, 0.55, 0.75])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("depth / layers", labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("behaviour rate")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0), handlelength=2.2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.5, h_pad=0.8)
    fig.savefig(OUT / "fig_depth.pdf")
    plt.close(fig)
    return [m[1] for m in with_sweep]


def f2(x):
    return f"{x:.2f}"


def tab_q17():
    S, M, J = DATA["q17"]
    rows = []
    for b in ORDER:
        R = S[b]
        cell = lambda k: (rf"{f2(R[k]['mean'])} \ci{{{f2(R[k]['ci'][0])}}}{{{f2(R[k]['ci'][1])}}}" if k in R else "--")
        sd = np.std(R["ours"]["seed_means"], ddof=1)
        gap = R["ours"]["insample"] - R["ours"]["mean"]
        jd = rf"{f2(J[b]['fitted']['judge'])} / {f2(J[b]['ours']['judge'])}" if b in J else "--"
        rows.append(rf"{BNAME[b]} & {cell('fitted')} & {cell('ours')} & {f2(sd)} & {gap:+.2f} & {cell('eye')} & "
                    rf"{cell('oracle')} & {jd} \\")
    (OUT / "tab_q17.tex").write_text("\n".join(rows) + "\n")


def tab_cases():
    head = "& " + " & ".join(r"\shortstack{" + m[1].replace(" ", r"\\") + "}" for m in MODELS) + r" \\"
    rows = []
    for b in ORDER:
        cells = []
        for t, _, _ in MODELS:
            if b not in DATA[t][0]:
                cells.append("")
                continue
            c = case(DATA[t][0][b])
            cells.append({"J": r"\caseJ", "U": r"\caseU", "E": r"\caseE", "S": r"\caseS", "-": r"\caseN"}[c])
        rows.append(rf"{BNAME[b]} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_cases_head.tex").write_text(head + "\n")
    (OUT / "tab_cases.tex").write_text("\n".join(rows) + "\n")


def tab_mix():
    with_mix = [m for m in MODELS if "mix_0.5" in DATA[m[0]][0]["caps"]]
    rows = []
    for b in ORDER:
        ms = [t for t, _, _ in with_mix if b in DATA[t][0]]
        f = [DATA[t][0][b]["fitted"]["mean"] for t in ms]
        o = [DATA[t][0][b]["ours"]["seed_means"][0] for t in ms]
        mx = {lam: [DATA[t][0][b][f"mix_{lam}"]["mean"] for t in ms] for lam in ("0.25", "0.5", "0.75")}
        best = [max(a, c) for a, c in zip(f, o)]
        bestmix = [max(mx[l][i] for l in mx) for i in range(len(ms))]
        wins = sum(bm > bs + 0.05 for bm, bs in zip(bestmix, best))
        rows.append(rf"{BNAME[b]} & {f2(np.mean(f))} & {f2(np.mean(o))} & " +
                    " & ".join(f2(np.mean(mx[l])) for l in ("0.75", "0.5", "0.25")) + rf" & {wins}/{len(ms)} \\")
    (OUT / "tab_mix.tex").write_text("\n".join(rows) + "\n")
    return [m[1] for m in with_mix]


def tab_judge():
    """Judge-selected rates (coefficient chosen on the other fold by the judged score) for hedging and sycophancy."""
    rows = []
    for t, name, _ in MODELS:
        fp = pathlib.Path("results") / t / "judge_haiku_full.json"
        if not fp.exists():
            continue
        F = json.load(open(fp))
        S = DATA[t][0]
        cells = []
        for b in ("hedge", "syco"):
            for k in ("none", "fitted", "ours_s0", "eye_s0", "oracle"):
                cells.append(f2(F[b][k]["judge_selected"]) if b in F and k in F[b] else "--")
        rows.append(rf"{name} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_judge.tex").write_text("\n".join(rows) + "\n")
    return len(rows)


def tab_cos():
    rows = []
    for b in ORDER:
        vals = []
        for t, _, _ in MODELS:
            c = DATA[t][1]["behaviours"].get(b, {}).get("cos_to_fitted", {})
            v = [c[k] for k in c if k.startswith("ours_s")]
            vals.append(f"{np.mean(v):+.2f}" if v else "--")
        rows.append(rf"{BNAME[b]} & " + " & ".join(vals) + r" \\")
    (OUT / "tab_cos.tex").write_text("\n".join(rows) + "\n")


def numbers():
    """Macros quoted in the appendix text."""
    out = {}
    gaps = [DATA[t][0][b][k]["insample"] - DATA[t][0][b][k]["mean"] for t, _, _ in MODELS for b in ORDER
            if b in DATA[t][0] for k in ("fitted", "ours", "eye") if k in DATA[t][0][b]]
    out["GapMean"] = f"{np.mean(gaps):.2f}"
    out["GapMax"] = f"{np.max(gaps):.2f}"
    sds = [np.std(DATA[t][0][b]["ours"]["seed_means"], ddof=1) for t, _, _ in MODELS for b in ORDER if b in DATA[t][0]]
    out["SeedSDMedian"] = f"{np.median(sds):.2f}"
    out["SeedSDMax"] = f"{np.max(sds):.2f}"
    hw = [(DATA[t][0][b][k]["ci"][1] - DATA[t][0][b][k]["ci"][0]) / 2 for t, _, _ in MODELS for b in ORDER
          if b in DATA[t][0] for k in ("fitted", "ours") if b != "syco"]
    out["CIHalfMedian"] = f"{np.median(hw):.2f}"
    out["NModels"] = str(len(MODELS))
    out["NMix"] = str(sum("mix_0.5" in DATA[t][0]["caps"] for t, _, _ in MODELS))
    out["NDepth"] = str(sum(bool(DATA[t][1].get("sweep")) for t, _, _ in MODELS))
    agree = [DATA[t][2][b]["_agreement"]["agree"] for t, _, _ in MODELS for b in ("hedge", "syco") if b in DATA[t][2]]
    out["JudgeAgreeMin"] = f"{min(agree):.2f}" if agree else "--"
    out["JudgeAgreeMax"] = f"{max(agree):.2f}" if agree else "--"
    cases = {}
    for t, _, _ in MODELS:
        for b in ORDER:
            if b in DATA[t][0]:
                cases.setdefault(case(DATA[t][0][b]), []).append((t, b))
    # mixtures: cells where the best mixture beats both endpoints by > 0.05
    wins = cells = 0
    for t, _, _ in MODELS:
        S = DATA[t][0]
        if "mix_0.5" not in S["caps"]:
            continue
        for b in ORDER:
            if b not in S:
                continue
            best = max(S[b]["fitted"]["mean"], S[b]["ours"]["seed_means"][0])
            bm = max(S[b][f"mix_{l}"]["mean"] for l in ("0.25", "0.5", "0.75"))
            wins += bm > best + 0.05
            cells += 1
    out["MixWins"], out["MixCells"] = str(wins), str(cells)
    cs = [np.mean([v for k, v in DATA[t][1]["behaviours"][b]["cos_to_fitted"].items() if k.startswith("ours_s")])
          for t, _, _ in MODELS for b in ORDER if "cos_to_fitted" in DATA[t][1]["behaviours"].get(b, {})]
    out["CosMax"] = f"{max(abs(c) for c in cs):.2f}"
    out["CosMedian"] = f"{np.median(np.abs(cs)):.2f}"
    for k in "JUES-":
        out["Case" + {"-": "N"}.get(k, k)] = str(sum(1 for t, _, _ in MODELS for b in ORDER
                                                    if b in DATA[t][0] and case(DATA[t][0][b]) == k))
    # judge-selected: does the U case (ours and J=I both work) survive the judge?
    for b in ("hedge", "syco"):
        n_ok = n_cls = 0
        for t, _, _ in MODELS:
            fp = pathlib.Path("results") / t / "judge_haiku_full.json"
            if not fp.exists():
                continue
            F = json.load(open(fp))[b]
            u = F["none"]["judge_selected"]
            f = F["fitted"]["judge_selected"] - u
            if f < FITTED_MIN:
                continue
            n_cls += 1
            n_ok += all(F[k]["judge_selected"] - u >= COMPETITIVE * f for k in ("ours_s0", "eye_s0"))
        out[f"Judge{b.title()}U"], out[f"Judge{b.title()}N"] = str(n_ok), str(n_cls)
    # keyword-positive hedging completions, as the judge sees them (all judged cells, all models)
    from analyze import load as load_cells
    hc = hi = nh = 0
    for t, _, _ in MODELS:
        fp = pathlib.Path("results") / t / "judge_haiku_hedge.jsonl"
        if not fp.exists():
            continue
        cells, _ = load_cells(t, "hedge")
        for line in open(fp):
            r = json.loads(line)
            if "error" in r:
                continue
            sc, deg = cells[r["dir"]][r["alpha"]]
            if deg[r["p"]] or sc[r["p"]] <= 0.5:
                continue
            if r["hedged"] and r["coherent"]:
                hc += 1
            elif r["hedged"]:
                hi += 1
            else:
                nh += 1
    n = max(hc + hi + nh, 1)
    out["KwHedgeN"] = f"{n:,}".replace(",", "{,}")
    out["KwHedgeCoh"], out["KwHedgeIncoh"], out["KwHedgeNot"] = (f"{round(100 * x / n)}\\%" for x in (hc, hi, nh))
    (OUT / "numbers.tex").write_text("".join(rf"\newcommand{{\F{k}}}{{{v}}}" + "\n" for k, v in out.items()))
    return out, {k: len(v) for k, v in cases.items()}


def pack_tables():
    """Wrap each table fragment as a macro (\\input cannot sit next to \\midrule inside a tabular)."""
    names = {"tab_cases_head": "FTabCasesHead", "tab_cases": "FTabCases", "tab_q17": "FTabQ", "tab_judge": "FTabJudge",
             "tab_cos": "FTabCos", "tab_mix": "FTabMix"}
    out = []
    for f, m in names.items():
        fp = OUT / f"{f}.tex"
        body = fp.read_text().rstrip("\n") if fp.exists() else ""
        out.append(rf"\newcommand{{\{m}}}{{%" + "\n" + body + "}")
    (OUT / "tables.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    fig_models()
    depth_models = fig_depth()
    tab_q17()
    tab_cases()
    mix_models = tab_mix()
    tab_cos()
    nj = tab_judge()
    nums, cases = numbers()
    pack_tables()
    print("models:", [m[1] for m in MODELS])
    print("depth sweep:", depth_models, "\nmixtures:", mix_models)
    print(nums, cases)
    for t, name, _ in MODELS:
        print(f"{name:<22}", " ".join(f"{SHORT[b]}={case(DATA[t][0][b]) if b in DATA[t][0] else ' '}" for b in ORDER))
