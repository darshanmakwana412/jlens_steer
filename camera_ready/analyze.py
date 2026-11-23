"""Score stored completions: quality gate, in-sample peak (submission protocol), two-fold cross-fitted rate with
bootstrap CIs, and paired differences against the fitted baseline.

uv run python analyze.py q17
"""
import collections
import json
import pathlib
import sys

import numpy as np

from jspace.behaviours import BEHAVIOURS, ORDER, is_degenerate

GATE = 0.30
N_BOOT = 4000
rng = np.random.default_rng(0)


def load(tag, b):
    """-> {dir: {alpha: (scores[n], degen[n])}} plus the unsteered row."""
    test = BEHAVIOURS[b]["test"]
    allow = ("LATIN",)
    cells = collections.defaultdict(dict)
    rows = collections.defaultdict(list)
    for fp in sorted((pathlib.Path("results") / tag).glob(f"gen_{b}*.jsonl")):
        for line in open(fp):
            r = json.loads(line)
            rows[(r["dir"], r["alpha"])].append(r)
    texts = {}
    for (d, a), rs in rows.items():
        rs.sort(key=lambda r: r["p"])
        deg = np.array([is_degenerate(r["text"], allow) for r in rs])
        sc = np.array([0.0 if g else test(r["text"]) for r, g in zip(rs, deg)])
        cells[d][a] = (sc, deg)
        texts[(d, a)] = [r["text"] for r in rs]
    return cells, texts


def select(cells_d, idx):
    """Best alpha on prompt subset idx among alphas whose degenerate share on idx is <= GATE (ties -> smaller)."""
    best, best_rate = None, -1.0
    for a in sorted(cells_d):
        sc, deg = cells_d[a]
        if deg[idx].mean() > GATE:
            continue
        r = sc[idx].mean()
        if r > best_rate + 1e-12:
            best, best_rate = a, r
    return best


def crossfit(cells_d, n):
    """Per-prompt scores where each fold is scored at the alpha selected on the other fold."""
    half = n // 2
    A, B = np.arange(half), np.arange(half, n)
    out = np.zeros(n)
    chosen = {}
    for sel, rep, name in ((A, B, "A->B"), (B, A, "B->A")):
        a = select(cells_d, sel)
        chosen[name] = a
        if a is not None:
            out[rep] = cells_d[a][0][rep]
    return out, chosen


def insample(cells_d, n):
    a = select(cells_d, np.arange(n))
    return (cells_d[a][0].mean() if a is not None else 0.0), a


def boot_ci(x):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ci_seeds(P):
    """P: seeds x prompts. Resample prompts and seeds jointly, so seed variance is in the interval."""
    P = np.asarray(P)
    k, n = P.shape
    pi = rng.integers(0, n, size=(N_BOOT, n))
    si = rng.integers(0, k, size=(N_BOOT, k))
    m = np.array([P[si[b]][:, pi[b]].mean() for b in range(N_BOOT)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def method_scores(cells, prefix, n):
    """Seed-averaged cross-fitted per-prompt scores for a method with seeds (prefix_s0..), or a single dir."""
    names = [d for d in cells if d == prefix or (d.startswith(prefix + "_s") and "@" not in d
                                                 and d[len(prefix) + 2:].isdigit())]
    per_seed, ins, chosen = [], [], []
    for d in sorted(names):
        x, ch = crossfit(cells[d], n)
        per_seed.append(x)
        ins.append(insample(cells[d], n)[0])
        chosen.append(ch)
    if not per_seed:
        return None
    X = np.mean(per_seed, 0)
    ci_prompt = boot_ci(X)
    ci = boot_ci_seeds(per_seed) if len(per_seed) > 1 else ci_prompt
    return dict(per_prompt=X, mean=float(X.mean()), ci=ci, ci_prompt=ci_prompt, seed_means=[float(s.mean()) for s in per_seed],
                insample=float(np.mean(ins)), chosen=chosen)


def analyze(tag):
    meta = json.load(open(pathlib.Path("results") / tag / "meta.json"))
    res = {}
    for b in ORDER:
        if not list((pathlib.Path("results") / tag).glob(f"gen_{b}*.jsonl")):
            continue
        cells, texts = load(tag, b)
        n = len(next(iter(cells["fitted"].values()))[0])
        R = {"n_prompts": n, "unsteered": float(cells["none"][0.0][0].mean())}
        prefixes = sorted({d.rsplit("_s", 1)[0] if d.rsplit("_s", 1)[-1].isdigit() else d for d in cells} - {"none"})
        for pre in prefixes:
            m = method_scores(cells, pre, n)
            if m is not None:
                R[pre] = m
        # paired differences vs fitted
        for pre in list(R):
            if isinstance(R[pre], dict) and pre != "fitted" and "per_prompt" in R[pre]:
                diff = R[pre]["per_prompt"] - R["fitted"]["per_prompt"]
                R[pre]["diff_vs_fitted"] = float(diff.mean())
                R[pre]["diff_ci"] = boot_ci(diff)
        res[b] = R
    return meta, res


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "per_prompt"}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    return o


if __name__ == "__main__":
    tag = sys.argv[1]
    meta, res = analyze(tag)
    json.dump(strip(res), open(pathlib.Path("results") / tag / "summary.json", "w"), indent=1)
    show = ["fitted", "ours", "ridge", "notop", "eye", "rand", "oracle", "ours_jpp", "ridge_jpp", "oracle_jpp", "artifact", "oracle_artifact"]
    print(f"{'behaviour':<9} {'unst':>5} " + " ".join(f"{s:>16}" for s in show))
    for b, R in res.items():
        cells = []
        for s in show:
            if s in R:
                lo, hi = R[s]["ci"]
                cells.append(f"{R[s]['mean']:.2f}[{lo:.2f},{hi:.2f}]")
            else:
                cells.append("-")
        print(f"{b:<9} {R['unsteered']:>5.2f} " + " ".join(f"{c:>16}" for c in cells))
    print("\nin-sample peak (submission protocol) vs cross-fitted:")
    for b, R in res.items():
        print(f"{b:<9} " + "  ".join(f"{s}: {R[s]['insample']:.2f}->{R[s]['mean']:.2f}" for s in ["fitted", "ours", "eye"] if s in R))
    print("\nother directions (cross-fitted mean, diff vs fitted [CI]):")
    for b, R in res.items():
        extra = [k for k in R if isinstance(R[k], dict) and k not in show]
        print(b, "  ".join(f"{k}={R[k]['mean']:.2f}" for k in sorted(extra)))
