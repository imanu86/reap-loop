"""paper_ci.py — bootstrap 95% CIs for every key ratio already on disk (GAPS M4).

Zero-GPU. Reads existing result files and emits percentile bootstrap 95% CIs
for the comparative ratios the paper reports as naked point estimates:

  (a) REAP paired per-chunk perplexity ratios vs full/dom (reap, rand) and
      rand/gen, from runs/reap/2026-07-05_eval_biasmask/results_raw.csv.
      Statistic: geometric mean of per-chunk ppl ratios (== exp(mean log-ratio)),
      paired bootstrap over CHUNKS. n is tiny (dom n=4, gen n=2) — reported.
  (b) DSpark spec2-vs-base decode wall-time ratio, from
      runs/dspark/20260705_fase_c_smoke_3060/out/runtimes.csv. Matched protocol
      (NTOK=100, draft=2). base={186,181}, spec2={168,166}. Bootstrap over the
      cartesian product of the two arms (ratio = spec2_wall/base_wall). n per
      arm = 2 — reported as [CI: dati insufficienti] where it applies.
  (c) Fase C spec2 acceptance 56/66 (per-cycle Bernoulli) — Wilson 95% CI, and
      a per-cycle bootstrap for cross-check.

Method: percentile bootstrap, B=10000 resamples, numpy seed 42. For a ratio of
means (wall-time) we resample each arm independently; for paired per-chunk
ratios we resample the chunk index with replacement (paired). Where the number
of independent replicates is < 3 the CI is emitted but explicitly flagged
insufficient, per the house rule "no invented numbers".

Usage: python scripts/paper_ci.py
Output: runs/paper_ci_results.json
"""
import json
import math
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10000
SEED = 42
PCT = (2.5, 97.5)

ROW = re.compile(
    r"^(?P<cfg>\w+),(?P<chunk>[\w]+),rc=(?P<rc>\d+),tokens=(?P<tokens>\d+) "
    r"scored=(?P<scored>\d+) nll=(?P<nll>[\d.]+) avg_nll=(?P<avg>[\d.]+) ppl=(?P<ppl>[\d.]+)")


def fmt(lo, hi):
    return [round(float(lo), 4), round(float(hi), 4)]


def crosses_one(lo, hi):
    return lo <= 1.0 <= hi


# ---------------------------------------------------------------- (a) REAP
def load_reap(path):
    """Return {cfg: {chunk: {'avg_nll':.., 'scored':..}}}."""
    rows = {}
    for line in open(path):
        m = ROW.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        assert d["rc"] == "0", f"run fallita: {line}"
        rows.setdefault(d["cfg"], {})[d["chunk"]] = {
            "avg_nll": float(d["avg"]), "scored": int(d["scored"])}
    return rows


def reap_ratio_ci(rows, cfg, corpus, rng):
    """Paired bootstrap over chunks of the geomean per-chunk ppl ratio cfg/full.

    Per-chunk log-ratio = avg_nll[cfg,chunk] - avg_nll[full,chunk].
    Point statistic = exp(mean(log-ratio))  (geomean of ppl ratios).
    """
    chunks = sorted(k for k in rows[cfg] if k.startswith(corpus))
    log_ratios = np.array([rows[cfg][k]["avg_nll"] - rows["full"][k]["avg_nll"]
                           for k in chunks])
    per_chunk_ppl_ratio = [round(float(math.exp(x)), 4) for x in log_ratios]
    point = float(np.exp(log_ratios.mean()))
    n = len(chunks)
    if n < 3:
        # bootstrap still computed but flagged insufficient
        idx = rng.integers(0, n, size=(B, n))
        boot = np.exp(log_ratios[idx].mean(axis=1))
        lo, hi = np.percentile(boot, PCT)
        return {
            "point_geomean_ppl_ratio": round(point, 4),
            "per_chunk_ppl_ratio": per_chunk_ppl_ratio,
            "n_chunks": n,
            "ci95": fmt(lo, hi),
            "crosses_1.0": bool(crosses_one(lo, hi)),
            "note": f"[CI: dati insufficienti, n={n} chunk] — CI mostrato ma non affidabile",
        }
    idx = rng.integers(0, n, size=(B, n))
    boot = np.exp(log_ratios[idx].mean(axis=1))
    lo, hi = np.percentile(boot, PCT)
    return {
        "point_geomean_ppl_ratio": round(point, 4),
        "per_chunk_ppl_ratio": per_chunk_ppl_ratio,
        "n_chunks": n,
        "ci95": fmt(lo, hi),
        "crosses_1.0": bool(crosses_one(lo, hi)),
    }


# --------------------------------------------------- (b) DSpark wall ratio
def load_runtimes(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("run,"):
            continue
        name, rc, wall = line.split(",")
        out[name] = {"rc": int(rc), "wall_s": float(wall)}
    return out


def wall_ratio_ci(base_arm, spec_arm, rng):
    """Bootstrap ratio spec/base over the cartesian product of the two arms.

    Resample each arm independently (independent runs), ratio = mean(spec)/mean(base)
    on each resample. Arms are tiny (n=2 each) → flagged.
    """
    base = np.array(base_arm, dtype=float)
    spec = np.array(spec_arm, dtype=float)
    point = float(spec.mean() / base.mean())
    nb, ns = len(base), len(spec)
    bi = rng.integers(0, nb, size=(B, nb))
    si = rng.integers(0, ns, size=(B, ns))
    boot = spec[si].mean(axis=1) / base[bi].mean(axis=1)
    lo, hi = np.percentile(boot, PCT)
    r = {
        "point_ratio_spec_over_base": round(point, 4),
        "base_wall_s": base_arm,
        "spec_wall_s": spec_arm,
        "n_base": nb, "n_spec": ns,
        "ci95": fmt(lo, hi),
        "crosses_1.0": bool(crosses_one(lo, hi)),
    }
    if nb < 3 or ns < 3:
        r["note"] = (f"[CI: dati insufficienti, n_base={nb} n_spec={ns}] — "
                     "CI mostrato ma le braccia hanno <3 repliche")
    return r


# ------------------------------------------------- (c) acceptance Wilson CI
def wilson_ci(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - half, centre + half


def accept_ci(k, n, rng):
    lo_w, hi_w = wilson_ci(k, n)
    draws = rng.binomial(n, k / n, size=B) / n  # parametric bootstrap check
    lo_b, hi_b = np.percentile(draws, PCT)
    return {
        "k": k, "n": n, "point": round(k / n, 4),
        "wilson_ci95": fmt(lo_w, hi_w),
        "bootstrap_ci95": fmt(lo_b, hi_b),
    }


def main():
    rng = np.random.default_rng(SEED)
    out = {
        "_meta": {
            "script": "scripts/paper_ci.py",
            "bootstrap_resamples": B,
            "seed": SEED,
            "ci": "percentile 95% (2.5–97.5)",
            "note": ("Bootstrap CIs on ratios already on disk (GAPS M4). Where the "
                     "number of independent replicates is <3 the CI is flagged "
                     "'dati insufficienti'. A CI that crosses 1.0 means the "
                     "comparison is not significant at 95%."),
        },
        "a_reap_perplexity_ratios": {},
        "b_dspark_wall_ratio": {},
        "c_acceptance": {},
    }

    # (a) REAP
    reap_raw = os.path.join(REPO, "runs/reap/2026-07-05_eval_biasmask/results_raw.csv")
    rows = load_reap(reap_raw)
    out["a_reap_perplexity_ratios"]["_source"] = \
        "runs/reap/2026-07-05_eval_biasmask/results_raw.csv"
    out["a_reap_perplexity_ratios"]["reap_over_full_dom"] = \
        reap_ratio_ci(rows, "reap", "dom", rng)
    out["a_reap_perplexity_ratios"]["rand_over_full_dom"] = \
        reap_ratio_ci(rows, "rand", "dom", rng)
    out["a_reap_perplexity_ratios"]["rand_over_full_gen"] = \
        reap_ratio_ci(rows, "rand", "gen", rng)

    # (b) DSpark wall ratio (matched protocol NTOK=100, draft=2)
    rt = load_runtimes(
        os.path.join(REPO, "runs/dspark/20260705_fase_c_smoke_3060/out/runtimes.csv"))
    base_arm = [rt["base_r1"]["wall_s"], rt["base_r2"]["wall_s"]]
    spec_arm = [rt["spec2_r1"]["wall_s"], rt["spec2_r2"]["wall_s"]]
    out["b_dspark_wall_ratio"]["_source"] = \
        "runs/dspark/20260705_fase_c_smoke_3060/out/runtimes.csv"
    out["b_dspark_wall_ratio"]["_note"] = (
        "Matched protocol: NTOK=100, draft=2. base_r1/r2 vs spec2_r1/r2. "
        "spec2_verbose excluded (n=30 tokens), spec4 excluded (draft=4), "
        "spec2_m3 excluded (margin-skip variant). speedup = 1/ratio.")
    out["b_dspark_wall_ratio"]["spec2_over_base"] = \
        wall_ratio_ci(base_arm, spec_arm, rng)

    # (c) Fase C spec2 acceptance 56/66 (per confidence cycle)
    out["c_acceptance"]["_source"] = \
        "runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log (56 committed / 66 cycles)"
    out["c_acceptance"]["fase_c_spec2"] = accept_ci(56, 66, rng)

    dst = os.path.join(REPO, "runs/paper_ci_results.json")
    json.dump(out, open(dst, "w"), indent=2)

    # human-readable summary
    def line(name, d):
        flag = "  ** CROSSES 1.0 (not significant) **" if d.get("crosses_1.0") else ""
        ins = "  [n<3: dati insufficienti]" if "note" in d else ""
        pt = d.get("point_geomean_ppl_ratio", d.get("point_ratio_spec_over_base"))
        print(f"  {name}: {pt}x  CI95 {d['ci95']}{flag}{ins}")

    print(f"B={B} seed={SEED}\n(a) REAP perplexity ratios (geomean, paired bootstrap over chunks):")
    line("reap/dom", out["a_reap_perplexity_ratios"]["reap_over_full_dom"])
    line("rand/dom", out["a_reap_perplexity_ratios"]["rand_over_full_dom"])
    line("rand/gen", out["a_reap_perplexity_ratios"]["rand_over_full_gen"])
    print("(b) DSpark spec2/base wall-time ratio:")
    line("spec2/base", out["b_dspark_wall_ratio"]["spec2_over_base"])
    print("(c) Fase C spec2 acceptance:")
    c = out["c_acceptance"]["fase_c_spec2"]
    print(f"  56/66 = {c['point']}  Wilson95 {c['wilson_ci95']}  boot95 {c['bootstrap_ci95']}")
    print("scritto", dst)


if __name__ == "__main__":
    main()
