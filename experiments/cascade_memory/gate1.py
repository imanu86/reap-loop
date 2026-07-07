"""Gate #1 analysis (pre-cascata): sensor separability + baseline cost/accuracy.

Primary decision metric: AUROC(confidence, correctness) on the baseline arm.
  GO   if AUROC >= 0.65
  NO-GO if AUROC <  0.60   -> STOP, report the honest negative (the sensor does
                              not discriminate, so an early-exit gate would be noise)
  GREY  in [0.60, 0.65)    -> inconclusive, needs more data / better sensor

All averages use the FIXED denominator N_total (never N_resolved). Leak-control
items (has_plant=False) are excluded from the task metrics and reported separately;
their accuracy must be ~chance (near 0) or leakage is present.
"""
from __future__ import annotations
import os
import glob
import json
import argparse
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def load_rows(paths):
    rows = []
    for p in paths:
        for pat in ([p] if os.path.isfile(p) else glob.glob(p)):
            with open(pat, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        rows.append(json.loads(l))
    return rows


def bootstrap_auroc(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb, sb = y[b], s[b]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, sb))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def auroc_report(rows, arm):
    task = [r for r in rows if r.get("has_plant", True) and r.get("arm") == arm]
    valid = [r for r in task if r.get("confidence") is not None
             and r.get("sensor_used") == "logprob"]
    fallback_n = sum(1 for r in task if r.get("sensor_used") != "logprob")
    y = [1 if r["correct"] else 0 for r in valid]
    s = [r["confidence"] for r in valid]
    out = {"arm": arm, "n_task": len(task), "n_valid_logprob": len(valid),
           "n_fallback_sensor": fallback_n,
           "acc": (sum(r["correct"] for r in task) / len(task)) if task else float("nan"),
           "abst": (sum(r["abstained"] for r in task) / len(task)) if task else float("nan")}
    if roc_auc_score is None:
        out["auroc"] = None; out["note"] = "sklearn missing"
        return out
    if len(set(y)) < 2:
        out["auroc"] = float("nan")
        out["note"] = "degenerate: only one correctness class (all right or all wrong)"
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auroc_ci95"] = bootstrap_auroc(y, s)
    # stop-too-early proxy: high-confidence-but-wrong mass (threshold = median conf)
    med = float(np.median(s))
    hi_wrong = sum(1 for r in valid if r["confidence"] > med and not r["correct"])
    out["p_highconf_wrong"] = hi_wrong / len(valid)
    return out


def frontier_table(rows, arm):
    task = [r for r in rows if r.get("has_plant", True) and r.get("arm") == arm]
    dists = sorted(set(r["distance"] for r in task))
    tbl = []
    for d in dists:
        sub = [r for r in task if r["distance"] == d]
        n = len(sub)
        tbl.append({
            "distance": d, "n": n,
            "acc": sum(r["correct"] for r in sub) / n,
            "abst": sum(r["abstained"] for r in sub) / n,
            "mean_prefill": sum(r["prefill_tokens"] for r in sub) / n,
            "mean_decode": sum(r["decode_tokens"] for r in sub) / n,
            "mean_total_tokens": sum(r["total_tokens"] for r in sub) / n,
            "mean_rag_calls": sum(r["n_rag_calls"] for r in sub) / n,
        })
    return tbl


def leak_control(rows, arm):
    ctl = [r for r in rows if not r.get("has_plant", True) and r.get("arm") == arm]
    if not ctl:
        return None
    return {"n": len(ctl),
            "acc": sum(r["correct"] for r in ctl) / len(ctl),
            "abst": sum(r["abstained"] for r in ctl) / len(ctl)}


def verdict(auroc):
    if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)):
        return "INCONCLUSIVE (no AUROC)"
    if auroc >= 0.65:
        return "GO"
    if auroc < 0.60:
        return "NO-GO (STOP: sensor does not discriminate)"
    return "GREY (0.60-0.65: inconclusive)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="JSONL file(s) or globs")
    ap.add_argument("--baseline-arm", default="B0_no_memory",
                    help="arm whose AUROC drives Gate #1")
    a = ap.parse_args()
    rows = load_rows(a.runs)
    arms_present = sorted(set(r["arm"] for r in rows))
    print(f"loaded {len(rows)} rows; arms present: {arms_present}")
    print("=" * 68)

    base = auroc_report(rows, a.baseline_arm)
    print(f"GATE #1 - sensor separability on baseline arm '{a.baseline_arm}'")
    print(f"  N_task={base['n_task']}  acc={base['acc']:.3f}  abst={base['abst']:.3f}")
    print(f"  sensor: logprob n={base['n_valid_logprob']}  fallback n={base['n_fallback_sensor']}")
    au = base.get("auroc")
    if isinstance(au, float) and not np.isnan(au):
        ci = base.get("auroc_ci95")
        print(f"  AUROC(confidence,correct) = {au:.3f}  CI95={('[%.3f, %.3f]' % ci) if ci else 'n/a'}")
        print(f"  P(high-conf & wrong) = {base['p_highconf_wrong']:.3f}")
    else:
        print(f"  AUROC = {au}  ({base.get('note','')})")
    print(f"  >>> VERDICT: {verdict(au)}")
    print("=" * 68)

    for arm in arms_present:
        print(f"\nBaseline frontier - {arm} (per turn-distance):")
        print(f"  {'dist':>5} {'n':>5} {'acc':>6} {'abst':>6} "
              f"{'prefill':>9} {'decode':>7} {'tot_tok':>8} {'rag':>5}")
        for r in frontier_table(rows, arm):
            print(f"  {r['distance']:>5} {r['n']:>5} {r['acc']:>6.3f} {r['abst']:>6.3f} "
                  f"{r['mean_prefill']:>9.1f} {r['mean_decode']:>7.1f} "
                  f"{r['mean_total_tokens']:>8.1f} {r['mean_rag_calls']:>5.2f}")
        lc = leak_control(rows, arm)
        if lc:
            flag = "OK" if lc["acc"] <= 0.05 else "LEAK?"
            print(f"  leak-control (no plant): n={lc['n']} acc={lc['acc']:.3f} [{flag}]")


if __name__ == "__main__":
    main()
