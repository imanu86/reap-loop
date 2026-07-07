"""Gate #2 analysis (post-cascade): Pareto frontier + adversarial controls.

Metrics (spec §1, KPI-1/KPI-4), denominator ALWAYS N_total plant items:
  * Per (arm, theta): accuracy_total = Sigma correct / N_total, mean cost (total_tokens),
    cost-vector means, and the early-exit rung distribution.
  * Frontier per arm: cascade arms sweep theta -> a curve; B0_sw/B1 are single points.
  * KPI-1 headline: Delta_cost @ matched-accuracy = cost of the cascade interpolated to a
    baseline's accuracy, minus the baseline cost. NEGATIVE = cascade cheaper at equal
    accuracy = good. Paired item bootstrap for CI95 (same items resampled across thetas).
  * Adversarial: B4 must beat BOTH B1 (always-RAG) and B3 (RANDOM order, equal budget).
    If B4 can't beat B3, the gain was the budget, not the cheap-first gate.
  * KPI-4: rung-0 share of successfully-recovered items; B5 (no rung-0) frontier vs B4.
  * stop-too-early: P(early stop & high confidence & wrong).

Pre-registered Gate #2 (PREREGISTRATION.md): GO-novelty only if (i) B4 dominates B1 AND
B3 with Delta_cost@matched-acc < 0, CI95 excludes 0; (ii) rung-0 resolves >=10% of
recovered items AND B5 is significantly worse; (iii) stop-too-early < 5%.
"""
from __future__ import annotations
import os
import glob
import json
import argparse
import numpy as np


def load_rows(paths):
    rows = []
    for p in paths:
        for pat in ([p] if os.path.isfile(p) else glob.glob(p)):
            with open(pat, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        rows.append(json.loads(l))
    return rows


def _task(rows):
    return [r for r in rows if r.get("has_plant", True) and r.get("error") is None
            and r.get("confidence") is not None]


def group_points(rows):
    """-> {arm: {theta: {item_id: (correct, total_tokens, rung_stop, confidence)}}}"""
    g = {}
    for r in _task(rows):
        arm = r["arm"]; th = r.get("theta")
        g.setdefault(arm, {}).setdefault(th, {})[r["item_id"]] = (
            int(r["correct"]), r["total_tokens"], r.get("rung_stop"), r["confidence"])
    return g


def _acc_cost(d, ids=None):
    ids = ids if ids is not None else list(d)
    n = len(ids)
    if not n:
        return float("nan"), float("nan")
    acc = sum(d[i][0] for i in ids) / n
    cost = sum(d[i][1] for i in ids) / n
    return acc, cost


def _frontier(arm_groups):
    """list of (theta, acc, cost) sorted by acc ascending (for interpolation)."""
    pts = []
    for th, d in arm_groups.items():
        acc, cost = _acc_cost(d)
        pts.append((th, acc, cost))
    pts.sort(key=lambda x: x[1])
    return pts


def _interp_cost_at_acc(pts, target_acc):
    """Linear-interpolate cascade cost at target_acc along its (acc,cost) curve."""
    accs = [p[1] for p in pts]
    costs = [p[2] for p in pts]
    if target_acc <= min(accs):
        return costs[accs.index(min(accs))], "at/below range"
    if target_acc >= max(accs):
        return costs[accs.index(max(accs))], "above range (cannot match acc)"
    for i in range(1, len(pts)):
        a0, a1 = accs[i - 1], accs[i]
        if a0 <= target_acc <= a1 and a1 > a0:
            c0, c1 = costs[i - 1], costs[i]
            frac = (target_acc - a0) / (a1 - a0)
            return c0 + frac * (c1 - c0), "interp"
    return float("nan"), "no bracket"


def delta_cost_matched(casc_groups, base_d, n_boot=1000, seed=0):
    """Delta = cascade_cost@base_acc - base_cost. Paired item bootstrap."""
    base_acc, base_cost = _acc_cost(base_d)
    pts = _frontier(casc_groups)
    point, flag = _interp_cost_at_acc(pts, base_acc)
    delta = point - base_cost
    # bootstrap over the common item set
    common = set(base_d)
    for d in casc_groups.values():
        common &= set(d)
    common = sorted(common)
    rng = np.random.default_rng(seed)
    deltas = []
    if common:
        idx = np.arange(len(common))
        for _ in range(n_boot):
            b = [common[i] for i in rng.choice(idx, size=len(idx), replace=True)]
            ba, bc = _acc_cost(base_d, b)
            bpts = sorted([(th, *_acc_cost(d, b)) for th, d in casc_groups.items()],
                          key=lambda x: x[1])
            cc, _ = _interp_cost_at_acc(bpts, ba)
            deltas.append(cc - bc)
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))) if deltas else (float("nan"),) * 2
    return {"base_acc": base_acc, "base_cost": base_cost, "delta": delta,
            "delta_ci95": ci, "flag": flag}


def rung_distribution(d):
    from collections import Counter
    c = Counter(v[2] for v in d.values())
    n = sum(c.values())
    return {k: c[k] / n for k in sorted(c, key=lambda x: str(x))}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--op-theta", type=float, default=None,
                    help="operating theta for rung/stop-early stats (default: median swept)")
    a = ap.parse_args()
    rows = load_rows(a.runs)
    g = group_points(rows)
    arms = sorted(g)
    print(f"loaded {len(rows)} rows; arms: {arms}")
    print("=" * 78)

    # frontier tables
    for arm in arms:
        print(f"\n{arm} frontier:")
        print(f"  {'theta':>7} {'N':>5} {'acc':>6} {'tot_tok':>8}  rung_dist")
        for th, d in sorted(g[arm].items(), key=lambda kv: (kv[0] is None, kv[0])):
            acc, cost = _acc_cost(d)
            rd, n = rung_distribution(d)
            rds = " ".join(f"{k}:{v:.2f}" for k, v in rd.items())
            print(f"  {str(th):>7} {n:>5} {acc:>6.3f} {cost:>8.1f}  {rds}")

    # KPI-1: Delta_cost @ matched accuracy vs B1 and B3
    print("\n" + "=" * 78)
    print("KPI-1  Delta_cost @ matched-accuracy (NEGATIVE = cascade cheaper; good)")
    b1 = g.get("B1_always_rag", {}).get(None)
    b3 = g.get("B3_random", {})
    for casc in ("B4_cascade", "B5_cascade_no_rung0"):
        if casc not in g:
            continue
        if b1:
            r = delta_cost_matched(g[casc], b1)
            print(f"  {casc} vs B1(always-RAG): base_acc={r['base_acc']:.3f} "
                  f"base_cost={r['base_cost']:.1f}  Delta={r['delta']:.1f} "
                  f"CI95=[{r['delta_ci95'][0]:.1f},{r['delta_ci95'][1]:.1f}] ({r['flag']})")
    if "B4_cascade" in g and b3:
        # compare B4 frontier to B3 frontier at B3's best-accuracy point
        b3_best = max((_acc_cost(d)[0], _acc_cost(d)[1], th) for th, d in b3.items())
        r = delta_cost_matched(g["B4_cascade"], b3[b3_best[2]])
        print(f"  B4_cascade vs B3(random,equal-budget) @theta={b3_best[2]}: "
              f"base_acc={r['base_acc']:.3f} Delta={r['delta']:.1f} "
              f"CI95=[{r['delta_ci95'][0]:.1f},{r['delta_ci95'][1]:.1f}]")

    # KPI-4: rung-0 contribution (at operating theta)
    print("\n" + "=" * 78)
    print("KPI-4  rung-0 contribution (B4_cascade)")
    if "B4_cascade" in g:
        thetas = sorted(g["B4_cascade"])
        op = a.op_theta if a.op_theta is not None else thetas[len(thetas) // 2]
        d = g["B4_cascade"].get(op) or g["B4_cascade"][thetas[len(thetas) // 2]]
        recovered = [v for v in d.values() if v[0] == 1 and v[2] != "native"]
        via0 = [v for v in recovered if v[2] == "0"]
        share = (len(via0) / len(recovered)) if recovered else float("nan")
        # stop-too-early: native accepted (conf>=theta => "high confidence") yet WRONG
        stop_early = [v for v in d.values() if v[2] == "native" and v[0] == 0]
        print(f"  operating theta={op}: recovered(correct via escalation)={len(recovered)}, "
              f"rung-0 share={share:.3f}  (pre-reg: >=0.10)")
        print(f"  stop-too-early (accepted native but WRONG) = "
              f"{len(stop_early)/len(d):.3f}  (pre-reg: <0.05)")


if __name__ == "__main__":
    main()
