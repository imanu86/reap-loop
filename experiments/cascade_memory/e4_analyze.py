"""E4 analysis: does the LEARNED trigger beat the ABSTENTION-ONLY trigger?

Both are B2_reactive (2-tier: small window -> one direct RAG on trigger). The trigger is
`abstained OR mean_logprob < theta`:
  * theta = -2.0        -> logprob gate never fires -> ABSTENTION-ONLY (T_abst, baseline point)
  * theta near 0 (-0.02..-0.001) -> also recovers confident-wrong non-abstainers (T_learned curve)

Amortized cost = mean total_tokens over ALL items (denominator N_total; recency-skewed
mixture, so most items are in-window and never recover). Pre-registered GO (PREREGISTRATION
E4): T_learned's frontier DOMINATES the T_abst point -> higher accuracy at amortized cost
<= 1.15x T_abst's, with paired-bootstrap CI95 on Delta-accuracy excluding 0.
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for pat in ([p] if os.path.isfile(p) else glob.glob(p)):
            with open(pat, encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return [r for r in rows if r.get("has_plant", True) and r.get("error") is None]


def by_theta(rows):
    g = {}
    for r in rows:
        if r["arm"] != "B2_reactive":
            continue
        g.setdefault(r.get("theta"), {})[r["item_id"]] = (
            int(r["correct"]), r["total_tokens"], r.get("rung_stop"))
    return g


def acc_cost_rec(d, ids=None):
    ids = ids or list(d)
    n = len(ids)
    acc = sum(d[i][0] for i in ids) / n
    cost = sum(d[i][1] for i in ids) / n
    rec = sum(1 for i in ids if d[i][2] not in (None, "native")) / n
    return acc, cost, rec


def paired_delta_acc(d_theta, d_base, n_boot=2000, seed=0):
    common = sorted(set(d_theta) & set(d_base))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(common))
    base = np.array([d_base[i][0] for i in common])
    cur = np.array([d_theta[i][0] for i in common])
    point = cur.mean() - base.mean()
    deltas = [ (cur[b].mean() - base[b].mean())
               for b in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n_boot)) ]
    return point, (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    rows = load(a.runs)
    g = by_theta(rows)
    b0 = [r for r in rows if r["arm"] == "B0_sw"]
    b1 = [r for r in rows if r["arm"] == "B1_always_rag"]
    if b0:
        n = len(b0); print(f"B0_sw (no recovery, floor)  : acc={sum(r['correct'] for r in b0)/n:.3f}  cost={sum(r['total_tokens'] for r in b0)/n:.1f}")
    if b1:
        n = len(b1); print(f"B1_always_rag (ceiling ref) : acc={sum(r['correct'] for r in b1)/n:.3f}  cost={sum(r['total_tokens'] for r in b1)/n:.1f}")

    thetas = sorted(g, key=lambda t: (t is None, t))
    t_abst = min(thetas)  # most negative = abstention-only
    a0, c0, r0 = acc_cost_rec(g[t_abst])
    print(f"\nT_abst = B2 @ theta={t_abst} (abstention-only): acc={a0:.3f} cost={c0:.1f} recovery={r0:.2f}")
    print("\nB2_reactive frontier (theta -> T_learned):")
    print(f"  {'theta':>7} {'acc':>6} {'cost':>7} {'recov':>6} {'d_acc':>7} {'d_acc_CI95':>18} {'cost_x':>7}")
    best = None
    for t in thetas:
        acc, cost, rec = acc_cost_rec(g[t])
        dacc, ci = paired_delta_acc(g[t], g[t_abst])
        cost_x = cost / c0
        flag = ""
        if t != t_abst and cost_x <= 1.15 and ci[0] > 0:
            flag = " <= GO-candidate"
            if best is None or acc > best[1]:
                best = (t, acc, cost, dacc, ci, cost_x)
        print(f"  {str(t):>7} {acc:>6.3f} {cost:>7.1f} {rec:>6.2f} {dacc:>+7.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] {cost_x:>6.2f}x{flag}")

    print("\n" + "=" * 60)
    if best:
        t, acc, cost, dacc, ci, cost_x = best
        print(f"VERDICT: GO - learned trigger dominates at theta={t}: "
              f"+{dacc:.3f} acc (CI95 [{ci[0]:+.3f},{ci[1]:+.3f}]) at {cost_x:.2f}x cost")
    else:
        print("VERDICT: NEGATIVE - no theta gives a significant accuracy gain within 1.15x "
              "amortized cost. The +0.029 AUROC edge (F3) does not convert to a frontier gain.")


if __name__ == "__main__":
    main()
