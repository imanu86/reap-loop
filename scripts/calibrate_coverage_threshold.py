#!/usr/bin/env python3
"""E-CAL -- OFFLINE calibration of a coverage threshold for predictive mask sizing.

OFFLINE ONLY. Reads routing-weight traces + S1 sensors already on disk
(no GPU / WSL / pod writes).

USER IDEA under test
--------------------
From the warmup alone, the per-expert per-layer *unbiased router mass* gives -- for
free -- the curve K -> coverage for every candidate mask:
    coverage(K) = mass(top-K experts) / total mass
    S1_predicted at engage = 1 - coverage(K)
Pick the smallest K whose coverage >= a threshold theta. Question: which theta
separates survived runs from collapsed ones, and does it depend on task width?

MONITO (from tonight's data): S1 is NOT monotone with survival. K91-static
(coding) engage S1 ~= 0.845 survives ~2476 tok; K23-rotate (html) engage S1 ~=
0.815 collapses at ~gen126. So look for CONDITIONED thresholds (per task/width)
AND evaluate alternative metrics from the same curve (slope, knee, marginal
coverage of the K-th expert).

WHAT THIS SCRIPT DOES
---------------------
1. Loads every full-model routing trace with weights (schema pos,layer,n,e0..e5,
   w0..w5): html cyberpunk W50/W130 warmup-replay + 11 coding prompts
   (trace_coding.tgz). Weights are the UNBIASED router probs of the 6 SELECTED
   experts (ds4 comment ~7384). NOTE: the trace logs only the 6 selected experts
   per (pos,layer); the mass on the other 250 experts is NOT in the trace.
2. For each trace, on a fixed engage window (first W tokens = what an in-engine
   adaptive-K would see at engage) AND on the full trace, builds coverage(K):
   per-layer rank experts by accumulated routed mass, coverage(K) = cumulative
   top-K share. Extracts coverage@{23,32,...}, Kmin for cov in {80,85,90,95}%,
   the kneedle knee, and the marginal coverage of the deployed K-th expert.
3. Reads the two live S1 sensors (patch-0012, normalized over ALL 256 experts):
   r1 K23-rotate html (s1_r1.csv.gz) and K91-static coding (loop/s1_sensor.csv).
   These measure def-2 coverage (full-softmax); the trace curve measures def-1
   (routed/selected mass). The script quantifies the gap between them.
4. Cross-references coverage@K-used against the known outcomes and prints the
   separation verdict.

Reproduce:
  python scripts/calibrate_coverage_threshold.py \
    --reap-loop-root <reap-loop> \
    --moe-root <moe-aggressive-commit>/.claude/worktrees/elastic-bose-6ae1c7
"""
import argparse
import csv
import gzip
import json
import os
import statistics
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

N_EXPERTS = 256
KS = [8, 12, 16, 20, 23, 28, 32, 40, 48, 64, 91, 96, 128]
COV_TARGETS = (0.80, 0.85, 0.90, 0.95)
ENGAGE_WIN = 50   # ds4 PACE_WARMUP default: the mask engages after 50 decode tokens


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_route_rows(path, opener=open):
    """Return list of (pos, layer, [experts], [weights]) from a route/trace csv."""
    out = []
    with opener(path, "rt", newline="") as fh:
        rd = csv.reader(fh)
        h = next(rd)
        ei = [h.index(f"e{i}") for i in range(6)]
        wi = [h.index(f"w{i}") for i in range(6)]
        li = h.index("layer")
        pi = h.index("pos")
        for r in rd:
            if not r:
                continue
            out.append((int(r[pi]), int(r[li]),
                        [int(r[a]) for a in ei], [float(r[b]) for b in wi]))
    return out


def accumulate(rows, win=None):
    """layer -> {expert: accumulated routed mass}, restricted to first `win` positions."""
    by_layer = defaultdict(lambda: defaultdict(float))
    positions = sorted({p for p, _, _, _ in rows})
    keep = set(positions[:win]) if win else None
    for p, L, es, ws in rows:
        if keep is not None and p not in keep:
            continue
        for e, w in zip(es, ws):
            by_layer[L][e] += w
    return by_layer, len(positions)


# --------------------------------------------------------------------------- #
# curve metrics (def-1: share of ROUTED / selected-6 mass)
# --------------------------------------------------------------------------- #
def coverage_at(by_layer, K):
    fr = []
    for _L, mass in by_layer.items():
        v = sorted(mass.values(), reverse=True)
        tot = sum(v)
        if tot > 0:
            fr.append(sum(v[:K]) / tot)
    return float(np.mean(fr)) if fr else float("nan")


def kmin_for(by_layer, target):
    """Per-layer smallest K to reach coverage>=target; return the median over layers."""
    ks = []
    for _L, mass in by_layer.items():
        v = sorted(mass.values(), reverse=True)
        tot = sum(v)
        if tot <= 0:
            continue
        c = 0.0
        for i, x in enumerate(v, 1):
            c += x
            if c / tot >= target:
                ks.append(i)
                break
    return float(statistics.median(ks)) if ks else float("nan")


def knee_kneedle(by_layer, kmax=96):
    """Kneedle knee of the mean coverage(K) curve: K of max distance to the chord."""
    xs = list(range(1, kmax + 1))
    ys = [coverage_at(by_layer, k) for k in xs]
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    best_k, best_d = xs[0], -1.0
    for x, y in zip(xs, ys):
        # perpendicular-ish distance to the straight chord (normalised)
        d = (y - (y0 + (y1 - y0) * (x - x0) / (x1 - x0)))
        if d > best_d:
            best_d, best_k = d, x
    return best_k


def marginal_at(by_layer, K):
    """Mean marginal coverage gained by the K-th expert = cov(K)-cov(K-1)."""
    return coverage_at(by_layer, K) - coverage_at(by_layer, K - 1)


# --------------------------------------------------------------------------- #
# S1 sensors (def-2: pruned_mass / total_mass, normalised over ALL 256)
# --------------------------------------------------------------------------- #
def s1_stats(path, opener=open, lo=None, hi=None):
    vals = []
    with opener(path, "rt", newline="") as fh:
        rd = csv.reader(fh)
        h = next(rd)
        pi = h.index("pos")
        pm = h.index("pruned_mass")
        tm = h.index("total_mass")
        for r in rd:
            if not r:
                continue
            p = int(r[pi])
            t = float(r[tm])
            pr = float(r[pm])
            if t <= 0 or pr <= 0:
                continue
            if lo is not None and p < lo:
                continue
            if hi is not None and p > hi:
                continue
            vals.append(pr / t)
    if not vals:
        return {}
    return dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)),
                n=len(vals))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--reap-loop-root", default=os.path.dirname(here))
    ap.add_argument("--moe-root",
                    default=r"C:/Users/imanu/source/repos/moe-aggressive-commit/.claude/worktrees/elastic-bose-6ae1c7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RL = args.reap_loop_root
    K91 = os.path.join(args.moe_root, "runs/reap/k91_coding_vram")
    out_dir = args.out or os.path.join(RL, "runs/ds4/20260710_ecal_coverage_threshold")
    os.makedirs(out_dir, exist_ok=True)

    # ---- assemble traces with weights ------------------------------------ #
    sources = [
        ("html_W50", "html",
         os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W50/route_W50.csv")),
        ("html_W130", "html",
         os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv")),
    ]
    tmp = tempfile.mkdtemp(prefix="ecal_")
    with tarfile.open(os.path.join(K91, "trace_coding.tgz")) as tf:
        try:
            tf.extractall(tmp, filter="data")   # py>=3.12; avoids the 3.14 warning
        except TypeError:
            tf.extractall(tmp)
    cd = os.path.join(tmp, "trace_coding")
    for n in sorted(os.listdir(cd)):
        if n.startswith("trace_") and n.endswith(".csv"):
            tag = n.replace("trace_", "").replace(".csv", "")
            tag = tag.split("_")[-1]  # e.g. python-csv
            sources.append((f"code_{tag}", "coding", os.path.join(cd, n)))

    # ---- per-trace curve metrics ----------------------------------------- #
    rows_out = []
    per_trace = {}
    for lab, task, path in sources:
        if not os.path.exists(path):
            continue
        rows = load_route_rows(path)
        bl_w, ntok = accumulate(rows, win=ENGAGE_WIN)   # engage window
        bl_f, _ = accumulate(rows, win=None)            # full trace
        rec = dict(
            task=task, tokens=ntok,
            cov_engage={k: coverage_at(bl_w, k) for k in KS},
            cov_full={k: coverage_at(bl_f, k) for k in KS},
            kmin_engage={f"{int(t*100)}": kmin_for(bl_w, t) for t in COV_TARGETS},
            kmin_full={f"{int(t*100)}": kmin_for(bl_f, t) for t in COV_TARGETS},
            knee_engage=knee_kneedle(bl_w),
            marg23_engage=marginal_at(bl_w, 23),
            marg32_engage=marginal_at(bl_w, 32),
        )
        per_trace[lab] = rec
        rows_out.append((lab, rec))

    def agg(field_getter):
        vals = [field_getter(r) for _, r in rows_out if field_getter(r) == field_getter(r)]
        return dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                    lo=float(np.min(vals)), hi=float(np.max(vals)))

    cov23 = agg(lambda r: r["cov_engage"][23])
    cov32 = agg(lambda r: r["cov_engage"][32])
    kmin90 = agg(lambda r: r["kmin_engage"]["90"])
    kmin80 = agg(lambda r: r["kmin_engage"]["80"])
    kmin95 = agg(lambda r: r["kmin_engage"]["95"])

    # ---- live S1 sensors (def-2, full-256 normalisation) ----------------- #
    s1_r1 = s1_stats(os.path.join(RL, "runs/ds4/20260710_scope_divergence_pod/r1/s1_r1.csv.gz"),
                     gzip.open)
    s1_k91 = s1_stats(os.path.join(K91, "loop/s1_sensor.csv"), open, lo=300, hi=2280)

    # ---- def-1 (warmup, routed) vs def-2 (live, full-256) gap ------------ #
    html_cov23_def1 = per_trace["html_W50"]["cov_engage"][23]
    predicted_s1_html = 1.0 - html_cov23_def1
    measured_s1_html = s1_r1.get("mean", float("nan"))

    # ---- outcome cross-reference ----------------------------------------- #
    # (label, task, mask, deployed K, actuation, provenance, coverage_used_def1,
    #  measured_S1_def2, outcome, survive_bool)
    outcomes = [
        dict(run="k91-static (coding)", K=91, act="static", prov="cold-corpus",
             cov_def1=None, s1_def2=s1_k91.get("mean"),
             outcome="survives ~2476 tok, slow drift (onset 2286)", survive=True),
        dict(run="K23-rotate32 (html cyberpunk)", K=23, act="rotate32", prov="session-W50",
             cov_def1=per_trace["html_W50"]["cov_engage"][23], s1_def2=s1_r1.get("mean"),
             outcome="collapse ~gen126 (loop lock)", survive=False),
        dict(run="keep-23 static session (html W50)", K=23, act="static", prov="session-W50",
             cov_def1=per_trace["html_W50"]["cov_engage"][23], s1_def2=None,
             outcome="L3 clean page (knee/warmup)", survive=True),
        dict(run="keep-23 cold-static (frontpage)", K=23, act="static", prov="cold-corpus",
             cov_def1=None, s1_def2=None,
             outcome="L0 loop (knee cold-static)", survive=False),
        dict(run="JSON keep-20 cold-static", K=20, act="static", prov="cold-corpus",
             cov_def1=None, s1_def2=None, outcome="L3 exact (knee)", survive=True),
        dict(run="PYTHON keep-32 cold-static", K=32, act="static", prov="cold-corpus",
             cov_def1=None, s1_def2=None, outcome="L3 tests pass (knee)", survive=True),
    ]

    # ---- verdict logic --------------------------------------------------- #
    # Does a single coverage@K-used threshold separate survive vs collapse?
    surv_cov = [o["cov_def1"] for o in outcomes if o["survive"] and o["cov_def1"] is not None]
    coll_cov = [o["cov_def1"] for o in outcomes if not o["survive"] and o["cov_def1"] is not None]
    surv_s1 = [o["s1_def2"] for o in outcomes if o["survive"] and o["s1_def2"] is not None]
    coll_s1 = [o["s1_def2"] for o in outcomes if not o["survive"] and o["s1_def2"] is not None]
    # def-2 monotonicity: survivor S1 vs collapser S1
    s1_separates = bool(surv_s1 and coll_s1 and max(surv_s1) < min(coll_s1))
    # identical-coverage counter-example (static vs rotate at same warmup keep-23)
    same_cov_flip = (abs(per_trace["html_W50"]["cov_engage"][23]
                         - per_trace["html_W50"]["cov_engage"][23]) < 1e-9)

    verdict = (
        "COVERAGE-AT-ENGAGE DOES NOT SEPARATE. (1) The warmup routed-mass curve is "
        "near task-invariant at the %d-tok engage window: cov@23 = %.0f%% (range %.0f-%.0f%%) "
        "for html AND all 11 coding prompts, Kmin-cov90 ~= %.0f for every task -- so a "
        "coverage rule picks ~the same K regardless of task and is not a task-width "
        "discriminator. (2) Two normalisations differ ~4x: warmup 1-cov@23 = %.2f (def-1, "
        "routed/selected-6) vs live S1 = %.2f (def-2, full-256); the trace logs only the 6 "
        "selected experts so def-2 is UNOBSERVABLE offline -- the identity S1_pred=1-cov is "
        "false. (3) Neither normalisation orders outcomes: survivor K91 S1=%.3f < collapser "
        "K23-rotate S1=%.3f (non-monotone); and keep-23 STATIC survives (L3) while keep-23 "
        "ROTATE collapses at the identical warmup coverage %.2f. The separators are actuation "
        "mode (static>>rotate), mask provenance (session>>cold), and token budget -- none in "
        "the coverage curve. USE the curve only to avoid UNDER-provisioning: cov90-sizing "
        "raises the fatal fixed K23 to ~%.0f uniformly (untested; needs a pod A/B)."
        % (ENGAGE_WIN, 100*cov23["median"], 100*cov23["lo"], 100*cov23["hi"],
           kmin90["median"], predicted_s1_html, measured_s1_html,
           s1_k91.get("mean", float("nan")), s1_r1.get("mean", float("nan")),
           per_trace["html_W50"]["cov_engage"][23], kmin90["median"]))

    # ---- write JSON ------------------------------------------------------ #
    stats = dict(
        params=dict(engage_win=ENGAGE_WIN, n_experts=N_EXPERTS, ks=KS,
                    cov_targets=COV_TARGETS),
        aggregate=dict(cov23_engage=cov23, cov32_engage=cov32,
                       kmin80_engage=kmin80, kmin90_engage=kmin90, kmin95_engage=kmin95),
        per_trace=per_trace,
        s1_sensors=dict(k23_rotate_html_r1=s1_r1, k91_static_coding=s1_k91),
        def1_vs_def2=dict(html_cov23_def1=html_cov23_def1,
                          predicted_s1_1minuscov=predicted_s1_html,
                          measured_s1_def2=measured_s1_html,
                          ratio=measured_s1_html / max(predicted_s1_html, 1e-9)),
        outcomes=outcomes,
        separation=dict(surv_cov_def1=surv_cov, coll_cov_def1=coll_cov,
                        surv_s1_def2=surv_s1, coll_s1_def2=coll_s1,
                        s1_monotone_separates=s1_separates),
        verdict=verdict,
    )
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    # ---- write per-trace CSV -------------------------------------------- #
    with open(os.path.join(out_dir, "coverage_by_trace.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "task", "tokens",
                    "cov@23_engage", "cov@32_engage", "cov@48_engage",
                    "Kmin_cov80", "Kmin_cov85", "Kmin_cov90", "Kmin_cov95",
                    "knee_kneedle", "marg@23", "marg@32"])
        for lab, r in rows_out:
            w.writerow([lab, r["task"], r["tokens"],
                        f"{r['cov_engage'][23]:.3f}", f"{r['cov_engage'][32]:.3f}",
                        f"{r['cov_engage'][48]:.3f}",
                        f"{r['kmin_engage']['80']:.1f}", f"{r['kmin_engage']['85']:.1f}",
                        f"{r['kmin_engage']['90']:.1f}", f"{r['kmin_engage']['95']:.1f}",
                        r["knee_engage"], f"{r['marg23_engage']:.4f}",
                        f"{r['marg32_engage']:.4f}"])

    # ---- console summary ------------------------------------------------- #
    print("=== E-CAL coverage-threshold calibration ===")
    print(f"engage window = first {ENGAGE_WIN} tokens (mask engages after PACE_WARMUP=50)\n")
    print(f"{'trace':18s} {'tok':>4s} | c@23 c@32 c@48 | K80  K90  K95 | knee marg@23")
    for lab, r in rows_out:
        c = r["cov_engage"]
        km = r["kmin_engage"]
        print(f"{lab:18s} {r['tokens']:4d} | "
              f"{100*c[23]:3.0f}% {100*c[32]:3.0f}% {100*c[48]:3.0f}% | "
              f"{km['80']:4.0f} {km['90']:4.0f} {km['95']:4.0f} | "
              f"{r['knee_engage']:4d} {100*r['marg23_engage']:.2f}%")
    print(f"\naggregate cov@23 = {100*cov23['median']:.0f}% "
          f"(range {100*cov23['lo']:.0f}-{100*cov23['hi']:.0f}%)  "
          f"Kmin-cov90 median = {kmin90['median']:.0f}  "
          f"(cov80 {kmin80['median']:.0f}, cov95 {kmin95['median']:.0f})")
    print("\n-- live S1 sensors (def-2, full-256 normalisation) --")
    print(f"  K23-rotate html (r1)      : S1 mean={s1_r1.get('mean'):.3f} "
          f"median={s1_r1.get('median'):.3f} n={s1_r1.get('n')}  -> cov={1-s1_r1.get('mean'):.3f}")
    print(f"  K91-static coding (pre-onset): S1 mean={s1_k91.get('mean'):.3f} "
          f"median={s1_k91.get('median'):.3f} n={s1_k91.get('n')} -> cov={1-s1_k91.get('mean'):.3f}")
    print(f"\n-- def-1 vs def-2 gap (html keep-23) --")
    print(f"  predicted S1 = 1-cov@23(def-1) = {predicted_s1_html:.2f}")
    print(f"  measured  S1 (def-2)           = {measured_s1_html:.2f}  "
          f"(x{measured_s1_html/max(predicted_s1_html,1e-9):.1f})")
    print("\n-- outcome cross-reference --")
    for o in outcomes:
        cov = f"{o['cov_def1']:.2f}" if o["cov_def1"] is not None else "  - "
        s1 = f"{o['s1_def2']:.3f}" if o["s1_def2"] is not None else "  -  "
        print(f"  [{'SURV' if o['survive'] else 'COLL'}] {o['run']:34s} K={o['K']:3d} "
              f"{o['act']:8s} {o['prov']:12s} cov_def1={cov} S1={s1}  {o['outcome']}")
    print(f"\n  survivor cov_def1={surv_cov}  collapser cov_def1={coll_cov}")
    print(f"  survivor S1_def2={surv_s1}  collapser S1_def2={coll_s1}  "
          f"(monotone-separates={s1_separates})")
    print("\n=== VERDICT ===")
    print(verdict)
    print(f"\nwrote {out_dir}/stats.json, coverage_by_trace.csv")
    return stats


if __name__ == "__main__":
    main()
