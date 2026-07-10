#!/usr/bin/env python3
"""E-MODEL — REAP-LOOP decision model v0 (design over trial).

OFFLINE ONLY. Reads routing-weight traces + graded-outcome ledger already on
disk (no GPU / WSL / pod). Turns the archive of "config attempts" into a
*calculated* decision model with three coupled components:

  (1) WIDTH SENSOR -> K* ESTIMATOR
      Identity-based task-width metrics from the weighted warmup traces (NOT
      mass — E-CAL proved routed-mass coverage is task-invariant at engage,
      cov@23=79% for html AND all 11 coding prompts). We compute three
      identity metrics on sliding windows and ask which one separates
      "K that holds" from "K that collapses" better than mass does.

  (2) COLLAPSE HAZARD
      Per-token collapse hazard fitted from every graded run with a known
      loop onset (retro-grade, M1a/M1b, armA, pod3, T4, knee ladder),
      stratified by (task width class, actuation mode). Exponential /
      piecewise, with Poisson CIs and declared confounders.

  (3) OBJECTIVE + DECISION TABLE
      SOTA-metric = good-tokens/second (wall) under an L2+ median constraint.
      good-tok/s = throughput(K) * E[useful fraction | K, width, corrections].
      throughput(K) from the E-LAT tier-latency model; useful fraction from
      the hazard model and the correction costs (rewind ~56 tok, breath ~70
      tok window + relearn). Emits K*, breath cadence and invariant-unit
      thresholds per width class, plus the minimal identification experiments
      the fit itself declares non-identifiable.

Reproduce:
    python scripts/decision_model.py \
        --reap-loop-root <reap-loop> \
        --moe-root <moe-aggressive-commit>/.claude/worktrees/elastic-bose-6ae1c7 \
        --out runs/ds4/20260711_emodel_decision

Every outcome datum carries a provenance string (see OUTCOME_LEDGER). No
absolute t/s is ever an *input* to the decision (P2): the controller thresholds
are in invariant units (identity-width, coverage%, hazard/token). t/s enters
only the objective's speed term, sourced from E-LAT and flagged HW-dependent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

# ===========================================================================
# Geometry constants (k91 meta.json; E1 / E-LAT provenance)
# ===========================================================================
N_ROUTED_LAYERS = 40        # 43 layers, first 3 dense -> 40 routed (E1 header)
TOP_K = 6                   # DS4_MAX_EXPERT_USED
N_EXPERTS = 256
WARMUP = 50                 # PACE_WARMUP: mask engages after 50 tok
COV_TARGET = 0.90           # cov90 anti-under-provisioning floor (E-CAL)

# ---------------------------------------------------------------------------
# Width-sensor window parameters. A window must be short enough to sit inside
# the 50-tok engage window in spirit, long enough to have >=1 selection of the
# stationary set. 24 tok * 6 = 144 selection events/layer over <=256 experts.
# ---------------------------------------------------------------------------
WIN = 24
STEP = 8
N_FIX = 49          # fixed engage window (shortest trace html_W50 = 49 tok) for
                    # LENGTH-CONTROLLED width metrics — union growth over the whole
                    # trace is confounded by trace length (short traces grow faster).


# ===========================================================================
# PART 1 — WIDTH SENSOR (identity-based, computed from the weighted traces)
# ===========================================================================
def load_trace_positional(path):
    """Return list of (pos, layer, [6 experts]) rows and the mask-detector
    median distinct-experts/layer (to flag already-masked traces)."""
    rows = []
    per_layer_sets = defaultdict(set)
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        eidx = [header.index(f"e{i}") for i in range(6)]
        lidx = header.index("layer")
        pidx = header.index("pos")
        for row in rd:
            if not row:
                continue
            pos = int(row[pidx]); layer = int(row[lidx])
            experts = [int(row[i]) for i in eidx]
            rows.append((pos, layer, experts))
            for e in experts:
                per_layer_sets[layer].add(e)
    distinct = [len(s) for s in per_layer_sets.values()]
    med_distinct = statistics.median(distinct) if distinct else 0
    return rows, med_distinct


def _windows(positions):
    """Yield (lo_pos, hi_pos) sliding windows over sorted unique positions."""
    ps = sorted(set(positions))
    n = len(ps)
    i = 0
    while i < n:
        lo = ps[i]
        hi_i = min(i + WIN, n)
        hi = ps[hi_i - 1]
        yield lo, hi
        if hi_i >= n:
            break
        i += STEP


def width_metrics(rows):
    """Three IDENTITY-based width metrics for a single trace.

    Returns dict with:
      union_cov90 : mean over (window,layer) of #distinct experts needed to
                    cover 90% of the *selection-frequency* mass in the window.
                    (identity diversity of REUSE; high = wide.)
      churn_topk  : mean over consecutive (window,layer) pairs of 1 - Jaccard
                    of the top-K(=23) experts by selection frequency.
                    (how fast the identity of the important set moves; high = wide.)
      union_slope : new distinct experts discovered per token, per layer,
                    normalised to /100 tok. Fit of cumulative-union vs token.
                    (does the task keep recruiting experts? high = wide.)
      sat_ratio   : union(all tokens) / union(first half). ~1 = saturated (narrow),
                    >>1 = still growing (wide). Robust companion to union_slope.
      union_fix   : LENGTH-CONTROLLED. distinct experts per layer discovered in
                    the FIRST N_FIX=49 tokens (the common engage window). This is
                    the clean width signal: union_slope/sat_ratio are confounded
                    by trace length; union_fix is measured on the same token count
                    for every trace.
      newexp_late : fraction of union_fix that is FIRST seen in tok 25-49 vs 0-24
                    (late-discovery rate). High = the task keeps recruiting new
                    experts inside the window = wide; ~0 = saturates early = narrow.
      n_tok, n_win
    """
    # index rows by layer -> ordered list of (pos, experts)
    by_layer = defaultdict(list)
    for pos, layer, experts in rows:
        by_layer[layer].append((pos, experts))
    for L in by_layer:
        by_layer[L].sort(key=lambda r: r[0])

    positions = sorted({pos for pos, _, _ in rows})
    n_tok = len(positions)

    cov90_vals = []
    churn_vals = []
    slope_vals = []
    sat_vals = []
    fix_vals = []
    late_vals = []

    for L, seq in by_layer.items():
        poslist = [p for p, _ in seq]
        # ---- per-window selection-frequency profile ----
        win_topk = []          # list of (set of top-K experts) per window
        for lo, hi in _windows(poslist):
            freq = defaultdict(int)
            total = 0
            for p, experts in seq:
                if lo <= p <= hi:
                    for e in experts:
                        freq[e] += 1
                        total += 1
            if total == 0:
                continue
            # union_cov90: distinct experts to reach 90% of selection freq
            counts = sorted(freq.values(), reverse=True)
            cum = 0; k90 = 0
            for c in counts:
                cum += c; k90 += 1
                if cum >= COV_TARGET * total:
                    break
            cov90_vals.append(k90)
            # top-K identity set for churn
            topk = set(sorted(freq, key=lambda e: freq[e], reverse=True)[:23])
            win_topk.append(topk)
        # churn between consecutive windows
        for a, b in zip(win_topk, win_topk[1:]):
            union = a | b
            if union:
                churn_vals.append(1.0 - len(a & b) / len(union))
        # ---- union growth: cumulative distinct experts vs token ----
        seen = set()
        cum_curve = []
        for p, experts in seq:
            for e in experts:
                seen.add(e)
            cum_curve.append(len(seen))
        if len(cum_curve) >= 8:
            x = np.arange(len(cum_curve), dtype=float)
            y = np.array(cum_curve, dtype=float)
            # slope of cumulative-union vs token, per 100 tok (LENGTH-CONFOUNDED)
            slope = float(np.polyfit(x, y, 1)[0]) * 100.0
            slope_vals.append(slope)
            half = len(cum_curve) // 2
            u_half = cum_curve[half - 1] if half >= 1 else cum_curve[-1]
            sat_vals.append(cum_curve[-1] / u_half if u_half else float("nan"))
        # ---- LENGTH-CONTROLLED: union over the FIRST N_FIX tokens ----
        first = seq[:N_FIX]
        if len(first) >= 24:
            seen_early = set()   # tok 0..24
            seen_all = set()     # tok 0..N_FIX
            for i, (p, experts) in enumerate(first):
                for e in experts:
                    seen_all.add(e)
                    if i < N_FIX // 2:
                        seen_early.add(e)
            fix_vals.append(len(seen_all))
            new_late = len(seen_all) - len(seen_early)
            late_vals.append(new_late / len(seen_all) if seen_all else float("nan"))

    def m(v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    return dict(
        union_cov90=m(cov90_vals),
        churn_topk=m(churn_vals),
        union_slope=m(slope_vals),
        sat_ratio=m(sat_vals),
        union_fix=m(fix_vals),
        newexp_late=m(late_vals),
        n_tok=n_tok,
        n_win=len(cov90_vals),
    )


def assemble_traces(reap_root, moe_root):
    """(label, task, path, rows, med_distinct) for every weighted full-model trace."""
    sources = [
        ("html_W50", "html",
         os.path.join(reap_root, "runs/ds4/20260710_pod_cache1024_warmup_replay/W50/route_W50.csv")),
        ("html_W130", "html",
         os.path.join(reap_root, "runs/ds4/20260710_pod_cache1024_warmup_replay/W130/route_W130.csv")),
    ]
    tgz = os.path.join(moe_root, "runs/reap/k91_coding_vram/trace_coding.tgz")
    tmp = None
    if os.path.exists(tgz):
        tmp = tempfile.mkdtemp(prefix="emodel_coding_")
        with tarfile.open(tgz) as tf:
            tf.extractall(tmp)
        cdir = os.path.join(tmp, "trace_coding")
        if os.path.isdir(cdir):
            for name in sorted(os.listdir(cdir)):
                if name.startswith("trace_") and name.endswith(".csv"):
                    tag = name.replace("trace_", "").replace(".csv", "")
                    sources.append((f"code_{tag}", "coding", os.path.join(cdir, name)))
    out = []
    for label, task, path in sources:
        if not os.path.exists(path):
            continue
        rows, med = load_trace_positional(path)
        out.append((label, task, path, rows, med))
    return out


def run_width_sensor(reap_root, moe_root):
    traces = assemble_traces(reap_root, moe_root)
    per_trace = {}
    for label, task, path, rows, med in traces:
        wm = width_metrics(rows)
        wm["task"] = task
        wm["med_distinct"] = med
        wm["masked"] = med < 30
        per_trace[label] = wm
    return per_trace


# ===========================================================================
# PART 2 — GRADED-OUTCOME LEDGER + COLLAPSE HAZARD
# ===========================================================================
# Every row is one *rollout* (n>=1) with a known outcome. onset = generated
# token at which the loop/degeneration starts (None = survived to `budget`
# with no detected loop -> right-censored). Provenance is a repo path.
# width_class in {narrow, medium, wide}; actuation in {static, rotate,
# frozen, admit, rotate_stop}. hold = final grade >= L2 OR clean close.
OUTCOME_LEDGER = [
    # --- cyberpunk html (WIDE): collapses at K23 under every actuation -----
    dict(id="M1a_W50_r01", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W50", budget=4000, hw="local3060",
         level=0, onset=None, hold=False,  # L0, no clean loop onset flagged (degenerate)
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1a_W50_r02", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W50", budget=4000, hw="local3060",
         level=0, onset=118, hold=False,
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1a_W50_r03", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W50", budget=4000, hw="local3060",
         level=0, onset=757, hold=False,
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1a_W100_r01", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W100", budget=4000, hw="local3060",
         level=0, onset=617, hold=False,
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1a_W100_r02", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W100", budget=4000, hw="local3060",
         level=1, onset=469, hold=False,
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1a_W100_r03", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate", provenance="session_W100", budget=4000, hw="local3060",
         level=0, onset=586, hold=False,
         src="runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/ANALYSIS.md"),
    # M1b: W50 rotate + anti-repeat stopper (airbag)
    dict(id="M1b_W50_r01", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate_stop", provenance="session_W50", budget=4000, hw="local3060",
         level=2, onset=685, hold=True,
         src="runs/ds4/20260710_m1b_w50_stopguard_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1b_W50_r02", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate_stop", provenance="session_W50", budget=4000, hw="local3060",
         level=0, onset=848, hold=False,
         src="runs/ds4/20260710_m1b_w50_stopguard_ctx8192_n3/ANALYSIS.md"),
    dict(id="M1b_W50_r03", task="cyberpunk", width_class="wide", K=23,
         actuation="rotate_stop", provenance="session_W50", budget=4000, hw="local3060",
         level=1, onset=328, hold=False,
         src="runs/ds4/20260710_m1b_w50_stopguard_ctx8192_n3/ANALYSIS.md"),
    # armA: cyberpunk STATIC K23 (byte-identical x3 -> 1 independent obs); loop
    dict(id="armA_static_cyber", task="cyberpunk", width_class="wide", K=23,
         actuation="static", provenance="session_W50_2phase", budget=4000, hw="pod3090",
         level=0, onset=120, hold=False, weight=1.0,  # deterministic x3 => weight 1
         src="runs/ds4/20260710_pod_static_ab_ctx8192/armA_k23/ (byte-identical, loop https.com)"),
    dict(id="pod3_frozen_cyber", task="cyberpunk", width_class="wide", K=23,
         actuation="frozen", provenance="session_W50", budget=4000, hw="pod3090",
         level=0, onset=68, hold=False,
         src="runs/ds4/20260710_pod3_s3_ab_frozen_vs_admit/REPORT.md (loop ' no, no,')"),
    # --- coffee (NARROW): holds at K23 static, budget 1200 -----------------
    dict(id="pod3_frozen_coffee_r0", task="coffee", width_class="narrow", K=23,
         actuation="frozen", provenance="session_W50", budget=1200, hw="pod3090",
         level=1, onset=None, hold=True,
         src="runs/ds4/20260710_pod3_s3_ab_frozen_vs_admit/REPORT.md (L1x3, closes)"),
    dict(id="pod3_frozen_coffee_r1", task="coffee", width_class="narrow", K=23,
         actuation="frozen", provenance="session_W50", budget=1200, hw="pod3090",
         level=1, onset=None, hold=True,
         src="runs/ds4/20260710_pod3_s3_ab_frozen_vs_admit/REPORT.md"),
    dict(id="pod3_frozen_coffee_r2", task="coffee", width_class="narrow", K=23,
         actuation="frozen", provenance="session_W50", budget=1200, hw="pod3090",
         level=1, onset=None, hold=True,
         src="runs/ds4/20260710_pod3_s3_ab_frozen_vs_admit/REPORT.md"),
    dict(id="s5_coffee_r00", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W50_2phase", budget=1200, hw="pod3090",
         level=2, onset=None, hold=True,
         src="runs/ds4/20260710_pod_static_ab_ctx8192/s5_coffee_k23/ (L2, closes+popup)"),
    dict(id="s5_coffee_r02", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W50_2phase", budget=1200, hw="pod3090",
         level=2, onset=None, hold=True,
         src="runs/ds4/20260710_pod_static_ab_ctx8192/s5_coffee_k23/"),
    dict(id="s5_coffee_r01", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W50_2phase", budget=1200, hw="pod3090",
         level=1, onset=70, hold=False,  # harness fence anomaly, not a mask collapse
         confound="harness_fence_strip", weight=0.0,  # excluded from hazard
         src="runs/ds4/20260710_pod_static_ab_ctx8192/s5_coffee_k23/ (r01 fence-strip bug)"),
    # T4 coffee W-sweep local (2-phase static K23, freeze-safe) -- medians
    dict(id="T4_coffee_W30", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W30_2phase", budget=1200, hw="local3060",
         level=1, onset=None, hold=False,  # L1 median (borderline)
         src="runs/ds4/20260710_t4_t5_w_sweep_local/t4_W030/summary_median.csv"),
    dict(id="T4_coffee_W50", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W50_2phase", budget=1200, hw="local3060",
         level=2, onset=None, hold=True,
         src="runs/ds4/20260710_t4_t5_w_sweep_local/t4_W050/summary_median.csv"),
    dict(id="T4_coffee_W130", task="coffee", width_class="narrow", K=23,
         actuation="static", provenance="session_W130_2phase", budget=1200, hw="local3060",
         level=2, onset=None, hold=True,
         src="runs/ds4/20260710_t4_t5_w_sweep_local/t4_W130/summary_median.csv"),
    # --- historical knee ladder (cold-static; label-only, no onset) --------
    dict(id="knee_JSON_k20", task="json", width_class="narrow", K=20,
         actuation="static", provenance="cold_corpus", budget=None, hw="pod",
         level=3, onset=None, hold=True,
         src="docs/CLAIMS_CURRENT.md CLAIM-005 (JSON keep-20 = L3 exact)"),
    dict(id="knee_Python_k32", task="python", width_class="medium", K=32,
         actuation="static", provenance="cold_corpus", budget=None, hw="pod",
         level=3, onset=None, hold=True,
         src="docs/CLAIMS_CURRENT.md CLAIM-005 (Python keep-32 = L3)"),
    dict(id="knee_Python_k28", task="python", width_class="medium", K=28,
         actuation="static", provenance="cold_corpus", budget=None, hw="pod",
         level=0, onset=None, hold=False,  # keep-28 already breaks
         src="docs/CLAIMS_CURRENT.md CLAIM-005 (Python keep-28 breaks)"),
    dict(id="knee_frontpage_k23_cold", task="frontpage", width_class="wide", K=23,
         actuation="static", provenance="cold_corpus", budget=None, hw="pod",
         level=0, onset=None, hold=False,  # cold keep-23 = L0 loop
         src="docs/paper/PAPER.md 5.8 + E-CAL 3b (cold keep-23 frontpage L0 loop)"),
    dict(id="knee_frontpage_k32p_cold", task="frontpage", width_class="wide", K=32,
         actuation="static", provenance="cold_corpus", budget=None, hw="pod",
         level=0, onset=None, hold=False,  # collapses at every cold K
         src="docs/CLAIMS_CURRENT.md CLAIM-005 (Frontpage >32 collapses cold, L0 loop)"),
    # K91 coding: mild static mask, slow erosion, holds ~2476 then drifts
    dict(id="K91_static_coding", task="coding", width_class="wide", K=91,
         actuation="static", provenance="cold_corpus", budget=2500, hw="pod",
         level=2, onset=2476, hold=True,  # coherent ~2200, text-lock 2476
         src="runs/ds4/20260710_scope_divergence_pod/README.md (K91 coherent ~2200, lock 2476)"),
]


def fit_exponential_hazard(rows):
    """MLE exponential hazard lambda = events / exposure, with a Poisson 95%
    CI on the event count. Each row: onset (event time in tok) or None
    (censored at budget). `weight` scales the contribution (deterministic
    duplicates -> weight<1; excluded confounds -> weight 0).

    Returns (lam, lo, hi, events, exposure, n) per stratum. Exposure for a
    censored/no-budget row falls back to a conservative horizon.
    """
    HORIZON_DEFAULT = 2000  # for label-only knee rows (no budget/onset)
    events = 0.0
    exposure = 0.0
    n = 0
    for r in rows:
        w = r.get("weight", 1.0)
        if w == 0:
            continue
        n += 1
        onset = r.get("onset")
        budget = r.get("budget") or HORIZON_DEFAULT
        if onset is not None:
            events += w
            exposure += w * onset
        else:
            # survived to budget (censored) OR label-only hold: full exposure
            exposure += w * budget
    if exposure <= 0:
        return dict(lam=float("nan"), lo=float("nan"), hi=float("nan"),
                    events=events, exposure=exposure, n=n)
    lam = events / exposure
    # Poisson CI on counts (Garwood), scaled by exposure
    if events > 0:
        lo_c = 0.5 * chi2_ppf(0.025, 2 * events)
        hi_c = 0.5 * chi2_ppf(0.975, 2 * (events + 1))
    else:
        lo_c = 0.0
        hi_c = 0.5 * chi2_ppf(0.975, 2)  # 3.688 for 0 events
    return dict(lam=lam, lo=lo_c / exposure, hi=hi_c / exposure,
                events=events, exposure=exposure, n=n)


def chi2_ppf(p, k):
    """Chi-square quantile via Wilson-Hilferty; adequate for CI reporting."""
    if k <= 0:
        return 0.0
    z = _norm_ppf(p)
    term = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return k * term ** 3


def _norm_ppf(p):
    """Acklam inverse-normal approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def run_hazard():
    """Strata are conditioned on (width, K-band, actuation). K-conditioning is
    essential: lumping wide-K23 (collapses ~100 tok) with wide-K91 (survives
    ~2476) hides the strongest driver — the coverage gap K vs the task knee."""
    def sel(width_class, actuations, klo, khi):
        return [r for r in OUTCOME_LEDGER
                if r["width_class"] == width_class
                and r["actuation"] in actuations
                and klo <= r["K"] <= khi]
    strata = {
        # WIDE, aggressive K23 — the best-measured collapse cell (the anchor)
        "wide_K23_rotate": sel("wide", ("rotate", "rotate_stop"), 1, 25),
        "wide_K23_static": sel("wide", ("static", "frozen"), 1, 25),
        # WIDE, mild K91 — the slow-erosion survivor (K91 static coding)
        "wide_K91_static": sel("wide", ("static", "frozen"), 80, 200),
        # NARROW K23 static — coffee, 0 collapses
        "narrow_K23_static": sel("narrow", ("static", "frozen"), 1, 25),
        # MEDIUM K28-32 static — Python knee (label-only)
        "medium_K_static": sel("medium", ("static", "frozen"), 1, 40),
    }
    fits = {name: fit_exponential_hazard(rows) for name, rows in strata.items()}
    return fits, strata


# ===========================================================================
# PART 3 — VELOCITY(K) FROM E-LAT + OBJECTIVE + DECISION TABLE
# ===========================================================================
# E-LAT local-3060 calibration (runs/ds4/20260710_elat_tier_latency/REPORT.md).
# t_ss = t_compute + 258*miss*t_b. Working-set(K) = K * N_ROUTED_LAYERS experts.
# Cache slots (12 GB) ~ 407 (E1). miss(K) rises as working-set exceeds cache.
ELAT_T_COMPUTE_MS = 74.9     # local 3060
ELAT_T_B_MS = 0.952          # per expert H2D copy, local
E_PER_TOKEN = 43 * TOP_K     # 258 (E-LAT uses 43 MoE layers for recall demand)
CACHE_SLOTS = 407            # max VRAM slots @12 GB (E1)
CORR_REWIND_TOK = 56         # detection ~40 (S1_REWIND FIRE median) + margin 16
CORR_BREATH_TOK = 70         # breath window (J28: breath 290->370 = 80 tok; ~70 useful lost)
BREATH_RELEARN_FRAC = 0.15   # D6b overhead 13-17% during a breath cycle


def miss_rate(K):
    """Fraction of the 258 per-token expert recalls that miss VRAM, as a
    function of mask size K. Working-set = K*40 experts. Below cache it trends
    to the measured floor (resident hit ~0 runtime -> ~1.0 miss at K23 today,
    E-LAT), above cache it saturates. Piecewise-linear surrogate anchored to
    the segmented-timing claim (keep-8 fits/accelerates, keep-32 stuck)."""
    ws = K * N_ROUTED_LAYERS
    # Anchors: keep-8 ws=320 fits (miss ~0.55 -> fast), keep-23 ws=920 (miss ~1.0
    # measured today, resident-hit bug), keep-32 ws=1280 > cache stuck (~1.0).
    if ws <= CACHE_SLOTS:
        return 0.55 * ws / CACHE_SLOTS
    return min(1.0, 0.55 + 0.45 * (ws - CACHE_SLOTS) / CACHE_SLOTS)


def tps_of_K(K):
    """Steady-state t/s on local 3060 for a static mask of size K (E-LAT)."""
    t_ss_ms = ELAT_T_COMPUTE_MS + E_PER_TOKEN * miss_rate(K) * ELAT_T_B_MS
    return 1000.0 / t_ss_ms


def expected_useful_fraction(K, lam, budget, mode):
    """E[useful tokens] / budget for a run of `budget` tokens at collapse
    hazard `lam` (per tok), under a correction `mode`:
      - 'none'   : first collapse ends usefulness -> useful = E[min(T,budget)]
      - 'rewind' : each collapse costs CORR_REWIND_TOK, then continues
      - 'breath' : periodic breath every 1/lam-ish tok; costs CORR_BREATH_TOK
                   and resets drift (hazard clock). Approx overhead model.
    Returns (useful_fraction, n_expected_corrections).
    """
    if lam <= 0 or not budget:
        return 1.0, 0.0
    if mode == "none":
        # expected productive tokens before first collapse, capped at budget
        # E[min(T,B)] for exponential = (1-e^{-lam B})/lam
        useful = (1 - math.exp(-lam * budget)) / lam
        return min(1.0, useful / budget), (1 - math.exp(-lam * budget))
    if mode == "rewind":
        # collapses recur; each caught after CORR_REWIND_TOK lost + regen.
        n_corr = lam * budget
        lost = n_corr * CORR_REWIND_TOK
        useful = max(0.0, budget - lost)
        # if correction latency exceeds mean time-to-collapse, it thrashes
        mttc = 1.0 / lam
        if CORR_REWIND_TOK >= mttc:
            return 0.0, n_corr           # thrash: can't outrun the hazard
        return useful / budget, n_corr
    if mode == "breath":
        # breathe on a cadence ~ a fraction of mean-time-to-collapse to pre-empt
        cadence = max(CORR_BREATH_TOK * 2, 0.6 / lam)
        n_breath = budget / cadence
        overhead = n_breath * CORR_BREATH_TOK * BREATH_RELEARN_FRAC
        # residual collapse hazard after breath resets: assume halved
        resid_lam = lam * 0.5
        surv = (1 - math.exp(-resid_lam * budget)) / resid_lam
        useful = min(budget, surv) - overhead
        return max(0.0, useful) / budget, n_breath
    return 1.0, 0.0


def good_tps(K, lam, budget, mode):
    """SOTA metric: good tokens per wall-second."""
    frac, _ = expected_useful_fraction(K, lam, budget, mode)
    return tps_of_K(K) * frac


# ---------------------------------------------------------------------------
# Hazard-vs-K model (honest, few-anchor): two additive terms.
#
#   lam(K, width) = DRIFT(width) + LAM_COV * max(0, exp((knee(width)-K)/SCALE) - 1)
#
#   DRIFT(width) : residual demand-shift hazard even when K >= knee (mask fully
#                  covers the task's *stationary* set). It is WIDTH-DEPENDENT
#                  because a wide task keeps shifting which experts it needs
#                  (E-PHASE: a frozen mask starves later structural phases,
#                  -17.6pt coverage), while a narrow task is ~one phase.
#                  Anchors: wide K91 coding drifts to a text-lock at ~2476 tok
#                  => DRIFT_wide ~ 1/2476 = 4.0e-4; narrow coffee K23 survives
#                  0/11600 tok => DRIFT_narrow < 1e-4 (below the measurement floor).
#   COV term     : ONE-SIDED coverage-gap hazard, active only when K < knee
#                  (under-provisioned). Vanishes at/above the knee. Anchor: wide
#                  cyberpunk K23 (knee 48, gap 25) MTTC ~1050 => lam ~9.5e-4.
#   SCALE        : gap decay length in K-units. knee(width) is the sizing target.
#
# This ties the WIDTH SENSOR to the hazard: identity-union-growth (non-stationary
# expert demand) is the same quantity that (a) raises the knee and (b) sets the
# residual DRIFT. The anchors are 3-4 points => wide CIs; the model names its own
# weakest legs as identification experiments (REPORT §5).
# ---------------------------------------------------------------------------
DRIFT = {"narrow": 1.0e-4, "medium": 2.0e-4, "wide": 4.0e-4}   # 1/MTTC at K>=knee
LAM_COV = 2.6e-4             # one-sided coverage-gap amplitude
HAZ_SCALE = 22.0            # K-units decay length
KNEE = {"narrow": 20, "medium": 32, "wide": 48}


def lam_of_K(K, cls):
    """Per-token collapse hazard as a function of mask size K and width class."""
    knee = KNEE[cls]
    cov_term = LAM_COV * max(0.0, math.exp((knee - K) / HAZ_SCALE) - 1.0)
    return DRIFT[cls] + cov_term


def build_decision_table(fits):
    """Compose objective for the three width classes and candidate K/mode."""
    classes = {
        "narrow": dict(budget=1500, Ks=[12, 16, 20, 23, 32]),
        "medium": dict(budget=2500, Ks=[16, 23, 28, 32, 38]),
        "wide":   dict(budget=4000, Ks=[12, 23, 38, 48, 64, 91]),
    }
    table = {}
    for cls, cfg in classes.items():
        rows = []
        for K in cfg["Ks"]:
            lamK = lam_of_K(K, cls)
            for mode in ("none", "rewind", "breath"):
                gt = good_tps(K, lamK, cfg["budget"], mode)
                frac, ncorr = expected_useful_fraction(K, lamK, cfg["budget"], mode)
                rows.append(dict(K=K, mode=mode, lamK=lamK, tps=tps_of_K(K),
                                 useful_frac=frac, n_corr=ncorr, good_tps=gt))
        best = max(rows, key=lambda r: r["good_tps"])
        table[cls] = dict(rows=rows, best=best, budget=cfg["budget"],
                          lam_K23=lam_of_K(23, cls), knee=KNEE[cls])
    return table


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--reap-loop-root", default=os.path.dirname(here))
    ap.add_argument("--moe-root",
                    default=r"C:/Users/imanu/source/repos/moe-aggressive-commit/.claude/worktrees/elastic-bose-6ae1c7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RL = args.reap_loop_root
    out_dir = args.out or os.path.join(RL, "runs/ds4/20260711_emodel_decision")
    os.makedirs(out_dir, exist_ok=True)

    # ---- PART 1: width sensor -------------------------------------------
    width = run_width_sensor(RL, args.moe_root)

    print("=" * 78)
    print("PART 1 — IDENTITY-BASED WIDTH SENSOR (per weighted full-model trace)")
    print("=" * 78)
    print(f"{'trace':24s} {'task':7s} {'tok':>4s} {'uFix49':>6s} {'late':>5s} "
          f"{'uCov90':>6s} {'churn':>6s} {'uSlope':>7s} {'satR':>5s}")
    by_task = defaultdict(list)
    for label in sorted(width):
        w = width[label]
        by_task[w["task"]].append(w)
        print(f"{label:24s} {w['task']:7s} {w['n_tok']:4d} {w['union_fix']:6.1f} "
              f"{w['newexp_late']:5.2f} {w['union_cov90']:6.2f} {w['churn_topk']:6.3f} "
              f"{w['union_slope']:7.2f} {w['sat_ratio']:5.2f}")

    def stat(task, key):
        vals = [w[key] for w in by_task[task] if w[key] == w[key]]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    METRICS = ("union_fix", "newexp_late", "union_cov90", "churn_topk",
               "union_slope", "sat_ratio")
    print("\n-- per-task means (identity width; union_fix/newexp_late are length-controlled) --")
    for task in sorted(by_task):
        line = f"  {task:8s} "
        for key in METRICS:
            m, s = stat(task, key)
            line += f"{key}={m:.2f}±{s:.2f}  "
        print(line)

    # discrimination: spread of each metric ACROSS traces (higher = more
    # task-informative). Compare to mass-cov@23 (E-CAL: 73-82%, ~flat).
    def cv(key):
        vals = [width[l][key] for l in width if width[l][key] == width[l][key]]
        return float(np.std(vals) / np.mean(vals)) if vals and np.mean(vals) else float("nan")
    disc = {k: cv(k) for k in METRICS}
    print("\n-- cross-trace coefficient-of-variation (discrimination power) --")
    for k, v in sorted(disc.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} CV = {v:.3f}")
    print("  (reference: mass cov@23 across the same 13 traces = 79% range 73-82%"
          " -> CV ~ 0.033, near-flat, E-CAL)")

    # ---- PART 2: hazard --------------------------------------------------
    fits, strata = run_hazard()
    print("\n" + "=" * 78)
    print("PART 2 — COLLAPSE HAZARD (exponential MLE, per stratum, Poisson 95% CI)")
    print("=" * 78)
    print(f"{'stratum':16s} {'n':>3s} {'events':>7s} {'exposure':>9s} "
          f"{'lam/tok':>9s} {'1/lam':>7s} {'95% CI (1/lam)':>20s}")
    for name, f in fits.items():
        mttc = 1.0 / f["lam"] if f["lam"] and f["lam"] == f["lam"] and f["lam"] > 0 else float("inf")
        ci_lo = 1.0 / f["hi"] if f["hi"] and f["hi"] > 0 else float("inf")
        ci_hi = 1.0 / f["lo"] if f["lo"] and f["lo"] > 0 else float("inf")
        print(f"{name:16s} {f['n']:3d} {f['events']:7.1f} {f['exposure']:9.0f} "
              f"{f['lam']:9.5f} {mttc:7.0f} {ci_lo:8.0f}-{ci_hi:<8.0f}")

    # ---- PART 3: objective + decision table ------------------------------
    table = build_decision_table(fits)
    print("\n" + "=" * 78)
    print("PART 3 — OBJECTIVE (good-tok/s = tps(K) * useful_fraction) + DECISION")
    print("=" * 78)
    for cls in ("narrow", "medium", "wide"):
        t = table[cls]
        print(f"\n[{cls.upper()}] knee={t['knee']}, lam@K23={t['lam_K23']:.5f}/tok, "
              f"budget={t['budget']}")
        print(f"  {'K':>3s} {'mode':>7s} {'lam(K)':>8s} {'tps':>5s} "
              f"{'useful':>6s} {'corr':>5s} {'good_tps':>8s}")
        for r in sorted(t["rows"], key=lambda x: -x["good_tps"])[:6]:
            print(f"  {r['K']:3d} {r['mode']:>7s} {r['lamK']:8.5f} {r['tps']:5.2f} "
                  f"{r['useful_frac']:6.2f} {r['n_corr']:5.1f} {r['good_tps']:8.2f}")
        b = t["best"]
        none_rows = [r for r in t["rows"] if r["mode"] == "none"]
        bn = max(none_rows, key=lambda r: r["good_tps"])
        t["best_none"] = bn
        print(f"  -> BEST overall : K={b['K']} mode={b['mode']} good_tps={b['good_tps']:.2f} "
              f"(tps {b['tps']:.2f} x useful {b['useful_frac']:.2f})")
        print(f"  -> BEST no-airbag (mode=none): K={bn['K']} good_tps={bn['good_tps']:.2f} "
              f"(useful {bn['useful_frac']:.2f}) <- fallback if rewind unproven")

    # recovery-ladder question: does K12 + rewind + breath beat higher-K static
    # on the WIDE class? (the user's proposed experiment)
    print("\n-- recovery-ladder check (WIDE, budget 4000): K12+rewind/breath vs high-K static --")
    for K, mode in [(12, "rewind"), (12, "breath"), (23, "none"), (38, "none"),
                    (48, "none"), (64, "none"), (91, "none")]:
        lamK = lam_of_K(K, "wide")
        gt = good_tps(K, lamK, 4000, mode)
        frac, nc = expected_useful_fraction(K, lamK, 4000, mode)
        mttc = 1.0 / lamK if lamK > 0 else float("inf")
        print(f"  K={K:2d} {mode:7s}: lam(K)={lamK:.5f} MTTC={mttc:5.0f} "
              f"tps={tps_of_K(K):.2f} useful={frac:.2f} good_tps={gt:.2f}")

    # ---- persist ---------------------------------------------------------
    payload = dict(
        width_sensor=width,
        width_discrimination_cv=disc,
        hazard_fits=fits,
        hazard_strata={k: [r["id"] for r in v] for k, v in strata.items()},
        decision_table={cls: dict(best=table[cls]["best"],
                                  knee=table[cls]["knee"],
                                  lam_K23=table[cls]["lam_K23"],
                                  budget=table[cls]["budget"],
                                  rows=table[cls]["rows"])
                        for cls in table},
        hazard_model=dict(DRIFT=DRIFT, LAM_COV=LAM_COV,
                          HAZ_SCALE=HAZ_SCALE, KNEE=KNEE),
        constants=dict(WIN=WIN, STEP=STEP, WARMUP=WARMUP, CACHE_SLOTS=CACHE_SLOTS,
                       ELAT_T_COMPUTE_MS=ELAT_T_COMPUTE_MS, ELAT_T_B_MS=ELAT_T_B_MS,
                       CORR_REWIND_TOK=CORR_REWIND_TOK, CORR_BREATH_TOK=CORR_BREATH_TOK,
                       BREATH_RELEARN_FRAC=BREATH_RELEARN_FRAC),
    )
    with open(os.path.join(out_dir, "decision_model.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {out_dir}/decision_model.json")
    return payload


if __name__ == "__main__":
    main()
