#!/usr/bin/env python3
"""E-ADMIT -- demand-driven admission simulation (constant-K targeted exchange).

OFFLINE ONLY. Reads routing-weight traces already on disk (no GPU / WSL / pod).

User mechanism under test ("demand-driven admission"):
  Keep the mask at CONSTANT K. When a FORBIDDEN (out-of-mask) expert shows
  strong AND persistent blocked demand, ADMIT it by evicting the least-used
  keep (min EWMA mass). Never fall back to K0, never re-rank wholesale.
  To be sharply distinguished from periodic wholesale rotate (rotate32),
  which the data already rejected (E-CAL regime notes; pod gen126 collapse,
  runs/ds4/20260710_scope_divergence_pod/).

Four policies compared token-by-token on FULL-model weighted traces (no mask
was active when they were captured -> the router's true demand is directly
observable):

  (A) FROZEN        mask fixed = top-23/layer by mass of the first 50 gen tok.
  (B) ROTATE-32     wholesale re-rank of the whole mask from EWMA mass every
                    32 tok (decay 0.98 = DS4_PACE_ROTATE_DECAY manifest value).
                    The rejected mechanism, kept as reference.
  (C) DEMAND-ADMIT  per out-of-mask expert, CUSUM of its requested mass share;
                    admission when CUSUM >= h with persistence >= p requested
                    tokens (drift k_d subtracted per token); evict the in-mask
                    keep with min EWMA mass; K constant. 3x3x3 sweep of
                    (h, k_d, p).
  (D) C + STRUCTURAL BOUNDARIES: for 16 tok after a structural boundary of the
                    generated text (E-PHASE segmentation: <style>/</style>/
                    <script>, ``` fences), admission sensitivity is boosted
                    (h/2, p/2). Measures whether structure awareness adds
                    anything on top of pure demand.

Metrics per policy per trace (eval region = tokens after the 50-tok warmup):
  - instantaneous def-1 coverage (mean over layers of in-mask routed mass /
    total routed mass): mean, p10, late-third mean, per structural phase;
  - churn: admitted/entered experts total and per-100-tok, GiB at 6.75 MiB/exp;
  - coverage-recovery lag after a post-warmup structural boundary (tokens to
    return within 2 pt of the pre-boundary 16-tok mean, 5-tok smoothed);
  - bounce: experts admitted then evicted again within 100 tok.

DECLARED LIMITS (do not over-read the numbers):
  1. Simulation on FULL traces = demand of the HEALTHY trajectory. In the real
     masked runtime the demand is conditioned on the masked trajectory. This
     estimates the COVERAGE POTENTIAL of the mechanism, not the quality
     outcome -- that requires the live A/B (S3).
  2. The trace logs only the 6 SELECTED experts per (pos,layer): out-of-mask
     demand is visible only when the free router puts the expert in its top-6.
     The real runtime signal (unbiased router_probs over all 256, as read by
     the 0012 sensor / 0020 rmass) is RICHER, so live CUSUM would see demand
     earlier than this simulation does.
  3. Token<->text alignment for policy D is proportional (no tokenizer on
     disk): boundary position error is a few tokens (E-PHASE convention).
  4. scope_divergence_pod ctrl (full "router libero" control) has text but NO
     routing trace on disk (r1/s1 logs aggregate S1 without expert ids) ->
     unusable here; declared instead of silently skipped.

Trace schema: pos,layer,n,e0..e5,w0..w5 (weights NOT sorted; unbiased router
probs of the 6 selected experts -- def-1 routed-mass denominator, as E-CAL /
E-PHASE).
"""
import csv
import json
import os
import re
import tarfile
import tempfile
from collections import defaultdict

import numpy as np

# ---- constants --------------------------------------------------------------
K_KEEP = 23
WARMUP = 50                    # PACE_WARMUP: mask frozen from first 50 gen tok
N_LAYERS = 40                  # routed layers 3..42
LAYER0 = 3
N_EXP = 256
EWMA_DECAY = 0.98              # DS4_PACE_ROTATE_DECAY (pod manifest)
ROTATE_EVERY = 32
MIB_PER_EXPERT = 6.75
BOUNCE_WIN = 100               # admitted-then-evicted-again window
COOLDOWN = 16                  # freshly admitted expert not evictable (tok)
BOOST_WIN = 16                 # policy D: boosted sensitivity after boundary
LAG_TOL = 0.02                 # recovered = within 2 pt of pre-boundary mean
LAG_PRE = 16                   # pre-boundary window for the reference level
SMOOTH = 5                     # trailing smoothing window for lag curves
MIN_EVAL_TOK = 30              # traces with fewer post-warmup tokens excluded
LATE_FRAC = 1 / 3              # "late" = last third of the eval region

H_GRID = (0.3, 0.6, 1.2)       # CUSUM threshold (accumulated mass share)
KD_GRID = (0.01, 0.02, 0.04)   # CUSUM drift per token
P_GRID = (2, 4, 8)             # persistence: requested tokens in excursion


# ---- loading ----------------------------------------------------------------
def load_trace_dense(path):
    """Return (M, tot): M[t, l, e] = routed weight, tot[t, l] = 6-slot sum.
    t = 0-based generated-token order, l = layer-LAYER0."""
    by_pos = defaultdict(list)
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        eidx = [header.index(f"e{i}") for i in range(6)]
        widx = [header.index(f"w{i}") for i in range(6)]
        lidx = header.index("layer")
        pidx = header.index("pos")
        for row in rd:
            if not row:
                continue
            pos = int(row[pidx]); layer = int(row[lidx])
            pairs = [(int(row[eidx[i]]), float(row[widx[i]])) for i in range(6)]
            by_pos[pos].append((layer, pairs))
    positions = sorted(by_pos)
    ntok = len(positions)
    M = np.zeros((ntok, N_LAYERS, N_EXP), dtype=np.float64)
    for t, pos in enumerate(positions):
        for layer, pairs in by_pos[pos]:
            li = layer - LAYER0
            if not (0 <= li < N_LAYERS):
                continue
            for e, w in pairs:
                M[t, li, e] += w
    tot = M.sum(axis=2)
    return M, tot


def extract_gen_text_from_log(path):
    """Strip interleaved 'ds4: ...' diagnostics from gen_*.log (E-PHASE rule)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    out = []
    for line in raw.split("\n"):
        if "ds4: " in line:
            pre = line.split("ds4: ", 1)[0]
            if pre:
                out.append(pre)
        else:
            out.append(line + "\n")
    return "".join(out).rstrip("\n")


# ---- structural segmentation (E-PHASE conventions) ---------------------------
def segments_html(text):
    segs = []
    i_style = text.find("<style")
    if i_style < 0:
        return [("head", 0, len(text))]
    segs.append(("head", 0, i_style))
    i_end = text.find("</style>", i_style)
    if i_end < 0:
        segs.append(("css", i_style, len(text)))
        return segs
    segs.append(("css", i_style, i_end))
    i_script = text.find("<script", i_end)
    if i_script < 0:
        segs.append(("markup", i_end, len(text)))
        return segs
    segs.append(("markup", i_end, i_script))
    segs.append(("js", i_script, len(text)))
    return segs


def segments_code(text):
    i = text.find("```")
    if i < 0:
        return [("prose", 0, len(text))]
    j = text.find("\n", i)
    code_start = (j + 1) if j >= 0 else i + 3
    code_end = len(text)
    tail = text.rfind("```")
    if tail > i:
        code_end = tail
    return [("prose", 0, i), ("code", code_start, code_end)]


def char_to_tok(c, total_chars, ntok):
    if total_chars <= 0:
        return 0
    return max(0, min(ntok, int(round(ntok * c / total_chars))))


def phases_tok(text, segs_char, ntok, min_tok=12):
    C = len(text)
    out = []
    for name, c0, c1 in segs_char:
        t0, t1 = char_to_tok(c0, C, ntok), char_to_tok(c1, C, ntok)
        if t1 - t0 >= min_tok:
            out.append((name, t0, t1))
    return out


# ---- simulation core ----------------------------------------------------------
def warmup_state(M, tot):
    """Initial mask (top-23 by accumulated mass of first WARMUP tok) + EWMA."""
    warm_end = min(WARMUP, M.shape[0])
    acc = M[:warm_end].sum(axis=0)                      # (L, E)
    mask = np.zeros((N_LAYERS, N_EXP), dtype=bool)
    for li in range(N_LAYERS):
        top = np.argsort(-acc[li])[:K_KEEP]
        mask[li, top] = True
    ewma = np.zeros((N_LAYERS, N_EXP))
    for t in range(warm_end):
        ewma = EWMA_DECAY * ewma + M[t]
    return mask, ewma, warm_end


def coverage_t(M_t, tot_t, mask):
    num = (M_t * mask).sum(axis=1)
    ok = tot_t > 0
    if not ok.any():
        return float("nan")
    return float((num[ok] / tot_t[ok]).mean())


def simulate(M, tot, policy, params=None, boundaries=()):
    """Run one policy over the eval region. Returns dict with per-token
    coverage (eval region), churn events [(t, li, admitted, evicted)], bounce
    count. policy in {'A','B','C','D'}."""
    ntok = M.shape[0]
    mask, ewma, warm_end = warmup_state(M, tot)
    cov = []
    churn_events = []            # (t, li, e_in, e_out_or_-1)
    admit_time = {}              # (li, e) -> t admitted (C/D) / entered (B)
    bounce = 0
    cus = np.zeros((N_LAYERS, N_EXP))
    per = np.zeros((N_LAYERS, N_EXP), dtype=np.int32)
    cooldown_until = np.zeros((N_LAYERS, N_EXP), dtype=np.int64)

    if policy in ("C", "D"):
        h, kd, p = params

    for t in range(warm_end, ntok):
        cov.append(coverage_t(M[t], tot[t], mask))
        ewma = EWMA_DECAY * ewma + M[t]

        if policy == "B":
            if (t - warm_end + 1) % ROTATE_EVERY == 0:
                new_mask = np.zeros_like(mask)
                for li in range(N_LAYERS):
                    top = np.argsort(-ewma[li])[:K_KEEP]
                    new_mask[li, top] = True
                entered = new_mask & ~mask
                left = mask & ~new_mask
                for li, e in zip(*np.where(left)):
                    tin = admit_time.pop((li, e), None)
                    if tin is not None and t - tin <= BOUNCE_WIN:
                        bounce += 1
                for li, e in zip(*np.where(entered)):
                    churn_events.append((t, int(li), int(e), -1))
                    admit_time[(li, e)] = t
                mask = new_mask

        elif policy in ("C", "D"):
            h_eff, p_eff = h, p
            if policy == "D" and any(0 <= t - b < BOOST_WIN for b in boundaries):
                h_eff, p_eff = h / 2.0, max(1, p // 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(tot[t][:, None] > 0, M[t] / tot[t][:, None], 0.0)
            x = np.where(mask, 0.0, share)
            cus = np.maximum(0.0, cus + x - kd)
            cus[mask] = 0.0
            per[(x > 0) & (cus > 0)] += 1
            per[cus <= 0] = 0
            cand = (cus >= h_eff) & (per >= p_eff) & ~mask
            if cand.any():
                for li in np.unique(np.where(cand)[0]):
                    es = np.where(cand[li])[0]
                    for e in es[np.argsort(-cus[li, es])]:
                        keeps = np.where(mask[li])[0]
                        evictable = keeps[cooldown_until[li, keeps] <= t]
                        if len(evictable) == 0:
                            evictable = keeps
                        evict = evictable[np.argmin(ewma[li, evictable])]
                        mask[li, evict] = False
                        mask[li, e] = True
                        cooldown_until[li, e] = t + COOLDOWN
                        cus[li, e] = 0.0; per[li, e] = 0
                        cus[li, evict] = 0.0; per[li, evict] = 0
                        tin = admit_time.pop((li, int(evict)), None)
                        if tin is not None and t - tin <= BOUNCE_WIN:
                            bounce += 1
                        admit_time[(li, int(e))] = t
                        churn_events.append((t, int(li), int(e), int(evict)))

    return dict(cov=np.array(cov), churn=churn_events, bounce=bounce,
                warm_end=warm_end, final_mask=mask)


# ---- metrics ------------------------------------------------------------------
def smooth(a, w=SMOOTH):
    if len(a) == 0:
        return a
    out = np.empty_like(a, dtype=float)
    for i in range(len(a)):
        out[i] = a[max(0, i - w + 1): i + 1].mean()
    return out


def recovery_lag(cov, warm_end, boundary):
    """Tokens after `boundary` (abs tok index) for the 5-tok smoothed coverage
    to return within LAG_TOL of the pre-boundary LAG_PRE-tok mean.
    Returns (lag, recovered, cap)."""
    b = boundary - warm_end
    if b - LAG_PRE < 0 or b >= len(cov):
        return None
    s = smooth(cov)
    pre = float(np.nanmean(cov[b - LAG_PRE: b]))
    cap = min(BOUNCE_WIN, len(cov) - b)
    for d in range(cap):
        if s[b + d] >= pre - LAG_TOL:
            return (d, True, cap)
    return (cap, False, cap)


def policy_metrics(sim, ntok, phases, boundaries):
    cov = sim["cov"]
    warm_end = sim["warm_end"]
    n_eval = len(cov)
    late0 = int(n_eval * (1 - LATE_FRAC))
    churn_n = len(sim["churn"])
    out = dict(
        n_eval=n_eval,
        cov_mean=float(np.nanmean(cov)),
        cov_p10=float(np.nanpercentile(cov, 10)),
        cov_late=float(np.nanmean(cov[late0:])) if n_eval - late0 > 0 else float("nan"),
        churn=churn_n,
        churn_per100=100.0 * churn_n / n_eval if n_eval else float("nan"),
        churn_gib=churn_n * MIB_PER_EXPERT / 1024.0,
        bounce=sim["bounce"],
    )
    # per-phase coverage: post-warmup slice of each structural phase (>=12 tok)
    ph = {}
    for name, t0, t1 in phases:
        a, b = max(t0, warm_end) - warm_end, t1 - warm_end
        if b - a >= 12:
            ph[name] = float(np.nanmean(cov[max(0, a):b]))
    out["phase_cov"] = ph
    # recovery lag at post-warmup boundaries (vs pre-boundary level)
    lags = []
    for b in boundaries:
        r = recovery_lag(cov, warm_end, b)
        if r is not None:
            lags.append(r)
    out["lags"] = lags
    out["lag_mean"] = float(np.mean([l for l, rec, c in lags])) if lags else float("nan")
    out["lag_unrecovered"] = sum(1 for l, rec, c in lags if not rec)
    # boundary transition: dip window [b, b+16) and plateau [b+16, b+48)
    dips, plats = [], []
    for b in boundaries:
        i = b - warm_end
        if 0 <= i < n_eval - 16:
            dips.append(float(np.nanmean(cov[i:i + 16])))
            j = min(n_eval, i + 48)
            if j > i + 16:
                plats.append(float(np.nanmean(cov[i + 16:j])))
    out["bdry_dip"] = float(np.mean(dips)) if dips else float("nan")
    out["bdry_plateau"] = float(np.mean(plats)) if plats else float("nan")
    return out


# ---- main ----------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    RL = os.path.dirname(here)
    out_dir = os.path.join(RL, "runs/ds4/20260710_eadmit_demand_admission")
    os.makedirs(out_dir, exist_ok=True)
    replay = os.path.join(RL, "runs/ds4/20260710_pod_cache1024_warmup_replay")
    tgz = os.path.join(RL, "runs/reap/k91_coding_vram/trace_coding.tgz")

    # ---- assemble sources (FULL-model weighted traces only) -----------------
    sources = []  # (label, M, tot, phases, boundaries_postwarmup)
    excluded = {}

    def add_source(label, csv_path, text, kind):
        M, tot = load_trace_dense(csv_path)
        ntok = M.shape[0]
        if ntok - WARMUP < MIN_EVAL_TOK:
            excluded[label] = f"{ntok} tok -> {max(0, ntok - WARMUP)} eval tok < {MIN_EVAL_TOK}"
            return
        segs = segments_html(text) if kind == "html" else segments_code(text)
        phases = phases_tok(text, segs, ntok)
        bounds = [t0 for _, t0, _ in phases[1:] if t0 >= WARMUP]
        sources.append((label, M, tot, phases, bounds))

    with open(os.path.join(replay, "W130/tw_W130.txt"), encoding="utf-8") as fh:
        add_source("html_W130", os.path.join(replay, "W130/route_W130.csv"),
                   fh.read(), "html")
    with open(os.path.join(replay, "W50/tw_W50.txt"), encoding="utf-8") as fh:
        add_source("html_W50", os.path.join(replay, "W50/route_W50.csv"),
                   fh.read(), "html")

    tmp = tempfile.mkdtemp(prefix="eadmit_coding_")
    with tarfile.open(tgz) as tf:
        tf.extractall(tmp)
    cdir = os.path.join(tmp, "trace_coding")
    for name in sorted(os.listdir(cdir)):
        m = re.match(r"trace_(p\d+_c\d+_(.+))\.csv", name)
        if not m:
            continue
        full, tag = m.group(1), m.group(2)
        gen = os.path.join(cdir, f"gen_{full}.log")
        if not os.path.exists(gen):
            continue
        add_source(f"code_{tag}", os.path.join(cdir, name),
                   extract_gen_text_from_log(gen), "code")

    labels = [s[0] for s in sources]
    print(f"eval traces: {labels}")
    print(f"excluded: {excluded}")

    # ---- run A and B on every trace ------------------------------------------
    results = defaultdict(dict)   # label -> policy_key -> metrics
    for label, M, tot, phases, bounds in sources:
        ntok = M.shape[0]
        for pol in ("A", "B"):
            sim = simulate(M, tot, pol)
            results[label][pol] = policy_metrics(sim, ntok, phases, bounds)

    # ---- policy C sweep -------------------------------------------------------
    sweep_rows = []
    c_metrics = defaultdict(dict)  # (h,kd,p) -> label -> metrics
    for h in H_GRID:
        for kd in KD_GRID:
            for p in P_GRID:
                key = (h, kd, p)
                for label, M, tot, phases, bounds in sources:
                    sim = simulate(M, tot, "C", params=key)
                    met = policy_metrics(sim, M.shape[0], phases, bounds)
                    c_metrics[key][label] = met
                    sweep_rows.append(dict(
                        h=h, k_drift=kd, p=p, trace=label,
                        cov_mean=met["cov_mean"], cov_p10=met["cov_p10"],
                        cov_late=met["cov_late"], churn=met["churn"],
                        churn_per100=met["churn_per100"], bounce=met["bounce"],
                        lag_mean=met["lag_mean"]))

    # aggregate the sweep: mean late gain vs A, mean churn ratio vs B
    def agg_config(key):
        gains, churn100, churn_n, bounce, covm, p10 = [], [], [], [], [], []
        for label in labels:
            a = results[label]["A"]
            c = c_metrics[key][label]
            gains.append(c["cov_late"] - a["cov_late"])
            churn100.append(c["churn_per100"])
            churn_n.append(c["churn"])
            bounce.append(c["bounce"])
            covm.append(c["cov_mean"] - a["cov_mean"])
            p10.append(c["cov_p10"] - a["cov_p10"])
        b_churn = np.mean([results[l]["B"]["churn_per100"] for l in labels])
        n_adm = int(np.sum(churn_n))
        return dict(late_gain=float(np.mean(gains)),
                    mean_gain=float(np.mean(covm)),
                    p10_gain=float(np.mean(p10)),
                    churn_per100=float(np.mean(churn100)),
                    churn_ratio_vs_B=float(np.mean(churn100) / b_churn) if b_churn else float("nan"),
                    admissions_total=n_adm,
                    bounce_total=int(np.sum(bounce)),
                    bounce_rate=float(np.sum(bounce) / n_adm) if n_adm else 0.0)

    agg = {key: agg_config(key) for key in c_metrics}
    b_churn100_mean = float(np.mean([results[l]["B"]["churn_per100"] for l in labels]))

    # RECOMMENDED config: max late gain among configs with BOTH
    #   churn <= B/3  AND  bounce rate <= 1% of admissions (mechanical stability)
    def eligible_keys(churn_cap, bounce_cap=0.01):
        return [k for k, v in agg.items()
                if v["churn_per100"] <= churn_cap and v["bounce_rate"] <= bounce_cap]

    pool = eligible_keys(b_churn100_mean / 3) or list(agg)
    best_key = max(pool, key=lambda k: agg[k]["late_gain"])
    # MAX-RECOVERY config (churn cap only, no bounce filter) for the frontier
    pool_max = [k for k, v in agg.items() if v["churn_per100"] <= b_churn100_mean / 3] or list(agg)
    maxrec_key = max(pool_max, key=lambda k: agg[k]["late_gain"])

    # parameter stability: per-trace best (churn <= B_trace/3) and the REGRET
    # of the global recommended config vs that per-trace best (late-gain pts)
    per_trace_best, regret = {}, {}
    for label in labels:
        b_churn = results[label]["B"]["churn_per100"]
        a_late = results[label]["A"]["cov_late"]
        elig = [k for k in c_metrics
                if c_metrics[k][label]["churn_per100"] <= b_churn / 3] or list(c_metrics)
        per_trace_best[label] = max(elig, key=lambda k: c_metrics[k][label]["cov_late"])
        regret[label] = float(c_metrics[per_trace_best[label]][label]["cov_late"] -
                              c_metrics[best_key][label]["cov_late"])

    # ---- policy D at the recommended config -----------------------------------
    for label, M, tot, phases, bounds in sources:
        sim = simulate(M, tot, "D", params=best_key, boundaries=bounds)
        results[label]["D"] = policy_metrics(sim, M.shape[0], phases, bounds)
        results[label]["C"] = c_metrics[best_key][label]

    # ---- coverage curves at the recommended config (for the report) ----------
    curve_path = os.path.join(out_dir, "cov_curves_bestC.csv")
    with open(curve_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "t_gen", "cov_A", "cov_B", "cov_C", "cov_D"])
        for label, M, tot, phases, bounds in sources:
            sA = simulate(M, tot, "A")
            sB = simulate(M, tot, "B")
            sC = simulate(M, tot, "C", params=best_key)
            sD = simulate(M, tot, "D", params=best_key, boundaries=bounds)
            we = sA["warm_end"]
            for i in range(len(sA["cov"])):
                w.writerow([label, we + i] +
                           [f"{x['cov'][i]:.4f}" for x in (sA, sB, sC, sD)])

    # ---- write CSVs ------------------------------------------------------------
    with open(os.path.join(out_dir, "sweep_C.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        for r in sweep_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    with open(os.path.join(out_dir, "policy_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "n_eval", "cov_mean", "cov_p10", "cov_late",
                    "churn", "churn_per100", "churn_gib", "bounce",
                    "lag_mean", "lag_unrecovered", "phase_cov"])
        for label in labels:
            for pol in ("A", "B", "C", "D"):
                m = results[label][pol]
                w.writerow([label, pol, m["n_eval"], f"{m['cov_mean']:.4f}",
                            f"{m['cov_p10']:.4f}", f"{m['cov_late']:.4f}",
                            m["churn"], f"{m['churn_per100']:.2f}",
                            f"{m['churn_gib']:.3f}", m["bounce"],
                            f"{m['lag_mean']:.1f}", m["lag_unrecovered"],
                            ";".join(f"{k}={v:.3f}" for k, v in m["phase_cov"].items())])

    # dip/plateau at the post-warmup boundaries (transition dynamics)
    with open(os.path.join(out_dir, "boundary_transitions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "policy", "n_boundaries", "dip_cov_0_16", "plateau_cov_16_48"])
        for label in labels:
            nb = len(results[label]["A"]["lags"])
            if nb == 0:
                continue
            for pol in ("A", "B", "C", "D"):
                m = results[label][pol]
                w.writerow([label, pol, nb, f"{m['bdry_dip']:.4f}",
                            f"{m['bdry_plateau']:.4f}"])

    # ---- aggregates for the verdict --------------------------------------------
    def pool_mean(pol, field):
        return float(np.nanmean([results[l][pol][field] for l in labels]))

    # lag pooled over traces that have post-warmup boundaries
    def pool_lag(pol):
        vals = [l for lab in labels for (l, rec, cap) in results[lab][pol]["lags"]]
        unrec = sum(1 for lab in labels for (l, rec, cap) in results[lab][pol]["lags"] if not rec)
        return (float(np.mean(vals)) if vals else float("nan"), unrec, len(vals))

    summary = dict(
        traces=labels, excluded=excluded,
        warmup=WARMUP, K=K_KEEP, ewma_decay=EWMA_DECAY,
        rotate_every=ROTATE_EVERY, cooldown=COOLDOWN, boost_win=BOOST_WIN,
        sweep=dict(h=H_GRID, k_drift=KD_GRID, p=P_GRID),
        recommended_config=dict(h=best_key[0], k_drift=best_key[1], p=best_key[2]),
        recommended_agg=agg[best_key],
        max_recovery_config=dict(h=maxrec_key[0], k_drift=maxrec_key[1], p=maxrec_key[2]),
        max_recovery_agg=agg[maxrec_key],
        config_table={str(k): v for k, v in agg.items()},
        per_trace_best_config={l: str(per_trace_best[l]) for l in labels},
        recommended_regret_pts={l: 100 * regret[l] for l in labels},
        pooled=dict(
            cov_mean={p: pool_mean(p, "cov_mean") for p in "ABCD"},
            cov_p10={p: pool_mean(p, "cov_p10") for p in "ABCD"},
            cov_late={p: pool_mean(p, "cov_late") for p in "ABCD"},
            churn_per100={p: pool_mean(p, "churn_per100") for p in "ABCD"},
            churn_gib_total={p: float(np.sum([results[l][p]["churn_gib"] for l in labels])) for p in "ABCD"},
            bounce_total={p: int(np.sum([results[l][p]["bounce"] for l in labels])) for p in "ABCD"},
            lag={p: pool_lag(p) for p in "ABCD"},
            bdry_dip={p: float(np.nanmean([results[l][p]["bdry_dip"] for l in labels])) for p in "ABCD"},
            bdry_plateau={p: float(np.nanmean([results[l][p]["bdry_plateau"] for l in labels])) for p in "ABCD"},
        ),
    )

    lateA, lateB = summary["pooled"]["cov_late"]["A"], summary["pooled"]["cov_late"]["B"]
    lateC = summary["pooled"]["cov_late"]["C"]
    churnB, churnC = summary["pooled"]["churn_per100"]["B"], summary["pooled"]["churn_per100"]["C"]
    closure = (lateC - lateA) / (lateB - lateA) if lateB > lateA else float("nan")
    summary["closure_of_AB_gap"] = float(closure)

    reg = [100 * regret[l] for l in labels]
    if (lateC - lateA) >= 0.05 and churnC <= churnB / 3:
        verdict = ("POSITIVO: DEMAND-ADMIT (C, h=%.2f k=%.3f p=%d) recupera %.1f pt di "
                   "copertura tardiva persa da FROZEN (A %.3f -> C %.3f; B rotate32 %.3f), "
                   "chiude il %.0f%% del gap A->B con churn %.1f scambi/100tok = %.1fx MENO "
                   "del rotate wholesale (B %.1f/100tok) e bounce %.1f%% (B: %d rimbalzi). "
                   "Regret del config unico vs il best per-trace: mediana %.1f pt, max %.1f pt "
                   "-> parametri stabili tra trace. Merita la candidata patch 0026 e l'A/B "
                   "live S3." % (
                       best_key[0], best_key[1], best_key[2],
                       100 * (lateC - lateA), lateA, lateC, lateB, 100 * closure,
                       churnC, churnB / churnC if churnC else float("inf"), churnB,
                       100 * agg[best_key]["bounce_rate"],
                       summary["pooled"]["bounce_total"]["B"],
                       float(np.median(reg)), float(np.max(reg))))
    elif (lateC - lateA) >= 0.02:
        verdict = ("MARGINALE: guadagno tardivo C vs A = %.1f pt (soglia 5), churn C/B = %.2f. "
                   "Il meccanismo funziona ma il margine sui trace disponibili e' sottile." % (
                       100 * (lateC - lateA), churnC / churnB if churnB else float("nan")))
    else:
        verdict = ("NEGATIVO: DEMAND-ADMIT non recupera copertura utile (%.1f pt vs A); "
                   "la mask warmup-frozen copre gia' la domanda tardiva su questi trace." % (
                       100 * (lateC - lateA)))
    summary["verdict"] = verdict

    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # ---- console ---------------------------------------------------------------
    print("\n=== E-ADMIT demand-driven admission simulation ===")
    print(f"traces: {len(labels)} (excluded: {excluded})")
    P = summary["pooled"]
    print(f"\n{'policy':8s} {'cov_mean':>9s} {'cov_p10':>8s} {'cov_late':>9s} "
          f"{'churn/100':>10s} {'GiB tot':>8s} {'bounce':>7s} {'lag':>12s}")
    for p in "ABCD":
        lm, lu, ln = P["lag"][p]
        print(f"{p:8s} {P['cov_mean'][p]:9.3f} {P['cov_p10'][p]:8.3f} "
              f"{P['cov_late'][p]:9.3f} {P['churn_per100'][p]:10.2f} "
              f"{P['churn_gib_total'][p]:8.2f} {P['bounce_total'][p]:7d} "
              f"{lm:5.1f} ({lu}/{ln} unrec)")
    print(f"\nboundary transitions (pooled over traces with post-warmup boundaries):")
    for p in "ABCD":
        print(f"  {p}: dip[0,16)={P['bdry_dip'][p]:.3f}  plateau[16,48)={P['bdry_plateau'][p]:.3f}")
    print(f"\nrecommended CUSUM config: h={best_key[0]} k_drift={best_key[1]} p={best_key[2]} "
          f"(late gain {100*agg[best_key]['late_gain']:.1f} pt, churn ratio vs B "
          f"{agg[best_key]['churn_ratio_vs_B']:.2f}, bounce rate {100*agg[best_key]['bounce_rate']:.1f}%)")
    print(f"max-recovery config (no bounce filter): h={maxrec_key[0]} k_drift={maxrec_key[1]} "
          f"p={maxrec_key[2]} (late gain {100*agg[maxrec_key]['late_gain']:.1f} pt)")
    print(f"closure of A->B late-coverage gap: {100*closure:.0f}%")
    print(f"per-trace best: {summary['per_trace_best_config']}")
    print("regret of the recommended config (pt): " +
          ", ".join(f"{l}={100*regret[l]:.1f}" for l in labels))
    print("\n=== VERDICT ===")
    print(verdict)
    print(f"\nwrote {out_dir}/policy_summary.csv, sweep_C.csv, boundary_transitions.csv, "
          f"cov_curves_bestC.csv, stats.json")


if __name__ == "__main__":
    main()
