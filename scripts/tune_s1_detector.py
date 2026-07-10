#!/usr/bin/env python3
"""E-DET: offline tuning of the S1 onset detector (patch 0020 family).

Goal: reduce onset-detection latency from ~50-60 tok (current design:
EWMA alpha=0.10 + slope over win=64 + stable=16) to ~10-20 tok, exploiting the
cost asymmetry of the controller ladder (rotate = ~free with 0021
delta-prefetch, rewind = ~10-20 tok margin, stopper = last resort).

ONLY offline analysis on recorded data. No GPU, no pod.

SCOPE (E-DET v2, binding). The S1-slope early warning only buys lead time in the
SLOW-EROSION regime: a mild/wide STATIC mask (K91-family) that drifts up over
hundreds of tokens (0.845 -> ~0.895, onset ~190 tok before the text lock). In the
AGGRESSIVE dynamic regime (W50+K23+rotate32) S1 is PINNED FLAT ~0.815 from the
instant the mask engages and the output loops almost at once (~gen 126) with NO
S1 lead. This detector is tuned for, and only claimed on, the slow-erosion regime
(runs/ds4/20260710_scope_divergence_pod/README.md). The aggressive pod series is
used here ONLY as an adversarial false-alarm control (a flat-but-high series a
slope/CUSUM/vote detector must NOT fire on).

Real inputs (read-only):
  A) s1_sensor.csv  (patch 0012, LIVE per-layer sensor: pos,layer,pruned_mass,
     total_mass; 40 layers, pos 138..2736) from the moe-aggressive-commit k91
     worktree, run `gen_s1` (pod 3090, ctx3072). Mask empty until pos~294,
     then K91-family mask ON (S1 jumps 0 -> ~0.82). Text (gen_s1.log) loops
     `// <cjk>` from char-proportional pos ~2476 (first marker; GROUND-TRUTH
     collapse), exact repetition-lock ~2508. This is the PRIMARY slow-erosion
     onset series (the scope repo's `k91_collapse.divergence` scene is this CSV).
  B) trace_k0.csv (patch 0006 top-6 routing trace of the FULL model, 799 tok)
     + reap_mask_coding_k91.json -> counterfactual S1 proxy per token/layer
     (mass of full-model top-6 falling on k91-pruned experts). Healthy series
     (K0 "router libero" never degenerates) for false-alarm measurement.
  C) runs/ds4/20260710_scope_divergence_pod/r1/s1_r1.csv.gz  (patch 0012 sensor,
     AGGRESSIVE W50+K23+rotate32 pod run; 98k rows, 40 layers, pos 128..2577).
     Aggregate S1 FLAT ~0.815, no drift. Used as the ADVERSARIAL false-alarm
     control for the aggressive regime (per-layer, so it also stresses the vote).
  D) runs/ds4/20260710_pod_smoke_0020_0021/s1trig.jsonl: sparse real events of
     the new 0020 code (forced params thr=1e-6 win=16 stable=4) - used as a
     sanity anchor for the replay semantics, not as a series.

Because only ONE real collapse-labeled series exists (A), the delay-from-onset
distribution (p50/p90) is estimated on SYNTHETIC series calibrated on (A): AR(1)
noise matched to the healthy-regime per-token std and lag-1 autocorrelation, plus
ramps of measured amplitude (+0.04..0.08 over 100-300 tok; provenance
CLAIM-011 +0.058/~200tok and the local sensor ramp +0.05/~300tok). On the real
series (A) we additionally report the single measured LEAD = ground-truth text
lock (2476) - first collapse-region fire (defensible, non-circular: the deadline
is the text lock, not the baseline-detected onset).

Usage:
  python scripts/tune_s1_detector.py \
      --k91-root "C:/Users/imanu/source/repos/moe-aggressive-commit/.claude/worktrees/elastic-bose-6ae1c7/runs/reap/k91_coding_vram" \
      --trace-k0 <extracted trace_k0.csv> \
      --out runs/ds4/20260710_edet_s1_detector_tuning [--quick]

trace_k0.csv is inside <k91-root>/loop/loop_traces.tgz; if --trace-k0 is not
given the script extracts it to a temp dir automatically.

Outputs in --out: calibration.json, real_series_eval.csv, curves_*.csv,
summary_table.md, derived series s1_sensor_agg.csv / s1_k0proxy_agg.csv.
Deterministic (seed 20260710).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import tarfile
import tempfile
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _open_text(path):
    """Open a possibly gzipped CSV as a text stream."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")

# ----------------------------------------------------------------------------
# Constants: labels on the real collapse series (provenance in the docstring)
# ----------------------------------------------------------------------------
SENSOR_MASK_ON = 294      # first pos with aggregate S1 > 0.1 (mask flip k91)
SENSOR_HEALTHY = (350, 2250)   # text confirmed clean until >=2476
SENSOR_TEXT_FIRST = 2476  # first loop-marker occurrence (GROUND-TRUTH collapse)
SENSOR_TEXT_LOCK = 2508   # exact repetition-lock (char-proportional)
# A collapse-region fire (used for the real LEAD metric) is the first fire past
# the healthy plateau; fires before are counted as false alarms. Fires after the
# text lock give a non-positive lead (too late).
COLLAPSE_REGION = (SENSOR_HEALTHY[1], 2577)  # (2250, end-of-sensor]
POD_R1 = os.path.join(REPO_ROOT, "runs", "ds4",
                      "20260710_scope_divergence_pod", "r1", "s1_r1.csv.gz")
SEED = 20260710


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_sensor_csv(path):
    """patch-0012 CSV -> (positions, agg S1 list, {layer: list aligned})."""
    pruned = defaultdict(float)
    total = defaultdict(float)
    per_layer = defaultdict(dict)
    with _open_text(path) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            pos, lay = int(row[0]), int(row[1])
            a, b = float(row[2]), float(row[3])
            pruned[pos] += a
            total[pos] += b
            per_layer[lay][pos] = a / b if b > 0 else 0.0
    poss = sorted(pruned)
    agg = [pruned[q] / total[q] if total[q] > 0 else 0.0 for q in poss]
    layers = {
        lay: [per_layer[lay].get(q, 0.0) for q in poss]
        for lay in sorted(per_layer)
    }
    return poss, agg, layers


def load_k0_proxy(trace_csv, mask_json):
    """0006 top-6 trace of the FULL model + k91 keep -> counterfactual S1."""
    mask = json.load(open(mask_json))
    keep = {int(l): set(v) for l, v in mask["keep"].items()}
    pruned = defaultdict(float)
    total = defaultdict(float)
    per_layer_p = defaultdict(lambda: defaultdict(float))
    per_layer_t = defaultdict(lambda: defaultdict(float))
    with open(trace_csv, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        iw0 = header.index("w0")
        for row in rd:
            try:
                pos, lay, n = int(row[0]), int(row[1]), int(row[2])
            except ValueError:
                continue
            if lay not in keep:
                continue
            for s in range(n):
                try:
                    e = int(row[3 + s])
                    w = float(row[iw0 + s])
                except (ValueError, IndexError):
                    continue
                if e < 0 or w != w:
                    continue
                total[pos] += w
                per_layer_t[lay][pos] += w
                if e not in keep[lay]:
                    pruned[pos] += w
                    per_layer_p[lay][pos] += w
    poss = sorted(total)
    agg = [pruned[q] / total[q] if total[q] > 0 else 0.0 for q in poss]
    layers = {}
    for lay in sorted(per_layer_t):
        layers[lay] = [
            (per_layer_p[lay][q] / per_layer_t[lay][q])
            if per_layer_t[lay][q] > 0 else 0.0
            for q in poss
        ]
    return poss, agg, layers


# ----------------------------------------------------------------------------
# Calibration helpers
# ----------------------------------------------------------------------------
def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def lag1_rho(xs):
    mu = sum(xs) / len(xs)
    num = sum((xs[i] - mu) * (xs[i + 1] - mu) for i in range(len(xs) - 1))
    den = sum((x - mu) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def fit_ramp_start(poss, agg, lo, hi, base_mu):
    """Least-squares flat-then-linear changepoint on [lo, hi] -> ramp t0."""
    idx = [i for i, q in enumerate(poss) if lo <= q <= hi]
    best_t0, best_sse = None, float("inf")
    for j in range(len(idx) - 20):
        t0i = idx[j]
        t0 = poss[t0i]
        # slope fitted to the points after t0
        pts = [(poss[i] - t0, agg[i] - base_mu) for i in idx[j:]]
        sxx = sum(p * p for p, _ in pts)
        sxy = sum(p * y for p, y in pts)
        b = sxy / sxx if sxx > 0 else 0.0
        if b <= 0:
            continue
        sse = sum((agg[i] - base_mu) ** 2 for i in idx[:j])
        sse += sum((y - b * p) ** 2 for p, y in pts)
        if sse < best_sse:
            best_sse, best_t0 = sse, (t0, b)
    return best_t0  # (pos, slope per token)


# ----------------------------------------------------------------------------
# Synthetic series (calibrated)
# ----------------------------------------------------------------------------
def gen_ar1(rng, n, mu, sigma, rho):
    innov_sd = sigma * math.sqrt(max(1e-12, 1.0 - rho * rho))
    x = [0.0] * n
    prev = rng.gauss(0.0, sigma)
    for i in range(n):
        prev = rho * prev + rng.gauss(0.0, innov_sd)
        x[i] = mu + prev
    return x

def make_synth(rng, n_series, length, mu, sigma, rho, collapse):
    """Returns list of (series, onset or None). Ramp amp U(0.04,0.08) over
    U(100,300) tok (25%: fast step over 20-50 tok), then plateau."""
    out = []
    for _ in range(n_series):
        x = gen_ar1(rng, length, mu, sigma, rho)
        onset = None
        if collapse:
            onset = rng.randrange(int(length * 0.25), int(length * 0.65))
            amp = rng.uniform(0.04, 0.08)
            dur = rng.randrange(20, 51) if rng.random() < 0.25 \
                else rng.randrange(100, 301)
            for i in range(onset, length):
                f = min(1.0, (i - onset) / dur)
                x[i] += amp * f
        out.append((x, onset))
    return out


def make_synth_layers(rng, n_series, length, lay_mu, lay_sigma, rho, rbar,
                      lay_beta, collapse):
    """Factor-model per-layer synth: x_l = mu_l + lambda_l*common + idio_l
    (+ beta_l * ramp). Average pairwise correlation matched to rbar."""
    out = []
    L = len(lay_mu)
    for _ in range(n_series):
        common = gen_ar1(rng, length, 0.0, 1.0, rho)
        onset = None
        ramp = [0.0] * length
        if collapse:
            onset = rng.randrange(int(length * 0.25), int(length * 0.65))
            amp_scale = rng.uniform(0.6, 1.3)   # scales the measured betas
            dur = rng.randrange(100, 301)
            for i in range(onset, length):
                ramp[i] = amp_scale * min(1.0, (i - onset) / dur)
        layers = []
        for l in range(L):
            lam = lay_sigma[l] * math.sqrt(max(0.0, rbar))
            id_sd = lay_sigma[l] * math.sqrt(max(1e-12, 1.0 - rbar))
            idio = gen_ar1(rng, length, 0.0, id_sd, rho)
            xs = [lay_mu[l] + lam * common[i] + idio[i] + lay_beta[l] * ramp[i]
                  for i in range(length)]
            layers.append(xs)
        out.append((layers, onset))
    return out


# ----------------------------------------------------------------------------
# Detectors — each returns the list of fire indices (0-based token index)
# ----------------------------------------------------------------------------
def det_slope_0020(x, alpha, win, thr, stable, ema_init_first=False,
                   start=0):
    """Exact replay of patch-0020 ds4_pace_s1_update semantics.

    ema starts at 0.0 (as coded) unless ema_init_first=True (proposed fix:
    seed the EWMA with the first sample). Refractory: one window after fire.
    `start` marks the first index at which the detector runs (e.g. mask-on).
    """
    fires = []
    ema = None if ema_init_first else 0.0
    ring = []
    head = 0
    over = 0
    last_fire = start  # s1_last_fire_tok starts at 0 relative to reset
    for t in range(start, len(x)):
        s1 = x[t]
        if ema is None:
            ema = s1
        ema = (1.0 - alpha) * ema + alpha * s1
        if len(ring) < win:
            ring.append(ema)
            slope = 0.0
        else:
            ring[head] = ema
            head = (head + 1) % win
            oldest = ring[head]
            slope = (ema - oldest) / (win - 1)
        if len(ring) >= win and (t - last_fire) >= win and slope >= thr:
            over += 1
        else:
            over = 0
        if over >= stable:
            fires.append(t)
            last_fire = t
            over = 0
    return fires


def det_cusum(x, k_sigma, h_sigma, pre_alpha=None, base_lag=32,
              base_win=128, min_base=32, calib_win=128, refractory=64,
              start=0):
    """One-sided CUSUM on x (raw or lightly-EWMA'd if pre_alpha).

    Reference mean = LAGGED moving average of the transformed signal over
    y[t-base_lag-base_win : t-base_lag]. The lag keeps the onset ramp out of
    the baseline for `base_lag` tokens (enough to detect a 100-300 tok ramp)
    while still tracking the benign slow drift the real sensor shows
    (0.85 -> 0.89 over ~2000 tok).

    sigma is SELF-CALIBRATED from the first `calib_win` transformed samples
    of the run (live-implementable: a calibration window after prefill).
    Detection starts after `min_base` baseline samples AND calibration.
    """
    fires = []
    g = 0.0
    y_ema = None
    last_fire = -10 ** 9
    sigma = None
    ys = []          # transformed samples (index-aligned to t - start)
    pref = [0.0]     # prefix sums of ys
    for t in range(start, len(x)):
        y = x[t]
        if pre_alpha is not None:
            y_ema = y if y_ema is None else (1 - pre_alpha) * y_ema + pre_alpha * y
            y = y_ema
        i = t - start
        if sigma is None and i >= calib_win:
            sigma = max(1e-4, mean_std(ys[:calib_win])[1])
        # baseline over ys[a:b] with b = i - base_lag (exclusive), lagged
        b = i - base_lag
        a = max(0, b - base_win)
        if sigma is not None and b - a >= min_base:
            mu = (pref[b] - pref[a]) / (b - a)
            g = max(0.0, g + (y - mu - k_sigma * sigma))
            if g > h_sigma * sigma and (t - last_fire) >= refractory:
                fires.append(t)
                g = 0.0
                last_fire = t
        ys.append(y)
        pref.append(pref[-1] + y)
    return fires


def det_vote_kofn(layers, sigmas, k_sigma, h_sigma, K, base_lag=32,
                  base_win=128, min_base=32, refractory=64, start=0):
    """Per-layer one-sided CUSUM (lagged-window baseline, same scheme as
    det_cusum) + k-of-N vote: fire when >= K layers have g_l > h*sigma_l."""
    L = len(layers)
    n = len(layers[0])
    g = [0.0] * L
    pref = [[0.0] for _ in range(L)]
    fires = []
    last_fire = -10 ** 9
    for t in range(start, n):
        i = t - start
        b = i - base_lag
        a = max(0, b - base_win)
        ready = (b - a) >= min_base
        count = 0
        for l in range(L):
            y = layers[l][t]
            if ready:
                mu = (pref[l][b] - pref[l][a]) / (b - a)
                g[l] = max(0.0, g[l] + (y - mu - k_sigma * sigmas[l]))
                if g[l] > h_sigma * sigmas[l]:
                    count += 1
            pref[l].append(pref[l][-1] + y)
        if ready and count >= K and (t - last_fire) >= refractory:
            fires.append(t)
            g = [0.0] * L
            last_fire = t
    return fires


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def eval_agg_config(name, runner, synth_collapse, synth_healthy,
                    real_healthy_segs, horizon=600):
    """runner(series, start) -> fires. Returns metrics dict."""
    delays, misses, pre_fires, pre_tokens = [], 0, 0, 0
    for x, onset in synth_collapse:
        fires = runner(x, 0)
        pre = [f for f in fires if f < onset]
        pre_fires += len(pre)
        pre_tokens += onset
        post = [f for f in fires if onset <= f <= onset + horizon]
        if post:
            delays.append(post[0] - onset)
        else:
            misses += 1
    fa_fires, fa_tokens = pre_fires, pre_tokens
    for x, _ in synth_healthy:
        fires = runner(x, 0)
        fa_fires += len(fires)
        fa_tokens += len(x)
    real_fa, real_tok = 0, 0
    for x, start in real_healthy_segs:
        fires = runner(x, start)
        real_fa += len(fires)
        real_tok += len(x) - start
    dsort = sorted(delays)
    def pct(p):
        if not dsort:
            return None
        i = min(len(dsort) - 1, int(math.ceil(p * len(dsort))) - 1)
        return dsort[i]
    n_c = len(synth_collapse)
    return {
        "name": name,
        "delay_p50": pct(0.50),
        "delay_p90": pct(0.90),
        "miss_rate": misses / n_c,
        "fa_per_1k_synth": 1000.0 * fa_fires / fa_tokens if fa_tokens else None,
        "fa_per_1k_real": 1000.0 * real_fa / real_tok if real_tok else None,
        "real_fa_fires": real_fa,
        "real_fa_tokens": real_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    default_root = (r"C:\Users\imanu\source\repos\moe-aggressive-commit"
                    r"\.claude\worktrees\elastic-bose-6ae1c7\runs\reap"
                    r"\k91_coding_vram")
    ap.add_argument("--k91-root", default=default_root)
    ap.add_argument("--trace-k0", default=None,
                    help="extracted trace_k0.csv (default: auto-extract from "
                         "<k91-root>/loop/loop_traces.tgz)")
    ap.add_argument("--pod-r1", default=POD_R1,
                    help="aggressive-regime pod S1 sensor (gz ok); adversarial "
                         "false-alarm control. Set to '' to skip.")
    ap.add_argument("--out", default="runs/ds4/20260710_edet_s1_detector_tuning")
    ap.add_argument("--quick", action="store_true",
                    help="smaller synthetic ensembles (smoke)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    root = args.k91_root
    sensor_csv = os.path.join(root, "loop", "s1_sensor.csv")
    mask_json = os.path.join(root, "reap_mask_coding_k91.json")

    # ---------------- load real series ----------------
    poss, agg, layers = load_sensor_csv(sensor_csv)
    pos_index = {q: i for i, q in enumerate(poss)}

    trace_k0 = args.trace_k0
    tmpdir = None
    if trace_k0 is None:
        tgz = os.path.join(root, "loop", "loop_traces.tgz")
        tmpdir = tempfile.mkdtemp(prefix="edet_")
        with tarfile.open(tgz) as tf:
            tf.extract("trace_k0.csv", tmpdir)
        trace_k0 = os.path.join(tmpdir, "trace_k0.csv")
    k0_poss, k0_agg, k0_layers = load_k0_proxy(trace_k0, mask_json)

    # ---- aggressive-regime pod series (adversarial false-alarm control) ----
    pod_poss = pod_agg = None
    pod_layers = {}
    pod_stats = None
    if args.pod_r1 and os.path.exists(args.pod_r1):
        pod_poss, pod_agg, pod_layers = load_sensor_csv(args.pod_r1)
        # full-run slope via least squares (per token) to show it is flat
        n = len(pod_agg)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(pod_agg) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((xs[i] - mx) * (pod_agg[i] - my) for i in range(n))
        pod_slope = sxy / sxx if sxx > 0 else 0.0
        pod_mu, pod_sd = mean_std(pod_agg)
        pod_stats = {
            "path": args.pod_r1,
            "tokens": n,
            "pos_range": [pod_poss[0], pod_poss[-1]],
            "n_layers": len(pod_layers),
            "mean": round(pod_mu, 5),
            "std_per_token": round(pod_sd, 5),
            "lag1_autocorr": round(lag1_rho(pod_agg), 4),
            "full_run_slope_per_tok": round(pod_slope, 8),
        }
        with open(os.path.join(args.out, "s1_pod_r1_agg.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pos", "s1"])
            for q, v in zip(pod_poss, pod_agg):
                w.writerow([q, f"{v:.6f}"])
    else:
        print("WARN: pod r1 series not found (%s); skipping the aggressive-"
              "regime false-alarm control." % args.pod_r1, file=sys.stderr)

    # export derived aggregates for reproducibility
    with open(os.path.join(args.out, "s1_sensor_agg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "s1"])
        for q, v in zip(poss, agg):
            w.writerow([q, f"{v:.6f}"])
    with open(os.path.join(args.out, "s1_k0proxy_agg.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "s1_counterfactual"])
        for q, v in zip(k0_poss, k0_agg):
            w.writerow([q, f"{v:.6f}"])

    # ---------------- calibration ----------------
    h_lo, h_hi = SENSOR_HEALTHY
    healthy_idx = [i for i, q in enumerate(poss) if h_lo <= q <= h_hi]
    healthy = [agg[i] for i in healthy_idx]
    mu0, sd0 = mean_std(healthy)
    rho0 = lag1_rho(healthy)

    ramp = fit_ramp_start(poss, agg, 2100, 2736, mu0)
    ramp_t0, ramp_slope = ramp if ramp else (None, None)

    # per-layer healthy stats + collapse betas (mean tail - healthy mean)
    lay_ids = sorted(layers)
    lay_mu, lay_sd, lay_beta = [], [], []
    tail_idx = [i for i, q in enumerate(poss) if q >= 2550]
    for lay in lay_ids:
        xs = [layers[lay][i] for i in healthy_idx]
        m, s = mean_std(xs)
        tail_m = sum(layers[lay][i] for i in tail_idx) / len(tail_idx)
        lay_mu.append(m)
        lay_sd.append(s)
        lay_beta.append(tail_m - m)
    # average pairwise correlation via correlation with the aggregate
    # (factor-model estimate: rbar ~ mean corr(layer, agg)^2)
    rbars = []
    agg_h = healthy
    mu_a = mu0
    den_a = sum((a - mu_a) ** 2 for a in agg_h)
    for li, lay in enumerate(lay_ids):
        xs = [layers[lay][i] for i in healthy_idx]
        m = lay_mu[li]
        num = sum((xs[j] - m) * (agg_h[j] - mu_a) for j in range(len(xs)))
        den_l = sum((x - m) ** 2 for x in xs)
        if den_l > 0 and den_a > 0:
            r = num / math.sqrt(den_l * den_a)
            rbars.append(r * r)
    rbar = sum(rbars) / len(rbars)

    calib = {
        "sensor_tokens": len(poss),
        "sensor_pos_range": [poss[0], poss[-1]],
        "mask_on_pos": SENSOR_MASK_ON,
        "healthy_segment": list(SENSOR_HEALTHY),
        "healthy_mean": round(mu0, 5),
        "healthy_std_per_token": round(sd0, 5),
        "healthy_lag1_autocorr": round(rho0, 4),
        "ramp_fit_t0_pos": ramp_t0,
        "ramp_fit_slope_per_tok": round(ramp_slope, 8) if ramp_slope else None,
        "text_first_marker_pos": SENSOR_TEXT_FIRST,
        "text_repetition_lock_pos": SENSOR_TEXT_LOCK,
        "k0_proxy_tokens": len(k0_poss),
        "k0_proxy_mean": round(sum(k0_agg) / len(k0_agg), 5),
        "k0_proxy_std": round(mean_std(k0_agg)[1], 5),
        "per_layer_rbar": round(rbar, 4),
        "per_layer_beta_mean": round(sum(lay_beta) / len(lay_beta), 5),
        "collapse_region": list(COLLAPSE_REGION),
        "pod_r1_aggressive": pod_stats,
        "seed": SEED,
    }
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print("calibration:", json.dumps(calib, indent=2))

    # ---------------- synthetic ensembles ----------------
    rng = random.Random(SEED)
    N = 30 if args.quick else 80
    LEN = 3000
    synth_c = make_synth(rng, N, LEN, mu0, sd0, rho0, collapse=True)
    synth_h = make_synth(rng, N, LEN, mu0, sd0, rho0, collapse=False)

    # real healthy segments: (series, start_index)
    # sensor healthy: run detector from mask-on, count fires inside [350,2250]
    sensor_healthy_series = [agg[i] for i in range(pos_index[SENSOR_MASK_ON],
                                                   healthy_idx[-1] + 1)]
    real_healthy = [
        (sensor_healthy_series, 0),
        (k0_agg, 0),
    ]

    # real collapse eval: run on full masked series, look at fires >= 2250
    sensor_masked = agg[pos_index[SENSOR_MASK_ON]:]
    sensor_masked_pos0 = SENSOR_MASK_ON  # poss offset ~ +1/token (contiguous)

    def real_collapse_fires(runner):
        fires = runner(sensor_masked, 0)
        # map index -> pos (positions are contiguous per token)
        return [poss[pos_index[SENSOR_MASK_ON]] + i for i in
                (f for f in fires)]

    # aggressive-regime pod: per-layer arrays + self-calibrated per-layer sigma
    # (the run is flat, so its own per-token std IS its noise scale). Used as an
    # adversarial false-alarm control for both aggregate and vote detectors.
    pod_lay = None
    pod_lay_sd = None
    if pod_agg is not None:
        pod_lay = [pod_layers[l] for l in lay_ids if l in pod_layers]
        pod_lay_sd = [max(1e-4, mean_std(xs)[1]) if len(xs) > 1 else 1e-4
                      for xs in pod_lay]

    def pod_fa_agg(runner):
        """fires/1k on the aggressive pod aggregate; None if pod unavailable."""
        if pod_agg is None:
            return None
        fires = runner(pod_agg, 0)
        return 1000.0 * len(fires) / len(pod_agg)

    def real_k91_lead(rc_fires):
        """(first collapse-region fire pos, lead = text_lock(2476) - pos).

        Non-circular: the deadline is the ground-truth text repetition-lock, not
        the baseline-detected onset. Positive lead = warned before the text
        visibly loops; <=0 = too late. Fires in [mask_on, 2250] are FALSE alarms
        (accounted separately) and are ignored here.
        """
        for p in (rc_fires or []):
            if COLLAPSE_REGION[0] < p <= COLLAPSE_REGION[1]:
                return p, SENSOR_TEXT_FIRST - p
        return None, None

    # ---------------- configs ----------------
    results = []
    curve_rows = []

    slope_variants = [
        # (label, alpha, win, stable, init_first)
        ("slope_base_a10_w64_s16_init0", 0.10, 64, 16, False),
        ("slope_base_a10_w64_s16_initx0", 0.10, 64, 16, True),
        ("slope_a10_w64_s8_initx0", 0.10, 64, 8, True),
        ("slope_a15_w32_s6_initx0", 0.15, 32, 6, True),
        ("slope_a20_w16_s4_initx0", 0.20, 16, 4, True),
        ("slope_a20_w16_s8_initx0", 0.20, 16, 8, True),
        ("slope_a20_w24_s6_initx0", 0.20, 24, 6, True),
        ("slope_a20_w32_s8_initx0", 0.20, 32, 8, True),
        ("slope_a30_w16_s4_initx0", 0.30, 16, 4, True),
    ]
    thr_grid = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3, 2e-3]

    for label, alpha, win, stable, initx in slope_variants:
        for thr in thr_grid:
            def runner(x, start, a=alpha, w=win, t=thr, s=stable, ix=initx):
                return det_slope_0020(x, a, w, t, s, ema_init_first=ix,
                                      start=start)
            m = eval_agg_config(f"{label}_thr{thr:g}", runner, synth_c,
                                synth_h, real_healthy)
            m["family"] = "slope"
            m["params"] = {"alpha": alpha, "win": win, "stable": stable,
                           "thr": thr, "ema_init_first": initx}
            m["real_collapse_fires"] = real_collapse_fires(runner)
            m["fa_per_1k_pod"] = pod_fa_agg(runner)
            m["real_k91_first_fire"], m["real_k91_lead"] = \
                real_k91_lead(m["real_collapse_fires"])
            results.append(m)
            curve_rows.append(m)
            print(f"{m['name']:44s} p50={m['delay_p50']} p90={m['delay_p90']} "
                  f"miss={m['miss_rate']:.2f} "
                  f"FA/1k synth={m['fa_per_1k_synth']:.2f} "
                  f"real={m['fa_per_1k_real']:.2f}")

    # CUSUM: sigma of the transformed healthy signal, from the REAL series
    def transformed_sigma(pre_alpha):
        if pre_alpha is None:
            return sd0
        y = None
        out = []
        for v in healthy:
            y = v if y is None else (1 - pre_alpha) * y + pre_alpha * v
            out.append(y)
        return mean_std(out[32:])[1]

    cusum_variants = [
        # (label, pre_alpha, base_lag, base_win)
        ("cusum_raw", None, 32, 128),
        ("cusum_ewma15", 0.15, 32, 128),
        ("cusum_ewma30", 0.30, 32, 128),
        ("cusum_ewma50", 0.50, 32, 128),
        ("cusum_ewma30_fast", 0.30, 16, 96),
    ]
    h_grid = [2, 3, 4, 6, 8, 12, 16, 24]
    for label, pre_alpha, base_lag, base_win in cusum_variants:
        sig = transformed_sigma(pre_alpha)
        for k_sigma in (0.25, 0.5, 1.0):
            for h_sigma in h_grid:
                def runner(x, start, k=k_sigma, h=h_sigma, p=pre_alpha,
                           bl=base_lag, bw=base_win):
                    # det_cusum self-calibrates sigma; pass k_sigma,h_sigma only
                    return det_cusum(x, k, h, pre_alpha=p, base_lag=bl,
                                     base_win=bw, start=start)
                m = eval_agg_config(
                    f"{label}_k{k_sigma:g}_h{h_sigma:g}", runner,
                    synth_c, synth_h, real_healthy)
                m["family"] = label
                m["params"] = {"k_sigma": k_sigma, "h_sigma": h_sigma,
                               "pre_alpha": pre_alpha, "base_lag": base_lag,
                               "base_win": base_win, "sigma_cal": round(sig, 5)}
                m["real_collapse_fires"] = real_collapse_fires(runner)
                m["fa_per_1k_pod"] = pod_fa_agg(runner)
                m["real_k91_first_fire"], m["real_k91_lead"] = \
                    real_k91_lead(m["real_collapse_fires"])
                results.append(m)
                curve_rows.append(m)
                print(f"{m['name']:44s} p50={m['delay_p50']} "
                      f"p90={m['delay_p90']} miss={m['miss_rate']:.2f} "
                      f"FA/1k synth={m['fa_per_1k_synth']:.2f} "
                      f"real={m['fa_per_1k_real']:.2f}")

    # ---------------- per-layer k-of-N vote ----------------
    # real per-layer series (sensor, masked segment) + factor-model synth
    start_i = pos_index[SENSOR_MASK_ON]
    real_lay = [[layers[l][i] for i in range(start_i, len(poss))]
                for l in lay_ids]
    k0_lay = [k0_layers[l] for l in sorted(k0_layers)]
    k0_lay_sd = []
    for xs in k0_lay:
        k0_lay_sd.append(mean_std(xs[64:])[1])

    Nv = 12 if args.quick else 30
    synth_lc = make_synth_layers(rng, Nv, LEN, lay_mu, lay_sd, rho0, rbar,
                                 lay_beta, collapse=True)
    synth_lh = make_synth_layers(rng, Nv, LEN, lay_mu, lay_sd, rho0, rbar,
                                 lay_beta, collapse=False)

    vote_rows = []
    for k_sigma in (0.25, 0.5):
      for K in (4, 8, 12, 16):
        for h_sigma in (2, 4, 6, 8, 12):
            delays, misses, fa_f, fa_t = [], 0, 0, 0
            for lays, onset in synth_lc:
                fires = det_vote_kofn(lays, lay_sd, k_sigma, h_sigma, K)
                fa_f += len([f for f in fires if f < onset])
                fa_t += onset
                post = [f for f in fires if onset <= f <= onset + 600]
                if post:
                    delays.append(post[0] - onset)
                else:
                    misses += 1
            for lays, _ in synth_lh:
                fires = det_vote_kofn(lays, lay_sd, k_sigma, h_sigma, K)
                fa_f += len(fires)
                fa_t += LEN
            # real healthy: sensor masked-healthy portion; count fires with
            # pos <= 2250. real collapse fires: pos > 2250.
            fires = det_vote_kofn(real_lay, lay_sd, k_sigma, h_sigma, K)
            fpos = [poss[start_i] + f for f in fires]
            real_h_f = len([p for p in fpos if p <= SENSOR_HEALTHY[1]])
            real_c_f = [p for p in fpos if p > SENSOR_HEALTHY[1]]
            # k0 proxy healthy (per-layer sigmas of the proxy itself)
            fires_k0 = det_vote_kofn(k0_lay, k0_lay_sd, k_sigma, h_sigma, K)
            # aggressive pod adversarial FA (per-layer, self-calibrated sigmas)
            pod_fa = None
            if pod_lay is not None:
                pfires = det_vote_kofn(pod_lay, pod_lay_sd, k_sigma, h_sigma, K)
                pod_fa = 1000.0 * len(pfires) / len(pod_lay[0])
            ff, lead = real_k91_lead(real_c_f)
            dsort = sorted(delays)
            def pct(p):
                if not dsort:
                    return None
                i = min(len(dsort) - 1, int(math.ceil(p * len(dsort))) - 1)
                return dsort[i]
            row = {
                "name": f"vote_{K}of40_k{k_sigma:g}_h{h_sigma:g}",
                "family": "vote_kofN",
                "params": {"K": K, "k_sigma": k_sigma, "h_sigma": h_sigma},
                "delay_p50": pct(0.50), "delay_p90": pct(0.90),
                "miss_rate": misses / max(1, len(synth_lc)),
                "fa_per_1k_synth": 1000.0 * fa_f / fa_t if fa_t else None,
                "fa_per_1k_real": 1000.0 * (real_h_f + len(fires_k0)) /
                                  ((SENSOR_HEALTHY[1] - SENSOR_MASK_ON) +
                                   len(k0_lay[0])),
                "real_fa_fires": real_h_f + len(fires_k0),
                "real_fa_tokens": (SENSOR_HEALTHY[1] - SENSOR_MASK_ON) +
                                  len(k0_lay[0]),
                "fa_per_1k_pod": pod_fa,
                "real_collapse_fires": real_c_f,
                "real_k91_first_fire": ff,
                "real_k91_lead": lead,
            }
            results.append(row)
            vote_rows.append(row)
            print(f"{row['name']:44s} p50={row['delay_p50']} "
                  f"p90={row['delay_p90']} miss={row['miss_rate']:.2f} "
                  f"FA/1k synth={row['fa_per_1k_synth']:.2f} "
                  f"real={row['fa_per_1k_real']:.2f}")

    # ---------------- outputs ----------------
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    with open(os.path.join(args.out, "curves_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "family", "params", "delay_p50", "delay_p90",
                    "miss_rate", "fa_per_1k_synth", "fa_per_1k_real",
                    "fa_per_1k_pod_aggressive", "real_fa_fires",
                    "real_fa_tokens", "real_collapse_first_fire_pos",
                    "real_k91_lead_vs_lock"])
        for m in results:
            rc = m.get("real_collapse_fires") or []
            rc_first = ""
            for p in rc:
                if p > SENSOR_HEALTHY[1]:
                    rc_first = p
                    break
            w.writerow([m["name"], m["family"], json.dumps(m["params"]),
                        fmt(m["delay_p50"]), fmt(m["delay_p90"]),
                        fmt(m["miss_rate"]), fmt(m["fa_per_1k_synth"]),
                        fmt(m["fa_per_1k_real"]), fmt(m.get("fa_per_1k_pod")),
                        m.get("real_fa_fires", ""),
                        m.get("real_fa_tokens", ""), rc_first,
                        fmt(m.get("real_k91_lead"))])

    # ---------------- recommendation: two operating profiles ----------------
    def _delay(m):
        v = m.get("delay_p50")
        return v if v is not None else 10 ** 9

    def _rfa(m):
        v = m.get("fa_per_1k_real")
        return v if v is not None else 10 ** 9

    def _pfa(m):
        v = m.get("fa_per_1k_pod")
        return v if v is not None else 0.0  # unavailable -> do not penalize

    def _lead(m):
        return m.get("real_k91_lead")

    # eligible: reliable on the synthetic slow ramps (few misses)
    elig = [m for m in results
            if m.get("delay_p50") is not None
            and (m.get("miss_rate") or 0.0) <= 0.10]

    def _pick(pool, cap_real, cap_pod):
        c = [m for m in pool if _rfa(m) <= cap_real and _pfa(m) <= cap_pod
             and _lead(m) is not None and _lead(m) > 0]
        return min(c, key=lambda m: (_delay(m), _rfa(m))) if c else None

    # RECOMMENDED detectors are restricted to the smoothed aggregate EWMA-CUSUM
    # family: it dominates the per-layer vote at conservative FA and is robust to
    # per-token spikes (a single S1 EWMA can drive both thresholds, cheaply).
    RECO_FAMILIES = ("cusum_ewma15", "cusum_ewma30", "cusum_ewma50",
                     "cusum_ewma30_fast")
    smoothed = [m for m in elig if m["family"] in RECO_FAMILIES]

    # arm-relearn/admit: cheap corrective action (rotate/widen/relearn/admit) ->
    # tolerate false alarms to cut latency. Cap FA ~8/1k healthy; minimize delay.
    admit = _pick(smoothed, 8.0, 4.0)
    # arm-rewind: expensive corrective action -> false alarms costly. Cap FA hard
    # (FA_real ~2/1k, near-zero on the aggressive control), then minimize delay.
    rewind = _pick(smoothed, 2.5, 0.5)

    # unconstrained delay-minimizers over ALL families (context: the pure
    # latency floor, which the tight-slope / vote configs reach only at high FA).
    admit_any = _pick(elig, 8.0, 6.0)
    rewind_any = _pick(elig, 2.5, 1.0)

    def _slim(m):
        if m is None:
            return None
        return {k: m.get(k) for k in (
            "name", "family", "params", "delay_p50", "delay_p90",
            "miss_rate", "fa_per_1k_synth", "fa_per_1k_real", "fa_per_1k_pod",
            "real_k91_first_fire", "real_k91_lead")}

    reco = {
        "note": ("Detector scoped to the SLOW-EROSION regime only (static/wide "
                 "masks, K91-family). No usable S1 lead in the aggressive "
                 "W50+K23+rotate32 regime (S1 pinned flat ~0.815); the pod r1 "
                 "series is used only as an adversarial false-alarm control. "
                 "delay_p50/p90 are synthetic ramps calibrated on the real K91 "
                 "noise; real_k91_lead is the single measured lead vs the "
                 "ground-truth text lock (pos 2476)."),
        "recommended_detector": ("aggregate EWMA-CUSUM (alpha=0.30, "
                                 "self-calibrated sigma over first 128 tok, "
                                 "lagged-window baseline base_lag=32 base_win=128); "
                                 "two thresholds on the normalized CUSUM g/sigma."),
        "profile_arm_relearn_admit": _slim(admit),
        "profile_arm_rewind": _slim(rewind),
        "delay_floor_any_family_admit_cap": _slim(admit_any),
        "delay_floor_any_family_rewind_cap": _slim(rewind_any),
        "baseline_0020_default_thr3e-4": _slim(next(
            (m for m in results
             if m["name"] == "slope_base_a10_w64_s16_init0_thr0.0003"), None)),
    }
    with open(os.path.join(args.out, "recommendation.json"), "w") as f:
        json.dump(reco, f, indent=2)

    # summary table (top configs by family at acceptable FA) for the report
    def _row_md(m):
        return ("| {name} | {fam} | {d50} | {d90} | {miss} | {rfa} | {pfa} | "
                "{ff} | {lead} |").format(
            name=m["name"], fam=m["family"],
            d50=fmt(m.get("delay_p50")), d90=fmt(m.get("delay_p90")),
            miss=fmt(m.get("miss_rate")), rfa=fmt(m.get("fa_per_1k_real")),
            pfa=fmt(m.get("fa_per_1k_pod")),
            ff=(m.get("real_k91_first_fire") or ""),
            lead=fmt(m.get("real_k91_lead")))

    with open(os.path.join(args.out, "summary_table.md"), "w") as f:
        f.write("# S1 onset detector tuning — summary (auto-generated)\n\n")
        f.write("Slow-erosion regime only. delay_p50/p90 = synthetic ramps; "
                "FA_real = K91 healthy plateau + K0 router-libero; "
                "FA_pod = aggressive pod r1 (adversarial); "
                "lead = 2476 - first collapse-region fire (real K91).\n\n")
        f.write("| config | family | d_p50 | d_p90 | miss | FA_real | FA_pod | "
                "K91 first fire | K91 lead |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for m in sorted(elig, key=lambda z: (_rfa(z), _delay(z)))[:25]:
            f.write(_row_md(m) + "\n")
        f.write("\n## Recommended profiles\n\n")
        f.write(json.dumps(reco, indent=2) + "\n")

    print("\nwrote", os.path.join(args.out, "curves_all.csv"))
    print("recommendation:", json.dumps(reco, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
