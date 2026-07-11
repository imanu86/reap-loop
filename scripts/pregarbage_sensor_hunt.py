#!/usr/bin/env python3
"""Pre-garbage early-warning sensor hunt.

QUESTION (operator): does any recorded signal detect the drift BEFORE the first
garbage character enters the context? If yes -> widen-to-K0 on the fly WITHOUT
rewind is viable (the poisoned span never enters the KV). If no -> the garbage is
inevitable and a rewind is required to erase it; S1/garbage-char then only serve
to FIRE the rewind.

This is deliberately narrower than the sibling shadow-replay
(runs/ds4/20260711_podC_edet_shadow), which measured lead vs the terminal
repetition-LOCK. Here the deadline is the FIRST-GARBAGE token (the onset of
mask-induced incoherence), which is what actually poisons the context.

Method (CPU-only, offline, read-only on recorded runs):
  1. Localize the first-garbage char in content.txt (first structurally broken
     token: tag missing its '>', doubled tag, malformed CSS) -> convert to a
     spex_trace_pos via the run's measured char/token ratio (anchored at
     prompt_len). Reported alongside the report's independent estimate.
  2. For every candidate signal available in the per-layer S1 sensor
     (pos,layer,pruned_mass,total_mass):
       - aggregate S1 EWMA-CUSUM (ARM k0.5/h4, FIRE k1.0/h8) -- the E-DET profile
       - PER-LAYER EWMA-CUSUM: does any single layer, or a k-of-N vote, ARM
         before the aggregate AND before first-garbage?
       - per-layer raw-threshold and per-layer slope (drift) onset
     measure alarm_pos and LEAD = first_garbage_pos - alarm_pos (positive =
     warned before the garbage entered the context).
  3. Report the structural floor: an EWMA-CUSUM cannot fire before its sigma
     calibration completes at pos_first + calib_win. Compare that floor to
     first-garbage.

Signals NOT available in any recorded run (declared, not silently skipped):
  entropy / logit-margin / top-1 prob per token -- no confidence logging exists.
  The verdict flags whether an instrumented run is required.

Reuses det_cusum + load_sensor_csv + mean_std from scripts/tune_s1_detector.py
verbatim. Deterministic. No GPU, no pod.

Usage: python scripts/pregarbage_sensor_hunt.py <out_dir>
"""
from __future__ import annotations
import csv, gzip, json, math, os, sys, importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "runs", "ds4", "20260711_pregarbage_sensor")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "tune", os.path.join(REPO, "scripts", "tune_s1_detector.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)
det_cusum = tune.det_cusum
load_sensor_csv = tune.load_sensor_csv
mean_std = tune.mean_std

# E-DET recommended profile (podC shadow, verbatim)
COMMON = dict(pre_alpha=0.50, base_lag=32, base_win=128, min_base=32,
              calib_win=128, refractory=64)
ARM = dict(k_sigma=0.5, h_sigma=4)
FIRE = dict(k_sigma=1.0, h_sigma=8)

# ---- Runs under test: (label, run_dir, prompt_len, first_garbage_finder) -----
# first_garbage_finder(content) -> (char_offset|None, description)


def fg_run1(c):
    # K23 cyber: first broken token is '<html' with no '>' (then doubled).
    i = c.find("<html\n")
    return (i if i >= 0 else None, "'<html' tag missing '>' (then doubled <html>)")


def fg_run2(c):
    # K38 cyber: valid doctype/html/head/meta, then '<style:' (colon, not '>').
    i = c.find("<style:")
    return (i if i >= 0 else None, "'<style:' colon instead of '>' -> comment-list")


def fg_none(c):
    return (None, "completes (L1, </html>); no garbage onset -> FA control")


RUNS = [
    ("podC_run1_K23_cyber",
     "runs/ds4/20260711_podC_edet_shadow/run1_r00", 78, fg_run1),
    ("podC_run2_K38_cyber",
     "runs/ds4/20260711_podC_edet_shadow/run2_r00", 78, fg_run2),
    ("podC_run3_K23_coffee",
     "runs/ds4/20260711_podC_edet_shadow/run3_r00", 78, fg_none),
    # scope_divergence r1: aggressive W50+K23+rotate32, per-layer sensor, no
    # content.txt (only pace). Analysed for signal shape / FA, garbage pos taken
    # from the pace 'first loop' if present; else flagged.
    ("scope_r1_K23_rotate32_aggr",
     "runs/ds4/20260710_scope_divergence_pod/r1", None, None),
]


def s1_path(run_dir):
    for name in ("s1.csv", "s1.csv.gz", "s1_r1.csv.gz", "s1_r1.csv"):
        p = os.path.join(REPO, run_dir, name)
        if os.path.exists(p):
            return p
    return None


def per_layer_cusum_onsets(layers, poss, prof):
    """Run det_cusum per layer with its own self-calibrated sigma. Return
    {layer: first_fire_pos|None} and the sorted list of (pos, layer)."""
    onsets = {}
    fired = []
    for lay in sorted(layers):
        fires = det_cusum(layers[lay], prof["k_sigma"], prof["h_sigma"], **COMMON)
        if fires:
            p = poss[fires[0]]
            onsets[lay] = p
            fired.append((p, lay))
        else:
            onsets[lay] = None
    fired.sort()
    return onsets, fired


def kofn_vote_onset(layers, poss, prof, K):
    """First pos at which >=K layers are simultaneously in fired state. We
    approximate 'fired state' as: layer has produced a CUSUM fire at or before
    this pos (latching). Returns pos or None. This bounds the earliest a k-of-N
    vote could trigger from these per-layer CUSUMs."""
    onsets, _ = per_layer_cusum_onsets(layers, poss, prof)
    fire_positions = sorted(p for p in onsets.values() if p is not None)
    if len(fire_positions) < K:
        return None
    return fire_positions[K - 1]  # Kth distinct layer to have fired


def raw_threshold_onset(series, poss, thr):
    for i, v in enumerate(series):
        if v >= thr:
            return poss[i]
    return None


def slope_onset(series, poss, win, thr):
    """First pos where the trailing-win least-squares slope >= thr (per token)."""
    n = len(series)
    for t in range(win, n):
        seg = series[t - win:t]
        mx = (win - 1) / 2.0
        my = sum(seg) / win
        num = sum((j - mx) * (seg[j] - my) for j in range(win))
        den = sum((j - mx) ** 2 for j in range(win))
        s = num / den if den else 0.0
        if s >= thr:
            return poss[t]
    return None


def analyze(label, run_dir, prompt_len, fg_finder):
    p = s1_path(run_dir)
    if p is None:
        return {"run": label, "error": "no s1 sensor"}
    poss, agg, layers = load_sensor_csv(p)
    pos_first = poss[0]
    calib_done_pos = pos_first + COMMON["calib_win"]

    # --- first-garbage localization ---
    fg_pos = None
    fg_char = None
    fg_desc = None
    content_path = os.path.join(REPO, run_dir, "content.txt")
    if fg_finder is not None and os.path.exists(content_path):
        content = open(content_path, encoding="utf-8", errors="replace").read()
        fg_char, fg_desc = fg_finder(content)
        if fg_char is not None and len(content) > 0:
            cpt = len(content) / len(poss)  # chars per generated token
            fg_pos = round(pos_first + fg_char / cpt)
    elif fg_finder is None:
        fg_desc = "no content.txt (pace-only run); garbage pos not localizable here"

    mu, sd = mean_std(agg)

    # --- aggregate CUSUM (E-DET profile) ---
    arm_fires = det_cusum(agg, ARM["k_sigma"], ARM["h_sigma"], **COMMON)
    fire_fires = det_cusum(agg, FIRE["k_sigma"], FIRE["h_sigma"], **COMMON)
    arm_at = poss[arm_fires[0]] if arm_fires else None
    fire_at = poss[fire_fires[0]] if fire_fires else None

    # --- per-layer CUSUM: earliest arming layer + how many lead first-garbage ---
    lay_onsets, lay_fired = per_layer_cusum_onsets(layers, poss, ARM)
    earliest_layer_pos = lay_fired[0][0] if lay_fired else None
    earliest_layer_id = lay_fired[0][1] if lay_fired else None
    n_layers_before_fg = (
        sum(1 for (pp, _) in lay_fired if fg_pos is not None and pp < fg_pos))
    n_layers_before_agg_arm = (
        sum(1 for (pp, _) in lay_fired if arm_at is not None and pp < arm_at))

    # --- k-of-N vote earliest trigger ---
    vote = {K: kofn_vote_onset(layers, poss, ARM, K) for K in (1, 2, 4, 8, 12)}

    # --- pre-garbage descriptive: is S1 already elevated / rising pre-garbage? ---
    pre_stats = None
    if fg_pos is not None:
        pre_idx = [i for i, q in enumerate(poss) if q < fg_pos]
        if len(pre_idx) >= 4:
            pre_series = [agg[i] for i in pre_idx]
            pm, psd = mean_std(pre_series)
            # slope over the whole pre-garbage window
            n = len(pre_series)
            mx = (n - 1) / 2.0
            mmy = sum(pre_series) / n
            num = sum((j - mx) * (pre_series[j] - mmy) for j in range(n))
            den = sum((j - mx) ** 2 for j in range(n))
            pre_slope = num / den if den else 0.0
            pre_stats = {
                "pre_garbage_tokens": len(pre_idx),
                "agg_mean_pre": round(pm, 4),
                "agg_std_pre": round(psd, 4),
                "agg_slope_pre_per_tok": round(pre_slope, 6),
                "agg_first_val": round(agg[0], 4),
            }

    def lead(x):
        return None if (x is None or fg_pos is None) else fg_pos - x

    return {
        "run": label,
        "regime": run_dir,
        "pos_first": pos_first,
        "pos_last": poss[-1],
        "n_tokens": len(poss),
        "n_layers": len(layers),
        "calib_completes_pos": calib_done_pos,
        "agg_mean": round(mu, 4),
        "agg_std": round(sd, 4),
        "first_garbage_char": fg_char,
        "first_garbage_pos_est": fg_pos,
        "first_garbage_desc": fg_desc,
        "calib_floor_minus_garbage": (
            None if fg_pos is None else calib_done_pos - fg_pos),
        # aggregate
        "AGG_ARM_at": arm_at, "AGG_ARM_lead_vs_garbage": lead(arm_at),
        "AGG_FIRE_at": fire_at, "AGG_FIRE_lead_vs_garbage": lead(fire_at),
        # per-layer
        "earliest_layer_id": earliest_layer_id,
        "earliest_layer_ARM_at": earliest_layer_pos,
        "earliest_layer_lead_vs_garbage": lead(earliest_layer_pos),
        "n_layers_ARM_before_garbage": n_layers_before_fg if fg_pos else None,
        "n_layers_ARM_before_agg_arm": n_layers_before_agg_arm if arm_at else None,
        "per_layer_first5_fired": [
            {"pos": pp, "layer": ll} for (pp, ll) in lay_fired[:5]],
        # vote
        "vote_kofN_earliest_pos": {str(k): v for k, v in vote.items()},
        "vote_1of40_lead_vs_garbage": lead(vote[1]),
        "vote_4of40_lead_vs_garbage": lead(vote[4]),
        # pre-garbage descriptive
        "pre_garbage_stats": pre_stats,
    }


def main():
    results = [analyze(*r) for r in RUNS]
    json.dump({"common": COMMON, "ARM": ARM, "FIRE": FIRE, "runs": results},
              open(os.path.join(OUT, "pregarbage_metrics.json"), "w"), indent=2)
    for r in results:
        print(json.dumps(r, indent=2))
    print("\nwrote", os.path.join(OUT, "pregarbage_metrics.json"))


if __name__ == "__main__":
    main()
