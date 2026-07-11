#!/usr/bin/env python3
"""Instrumented-collapse lead analysis (K12 wide, live per-layer S1 sensor).

Companion to scripts/pregarbage_sensor_hunt.py. That script mined RECORDED podC
K23/K38 runs; this one analyses a FRESH, deliberately-instrumented K12 (wide,
fast-collapse) run produced by run_w_sweep_freeze_safe.py with
DS4_COLLAPSE_INSTRUMENT=1 (per-run phase-2 s1_perlayer.csv + route_p2.csv, plus
conf.csv/tokens.csv when the binary carries the DS4_DIAG_CONF_LOG diagnostic
patch / patch 0028).

QUESTION (operator): does any recorded signal (S1 per-layer, or -- if present --
per-token confidence entropy / logit margin) move BEFORE the first garbage char
enters the phase-2 context? Positive lead => widen-to-K0 on the fly WITHOUT
rewind is viable. Non-positive => the garbage is inevitable, a rewind is
required and the signals only FIRE it.

Deterministic, CPU-only, read-only. Reuses det_cusum / det_vote_kofn /
load_sensor_csv / mean_std from scripts/tune_s1_detector.py verbatim.

Usage:
  python scripts/analyze_instrumented_collapse.py <run_dir> \
      [--fg-substr "<html\n"] [--label k12_cyber_r00] [--out metrics.json]
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "tune", os.path.join(REPO, "scripts", "tune_s1_detector.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)
det_cusum = tune.det_cusum
det_vote_kofn = tune.det_vote_kofn
load_sensor_csv = tune.load_sensor_csv
mean_std = tune.mean_std

COMMON = dict(pre_alpha=0.50, base_lag=32, base_win=128, min_base=32,
              calib_win=128, refractory=64)
ARM = dict(k_sigma=0.5, h_sigma=4)
FIRE = dict(k_sigma=1.0, h_sigma=8)

# ---- first-garbage heuristics (earliest structural break in phase-2 text) ----
# Each returns a char offset or -1. Ordered generic->specific; caller may also
# pass an exact --fg-substr once the content has been eyeballed (determinism).
_FG_PATTERNS = [
    "<html\n", "<html ", "<style:", "<head:", "<body:", "<title:",
    "<div:", "<meta:", "<link:", "<!doctype html>\n<!doctype",
]


def find_first_garbage(text, fg_substr=None):
    # An explicit marker (eyeballed from the deterministic content) is
    # authoritative: use it alone, never min() it against the generic
    # patterns (which also match the legitimate doc-restart tags).
    if fg_substr:
        i = text.find(fg_substr.encode().decode("unicode_escape"))
        return (i if i >= 0 else None,
                f"explicit --fg-substr {fg_substr!r}" if i >= 0
                else f"--fg-substr {fg_substr!r} NOT FOUND")
    cands = []
    for p in _FG_PATTERNS:
        i = text.find(p)
        if i >= 0:
            cands.append((i, f"pattern {p!r}"))
    # doubled tag: <html>...<html>  or repeated <!doctype
    for tag in ("<html>", "<!doctype html>", "<head>", "<body>"):
        i = text.find(tag)
        if i >= 0:
            j = text.find(tag, i + len(tag))
            if j >= 0:
                cands.append((j, f"doubled {tag!r}"))
    if not cands:
        return None, "no structural-break pattern matched (declare manually)"
    off, desc = min(cands, key=lambda t: t[0])
    return off, desc


def char_to_pos_via_tokens(tokens_csv, prefix_len_chars, garbage_char):
    """Exact map using patch-0028 tokens.csv (pos,token_id,piece)."""
    if not (tokens_csv and os.path.exists(tokens_csv)):
        return None
    acc = 0
    with open(tokens_csv, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            try:
                pos = int(row[0]); piece = row[2] if len(row) > 2 else ""
            except (ValueError, IndexError):
                continue
            acc += len(piece)
            if acc >= garbage_char:
                return pos
    return None


def first_fire(fires):
    return fires[0] if fires else None


def zscore_onset(poss, series, base_n=32, z=3.0, sustain=2, worse_up=True):
    """Short-baseline onset: first pos where the signal deviates >= z sigma
    from the FIRST `base_n` (assumed-healthy, post-prefill) samples, sustained
    for `sustain` consecutive tokens. Robust in the fast-collapse regime where
    the 128-token CUSUM calibration never completes before garbage.

    worse_up=True  -> alarm on a RISE (entropy).
    worse_up=False -> alarm on a FALL (margin, top-1 prob).
    Returns (onset_pos|None, base_mean, base_std).
    """
    if len(series) <= base_n + sustain:
        return None, None, None
    base = series[:base_n]
    mu = sum(base) / len(base)
    var = sum((v - mu) ** 2 for v in base) / max(1, len(base) - 1)
    sd = max(1e-9, var ** 0.5)
    run = 0
    for i in range(base_n, len(series)):
        dev = (series[i] - mu) if worse_up else (mu - series[i])
        if dev >= z * sd:
            run += 1
            if run >= sustain:
                return poss[i - sustain + 1], mu, sd
        else:
            run = 0
    return None, mu, sd


def analyze(run_dir, fg_substr=None, label=None):
    d = run_dir
    sensor = os.path.join(d, "s1_perlayer.csv")
    if not os.path.exists(sensor):
        raise SystemExit(f"no s1_perlayer.csv in {d}")
    poss, agg, layers = load_sensor_csv(sensor)
    n = len(poss)
    pos_first, pos_last = poss[0], poss[-1]

    # phase-2 continuation text = trest.txt (collapse lives here)
    trest = ""
    for cand in ("trest.txt", "deliverable.html"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            trest = open(p, encoding="utf-8", errors="ignore").read()
            trest_src = cand
            break
    fg_char, fg_desc = find_first_garbage(trest, fg_substr)

    # char -> pos. Prefer exact tokens.csv; else char/token ratio anchored at
    # the first decode pos (poss[0]).
    fg_pos = None
    map_method = None
    tokens_csv = os.path.join(d, "tokens.csv")
    if fg_char is not None:
        fg_pos = char_to_pos_via_tokens(tokens_csv, 0, fg_char)
        if fg_pos is not None:
            map_method = "tokens.csv exact (0028)"
        else:
            ratio = len(trest) / n if n else 1.0  # chars per decoded token
            didx = min(n - 1, max(0, round(fg_char / ratio))) if ratio else 0
            fg_pos = poss[didx]
            map_method = f"char/token ratio {ratio:.3f} (no tokens.csv)"

    # ---- aggregate E-DET (S1) ----
    agg_arm = first_fire(det_cusum(agg, ARM["k_sigma"], ARM["h_sigma"], **COMMON))
    agg_fire = first_fire(det_cusum(agg, FIRE["k_sigma"], FIRE["h_sigma"], **COMMON))
    agg_arm_pos = poss[agg_arm] if agg_arm is not None else None
    agg_fire_pos = poss[agg_fire] if agg_fire is not None else None

    # ---- per-layer earliest ARM + k-of-N vote ----
    lay_ids = sorted(layers)
    lay_series = [layers[l] for l in lay_ids]
    sigmas = []
    for s in lay_series:
        sigmas.append(max(1e-4, mean_std(s[:COMMON["calib_win"]])[1]))
    per_layer_arm = {}
    for l, s in zip(lay_ids, lay_series):
        f = first_fire(det_cusum(s, ARM["k_sigma"], ARM["h_sigma"], **COMMON))
        if f is not None:
            per_layer_arm[l] = poss[f]
    earliest_layer = None
    if per_layer_arm:
        earliest_layer = min(per_layer_arm.items(), key=lambda kv: kv[1])
    votes = {}
    for K in (1, 2, 4, 8, 12):
        vf = det_vote_kofn(lay_series, sigmas, ARM["k_sigma"], ARM["h_sigma"], K,
                           base_lag=COMMON["base_lag"], base_win=COMMON["base_win"],
                           min_base=COMMON["min_base"], refractory=COMMON["refractory"])
        f = first_fire(vf)
        votes[K] = poss[f] if f is not None else None

    calib_floor = pos_first + COMMON["calib_win"]

    def lead(alarm_pos):
        if alarm_pos is None or fg_pos is None:
            return None
        return fg_pos - alarm_pos  # positive = warned BEFORE garbage

    # pre-garbage raw slope of the aggregate S1
    pre = [(poss[i], agg[i]) for i in range(n) if fg_pos is not None and poss[i] < fg_pos]
    pre_stats = None
    if len(pre) >= 3:
        xs = [p for p, _ in pre]; ys = [v for _, v in pre]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        pre_stats = dict(pre_garbage_tokens=len(pre), agg_mean_pre=round(my, 4),
                         agg_first=round(ys[0], 4), agg_last=round(ys[-1], 4),
                         agg_slope_per_tok=round(num / den, 6))

    # ---- aggregate S1 short-baseline z-onset (robust to fast collapse) ----
    s1_zpos, s1_zmu, s1_zsd = zscore_onset(poss, agg, base_n=32, z=3.0,
                                           sustain=2, worse_up=True)

    # ---- optional confidence sidecar (DS4_DIAG_CONF_LOG diagnostic patch) ----
    conf = analyze_conf(os.path.join(d, "conf.csv"), poss, fg_pos)

    res = dict(
        label=label or os.path.basename(d.rstrip("/")),
        run_dir=os.path.relpath(d, REPO) if d.startswith(REPO) else d,
        trest_src=trest_src, n_tokens=n, n_layers=len(lay_ids),
        pos_first=pos_first, pos_last=pos_last,
        calib_completes_pos=calib_floor,
        first_garbage_char=fg_char, first_garbage_desc=fg_desc,
        first_garbage_pos=fg_pos, char_to_pos_method=map_method,
        agg_ARM_pos=agg_arm_pos, agg_ARM_lead=lead(agg_arm_pos),
        agg_FIRE_pos=agg_fire_pos, agg_FIRE_lead=lead(agg_fire_pos),
        earliest_layer_id=(earliest_layer[0] if earliest_layer else None),
        earliest_layer_ARM_pos=(earliest_layer[1] if earliest_layer else None),
        earliest_layer_lead=(lead(earliest_layer[1]) if earliest_layer else None),
        aggS1_zonset_pos=s1_zpos, aggS1_zonset_lead=lead(s1_zpos),
        n_layers_ARM_before_garbage=sum(
            1 for l, p in per_layer_arm.items() if fg_pos is not None and p < fg_pos),
        vote_pos={str(k): v for k, v in votes.items()},
        vote_lead={str(k): lead(v) for k, v in votes.items()},
        calib_floor_minus_garbage=(calib_floor - fg_pos if fg_pos is not None else None),
        pre_garbage_stats=pre_stats,
        confidence=conf,
    )
    return res


def analyze_conf(conf_csv, poss, fg_pos):
    """DS4_DIAG_CONF_LOG sidecar: pos + per-token confidence columns.

    Flexible on schema: any of entropy/margin/top1/top1_prob/logit_margin.
    Runs the same E-DET CUSUM on the WORSENING direction (entropy up => use +x;
    margin/top1 down => use -x) and reports lead vs first-garbage.
    """
    if not os.path.exists(conf_csv):
        return dict(available=False,
                    note="no conf.csv -- binary lacks DS4_DIAG_CONF_LOG; "
                         "entropy/margin/top-1 NOT captured this run")
    rows = {}
    header = None
    with open(conf_csv, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        idx = {name: i for i, name in enumerate(header or [])}
        for row in rd:
            try:
                pos = int(row[idx.get("pos", 0)])
            except (ValueError, IndexError):
                continue
            rows[pos] = row
    if not rows:
        return dict(available=True, empty=True, header=header)
    out = dict(available=True, header=header, n_rows=len(rows), signals={})
    common_pos = [p for p in poss if p in rows]
    for col, worse_up in (("entropy", True), ("margin", False),
                          ("logit_margin", False), ("top1_prob", False),
                          ("top1", False)):
        if col not in (idx := {n: i for i, n in enumerate(header)}):
            continue
        series, sp = [], []
        for p in common_pos:
            try:
                v = float(rows[p][idx[col]])
            except (ValueError, IndexError):
                continue
            series.append(v if worse_up else -v); sp.append(p)
        if len(series) < COMMON["calib_win"] + COMMON["min_base"]:
            out["signals"][col] = dict(note="too short to calibrate", n=len(series))
            continue
        arm = det_cusum(series, ARM["k_sigma"], ARM["h_sigma"], **COMMON)
        arm_pos = sp[arm[0]] if arm else None
        lead = (fg_pos - arm_pos) if (arm_pos is not None and fg_pos is not None) else None
        # short-baseline z-onset (series already oriented worse=higher)
        zpos, zmu, zsd = zscore_onset(sp, series, base_n=32, z=3.0, sustain=2,
                                      worse_up=True)
        zlead = (fg_pos - zpos) if (zpos is not None and fg_pos is not None) else None
        out["signals"][col] = dict(worsening_dir="up" if worse_up else "down",
                                   ARM_pos=arm_pos, lead_vs_garbage=lead,
                                   zonset_pos=zpos, zonset_lead=zlead)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--fg-substr", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for rd in a.run_dirs:
        results.append(analyze(rd, a.fg_substr, a.label))
    blob = dict(common=COMMON, ARM=ARM, FIRE=FIRE, runs=results)
    txt = json.dumps(blob, indent=2)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
