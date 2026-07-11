#!/usr/bin/env python3
"""E-DET shadow replay: run the recommended EWMA-CUSUM detector (ARM + FIRE
profiles) offline on a podC S1 sensor CSV and compare its fire positions with
the real text collapse onset. Reuses det_cusum + load_sensor_csv from the
offline tuning study (scripts/tune_s1_detector.py) verbatim.

Usage: shadow_replay.py <reap_loop_repo> <run_dir> <prompt_len_tokens>
  run_dir must contain s1.csv, content.txt, summary.json
"""
import json, os, sys, importlib.util

REPO = sys.argv[1]
RUN = sys.argv[2]
PROMPT_LEN = int(sys.argv[3])

spec = importlib.util.spec_from_file_location(
    "tune_s1", os.path.join(REPO, "scripts", "tune_s1_detector.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

# Recommended detector (REPORT §4): EWMA alpha=0.50 front-end, lagged baseline
# lag32/win128, sigma self-calibrated over first 128 tok.
COMMON = dict(pre_alpha=0.50, base_lag=32, base_win=128, min_base=32,
              calib_win=128, refractory=64)
ARM = dict(k_sigma=0.5, h_sigma=4)     # arm-relearn/admit
FIRE = dict(k_sigma=1.0, h_sigma=8)    # arm-rewind


def find_loop_onset(text):
    """Return (char_offset, period, unit) of the first consecutive repetition
    lock, or (None, None, None) if no loop. Detect the period from the tail,
    then walk back to the first index where >=3 consecutive copies start."""
    n = len(text)
    if n < 200:
        return (None, None, None)
    tail = text[-min(3000, n):]
    best = None
    for p in range(1, 400):
        if 2 * p > len(tail):
            break
        reps = 0
        i = len(tail) - p
        unit = tail[i:i + p]
        if not unit.strip():
            continue
        j = i
        while j - p >= 0 and text_slice(tail, j - p, p) == unit:
            reps += 1
            j -= p
        if reps >= 3:
            # candidate period; prefer smallest period with many reps
            span = reps * p
            best = (p, span, unit)
            break
    if not best:
        return (None, None, None)
    p, span, unit = best
    # find first occurrence index in full text where >=3 consecutive copies start
    onset = None
    i = 0
    while i + p <= n:
        if text[i:i + p] == unit:
            k = i
            c = 0
            while k + p <= n and text[k:k + p] == unit:
                c += 1
                k += p
            if c >= 3:
                onset = i
                break
            i = k
        else:
            i += 1
    return (onset, p, unit)


def text_slice(s, start, length):
    if start < 0:
        return None
    return s[start:start + length]


def grade(text, repo):
    try:
        spec2 = importlib.util.spec_from_file_location(
            "fg", os.path.join(repo, "scripts", "functional_grade.py"))
        fg = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fg)
        for fn in ("grade_frontpage", "grade", "grade_html"):
            if hasattr(fg, fn):
                r = getattr(fg, fn)(text)
                if isinstance(r, (int, float)):
                    return int(r)
                if isinstance(r, dict):
                    return r.get("level", r.get("l0l3"))
                if isinstance(r, (list, tuple)):
                    return r[0]
    except Exception as e:
        return f"grade_err:{e}"
    return None


def main():
    s1 = os.path.join(RUN, "s1.csv")
    if not os.path.exists(s1) and os.path.exists(s1 + ".gz"):
        s1 = s1 + ".gz"
    poss, agg, layers = tune.load_sensor_csv(s1)
    content = open(os.path.join(RUN, "content.txt"), encoding="utf-8", errors="replace").read()
    summ = json.load(open(os.path.join(RUN, "summary.json")))
    ctok = summ.get("completion_tokens")

    mu, sd = tune.mean_std(agg)
    # full-run slope (least squares over index)
    n = len(agg)
    if n > 1:
        xs = list(range(n)); mx = sum(xs) / n; my = mu
        num = sum((xs[i] - mx) * (agg[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        slope_per_row = num / den if den else 0.0
    else:
        slope_per_row = 0.0

    arm_fires = tune.det_cusum(agg, ARM["k_sigma"], ARM["h_sigma"], **COMMON)
    fire_fires = tune.det_cusum(agg, FIRE["k_sigma"], FIRE["h_sigma"], **COMMON)
    arm_at = poss[arm_fires[0]] if arm_fires else None
    fire_at = poss[fire_fires[0]] if fire_fires else None

    onset_char, period, unit = find_loop_onset(content)
    has_close = "</html>" in content
    total_chars = len(content)
    onset_pos = None
    if onset_char is not None and total_chars > 0 and isinstance(ctok, int):
        gen_frac = onset_char / total_chars
        gen_tok = round(gen_frac * ctok)
        onset_pos = PROMPT_LEN + gen_tok   # spex_trace_pos coordinate

    def lead(fire):
        if fire is None or onset_pos is None:
            return None
        return onset_pos - fire   # positive = warned before text lock

    # per-layer erosion check: max per-layer slope vs aggregate
    lay_slopes = {}
    for l, ser in layers.items():
        m = len(ser)
        if m > 1:
            mxx = (m - 1) / 2.0
            myy = sum(ser) / m
            nu = sum((i - mxx) * (ser[i] - myy) for i in range(m))
            de = sum((i - mxx) ** 2 for i in range(m))
            lay_slopes[l] = nu / de if de else 0.0
    max_lay_slope = max(lay_slopes.values()) if lay_slopes else None

    out = {
        "run": os.path.basename(RUN.rstrip("/")),
        "sensor_rows": sum(len(v) for v in layers.values()),
        "pos_first": poss[0] if poss else None,
        "pos_last": poss[-1] if poss else None,
        "agg_mean": round(mu, 4), "agg_std": round(sd, 4),
        "agg_slope_per_tok": round(slope_per_row, 8),
        "agg_delta_over_run": round(slope_per_row * (n - 1), 4) if n > 1 else 0.0,
        "max_layer_slope_per_tok": round(max_lay_slope, 8) if max_lay_slope is not None else None,
        "completion_tokens": ctok,
        "content_chars": total_chars,
        "has_html_close": has_close,
        "loop_detected": onset_char is not None,
        "loop_period_chars": period,
        "loop_unit_preview": (unit[:40] if unit else None),
        "text_onset_char": onset_char,
        "text_onset_pos_est": onset_pos,
        "grade_l0l3": grade(content, REPO),
        "ARM_at": arm_at, "ARM_n_fires": len(arm_fires),
        "FIRE_at": fire_at, "FIRE_n_fires": len(fire_fires),
        "ARM_lead": lead(arm_at), "FIRE_lead": lead(fire_at),
    }
    # classification vs text onset
    def classify(fire, lead_v):
        if onset_pos is None:            # no text collapse
            return "false_alarm" if fire is not None else "correct_silent"
        if fire is None:
            return "missed"
        if lead_v is not None and lead_v > 0:
            return "true_lead"
        return "late"                     # fired after text lock
    out["ARM_verdict"] = classify(arm_at, out["ARM_lead"])
    out["FIRE_verdict"] = classify(fire_at, out["FIRE_lead"])
    json.dump(out, open(os.path.join(RUN, "shadow.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
