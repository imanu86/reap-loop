#!/usr/bin/env python3
"""E-LAT: per-tier expert-recall latency model for REAP-LOOP (offline analysis).

Builds the per-tier latency table for expert recall (VRAM cache / RAM
page-cache / cold SSD / CQ1 decompress), calibrates a per-token throughput
model on hit~1 runs, validates it against measured (config, t/s) pairs from
runs/ds4 summary.csv artifacts, and projects differentiated-quantization
scenarios (status quo / async CQ1 cold / full tiering) on the local 3060.

OFFLINE ONLY: reads CSV/MD artifacts already on disk; no GPU, no server.

Usage:
    python scripts/latency_tier_model.py [--runs-root runs/ds4] [--csv-out PATH]

Provenance of every constant is in TIER_TABLE / CALIBRATION below and in
runs/ds4/20260710_elat_tier_latency/REPORT.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys

# ---------------------------------------------------------------------------
# Architecture constants (provenance: pod smoke 0021 README + J35)
# ---------------------------------------------------------------------------
EXPERT_MIB = 6.75          # bytes/expert 2-bit IQ2_XXS+Q2_K: 4466147328/631 experts
                           # = 6.7503 MiB (runs/ds4/20260710_pod_smoke_0020_0021/
                           # README.md, smoke c) and J35 gguf_inspect (ledger).
MOE_LAYERS = 43            # FLASH: 43 layers x 256 experts (docs/PACE_DESIGN.md #2)
TOP_K = 6                  # DS4_MAX_EXPERT_USED=6 (docs/PACE_DESIGN.md #2)
E_PER_TOKEN = MOE_LAYERS * TOP_K   # 258 expert recalls per decode token
GIB_PER_TOKEN = E_PER_TOKEN * EXPERT_MIB / 1024.0  # 1.70 GiB/token if all miss

# ---------------------------------------------------------------------------
# Per-tier recall latency (per single expert, ms) with provenance.
# Hardware is stated for every number; rates do NOT transfer across hosts.
# ---------------------------------------------------------------------------
TIER_TABLE = [
    # (tier, hw, ms_lo, ms_hi, provenance)
    ("a. VRAM resident cache hit", "3060/3090", 0.0, 0.0,
     "inside t_compute; no copy. Resident hit path exists but local runtime "
     "measures hit~0 (J31: resident=1/6923 events)."),
    ("b. RAM page-cache -> VRAM (H2D copy, streamed selected)", "3060 (WSL)",
     0.95, 2.31,
     "copy_ms_batch/6: 5.712/6=0.95 ms (cache128 html320, "
     "runs/ds4/20260709_local_cache_sweep_k23_combined.csv) to 13.853/6=2.31 ms "
     "(cache258 same file). REAP-era with SSD+trace contention: 59/6~9.8 ms "
     "(docs/PACE_DESIGN.md #1, hit_rate 0.83)."),
    ("b'. RAM page-in bulk (WRAP/fattorino, warm)", "3060 (WSL)", 0.22, 0.83,
     "6.07 GiB/198 ms = 30.7 GiB/s (local_cache_sweep_k23_html320_rev cache258) "
     "to 6.07 GiB/761 ms = 8.0 GiB/s (20260710_w50_..._html4000 server.stderr: "
     "16 thread mlock). Historic WRAP 6.07 GiB/445 ms (PACE_DESIGN "
     "PREFILL_WAIT_WRAP row, J3) = 13.6 GiB/s -> 0.48 ms/expert."),
    ("c. SSD cold, bulk threaded page-in", "3060 (WSL, NVMe)", 1.5, 2.7,
     "coldest prefetch rates: 12.14 GiB/3844 ms = 3.16 GiB/s (rotate_smoke_"
     "gatefix), 14.26/4612 = 3.09 (breath_k0 html800), 75.78/30291 = 2.50 "
     "(20260710_stepdown_relearn_only), 6.07/2007 = 3.02 (prebreath_adapt) "
     "-> 2.5-4.4 GiB/s -> 6.75 MiB / rate."),
    ("c'. SSD cold, synchronous in decode path (miss batch)", "3060 (WSL)",
     50.0, 59.0,
     "copy_ms_per_batch ~59 ms REAP-era SSD-bound (PACE_DESIGN #1); rotate on "
     "cold cache258: copy_ms_batch 49.6-54.9 (20260709_rotate_smoke*, "
     "cache_pattern_table.csv). Per missing expert if 1 miss/batch."),
    ("c''. SSD cold storm (uncached decode, QD1 faulting)", "3060 (WSL)",
     230.0, 230.0,
     "'cold decode can stall >60 s/token' (PACE_DESIGN #7) / 258 experts "
     "= ~230 ms/expert equivalent worst tail."),
    ("d. CQ1 sync decompress+repack+copy", "3060 (CPU, WSL)", 64.0, 74.0,
     "J38 (ledger): after CQ1 admission last 14 tok at 0.06 t/s; server log "
     "(runs/ds4/20260709_cq1_parallel/local_3060_cq1_native50/server_log_tail"
     ".txt): 277.114-32.910 = 244.2 s for 14 tok = 17.4 s/tok / 258 = 67.6 ms; "
     "3612 copies = 258/tok, repacked 24381 MiB. J34 lossless RAM sidecar "
     "(no decompress) ~39 ms/copy at prefill (158.9 s, 4056 copies)."),
    ("b-pod. RAM page-cache bulk (1TB-RAM pod)", "3090 pod", 0.033, 0.23,
     "delta-prefetch 0.693 GiB/30 ms = 23.1 GiB/s down to 4.167 GiB/983 ms "
     "= 4.24 GiB/s cold-ish first touch (20260710_pod_smoke_0020_0021/"
     "delta.diag). Pod numbers do NOT transfer to the 3060."),
]

# ---------------------------------------------------------------------------
# Calibration constants (derived below in calibrate(); listed for reference)
# Local 3060: html_local_k23_cache128_r01 (20260709_local_cache_sweep_k23_html320)
#   last_tps 3.12 -> t_ss = 320.5 ms; copy_ms_batch 5.712 -> t_b = 0.952 ms
#   t_compute = t_ss - 258*t_b = 74.9 ms;  first50 2.73 -> t_k0 = 366 ms
# Pod 3090: html_local_k23_cache1024_r01 (20260710_pod_cache1024_html800)
#   last_chunk 24.79 -> t_compute = 40.3 ms (hit~1); first50 2.18 -> t_k0 = 459 ms
# Pod 4070TiClass (e7w4/id63/qo6 families, J40 "RTX 4070 Ti pod tests"):
#   static64 last 3.43 -> t_ss = 291.5 ms; t_b = (291.5-40.3)/258 = 0.974 ms
#   (t_compute borrowed from 3090 pod -- stated assumption)
# Rotate churn (mask apply + cache invalidation, prefetch excluded):
#   local: requested4_html800_cache128 rotate32 vs baseline -> 0.67 s/rotate
#   pod:   id63 rotate16 vs e7w4 static64 -> 0.79 s/rotate
# ---------------------------------------------------------------------------

CAL_STEMS = {
    # (family_substring, stem) used for calibration -> excluded from validation
    ("20260709_local_cache_sweep_k23_html320", "html_local_k23_cache128_r01"),
    ("20260710_pod_cache1024_html800", "html_local_k23_cache1024_r01"),
    ("20260709_pod_e7w4_static64", "html_pod_k23_static_no_breath_64_r01"),
    ("20260709_requested4_html800_cache128", "html_local_k23_cache128_r01"),
    ("20260709_requested4_html800_cache128", "html_local_k23_rotate32_cache128_r01"),
    ("20260709_pod_id63_rotate16_64", "html_pod_k23_rotate_every16_64_r01"),
}

POD_3090_FAMILIES = ("pod_cache1024", "pod_smoke", "pod_t1")
POD_4070_FAMILIES = ("pod_e7w4", "pod_id63", "pod_qo6")
SKIP_FAMILIES = ("m1a_", "m1b_")   # untracked WIP dirs -- do not touch


def hw_class(family: str) -> str:
    if any(k in family for k in POD_3090_FAMILIES):
        return "pod3090"
    if any(k in family for k in POD_4070_FAMILIES):
        return "pod4070ti"
    return "local3060"


def fnum(row, *keys, default=None):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                continue
    return default


def load_runs(runs_root: str):
    recs = []
    pats = [os.path.join(runs_root, "2026070*", "summary.csv"),
            os.path.join(runs_root, "2026071*", "summary.csv")]
    for p in sorted(set(sum((glob.glob(x) for x in pats), []))):
        family = os.path.basename(os.path.dirname(p))
        if any(s in family for s in SKIP_FAMILIES):
            continue
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                n = fnum(row, "completion_tokens")
                avg = fnum(row, "avg_tps")
                if not n or not avg:
                    continue
                recs.append({
                    "family": family,
                    "stem": row.get("stem", ""),
                    "hw": hw_class(family),
                    "N": n,
                    "avg": avg,
                    "last": fnum(row, "last_chunk_tps", "last_tps"),
                    "first50": fnum(row, "first50_tps"),
                    "pf_s": (fnum(row, "prefetch_ms", default=0.0) or 0.0) / 1000.0,
                    "rot": fnum(row, "pace_rotates", "rotates", default=0.0) or 0.0,
                    "W": fnum(row, "first_learned_tok", default=50.0) or 50.0,
                    "cache": fnum(row, "cache_experts", "cache", default=None),
                })
    # combined sweep CSV (different schema, per-run tier counters)
    sweep = os.path.join(runs_root, "20260709_local_cache_sweep_k23_combined.csv")
    if os.path.exists(sweep):
        with open(sweep, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                suite = row.get("suite", "")
                stem = row.get("stem", "")
                if any((r["family"] == suite and r["stem"] == stem) for r in recs):
                    continue
                n = fnum(row, "completion_tokens")
                avg = fnum(row, "avg_tps")
                if not n or not avg:
                    continue
                recs.append({
                    "family": suite, "stem": stem, "hw": "local3060",
                    "N": n, "avg": avg, "last": fnum(row, "last_tps"),
                    "first50": fnum(row, "first50_tps"),
                    "pf_s": 0.0, "rot": 0.0, "W": 50.0,
                    "cache": fnum(row, "cache", default=None),
                })
    return recs


def calibrate(recs):
    def pick(fam_sub, stem):
        for r in recs:
            if fam_sub in r["family"] and r["stem"] == stem:
                return r
        raise SystemExit(f"calibration run not found: {fam_sub}/{stem}")

    cal = {}
    # --- local 3060 ---
    c128 = pick("20260709_local_cache_sweep_k23_html320", "html_local_k23_cache128_r01")
    t_b_local = 5.712 / TOP_K / 1000.0   # copy_ms_batch from combined sweep CSV
    t_ss_local = 1.0 / c128["last"]
    cal["local3060"] = {
        "t_b": t_b_local,
        "t_ss": t_ss_local,
        "t_compute": t_ss_local - E_PER_TOKEN * t_b_local,
        "t_k0": 1.0 / c128["first50"],
    }
    # --- pod 3090 (hit~1 at cache1024) ---
    p1024 = pick("20260710_pod_cache1024_html800", "html_local_k23_cache1024_r01")
    cal["pod3090"] = {
        "t_b": 0.0004,                    # delta.diag warm rate ~15 GiB/s
        "t_ss": 1.0 / p1024["last"],      # miss ~0 -> t_ss = t_compute
        "t_compute": 1.0 / p1024["last"],
        "t_k0": 1.0 / p1024["first50"],
    }
    # --- pod 4070Ti-class (RAM-hot streamed) ---
    s64 = pick("20260709_pod_e7w4_static64", "html_pod_k23_static_no_breath_64_r01")
    t_ss_4070 = 1.0 / s64["last"]
    t_c_4070 = cal["pod3090"]["t_compute"]      # stated assumption
    cal["pod4070ti"] = {
        "t_b": (t_ss_4070 - t_c_4070) / E_PER_TOKEN,
        "t_ss": t_ss_4070,
        "t_compute": t_c_4070,
        "t_k0": 1.0 / s64["first50"],
    }
    # --- rotate churn cost (s/rotate), prefetch excluded ---
    base = pick("20260709_requested4_html800_cache128", "html_local_k23_cache128_r01")
    rot = pick("20260709_requested4_html800_cache128", "html_local_k23_rotate32_cache128_r01")
    d_base = base["N"] / base["avg"]
    d_rot = rot["N"] / rot["avg"]
    cal["c_rot_local"] = max(0.0, (d_rot - d_base - rot["pf_s"]) / rot["rot"])
    prot = pick("20260709_pod_id63_rotate16_64", "html_pod_k23_rotate_every16_64_r01")
    d_ps = s64["N"] / s64["avg"]
    d_pr = prot["N"] / prot["avg"]
    cal["c_rot_pod"] = max(0.0, (d_pr - d_ps - prot["pf_s"]) / prot["rot"])
    return cal


def predict(rec, cal):
    hwc = cal[rec["hw"]]
    c_rot = cal["c_rot_local"] if rec["hw"] == "local3060" else cal["c_rot_pod"]
    w = min(rec["W"], rec["N"])
    decode = (w * hwc["t_k0"]
              + (rec["N"] - w) * hwc["t_ss"]
              + rec["pf_s"]
              + rec["rot"] * c_rot)
    return rec["N"] / decode


def is_calibration(rec):
    return any(f in rec["family"] and rec["stem"] == s for f, s in CAL_STEMS)


def validate(recs, cal):
    rows = []
    for r in recs:
        if is_calibration(r) or r["N"] < 96:
            continue
        pred = predict(r, cal)
        err = (pred - r["avg"]) / r["avg"] * 100.0
        # steady-state check where a last-chunk rate exists
        hwc = cal[r["hw"]]
        st_pred = 1.0 / hwc["t_ss"]
        st_err = ((st_pred - r["last"]) / r["last"] * 100.0) if r["last"] else None
        rows.append({**r, "pred": pred, "err": err,
                     "st_pred": st_pred, "st_err": st_err})
    return rows


# ---------------------------------------------------------------------------
# Scenarios (local 3060) -- differentiated quantization
# Hit-rate provenance: LRU sims J17/J31 (cap258 0.34-0.345, cap512 0.59-0.61,
# cap1024 0.74-0.76, cap2048 0.81-0.82); prompt-preload J32 (cap1024 hot-hit
# 0.849, cap512 0.681). CQ1 sizes J35: cq1g32 48.38 GiB, cq1g256 34.27 GiB vs
# native 72.56 GiB (x1.50 / x2.12 capacity, NOT x3 -- x3 needs sub-CQ1).
# ---------------------------------------------------------------------------
def t_token_ms(t_compute_ms, miss_ram, t_b_ms, miss_ssd=0.0, t_ssd_ms=0.0):
    return (t_compute_ms
            + E_PER_TOKEN * miss_ram * t_b_ms
            + E_PER_TOKEN * miss_ssd * t_ssd_ms)


def scenarios(cal):
    tc = cal["local3060"]["t_compute"] * 1000.0
    tb = cal["local3060"]["t_b"] * 1000.0
    out = []
    # (a) status quo: cache256 2-bit, resident hit ~0 measured -> all tier-b
    a = t_token_ms(tc, 1.00, tb)
    out.append(("a. status quo cache256 2-bit (resident hit~0 measured, J31)",
                a, 1000.0 / a,
                "matches measured steady 2.9-3.4 t/s; SSD cliffs stay: breath "
                "25.34 GiB/2.7 s (J28), cold storms up to 60 s/tok"))
    # (b) + CQ1 async on colds in RAM: steady path unchanged, cliffs removed
    b = t_token_ms(tc, 1.00, tb)
    out.append(("b. + CQ1 async cold-in-RAM (capacity x1.5-x2.1 measured J35)",
                b, 1000.0 / b,
                "steady unchanged (path is H2D-copy-bound); removes tier-c "
                "from working-set shifts; first recall of a cold expert costs "
                "one-off 64-74 ms IF sync, ~0 if promoted >=1 token ahead"))
    # (c) full tiering: hot native VRAM (258..407) actually hitting + warm RAM
    for cap, hit, prov in ((258, 0.34, "LRU sim J17/J31"),
                           (407, 0.50, "interp J17/J31 258->512; 407 = max "
                                       "VRAM slots per E1 note"),
                           (407, 0.60, "preload-promote upside, J32 shape")):
        c = t_token_ms(tc, 1.0 - hit, tb, miss_ssd=0.005, t_ssd_ms=54.0)
        out.append((f"c. full tiering, VRAM hot cap{cap} hit={hit:.2f} ({prov}), "
                    "0.5% frozen-SSD sync misses",
                    c, 1000.0 / c, "hot native + warm native RAM + cold CQ1 "
                    "RAM + frozen SSD"))
    # theoretical ceiling: hot-hit 0.85 (needs cap~2048-equivalent or predictor)
    d = t_token_ms(tc, 0.15, tb)
    out.append(("ceiling: hot-hit 0.85 (cap2048-equiv, J25/J31 -- NOT reachable "
                "in 12 GiB native)", d, 1000.0 / d,
                "upper bound only; requires predictor (SPEX) or compressed "
                "VRAM formats"))
    return out


def worst_case_recall(cal):
    """Single-expert SSD recall cost in ms and lost tokens at scenario-c rate."""
    rows = []
    for label, ms in (("batched/prefetched cold page-in (2.5-4.4 GiB/s)", (1.5, 2.7)),
                      ("sync miss-batch in decode path (copy_ms 50-59)", (50.0, 59.0)),
                      ("CQ1 sync fallback (decompress+repack+copy)", (64.0, 74.0)),
                      ("fault-storm tail (60 s/tok / 258)", (230.0, 230.0))):
        rows.append((label, ms))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=os.path.join("runs", "ds4"))
    ap.add_argument("--csv-out", default=None,
                    help="write per-run validation table as CSV")
    args = ap.parse_args()

    recs = load_runs(args.runs_root)
    cal = calibrate(recs)
    rows = validate(recs, cal)

    print("=" * 78)
    print("E-LAT per-tier expert recall latency (per single 6.75 MiB expert)")
    print("=" * 78)
    for tier, hw, lo, hi, prov in TIER_TABLE:
        rng = f"{lo:.2f}" if lo == hi else f"{lo:.2f}-{hi:.2f}"
        print(f"- {tier} [{hw}]: {rng} ms\n    {prov}")

    print()
    print("=" * 78)
    print("Calibration (t/s model: decode = W*t_k0 + (N-W)*t_ss + prefetch_s"
          " + rot*c_rot)")
    print("=" * 78)
    for hw in ("local3060", "pod3090", "pod4070ti"):
        c = cal[hw]
        print(f"{hw}: t_compute={c['t_compute']*1000:.1f} ms  "
              f"t_b={c['t_b']*1000:.3f} ms/expert  "
              f"t_ss={c['t_ss']*1000:.1f} ms/tok ({1/c['t_ss']:.2f} t/s)  "
              f"t_k0={c['t_k0']*1000:.0f} ms/tok")
    print(f"c_rotate: local={cal['c_rot_local']:.2f} s, pod={cal['c_rot_pod']:.2f} s")
    print(f"demand: {E_PER_TOKEN} experts/token = {GIB_PER_TOKEN:.2f} GiB/token"
          " if all recalls miss VRAM")

    print()
    print("=" * 78)
    print(f"Validation on {len(rows)} non-calibration runs (avg t/s, whole decode)")
    print("=" * 78)
    hdr = f"{'family':44s} {'stem':52s} {'hw':10s} {'N':>5s} {'act':>5s} {'pred':>5s} {'err%':>7s}"
    print(hdr)
    for r in sorted(rows, key=lambda x: abs(x["err"])):
        print(f"{r['family'][:44]:44s} {r['stem'][:52]:52s} {r['hw']:10s} "
              f"{r['N']:5.0f} {r['avg']:5.2f} {r['pred']:5.2f} {r['err']:+7.1f}")
    errs = [abs(r["err"]) for r in rows]
    st_errs = [abs(r["st_err"]) for r in rows if r["st_err"] is not None]
    print(f"\nmedian |err| avg-t/s model: {statistics.median(errs):.1f}%  "
          f"(IQR {statistics.quantiles(errs, n=4)[0]:.1f}-"
          f"{statistics.quantiles(errs, n=4)[2]:.1f}%, n={len(errs)})")
    print(f"median |err| steady-state (1/t_ss vs last-chunk): "
          f"{statistics.median(st_errs):.1f}% (n={len(st_errs)})")
    # subset: runs in the modeled regime (no cache>VRAM overcommit, no
    # stepdown/prebreath churn outside the rotate/prefetch terms)
    clean = [r for r in rows
             if not any(k in r["family"] for k in
                        ("prebreath", "stepdown", "cache512", "cache768",
                         "cache854", "rotate_smoke", "pace_"))
             and (r["cache"] is None or r["cache"] <= 258 or r["hw"] != "local3060")]
    cerrs = [abs(r["err"]) for r in clean]
    print(f"median |err| modeled-regime subset (cache<=258 local, no "
          f"stepdown/prebreath churn): {statistics.median(cerrs):.1f}% "
          f"(n={len(cerrs)})")

    print()
    print("=" * 78)
    print("Scenarios -- differentiated quantization on local 3060")
    print("=" * 78)
    for label, ms, tps, note in scenarios(cal):
        print(f"- {label}\n    t_token={ms:.0f} ms -> {tps:.2f} t/s. {note}")

    print()
    print("=" * 78)
    print("Worst-case single-expert SSD/cold recall (3060/WSL) -- the user's number")
    print("=" * 78)
    tc = cal["local3060"]
    steady_ms = tc["t_ss"] * 1000.0
    for label, (lo, hi) in worst_case_recall(cal):
        tok_lo, tok_hi = lo / steady_ms, hi / steady_ms
        rng = f"{lo:.0f} ms" if lo == hi else f"{lo:.0f}-{hi:.0f} ms"
        trng = (f"{tok_lo:.2f}" if lo == hi
                else f"{tok_lo:.2f}-{tok_hi:.2f}")
        print(f"- {label}: {rng}  (~{trng} tokens lost at "
              f"{1/tc['t_ss']:.1f} t/s steady)")

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["family", "stem", "hw", "N", "W", "rotates",
                        "prefetch_s", "avg_tps_actual", "avg_tps_pred",
                        "err_pct", "last_tps_actual", "steady_tps_pred"])
            for r in sorted(rows, key=lambda x: (x["family"], x["stem"])):
                w.writerow([r["family"], r["stem"], r["hw"], int(r["N"]),
                            int(r["W"]), int(r["rot"]), f"{r['pf_s']:.3f}",
                            r["avg"], f"{r['pred']:.3f}", f"{r['err']:.1f}",
                            r["last"] if r["last"] else "",
                            f"{r['st_pred']:.3f}"])
        print(f"\nvalidation CSV written: {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
