#!/usr/bin/env python3
"""Tripwire monitor for the counterfactual-admission (0046) test protocol.

Watches a running ds4-server's LIVEMASK JSONL log (adaptive_k / cf_admit
events) and the client's stream_events.jsonl in real time, and kills the
server (by PID, never pkill -f) the moment one of three "K0-with-extra-steps"
tripwires fires:

  1. UNION: the cumulative set of (layer, expert) ever admitted by the
     counterfactual-admission actuator, for any single layer, exceeds
     `--union-pct-limit` percent of the expert pool. This is the direct
     signature of the adaptive mask degenerating into "eventually everyone is
     admitted somewhere" = K0 in disguise.
  2. CHURN: the admit-events-per-token rate, bucketed in windows of
     `--admit-bucket-tokens` tokens, never decays after a peak -- i.e. more
     than `--admit-nondecay-tokens` tokens after the peak bucket, the recent
     bucket rate is still >= 50% of the peak. Constant high churn after a
     phase transition should settle; if it does not, this is rotate-style
     behavior wearing an adaptive-K costume.
  3. SPEED: decode throughput (estimated from stream_events.jsonl SSE
     arrival timestamps) stays below `--tps-min` for more than
     `--tps-window-tokens` consecutive observed tokens.

It also watches for an external WATCHDOG_KILL.txt (written by a separate,
coordinator-run watchdog that judges output degeneration) and for the
run's own response.json (normal completion) to know when to stop watching.

This script does not judge text quality. It only judges the mechanism: does
the counterfactual-admission actuator stay a *bounded, decaying, on-signal*
process, or does it decay into the old rotate/K0 pathology.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def read_new_lines(fh, state_key, pos_state):
    """Read whatever new complete lines are available since last read."""
    pos = pos_state.get(state_key, 0)
    fh.seek(pos)
    lines = fh.readlines()
    # Only keep complete lines (last one may be a partial write-in-progress).
    if lines and not lines[-1].endswith("\n"):
        partial = lines.pop()
        pos_state[state_key] = fh.tell() - len(partial.encode("utf-8", "replace"))
    else:
        pos_state[state_key] = fh.tell()
    return [ln.strip() for ln in lines if ln.strip()]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def kill_pid(pid: int, log):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log(f"pid {pid} already gone at SIGTERM time")
        return
    except Exception as exc:  # noqa: BLE001
        log(f"SIGTERM failed on pid {pid}: {exc}")
    for _ in range(20):
        time.sleep(0.25)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"pid {pid} exited after SIGTERM")
            return
    try:
        os.kill(pid, signal.SIGKILL)
        log(f"pid {pid} did not exit; sent SIGKILL")
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--pid-file", type=Path, required=True)
    ap.add_argument("--livemask-log", type=Path, required=True)
    ap.add_argument("--events-log", type=Path, required=True)
    ap.add_argument("--response", type=Path, required=True)
    ap.add_argument("--watchdog-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--union-pct-limit", type=float, default=50.0)
    ap.add_argument("--tps-min", type=float, default=1.5)
    ap.add_argument("--tps-window-tokens", type=int, default=50)
    ap.add_argument("--admit-bucket-tokens", type=int, default=25)
    ap.add_argument("--admit-nondecay-tokens", type=int, default=150)
    ap.add_argument("--admit-peak-floor", type=float, default=5.0,
                     help="minimum peak bucket rate before non-decay can fire "
                          "(avoids false triggers on trivially low churn)")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--max-wall-s", type=float, default=7500.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pos_state = {}
    k_samples = []
    admit_layer_sets = {}   # layer -> set(expert)
    admit_tok_buckets = {}  # bucket_idx -> count
    admit_events_total = 0
    cf_admit_source_counts = {"cf": 0, "mass10": 0}
    tok_events = []  # (t_s, event_count) from stream_events.jsonl

    def log(msg):
        print(f"[cf_tripwire_monitor] {msg}", file=sys.stderr, flush=True)

    def write_summary(extra=None):
        k_sorted = sorted(k_samples)
        summary = {
            "run_dir": str(args.run_dir),
            "elapsed_s": round(time.time() - t0, 1),
            "k_updates": len(k_samples),
            "k_avg": (sum(k_samples) / len(k_samples)) if k_samples else None,
            "k_p90": percentile(k_sorted, 90),
            "k_max": max(k_samples) if k_samples else None,
            "admit_events_total": admit_events_total,
            "admit_source_counts": dict(cf_admit_source_counts),
            "union_by_layer_top5": sorted(
                ((layer, len(s)) for layer, s in admit_layer_sets.items()),
                key=lambda kv: -kv[1],
            )[:5],
            "union_max_count": max((len(s) for s in admit_layer_sets.values()), default=0),
            "union_max_pct": round(
                100.0 * max((len(s) for s in admit_layer_sets.values()), default=0)
                / args.pool_size,
                2,
            ),
            "admit_bucket_rates": dict(sorted(admit_tok_buckets.items())),
            "tps_recent": tps_recent(),
        }
        if extra:
            summary.update(extra)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def tps_recent():
        if len(tok_events) < 2:
            return None
        window = tok_events[-args.tps_window_tokens:]
        if len(window) < 2:
            return None
        dt = window[-1][0] - window[0][0]
        if dt <= 0:
            return None
        return round((len(window) - 1) / dt, 3)

    def trigger(reason, detail):
        stop_obj = {
            "reason": reason,
            "detail": detail,
            "elapsed_s": round(time.time() - t0, 1),
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.stop_file.write_text(json.dumps(stop_obj, ensure_ascii=False, indent=2))
        write_summary({"tripwire_triggered": stop_obj})
        log(f"TRIPWIRE FIRED: {reason} :: {detail}")
        pid = read_pid()
        if pid:
            kill_pid(pid, log)

    def read_pid():
        try:
            return int(args.pid_file.read_text().strip())
        except Exception:  # noqa: BLE001
            return None

    t0 = time.time()
    log(f"watching run-dir={args.run_dir}")

    below_tps_since = None
    lm_fh = None
    ev_fh = None

    while True:
        now = time.time()
        if now - t0 > args.max_wall_s:
            trigger("max_wall_time", f"exceeded {args.max_wall_s}s without completion")
            break
        if args.watchdog_file.exists():
            try:
                wd = json.loads(args.watchdog_file.read_text())
            except Exception:  # noqa: BLE001
                wd = {"raw": args.watchdog_file.read_text()[:500]}
            write_summary({"watchdog_triggered": wd})
            log(f"external WATCHDOG_KILL.txt observed: {wd}")
            break
        if args.response.exists():
            write_summary({"completed_normally": True})
            log("response.json present; run completed, stopping monitor")
            break

        # -- tail livemask.jsonl --
        if lm_fh is None and args.livemask_log.exists():
            lm_fh = args.livemask_log.open("r", encoding="utf-8", errors="replace")
        if lm_fh is not None:
            for line in read_new_lines(lm_fh, "lm", pos_state):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = obj.get("ev")
                if ev == "adaptive_k":
                    if "new_k" in obj:
                        k_samples.append(obj["new_k"])
                elif ev == "cf_admit":
                    admit_events_total += 1
                    layer = obj.get("layer")
                    expert = obj.get("expert")
                    src = obj.get("source", "mass10")
                    cf_admit_source_counts[src] = cf_admit_source_counts.get(src, 0) + 1
                    if layer is not None and expert is not None:
                        admit_layer_sets.setdefault(layer, set()).add(expert)
                    tok = obj.get("tok", 0)
                    bucket = tok // args.admit_bucket_tokens
                    admit_tok_buckets[bucket] = admit_tok_buckets.get(bucket, 0) + 1

        # -- tail stream_events.jsonl for t/s --
        if ev_fh is None and args.events_log.exists():
            ev_fh = args.events_log.open("r", encoding="utf-8", errors="replace")
        if ev_fh is not None:
            for line in read_new_lines(ev_fh, "ev", pos_state):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t_s = obj.get("t_s")
                if t_s is not None:
                    tok_events.append((t_s, obj.get("event")))

        # -- tripwire 1: union --
        union_max = max((len(s) for s in admit_layer_sets.values()), default=0)
        union_max_pct = 100.0 * union_max / args.pool_size
        if union_max_pct > args.union_pct_limit:
            worst_layer = max(admit_layer_sets.items(), key=lambda kv: len(kv[1]))[0]
            trigger(
                "union_to_k0",
                f"layer {worst_layer} union={union_max}/{args.pool_size} "
                f"({union_max_pct:.1f}% > {args.union_pct_limit}%)",
            )
            break

        # -- tripwire 2: non-decaying churn --
        if admit_tok_buckets:
            peak_bucket, peak_rate = max(admit_tok_buckets.items(), key=lambda kv: kv[1])
            last_bucket = max(admit_tok_buckets.keys())
            tokens_since_peak = (last_bucket - peak_bucket) * args.admit_bucket_tokens
            if (
                peak_rate >= args.admit_peak_floor
                and tokens_since_peak >= args.admit_nondecay_tokens
            ):
                recent_rate = admit_tok_buckets.get(last_bucket, 0)
                if recent_rate >= 0.5 * peak_rate:
                    trigger(
                        "churn_no_decay",
                        f"peak_rate={peak_rate}@bucket{peak_bucket}, "
                        f"recent_rate={recent_rate}@bucket{last_bucket}, "
                        f"{tokens_since_peak} tok since peak without decay <50%",
                    )
                    break

        # -- tripwire 3: sustained low t/s --
        cur_tps = tps_recent()
        if cur_tps is not None:
            if cur_tps < args.tps_min:
                if below_tps_since is None:
                    below_tps_since = len(tok_events)
                elif len(tok_events) - below_tps_since > args.tps_window_tokens:
                    trigger(
                        "speed_to_k0",
                        f"t/s={cur_tps} < {args.tps_min} sustained over "
                        f">{args.tps_window_tokens} tokens",
                    )
                    break
            else:
                below_tps_since = None

        write_summary()
        time.sleep(args.poll_interval)

    if lm_fh:
        lm_fh.close()
    if ev_fh:
        ev_fh.close()


if __name__ == "__main__":
    main()
