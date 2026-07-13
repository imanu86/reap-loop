#!/usr/bin/env python3
"""Stream a chat completion, record per-token wall times, compute steady-state decode t/s."""
import argparse, json, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--request", required=True)
ap.add_argument("--out", required=True)         # summary json path
ap.add_argument("--live", required=True)         # live text path
ap.add_argument("--drop", type=int, default=32)  # decode-ramp tokens to drop from steady-state
args = ap.parse_args()

req = json.load(open(args.request, encoding="utf-8"))
data = json.dumps(req).encode("utf-8")
r = urllib.request.Request(args.url, data=data, headers={"Content-Type": "application/json"})

t_start = time.monotonic()
tok_times = []   # monotonic time when each content delta arrived
pieces = []
usage = None
live = open(args.live, "w", encoding="utf-8")
with urllib.request.urlopen(r, timeout=3600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            delta = ch.get("delta") or {}
            c = delta.get("content")
            if c:
                tok_times.append(time.monotonic())
                pieces.append(c)
                live.write(c); live.flush()
live.close()

n = len(tok_times)
t_first = (tok_times[0] - t_start) if n else None
summary = {"n_deltas": n, "ttft_s": t_first, "usage": usage}
if n > args.drop + 8:
    a = tok_times[args.drop]
    b = tok_times[-1]
    steady_n = (n - 1) - args.drop
    steady_tps = steady_n / (b - a) if b > a else None
    summary["steady_decode_tps"] = steady_tps
    summary["steady_tokens"] = steady_n
    summary["steady_window_s"] = b - a
    # also full-decode t/s (excluding ttft)
    if n > 1:
        summary["full_decode_tps"] = (n - 1) / (tok_times[-1] - tok_times[0])
open(args.out, "w", encoding="utf-8").write(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
