#!/usr/bin/env python3
"""Stream a completion and fail closed if the SSE response is incomplete."""
import argparse
import json
import math
import pathlib
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--request", required=True)
ap.add_argument("--out", required=True)         # summary json path
ap.add_argument("--live", required=True)         # live text path
ap.add_argument("--content", required=True)      # exact UTF-8 response bytes
ap.add_argument("--drop", type=int, default=32)  # decode-ramp tokens to drop from steady-state
args = ap.parse_args()

req = json.load(open(args.request, encoding="utf-8"))
data = json.dumps(req).encode("utf-8")
r = urllib.request.Request(args.url, data=data, headers={"Content-Type": "application/json"})

t_start = time.monotonic()
tok_times = []   # monotonic time when each content delta arrived
pieces = []
usage = None
saw_done = False
finish_reasons = []
stream_errors = []
with open(args.live, "w", encoding="utf-8") as live:
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for raw in resp:
            try:
                line = raw.decode("utf-8", "strict").strip()
            except UnicodeDecodeError:
                stream_errors.append("invalid_utf8")
                break
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                stream_errors.append("malformed_sse_line")
                break
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                obj = json.loads(payload)
            except (TypeError, ValueError):
                stream_errors.append("invalid_sse_json")
                break
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason") is not None:
                    finish_reasons.append(ch["finish_reason"])
                delta = ch.get("delta") or {}
                c = delta.get("content")
                if c:
                    tok_times.append(time.monotonic())
                    pieces.append(c)
                    live.write(c)
                    live.flush()
content = "".join(pieces)
pathlib.Path(args.content).write_bytes(content.encode("utf-8"))

n = len(tok_times)
t_first = (tok_times[0] - t_start) if n else None
summary = {
    "n_deltas": n,
    "ttft_s": t_first,
    "usage": usage,
    "saw_done": saw_done,
    "finish_reasons": finish_reasons,
    "stream_errors": stream_errors,
    "content_bytes": len(content.encode("utf-8")),
}
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
errors = []
errors.extend(stream_errors)
if not saw_done:
    errors.append("missing_sse_done")
completion_tokens = usage.get("completion_tokens") if usage else None
if not isinstance(completion_tokens, int) or completion_tokens <= 0:
    errors.append("missing_completion_usage")
if not any(isinstance(reason, str) and reason for reason in finish_reasons):
    errors.append("missing_finish_reason")
one_delta_per_token = isinstance(completion_tokens, int) and completion_tokens > 0 and n == completion_tokens
summary["one_delta_per_token"] = one_delta_per_token
if not one_delta_per_token:
    errors.append("delta_token_contract_mismatch")
if n <= args.drop + 8:
    errors.append("insufficient_stream_deltas")
for key in ("ttft_s", "steady_decode_tps", "full_decode_tps"):
    value = summary.get(key)
    if value is None or not math.isfinite(value) or value <= 0:
        errors.append(f"invalid_{key}")
summary["validation_errors"] = errors
summary["valid"] = not errors

open(args.out, "w", encoding="utf-8").write(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if errors:
    raise SystemExit(2)
