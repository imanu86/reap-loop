"""Parse v2 gate-a artifacts: rewind events + wall-time cost of the restore+rewind cycle.

Input: a_run.log with lines prefixed by ts.py monotonic seconds, containing
  - "ds4: gpu decode eval N took X ms" lines (DS4_TOKEN_TIMING=1)
  - "ds4: PACE rewind ..." debug lines (DS4_PACE_DEBUG=1)
and a_events.jsonl with rewind_arm / rewind events.

Cost model:
  - regen_tokens = from - to (positions discarded and re-decoded)
  - cycle_wall  = timestamp gap between the last eval line BEFORE the 'rewind'
                  debug line and the first eval line AFTER it, minus the median
                  inter-eval gap (the eval that would have happened anyway).
"""
import json, re, sys, statistics

log_path, ev_path = sys.argv[1], sys.argv[2]

evals = []   # (ts, eval_idx, ms)
marks = []   # (ts, line) for rewind debug lines
pat_eval = re.compile(r"^\s*([0-9.]+)\s+ds4: gpu decode eval (\d+) took ([0-9.]+) ms")
pat_rw   = re.compile(r"^\s*([0-9.]+)\s+ds4: PACE (rewind(?:_arm|_skip)?)\s")
for line in open(log_path, encoding="utf-8", errors="ignore"):
    m = pat_eval.match(line)
    if m:
        evals.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
        continue
    m = pat_rw.match(line)
    if m:
        marks.append((float(m.group(1)), m.group(2)))

rewinds = []
for l in open(ev_path, encoding="utf-8", errors="ignore"):
    try:
        d = json.loads(l)
    except Exception:
        continue
    if d.get("ev") == "rewind":
        rewinds.append(d)

print(f"eval_lines={len(evals)}  rewind_debug_marks={[m[1] for m in marks]}")
print(f"rewind_events={len(rewinds)}")
for d in rewinds:
    print("  ", json.dumps(d))

if not evals:
    sys.exit("no eval timing lines — was DS4_TOKEN_TIMING=1 set?")

gaps = [b[0] - a[0] for a, b in zip(evals, evals[1:])]
med_gap = statistics.median(gaps)
print(f"median_inter_eval_gap_s={med_gap:.4f}  (n={len(gaps)})")

fire_marks = [m for m in marks if m[1] == "rewind"]
for ts, _ in fire_marks:
    before = [e for e in evals if e[0] < ts]
    after  = [e for e in evals if e[0] >= ts]
    if not before or not after:
        print("fire at", ts, ": missing surrounding evals")
        continue
    gap = after[0][0] - before[-1][0]
    cycle = gap - med_gap
    print(f"fire@{ts:.3f}s: last_eval_before={before[-1][0]:.3f} first_eval_after={after[0][0]:.3f} "
          f"raw_gap={gap:.3f}s  cycle_wall≈{cycle:.3f}s (raw_gap - median_gap)")

# regen decode time: regen_tokens * median eval wall (the re-decode of the reverted span)
if rewinds:
    for d in rewinds:
        regen = d["from"] - d["to"]
        print(f"rewind from={d['from']} to={d['to']}: regen_tokens={regen}  "
              f"regen_decode_est={regen * med_gap:.2f}s @ median gap")
