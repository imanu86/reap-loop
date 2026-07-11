#!/usr/bin/env python3
"""Verify mask enforcement in an emitted route trace: every picked expert (e0..e5)
must be in the keep-set for its layer. Prints violation rate + distinct-used/layer."""
import csv, json, sys
from collections import defaultdict

route_csv, mask_json = sys.argv[1], sys.argv[2]
keep = {int(k): set(v) for k, v in json.load(open(mask_json))["keep"].items()}
picks = 0
viol = 0
used = defaultdict(set)
with open(route_csv, newline="") as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        if len(r) < 9:
            continue
        L = int(r[1])
        for s in range(min(int(r[2]), 6)):
            e = int(r[3 + s])
            if e < 0:
                continue
            picks += 1
            used[L].add(e)
            if L in keep and e not in keep[L]:
                viol += 1
du = [len(v) for v in used.values()]
avg = sum(du) / len(du) if du else 0
print(f"{route_csv.split('/')[-2]}: picks={picks} violations={viol} "
      f"({100*viol/max(picks,1):.3f}% on pruned) distinct-used/layer avg={avg:.1f} "
      f"min={min(du)} max={max(du)}")
