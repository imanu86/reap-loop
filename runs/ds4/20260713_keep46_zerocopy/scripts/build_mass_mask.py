#!/usr/bin/env python3
"""Build a per-layer top-K-by-mass REAP bias mask ("virtual bake" study, 2026-07-12).

Reads one or more DS4_SPEX_TRACE_ROUTING(+_WEIGHTS) route.csv traces
(header: pos,layer,n,e0,e1,e2,e3,e4,e5,w0,w1,w2,w3,w4,w5), aggregates the
per-layer expert mass = sum of routing weight w over every occurrence of
expert E at layer L across every input trace, then keeps the top `--keep`
experts per layer by mass (fixed count, ties broken by expert id asc for
determinism). Every other expert in [0,256) at that layer is written to the
output mask (BLOCKED expert format expected by DS4_REAP_MASK_FILE: one
"<layer> <expert>" per line, no header).

No numpy dependency (oracle50 hit a missing-numpy failure on this box on a
later step; keep this pure stdlib).

Usage:
  build_mass_mask.py --keep 141 --out mask55_family.txt trace1/route.csv trace2/route.csv ...
"""
import argparse
import collections
import csv
import sys

NUM_EXPERTS = 256
MOE_LAYER_MIN, MOE_LAYER_MAX = 3, 42  # 40 MoE layers


def load_mass(paths):
    mass = collections.defaultdict(lambda: collections.defaultdict(float))
    rows_total = 0
    for p in paths:
        rows = 0
        skipped = 0
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            next(r, None)  # header
            for row in r:
                if len(row) < 15:
                    skipped += 1
                    continue
                try:
                    layer = int(row[1])
                    n = int(row[2])
                except Exception:
                    skipped += 1
                    continue
                if layer < MOE_LAYER_MIN or layer > MOE_LAYER_MAX:
                    continue
                ok = False
                for i in range(min(n, 6)):
                    try:
                        e = int(row[3 + i])
                        w = float(row[9 + i])
                    except Exception:
                        continue
                    mass[layer][e] += w
                    ok = True
                if ok:
                    rows += 1
        rows_total += rows
        print(f"[mass] {p}: rows_used={rows} rows_skipped={skipped}", file=sys.stderr)
    print(f"[mass] TOTAL rows_used={rows_total} layers_with_data={len(mass)}", file=sys.stderr)
    return mass


def build_mask(mass, keep):
    lines = []
    per_layer_keep = {}
    for L in range(MOE_LAYER_MIN, MOE_LAYER_MAX + 1):
        layer_mass = mass.get(L, {})
        ranked = sorted(range(NUM_EXPERTS), key=lambda e: (-layer_mass.get(e, 0.0), e))
        keep_set = set(ranked[:keep])
        per_layer_keep[L] = len(keep_set)
        for e in range(NUM_EXPERTS):
            if e not in keep_set:
                lines.append(f"{L} {e}")
    return lines, per_layer_keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="route.csv trace file(s), mass is summed across all")
    ap.add_argument("--keep", type=int, required=True, help="experts kept per layer (fixed count)")
    ap.add_argument("--out", required=True, help="output mask file (blocked-expert format)")
    args = ap.parse_args()

    mass = load_mass(args.traces)
    n_layers_moe = MOE_LAYER_MAX - MOE_LAYER_MIN + 1
    missing = [L for L in range(MOE_LAYER_MIN, MOE_LAYER_MAX + 1) if L not in mass]
    if missing:
        print(f"[mask] WARNING: no data for layers {missing} (top-{args.keep} on those layers "
              f"is UNDEFINED -> tie-break by expert-id ASC, i.e. keeps 0..{args.keep-1})",
              file=sys.stderr)

    lines, per_layer_keep = build_mask(mass, args.keep)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    pct = 100.0 * args.keep / NUM_EXPERTS
    expect_lines = (NUM_EXPERTS - args.keep) * n_layers_moe
    keeps = list(per_layer_keep.values())
    print(f"[mask] out={args.out}")
    print(f"[mask] keep={args.keep} ({pct:.1f}%) fixed per layer, {n_layers_moe} MoE layers")
    print(f"[mask] blocked_lines={len(lines)} (expected {expect_lines})")
    print(f"[mask] per-layer keep: min={min(keeps)} max={max(keeps)} "
          f"(should both == {args.keep} for the fixed-K scheme)")
    if len(lines) != expect_lines:
        print("[mask] ERROR: line count mismatch!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
