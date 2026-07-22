#!/usr/bin/env python3
"""Coffee per-layer mass coverage curve (decisive test, 2026-07-13).

Uses the IDENTICAL mass definition as build_mass_mask.py:
  per-layer expert mass = sum of routing weight w over every occurrence of
  expert E at layer L (top-6 slots), across the trace, layers 3..42.

Reports, for the coffee K0 trace:
  - median distinct experts touched / layer
  - for a per-layer coverage target c, the per-layer count of top-by-mass
    experts needed to reach c (variable) -> median/mean of that count
  - for a FIXED K (mask60_self semantics), the aggregate mass coverage =
    sum over layers of (mass of top-K) / sum over layers of (total mass),
    plus mean per-layer coverage. Finds the fixed K that hits 90% / 92.5%.
"""
import collections, csv, gzip, sys, statistics

NUM_EXPERTS = 256
MOE_LAYER_MIN, MOE_LAYER_MAX = 3, 42


def load_mass(path):
    mass = collections.defaultdict(lambda: collections.defaultdict(float))
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 15:
                continue
            try:
                layer = int(row[1]); n = int(row[2])
            except Exception:
                continue
            if layer < MOE_LAYER_MIN or layer > MOE_LAYER_MAX:
                continue
            for i in range(min(n, 6)):
                try:
                    e = int(row[3 + i]); w = float(row[9 + i])
                except Exception:
                    continue
                mass[layer][e] += w
    return mass


def main():
    path = sys.argv[1]
    mass = load_mass(path)
    layers = list(range(MOE_LAYER_MIN, MOE_LAYER_MAX + 1))

    distinct = [len(mass.get(L, {})) for L in layers]
    print(f"layers with data: {sum(1 for L in layers if L in mass)}/{len(layers)}")
    print(f"distinct experts touched/layer: median={statistics.median(distinct):.0f} "
          f"min={min(distinct)} max={max(distinct)}")

    # per-layer sorted masses (desc)
    sorted_mass = {}
    layer_total = {}
    for L in layers:
        lm = mass.get(L, {})
        vals = sorted(lm.values(), reverse=True)
        sorted_mass[L] = vals
        layer_total[L] = sum(vals)

    def experts_to_cover(L, c):
        tot = layer_total[L]
        if tot <= 0:
            return 0
        acc = 0.0
        for i, v in enumerate(sorted_mass[L], 1):
            acc += v
            if acc >= c * tot:
                return i
        return len(sorted_mass[L])

    for c in (0.90, 0.925):
        counts = [experts_to_cover(L, c) for L in layers]
        print(f"[variable-coverage {c*100:.1f}%] experts/layer needed: "
              f"median={statistics.median(counts):.0f} mean={statistics.mean(counts):.1f} "
              f"min={min(counts)} max={max(counts)}")

    def fixed_k_cov(K):
        # aggregate mass coverage and mean per-layer coverage for fixed top-K
        num = sum(sum(sorted_mass[L][:K]) for L in layers)
        den = sum(layer_total[L] for L in layers)
        per = [ (sum(sorted_mass[L][:K]) / layer_total[L]) if layer_total[L] > 0 else 0.0
                for L in layers ]
        return num / den, statistics.mean(per)

    print("\nfixed-K (mask60_self semantics) coverage:")
    print(f"  {'K':>4} {'agg_cov':>8} {'mean_layer_cov':>14} {'window_GiB@11MiB':>16}")
    # find K crossing targets by aggregate coverage
    cross = {0.90: None, 0.925: None}
    for K in range(40, 130):
        agg, meanc = fixed_k_cov(K)
        for tgt in cross:
            if cross[tgt] is None and agg >= tgt:
                cross[tgt] = K
    for K in [60, 65, 70, 71, 72, 75, 80, 83, 85, 90, 102, 154]:
        agg, meanc = fixed_k_cov(K)
        gib = K * 40 * 11 / 1024.0
        print(f"  {K:>4} {agg*100:>7.2f}% {meanc*100:>13.2f}% {gib:>15.1f}")
    print(f"\n  fixed-K crossing 90.0% aggregate mass: K={cross[0.90]}")
    print(f"  fixed-K crossing 92.5% aggregate mass: K={cross[0.925]}")


if __name__ == "__main__":
    main()
