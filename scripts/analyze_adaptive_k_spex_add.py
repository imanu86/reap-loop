#!/usr/bin/env python3
"""Offline screen for adaptive K16..K50 plus additive SPEX 1/2/4.

This measures routing coverage only. It does not grade generated quality.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics
from pathlib import Path


def read_rows(path: Path, pred: bool = False):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                pos = int(row["pos"])
                layer = int(row["layer"])
                n = int(row.get("n", 0))
                prefix = "p" if pred else "e"
                score_prefix = "s" if pred else "w"
                ids = [int(row[f"{prefix}{i}"]) for i in range(n)]
                scores = [float(row.get(f"{score_prefix}{i}", 1.0)) for i in range(n)]
            except (KeyError, TypeError, ValueError):
                continue
            if not pred:
                total = sum(max(0.0, x) for x in scores)
                if total > 0.0:
                    scores = [max(0.0, x) / total for x in scores]
            rows[(pos, layer)] = list(zip(ids, scores))
    return rows


def top_ids(score, k, excluded=frozenset()):
    ranked = sorted(
        ((e, s) for e, s in score.items() if e not in excluded),
        key=lambda item: (-item[1], item[0]),
    )
    return [e for e, _ in ranked[:k]]


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def simulate(demand, predictions, *, add, mode, window=10, kmin=16, kmax=50):
    keys_by_layer = collections.defaultdict(list)
    for pos, layer in demand:
        if layer >= 3:
            keys_by_layer[layer].append(pos)

    total_rows = total_ids = missed_ids = 0
    total_weight = missed_weight = 0.0
    add_slots = add_hits = 0
    add_weight = 0.0
    turnover = 0
    ks = []

    for layer, positions in keys_by_layer.items():
        ring = collections.deque()
        mass = collections.Counter()
        previous_d = 0.0
        previous_keep = None
        for pos in sorted(set(positions)):
            current_pairs = demand[(pos, layer)]
            current = dict(current_pairs)
            if len(ring) < window:
                ring.append(current)
                mass.update(current)
                continue

            mean = {e: value / window for e, value in mass.items()}
            union = set(mean) | set(current)
            immediate_d = 0.5 * sum(abs(current.get(e, 0.0) - mean.get(e, 0.0)) for e in union)
            used_d = immediate_d if mode == "immediate" else previous_d
            k = max(kmin, min(kmax, kmin + round((kmax - kmin) * used_d)))

            core = set(top_ids(mass, k))
            pred_score = dict(predictions.get((pos, layer), []))
            extra = set(top_ids(pred_score, add, core))
            keep = core | extra

            current_ids = set(current)
            missing = current_ids - keep
            total_rows += 1
            total_ids += len(current_ids)
            missed_ids += len(missing)
            total_weight += sum(current.values())
            missed_weight += sum(current[e] for e in missing)
            add_slots += len(extra)
            useful = extra & current_ids
            add_hits += len(useful)
            add_weight += sum(current[e] for e in useful)
            if previous_keep is not None:
                turnover += len(keep.symmetric_difference(previous_keep))
            previous_keep = keep
            ks.append(k)
            previous_d = immediate_d

            outgoing = ring.popleft()
            for e, value in outgoing.items():
                mass[e] -= value
                if abs(mass[e]) < 1e-12:
                    del mass[e]
            ring.append(current)
            mass.update(current)

    return {
        "mode": mode,
        "add": add,
        "rows": total_rows,
        "miss_ids_pct": 100.0 * missed_ids / total_ids if total_ids else 0.0,
        "miss_weight_pct": 100.0 * missed_weight / total_weight if total_weight else 0.0,
        "add_precision_pct": 100.0 * add_hits / add_slots if add_slots else 0.0,
        "add_captured_weight_pct": 100.0 * add_weight / total_weight if total_weight else 0.0,
        "avg_k": statistics.fmean(ks) if ks else 0.0,
        "p50_k": percentile(ks, 0.50),
        "p90_k": percentile(ks, 0.90),
        "max_k": max(ks, default=0),
        "turnover_per_row": turnover / total_rows if total_rows else 0.0,
    }


def simulate_knock_accel(
    demand,
    predictions,
    *,
    add,
    threshold,
    gain,
    window=10,
    warmup=16,
    kmin=16,
    kmax=50,
    controller="integral",
    anchored=True,
    max_step=4,
    max_step_down=1,
    deadband=2.0,
    scope="global",
    update_every=1,
):
    positions = sorted({pos for pos, layer in demand if layer >= 3})
    layers = sorted({layer for pos, layer in demand if layer >= 3})
    rings = {layer: collections.deque() for layer in layers}
    masses = {layer: collections.Counter() for layer in layers}
    previous_keep = {}
    anchored_core = {}
    knock_ring = collections.deque()
    current_k = kmin
    layer_k = {layer: kmin for layer in layers}
    layer_knock_ring = {layer: collections.deque() for layer in layers}

    total_rows = total_ids = missed_ids = 0
    total_weight = missed_weight = 0.0
    add_slots = add_hits = 0
    add_weight = 0.0
    turnover = 0
    ks = []
    knocks = []
    accelerations = []

    for token_index, pos in enumerate(positions):
        token_rows = []
        for layer in layers:
            pairs = demand.get((pos, layer))
            if pairs is not None:
                token_rows.append((layer, dict(pairs)))

        if token_index >= warmup:
            token_knocks = 0
            for layer, current in token_rows:
                used_k = layer_k[layer] if scope == "layer" else current_k
                if anchored:
                    core = anchored_core.setdefault(
                        layer, set(top_ids(masses[layer], used_k))
                    )
                    while len(core) < used_k:
                        candidates = top_ids(
                            masses[layer], used_k - len(core), core
                        )
                        if not candidates:
                            break
                        core.update(candidates)
                    while len(core) > used_k:
                        victim = min(
                            core, key=lambda e: (masses[layer].get(e, 0.0), e)
                        )
                        core.remove(victim)
                else:
                    core = set(top_ids(masses[layer], used_k))
                pred_score = dict(predictions.get((pos, layer), []))
                extra = set(top_ids(pred_score, add, core))
                keep = core | extra
                current_ids = set(current)
                missing = current_ids - keep

                total_rows += 1
                total_ids += len(current_ids)
                missed_ids += len(missing)
                total_weight += sum(current.values())
                missed_weight += sum(current[e] for e in missing)
                local_knocks = sum(
                    1 for e in missing if current[e] >= threshold
                )
                token_knocks += local_knocks
                add_slots += len(extra)
                useful = extra & current_ids
                add_hits += len(useful)
                add_weight += sum(current[e] for e in useful)
                if layer in previous_keep:
                    turnover += len(keep.symmetric_difference(previous_keep[layer]))
                previous_keep[layer] = keep

                if scope == "layer":
                    local_ring = layer_knock_ring[layer]
                    local_mean = statistics.fmean(local_ring) if local_ring else 0.0
                    local_accel = local_knocks - local_mean
                    due = (token_index - warmup) % update_every == 0
                    delta_k = 0 if not due or abs(local_accel) <= deadband else round(gain * local_accel)
                    delta_k = max(-max_step_down, min(max_step, delta_k))
                    layer_k[layer] = max(
                        kmin, min(kmax, used_k + delta_k)
                    )
                    local_ring.append(local_knocks)
                    if len(local_ring) > window:
                        local_ring.popleft()
                    ks.append(used_k)
                    knocks.append(local_knocks)
                    accelerations.append(local_accel)

            if scope == "global":
                mean_knocks = statistics.fmean(knock_ring) if knock_ring else 0.0
                acceleration = token_knocks - mean_knocks
                if controller == "integral":
                    due = (token_index - warmup) % update_every == 0
                    delta_k = 0 if not due or abs(acceleration) <= deadband else round(gain * acceleration)
                    delta_k = max(-max_step_down, min(max_step, delta_k))
                    next_k = current_k + delta_k
                else:
                    next_k = kmin + round(gain * max(0.0, acceleration))
                next_k = max(kmin, min(kmax, next_k))
                ks.extend([current_k] * len(token_rows))
                knocks.append(token_knocks)
                accelerations.append(acceleration)
                knock_ring.append(token_knocks)
                if len(knock_ring) > window:
                    knock_ring.popleft()
                current_k = next_k

        for layer, current in token_rows:
            ring = rings[layer]
            mass = masses[layer]
            ring.append(current)
            mass.update(current)
            if len(ring) > window:
                outgoing = ring.popleft()
                for e, value in outgoing.items():
                    mass[e] -= value
                    if abs(mass[e]) < 1e-12:
                        del mass[e]

    return {
        "threshold": threshold,
        "gain": gain,
        "add": add,
        "controller": controller,
        "anchored": int(anchored),
        "max_step": max_step,
        "max_step_down": max_step_down,
        "deadband": deadband,
        "scope": scope,
        "update_every": update_every,
        "rows": total_rows,
        "miss_ids_pct": 100.0 * missed_ids / total_ids if total_ids else 0.0,
        "miss_weight_pct": 100.0 * missed_weight / total_weight if total_weight else 0.0,
        "add_precision_pct": 100.0 * add_hits / add_slots if add_slots else 0.0,
        "add_captured_weight_pct": 100.0 * add_weight / total_weight if total_weight else 0.0,
        "avg_k": statistics.fmean(ks) if ks else 0.0,
        "p90_k": percentile(ks, 0.90),
        "at_k50_pct": 100.0 * sum(k == kmax for k in ks) / len(ks) if ks else 0.0,
        "avg_knocks": statistics.fmean(knocks) if knocks else 0.0,
        "p90_knocks": percentile(knocks, 0.90),
        "positive_accel_pct": 100.0 * sum(a > 0 for a in accelerations) / len(accelerations) if accelerations else 0.0,
        "turnover_per_row": turnover / total_rows if total_rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--policy", choices=("tv", "knock"), default="knock")
    parser.add_argument("--thresholds", default="0.10,0.15,0.20")
    parser.add_argument("--gains", default="0.5,1,2,4")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--controller", choices=("direct", "integral"), default="integral")
    parser.add_argument("--anchored", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-step", type=int, default=4)
    parser.add_argument("--max-step-down", type=int, default=1)
    parser.add_argument("--deadband", type=float, default=2.0)
    parser.add_argument("--scope", choices=("global", "layer"), default="global")
    parser.add_argument("--update-every", type=int, default=1)
    args = parser.parse_args()

    demand = read_rows(args.trace, pred=False)
    predictions = read_rows(args.predictions, pred=True) if args.predictions else {}
    if args.policy == "tv":
        fields = [
            "mode", "add", "rows", "miss_ids_pct", "miss_weight_pct",
            "add_precision_pct", "add_captured_weight_pct", "avg_k", "p50_k",
            "p90_k", "max_k", "turnover_per_row",
        ]
        results = (
            simulate(demand, predictions, add=add, mode=mode, window=args.window)
            for mode in ("lag1", "immediate")
            for add in (0, 1, 2, 4)
        )
    else:
        fields = [
            "threshold", "gain", "add", "controller", "anchored", "max_step", "max_step_down", "deadband", "scope", "update_every", "rows", "miss_ids_pct",
            "miss_weight_pct", "add_precision_pct", "add_captured_weight_pct",
            "avg_k", "p90_k", "at_k50_pct", "avg_knocks", "p90_knocks",
            "positive_accel_pct", "turnover_per_row",
        ]
        thresholds = [float(x) for x in args.thresholds.split(",") if x]
        gains = [float(x) for x in args.gains.split(",") if x]
        results = (
            simulate_knock_accel(
                demand,
                predictions,
                add=add,
                threshold=threshold,
                gain=gain,
                window=args.window,
                warmup=args.warmup,
                controller=args.controller,
                anchored=bool(args.anchored),
                max_step=args.max_step,
                max_step_down=args.max_step_down,
                deadband=args.deadband,
                scope=args.scope,
                update_every=max(1, args.update_every),
            )
            for threshold in thresholds
            for gain in gains
            for add in (0, 1, 2, 4)
        )
    print(",".join(fields))
    for result in results:
        print(",".join(
            str(result[name]) if isinstance(result[name], (int, str))
            else f"{result[name]:.6f}"
            for name in fields
        ))


if __name__ == "__main__":
    main()
