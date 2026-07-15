#!/usr/bin/env python3
"""Measure how well a REAP mask covers a weighted DS4 routing trace."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


N_EXPERT = 256


def load_retained(mask_path: pathlib.Path) -> dict[int, set[int]]:
    blocked: dict[int, set[int]] = {}
    with mask_path.open(encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{mask_path}:{lineno}: expected layer and expert")
            layer, expert = map(int, fields)
            if layer < 0 or expert < 0 or expert >= N_EXPERT:
                raise ValueError(f"{mask_path}:{lineno}: invalid layer/expert")
            blocked.setdefault(layer, set()).add(expert)
    if not blocked:
        raise ValueError(f"empty mask: {mask_path}")
    return {layer: set(range(N_EXPERT)) - ids for layer, ids in blocked.items()}


def score(trace_path: pathlib.Path, mask_path: pathlib.Path) -> dict:
    retained = load_retained(mask_path)
    calls = hits = complete_rows = rows = 0
    total_mass = hit_mass = 0.0
    by_layer: dict[int, dict[str, float | int]] = {}

    with trace_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            layer = int(row["layer"])
            if layer not in retained:
                continue
            n = int(row["n"])
            layer_stats = by_layer.setdefault(
                layer, {"calls": 0, "hits": 0, "mass": 0.0, "hit_mass": 0.0}
            )
            row_complete = True
            rows += 1
            for index in range(n):
                expert = int(row[f"e{index}"])
                weight = float(row[f"w{index}"])
                is_hit = expert in retained[layer]
                calls += 1
                total_mass += weight
                layer_stats["calls"] += 1
                layer_stats["mass"] += weight
                if is_hit:
                    hits += 1
                    hit_mass += weight
                    layer_stats["hits"] += 1
                    layer_stats["hit_mass"] += weight
                else:
                    row_complete = False
            complete_rows += int(row_complete)

    if calls == 0 or total_mass == 0.0:
        raise ValueError("trace has no rows for maskable layers")

    layer_rows = []
    for layer, stats in sorted(by_layer.items()):
        layer_rows.append({
            "layer": layer,
            "calls": stats["calls"],
            "misses": stats["calls"] - stats["hits"],
            "call_coverage": stats["hits"] / stats["calls"],
            "mass_coverage": stats["hit_mass"] / stats["mass"],
        })

    return {
        "trace": str(trace_path),
        "mask": str(mask_path),
        "rows": rows,
        "selected_calls": calls,
        "misses": calls - hits,
        "call_coverage": hits / calls,
        "mass_coverage": hit_mass / total_mass,
        "all_selected_row_rate": complete_rows / rows,
        "minimum_layer_mass_coverage": min(
            item["mass_coverage"] for item in layer_rows
        ),
        "layers": layer_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--mask", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    result = score(args.trace, args.mask)
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
