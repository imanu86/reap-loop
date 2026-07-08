#!/usr/bin/env python3
"""Summarize DS4_EXPERT_TIERING=observe JSONL logs."""

from __future__ import annotations

import argparse
import collections
import json
from collections import OrderedDict
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: bad JSON: {exc}") from exc
                if row.get("event") == "tiering_observe":
                    rows.append(row)
    return rows


def isum(rows: list[dict], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def mib(n: int) -> float:
    return n / 1048576.0


def simulate_lru(rows: list[dict], cap: int) -> tuple[int, int, int]:
    if cap <= 0:
        return (0, 0, 0)
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = 0
    misses = 0
    requests = 0
    for row in rows:
        layer = int(row.get("layer") or 0)
        ids = row.get("compact_ids")
        if not isinstance(ids, list):
            continue
        for raw in ids:
            try:
                expert = int(raw)
            except (TypeError, ValueError):
                continue
            if expert < 0:
                continue
            requests += 1
            key = (layer, expert)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                misses += 1
                cache[key] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return requests, hits, misses


def capacity_cost(cap: int, slot_mib: float, scales: list[float]) -> str:
    if cap <= 0 or slot_mib <= 0:
        return ""
    native = cap * slot_mib
    parts = [f"native={native:.1f}MiB"]
    for scale in scales:
        if scale <= 0:
            continue
        parts.append(f"x{scale:g}={native * scale:.1f}MiB")
    return " " + " ".join(parts)


def summarize(
    rows: list[dict],
    top: int,
    simulate_cap: list[int],
    slot_mib: float,
    capacity_scales: list[float],
) -> str:
    if not rows:
        return "No tiering_observe rows found."

    by_path = collections.Counter(str(row.get("path") or "unknown") for row in rows)
    by_capacity = collections.Counter(int(row.get("cache_capacity") or 0) for row in rows)
    total = len(rows)
    hits = isum(rows, "hits")
    misses = isum(rows, "misses")
    direct = isum(rows, "direct")
    evictions = isum(rows, "evictions")
    lookups = hits + misses
    hit_rate = hits / lookups if lookups else 0.0
    direct_share = by_path.get("selected_direct", 0) / total

    lines = [
        f"events={total}",
        "paths=" + ", ".join(f"{k}:{v}" for k, v in by_path.most_common()),
        "cache_capacity="
        + ", ".join(f"{k}:{v}" for k, v in by_capacity.most_common()),
        (
            f"resident hit_rate={hit_rate:.4f} hits={hits} misses={misses} "
            f"direct_loads={direct} evictions={evictions}"
        ),
        (
            f"bytes direct={mib(isum(rows, 'direct_bytes')):.2f} MiB "
            f"compact={mib(isum(rows, 'compact_bytes')):.2f} MiB"
        ),
        f"selected_direct_event_share={direct_share:.3f}",
    ]

    rows_with_ids = [row for row in rows if isinstance(row.get("compact_ids"), list)]
    if rows_with_ids:
        compact_req = sum(len(row.get("compact_ids") or []) for row in rows_with_ids)
        selected_req = sum(len(row.get("selected") or []) for row in rows_with_ids)
        unique_compact = {
            (int(row.get("layer") or 0), int(expert))
            for row in rows_with_ids
            for expert in (row.get("compact_ids") or [])
            if isinstance(expert, int) and expert >= 0
        }
        unique_selected = {
            (int(row.get("layer") or 0), int(expert))
            for row in rows_with_ids
            for expert in (row.get("selected") or [])
            if isinstance(expert, int) and expert >= 0
        }
        lines.extend(
            [
                (
                    f"id_trace rows={len(rows_with_ids)} selected_req={selected_req} "
                    f"compact_req={compact_req}"
                ),
                (
                    f"id_trace unique_selected={len(unique_selected)} "
                    f"unique_compact={len(unique_compact)}"
                ),
            ]
        )
        if simulate_cap:
            lines.append("lru_sim:")
            for cap in simulate_cap:
                requests, sim_hits, sim_misses = simulate_lru(rows, cap)
                rate = sim_hits / requests if requests else 0.0
                lines.append(
                    f"  cap={cap} requests={requests} hit_rate={rate:.4f} "
                    f"hits={sim_hits} misses={sim_misses}"
                    f"{capacity_cost(cap, slot_mib, capacity_scales)}"
                )

    layer_stats: dict[int, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for row in rows:
        layer = int(row.get("layer") or 0)
        s = layer_stats[layer]
        s["events"] += 1
        for key in (
            "slots",
            "compact",
            "hits",
            "misses",
            "direct",
            "evictions",
            "direct_bytes",
            "compact_bytes",
        ):
            s[key] += int(row.get(key) or 0)
        s["max_capacity"] = max(s["max_capacity"], int(row.get("cache_capacity") or 0))

    ranked = sorted(
        layer_stats.items(),
        key=lambda item: (
            item[1]["direct"] + item[1]["misses"] + item[1]["evictions"],
            item[1]["direct_bytes"],
        ),
        reverse=True,
    )
    lines.append("worst_layers:")
    for layer, s in ranked[:top]:
        layer_lookups = s["hits"] + s["misses"]
        layer_hit_rate = s["hits"] / layer_lookups if layer_lookups else 0.0
        lines.append(
            f"  layer={layer} events={s['events']} slots={s['slots']} "
            f"compact={s['compact']} cap={s['max_capacity']} "
            f"hit_rate={layer_hit_rate:.4f} misses={s['misses']} "
            f"direct={s['direct']} evictions={s['evictions']} "
            f"direct_mib={mib(s['direct_bytes']):.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--simulate-cap", type=int, nargs="*", default=[])
    ap.add_argument(
        "--slot-mib",
        type=float,
        default=6.75,
        help="Resident expert slot size used for capacity-cost estimates.",
    )
    ap.add_argument(
        "--capacity-scale",
        type=float,
        nargs="*",
        default=[],
        help="Optional footprint multipliers for compressed tiers, e.g. 0.5 0.33.",
    )
    args = ap.parse_args()
    rows = load_rows(args.jsonl)
    print(
        summarize(
            rows,
            args.top,
            args.simulate_cap,
            args.slot_mib,
            args.capacity_scale,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
