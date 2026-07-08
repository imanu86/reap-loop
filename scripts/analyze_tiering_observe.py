#!/usr/bin/env python3
"""Summarize DS4_EXPERT_TIERING=observe JSONL logs."""

from __future__ import annotations

import argparse
import collections
import json
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


def summarize(rows: list[dict], top: int) -> str:
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
    args = ap.parse_args()
    rows = load_rows(args.jsonl)
    print(summarize(rows, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
