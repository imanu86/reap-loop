#!/usr/bin/env python3
"""Summarize DS4 cache sweep runs from runner artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_run(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary_path = run_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    for row in json.loads(summary_path.read_text(encoding="utf-8")):
        leaf = run_root / row["stem"]
        env = json.loads((leaf / "server_env.json").read_text(encoding="utf-8"))
        manifest = json.loads((leaf / "runner_manifest.json").read_text(encoding="utf-8"))
        log = (leaf / "server.stderr.log").read_text(encoding="utf-8", errors="replace")
        tier = re.search(
            r"expert tiering observe summary .*?hits=(\d+) misses=(\d+) "
            r"direct=(\d+) evictions=(\d+).*?direct=([0-9.]+) MiB compact=([0-9.]+) MiB",
            log,
        )
        spex = re.search(
            r"SPEX stats: .*?cache_hits=(\d+) cache_misses=(\d+).*?"
            r"direct_loads=(\d+).*?copy_ms_per_batch=([0-9.]+).*?"
            r"sync_ms_per_batch=([0-9.]+)",
            log,
        )
        rows.append(
            {
                "suite": run_root.name,
                "stem": row["stem"],
                "prompt": row["prompt"],
                "variant": row["variant"],
                "cache": manifest["server"]["cache_experts"],
                "pace_target": env.get("DS4_PACE_CACHE_TARGET_SLOTS"),
                "wall_s": row["wall_s"],
                "prompt_s": row["prompt_s"],
                "first50_tps": row["first50_tps"],
                "avg_tps": row["avg_tps"],
                "last_tps": row["last_chunk_tps"],
                "completion_tokens": row["completion_tokens"],
                "repeat": row["repeat_flag"],
                "tier_hits": tier.group(1) if tier else "",
                "tier_misses": tier.group(2) if tier else "",
                "tier_direct": tier.group(3) if tier else "",
                "tier_evictions": tier.group(4) if tier else "",
                "direct_mib": tier.group(5) if tier else "",
                "compact_mib": tier.group(6) if tier else "",
                "spex_hits": spex.group(1) if spex else "",
                "spex_misses": spex.group(2) if spex else "",
                "spex_direct": spex.group(3) if spex else "",
                "copy_ms_batch": spex.group(4) if spex else "",
                "sync_ms_batch": spex.group(5) if spex else "",
                "prefix": row["prefix"],
            }
        )
    return rows


def write_outputs(rows: list[dict[str, object]], csv_path: Path, md_path: Path) -> None:
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Local DS4 K23 Cache Sweep - 2026-07-09",
        "",
        "Measured on local RTX 3060 with the same DS4 binary. Each row uses one "
        "64-token warmup before the measured request, `PACE_WARMUP=50`, fixed "
        "K23, no breath, no prebreath, no rotation, routing trace off.",
        "",
        "| suite | prompt | cache | wall_s | prompt_s | first50 | avg t/s | last t/s | tier miss/evict | spex miss | copy ms/b | repeat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["suite"]).replace("20260709_local_cache_sweep_k23_", ""),
                    str(row["prompt"]),
                    str(row["cache"]),
                    str(row["wall_s"]),
                    str(row["prompt_s"]),
                    str(row["first50_tps"]),
                    str(row["avg_tps"]),
                    str(row["last_tps"]),
                    f"{row['tier_misses']}/{row['tier_evictions']}",
                    str(row["spex_misses"]),
                    str(row["copy_ms_batch"]),
                    str(row["repeat"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `cache64` is consistently slower and shows large eviction pressure.",
            "- `cache128` is the best local measured point in these warm K23 sweeps.",
            "- `cache258` removes almost all tiering evictions. When it is not the first cold run it is close to `cache128`, but still slightly behind in these measurements.",
            "- First-in-order cold effects are large: the cold `cache258` rows are not comparable to the warm rows without this caveat.",
            "- HTML quality is still fragile at fixed K23: all HTML rows tripped `repeat_flag=1`. This cache sweep is a throughput/path test, not a quality win.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for run_root in args.runs:
        rows.extend(parse_run(run_root))
    write_outputs(rows, args.csv, args.md)
    print(args.csv)
    print(args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
