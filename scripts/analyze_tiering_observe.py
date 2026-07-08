#!/usr/bin/env python3
"""Summarize DS4_EXPERT_TIERING=observe JSONL logs."""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import tarfile
from collections import OrderedDict
from pathlib import Path


def routing_csv_rows(fh, source: str) -> list[dict]:
    rows: list[dict] = []
    reader = csv.DictReader(fh)
    for lineno, row in enumerate(reader, 2):
        try:
            layer = int(row.get("layer") or 0)
            n = int(row.get("n") or 0)
        except ValueError as exc:
            raise SystemExit(f"{source}:{lineno}: bad layer/n: {exc}") from exc
        experts: list[int] = []
        for idx in range(n):
            raw = row.get(f"e{idx}")
            if raw is None:
                continue
            try:
                expert = int(raw)
            except ValueError as exc:
                raise SystemExit(
                    f"{source}:{lineno}: bad expert e{idx}: {exc}"
                ) from exc
            if expert >= 0:
                experts.append(expert)
        if not experts:
            continue
        rows.append(
            {
                "event": "tiering_observe",
                "path": "routing_csv",
                "layer": layer,
                "slots": len(experts),
                "compact": len(experts),
                "selected": experts,
                "compact_ids": experts,
                "source": source,
            }
        )
    return rows


def load_routing_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        return routing_csv_rows(fh, str(path))


def load_routing_tgz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".csv"):
                continue
            raw = tf.extractfile(member)
            if raw is None:
                continue
            with raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                rows.extend(routing_csv_rows(text, f"{path}:{member.name}"))
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        name = path.name.lower()
        if name.endswith(".csv"):
            rows.extend(load_routing_csv(path))
        elif name.endswith(".tgz") or name.endswith(".tar.gz"):
            rows.extend(load_routing_tgz(path))
        else:
            rows.extend(load_jsonl(path))
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


def row_expert_ids(row: dict, source: str) -> list[tuple[int, int]]:
    layer = int(row.get("layer") or 0)
    ids = row.get(source)
    if not isinstance(ids, list):
        return []
    out: list[tuple[int, int]] = []
    for raw in ids:
        try:
            expert = int(raw)
        except (TypeError, ValueError):
            continue
        if expert >= 0:
            out.append((layer, expert))
    return out


def infer_first_layer_cycle_rows(rows: list[dict]) -> int:
    """Return the first contiguous layer cycle, usually one prompt/prefill batch."""

    if not rows:
        return 0
    prev_layer = int(rows[0].get("layer") or 0)
    for idx, row in enumerate(rows[1:], 1):
        layer = int(row.get("layer") or 0)
        if layer <= prev_layer:
            return idx
        prev_layer = layer
    return len(rows)


def resolve_prefill_rows(value: str, rows: list[dict]) -> int:
    if value.lower() == "auto":
        return infer_first_layer_cycle_rows(rows)
    try:
        resolved = int(value)
    except ValueError as exc:
        raise SystemExit(f"bad --tier-prefill-rows value {value!r}") from exc
    if resolved < 0:
        raise SystemExit("--tier-prefill-rows must be >= 0 or 'auto'")
    return min(resolved, len(rows))


def prompt_top_keys(
    rows: list[dict],
    limit: int,
    source: str,
) -> list[tuple[int, int]]:
    if limit <= 0:
        return []
    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    first_seen: dict[tuple[int, int], int] = {}
    for row_idx, row in enumerate(rows):
        for key in row_expert_ids(row, source):
            counts[key] += 1
            first_seen.setdefault(key, row_idx)
    ranked = sorted(
        counts,
        key=lambda key: (-counts[key], first_seen[key], key[0], key[1]),
    )
    return ranked[:limit]


def simulate_tier_policy(
    rows: list[dict],
    cap: int,
    source: str,
    warm_grace: int,
    freeze_after: int,
    initial_hot: list[tuple[int, int]] | None = None,
) -> dict[str, int | float]:
    """Replay a metadata-only hot/warm/cold/frozen tier policy.

    The simulation intentionally does not model compression latency. It answers a
    prior question: if hot capacity is `cap`, how often would a request be served
    from hot, warm grace, cold compressed storage, or frozen/SSD fallback?
    """

    if cap <= 0:
        return {"cap": cap, "requests": 0}

    hot: OrderedDict[tuple[int, int], None] = OrderedDict()
    warm_until: dict[tuple[int, int], int] = {}
    cold: set[tuple[int, int]] = set()
    frozen: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    last_seen: dict[tuple[int, int], int] = {}
    out: collections.defaultdict[str, int] = collections.defaultdict(int)
    peak_hot = peak_warm = peak_cold = peak_frozen = 0

    for key in reversed((initial_hot or [])[:cap]):
        hot[key] = None
        seen.add(key)
        last_seen[key] = -1
    out["preloaded"] = len(hot)
    peak_hot = len(hot)

    def expire_warm(event_index: int) -> None:
        expired = [key for key, until in warm_until.items() if until < event_index]
        for key in expired:
            warm_until.pop(key, None)
            if key not in hot:
                cold.add(key)
                out["warm_demotions"] += 1

    def freeze_idle(event_index: int) -> None:
        if freeze_after <= 0:
            return
        expired = [
            key for key in cold
            if event_index - last_seen.get(key, event_index) > freeze_after
        ]
        for key in expired:
            cold.discard(key)
            frozen.add(key)
            out["cold_freezes"] += 1

    for event_index, row in enumerate(rows):
        expire_warm(event_index)
        freeze_idle(event_index)
        for key in row_expert_ids(row, source):
            out["requests"] += 1
            if key in hot:
                out["hot_hits"] += 1
                hot.move_to_end(key)
            elif key in warm_until:
                out["warm_hits"] += 1
                warm_until.pop(key, None)
                cold.discard(key)
                frozen.discard(key)
                hot[key] = None
            else:
                if key in frozen:
                    out["frozen_recalls"] += 1
                    frozen.discard(key)
                elif key in cold:
                    out["cold_recalls"] += 1
                    cold.discard(key)
                elif key in seen:
                    out["cold_recalls"] += 1
                else:
                    out["initial_loads"] += 1
                hot[key] = None

            seen.add(key)
            last_seen[key] = event_index

            while len(hot) > cap:
                evicted, _ = hot.popitem(last=False)
                out["hot_evictions"] += 1
                if warm_grace > 0:
                    warm_until[evicted] = event_index + warm_grace
                    cold.discard(evicted)
                    frozen.discard(evicted)
                else:
                    cold.add(evicted)
                    frozen.discard(evicted)
                    out["warm_demotions"] += 1

            peak_hot = max(peak_hot, len(hot))
            peak_warm = max(peak_warm, len(warm_until))
            peak_cold = max(peak_cold, len(cold))
            peak_frozen = max(peak_frozen, len(frozen))

    expire_warm(len(rows) + max(0, warm_grace) + 1)
    freeze_idle(len(rows) + max(0, freeze_after) + 1)
    out["cap"] = cap
    out["unique"] = len(seen)
    out["peak_hot"] = peak_hot
    out["peak_warm"] = peak_warm
    out["peak_cold"] = max(peak_cold, len(cold))
    out["peak_frozen"] = max(peak_frozen, len(frozen))
    out["hot_hit_rate"] = (
        out["hot_hits"] / out["requests"] if out["requests"] else 0.0
    )
    out["served_hit_rate"] = (
        (out["hot_hits"] + out["warm_hits"]) / out["requests"]
        if out["requests"] else 0.0
    )
    promotions = out["initial_loads"] + out["cold_recalls"] + out["frozen_recalls"]
    out["promotions"] = promotions
    out["total_promotions"] = promotions + out["preloaded"]
    out["promotion_rate"] = promotions / out["requests"] if out["requests"] else 0.0
    hot_misses = out["requests"] - out["hot_hits"]
    out["warm_rescue_rate"] = out["warm_hits"] / hot_misses if hot_misses else 0.0
    return dict(out)


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


def tier_footprint(
    result: dict[str, int | float],
    slot_mib: float,
    warm_scale: float,
    cold_scale: float,
    frozen_scale: float,
) -> str:
    if slot_mib <= 0:
        return ""
    hot = int(result.get("peak_hot") or 0)
    warm = int(result.get("peak_warm") or 0)
    cold = int(result.get("peak_cold") or 0)
    frozen = int(result.get("peak_frozen") or 0)
    native = (hot + warm + cold + frozen) * slot_mib
    scaled = (
        hot * slot_mib
        + warm * warm_scale * slot_mib
        + cold * cold_scale * slot_mib
        + frozen * frozen_scale * slot_mib
    )
    return f" footprint_native={native:.1f}MiB footprint_scaled={scaled:.1f}MiB"


def summarize(
    rows: list[dict],
    top: int,
    simulate_cap: list[int],
    slot_mib: float,
    capacity_scales: list[float],
    target_hit_rate: list[float],
    tier_sim_cap: list[int],
    tier_sim_source: str,
    tier_warm_grace: int,
    tier_freeze_after: int,
    tier_warm_scale: float,
    tier_cold_scale: float,
    tier_frozen_scale: float,
    tier_prefill_rows: str,
    tier_preload_top: int,
    tier_preload_source: str,
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
            sim_results: list[tuple[int, int, int, int, float]] = []
            for cap in simulate_cap:
                requests, sim_hits, sim_misses = simulate_lru(rows, cap)
                rate = sim_hits / requests if requests else 0.0
                sim_results.append((cap, requests, sim_hits, sim_misses, rate))
                lines.append(
                    f"  cap={cap} requests={requests} hit_rate={rate:.4f} "
                    f"hits={sim_hits} misses={sim_misses}"
                    f"{capacity_cost(cap, slot_mib, capacity_scales)}"
                )
            for target in target_hit_rate:
                if target <= 0:
                    continue
                winner = next(
                    (item for item in sim_results if item[4] >= target),
                    None,
                )
                if winner is None:
                    lines.append(f"  target_hit_rate={target:.4f} first_cap=unmet")
                else:
                    cap, requests, sim_hits, sim_misses, rate = winner
                    lines.append(
                        f"  target_hit_rate={target:.4f} first_cap={cap} "
                        f"hit_rate={rate:.4f} hits={sim_hits} misses={sim_misses}"
                        f"{capacity_cost(cap, slot_mib, capacity_scales)}"
                    )
        if tier_sim_cap:
            prefill_count = resolve_prefill_rows(tier_prefill_rows, rows)
            prefill_rows = rows[:prefill_count]
            replay_rows = rows[prefill_count:] if prefill_count else rows
            lines.append(
                "tier_sim:"
                f" source={tier_sim_source}"
                f" warm_grace={tier_warm_grace}"
                f" freeze_after={tier_freeze_after}"
                f" prefill_rows={prefill_count}"
                f" preload_source={tier_preload_source}"
            )
            for cap in tier_sim_cap:
                preload_limit = min(cap, tier_preload_top or cap)
                initial_hot = prompt_top_keys(
                    prefill_rows,
                    preload_limit,
                    tier_preload_source,
                )
                result = simulate_tier_policy(
                    replay_rows,
                    cap=cap,
                    source=tier_sim_source,
                    warm_grace=tier_warm_grace,
                    freeze_after=tier_freeze_after,
                    initial_hot=initial_hot,
                )
                lines.append(
                    f"  cap={cap} requests={int(result.get('requests') or 0)} "
                    f"hot_hit_rate={float(result.get('hot_hit_rate') or 0.0):.4f} "
                    f"served_hit_rate={float(result.get('served_hit_rate') or 0.0):.4f} "
                    f"hot_hits={int(result.get('hot_hits') or 0)} "
                    f"warm_hits={int(result.get('warm_hits') or 0)} "
                    f"cold_recalls={int(result.get('cold_recalls') or 0)} "
                    f"frozen_recalls={int(result.get('frozen_recalls') or 0)} "
                    f"initial_loads={int(result.get('initial_loads') or 0)} "
                    f"preloaded={int(result.get('preloaded') or 0)} "
                    f"promotions={int(result.get('promotions') or 0)} "
                    f"total_promotions={int(result.get('total_promotions') or 0)} "
                    f"promotion_rate={float(result.get('promotion_rate') or 0.0):.4f} "
                    f"warm_rescue_rate={float(result.get('warm_rescue_rate') or 0.0):.4f} "
                    f"hot_evictions={int(result.get('hot_evictions') or 0)} "
                    f"warm_demotions={int(result.get('warm_demotions') or 0)} "
                    f"unique={int(result.get('unique') or 0)} "
                    f"peak_hot={int(result.get('peak_hot') or 0)} "
                    f"peak_warm={int(result.get('peak_warm') or 0)} "
                    f"peak_cold={int(result.get('peak_cold') or 0)} "
                    f"peak_frozen={int(result.get('peak_frozen') or 0)}"
                    f"{tier_footprint(result, slot_mib, tier_warm_scale, tier_cold_scale, tier_frozen_scale)}"
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
    ap.add_argument(
        "trace",
        nargs="+",
        type=Path,
        help="Tiering observe JSONL, routing CSV, or .tgz containing routing CSVs.",
    )
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
    ap.add_argument(
        "--target-hit-rate",
        type=float,
        nargs="*",
        default=[],
        help="Report the first simulated cap that reaches each target hit rate.",
    )
    ap.add_argument(
        "--tier-sim-cap",
        type=int,
        nargs="*",
        default=[],
        help="Run metadata-only hot/warm/cold/frozen tier simulation for these hot caps.",
    )
    ap.add_argument(
        "--tier-sim-source",
        choices=("compact_ids", "selected"),
        default="compact_ids",
        help="Which ID array to replay for tier simulation.",
    )
    ap.add_argument(
        "--tier-warm-grace",
        type=int,
        default=0,
        help="Rows after hot eviction before warm demotes to cold.",
    )
    ap.add_argument(
        "--tier-freeze-after",
        type=int,
        default=0,
        help="Rows of cold idle time before a recall is counted as frozen.",
    )
    ap.add_argument(
        "--tier-warm-scale",
        type=float,
        default=1.0,
        help="Footprint multiplier for warm tier in tier simulation.",
    )
    ap.add_argument(
        "--tier-cold-scale",
        type=float,
        default=0.5,
        help="Footprint multiplier for cold tier in tier simulation.",
    )
    ap.add_argument(
        "--tier-frozen-scale",
        type=float,
        default=0.0,
        help="Footprint multiplier for frozen tier in tier simulation.",
    )
    ap.add_argument(
        "--tier-prefill-rows",
        default="0",
        help=(
            "Use the first N rows, or 'auto' for the first layer cycle, as a "
            "router signal to preload hot experts before replaying the rest."
        ),
    )
    ap.add_argument(
        "--tier-preload-top",
        type=int,
        default=0,
        help="Number of prompt-ranked experts to preload. 0 means preload up to cap.",
    )
    ap.add_argument(
        "--tier-preload-source",
        choices=("compact_ids", "selected"),
        default="selected",
        help="Which ID array scores prompt-derived preloads.",
    )
    args = ap.parse_args()
    rows = load_rows(args.trace)
    print(
        summarize(
            rows,
            args.top,
            args.simulate_cap,
            args.slot_mib,
            args.capacity_scale,
            args.target_hit_rate,
            args.tier_sim_cap,
            args.tier_sim_source,
            args.tier_warm_grace,
            args.tier_freeze_after,
            args.tier_warm_scale,
            args.tier_cold_scale,
            args.tier_frozen_scale,
            args.tier_prefill_rows,
            args.tier_preload_top,
            args.tier_preload_source,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
