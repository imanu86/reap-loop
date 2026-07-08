#!/usr/bin/env python3
"""Evaluate a DS4 hidden SPX1 predictor against DSH1 hidden traces.

Inputs come from ds4.c runtime diagnostics:
  DS4_SPEX_TRACE_HIDDEN=/path/trace.dsh
  DS4_SPEX_TRACE_ROUTING=/path/routing.csv
  DS4_SPEX_TRACE_ROUTING_WEIGHTS=1   # optional, enables weighted recall

For a hidden row at (pos, layer L), SPX1 weights[L] predict experts for layer
L+1. This script compares top-K predictions against the selected experts logged
by the routing CSV at (pos, L+1).
"""

from __future__ import annotations

import argparse
import csv
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("NumPy is required for hidden SPX1 scoring.") from exc


SPX1_HEADER = struct.Struct("<4sIIIIIII")
DSH1_HEADER = struct.Struct("<4sIII")
DSH1_ROW_PREFIX = struct.Struct("<II")


@dataclass
class RoutingRow:
    experts: tuple[int, ...]
    weights: tuple[float, ...] | None


@dataclass
class ScoreStats:
    rows: int = 0
    target_experts: int = 0
    hits: int = 0
    hit_any: int = 0
    hit_all: int = 0
    weight_total: float = 0.0
    weight_hit: float = 0.0

    def add(self, hit_count: int, target_count: int, weighted_hit: float, weight_total: float) -> None:
        self.rows += 1
        self.target_experts += target_count
        self.hits += hit_count
        self.hit_any += 1 if hit_count else 0
        self.hit_all += 1 if target_count and hit_count == target_count else 0
        self.weight_hit += weighted_hit
        self.weight_total += weight_total

    def line(self, label: str) -> str:
        recall = self.hits / self.target_experts if self.target_experts else 0.0
        any_rate = self.hit_any / self.rows if self.rows else 0.0
        all_rate = self.hit_all / self.rows if self.rows else 0.0
        weighted = self.weight_hit / self.weight_total if self.weight_total else 0.0
        return (
            f"{label}: rows={self.rows} recall={recall:.4f} "
            f"hit_any={any_rate:.4f} hit_all={all_rate:.4f} "
            f"weighted_recall={weighted:.4f}"
        )


def load_spx1(path: Path) -> tuple[np.memmap, int, int, int]:
    with path.open("rb") as fh:
        data = fh.read(SPX1_HEADER.size)
    magic, version, predictor, n_layer, n_embd, n_expert, _ridge_bits, reserved = SPX1_HEADER.unpack(data)
    if magic != b"SPX1" or version != 1 or predictor != 2 or reserved != 0:
        raise SystemExit(f"{path}: unsupported SPX1 header")
    expected = SPX1_HEADER.size + n_layer * n_embd * n_expert * 2
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"{path}: size mismatch, got {actual}, expected {expected}")
    weights = np.memmap(
        path,
        dtype="<f2",
        mode="r",
        offset=SPX1_HEADER.size,
        shape=(n_layer, n_embd, n_expert),
    )
    return weights, n_layer, n_embd, n_expert


def load_routing(path: Path, max_selected: int) -> dict[tuple[int, int], RoutingRow]:
    rows: dict[tuple[int, int], RoutingRow] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for lineno, row in enumerate(reader, 1):
            if not row:
                continue
            try:
                pos = int(row[0])
                layer = int(row[1])
                n_selected = int(row[2])
            except (ValueError, IndexError):
                continue
            expert_cols = row[3 : 3 + max_selected]
            experts = tuple(int(x) for x in expert_cols[:n_selected] if int(x) >= 0)
            weights = None
            weight_cols = row[3 + max_selected : 3 + max_selected * 2]
            if len(weight_cols) >= n_selected:
                parsed: list[float] = []
                for x in weight_cols[:n_selected]:
                    try:
                        parsed.append(float(x))
                    except ValueError:
                        parsed.append(float("nan"))
                if any(np.isfinite(parsed)):
                    weights = tuple(parsed)
            rows[(pos, layer)] = RoutingRow(experts=experts, weights=weights)
    return rows


def iter_hidden(path: Path, n_embd_expected: int):
    with path.open("rb") as fh:
        header = fh.read(DSH1_HEADER.size)
        if len(header) != DSH1_HEADER.size:
            raise SystemExit(f"{path}: truncated DSH1 header")
        magic, version, n_layer, n_embd = DSH1_HEADER.unpack(header)
        if magic != b"DSH1" or version != 1:
            raise SystemExit(f"{path}: unsupported DSH1 header")
        if n_embd != n_embd_expected:
            raise SystemExit(f"{path}: n_embd={n_embd}, SPX1 expects {n_embd_expected}")
        row_bytes = DSH1_ROW_PREFIX.size + n_embd * 2
        idx = 0
        while True:
            blob = fh.read(row_bytes)
            if not blob:
                break
            if len(blob) != row_bytes:
                raise SystemExit(f"{path}: truncated DSH1 row {idx}")
            pos, layer = DSH1_ROW_PREFIX.unpack_from(blob, 0)
            hidden = np.frombuffer(blob, dtype="<f2", count=n_embd, offset=DSH1_ROW_PREFIX.size)
            yield idx, pos, layer, hidden
            idx += 1


def topk(scores: np.ndarray, k: int) -> set[int]:
    k = min(k, scores.shape[0])
    if k <= 0:
        return set()
    part = np.argpartition(scores, -k)[-k:]
    ordered = part[np.argsort(scores[part])[::-1]]
    return {int(x) for x in ordered}


def evaluate(args: argparse.Namespace) -> str:
    weights, n_layer, n_embd, n_expert = load_spx1(args.spex)
    routing = load_routing(args.routing_csv, args.max_selected)
    ks = sorted(set(args.k))
    totals = {k: ScoreStats() for k in ks}
    by_layer: dict[int, dict[int, ScoreStats]] = defaultdict(lambda: {k: ScoreStats() for k in ks})
    w32_cache: dict[int, np.ndarray] = {}
    skipped_no_target = 0
    skipped_last_layer = 0

    for idx, pos, layer, hidden in iter_hidden(args.hidden_trace, n_embd):
        if args.limit and idx >= args.limit:
            break
        if layer + 1 >= n_layer:
            skipped_last_layer += 1
            continue
        target = routing.get((pos, layer + 1))
        if not target or not target.experts:
            skipped_no_target += 1
            continue
        if layer not in w32_cache:
            w32_cache[layer] = np.asarray(weights[layer], dtype=np.float32)
        scores = np.asarray(hidden, dtype=np.float32) @ w32_cache[layer]
        target_set = set(target.experts)
        target_count = len(target_set)
        finite_weights = target.weights and len(target.weights) >= target_count

        for k in ks:
            pred = topk(scores, k)
            hit_count = len(pred & target_set)
            weighted_hit = 0.0
            weight_total = 0.0
            if finite_weights:
                for expert, weight in zip(target.experts, target.weights or ()):
                    if not np.isfinite(weight):
                        continue
                    weight_total += float(weight)
                    if expert in pred:
                        weighted_hit += float(weight)
            totals[k].add(hit_count, target_count, weighted_hit, weight_total)
            by_layer[layer + 1][k].add(hit_count, target_count, weighted_hit, weight_total)

    lines = [
        f"spex={args.spex}",
        f"hidden_trace={args.hidden_trace}",
        f"routing_csv={args.routing_csv}",
        f"shape=L{n_layer} D{n_embd} E{n_expert}",
        f"routing_rows={len(routing)} skipped_no_target={skipped_no_target} skipped_last_layer={skipped_last_layer}",
    ]
    for k in ks:
        lines.append(totals[k].line(f"top{k}"))
    if args.by_layer:
        lines.append("by_layer:")
        for layer in sorted(by_layer):
            best = " ".join(by_layer[layer][k].line(f"k{k}") for k in ks)
            lines.append(f"  layer={layer} {best}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spex", required=True, type=Path, help="Hidden SPX1 predictor file")
    ap.add_argument("--hidden-trace", required=True, type=Path, help="DSH1 hidden trace")
    ap.add_argument("--routing-csv", required=True, type=Path, help="DS4 routing CSV")
    ap.add_argument("--k", type=int, nargs="+", default=[6, 8, 12, 16, 23])
    ap.add_argument("--max-selected", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="Limit hidden rows for quick checks")
    ap.add_argument("--by-layer", action="store_true")
    args = ap.parse_args()
    print(evaluate(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
