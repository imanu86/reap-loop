from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def recall(pred: list[int], target: list[int]) -> float:
    return len(set(pred) & set(target)) / max(len(set(target)), 1)


def top(counter: Counter[int], cap: int) -> list[int]:
    return [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]]


def analyze(path: Path) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                (
                    int(r["pos"]),
                    int(r["layer"]),
                    [int(r[f"e{i}"]) for i in range(6)],
                )
            )

    by = {(pos, layer): experts for pos, layer, experts in rows}
    positions = sorted({pos for pos, _, _ in rows})
    layers = sorted({layer for _, layer, _ in rows})
    metrics: dict[str, list[float]] = defaultdict(list)

    for pos in positions:
        for layer in layers:
            target = by.get((pos, layer))
            if target is None:
                continue
            if (pos, layer - 1) in by:
                metrics["same_token_prev_layer_top6"].append(recall(by[(pos, layer - 1)], target))
            if (pos - 1, layer) in by:
                metrics["prev_token_same_layer_top6"].append(recall(by[(pos - 1, layer)], target))
            if (pos - 1, layer - 1) in by:
                metrics["prev_token_prev_layer_top6"].append(recall(by[(pos - 1, layer - 1)], target))

            for w in (2, 4, 8):
                hist = Counter()
                for prev in range(max(positions[0], pos - w), pos):
                    if (prev, layer) in by:
                        hist.update(by[(prev, layer)])
                if hist:
                    metrics[f"window{w}_same_layer_top6"].append(recall(top(hist, 6), target))
                    metrics[f"window{w}_same_layer_top12"].append(recall(top(hist, 12), target))

                combo = Counter(hist)
                if (pos, layer - 1) in by:
                    combo.update(by[(pos, layer - 1)])
                if combo:
                    metrics[f"prev_layer_plus_window{w}_top6"].append(recall(top(combo, 6), target))
                    metrics[f"prev_layer_plus_window{w}_top12"].append(recall(top(combo, 12), target))

    return {
        "path": str(path),
        "rows": len(rows),
        "positions": [min(positions), max(positions), len(positions)] if positions else [0, 0, 0],
        "layers": [min(layers), max(layers), len(layers)] if layers else [0, 0, 0],
        "metrics": {k: sum(v) / len(v) for k, v in sorted(metrics.items()) if v},
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_ds4_routing_trace.py trace.csv")
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
