#!/usr/bin/env python3
"""Export top-K predictions from a pre-existing SPX1 hidden predictor.

This performs no training. It reads a DSH1 hidden trace, scores each hidden row
with the SPX1 weights for that source layer, and writes a prediction CSV for the
target routing layer. The output can feed analyze_spex_predictive_mask.py.
"""
from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required for SPX1 scoring.") from exc

SPX1_HEADER = struct.Struct("<4sIIIIIII")
DSH1_HEADER = struct.Struct("<4sIII")
DSH1_ROW_PREFIX = struct.Struct("<II")


def load_spx1(path: Path) -> tuple[np.memmap, int, int, int]:
    with path.open("rb") as fh:
        data = fh.read(SPX1_HEADER.size)
    magic, version, predictor, n_layer, n_embd, n_expert, _ridge_bits, reserved = SPX1_HEADER.unpack(data)
    if magic != b"SPX1" or version != 1 or predictor != 2 or reserved != 0:
        raise SystemExit(f"{path}: unsupported SPX1 header")
    expected = SPX1_HEADER.size + n_layer * n_embd * n_expert * 2
    if path.stat().st_size != expected:
        raise SystemExit(f"{path}: size mismatch")
    weights = np.memmap(path, dtype="<f2", mode="r", offset=SPX1_HEADER.size, shape=(n_layer, n_embd, n_expert))
    return weights, n_layer, n_embd, n_expert


def iter_hidden(path: Path, n_embd_expected: int, prefix_order: str, pos_offset: int):
    with path.open("rb") as fh:
        header = fh.read(DSH1_HEADER.size)
        if len(header) != DSH1_HEADER.size:
            raise SystemExit(f"{path}: truncated DSH1 header")
        magic, version, _n_layer, n_embd = DSH1_HEADER.unpack(header)
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
            a, b = DSH1_ROW_PREFIX.unpack_from(blob, 0)
            if prefix_order == "layer_pos":
                layer, pos = a, b
            else:
                pos, layer = a, b
            pos += pos_offset
            hidden = np.frombuffer(blob, dtype="<f2", count=n_embd, offset=DSH1_ROW_PREFIX.size)
            yield idx, pos, layer, hidden
            idx += 1


def topk_ordered(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    part = np.argpartition(scores, -k)[-k:]
    ordered = part[np.argsort(scores[part])[::-1]]
    return ordered.astype(np.int64), scores[ordered].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spex", required=True, type=Path)
    ap.add_argument("--hidden-trace", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--topk", type=int, default=23)
    ap.add_argument("--hidden-prefix-order", choices=["pos_layer", "layer_pos"], default="pos_layer")
    ap.add_argument("--hidden-pos-offset", type=int, default=0)
    ap.add_argument("--target-layer-delta", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    weights, n_layer, n_embd, _n_expert = load_spx1(args.spex)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = ["pos", "layer", "n"] + [f"p{i}" for i in range(args.topk)] + [f"s{i}" for i in range(args.topk)]
    w32_cache: dict[int, np.ndarray] = {}
    rows = skipped = 0
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for idx, pos, layer, hidden in iter_hidden(args.hidden_trace, n_embd, args.hidden_prefix_order, args.hidden_pos_offset):
            if args.limit and idx >= args.limit:
                break
            target_layer = layer + args.target_layer_delta
            if target_layer < 0 or target_layer >= n_layer:
                skipped += 1
                continue
            if layer not in w32_cache:
                w32_cache[layer] = np.asarray(weights[layer], dtype=np.float32)
            scores = np.asarray(hidden, dtype=np.float32) @ w32_cache[layer]
            ids, vals = topk_ordered(scores, args.topk)
            writer.writerow([pos, target_layer, len(ids), *ids.tolist(), *[f"{float(x):.9g}" for x in vals]])
            rows += 1
    print(f"out={args.out}")
    print(f"rows={rows} skipped={skipped} topk={args.topk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
