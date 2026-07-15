#!/usr/bin/env python3
"""Learn/self DS4 full-decode mass coverage curves from weighted routing CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import tempfile
from collections import defaultdict


VERSION = "1"
MASKABLE_LAYER_MIN = 3
MASKABLE_LAYER_MAX = 42
N_EXPERT_DEFAULT = 256


def parse_keeps(value: str) -> list[int]:
    keeps: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            if not left or not right:
                raise ValueError(f"bad keep range: {part}")
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f"descending keep range: {part}")
            keeps.update(range(start, end + 1))
        else:
            keeps.add(int(part))
    if not keeps:
        raise ValueError("--keeps did not contain any K values")
    bad = [k for k in keeps if k < 0 or k > N_EXPERT_DEFAULT]
    if bad:
        raise ValueError(f"keep values out of [0,{N_EXPERT_DEFAULT}]: {bad}")
    return sorted(keeps)


def numeric_columns(header: list[str], prefix: str) -> list[int]:
    found: list[tuple[int, int]] = []
    for idx, name in enumerate(header):
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            found.append((int(name[len(prefix) :]), idx))
    return [idx for _, idx in sorted(found)]


def header_plan(header: list[str], path: pathlib.Path) -> dict[str, object]:
    lowered = [h.strip().lower() for h in header]
    try:
        layer_idx = lowered.index("layer")
        n_idx = lowered.index("n")
    except ValueError as exc:
        raise ValueError(f"{path}: header must contain layer,n columns") from exc

    expert_cols = numeric_columns(lowered, "e")
    weight_cols = numeric_columns(lowered, "w")
    if not expert_cols:
        raise ValueError(f"{path}: header has no e0/e1/... expert columns")
    if not weight_cols:
        raise ValueError(f"{path}: weighted trace required, header has no w0/w1/... columns")
    if len(weight_cols) < len(expert_cols):
        raise ValueError(
            f"{path}: weighted trace has fewer weight columns ({len(weight_cols)}) "
            f"than expert columns ({len(expert_cols)})"
        )
    return {"layer_idx": layer_idx, "n_idx": n_idx, "expert_cols": expert_cols, "weight_cols": weight_cols}


def _parse_int(value: str, what: str, path: pathlib.Path, line_no: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: invalid {what}: {value!r}") from exc


def _parse_weight(value: str, path: pathlib.Path, line_no: int) -> float:
    try:
        weight = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: invalid weight: {value!r}") from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{path}:{line_no}: weight must be finite and non-negative: {value!r}")
    return weight


def read_traces(paths: list[pathlib.Path], *, layer_min: int, layer_max: int, n_expert: int):
    mass = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(lambda: defaultdict(int))
    rows: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
    input_stats = []

    for path in paths:
        data_rows = 0
        used_rows = 0
        with path.open(newline="", encoding="utf-8-sig", errors="strict") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{path}: missing CSV header")
            plan = header_plan(header, path)
            expert_cols = plan["expert_cols"]
            weight_cols = plan["weight_cols"]
            max_top = min(len(expert_cols), len(weight_cols))

            for line_no, row in enumerate(reader, start=2):
                if not row or all(cell == "" for cell in row):
                    continue
                data_rows += 1
                min_cols = max(expert_cols[:1] + weight_cols[:1] + [plan["layer_idx"], plan["n_idx"]]) + 1
                if len(row) < min_cols:
                    raise ValueError(f"{path}:{line_no}: row has too few columns")
                layer = _parse_int(row[plan["layer_idx"]], "layer", path, line_no)
                n = _parse_int(row[plan["n_idx"]], "n", path, line_no)
                if n < 0:
                    raise ValueError(f"{path}:{line_no}: n must be non-negative")
                if layer < layer_min or layer > layer_max:
                    continue

                take = min(n, max_top)
                experts: list[int] = []
                weights: list[float] = []
                for slot in range(take):
                    e_idx = expert_cols[slot]
                    w_idx = weight_cols[slot]
                    if e_idx >= len(row) or w_idx >= len(row):
                        raise ValueError(f"{path}:{line_no}: row missing expert/weight slot {slot}")
                    expert = _parse_int(row[e_idx], f"expert e{slot}", path, line_no)
                    weight = _parse_weight(row[w_idx], path, line_no)
                    if expert < 0:
                        continue
                    if expert >= n_expert:
                        raise ValueError(f"{path}:{line_no}: expert id {expert} outside [0,{n_expert})")
                    experts.append(expert)
                    weights.append(weight)
                    mass[layer][expert] += weight
                    calls[layer][expert] += 1
                rows.append((layer, tuple(experts), tuple(weights)))
                used_rows += 1

        if data_rows == 0:
            raise ValueError(f"{path}: no data rows")
        input_stats.append({"path": str(path), "resolved_path": str(path.resolve()), "data_rows": data_rows, "used_rows": used_rows})

    if not rows:
        raise ValueError(f"no rows in maskable layers {layer_min}..{layer_max}")
    total_mass = sum(sum(by_expert.values()) for by_expert in mass.values())
    if total_mass <= 0.0:
        raise ValueError("weighted trace has no positive selected mass in maskable layers")
    return {"mass": mass, "calls": calls, "rows": rows, "inputs": input_stats}


def build_rankings(mass: dict[int, dict[int, float]]) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    for layer in sorted(mass):
        rankings[layer] = sorted(mass[layer], key=lambda expert: (-mass[layer][expert], expert))
    return rankings


def evaluate(data: dict[str, object], keeps: list[int], *, n_expert: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    mass = data["mass"]
    calls = data["calls"]
    rows = data["rows"]
    rankings = build_rankings(mass)
    layers = sorted(rankings)

    total_calls = sum(sum(by_expert.values()) for by_expert in calls.values())
    total_mass = sum(sum(by_expert.values()) for by_expert in mass.values())
    by_layer_mass = {layer: sum(mass[layer].values()) for layer in layers}
    results: list[dict[str, object]] = []

    for keep in keeps:
        keep_sets = {layer: set(rankings[layer][: min(keep, n_expert)]) for layer in layers}
        covered_calls = 0
        covered_mass = 0.0
        layer_mass_coverage: dict[int, float] = {}
        for layer in layers:
            kept = keep_sets[layer]
            layer_hit_mass = sum(weight for expert, weight in mass[layer].items() if expert in kept)
            layer_total_mass = by_layer_mass[layer]
            layer_mass_coverage[layer] = layer_hit_mass / layer_total_mass if layer_total_mass else 0.0
            covered_mass += layer_hit_mass
            covered_calls += sum(count for expert, count in calls[layer].items() if expert in kept)

        all_selected = 0
        for layer, experts, _weights in rows:
            kept = keep_sets.get(layer, set())
            if all(expert in kept for expert in experts):
                all_selected += 1

        result = {
            "keep": keep,
            "evaluation_label": "learn_self_coverage",
            "layers": len(layers),
            "rows": len(rows),
            "selected_calls": total_calls,
            "selected_mass": total_mass,
            "call_coverage": covered_calls / total_calls if total_calls else 0.0,
            "mass_coverage": covered_mass / total_mass if total_mass else 0.0,
            "all_selected_row_rate": all_selected / len(rows) if rows else 0.0,
            "worst_layer_mass_coverage": min(layer_mass_coverage.values()) if layer_mass_coverage else 0.0,
        }
        results.append(result)

    ranking_json = {
        str(layer): [
            {"expert": expert, "mass": mass[layer][expert], "calls": calls[layer][expert]}
            for expert in rankings[layer]
        ]
        for layer in layers
    }
    return results, {"layers": layers, "ranking": ranking_json}


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "keep",
        "evaluation_label",
        "layers",
        "rows",
        "selected_calls",
        "selected_mass",
        "call_coverage",
        "mass_coverage",
        "all_selected_row_rate",
        "worst_layer_mass_coverage",
    ]
    lines = []
    sink = _CsvStringWriter(lines)
    writer = csv.DictWriter(sink, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    write_text_atomic(path, "".join(lines))


class _CsvStringWriter:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def write(self, text: str) -> int:
        self.lines.append(text)
        return len(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn per-layer top-K expert rankings by sum(weight) from weighted DS4 "
            "routing CSVs, then report learn/self coverage curves."
        )
    )
    parser.add_argument("--trace", action="append", required=True, type=pathlib.Path, help="Input weighted routing CSV; repeatable.")
    parser.add_argument("--keeps", required=True, help="Comma list/ranges of K values, e.g. 8,16,23-25,64.")
    parser.add_argument("--json-out", type=pathlib.Path, help="Output JSON path.")
    parser.add_argument("--csv-out", type=pathlib.Path, help="Output CSV path.")
    parser.add_argument("--n-expert", type=int, default=N_EXPERT_DEFAULT, help="Experts per layer; default 256.")
    parser.add_argument("--layer-min", type=int, default=MASKABLE_LAYER_MIN, help="First maskable layer; default 3.")
    parser.add_argument("--layer-max", type=int, default=MASKABLE_LAYER_MAX, help="Last maskable layer; default 42.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.n_expert != N_EXPERT_DEFAULT:
            raise ValueError("DS4 mass curve is fixed to 256 experts; --n-expert must remain 256")
        if args.layer_min > args.layer_max:
            raise ValueError("--layer-min cannot exceed --layer-max")
        keeps = parse_keeps(args.keeps)
        data = read_traces(args.trace, layer_min=args.layer_min, layer_max=args.layer_max, n_expert=args.n_expert)
        results, learned = evaluate(data, keeps, n_expert=args.n_expert)
        payload = {
            "tool": "full_decode_mass_curve.py",
            "version": VERSION,
            "evaluation_label": "learn_self_coverage",
            "note": "Ranking is learned on the same trace(s) being scored; this is not held-out coverage.",
            "policy": {
                "rank_by": "sum(weight)",
                "tie_break": "expert_id_ascending",
                "maskable_layers": [args.layer_min, args.layer_max],
                "n_expert": args.n_expert,
            },
            "inputs": data["inputs"],
            "keeps": keeps,
            "results": results,
            "learned": learned,
        }
        if args.json_out:
            write_json(args.json_out, payload)
        if args.csv_out:
            write_csv(args.csv_out, results)
        if not args.json_out and not args.csv_out:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
