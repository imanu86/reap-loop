#!/usr/bin/env python3
"""Build a fixed-width REAP mask ranked by mean selected gate weight.

The input is the learned JSON emitted by ``full_decode_mass_curve.py``.  That
file records cumulative gate mass and call count for each expert.  This tool
deliberately removes frequency from the ranking score:

    mean_selected_weight = sum(selected_weight) / selected_calls

Experts never selected in the learn split receive score zero.  Ties are broken
by ascending expert id.  The output mask lists blocked ``layer expert`` pairs,
as expected by ``DS4_REAP_MASK_FILE``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile


N_EXPERT = 256
LAYER_MIN = 3
LAYER_MAX = 42
VERSION = "1"


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
    ) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = pathlib.Path(stream.name)
    os.replace(temporary, path)


def load_learned(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    ranking = payload.get("learned", {}).get("ranking")
    if not isinstance(ranking, dict):
        raise ValueError(f"{path}: missing learned.ranking object")
    return payload


def build(payload: dict, keep: int) -> tuple[list[str], dict]:
    if keep < 1 or keep > N_EXPERT:
        raise ValueError(f"keep must be in [1,{N_EXPERT}]")

    learned = payload["learned"]["ranking"]
    mask_lines: list[str] = []
    manifest_layers: list[dict] = []

    for layer in range(LAYER_MIN, LAYER_MAX + 1):
        entries = learned.get(str(layer))
        if not isinstance(entries, list):
            raise ValueError(f"missing learned ranking for layer {layer}")

        observed: dict[int, tuple[float, int]] = {}
        for entry in entries:
            expert = int(entry["expert"])
            mass = float(entry["mass"])
            calls = int(entry["calls"])
            if expert < 0 or expert >= N_EXPERT:
                raise ValueError(f"layer {layer}: expert {expert} outside [0,{N_EXPERT})")
            if mass < 0.0 or calls < 0:
                raise ValueError(f"layer {layer} expert {expert}: negative mass/calls")
            if expert in observed:
                raise ValueError(f"layer {layer}: duplicate expert {expert}")
            observed[expert] = (mass, calls)

        scored = []
        for expert in range(N_EXPERT):
            mass, calls = observed.get(expert, (0.0, 0))
            score = mass / calls if calls else 0.0
            scored.append(
                {
                    "expert": expert,
                    "mean_selected_weight": score,
                    "mass": mass,
                    "calls": calls,
                }
            )
        scored.sort(key=lambda row: (-row["mean_selected_weight"], row["expert"]))
        retained = {row["expert"] for row in scored[:keep]}
        mask_lines.extend(f"{layer} {expert}" for expert in range(N_EXPERT) if expert not in retained)
        manifest_layers.append(
            {
                "layer": layer,
                "retained": sorted(retained),
                "ranking": scored,
            }
        )

    expected_blocked = (N_EXPERT - keep) * (LAYER_MAX - LAYER_MIN + 1)
    if len(mask_lines) != expected_blocked:
        raise AssertionError(f"blocked line count {len(mask_lines)} != {expected_blocked}")

    manifest = {
        "tool": "build_mean_weight_mask.py",
        "version": VERSION,
        "policy": {
            "rank_by": "sum(selected_weight) / selected_calls",
            "frequency_component": "excluded",
            "tie_break": "expert_id_ascending",
            "n_expert": N_EXPERT,
            "maskable_layers": [LAYER_MIN, LAYER_MAX],
            "keep": keep,
        },
        "blocked_lines": len(mask_lines),
        "layers": manifest_layers,
    }
    return mask_lines, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learn-json", required=True, type=pathlib.Path)
    parser.add_argument("--keep", required=True, type=int)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--manifest-out", required=True, type=pathlib.Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = load_learned(args.learn_json)
        mask_lines, manifest = build(payload, args.keep)
        manifest["input"] = str(args.learn_json)
        write_text_atomic(args.out, "\n".join(mask_lines) + "\n")
        write_text_atomic(args.manifest_out, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"[weight-mask] out={args.out}")
        print(f"[weight-mask] keep={args.keep} blocked_lines={len(mask_lines)}")
        return 0
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
