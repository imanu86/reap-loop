#!/usr/bin/env python3
"""Measure an exact nested base/residual split for DS4 routed experts.

The base keeps a subset of the original IQ2_XXS/Q2_K metadata and bitplanes.
Appending the residual reconstructs the original bytes exactly.  The base can
also be evaluated alone as a cheap one-token fallback while the residual is
being fetched.  This is a CPU-only analysis tool; it does not modify a model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import mmap
from pathlib import Path
import random
import statistics
import struct
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
COLD_LAB_PATH = HERE / "ds4_cold_codec_lab.py"
SPEC = importlib.util.spec_from_file_location("ds4_cold_codec_lab", COLD_LAB_PATH)
assert SPEC and SPEC.loader
cold_lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cold_lab
SPEC.loader.exec_module(cold_lab)


IQ2_XXS_BASE_BYTES = 34
IQ2_XXS_RESIDUAL_BYTES = 32
Q2_K_BASE_BYTES = 52
Q2_K_RESIDUAL_BYTES = 32
ACTIVE_ROUTED_EXPERTS = 40 * 256


def split_iq2_xxs(block: bytes) -> tuple[bytes, bytes]:
    if len(block) != 66:
        raise ValueError("IQ2_XXS block must be 66 bytes")
    base = bytearray(block[:2])
    residual = bytearray()
    for group in range(8):
        offset = 2 + group * 8
        residual.extend(block[offset : offset + 4])
        base.extend(block[offset + 4 : offset + 8])
    return bytes(base), bytes(residual)


def join_iq2_xxs(base: bytes, residual: bytes) -> bytes:
    if len(base) != IQ2_XXS_BASE_BYTES or len(residual) != IQ2_XXS_RESIDUAL_BYTES:
        raise ValueError("invalid nested IQ2_XXS payload sizes")
    out = bytearray(base[:2])
    for group in range(8):
        out.extend(residual[group * 4 : group * 4 + 4])
        out.extend(base[2 + group * 4 : 2 + group * 4 + 4])
    return bytes(out)


def decode_iq2_xxs_base(base: bytes, signs: list[int], grid: list[int]) -> list[float]:
    if len(base) != IQ2_XXS_BASE_BYTES:
        raise ValueError("nested IQ2_XXS base must be 34 bytes")
    d = cold_lab.f16_to_f32(struct.unpack_from("<H", base, 0)[0])
    mean_magnitude = [
        statistics.mean((entry >> (8 * position)) & 0xFF for entry in grid)
        for position in range(8)
    ]
    out: list[float] = []
    for group in range(8):
        aux1 = struct.unpack_from("<I", base, 2 + group * 4)[0]
        group_scale = 2 * (aux1 >> 28) + 1
        for lane in range(4):
            sign_index = (aux1 >> (7 * lane)) & 127
            sign_bits = cold_lab.unpack_iq2_signs(signs[sign_index])
            out.extend(
                0.125 * d * group_scale * magnitude * (-1.0 if (sign_bits >> i) & 1 else 1.0)
                for i, magnitude in enumerate(mean_magnitude)
            )
    return out


def _q2_location(group: int, lane: int) -> tuple[int, int]:
    half = group // 8
    within = group % 8
    qbase = half * 32 + (16 if within % 2 else 0)
    shift = (within // 2) * 2
    return qbase + lane, shift


def split_q2_k(block: bytes) -> tuple[bytes, bytes]:
    if len(block) != 84:
        raise ValueError("Q2_K block must be 84 bytes")
    scales = block[:16]
    qs = block[16:80]
    base_bits = bytearray(32)
    residual_bits = bytearray(32)
    linear = 0
    for group in range(16):
        for lane in range(16):
            qbyte, shift = _q2_location(group, lane)
            q = (qs[qbyte] >> shift) & 3
            base_bits[linear // 8] |= ((q >> 1) & 1) << (linear % 8)
            residual_bits[linear // 8] |= (q & 1) << (linear % 8)
            linear += 1
    return scales + block[80:84] + bytes(base_bits), bytes(residual_bits)


def join_q2_k(base: bytes, residual: bytes) -> bytes:
    if len(base) != Q2_K_BASE_BYTES or len(residual) != Q2_K_RESIDUAL_BYTES:
        raise ValueError("invalid nested Q2_K payload sizes")
    qs = bytearray(64)
    base_bits = base[20:52]
    for linear in range(256):
        group, lane = divmod(linear, 16)
        msb = (base_bits[linear // 8] >> (linear % 8)) & 1
        lsb = (residual[linear // 8] >> (linear % 8)) & 1
        qbyte, shift = _q2_location(group, lane)
        qs[qbyte] |= ((msb << 1) | lsb) << shift
    return base[:16] + bytes(qs) + base[16:20]


def decode_q2_k_base(base: bytes) -> list[float]:
    if len(base) != Q2_K_BASE_BYTES:
        raise ValueError("nested Q2_K base must be 52 bytes")
    scales = base[:16]
    d = cold_lab.f16_to_f32(struct.unpack_from("<H", base, 16)[0])
    dmin = cold_lab.f16_to_f32(struct.unpack_from("<H", base, 18)[0])
    bits = base[20:52]
    out: list[float] = []
    for linear in range(256):
        group = linear // 16
        msb = (bits[linear // 8] >> (linear % 8)) & 1
        q_midpoint = 0.5 + 2.0 * msb
        scale = scales[group] & 0x0F
        min_scale = scales[group] >> 4
        out.append(d * scale * q_midpoint - dmin * min_scale)
    return out


def rms(values: Iterable[float]) -> float:
    vals = list(values)
    return math.sqrt(sum(value * value for value in vals) / max(1, len(vals)))


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def random_q8(rng: random.Random) -> list[int]:
    return [max(-127, min(127, int(round(rng.gauss(0.0, 24.0))))) for _ in range(256)]


def sample_model(args: argparse.Namespace) -> dict[str, object]:
    table_path = Path(args.ds4_root) / "ds4_iq2_tables_cuda.inc"
    signs = cold_lab.parse_cuda_table(table_path, "cuda_ksigns_iq2xs")
    grid = cold_lab.parse_cuda_table(table_path, "cuda_iq2xxs_grid")
    gguf = cold_lab.gguf_inspect_ds4.parse_gguf(args.model)
    layers = cold_lab.gguf_inspect_ds4.discover_expert_layers(gguf)
    rng = random.Random(args.seed)
    rows: dict[str, list[float]] = {
        "iq2_xxs_weight_nrmse": [],
        "iq2_xxs_dot_nmae": [],
        "q2_k_weight_nrmse": [],
        "q2_k_dot_nmae": [],
    }
    exact_reconstructions = 0

    with open(args.model, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for _ in range(args.blocks):
                layer = rng.choice(sorted(layers))
                kind = rng.choice(["gate", "up", "down"])
                tensor = layers[layer][kind]
                expert = rng.randrange(tensor.dims[2])
                row = rng.randrange(tensor.dims[1])
                block_index = rng.randrange(tensor.dims[0] // 256)
                row_bytes = cold_lab.gguf_inspect_ds4.routed_expert_row_bytes(tensor)
                expert_bytes = cold_lab.gguf_inspect_ds4.routed_expert_nbytes(tensor)
                block_bytes = 66 if tensor.type_id == 16 else 84
                offset = tensor.abs_offset + expert * expert_bytes + row * row_bytes + block_index * block_bytes
                block = bytes(mm[offset : offset + block_bytes])
                if tensor.type_id == 16:
                    base, residual = split_iq2_xxs(block)
                    if join_iq2_xxs(base, residual) != block:
                        raise RuntimeError("IQ2_XXS exact reconstruction failed")
                    exact = cold_lab.decode_iq2_xxs_block(block, grid, signs)
                    approx = decode_iq2_xxs_base(base, signs, grid)
                    prefix = "iq2_xxs"
                elif tensor.type_id == 10:
                    base, residual = split_q2_k(block)
                    if join_q2_k(base, residual) != block:
                        raise RuntimeError("Q2_K exact reconstruction failed")
                    exact = cold_lab.decode_q2_k_block(block)
                    approx = decode_q2_k_base(base)
                    prefix = "q2_k"
                else:
                    continue
                exact_reconstructions += 1
                exact_norm = rms(exact)
                rows[f"{prefix}_weight_nrmse"].append(rms(a - b for a, b in zip(approx, exact)) / max(1e-12, exact_norm))
                for _ in range(args.dot_vectors):
                    q8 = random_q8(rng)
                    denom = max(1e-12, rms(exact) * rms(q8) * 256)
                    exact_dot = sum(a * b for a, b in zip(exact, q8))
                    approx_dot = sum(a * b for a, b in zip(approx, q8))
                    rows[f"{prefix}_dot_nmae"].append(abs(approx_dot - exact_dot) / denom)
        finally:
            mm.close()

    active_base_gib = ACTIVE_ROUTED_EXPERTS * 3.75 / 1024
    active_residual_gib = ACTIVE_ROUTED_EXPERTS * 3.0 / 1024
    return {
        "schema": "ds4_nested_residual_lab_v1",
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "sampled_blocks": exact_reconstructions,
        "exact_reconstruction_failures": 0,
        "formats": {
            "IQ2_XXS": {"native_bytes": 66, "base_bytes": 34, "residual_bytes": 32},
            "Q2_K": {"native_bytes": 84, "base_bytes": 52, "residual_bytes": 32},
        },
        "expert_mib": {"native": 6.75, "nested_base": 3.75, "exact_residual": 3.0},
        "active_routed_gib": {
            "native": active_base_gib + active_residual_gib,
            "nested_base": active_base_gib,
            "exact_residual": active_residual_gib,
        },
        "quality_proxy": {key: summarize(values) for key, values in rows.items() if values},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--ds4-root", required=True)
    parser.add_argument("--blocks", type=int, default=768)
    parser.add_argument("--dot-vectors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = sample_model(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
