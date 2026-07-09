#!/usr/bin/env python3
"""Probe lossy cold-tier codecs on real DS4 routed expert blocks.

The goal is to reject bad cold formats cheaply.  This script samples native
IQ2_XXS/Q2_K blocks from a DS4 GGUF, decodes them to approximate float values,
then evaluates simple sign+scale CQ1 variants against random Q8-like vectors.
It does not change the model and does not require CUDA.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import math
import mmap
from pathlib import Path
import random
import re
import statistics
import struct
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
INSPECT_PATH = HERE / "gguf_inspect_ds4.py"
SPEC = importlib.util.spec_from_file_location("gguf_inspect_ds4", INSPECT_PATH)
assert SPEC and SPEC.loader
gguf_inspect_ds4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gguf_inspect_ds4
SPEC.loader.exec_module(gguf_inspect_ds4)


def f16_to_f32(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def parse_cuda_table(path: Path, symbol: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(symbol)}\[[^\]]+\]\s*=\s*\{{(.*?)\}};", text, re.S)
    if not match:
        raise ValueError(f"cannot find {symbol} in {path}")
    body = re.sub(r"//.*", "", match.group(1))
    return [int(tok.strip(), 0) for tok in body.replace("\n", " ").split(",") if tok.strip()]


def unpack_iq2_signs(v: int) -> int:
    parity = v.bit_count() & 1
    return v ^ (parity << 7)


def signed_iq2_grid_values(grid_value: int, sign_value: int) -> list[int]:
    sign_bits = unpack_iq2_signs(sign_value)
    out = []
    for i in range(8):
        mag = (grid_value >> (8 * i)) & 0xFF
        out.append(-mag if ((sign_bits >> i) & 1) else mag)
    return out


def decode_iq2_xxs_block(block: bytes, grid: list[int], signs: list[int]) -> list[float]:
    if len(block) != 66:
        raise ValueError("IQ2_XXS block must be 66 bytes")
    d = f16_to_f32(struct.unpack_from("<H", block, 0)[0])
    q2 = struct.unpack_from("<32H", block, 2)
    out: list[float] = []
    qpos = 0
    for _ in range(8):
        aux0 = q2[qpos] | (q2[qpos + 1] << 16)
        aux1 = q2[qpos + 2] | (q2[qpos + 3] << 16)
        qpos += 4
        group_scale = 2 * (aux1 >> 28) + 1
        grid_indices = [
            aux0 & 0xFF,
            (aux0 >> 8) & 0xFF,
            (aux0 >> 16) & 0xFF,
            (aux0 >> 24) & 0xFF,
        ]
        sign_indices = [
            (aux1 >> 0) & 127,
            (aux1 >> 7) & 127,
            (aux1 >> 14) & 127,
            (aux1 >> 21) & 127,
        ]
        for grid_idx, sign_idx in zip(grid_indices, sign_indices):
            vals = signed_iq2_grid_values(grid[grid_idx], signs[sign_idx])
            out.extend(0.125 * d * group_scale * v for v in vals)
    return out


def decode_q2_k_block(block: bytes) -> list[float]:
    if len(block) != 84:
        raise ValueError("Q2_K block must be 84 bytes")
    scales = block[:16]
    qs = block[16:80]
    d = f16_to_f32(struct.unpack_from("<H", block, 80)[0])
    dmin = f16_to_f32(struct.unpack_from("<H", block, 82)[0])
    out = [0.0] * 256
    for group in range(16):
        half = group // 8
        within = group % 8
        qbase = half * 32 + (16 if within % 2 else 0)
        shift = (within // 2) * 2
        scale = scales[group] & 0x0F
        min_scale = scales[group] >> 4
        base = group * 16
        for i in range(16):
            q = (qs[qbase + i] >> shift) & 0x03
            out[base + i] = d * scale * q - dmin * min_scale
    return out


def cq1(values: list[float], group_size: int) -> list[float]:
    out = [0.0] * len(values)
    for base in range(0, len(values), group_size):
        chunk = values[base : base + group_size]
        scale = sum(abs(v) for v in chunk) / max(1, len(chunk))
        for i, value in enumerate(chunk):
            out[base + i] = scale if value >= 0 else -scale
    return out


def rms(values: Iterable[float]) -> float:
    vals = list(values)
    return math.sqrt(sum(v * v for v in vals) / max(1, len(vals)))


def dot(a: list[float], b: list[int]) -> float:
    return sum(x * y for x, y in zip(a, b))


def random_q8(rng: random.Random) -> list[int]:
    out = []
    for _ in range(256):
        value = int(round(rng.gauss(0.0, 24.0)))
        out.append(max(-127, min(127, value)))
    return out


def codec_payload_bytes(group_size: int) -> int:
    # 2-byte fp16 scale per group + one sign bit per value.
    return (256 // group_size) * 2 + 32


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    vals = sorted(values)
    return {
        "avg": statistics.mean(vals),
        "p50": vals[len(vals) // 2],
        "p95": vals[min(len(vals) - 1, int(0.95 * (len(vals) - 1)))],
        "max": vals[-1],
    }


def sample_blocks(args: argparse.Namespace) -> list[dict[str, object]]:
    table_path = Path(args.ds4_root) / "ds4_iq2_tables_cuda.inc"
    iq2_signs = parse_cuda_table(table_path, "cuda_ksigns_iq2xs")
    iq2_grid = parse_cuda_table(table_path, "cuda_iq2xxs_grid")

    gguf = gguf_inspect_ds4.parse_gguf(args.model)
    layers = gguf_inspect_ds4.discover_expert_layers(gguf)
    if not layers:
        raise SystemExit("no routed expert layers found")

    rng = random.Random(args.seed)
    layer_ids = sorted(layers)
    rows = []
    codecs = {name: int(name.removeprefix("cq1g")) for name in args.codecs.split(",") if name.strip()}

    with open(args.model, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for _ in range(args.blocks):
                layer = rng.choice(layer_ids)
                kind = rng.choice(["gate", "up", "down"])
                tensor = layers[layer][kind]
                expert = rng.randrange(tensor.dims[2])
                row = rng.randrange(tensor.dims[1])
                block_idx = rng.randrange(tensor.dims[0] // 256)

                row_bytes = gguf_inspect_ds4.routed_expert_row_bytes(tensor)
                expert_bytes = gguf_inspect_ds4.routed_expert_nbytes(tensor)
                block_bytes = 66 if tensor.type_id == 16 else 84 if tensor.type_id == 10 else 0
                if block_bytes == 0:
                    continue
                offset = tensor.abs_offset + expert * expert_bytes + row * row_bytes + block_idx * block_bytes
                block = mm[offset : offset + block_bytes]
                exact = decode_iq2_xxs_block(block, iq2_grid, iq2_signs) if tensor.type_id == 16 else decode_q2_k_block(block)
                exact_rms = rms(exact)
                q8_vectors = [random_q8(rng) for _ in range(args.dot_vectors)]
                dot_norms = [math.sqrt(sum(v * v for v in exact)) * math.sqrt(sum(q * q for q in q8)) for q8 in q8_vectors]
                exact_dots = [dot(exact, q8) for q8 in q8_vectors]

                for codec_name, group_size in codecs.items():
                    approx = cq1(exact, group_size)
                    err_rms = rms(a - b for a, b in zip(approx, exact))
                    approx_dots = [dot(approx, q8) for q8 in q8_vectors]
                    dot_nmae = [
                        abs(a - e) / max(1e-9, norm)
                        for a, e, norm in zip(approx_dots, exact_dots, dot_norms)
                    ]
                    rows.append(
                        {
                            "tensor_type": tensor.type_name,
                            "kind": kind,
                            "codec": codec_name,
                            "native_block_bytes": block_bytes,
                            "cold_block_bytes": codec_payload_bytes(group_size),
                            "weight_nrmse": err_rms / max(1e-9, exact_rms),
                            "dot_nmae": statistics.mean(dot_nmae),
                            "layer": layer,
                            "expert": expert,
                        }
                    )
        finally:
            mm.close()
    return rows


NATIVE_GATE_MIB = 2.0625
NATIVE_UP_MIB = 2.0625
NATIVE_DOWN_MIB = 2.625
DS4_ROUTED_EXPERTS = 43 * 256


def cold_kind_mib(group_size: int, kind: str) -> float:
    cold_block = codec_payload_bytes(group_size)
    if kind in {"gate", "up"}:
        total = (4096 // 256) * cold_block * 2048
    elif kind == "down":
        total = (2048 // 256) * cold_block * 4096
    else:
        raise ValueError(kind)
    return total / (1024 * 1024)


def policy_expert_mib(group_size: int, policy: str) -> float:
    if policy == "all":
        return cold_kind_mib(group_size, "gate") + cold_kind_mib(group_size, "up") + cold_kind_mib(group_size, "down")
    if policy == "down-only":
        return NATIVE_GATE_MIB + NATIVE_UP_MIB + cold_kind_mib(group_size, "down")
    if policy == "gate-up-only":
        return cold_kind_mib(group_size, "gate") + cold_kind_mib(group_size, "up") + NATIVE_DOWN_MIB
    raise ValueError(policy)


def print_report(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_kind: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sizes: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        key = (str(row["tensor_type"]), str(row["codec"]))
        grouped[key]["weight_nrmse"].append(float(row["weight_nrmse"]))
        grouped[key]["dot_nmae"].append(float(row["dot_nmae"]))
        sizes[key] = (int(row["native_block_bytes"]), int(row["cold_block_bytes"]))
        kind_key = (str(row["kind"]), str(row["tensor_type"]), str(row["codec"]))
        grouped_kind[kind_key]["weight_nrmse"].append(float(row["weight_nrmse"]))
        grouped_kind[kind_key]["dot_nmae"].append(float(row["dot_nmae"]))

    print("| type | codec | block bytes | size ratio | weight nRMSE avg/p95 | dot nMAE avg/p95 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key in sorted(grouped):
        native, cold = sizes[key]
        weight = summarize(grouped[key]["weight_nrmse"])
        dot_err = summarize(grouped[key]["dot_nmae"])
        print(
            f"| {key[0]} | {key[1]} | {cold}/{native} | {cold / native:.3f} | "
            f"{weight['avg']:.3f}/{weight['p95']:.3f} | {dot_err['avg']:.3f}/{dot_err['p95']:.3f} |"
        )

    print("\nBy matrix kind:")
    print("| kind | type | codec | samples | weight nRMSE avg/p95 | dot nMAE avg/p95 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key in sorted(grouped_kind):
        weight = summarize(grouped_kind[key]["weight_nrmse"])
        dot_err = summarize(grouped_kind[key]["dot_nmae"])
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(grouped_kind[key]['dot_nmae'])} | "
            f"{weight['avg']:.3f}/{weight['p95']:.3f} | {dot_err['avg']:.3f}/{dot_err['p95']:.3f} |"
        )

    print("\nExpert-size estimate, cold expert only:")
    for group_size in (256, 64, 32):
        expert_mib = policy_expert_mib(group_size, "all")
        all_gib = expert_mib * 43 * 256 / 1024
        print(f"  cq1g{group_size}: {expert_mib:.3f} MiB/expert, all routed experts ~= {all_gib:.2f} GiB")

    print("\nMiddle-ground policies:")
    native_expert_mib = NATIVE_GATE_MIB + NATIVE_UP_MIB + NATIVE_DOWN_MIB
    print(f"  native: {native_expert_mib:.3f} MiB/expert, all routed experts ~= {native_expert_mib * DS4_ROUTED_EXPERTS / 1024:.2f} GiB")
    for group_size in (256, 64, 32):
        for policy in ("down-only", "gate-up-only", "all"):
            expert_mib = policy_expert_mib(group_size, policy)
            saved = 1.0 - expert_mib / native_expert_mib
            print(
                f"  {policy}:cq1g{group_size}: {expert_mib:.3f} MiB/expert "
                f"({saved * 100:.1f}% saved), all ~= {expert_mib * DS4_ROUTED_EXPERTS / 1024:.2f} GiB"
            )

    print("\nDynamic tier examples:")
    for hot_cap in (512, 1024, 2048):
        for policy, group_size in (("down-only", 64), ("all", 32), ("all", 64)):
            cold_mib = policy_expert_mib(group_size, policy)
            total_gib = (hot_cap * native_expert_mib + (DS4_ROUTED_EXPERTS - hot_cap) * cold_mib) / 1024
            print(f"  hot_native={hot_cap:4d} + cold={policy}:cq1g{group_size}: ~= {total_gib:.2f} GiB routed-expert RAM")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="DS4 GGUF path")
    parser.add_argument("--ds4-root", default="/root/ds4", help="Path containing ds4_iq2_tables_cuda.inc")
    parser.add_argument("--blocks", type=int, default=256, help="Random native blocks to sample")
    parser.add_argument("--dot-vectors", type=int, default=8, help="Random Q8-like dot vectors per block")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--codecs", default="cq1g256,cq1g64,cq1g32")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = sample_blocks(args)
    print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
