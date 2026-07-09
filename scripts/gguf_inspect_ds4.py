#!/usr/bin/env python3
"""Inspect DS4 GGUF files and sample routed-expert compressibility.

This intentionally stays dependency-free so it can run on fresh pods and local
WSL installs.  The parser covers the GGUF header, metadata, tensor directory,
and enough tensor layout math to extract a single routed expert
gate/up/down payload from a DS4-style model.
"""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from dataclasses import dataclass
import json
import lzma
import math
import mmap
import os
import random
import re
import statistics
import struct
import sys
import time
import zlib


VALUE_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}

VALUE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# Mirrors ds4.c gguf_types[] for the types present in DS4 Flash.
TYPE_INFO = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22),
    7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34),
    9: ("q8_1", 32, 40),
    10: ("q2_k", 256, 84),
    11: ("q3_k", 256, 110),
    12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176),
    14: ("q6_k", 256, 210),
    15: ("q8_k", 256, 292),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 110),
    20: ("iq4_nl", 256, 50),
    21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82),
    23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1),
    25: ("i16", 1, 2),
    26: ("i32", 1, 4),
    27: ("i64", 1, 8),
    28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2),
}

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps(?:\.weight)?$")


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    type_id: int
    rel_offset: int
    abs_offset: int
    nbytes: int

    @property
    def type_name(self) -> str:
        return TYPE_INFO.get(self.type_id, (f"unknown:{self.type_id}", 0, 0))[0]


@dataclass
class GGUF:
    path: str
    version: int
    kv: dict[str, object]
    tensors: list[Tensor]
    data_start: int
    alignment: int


def read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected EOF while reading GGUF")
    return data


def read_u32(f) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_str(f) -> str:
    n = read_u64(f)
    return read_exact(f, n).decode("utf-8", "replace")


def scalar_size(value_type: int) -> int:
    fmt = VALUE_FORMATS.get(value_type)
    return struct.calcsize(fmt) if fmt else 0


def skip_value(f, value_type: int) -> None:
    size = scalar_size(value_type)
    if size:
        f.seek(size, os.SEEK_CUR)
        return
    if value_type == 8:
        f.seek(read_u64(f), os.SEEK_CUR)
        return
    if value_type == 9:
        item_type = read_u32(f)
        n = read_u64(f)
        item_size = scalar_size(item_type)
        if item_size:
            f.seek(item_size * n, os.SEEK_CUR)
            return
        for _ in range(n):
            skip_value(f, item_type)
        return
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def read_value(f, value_type: int, max_array: int = 512) -> object:
    fmt = VALUE_FORMATS.get(value_type)
    if fmt:
        return struct.unpack(fmt, read_exact(f, struct.calcsize(fmt)))[0]
    if value_type == 8:
        return read_str(f)
    if value_type == 9:
        item_type = read_u32(f)
        n = read_u64(f)
        if n <= max_array:
            return [read_value(f, item_type, max_array=max_array) for _ in range(n)]
        skip_value_payload(f, item_type, n)
        name = VALUE_NAMES.get(item_type, str(item_type))
        return f"<array {name} len {n}>"
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def skip_value_payload(f, item_type: int, n: int) -> None:
    item_size = scalar_size(item_type)
    if item_size:
        f.seek(item_size * n, os.SEEK_CUR)
        return
    for _ in range(n):
        skip_value(f, item_type)


def align_up(value: int, alignment: int) -> int:
    rem = value % alignment
    return value if rem == 0 else value + alignment - rem


def tensor_nbytes(type_id: int, elements: int) -> int:
    if type_id not in TYPE_INFO:
        return 0
    _, block_elems, block_bytes = TYPE_INFO[type_id]
    return math.ceil(elements / block_elems) * block_bytes


def parse_gguf(path: str) -> GGUF:
    with open(path, "rb") as f:
        magic = read_exact(f, 4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)

        kv: dict[str, object] = {}
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            value_type = read_u32(f)
            value = read_value(f, value_type)
            kv[key] = value
            if key == "general.alignment" and isinstance(value, int) and value:
                alignment = value

        raw_tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            ndim = read_u32(f)
            dims = tuple(read_u64(f) for _ in range(ndim))
            type_id = read_u32(f)
            rel_offset = read_u64(f)
            elements = math.prod(dims)
            raw_tensors.append((name, dims, type_id, rel_offset, tensor_nbytes(type_id, elements)))

        data_start = align_up(f.tell(), alignment)
        tensors = [
            Tensor(
                name=name,
                dims=dims,
                type_id=type_id,
                rel_offset=rel_offset,
                abs_offset=data_start + rel_offset,
                nbytes=nbytes,
            )
            for name, dims, type_id, rel_offset, nbytes in raw_tensors
        ]
    return GGUF(path=path, version=version, kv=kv, tensors=tensors, data_start=data_start, alignment=alignment)


def routed_expert_row_bytes(tensor: Tensor) -> int:
    if tensor.type_id not in TYPE_INFO:
        raise ValueError(f"unsupported routed expert type {tensor.type_id} in {tensor.name}")
    _, block_elems, block_bytes = TYPE_INFO[tensor.type_id]
    if not tensor.dims or tensor.dims[0] % block_elems != 0:
        raise ValueError(f"{tensor.name} dim[0]={tensor.dims[0]} is not aligned to {block_elems}")
    return (tensor.dims[0] // block_elems) * block_bytes


def routed_expert_nbytes(tensor: Tensor) -> int:
    if len(tensor.dims) != 3:
        raise ValueError(f"{tensor.name} is not a routed expert tensor")
    return routed_expert_row_bytes(tensor) * tensor.dims[1]


def discover_expert_layers(gguf: GGUF) -> dict[int, dict[str, Tensor]]:
    layers: dict[int, dict[str, Tensor]] = {}
    for tensor in gguf.tensors:
        match = EXPERT_RE.match(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        kind = match.group(2)
        layers.setdefault(layer, {})[kind] = tensor
    return {layer: parts for layer, parts in layers.items() if {"gate", "up", "down"} <= parts.keys()}


def parse_int_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(chunk))
    return out


def expert_blob(mm: mmap.mmap, tensors: dict[str, Tensor], expert: int) -> bytes:
    chunks = []
    for kind in ("gate", "up", "down"):
        tensor = tensors[kind]
        nbytes = routed_expert_nbytes(tensor)
        if expert >= tensor.dims[2]:
            raise ValueError(f"expert {expert} outside {tensor.name} dims[2]={tensor.dims[2]}")
        offset = tensor.abs_offset + expert * nbytes
        chunks.append(mm[offset : offset + nbytes])
    return b"".join(chunks)


def entropy_bits_per_byte(data: bytes) -> float:
    counts = Counter(data)
    n = len(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def compress_one(data: bytes, name: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    if name == "zlib1":
        compressed = zlib.compress(data, 1)
    elif name == "zlib6":
        compressed = zlib.compress(data, 6)
    elif name == "bz2-1":
        compressed = bz2.compress(data, compresslevel=1)
    elif name == "lzma0":
        compressed = lzma.compress(data, preset=0)
    else:
        raise ValueError(f"unknown compression algorithm {name}")
    ms = (time.perf_counter() - t0) * 1000.0
    return len(compressed), ms


def sample_compression(args: argparse.Namespace) -> list[dict[str, object]]:
    gguf = parse_gguf(args.model)
    layers = discover_expert_layers(gguf)
    if not layers:
        raise SystemExit("no DS4 routed expert layers found")

    wanted_layers = parse_int_list(args.layers)
    wanted_experts = parse_int_list(args.experts)
    available_layers = sorted(layers)
    if wanted_layers is not None:
        available_layers = [layer for layer in wanted_layers if layer in layers]
    if not available_layers:
        raise SystemExit("no requested layers are present")

    expert_count = min(parts["gate"].dims[2] for parts in layers.values())
    available_experts = list(range(expert_count))
    if wanted_experts is not None:
        available_experts = [expert for expert in wanted_experts if 0 <= expert < expert_count]
    if not available_experts:
        raise SystemExit("no requested experts are present")

    pairs = [(layer, expert) for layer in available_layers for expert in available_experts]
    rng = random.Random(args.seed)
    if args.samples and args.samples < len(pairs):
        pairs = rng.sample(pairs, args.samples)

    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    rows = []
    with open(args.model, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for layer, expert in pairs:
                data = expert_blob(mm, layers[layer], expert)
                counts = Counter(data)
                row: dict[str, object] = {
                    "layer": layer,
                    "expert": expert,
                    "native_bytes": len(data),
                    "native_mib": len(data) / (1024 * 1024),
                    "entropy_bits_per_byte": entropy_bits_per_byte(data),
                    "zero_ratio": counts.get(0, 0) / len(data),
                    "ff_ratio": counts.get(255, 0) / len(data),
                }
                for algo in algorithms:
                    clen, ms = compress_one(data, algo)
                    row[f"{algo}_bytes"] = clen
                    row[f"{algo}_ratio"] = clen / len(data)
                    row[f"{algo}_ms"] = ms
                rows.append(row)
        finally:
            mm.close()
    return rows


def print_inspect(gguf: GGUF, dump_prefixes: list[str]) -> None:
    print(f"== {gguf.path}: gguf v{gguf.version}, {len(gguf.tensors)} tensors")
    print(f"  tensor_data_pos={gguf.data_start} alignment={gguf.alignment}")
    for key, value in gguf.kv.items():
        if "expert" in key or "block_count" in key or "hash" in key or key.startswith("general."):
            shown = value[:8] if isinstance(value, list) else value
            print(f"  KV {key} = {shown}")

    seen = {}
    for tensor in gguf.tensors:
        for prefix in dump_prefixes:
            if prefix in tensor.name:
                key = tensor.name.split(".")
                base = ".".join(key[2:]) if key[0] == "blk" else tensor.name
                seen.setdefault(base, tensor)
    print("  -- sample tensors:")
    for base, tensor in sorted(seen.items()):
        print(f"  {tensor.name:48s} dims={list(tensor.dims)} type={tensor.type_name}({tensor.type_id})")

    layers = discover_expert_layers(gguf)
    if layers:
        first_layer = layers[sorted(layers)[0]]
        per = sum(routed_expert_nbytes(first_layer[k]) for k in ("gate", "up", "down"))
        print(f"  routed expert layers: {len(layers)}; experts/layer={first_layer['gate'].dims[2]}; native/expert={per / (1024 * 1024):.3f} MiB")
    print(
        "  exp_probs_b: %d   tid2eid: %d   ffn_gate_exps: %d"
        % (
            sum(1 for t in gguf.tensors if "exp_probs_b" in t.name),
            sum(1 for t in gguf.tensors if "tid2eid" in t.name),
            sum(1 for t in gguf.tensors if "ffn_gate_exps" in t.name),
        )
    )


def print_compression(rows: list[dict[str, object]], algorithms: list[str]) -> None:
    headers = ["layer", "expert", "MiB", "entropy", "zero"]
    for algo in algorithms:
        headers += [f"{algo} ratio", f"{algo} ms"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            str(row["layer"]),
            str(row["expert"]),
            f"{row['native_mib']:.3f}",
            f"{row['entropy_bits_per_byte']:.3f}",
            f"{100.0 * row['zero_ratio']:.2f}%",
        ]
        for algo in algorithms:
            values += [f"{row[f'{algo}_ratio']:.4f}", f"{row[f'{algo}_ms']:.1f}"]
        print("| " + " | ".join(values) + " |")

    print("\nSummary:")
    print(f"  samples={len(rows)} native_avg_mib={statistics.mean(float(r['native_mib']) for r in rows):.3f}")
    for algo in algorithms:
        ratios = [float(row[f"{algo}_ratio"]) for row in rows]
        times = [float(row[f"{algo}_ms"]) for row in rows]
        print(
            f"  {algo}: ratio avg={statistics.mean(ratios):.4f} "
            f"min={min(ratios):.4f} max={max(ratios):.4f}; "
            f"time avg={statistics.mean(times):.1f} ms"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=None, help="GGUF model path")
    parser.add_argument(
        "--dump-prefix",
        action="append",
        default=["ffn_gate_inp", "exp_probs_b", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "tid2eid"],
        help="Tensor-name substring to show during inspection; repeatable.",
    )
    parser.add_argument("--compress-sample", action="store_true", help="Sample routed expert gate/up/down compression")
    parser.add_argument("--samples", type=int, default=8, help="Maximum sampled layer/expert pairs")
    parser.add_argument("--layers", default=None, help="Layer list/ranges, e.g. 3,10,20-22")
    parser.add_argument("--experts", default=None, help="Expert list/ranges, e.g. 0,7,31-35")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampled pairs")
    parser.add_argument("--algorithms", default="zlib1,zlib6,lzma0", help="Comma-separated: zlib1,zlib6,bz2-1,lzma0")
    parser.add_argument("--json-out", default=None, help="Optional path for compression sample JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.model:
        raise SystemExit("model path required")

    gguf = parse_gguf(args.model)
    print_inspect(gguf, args.dump_prefix)

    if args.compress_sample:
        print()
        rows = sample_compression(args)
        algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
        print_compression(rows, algorithms)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
