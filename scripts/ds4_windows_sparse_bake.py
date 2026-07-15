#!/usr/bin/env python3
"""Build and unpack a self-contained sparse DS4 bake for native Windows.

The pack preserves every original GGUF tensor offset. Masked routed-expert
slices are omitted from the payload and become sparse holes when unpacked.
The unpacked file receives an embedded DS4 bake manifest after the original
GGUF logical end; a compatible ds4-win loader must validate and apply it before
routing. Expert ids therefore remain the original DS4 ids.

The ``plan``, ``pack``, and ``pack-stream`` commands require the ``gguf``
Python package. The ``unpack`` command uses only the standard library and marks
the output sparse through FSCTL_SET_SPARSE on Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import zlib
from dataclasses import dataclass


PACK_MAGIC = b"DS4BAKEPACKv1\0\0\0"
PACK_END_MAGIC = b"DS4BAKEENDv1\0\0\0\0"
FILE_END_MAGIC = b"DS4BAKEFILEv1\0\0\0"
FOOTER = struct.Struct("<16sQ32s32s")
FILE_FOOTER = struct.Struct("<16sIIIIQQII")
ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
N_EXPERT = 256
MASK_BYTES_PER_LAYER = N_EXPERT // 8

assert len(PACK_MAGIC) == 16
assert len(PACK_END_MAGIC) == 16
assert len(FILE_END_MAGIC) == 16


def retained_mask_blob(manifest: dict) -> tuple[bytes, int]:
    routed_layers = [int(item["layer"]) for item in manifest["routed_tensors"]]
    if not routed_layers:
        raise ValueError("manifest contains no routed layers")
    n_layers = max(routed_layers) + 1
    selected_by_layer = manifest["selected_experts_by_layer"]
    blob = bytearray(n_layers * MASK_BYTES_PER_LAYER)
    for layer in range(n_layers):
        selected = selected_by_layer.get(str(layer), list(range(N_EXPERT)))
        if len(selected) < 6:
            raise ValueError(f"layer {layer}: bake retains fewer than router top-k experts")
        for expert in selected:
            expert = int(expert)
            if expert < 0 or expert >= N_EXPERT:
                raise ValueError(f"layer {layer}: invalid selected expert {expert}")
            blob[layer * MASK_BYTES_PER_LAYER + expert // 8] |= 1 << (expert % 8)
    return bytes(blob), n_layers


@dataclass(order=True)
class Extent:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class PackWriteResult:
    total_bytes: int
    payload_sha256: str
    manifest_sha256: str
    full_pack_sha256: str


def sha256_file(path: pathlib.Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: pathlib.Path) -> tuple[dict[int, set[int]], str]:
    blocked: dict[int, set[int]] = {}
    with path.open(encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected '<layer> <expert>'")
            layer, expert = map(int, parts)
            if layer < 0 or expert < 0 or expert >= N_EXPERT:
                raise ValueError(f"{path}:{lineno}: invalid layer/expert")
            blocked.setdefault(layer, set()).add(expert)
    if not blocked:
        raise ValueError(f"empty mask: {path}")
    return blocked, sha256_file(path)


def merge_extents(extents: list[Extent], max_gap: int) -> list[Extent]:
    merged: list[Extent] = []
    for current in sorted(extents):
        if current.length <= 0:
            continue
        if not merged or current.offset > merged[-1].end + max_gap:
            merged.append(Extent(current.offset, current.length))
            continue
        previous = merged[-1]
        previous.length = max(previous.end, current.end) - previous.offset
    return merged


def build_plan(
    model_path: pathlib.Path,
    mask_path: pathlib.Path,
    source_sha256: str | None,
    merge_gap: int,
) -> dict:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise SystemExit("plan/pack require: pip install gguf") from exc

    blocked, mask_sha256 = load_mask(mask_path)
    reader = GGUFReader(str(model_path))
    model_size = model_path.stat().st_size
    if not reader.tensors:
        raise ValueError("GGUF contains no tensors")

    first_data = min(int(t.data_offset) for t in reader.tensors)
    extents = [Extent(0, first_data)]
    routed = []
    selected_by_layer: dict[str, list[int]] = {}
    full_routed_layers: set[int] = set()

    for tensor in reader.tensors:
        offset = int(tensor.data_offset)
        n_bytes = int(tensor.n_bytes)
        match = ROUTED_RE.match(tensor.name)
        if not match:
            extents.append(Extent(offset, n_bytes))
            continue

        layer = int(match.group(1))
        if n_bytes % N_EXPERT:
            raise ValueError(f"{tensor.name}: bytes not divisible by {N_EXPERT}")
        slice_bytes = n_bytes // N_EXPERT
        if layer not in blocked:
            full_routed_layers.add(layer)
            selected = list(range(N_EXPERT))
            extents.append(Extent(offset, n_bytes))
        else:
            selected = sorted(set(range(N_EXPERT)) - blocked[layer])
            if not selected:
                raise ValueError(f"layer {layer}: mask retains no experts")
            selected_by_layer.setdefault(str(layer), selected)
            for expert in selected:
                extents.append(Extent(offset + expert * slice_bytes, slice_bytes))
        routed.append(
            {
                "name": tensor.name,
                "layer": layer,
                "kind": match.group(2),
                "offset": offset,
                "bytes": n_bytes,
                "slice_bytes": slice_bytes,
                "tensor_type": int(tensor.tensor_type),
                "selected_count": len(selected),
            }
        )

    planned_layers = {int(layer) for layer in selected_by_layer}
    if planned_layers != set(blocked):
        missing = sorted(set(blocked) - planned_layers)
        raise ValueError(f"mask layers absent from routed tensors: {missing}")

    merged = merge_extents(extents, merge_gap)
    for left, right in zip(merged, merged[1:]):
        if left.end > right.offset:
            raise AssertionError("overlapping extents after merge")
    payload_bytes = sum(extent.length for extent in merged)
    retained_counts = {layer: len(ids) for layer, ids in selected_by_layer.items()}

    return {
        "format": "ds4-windows-sparse-bake",
        "version": 1,
        "source_model_name": model_path.name,
        "source_model_size": model_size,
        "source_model_sha256": source_sha256,
        "mask_name": mask_path.name,
        "mask_sha256": mask_sha256,
        "tensor_count": len(reader.tensors),
        "routed_tensor_count": len(routed),
        "full_routed_layers": sorted(full_routed_layers),
        "selected_experts_by_layer": selected_by_layer,
        "retained_count_by_layer": retained_counts,
        "merge_gap": merge_gap,
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / (1 << 30),
        "logical_savings_bytes": model_size - payload_bytes,
        "logical_savings_gib": (model_size - payload_bytes) / (1 << 30),
        "extents": [[extent.offset, extent.length] for extent in merged],
        "routed_tensors": routed,
    }


def _write_all(target, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = target.write(view)
        if written is None:
            return
        if written <= 0:
            raise BrokenPipeError("write returned no bytes")
        view = view[written:]


def write_pack_stream(model_path: pathlib.Path, target, plan: dict) -> PackWriteResult:
    payload_digest = hashlib.sha256()
    full_digest = hashlib.sha256()
    copied = 0
    total_bytes = 0

    def write(data: bytes) -> None:
        nonlocal total_bytes
        _write_all(target, data)
        full_digest.update(data)
        total_bytes += len(data)

    with model_path.open("rb") as source:
        write(PACK_MAGIC)
        for offset, length in plan["extents"]:
            source.seek(offset)
            remaining = length
            while remaining:
                block = source.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError(f"source ended in extent offset={offset} length={length}")
                write(block)
                payload_digest.update(block)
                copied += len(block)
                remaining -= len(block)
        if copied != plan["payload_bytes"]:
            raise AssertionError("payload byte count mismatch")
        plan = dict(plan)
        plan["payload_sha256"] = payload_digest.hexdigest()
        manifest = json.dumps(plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
        manifest_digest = hashlib.sha256(manifest)
        write(manifest)
        write(
            FOOTER.pack(
                PACK_END_MAGIC,
                len(manifest),
                payload_digest.digest(),
                manifest_digest.digest(),
            )
        )
    return PackWriteResult(
        total_bytes=total_bytes,
        payload_sha256=payload_digest.hexdigest(),
        manifest_sha256=manifest_digest.hexdigest(),
        full_pack_sha256=full_digest.hexdigest(),
    )


def write_pack(model_path: pathlib.Path, pack_path: pathlib.Path, plan: dict) -> PackWriteResult:
    with pack_path.open("wb") as target:
        result = write_pack_stream(model_path, target, plan)
    print(json.dumps({"pack": str(pack_path), "bytes": pack_path.stat().st_size,
                      "payload_sha256": result.payload_sha256}, indent=2))
    return result


def write_status_atomic(path: pathlib.Path, result: PackWriteResult) -> None:
    payload = json.dumps(
        {
            "total_bytes": result.total_bytes,
            "payload_sha256": result.payload_sha256,
            "manifest_sha256": result.manifest_sha256,
            "full_pack_sha256": result.full_pack_sha256,
        },
        indent=2,
        sort_keys=True,
    )
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path)


def read_pack_manifest(pack_path: pathlib.Path) -> tuple[dict, bytes, bytes]:
    size = pack_path.stat().st_size
    if size < len(PACK_MAGIC) + FOOTER.size:
        raise ValueError("pack is too small")
    with pack_path.open("rb") as stream:
        if stream.read(len(PACK_MAGIC)) != PACK_MAGIC:
            raise ValueError("invalid pack magic")
        stream.seek(-FOOTER.size, os.SEEK_END)
        magic, manifest_len, payload_hash, manifest_hash = FOOTER.unpack(stream.read(FOOTER.size))
        if magic != PACK_END_MAGIC:
            raise ValueError("invalid pack footer")
        manifest_offset = size - FOOTER.size - manifest_len
        if manifest_offset < len(PACK_MAGIC):
            raise ValueError("invalid manifest length")
        stream.seek(manifest_offset)
        manifest_bytes = stream.read(manifest_len)
    if hashlib.sha256(manifest_bytes).digest() != manifest_hash:
        raise ValueError("manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    return manifest, payload_hash, manifest_bytes


def mark_sparse_windows(stream) -> None:
    if os.name != "nt":
        return
    import ctypes
    import msvcrt

    fsctl_set_sparse = 0x000900C4
    handle = msvcrt.get_osfhandle(stream.fileno())
    returned = ctypes.c_ulong(0)
    ok = ctypes.windll.kernel32.DeviceIoControl(
        ctypes.c_void_p(handle),
        fsctl_set_sparse,
        None,
        0,
        None,
        0,
        ctypes.byref(returned),
        None,
    )
    if not ok:
        raise ctypes.WinError()


def unpack(pack_path: pathlib.Path, output_path: pathlib.Path) -> None:
    manifest, expected_payload_hash, manifest_bytes = read_pack_manifest(pack_path)
    original_size = int(manifest["source_model_size"])
    mask_blob, n_layers = retained_mask_blob(manifest)
    extents = [Extent(int(offset), int(length)) for offset, length in manifest["extents"]]
    payload_bytes = int(manifest["payload_bytes"])
    payload_end = len(PACK_MAGIC) + payload_bytes
    digest = hashlib.sha256()

    with pack_path.open("rb") as source, output_path.open("w+b") as target:
        mark_sparse_windows(target)
        target.truncate(original_size)
        source.seek(len(PACK_MAGIC))
        copied = 0
        for extent in extents:
            target.seek(extent.offset)
            remaining = extent.length
            while remaining:
                block = source.read(min(8 << 20, remaining))
                if not block:
                    raise EOFError("pack payload ended early")
                target.write(block)
                digest.update(block)
                copied += len(block)
                remaining -= len(block)
        if copied != payload_bytes or source.tell() != payload_end:
            raise ValueError("pack payload size mismatch")
        if digest.digest() != expected_payload_hash:
            raise ValueError("pack payload checksum mismatch")

        target.seek(original_size)
        target.write(manifest_bytes)
        target.write(mask_blob)
        target.write(
            FILE_FOOTER.pack(
                FILE_END_MAGIC,
                1,
                n_layers,
                N_EXPERT,
                len(mask_blob),
                original_size,
                len(manifest_bytes),
                zlib.crc32(manifest_bytes),
                zlib.crc32(mask_blob),
            )
        )
        target.flush()
        os.fsync(target.fileno())

    print(json.dumps({"output": str(output_path), "logical_bytes": output_path.stat().st_size,
                      "payload_sha256": digest.hexdigest(),
                      "embedded_mask_bytes": len(mask_blob),
                      "embedded_mask_crc32": f"{zlib.crc32(mask_blob):08x}"}, indent=2))


def inspect_bake(path: pathlib.Path) -> dict:
    size = path.stat().st_size
    if size < FILE_FOOTER.size:
        raise ValueError("bake is too small")
    with path.open("rb") as stream:
        stream.seek(-FILE_FOOTER.size, os.SEEK_END)
        fields = FILE_FOOTER.unpack(stream.read(FILE_FOOTER.size))
        (magic, version, n_layers, n_experts, mask_len, source_size,
         manifest_len, manifest_crc32, mask_crc32) = fields
        if magic != FILE_END_MAGIC:
            raise ValueError("invalid DS4BAKE file footer")
        if version != 1 or n_experts != N_EXPERT:
            raise ValueError("unsupported DS4BAKE dimensions/version")
        if mask_len != n_layers * MASK_BYTES_PER_LAYER:
            raise ValueError("invalid DS4BAKE mask length")
        trailer_len = manifest_len + mask_len + FILE_FOOTER.size
        if source_size + trailer_len != size:
            raise ValueError("DS4BAKE logical size/trailer mismatch")
        stream.seek(source_size)
        manifest_bytes = stream.read(manifest_len)
        mask_blob = stream.read(mask_len)
    if zlib.crc32(manifest_bytes) != manifest_crc32:
        raise ValueError("DS4BAKE manifest CRC32 mismatch")
    if zlib.crc32(mask_blob) != mask_crc32:
        raise ValueError("DS4BAKE retained-mask CRC32 mismatch")
    manifest = json.loads(manifest_bytes)
    expected_mask, expected_layers = retained_mask_blob(manifest)
    if expected_layers != n_layers or expected_mask != mask_blob:
        raise ValueError("DS4BAKE bitset does not match audit manifest")
    retained = []
    for layer in range(n_layers):
        base = layer * MASK_BYTES_PER_LAYER
        retained.append(sum(mask_blob[base + e // 8] >> (e % 8) & 1
                            for e in range(N_EXPERT)))
    return {
        "path": str(path),
        "version": version,
        "source_model_size": source_size,
        "manifest_bytes": manifest_len,
        "mask_bytes": mask_len,
        "manifest_crc32": f"{manifest_crc32:08x}",
        "mask_crc32": f"{mask_crc32:08x}",
        "retained_by_layer": retained,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "pack", "pack-stream"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", type=pathlib.Path, required=True)
        cmd.add_argument("--mask", type=pathlib.Path, required=True)
        cmd.add_argument("--source-sha256")
        cmd.add_argument("--verify-source", action="store_true")
        cmd.add_argument("--merge-gap", type=int, default=4096)
        if name == "pack":
            cmd.add_argument("--out", type=pathlib.Path, required=True)
        if name == "pack-stream":
            cmd.add_argument("--status-out", type=pathlib.Path, required=True)

    cmd = sub.add_parser("unpack")
    cmd.add_argument("--pack", type=pathlib.Path, required=True)
    cmd.add_argument("--out", type=pathlib.Path, required=True)
    cmd = sub.add_parser("inspect")
    cmd.add_argument("--bake", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_bake(args.bake), indent=2))
        return 0
    if args.command == "unpack":
        unpack(args.pack, args.out)
        return 0

    source_hash = args.source_sha256
    if args.verify_source:
        measured = sha256_file(args.model)
        if source_hash and measured.lower() != source_hash.lower():
            raise SystemExit(f"source SHA256 mismatch: expected {source_hash}, measured {measured}")
        source_hash = measured
    plan = build_plan(args.model, args.mask, source_hash, args.merge_gap)
    if args.command == "plan":
        compact = {key: value for key, value in plan.items()
                   if key not in ("extents", "routed_tensors", "selected_experts_by_layer")}
        compact["extent_count"] = len(plan["extents"])
        print(json.dumps(compact, indent=2, sort_keys=True))
        return 0
    if args.command == "pack-stream":
        try:
            result = write_pack_stream(args.model, sys.stdout.buffer, plan)
            sys.stdout.buffer.flush()
        except BrokenPipeError as exc:
            raise SystemExit("pack-stream failed: stdout pipe closed") from exc
        write_status_atomic(args.status_out, result)
        return 0
    write_pack(args.model, args.out, plan)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        finally:
            raise SystemExit(1)
