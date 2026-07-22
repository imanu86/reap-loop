#!/usr/bin/env python3
"""Pack and validate DS4 nested residual sidecars.

The sidecar stores only exact residual bytes for routed expert tensors.  The
source GGUF identity and tensor geometry are fixed in a little-endian binary
header plus fixed-size records, so validation does not need a JSON manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

try:
    import numpy as np
except ImportError:  # pragma: no cover - scalar fallback keeps the tool portable
    np = None


HERE = Path(__file__).resolve().parent

LAB_SPEC = importlib.util.spec_from_file_location(
    "ds4_nested_residual_lab", HERE / "ds4_nested_residual_lab.py"
)
assert LAB_SPEC and LAB_SPEC.loader
lab = importlib.util.module_from_spec(LAB_SPEC)
sys.modules[LAB_SPEC.name] = lab
LAB_SPEC.loader.exec_module(lab)

GGUF_SPEC = importlib.util.spec_from_file_location("gguf_inspect_ds4", HERE / "gguf_inspect_ds4.py")
assert GGUF_SPEC and GGUF_SPEC.loader
gguf_inspect_ds4 = importlib.util.module_from_spec(GGUF_SPEC)
sys.modules[GGUF_SPEC.name] = gguf_inspect_ds4
GGUF_SPEC.loader.exec_module(gguf_inspect_ds4)


MAGIC = b"DS4NRSC\0"
VERSION = 2
HEADER_STRUCT = struct.Struct("<8sIIQ32sQ32sIQI")
HEADER_CRC_OFFSET = 8 + 4 + 4 + 8 + 32 + 8 + 32
HEADER_BYTES = HEADER_STRUCT.size
RECORD_STRUCT = struct.Struct("<IIIQQQQQQQQQQIIII")
RECORD_SIZE = RECORD_STRUCT.size

KIND_ENUM = {"gate": 1, "up": 2, "down": 3}
KIND_NAME = {value: key for key, value in KIND_ENUM.items()}
TYPE_FORMATS = {
    16: (
        66,
        lab.IQ2_XXS_BASE_BYTES,
        lab.IQ2_XXS_RESIDUAL_BYTES,
        lab.split_iq2_xxs,
        lab.join_iq2_xxs,
    ),
    10: (84, lab.Q2_K_BASE_BYTES, lab.Q2_K_RESIDUAL_BYTES, lab.split_q2_k, lab.join_q2_k),
}
STREAM_CHUNK = 16 * 1024 * 1024

IQ2_XXS_RESIDUAL_COLUMNS = tuple(
    2 + group * 8 + lane for group in range(8) for lane in range(4)
)
Q2_K_QBYTE_SHIFT = tuple(
    (
        (group // 8) * 32 + (16 if (group % 8) % 2 else 0) + lane,
        ((group % 8) // 2) * 2,
    )
    for group in range(16)
    for lane in range(16)
)


def extract_residuals(native: bytes, source_type: int) -> bytes:
    """Extract block residuals in bulk while preserving block order."""
    if source_type not in TYPE_FORMATS:
        raise ValueError(f"unsupported source type {source_type}")
    native_block, _, residual_block, split_block, _ = TYPE_FORMATS[source_type]
    if len(native) % native_block:
        raise ValueError("native payload is not block aligned")
    if not native:
        return b""
    if np is None:
        return b"".join(split_block(native[offset : offset + native_block])[1] for offset in range(0, len(native), native_block))

    blocks = np.frombuffer(native, dtype=np.uint8).reshape(-1, native_block)
    if source_type == 16:
        residual = blocks[:, IQ2_XXS_RESIDUAL_COLUMNS]
    else:
        qs = blocks[:, 16:80]
        qbytes = np.asarray([entry[0] for entry in Q2_K_QBYTE_SHIFT], dtype=np.intp)
        shifts = np.asarray([entry[1] for entry in Q2_K_QBYTE_SHIFT], dtype=np.uint8)
        bits = (qs[:, qbytes] >> shifts) & 1
        residual = np.packbits(bits, axis=1, bitorder="little")
    if residual.shape[1] != residual_block:
        raise AssertionError("bulk residual extraction returned the wrong block size")
    return residual.tobytes(order="C")


@dataclass(frozen=True)
class Header:
    source_size: int
    source_sha256: bytes
    payload_bytes: int
    payload_sha256: bytes
    record_count: int
    record_size: int = RECORD_SIZE
    header_bytes: int = HEADER_BYTES
    header_crc32: int = 0


@dataclass(frozen=True)
class Record:
    layer: int
    kind: int
    source_type: int
    ncols: int
    nrows: int
    nexperts: int
    source_offset: int
    source_bytes: int
    residual_offset: int
    residual_bytes: int
    expert_stride: int
    kind_offset_within_expert: int
    residual_expert_bytes: int
    native_block_bytes: int
    base_block_bytes: int
    residual_block_bytes: int
    reserved: int = 0

    @property
    def kind_name(self) -> str:
        return KIND_NAME.get(self.kind, f"unknown:{self.kind}")


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(STREAM_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.digest()


def sha256_region(path: Path, offset: int, nbytes: int) -> bytes:
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            chunk = handle.read(min(STREAM_CHUNK, remaining))
            if not chunk:
                raise ValueError("unexpected EOF while hashing sidecar payload")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.digest()


def _pack_header(header: Header, crc: int) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        header.header_bytes,
        header.source_size,
        header.source_sha256,
        header.payload_bytes,
        header.payload_sha256,
        crc,
        header.record_count,
        header.record_size,
    )


def _header_crc(header_bytes: bytes, record_table: bytes) -> int:
    data = bytearray(header_bytes)
    struct.pack_into("<I", data, HEADER_CRC_OFFSET, 0)
    return zlib.crc32(record_table, zlib.crc32(data)) & 0xFFFFFFFF


def pack_header(header: Header, record_table: bytes) -> bytes:
    if len(record_table) != header.record_count * header.record_size:
        raise ValueError("record table size does not match sidecar header")
    raw = _pack_header(header, 0)
    return _pack_header(header, _header_crc(raw, record_table))


def unpack_header(data: bytes, record_table: bytes) -> Header:
    if len(data) != HEADER_BYTES:
        raise ValueError("truncated sidecar header")
    magic, version, header_bytes, source_size, source_sha, payload_bytes, payload_sha, crc, count, record_size = (
        HEADER_STRUCT.unpack(data)
    )
    if magic != MAGIC:
        raise ValueError("invalid sidecar magic")
    if version != VERSION:
        raise ValueError(f"unsupported sidecar version {version}")
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"unsupported sidecar header size {header_bytes}")
    if record_size != RECORD_SIZE:
        raise ValueError(f"unsupported sidecar record size {record_size}")
    if len(record_table) != count * record_size:
        raise ValueError("truncated sidecar record table")
    if _header_crc(data, record_table) != crc:
        raise ValueError("invalid sidecar header/record-table CRC")
    return Header(
        source_size,
        source_sha,
        payload_bytes,
        payload_sha,
        count,
        record_size,
        header_bytes,
        crc,
    )


def pack_record(record: Record) -> bytes:
    return RECORD_STRUCT.pack(
        record.layer,
        record.kind,
        record.source_type,
        record.ncols,
        record.nrows,
        record.nexperts,
        record.source_offset,
        record.source_bytes,
        record.residual_offset,
        record.residual_bytes,
        record.expert_stride,
        record.kind_offset_within_expert,
        record.residual_expert_bytes,
        record.native_block_bytes,
        record.base_block_bytes,
        record.residual_block_bytes,
        record.reserved,
    )


def unpack_record(data: bytes) -> Record:
    if len(data) != RECORD_SIZE:
        raise ValueError("truncated sidecar record")
    return Record(*RECORD_STRUCT.unpack(data))


def _expected_bytes(record: Record) -> tuple[int, int, int]:
    if record.source_type not in TYPE_FORMATS:
        raise ValueError(f"unsupported source type {record.source_type}")
    native_block, base_block, residual_block, _, _ = TYPE_FORMATS[record.source_type]
    if record.native_block_bytes != native_block:
        raise ValueError("record native block size does not match source type")
    if record.base_block_bytes != base_block:
        raise ValueError("record base block size does not match source type")
    if record.residual_block_bytes != residual_block:
        raise ValueError("record residual block size does not match source type")
    if record.kind not in KIND_NAME:
        raise ValueError(f"invalid record kind {record.kind}")
    if record.reserved != 0:
        raise ValueError("record reserved field must be zero")
    if record.ncols <= 0 or record.nrows <= 0 or record.nexperts <= 0:
        raise ValueError("record dimensions must be positive")
    if record.ncols % 256:
        raise ValueError("record ncols must be divisible by 256")
    blocks_per_expert = (record.ncols // 256) * record.nrows
    return blocks_per_expert * record.nexperts * native_block, blocks_per_expert * record.nexperts * residual_block, blocks_per_expert * residual_block


def records_by_layer(records: list[Record]) -> dict[int, dict[int, Record]]:
    layers: dict[int, dict[int, Record]] = {}
    for record in records:
        parts = layers.setdefault(record.layer, {})
        if record.kind in parts:
            raise ValueError(f"duplicate layer/kind record layer={record.layer} kind={record.kind_name}")
        parts[record.kind] = record
    return layers


def read_sidecar(path: Path) -> tuple[Header, list[Record]]:
    with path.open("rb") as handle:
        raw_header = handle.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            raise ValueError("truncated sidecar header")
        fields = HEADER_STRUCT.unpack(raw_header)
        record_count = fields[8]
        record_size = fields[9]
        if record_size != RECORD_SIZE or record_count > 4096:
            raise ValueError("invalid sidecar record table geometry")
        record_table = handle.read(record_count * record_size)
        header = unpack_header(raw_header, record_table)
        records = [
            unpack_record(record_table[offset : offset + RECORD_SIZE])
            for offset in range(0, len(record_table), RECORD_SIZE)
        ]
    return header, records


def validate_records(header: Header, records: list[Record], sidecar_size: int) -> None:
    if len(records) != header.record_count:
        raise ValueError("record count mismatch")
    records_end = HEADER_BYTES + header.record_count * RECORD_SIZE
    payload_end = records_end + header.payload_bytes
    if payload_end != sidecar_size:
        raise ValueError("sidecar size does not match header payload size")
    ranges: list[tuple[int, int]] = []
    for record in records:
        expected_source, expected_residual, per_expert_residual = _expected_bytes(record)
        if record.source_bytes != expected_source:
            raise ValueError("record source byte count does not match geometry")
        if record.residual_bytes != expected_residual:
            raise ValueError("record residual byte count does not match geometry")
        if record.expert_stride <= 0 or record.residual_expert_bytes <= 0:
            raise ValueError("record expert residual layout must be positive")
        if record.expert_stride != record.residual_expert_bytes:
            raise ValueError("record expert_stride must match residual_expert_bytes in v1")
        if record.kind_offset_within_expert + per_expert_residual > record.residual_expert_bytes:
            raise ValueError("record kind residual exceeds expert layout bounds")
        if record.residual_offset < records_end:
            raise ValueError("record residual range exceeds sidecar payload bounds")
        if record.source_offset + record.source_bytes > header.source_size:
            raise ValueError("record source range exceeds declared source size")
        for expert in range(record.nexperts):
            start = record.residual_offset + expert * record.expert_stride
            end = start + per_expert_residual
            if start < records_end or end > payload_end:
                raise ValueError("record residual range exceeds sidecar payload bounds")
            ranges.append((start, end))
    for (_, prev_end), (start, _) in zip(sorted(ranges), sorted(ranges)[1:]):
        if start < prev_end:
            raise ValueError("record residual ranges overlap")
    layers = records_by_layer(records)
    expected_layer_start = records_end
    for layer, parts in sorted(layers.items()):
        if set(parts) != set(KIND_NAME):
            raise ValueError(f"layer {layer} does not have gate/up/down residual records")
        ordered = [parts[KIND_ENUM[name]] for name in ("gate", "up", "down")]
        nexperts = ordered[0].nexperts
        stride = ordered[0].residual_expert_bytes
        layer_start = min(record.residual_offset - record.kind_offset_within_expert for record in ordered)
        if layer_start != expected_layer_start:
            raise ValueError(f"layer {layer} does not start at the expected layer-major payload offset")
        cursor = 0
        for record in ordered:
            _, _, per_expert_residual = _expected_bytes(record)
            if record.nexperts != nexperts:
                raise ValueError(f"layer {layer} records disagree on expert count")
            if record.expert_stride != stride or record.residual_expert_bytes != stride:
                raise ValueError(f"layer {layer} records disagree on residual expert stride")
            if record.kind_offset_within_expert != cursor:
                raise ValueError(f"layer {layer} kind offsets are not gate/up/down contiguous")
            if record.residual_offset != layer_start + cursor:
                raise ValueError(f"layer {layer} residual offsets do not match interleaved layout")
            cursor += per_expert_residual
        if cursor != stride:
            raise ValueError(f"layer {layer} residual expert bytes do not match kind geometry")
        expected_layer_start += stride * nexperts
    if expected_layer_start != payload_end:
        raise ValueError("layer-major payload layout does not consume the declared payload exactly")


def validate_reconstruction(sidecar_path: Path, source_path: Path, records: list[Record]) -> None:
    with source_path.open("rb") as source, sidecar_path.open("rb") as sidecar:
        for record in records:
            native_block, _, residual_block, _, _ = TYPE_FORMATS[record.source_type]
            row_blocks = record.ncols // 256
            source_row_bytes = row_blocks * native_block
            for expert in range(record.nexperts):
                expert_source = record.source_offset + expert * record.nrows * source_row_bytes
                residual_at = record.residual_offset + expert * record.expert_stride
                expert_source_bytes = record.nrows * source_row_bytes
                expert_residual_bytes = record.nrows * row_blocks * residual_block
                source.seek(expert_source)
                native = source.read(expert_source_bytes)
                sidecar.seek(residual_at)
                residual = sidecar.read(expert_residual_bytes)
                if len(native) != expert_source_bytes:
                    raise ValueError("unexpected EOF while reading source expert")
                if len(residual) != expert_residual_bytes:
                    raise ValueError("unexpected EOF while reading residual expert part")
                if extract_residuals(native, record.source_type) != residual:
                    raise ValueError(
                        f"exact reconstruction failed for layer={record.layer} "
                        f"kind={record.kind_name} expert={expert}"
                    )
                residual_at += expert_residual_bytes
                expected = (
                    record.residual_offset
                    + expert * record.expert_stride
                    + (record.residual_bytes // record.nexperts)
                )
                if residual_at != expected:
                    raise ValueError("record residual traversal ended at the wrong offset")


def validate_sidecar(
    sidecar_path: Path,
    source_path: Path,
    reconstruct: bool = True,
    trusted_source_sha256: bytes | None = None,
) -> tuple[Header, list[Record]]:
    header, records = read_sidecar(sidecar_path)
    source_size = source_path.stat().st_size
    if source_size != header.source_size:
        raise ValueError("source size does not match sidecar header")
    if trusted_source_sha256 is not None and len(trusted_source_sha256) != 32:
        raise ValueError("trusted source SHA256 must contain exactly 32 bytes")
    observed_source_sha = (
        trusted_source_sha256
        if trusted_source_sha256 is not None
        else sha256_file(source_path)
    )
    if observed_source_sha != header.source_sha256:
        raise ValueError("source SHA256 does not match sidecar header")
    sidecar_size = sidecar_path.stat().st_size
    validate_records(header, records, sidecar_size)
    payload_start = HEADER_BYTES + header.record_count * RECORD_SIZE
    if sha256_region(sidecar_path, payload_start, header.payload_bytes) != header.payload_sha256:
        raise ValueError("payload SHA256 does not match sidecar header")
    if reconstruct:
        validate_reconstruction(sidecar_path, source_path, records)
    return header, records


def write_sidecar(
    source_path: Path,
    sidecar_path: Path,
    records: list[Record],
    trusted_source_sha256: bytes | None = None,
) -> Header:
    source_path = source_path.resolve()
    sidecar_path = sidecar_path.resolve()
    if source_path == sidecar_path:
        raise ValueError("source and sidecar paths must differ")
    partial_path = sidecar_path.with_name(sidecar_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    source_size = source_path.stat().st_size
    if trusted_source_sha256 is not None and len(trusted_source_sha256) != 32:
        raise ValueError("trusted source SHA256 must contain exactly 32 bytes")
    source_sha = (
        trusted_source_sha256
        if trusted_source_sha256 is not None
        else sha256_file(source_path)
    )
    record_table = b"".join(pack_record(record) for record in records)
    placeholder = Header(
        source_size,
        source_sha,
        sum(record.residual_bytes for record in records),
        b"\0" * 32,
        len(records),
    )
    payload_digest = hashlib.sha256()
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb") as source, partial_path.open("xb") as sidecar:
            sidecar.write(pack_header(placeholder, record_table))
            sidecar.write(record_table)
            for _, parts in sorted(records_by_layer(records).items()):
                ordered = [parts[KIND_ENUM[name]] for name in ("gate", "up", "down")]
                for expert in range(ordered[0].nexperts):
                    for record in ordered:
                        expected_offset = record.residual_offset + expert * record.expert_stride
                        if sidecar.tell() != expected_offset:
                            raise ValueError("record layout does not match layer-major expert-major payload order")
                        native_block, _, residual_block, _, _ = TYPE_FORMATS[record.source_type]
                        row_blocks = record.ncols // 256
                        row_bytes = row_blocks * native_block
                        expert_offset = record.source_offset + expert * record.nrows * row_bytes
                        expert_bytes = record.nrows * row_bytes
                        source.seek(expert_offset)
                        native = source.read(expert_bytes)
                        if len(native) != expert_bytes:
                            raise ValueError("unexpected EOF while packing source expert")
                        residual = extract_residuals(native, record.source_type)
                        expected_residual = record.nrows * row_blocks * residual_block
                        if len(residual) != expected_residual:
                            raise ValueError("bulk extraction returned unexpected residual size")
                        sidecar.write(residual)
                        payload_digest.update(residual)
            header = Header(source_size, source_sha, placeholder.payload_bytes, payload_digest.digest(), len(records))
            sidecar.seek(0)
            sidecar.write(pack_header(header, record_table))
            sidecar.flush()
        partial_path.replace(sidecar_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise
    return header


def _records_from_specs(specs: list[tuple[int, str, int, int, int, int, int]]) -> list[Record]:
    payload_offset = HEADER_BYTES + len(specs) * RECORD_SIZE
    records = []
    layer_order = sorted(dict.fromkeys(spec[0] for spec in specs))
    for layer in layer_order:
        by_kind: dict[str, tuple[int, str, int, int, int, int, int]] = {}
        for spec in specs:
            if spec[0] != layer:
                continue
            if spec[1] in by_kind:
                raise ValueError(f"duplicate layer/kind spec layer={layer} kind={spec[1]}")
            by_kind[spec[1]] = spec
        if set(by_kind) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer} must include gate/up/down records")
        per_kind: dict[str, int] = {}
        for kind_name, spec in by_kind.items():
            _, _, source_type, ncols, nrows, nexperts, _ = spec
            residual_block = TYPE_FORMATS[source_type][2]
            per_kind[kind_name] = (ncols // 256) * nrows * residual_block
        residual_expert_bytes = sum(per_kind[name] for name in ("gate", "up", "down"))
        nexperts_values = {spec[5] for spec in by_kind.values()}
        if len(nexperts_values) != 1:
            raise ValueError(f"layer {layer} records disagree on expert count")
        kind_offset = 0
        for kind_name in ("gate", "up", "down"):
            _, _, source_type, ncols, nrows, nexperts, source_offset = by_kind[kind_name]
            native_block, base_block, residual_block, _, _ = TYPE_FORMATS[source_type]
            blocks_per_expert = (ncols // 256) * nrows
            source_bytes = blocks_per_expert * nexperts * native_block
            residual_bytes = blocks_per_expert * nexperts * residual_block
            records.append(
                Record(
                    layer,
                    KIND_ENUM[kind_name],
                    source_type,
                    ncols,
                    nrows,
                    nexperts,
                    source_offset,
                    source_bytes,
                    payload_offset + kind_offset,
                    residual_bytes,
                    residual_expert_bytes,
                    kind_offset,
                    residual_expert_bytes,
                    native_block,
                    base_block,
                    residual_block,
                )
            )
            kind_offset += per_kind[kind_name]
        payload_offset += residual_expert_bytes * next(iter(nexperts_values))
    return records


def write_synthetic_fixture(source_path: Path, sidecar_path: Path) -> tuple[Header, list[Record]]:
    specs: list[tuple[int, str, int, int, int, int, int]] = []
    source_offset = 0
    payload = bytearray()
    for layer, kind_name, source_type, ncols, nrows, nexperts in (
        (0, "gate", 16, 512, 2, 2),
        (0, "up", 10, 256, 3, 2),
        (0, "down", 16, 256, 1, 2),
    ):
        native_block = TYPE_FORMATS[source_type][0]
        blocks = (ncols // 256) * nrows * nexperts
        specs.append((layer, kind_name, source_type, ncols, nrows, nexperts, source_offset))
        for index in range(blocks):
            seed = len(payload) + index * 17 + layer * 31 + source_type
            payload.extend(bytes((seed + i * 13) % 256 for i in range(native_block)))
        source_offset += blocks * native_block
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(bytes(payload))
    records = _records_from_specs(specs)
    header = write_sidecar(source_path, sidecar_path, records)
    return header, records


def records_from_gguf(model_path: Path, layer_text: str | None) -> list[Record]:
    gguf = gguf_inspect_ds4.parse_gguf(str(model_path))
    layers = gguf_inspect_ds4.discover_expert_layers(gguf)
    wanted_layers = gguf_inspect_ds4.parse_int_list(layer_text) if layer_text else sorted(layers)
    specs: list[tuple[int, str, int, int, int, int, int]] = []
    for layer in wanted_layers:
        if layer not in layers:
            raise ValueError(f"requested layer {layer} is not present")
        for kind_name in ("gate", "up", "down"):
            tensor = layers[layer][kind_name]
            if tensor.type_id not in TYPE_FORMATS:
                raise ValueError(f"unsupported tensor type {tensor.type_id} in {tensor.name}")
            if len(tensor.dims) != 3:
                raise ValueError(f"{tensor.name} is not a routed expert tensor")
            specs.append(
                (
                    layer,
                    kind_name,
                    tensor.type_id,
                    tensor.dims[0],
                    tensor.dims[1],
                    tensor.dims[2],
                    tensor.abs_offset,
                )
            )
    return _records_from_specs(specs)


def write_manifest(path: Path, source_path: Path, sidecar_path: Path, header: Header, records: list[Record]) -> None:
    receipt = {
        "schema": "ds4_nested_residual_sidecar_receipt_v1",
        "source": str(source_path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
        "source_size": header.source_size,
        "source_sha256": header.source_sha256.hex(),
        "payload_bytes": header.payload_bytes,
        "payload_sha256": header.payload_sha256.hex(),
        "record_count": header.record_count,
        "records": [
            {
                "layer": record.layer,
                "kind": record.kind_name,
                "source_type": record.source_type,
                "dims": [record.ncols, record.nrows, record.nexperts],
                "source_offset": record.source_offset,
                "source_bytes": record.source_bytes,
                "residual_offset": record.residual_offset,
                "residual_bytes": record.residual_bytes,
                "expert_stride": record.expert_stride,
                "kind_offset_within_expert": record.kind_offset_within_expert,
                "residual_expert_bytes": record.residual_expert_bytes,
            }
            for record in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fixture = sub.add_parser("fixture", help="write a tiny synthetic source plus sidecar")
    fixture.add_argument("--source", required=True)
    fixture.add_argument("--sidecar", required=True)
    fixture.add_argument("--manifest")
    pack = sub.add_parser("pack-gguf", help="pack residuals from DS4 GGUF routed expert tensors")
    pack.add_argument("model")
    pack.add_argument("--sidecar", required=True)
    pack.add_argument("--layers", help="Layer list/ranges, e.g. 3,10,20-22")
    pack.add_argument("--manifest")
    pack.add_argument(
        "--source-sha256",
        help="reuse a previously verified 64-hex source SHA-256",
    )
    validate = sub.add_parser("validate", help="fail-closed sidecar validation")
    validate.add_argument("--source", required=True)
    validate.add_argument("--sidecar", required=True)
    validate.add_argument("--no-reconstruct", action="store_true")
    validate.add_argument(
        "--source-sha256",
        help="reuse a previously verified 64-hex source SHA-256",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "fixture":
        header, records = write_synthetic_fixture(Path(args.source), Path(args.sidecar))
        if args.manifest:
            write_manifest(Path(args.manifest), Path(args.source), Path(args.sidecar), header, records)
        print(f"wrote fixture sidecar records={len(records)} payload_bytes={header.payload_bytes}")
        return 0
    if args.cmd == "pack-gguf":
        source_path = Path(args.model)
        records = records_from_gguf(source_path, args.layers)
        trusted_source_sha = (
            bytes.fromhex(args.source_sha256) if args.source_sha256 else None
        )
        header = write_sidecar(
            source_path,
            Path(args.sidecar),
            records,
            trusted_source_sha256=trusted_source_sha,
        )
        if args.manifest:
            write_manifest(Path(args.manifest), source_path, Path(args.sidecar), header, records)
        print(f"wrote sidecar records={len(records)} payload_bytes={header.payload_bytes}")
        return 0
    if args.cmd == "validate":
        trusted_source_sha = (
            bytes.fromhex(args.source_sha256) if args.source_sha256 else None
        )
        header, records = validate_sidecar(
            Path(args.sidecar),
            Path(args.source),
            reconstruct=not args.no_reconstruct,
            trusted_source_sha256=trusted_source_sha,
        )
        print(f"valid sidecar records={len(records)} payload_bytes={header.payload_bytes}")
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
