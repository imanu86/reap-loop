from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import struct
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ds4_nested_residual_sidecar.py"
SPEC = importlib.util.spec_from_file_location("ds4_nested_residual_sidecar", SCRIPT)
assert SPEC and SPEC.loader
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)


def _repair_header_crc(data: bytearray) -> None:
    fields = sidecar.HEADER_STRUCT.unpack(bytes(data[: sidecar.HEADER_BYTES]))
    count = fields[8]
    record_size = fields[9]
    table_end = sidecar.HEADER_BYTES + count * record_size
    record_table = bytes(data[sidecar.HEADER_BYTES : table_end])
    header = sidecar.Header(
        fields[3], fields[4], fields[5], fields[6], count,
        record_size, fields[2], 0,
    )
    data[: sidecar.HEADER_BYTES] = sidecar.pack_header(header, record_table)


def test_bulk_residual_extraction_matches_reference() -> None:
    rng = random.Random(20260718)
    for source_type, blocks in ((16, 37), (10, 29)):
        native_block = sidecar.TYPE_FORMATS[source_type][0]
        split_block = sidecar.TYPE_FORMATS[source_type][3]
        native = bytes(rng.randrange(256) for _ in range(native_block * blocks))
        expected = b"".join(
            split_block(native[offset : offset + native_block])[1]
            for offset in range(0, len(native), native_block)
        )
        assert sidecar.extract_residuals(native, source_type) == expected


def test_fixture_sidecar_validates_and_reconstructs(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    header, records = sidecar.write_synthetic_fixture(source, residual)

    assert header.record_count == 3
    assert header.payload_bytes == sum(record.residual_bytes for record in records)
    checked_header, checked_records = sidecar.validate_sidecar(residual, source)
    assert checked_header.payload_sha256 == header.payload_sha256
    assert [record.kind_name for record in checked_records] == ["gate", "up", "down"]


def test_fixture_payload_is_layer_major_expert_major(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    _, records = sidecar.write_synthetic_fixture(source, residual)
    gate, up, down = records

    assert gate.kind_offset_within_expert == 0
    assert up.kind_offset_within_expert == gate.residual_bytes // gate.nexperts
    assert down.kind_offset_within_expert == up.kind_offset_within_expert + up.residual_bytes // up.nexperts
    assert gate.expert_stride == up.expert_stride == down.expert_stride
    assert gate.residual_expert_bytes == gate.expert_stride
    assert up.residual_offset == gate.residual_offset + up.kind_offset_within_expert
    assert down.residual_offset == gate.residual_offset + down.kind_offset_within_expert
    first_expert_end = down.residual_offset + down.residual_bytes // down.nexperts
    assert gate.residual_offset + gate.expert_stride == first_expert_end


def test_sidecar_header_crc_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(residual.read_bytes())
    data[32] ^= 0x01
    residual.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "CRC" in str(exc)
    else:
        raise AssertionError("corrupt header CRC was accepted")


def test_sidecar_payload_sha_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(residual.read_bytes())
    data[-1] ^= 0x80
    residual.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "payload SHA256" in str(exc)
    else:
        raise AssertionError("corrupt payload was accepted")


def test_sidecar_record_bounds_are_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(residual.read_bytes())
    first_record = sidecar.HEADER_BYTES
    residual_offset_field = first_record + struct.calcsize("<IIIQQQQQ")
    struct.pack_into("<Q", data, residual_offset_field, residual.stat().st_size + 128)
    _repair_header_crc(data)
    residual.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "bounds" in str(exc)
    else:
        raise AssertionError("out-of-bounds record was accepted")


def test_sidecar_interleaved_layout_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(residual.read_bytes())
    second_record = sidecar.HEADER_BYTES + sidecar.RECORD_SIZE
    kind_offset_field = second_record + struct.calcsize("<IIIQQQQQQQQ")
    struct.pack_into("<Q", data, kind_offset_field, 1)
    _repair_header_crc(data)
    residual.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("bad interleaved kind offset was accepted")


def test_sidecar_record_table_crc_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(residual.read_bytes())
    data[sidecar.HEADER_BYTES + 7] ^= 0x40
    residual.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "record-table CRC" in str(exc)
    else:
        raise AssertionError("corrupt record table CRC was accepted")


def test_source_identity_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    sidecar.write_synthetic_fixture(source, residual)
    data = bytearray(source.read_bytes())
    data[0] ^= 0x01
    source.write_bytes(data)

    try:
        sidecar.validate_sidecar(residual, source)
    except ValueError as exc:
        assert "source SHA256" in str(exc)
    else:
        raise AssertionError("wrong source identity was accepted")


def test_writer_refuses_source_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "fixture.gguf.bin"
    source.write_bytes(b"source")
    try:
        sidecar.write_sidecar(source, source, [])
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("writer accepted source as sidecar output")


def test_trusted_source_sha_avoids_rehash_and_remains_bound(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    source.write_bytes(bytes(range(64)))
    trusted = sidecar.sha256_file(source)
    records = sidecar._records_from_specs(
        [(0, "gate", 16, 256, 1, 1, 0),
         (0, "up", 16, 256, 1, 1, 66),
         (0, "down", 10, 256, 1, 1, 132)]
    )
    source.write_bytes(source.read_bytes() + bytes(84 + 68))
    trusted = sidecar.sha256_file(source)
    monkeypatch.setattr(
        sidecar, "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected rehash")),
    )

    header = sidecar.write_sidecar(
        source, residual, records, trusted_source_sha256=trusted
    )
    assert header.source_sha256 == trusted
    checked, _ = sidecar.validate_sidecar(
        residual,
        source,
        reconstruct=True,
        trusted_source_sha256=trusted,
    )
    assert checked.source_sha256 == trusted


def test_cli_fixture_and_validate(tmp_path: Path, capsys) -> None:
    source = tmp_path / "fixture.gguf.bin"
    residual = tmp_path / "fixture.dnr"
    assert sidecar.main(["fixture", "--source", str(source), "--sidecar", str(residual)]) == 0
    assert sidecar.main(["validate", "--source", str(source), "--sidecar", str(residual)]) == 0
    output = capsys.readouterr().out
    assert "wrote fixture sidecar" in output
    assert "valid sidecar" in output
