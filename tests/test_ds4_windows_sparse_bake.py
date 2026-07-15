from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ds4_windows_sparse_bake.py"
SPEC = importlib.util.spec_from_file_location("ds4_windows_sparse_bake", SCRIPT)
assert SPEC and SPEC.loader
ds4_windows_sparse_bake = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ds4_windows_sparse_bake
SPEC.loader.exec_module(ds4_windows_sparse_bake)


class BinaryStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class BrokenStdout:
    class Buffer:
        def write(self, data: bytes) -> int:
            raise BrokenPipeError("closed")

        def flush(self) -> None:
            pass

    def __init__(self) -> None:
        self.buffer = self.Buffer()


def fake_plan(source: Path) -> dict:
    return {
        "format": "ds4-windows-sparse-bake",
        "version": 1,
        "source_model_name": source.name,
        "source_model_size": source.stat().st_size,
        "source_model_sha256": None,
        "mask_name": "fake.mask",
        "mask_sha256": "0" * 64,
        "tensor_count": 2,
        "routed_tensor_count": 1,
        "full_routed_layers": [],
        "selected_experts_by_layer": {"0": list(range(6))},
        "retained_count_by_layer": {"0": 6},
        "merge_gap": 0,
        "payload_bytes": 11,
        "payload_gib": 11 / (1 << 30),
        "logical_savings_bytes": source.stat().st_size - 11,
        "logical_savings_gib": (source.stat().st_size - 11) / (1 << 30),
        "extents": [[0, 5], [10, 6]],
        "routed_tensors": [
            {
                "name": "blk.0.ffn_gate_exps.weight",
                "layer": 0,
                "kind": "gate",
                "offset": 10,
                "bytes": 6,
                "slice_bytes": 1,
                "tensor_type": 0,
                "selected_count": 6,
            }
        ],
    }


def expected_pack_bytes(source_bytes: bytes, plan: dict) -> tuple[bytes, dict[str, str | int]]:
    payload = b"".join(source_bytes[offset:offset + length] for offset, length in plan["extents"])
    payload_digest = hashlib.sha256(payload)
    manifest_plan = dict(plan)
    manifest_plan["payload_sha256"] = payload_digest.hexdigest()
    manifest = json.dumps(manifest_plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest)
    footer = ds4_windows_sparse_bake.FOOTER.pack(
        ds4_windows_sparse_bake.PACK_END_MAGIC,
        len(manifest),
        payload_digest.digest(),
        manifest_digest.digest(),
    )
    pack = ds4_windows_sparse_bake.PACK_MAGIC + payload + manifest + footer
    status = {
        "total_bytes": len(pack),
        "payload_sha256": payload_digest.hexdigest(),
        "manifest_sha256": manifest_digest.hexdigest(),
        "full_pack_sha256": hashlib.sha256(pack).hexdigest(),
    }
    return pack, status


class Ds4WindowsSparseBakeTests(unittest.TestCase):
    def test_pack_file_and_stream_are_byte_identical_with_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source_bytes = bytes(range(32))
            source.write_bytes(source_bytes)
            plan = fake_plan(source)
            pack_path = root / "out.pack"

            with contextlib.redirect_stdout(io.StringIO()):
                file_result = ds4_windows_sparse_bake.write_pack(source, pack_path, plan)
            stream = io.BytesIO()
            stream_result = ds4_windows_sparse_bake.write_pack_stream(source, stream, plan)

            expected_pack, expected_status = expected_pack_bytes(source_bytes, plan)
            self.assertEqual(pack_path.read_bytes(), expected_pack)
            self.assertEqual(stream.getvalue(), expected_pack)
            self.assertEqual(file_result, stream_result)
            self.assertEqual(file_result.total_bytes, expected_status["total_bytes"])
            self.assertEqual(file_result.payload_sha256, expected_status["payload_sha256"])
            self.assertEqual(file_result.manifest_sha256, expected_status["manifest_sha256"])
            self.assertEqual(file_result.full_pack_sha256, expected_status["full_pack_sha256"])

    def test_pack_stream_cli_writes_only_pack_bytes_to_stdout_and_status_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            mask = root / "mask.txt"
            status = root / "status.json"
            source_bytes = bytes(range(32))
            source.write_bytes(source_bytes)
            mask.write_text("0 6\n", encoding="utf-8")
            plan = fake_plan(source)
            stdout = BinaryStdout()

            argv = [
                "ds4_windows_sparse_bake.py",
                "pack-stream",
                "--model",
                str(source),
                "--mask",
                str(mask),
                "--status-out",
                str(status),
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(sys, "stdout", stdout), \
                    mock.patch.object(ds4_windows_sparse_bake, "build_plan", return_value=plan), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(ds4_windows_sparse_bake.main(), 0)

            expected_pack, expected_status = expected_pack_bytes(source_bytes, plan)
            self.assertEqual(stdout.buffer.getvalue(), expected_pack)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(status.read_text(encoding="utf-8")), expected_status)

    def test_pack_stream_broken_pipe_does_not_replace_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            mask = root / "mask.txt"
            status = root / "status.json"
            source.write_bytes(bytes(range(32)))
            mask.write_text("0 6\n", encoding="utf-8")
            status.write_text('{"old": true}\n', encoding="utf-8")
            plan = fake_plan(source)

            argv = [
                "ds4_windows_sparse_bake.py",
                "pack-stream",
                "--model",
                str(source),
                "--mask",
                str(mask),
                "--status-out",
                str(status),
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(sys, "stdout", BrokenStdout()), \
                    mock.patch.object(ds4_windows_sparse_bake, "build_plan", return_value=plan):
                with self.assertRaises(SystemExit):
                    ds4_windows_sparse_bake.main()

            self.assertEqual(json.loads(status.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(root.glob(f".{status.name}.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
