#!/usr/bin/env python3
import gzip
import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("archive_routing_trace.py")
SPEC = importlib.util.spec_from_file_location("archive_routing_trace", SCRIPT)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class ArchiveRoutingTraceTest(unittest.TestCase):
    def test_archives_deterministic_gzip_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src = root / "routing.csv"
            out = root / "routing.csv.gz"
            manifest = root / "routing.manifest.json"
            raw = b"route,token,bytes\r\nalpha,1,x\r\nbeta,2,y\r\n"
            src.write_bytes(raw)

            rc = ARCHIVE.main(
                ["--input", str(src), "--output", str(out), "--manifest", str(manifest)]
            )

            self.assertEqual(rc, 0)
            self.assertEqual(gzip.decompress(out.read_bytes()), raw)

            first_gzip = out.read_bytes()
            first = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(first["input"]["size_bytes"], len(raw))
            self.assertEqual(first["input"]["sha256"], ARCHIVE.hashlib.sha256(raw).hexdigest())
            self.assertEqual(first["input"]["line_count"], 3)
            self.assertEqual(first["input"]["data_row_count"], 2)
            self.assertEqual(first["output"]["size_bytes"], len(first_gzip))
            self.assertEqual(
                first["output"]["sha256"], ARCHIVE.hashlib.sha256(first_gzip).hexdigest()
            )
            self.assertEqual(first["gzip"]["mtime"], 0)
            self.assertEqual(first["gzip"]["filename"], "")
            self.assertTrue(first["roundtrip"]["verified"])
            self.assertEqual(
                first["reconstruct"]["method"], "gzip_decompress_output_bytes"
            )

            out.unlink()
            manifest.unlink()
            rc = ARCHIVE.main(
                ["--input", str(src), "--output", str(out), "--manifest", str(manifest)]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(out.read_bytes(), first_gzip)

    def test_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src = root / "empty.csv"
            src.write_bytes(b"")

            with self.assertRaises(SystemExit) as ctx:
                ARCHIVE.main(
                    [
                        "--input",
                        str(src),
                        "--output",
                        str(root / "out.csv.gz"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_header_only_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src = root / "header_only.csv"
            src.write_bytes(b"a,b\n")

            with self.assertRaises(SystemExit) as ctx:
                ARCHIVE.main(
                    [
                        "--input",
                        str(src),
                        "--output",
                        str(root / "out.csv.gz"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_output_input_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src = root / "routing.csv"
            src.write_bytes(b"a,b\n1,2\n")

            with self.assertRaises(SystemExit) as ctx:
                ARCHIVE.main(
                    [
                        "--input",
                        str(src),
                        "--output",
                        str(src),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_existing_output_or_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            src = root / "routing.csv"
            out = root / "routing.csv.gz"
            manifest = root / "manifest.json"
            src.write_bytes(b"a,b\n1,2\n")
            out.write_bytes(b"occupied")

            with self.assertRaises(SystemExit) as ctx:
                ARCHIVE.main(
                    ["--input", str(src), "--output", str(out), "--manifest", str(manifest)]
                )
            self.assertEqual(ctx.exception.code, 2)

            out.unlink()
            manifest.write_bytes(b"occupied")
            with self.assertRaises(SystemExit) as ctx:
                ARCHIVE.main(
                    ["--input", str(src), "--output", str(out), "--manifest", str(manifest)]
                )
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
