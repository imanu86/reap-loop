#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("merge_full_decode_traces.py")
SPEC = importlib.util.spec_from_file_location("merge_full_decode_traces", SCRIPT)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class MergeFullDecodeTracesTest(unittest.TestCase):
    def test_merge_preserves_body_order_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            one = root / "one.csv"
            two = root / "two.csv"
            out = root / "merged.csv"
            manifest = root / "manifest.json"

            one.write_bytes(b"route,token,bytes\r\nalpha,1,x\r\nbeta,2,y\r\n")
            two.write_bytes(b"route,token,bytes\r\ngamma,3,z\r\n")

            rc = MERGE.main(
                [
                    str(one),
                    str(two),
                    "--output",
                    str(out),
                    "--manifest",
                    str(manifest),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertEqual(
                out.read_bytes(),
                b"route,token,bytes\r\nalpha,1,x\r\nbeta,2,y\r\ngamma,3,z\r\n",
            )

            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["policy"]["dedupe"], False)
            self.assertEqual(data["policy"]["cutoff"], False)
            self.assertEqual(data["output"]["data_line_count"], 3)
            self.assertEqual([item["data_line_count"] for item in data["inputs"]], [2, 1])
            self.assertEqual(data["output"]["sha256"], MERGE.sha256_bytes(out.read_bytes()))

    def test_rejects_header_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            one = root / "one.csv"
            two = root / "two.csv"
            one.write_bytes(b"a,b\n1,2\n")
            two.write_bytes(b"a,c\n1,2\n")

            with self.assertRaises(SystemExit) as ctx:
                MERGE.main(
                    [
                        str(one),
                        str(two),
                        "--output",
                        str(root / "out.csv"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_empty_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            empty = root / "empty.csv"
            empty.write_bytes(b"")

            with self.assertRaises(SystemExit) as ctx:
                MERGE.main(
                    [
                        str(empty),
                        "--output",
                        str(root / "out.csv"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_header_only_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            header_only = root / "header_only.csv"
            header_only.write_bytes(b"a,b\n")

            with self.assertRaises(SystemExit) as ctx:
                MERGE.main(
                    [
                        str(header_only),
                        "--output",
                        str(root / "out.csv"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_body_without_trailing_lf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            unterminated = root / "unterminated.csv"
            unterminated.write_bytes(b"a,b\n1,2")

            with self.assertRaises(SystemExit) as ctx:
                MERGE.main(
                    [
                        str(unterminated),
                        "--output",
                        str(root / "out.csv"),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_output_input_collision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            one = root / "one.csv"
            one.write_bytes(b"a,b\n1,2\n")

            with self.assertRaises(SystemExit) as ctx:
                MERGE.main(
                    [
                        str(one),
                        "--output",
                        str(one),
                        "--manifest",
                        str(root / "manifest.json"),
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
