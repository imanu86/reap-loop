#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("build_mean_weight_mask.py")
SPEC = importlib.util.spec_from_file_location("build_mean_weight_mask", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload_for_layers(entries):
    return {
        "learned": {
            "ranking": {
                str(layer): list(entries)
                for layer in range(MODULE.LAYER_MIN, MODULE.LAYER_MAX + 1)
            }
        }
    }


class MeanWeightMaskTests(unittest.TestCase):
    def test_mean_weight_excludes_frequency(self):
        payload = payload_for_layers(
            [
                {"expert": 5, "mass": 8.0, "calls": 100},
                {"expert": 7, "mass": 1.0, "calls": 2},
            ]
        )
        lines, manifest = MODULE.build(payload, keep=1)
        self.assertNotIn("3 7", lines)
        self.assertIn("3 5", lines)
        self.assertEqual(manifest["layers"][0]["ranking"][0]["expert"], 7)
        self.assertEqual(manifest["policy"]["frequency_component"], "excluded")

    def test_ties_use_expert_id_ascending(self):
        payload = payload_for_layers(
            [
                {"expert": 9, "mass": 1.0, "calls": 2},
                {"expert": 4, "mass": 2.0, "calls": 4},
            ]
        )
        lines, _manifest = MODULE.build(payload, keep=1)
        self.assertNotIn("3 4", lines)
        self.assertIn("3 9", lines)

    def test_missing_experts_score_zero_and_line_count_is_exact(self):
        payload = payload_for_layers([{"expert": 3, "mass": 1.0, "calls": 1}])
        lines, manifest = MODULE.build(payload, keep=154)
        self.assertEqual(len(lines), (256 - 154) * 40)
        self.assertEqual(manifest["blocked_lines"], len(lines))

    def test_cli_writes_mask_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            learn_json = root / "learn.json"
            mask = root / "mask.txt"
            manifest = root / "manifest.json"
            import json

            learn_json.write_text(json.dumps(payload_for_layers([{"expert": 1, "mass": 1.0, "calls": 1}])), encoding="utf-8")
            rc = MODULE.main(
                [
                    "--learn-json",
                    str(learn_json),
                    "--keep",
                    "1",
                    "--out",
                    str(mask),
                    "--manifest-out",
                    str(manifest),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(mask.exists())
            self.assertTrue(manifest.exists())
            self.assertEqual(len(mask.read_text(encoding="utf-8").splitlines()), 255 * 40)


if __name__ == "__main__":
    unittest.main()
