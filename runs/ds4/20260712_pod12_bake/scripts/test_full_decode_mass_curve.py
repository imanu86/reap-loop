#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("full_decode_mass_curve.py")
SPEC = importlib.util.spec_from_file_location("full_decode_mass_curve", SCRIPT)
CURVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURVE)


def write_trace(path, rows, *, weights=True, header=True):
    lines = []
    if header:
        head = ["pos", "layer", "n", "e0", "e1", "e2"]
        if weights:
            head += ["w0", "w1", "w2"]
        lines.append(",".join(head))
    for pos, (layer, pairs) in enumerate(rows):
        experts = [str(expert) for expert, _weight in pairs]
        weights_s = [str(weight) for _expert, weight in pairs]
        experts += ["-1"] * (3 - len(experts))
        weights_s += ["0"] * (3 - len(weights_s))
        cols = [str(pos), str(layer), str(len(pairs))] + experts
        if weights:
            cols += weights_s
        lines.append(",".join(cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FullDecodeMassCurveTest(unittest.TestCase):
    def test_learn_self_curve_exact_values_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            trace = root / "route.csv"
            json_out = root / "curve.json"
            csv_out = root / "curve.csv"

            write_trace(
                trace,
                [
                    (3, [(1, 0.4), (2, 0.6)]),
                    (3, [(2, 0.7), (1, 0.3)]),
                    (4, [(9, 0.5), (8, 0.5)]),
                    (4, [(7, 0.9), (8, 0.1)]),
                ],
            )

            rc = CURVE.main(
                [
                    "--trace",
                    str(trace),
                    "--keeps",
                    "1,2",
                    "--json-out",
                    str(json_out),
                    "--csv-out",
                    str(csv_out),
                ]
            )

            self.assertEqual(rc, 0)
            obj = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(obj["evaluation_label"], "learn_self_coverage")
            self.assertIn("not held-out", obj["note"])
            self.assertEqual(obj["learned"]["ranking"]["3"][0]["expert"], 2)
            self.assertEqual(obj["learned"]["ranking"]["3"][1]["expert"], 1)
            self.assertEqual(obj["learned"]["ranking"]["4"][0]["expert"], 7)
            self.assertEqual(obj["learned"]["ranking"]["4"][1]["expert"], 8)

            k1, k2 = obj["results"]
            self.assertEqual(k1["keep"], 1)
            self.assertAlmostEqual(k1["call_coverage"], 3 / 8)
            self.assertAlmostEqual(k1["mass_coverage"], 2.2 / 4.0)
            self.assertAlmostEqual(k1["all_selected_row_rate"], 0.0)
            self.assertAlmostEqual(k1["worst_layer_mass_coverage"], 0.45)

            self.assertEqual(k2["keep"], 2)
            self.assertAlmostEqual(k2["call_coverage"], 7 / 8)
            self.assertAlmostEqual(k2["mass_coverage"], 3.5 / 4.0)
            self.assertAlmostEqual(k2["all_selected_row_rate"], 3 / 4)
            self.assertAlmostEqual(k2["worst_layer_mass_coverage"], 0.75)

            csv_text = csv_out.read_text(encoding="utf-8")
            self.assertIn("evaluation_label", csv_text)
            self.assertIn("learn_self_coverage", csv_text)

    def test_keep_range_and_tie_break_expert_id_ascending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            trace = root / "route.csv"
            write_trace(trace, [(3, [(5, 1.0), (4, 1.0)])])

            data = CURVE.read_traces([trace], layer_min=3, layer_max=42, n_expert=256)
            results, learned = CURVE.evaluate(data, CURVE.parse_keeps("1-2"), n_expert=256)

            self.assertEqual([row["keep"] for row in results], [1, 2])
            self.assertEqual([entry["expert"] for entry in learned["ranking"]["3"][:2]], [4, 5])
            self.assertAlmostEqual(results[0]["mass_coverage"], 0.5)
            self.assertAlmostEqual(results[1]["mass_coverage"], 1.0)

    def test_repeatable_trace_aggregates_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            one = root / "one.csv"
            two = root / "two.csv"
            write_trace(one, [(3, [(1, 0.2)])])
            write_trace(two, [(3, [(2, 0.8)])])

            data = CURVE.read_traces([one, two], layer_min=3, layer_max=42, n_expert=256)
            results, learned = CURVE.evaluate(data, [1], n_expert=256)

            self.assertEqual([item["used_rows"] for item in data["inputs"]], [1, 1])
            self.assertEqual(learned["ranking"]["3"][0]["expert"], 2)
            self.assertAlmostEqual(results[0]["mass_coverage"], 0.8)
            self.assertAlmostEqual(results[0]["call_coverage"], 0.5)

    def test_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = pathlib.Path(tmpdir) / "route.csv"
            write_trace(trace, [(3, [(1, 1.0)])], weights=False)

            with self.assertRaises(SystemExit) as ctx:
                CURVE.main(["--trace", str(trace), "--keeps", "1"])
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_header_only_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = pathlib.Path(tmpdir) / "route.csv"
            write_trace(trace, [], weights=True)

            with self.assertRaises(SystemExit) as ctx:
                CURVE.main(["--trace", str(trace), "--keeps", "1"])
            self.assertEqual(ctx.exception.code, 2)

    def test_rejects_zero_mass_in_maskable_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = pathlib.Path(tmpdir) / "route.csv"
            write_trace(trace, [(3, [(1, 0.0)])])

            with self.assertRaises(SystemExit) as ctx:
                CURVE.main(["--trace", str(trace), "--keeps", "1"])
            self.assertEqual(ctx.exception.code, 2)

    def test_ignores_non_maskable_layers_but_fails_if_none_remain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = pathlib.Path(tmpdir) / "route.csv"
            write_trace(trace, [(2, [(1, 1.0)]), (43, [(2, 1.0)])])

            with self.assertRaises(SystemExit) as ctx:
                CURVE.main(["--trace", str(trace), "--keeps", "1"])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
