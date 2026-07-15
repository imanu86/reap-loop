from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_ds4_experiment_ledger.py"
SPEC = importlib.util.spec_from_file_location("build_ds4_experiment_ledger", SCRIPT)
assert SPEC and SPEC.loader
build_ds4_experiment_ledger = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_ds4_experiment_ledger
SPEC.loader.exec_module(build_ds4_experiment_ledger)


class BuildDs4ExperimentLedgerTests(unittest.TestCase):
    def test_quality_full_decode_mass_weight_rows_preserve_run_fields(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        rows = build_ds4_experiment_ledger.parse_quality_full_decode_mass_weight_rows(repo)

        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["row_id"] for row in rows}), 24)

        first = rows[0]
        self.assertEqual(first["source_kind"], "quality_full_decode_mass_weight_csv")
        self.assertEqual(first["category"], "linux_pod_oracle_quality_full_decode")
        self.assertEqual(first["runtime_platform"], "linux_pod_oracle")
        self.assertEqual(first["hardware"], "Linux pod oracle")
        self.assertEqual(first["prompt_name"], "software_build_dashboard_html")
        self.assertNotIn("windows_native", first["benchmark_usable"])
        self.assertEqual(
            first["suite"],
            "runs/ds4/20260712_pod12_bake/windows_bake_quality_full_decode_a_20260715",
        )
        self.assertEqual(first["variant"], "k0")
        self.assertEqual(
            first["run_id"],
            "windows_bake_quality_full_decode_a_20260715/k0/run1/temp0",
        )
        self.assertEqual(first["gpu"], "NVIDIA GeForce RTX 3090 Ti")
        self.assertEqual(first["model"], "deepseek-v4-flash")
        self.assertEqual(first["ctx"], "4096")
        self.assertEqual(first["server_cache_experts"], "1024")
        self.assertEqual(first["prefill_chunk"], "512")
        self.assertEqual(first["server_max_tokens"], "3328")
        self.assertEqual(first["request_max_tokens"], "3200")
        self.assertEqual(first["temperature"], "0")
        self.assertEqual(first["think"], "false")
        self.assertEqual(first["stream"], "true")
        self.assertEqual(first["l0l3"], "L2")
        self.assertEqual(first["client_stop_reason"], "client_stop_html_close")
        self.assertEqual(first["wall_s"], "449.15")
        self.assertEqual(first["finish_s"], "449.15")
        self.assertEqual(first["trace_rows"], "3004")
        self.assertEqual(first["avg_tps"], "6.68819")
        self.assertEqual(first["completion_tokens"], "")
        self.assertEqual(first["prompt_sha16"], "35c7b4c82bef5ec1")
        self.assertEqual(
            first["harness_sha256"],
            "17192b6a48a1b7897474d4fcdb9c1a4089312327f9ba99edc8cd7379261913d4",
        )
        self.assertEqual(
            first["executable_sha256"],
            "4bb29874a7028cc06c7a1d1f6696528a854694e6f7d8de626f875abc3ecf2f76",
        )
        self.assertIn("mask_sha256=", first["setup_text"])
        self.assertNotIn("mask_sha256=0", first["setup_text"])

    def test_quality_full_decode_temperatures_are_robustness_conditions(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        rows = build_ds4_experiment_ledger.parse_quality_full_decode_mass_weight_rows(repo)

        k60 = [
            row
            for row in rows
            if row["suite"].endswith("windows_bake_quality_full_decode_a_20260715")
            and row["variant"] == "k60"
        ]

        self.assertEqual([row["temperature"] for row in k60], ["0", "0.2", "0.7"])
        self.assertTrue(all(row["repeats"] == "1" for row in k60))
        self.assertTrue(
            all(
                row["replication_scope"] == "temperature_robustness_condition_not_iid"
                for row in k60
            )
        )
        self.assertTrue(
            all("temperature_condition=robustness_not_iid" in row["quality_signal"] for row in k60)
        )
        self.assertTrue(all("mask_sha256=5b6d98504ba830c1" in row["setup_text"] for row in k60))


if __name__ == "__main__":
    unittest.main()
