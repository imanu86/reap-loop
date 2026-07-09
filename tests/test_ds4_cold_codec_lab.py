from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ds4_cold_codec_lab.py"
SPEC = importlib.util.spec_from_file_location("ds4_cold_codec_lab", SCRIPT)
assert SPEC and SPEC.loader
ds4_cold_codec_lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ds4_cold_codec_lab
SPEC.loader.exec_module(ds4_cold_codec_lab)


def test_cq1_payload_sizes() -> None:
    assert ds4_cold_codec_lab.codec_payload_bytes(256) == 34
    assert ds4_cold_codec_lab.codec_payload_bytes(64) == 40
    assert ds4_cold_codec_lab.codec_payload_bytes(32) == 48


def test_cq1_uses_group_mean_abs_scale() -> None:
    values = [-2.0, -1.0, 1.0, 4.0]
    assert ds4_cold_codec_lab.cq1(values, 4) == [-2.0, -2.0, 2.0, 2.0]


def test_middle_policy_sizes_are_between_native_and_full_cold() -> None:
    native = (
        ds4_cold_codec_lab.NATIVE_GATE_MIB
        + ds4_cold_codec_lab.NATIVE_UP_MIB
        + ds4_cold_codec_lab.NATIVE_DOWN_MIB
    )
    down_only = ds4_cold_codec_lab.policy_expert_mib(64, "down-only")
    all_cold = ds4_cold_codec_lab.policy_expert_mib(64, "all")
    assert all_cold < down_only < native
