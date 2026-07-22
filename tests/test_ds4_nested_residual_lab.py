from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import struct
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ds4_nested_residual_lab.py"
SPEC = importlib.util.spec_from_file_location("ds4_nested_residual_lab", SCRIPT)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


def test_iq2_xxs_split_is_exact() -> None:
    rng = random.Random(7)
    block = struct.pack("<H", 0x3C00) + bytes(rng.randrange(256) for _ in range(64))
    base, residual = lab.split_iq2_xxs(block)
    assert len(base) == 34
    assert len(residual) == 32
    assert lab.join_iq2_xxs(base, residual) == block


def test_q2_k_split_is_exact() -> None:
    rng = random.Random(11)
    block = bytes(rng.randrange(256) for _ in range(80)) + struct.pack("<HH", 0x3C00, 0x3800)
    base, residual = lab.split_q2_k(block)
    assert len(base) == 52
    assert len(residual) == 32
    assert lab.join_q2_k(base, residual) == block


def test_q2_k_base_decoder_has_expected_shape() -> None:
    block = bytes([0x11] * 16) + bytes([0xE4] * 64) + struct.pack("<HH", 0x3C00, 0x3800)
    base, _ = lab.split_q2_k(block)
    values = lab.decode_q2_k_base(base)
    assert len(values) == 256
    assert len(set(values)) == 2


def test_active_routed_size_math() -> None:
    assert lab.ACTIVE_ROUTED_EXPERTS == 10240
    assert lab.ACTIVE_ROUTED_EXPERTS * 3.75 / 1024 == 37.5
    assert lab.ACTIVE_ROUTED_EXPERTS * 3.0 / 1024 == 30.0
