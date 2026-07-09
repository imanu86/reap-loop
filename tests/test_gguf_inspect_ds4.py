from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gguf_inspect_ds4.py"
SPEC = importlib.util.spec_from_file_location("gguf_inspect_ds4", SCRIPT)
assert SPEC and SPEC.loader
gguf_inspect_ds4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gguf_inspect_ds4
SPEC.loader.exec_module(gguf_inspect_ds4)


def test_ds4_quantized_tensor_sizes_match_runtime_table() -> None:
    assert gguf_inspect_ds4.tensor_nbytes(16, 256) == 66
    assert gguf_inspect_ds4.tensor_nbytes(10, 256) == 84
    assert gguf_inspect_ds4.tensor_nbytes(12, 256) == 144
    assert gguf_inspect_ds4.tensor_nbytes(16, 512) == 132


def test_parse_int_list_supports_ranges() -> None:
    assert gguf_inspect_ds4.parse_int_list("1,3-5,8") == [1, 3, 4, 5, 8]
    assert gguf_inspect_ds4.parse_int_list("5-3") == [5, 4, 3]
    assert gguf_inspect_ds4.parse_int_list(None) is None
