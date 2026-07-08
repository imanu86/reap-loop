from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "analyze_tiering_observe.py"
    spec = importlib.util.spec_from_file_location("analyze_tiering_observe", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lru_capacity_cost_is_reported():
    mod = _load_module()
    rows = [
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 2], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 3], "path": "selected_direct"},
    ]

    text = mod.summarize(
        rows,
        top=1,
        simulate_cap=[2],
        slot_mib=6.75,
        capacity_scales=[0.5, 0.33],
    )

    assert "cap=2 requests=4 hit_rate=0.2500 hits=1 misses=3" in text
    assert "native=13.5MiB" in text
    assert "x0.5=6.8MiB" in text
    assert "x0.33=4.5MiB" in text
