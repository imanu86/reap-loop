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
        target_hit_rate=[0.2, 0.5],
        tier_sim_cap=[],
        tier_sim_source="compact_ids",
        tier_warm_grace=0,
        tier_freeze_after=0,
        tier_warm_scale=1.0,
        tier_cold_scale=0.5,
        tier_frozen_scale=0.0,
        tier_prefill_rows="0",
        tier_preload_top=0,
        tier_preload_source="selected",
    )

    assert "cap=2 requests=4 hit_rate=0.2500 hits=1 misses=3" in text
    assert "target_hit_rate=0.2000 first_cap=2 hit_rate=0.2500" in text
    assert "target_hit_rate=0.5000 first_cap=unmet" in text
    assert "native=13.5MiB" in text
    assert "x0.5=6.8MiB" in text
    assert "x0.33=4.5MiB" in text


def test_tier_policy_tracks_hot_warm_cold_and_frozen_recalls():
    mod = _load_module()
    rows = [
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 2], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 3], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [2], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [4], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [5], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [6], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [7], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [2], "path": "selected_direct"},
    ]

    result = mod.simulate_tier_policy(
        rows,
        cap=2,
        source="compact_ids",
        warm_grace=1,
        freeze_after=2,
    )

    assert result["requests"] == 10
    assert result["hot_hits"] == 1
    assert result["warm_hits"] == 1
    assert result["frozen_recalls"] == 1
    assert result["initial_loads"] == 7
    assert result["promotions"] == 8
    assert result["promotion_rate"] == 0.8
    assert result["peak_hot"] == 2
    assert result["peak_warm"] >= 1


def test_tier_policy_is_reported_in_summary():
    mod = _load_module()
    rows = [
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 2], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "compact_ids": [1, 3], "path": "selected_direct"},
    ]

    text = mod.summarize(
        rows,
        top=1,
        simulate_cap=[],
        slot_mib=10.0,
        capacity_scales=[],
        target_hit_rate=[],
        tier_sim_cap=[2],
        tier_sim_source="compact_ids",
        tier_warm_grace=1,
        tier_freeze_after=4,
        tier_warm_scale=1.0,
        tier_cold_scale=0.25,
        tier_frozen_scale=0.0,
        tier_prefill_rows="0",
        tier_preload_top=0,
        tier_preload_source="selected",
    )

    assert "tier_sim: source=compact_ids warm_grace=1 freeze_after=4" in text
    assert "cap=2 requests=4 hot_hit_rate=0.2500 served_hit_rate=0.2500" in text
    assert "promotions=3 promotion_rate=0.7500" in text
    assert "footprint_scaled=" in text


def test_routing_csv_can_feed_tier_simulation(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "routing.csv"
    csv_path.write_text(
        "\n".join(
            [
                "pos,layer,n,e0,e1,e2,w0,w1,w2",
                "10,3,3,1,2,3,0.5,0.4,0.1",
                "11,3,3,1,4,5,0.6,0.3,0.1",
            ]
        ),
        encoding="utf-8",
    )

    rows = mod.load_rows([csv_path])
    result = mod.simulate_tier_policy(
        rows,
        cap=3,
        source="compact_ids",
        warm_grace=0,
        freeze_after=0,
    )

    assert len(rows) == 2
    assert rows[0]["path"] == "routing_csv"
    assert rows[0]["compact_ids"] == [1, 2, 3]
    assert result["requests"] == 6
    assert result["hot_hits"] == 1


def test_prompt_preload_replays_after_prefill_rows():
    mod = _load_module()
    rows = [
        {"event": "tiering_observe", "layer": 0, "selected": [1, 1, 2], "compact_ids": [1, 2], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 1, "selected": [3], "compact_ids": [3], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 0, "selected": [1], "compact_ids": [1], "path": "selected_direct"},
        {"event": "tiering_observe", "layer": 1, "selected": [3], "compact_ids": [3], "path": "selected_direct"},
    ]

    assert mod.infer_first_layer_cycle_rows(rows) == 2
    text = mod.summarize(
        rows,
        top=1,
        simulate_cap=[],
        slot_mib=10.0,
        capacity_scales=[],
        target_hit_rate=[],
        tier_sim_cap=[2],
        tier_sim_source="compact_ids",
        tier_warm_grace=0,
        tier_freeze_after=0,
        tier_warm_scale=1.0,
        tier_cold_scale=0.25,
        tier_frozen_scale=0.0,
        tier_prefill_rows="auto",
        tier_preload_top=2,
        tier_preload_source="selected",
    )

    assert "prefill_rows=2 preload_source=selected" in text
    assert "requests=2" in text
    assert "preloaded=2" in text
    assert "hot_hits=1" in text
