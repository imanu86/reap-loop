"""test_hierarchy_tools.py — calibratore (probe->residenza) + metriche di capacita'.

CPU-only, deterministico, isolato: usa solo `tmp_path`, la stdlib e `msc.hierarchy`. Nessun
torch/transformers, nessuna GPU, nessun modulo implementato da altri agent.

Si verifica:
  - resident_from_trace: top-n giusti per frequenza, e nesting per n crescenti (il set a n piccolo
    e' sottoinsieme del set a n piu' grande).
  - blind_resident: set fisso 0..n-1 su ogni layer, clamp a n_experts, nessun aliasing.
  - capacity_report: fetch_rate e capacity_ratio corretti su uno stats-dict sintetico (schema
    CacheHandle.stats() dell'INTERFACCIA) + estimate_vram_expert_gb coerente.
"""

from __future__ import annotations

import json

from msc.hierarchy.calibrate import blind_resident, resident_from_trace
from msc.hierarchy.metrics import capacity_report, estimate_vram_expert_gb


# --------------------------------------------------------------------------- #
# Helper: costruzione di una traccia jsonl sintetica con frequenze NOTE.       #
# --------------------------------------------------------------------------- #
def _write_trace(path, records) -> None:
    """Scrive una lista di dict (schema msc.instrument.trace) come jsonl."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _rec(layer, topk_ids):
    """Una riga di routing minimale: solo i campi che il calibratore legge (layer, topk_ids)."""
    return {"layer": layer, "topk_ids": list(topk_ids)}


def _build_trace(path):
    """Traccia a frequenze controllate.

    Layer 0 (occorrenze totali per expert nei topk_ids):
        expert 5 -> 5, expert 3 -> 4, expert 1 -> 3, expert 9 -> 2, expert 7 -> 1
        => ordine di frequenza desc: [5, 3, 1, 9, 7]
    Layer 1:
        expert 2 -> 3, expert 8 -> 2, expert 4 -> 1
        => ordine: [2, 8, 4]
    """
    records = []
    # Layer 0 — emetto conteggi esatti distribuendo gli id nei topk.
    l0_counts = {5: 5, 3: 4, 1: 3, 9: 2, 7: 1}
    for eid, c in l0_counts.items():
        for _ in range(c):
            records.append(_rec(0, [eid]))
    # Layer 1.
    l1_counts = {2: 3, 8: 2, 4: 1}
    for eid, c in l1_counts.items():
        for _ in range(c):
            records.append(_rec(1, [eid]))
    _write_trace(path, records)
    return path


# --------------------------------------------------------------------------- #
# resident_from_trace                                                          #
# --------------------------------------------------------------------------- #
def test_resident_from_trace_topn_correct(tmp_path):
    """Per ogni layer i top-n expert per frequenza sono quelli attesi."""
    path = _build_trace(tmp_path / "trace.jsonl")

    res2 = resident_from_trace(str(path), n_per_layer=2)
    assert res2[0] == {5, 3}      # i due piu' frequenti del layer 0
    assert res2[1] == {2, 8}      # i due piu' frequenti del layer 1

    res3 = resident_from_trace(str(path), n_per_layer=3)
    assert res3[0] == {5, 3, 1}
    assert res3[1] == {2, 8, 4}


def test_resident_from_trace_nested_for_growing_n(tmp_path):
    """I set a n crescente sono annidati: top-n ⊆ top-(n+1) (per ogni layer)."""
    path = _build_trace(tmp_path / "trace.jsonl")

    res1 = resident_from_trace(str(path), n_per_layer=1)
    res2 = resident_from_trace(str(path), n_per_layer=2)
    res4 = resident_from_trace(str(path), n_per_layer=4)

    for layer in (0, 1):
        assert res1[layer] <= res2[layer] <= res4[layer]

    # il singolo top-1 e' l'expert piu' frequente.
    assert res1[0] == {5}
    assert res1[1] == {2}


def test_resident_from_trace_n_exceeds_distinct_returns_all(tmp_path):
    """n piu' grande del numero di expert distinti -> ritorna tutti gli osservati (no padding)."""
    path = _build_trace(tmp_path / "trace.jsonl")
    res = resident_from_trace(str(path), n_per_layer=100)
    assert res[0] == {5, 3, 1, 9, 7}   # tutti e 5 i distinti del layer 0
    assert res[1] == {2, 8, 4}


def test_resident_from_trace_multi_id_topk_counts_each(tmp_path):
    """Ogni id in topk_ids conta come un'attivazione (top-k reale, non solo top-1)."""
    path = tmp_path / "trace_multi.jsonl"
    # layer 0: expert 1 compare in ogni riga (4 volte), 2 in 3 righe, 3 in 2, 4 in 1.
    records = [
        _rec(0, [1, 2, 3, 4]),
        _rec(0, [1, 2, 3]),
        _rec(0, [1, 2]),
        _rec(0, [1]),
    ]
    _write_trace(path, records)
    res = resident_from_trace(str(path), n_per_layer=2)
    assert res[0] == {1, 2}


def test_resident_from_trace_ignores_blank_and_malformed_lines(tmp_path):
    """Righe vuote o prive di layer/topk_ids vengono ignorate senza errori."""
    path = tmp_path / "trace_blanks.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_rec(0, [7, 7, 9])) + "\n")
        fh.write(json.dumps({"layer": 0}) + "\n")          # manca topk_ids -> ignorata
        fh.write(json.dumps({"topk_ids": [1, 2]}) + "\n")  # manca layer -> ignorata
        fh.write("   \n")
        fh.write(json.dumps(_rec(0, [9])) + "\n")
    res = resident_from_trace(str(path), n_per_layer=1)
    # expert 7 -> 2, expert 9 -> 2 ... tie: most_common e' deterministico (ordine di comparsa: 7).
    assert res[0] == {7}


def test_resident_from_trace_n_zero_gives_empty_sets(tmp_path):
    """n_per_layer=0 -> set vuoto per ogni layer osservato (non KeyError)."""
    path = _build_trace(tmp_path / "trace.jsonl")
    res = resident_from_trace(str(path), n_per_layer=0)
    assert res == {0: set(), 1: set()}


# --------------------------------------------------------------------------- #
# blind_resident                                                              #
# --------------------------------------------------------------------------- #
def test_blind_resident_fixed_low_ids_per_layer():
    """Baseline cieca: stesso set 0..n-1 su ogni layer, forma uguale a resident_from_trace."""
    res = blind_resident(range(4), n_per_layer=3)
    assert set(res.keys()) == {0, 1, 2, 3}
    for layer in range(4):
        assert res[layer] == {0, 1, 2}


def test_blind_resident_clamps_to_n_experts():
    """n_per_layer non puo' superare n_experts (non si pinnano expert inesistenti)."""
    res = blind_resident([0, 1], n_per_layer=10, n_experts=4)
    assert res[0] == {0, 1, 2, 3}
    assert res[1] == {0, 1, 2, 3}


def test_blind_resident_no_aliasing_between_layers():
    """Ogni layer ha un proprio set: modificarne uno non tocca gli altri (no aliasing)."""
    res = blind_resident(range(3), n_per_layer=2)
    res[0].add(99)
    assert res[1] == {0, 1}
    assert res[2] == {0, 1}


def test_blind_resident_same_shape_as_probe(tmp_path):
    """probe e cieco hanno la STESSA forma di output (layer_idx -> set[int])."""
    path = _build_trace(tmp_path / "trace.jsonl")
    probe = resident_from_trace(str(path), n_per_layer=2)
    blind = blind_resident(probe.keys(), n_per_layer=2)
    assert set(probe.keys()) == set(blind.keys())
    for layer in probe:
        assert isinstance(probe[layer], set) and isinstance(blind[layer], set)


# --------------------------------------------------------------------------- #
# capacity_report + estimate_vram_expert_gb                                    #
# --------------------------------------------------------------------------- #
def _synthetic_cache_stats():
    """Stats-dict sintetico nello schema CacheHandle.stats() dell'INTERFACCIA.

    total: hits=70, misses=30 -> fetch_rate atteso = 0.30.
    """
    per_layer = {
        0: {"hits": 40, "misses": 10, "fetched_bytes": 100, "resident": 8},
        1: {"hits": 30, "misses": 20, "fetched_bytes": 200, "resident": 8},
    }
    total = {"hits": 70, "misses": 30, "fetched_bytes": 300, "resident": 16}
    return {
        "per_layer": per_layer,
        "total": total,
        "fetch_rate": 30 / (70 + 30),   # 0.30
        "fetched_gb": 2.0,
        "resident_experts": 16,
    }


def test_capacity_report_fetch_rate_and_ratio_correct():
    """fetch_rate dallo stats-dict e capacity_ratio = model_total_gb / vram_expert_gb."""
    stats = _synthetic_cache_stats()
    rep = capacity_report(stats, model_total_gb=40.0, vram_expert_gb=8.0)

    assert rep["fetch_rate"] == 0.30
    assert rep["fetched_gb"] == 2.0
    assert rep["resident_experts"] == 16
    # 40 GB di modello per 8 GB di VRAM-expert -> 5x di capacita'.
    assert rep["capacity_ratio"] == 5.0
    assert isinstance(rep["summary"], str) and "5.0x" in rep["summary"]


def test_capacity_report_recomputes_fetch_rate_if_missing():
    """Se manca la chiave fetch_rate, viene ricalcolata da total (misses/(hits+misses))."""
    stats = _synthetic_cache_stats()
    del stats["fetch_rate"]
    rep = capacity_report(stats, model_total_gb=10.0, vram_expert_gb=2.0)
    assert rep["fetch_rate"] == 0.30        # 30 / 100
    assert rep["capacity_ratio"] == 5.0     # 10 / 2


def test_capacity_report_zero_vram_no_div_by_zero():
    """vram_expert_gb=0 -> capacity_ratio 0.0 (nessuna divisione per zero)."""
    stats = _synthetic_cache_stats()
    rep = capacity_report(stats, model_total_gb=40.0, vram_expert_gb=0.0)
    assert rep["capacity_ratio"] == 0.0


def test_capacity_report_zero_access_fetch_rate_zero():
    """Nessun accesso (hits+misses=0) e fetch_rate assente -> fetch_rate 0.0."""
    stats = {
        "per_layer": {},
        "total": {"hits": 0, "misses": 0, "fetched_bytes": 0, "resident": 0},
        "fetched_gb": 0.0,
        "resident_experts": 0,
    }
    rep = capacity_report(stats, model_total_gb=40.0, vram_expert_gb=8.0)
    assert rep["fetch_rate"] == 0.0
    assert rep["capacity_ratio"] == 5.0


def test_estimate_vram_expert_gb():
    """VRAM-expert = capacity_per_layer * n_layers * bytes_per_expert / 2**30."""
    gib = float(1 << 30)
    # 8 residenti * 16 layer * 0.5 GiB l'uno = 64 GiB.
    vram = estimate_vram_expert_gb(
        capacity_per_layer=8, n_layers=16, bytes_per_expert=0.5 * gib
    )
    assert vram == 64.0

    # caso degenere: zero capienza -> 0 GB.
    assert estimate_vram_expert_gb(0, 16, 1.0 * gib) == 0.0


def test_estimate_and_report_compose():
    """estimate_vram_expert_gb alimenta capacity_report: capacity_ratio coerente end-to-end."""
    gib = float(1 << 30)
    # 4 residenti * 16 layer * 0.25 GiB = 16 GiB di VRAM-expert.
    vram_gb = estimate_vram_expert_gb(4, 16, 0.25 * gib)
    assert vram_gb == 16.0
    rep = capacity_report(_synthetic_cache_stats(), model_total_gb=160.0, vram_expert_gb=vram_gb)
    assert rep["capacity_ratio"] == 10.0   # 160 / 16
