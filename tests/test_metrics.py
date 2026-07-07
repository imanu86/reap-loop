"""test_metrics.py — la riga di metriche per cella deve essere piatta e separare VRAM-KV da VRAM-expert.

GATE concettuale del rischio R6: a contesto lungo la KV cache puo' dominare; se nel CSV la VRAM-KV
fosse sommata a quella degli expert, il "guadagno della policy" a 64k sembrerebbe svanire per colpa
della KV, non della policy. Quindi to_row() DEVE esporre colonne separate.

CPU-only, deterministico, isolato: usa solo msc.experiment.metrics (niente torch/transformers,
niente moduli implementati da altri agent).
"""

from __future__ import annotations

from msc.experiment.metrics import CellMetrics, VramBreakdown, rows_to_dataframe


def _make_cell() -> CellMetrics:
    """Costruisce una CellMetrics deterministica con valori VRAM facilmente riconoscibili."""
    vram = VramBreakdown(
        backbone=1_000,
        kv_cache=2_000,
        experts_resident=4_000,
        overhead=8_000,
    )
    return CellMetrics(
        model_id="OLMoE-1B-7B",
        policy="AGGRESSIVE-COMMIT",
        k_fraction=0.25,
        ctx_len=65_536,
        miss_mode="precision-cascade",
        sparsity_ratio=8 / 64,
        accuracy=0.91,
        accuracy_drop_vs_full=0.015,
        mean_n_eff=6.5,
        mean_entropy_norm=0.42,
        vram=vram,
        miss_rate=0.07,
        latency_ms_per_token=12.3,
        working_set_converged=True,
    )


def test_vram_total_is_sum_of_parts():
    """total e' esattamente la somma dei quattro contributi (potenze di due distinte -> niente collisioni)."""
    vram = VramBreakdown(backbone=1_000, kv_cache=2_000, experts_resident=4_000, overhead=8_000)
    assert vram.total == 15_000
    assert vram.total == vram.backbone + vram.kv_cache + vram.experts_resident + vram.overhead


def test_to_row_is_flat_dict_of_scalars():
    """Ogni valore della riga deve essere scalare: nessun VramBreakdown annidato."""
    row = _make_cell().to_row()
    assert isinstance(row, dict)
    # nessun valore deve essere una VramBreakdown (cioe' deve essere stato scorporato)
    assert not any(isinstance(v, VramBreakdown) for v in row.values())


def test_to_row_separates_kv_from_experts():
    """R6: VRAM-KV e VRAM-expert come colonne SEPARATE, non aggregate."""
    row = _make_cell().to_row()
    # colonne separate presenti
    assert "vram_kv_cache" in row
    assert "vram_experts_resident" in row
    assert "vram_backbone" in row
    assert "vram_overhead" in row
    # valori corretti e DISTINTI (non sommati tra loro)
    assert row["vram_kv_cache"] == 2_000
    assert row["vram_experts_resident"] == 4_000
    assert row["vram_kv_cache"] != row["vram_experts_resident"]
    # il totale e' una colonna a parte, derivata, NON sovrascrive i contributi
    assert row["vram_total"] == 15_000


def test_to_row_contains_accuracy_drop_vs_full():
    """Il drop vs FULL (su cui si valuta il < 2%) deve avere una sua colonna."""
    row = _make_cell().to_row()
    assert "accuracy_drop_vs_full" in row
    assert row["accuracy_drop_vs_full"] == 0.015
    # e l'accuratezza assoluta resta una colonna distinta
    assert "accuracy" in row
    assert row["accuracy"] == 0.91


def test_to_row_preserves_identity_fields():
    """Gli assi della cella (id, policy, K, ctx, miss_mode) restano colonne piatte."""
    row = _make_cell().to_row()
    assert row["model_id"] == "OLMoE-1B-7B"
    assert row["policy"] == "AGGRESSIVE-COMMIT"
    assert row["k_fraction"] == 0.25
    assert row["ctx_len"] == 65_536
    assert row["miss_mode"] == "precision-cascade"
    assert row["working_set_converged"] is True
    # la chiave annidata "vram" non deve sopravvivere come tale
    assert "vram" not in row


def test_round_trip_to_dataframe():
    """rows_to_dataframe: round-trip di piu' celle in un DataFrame con le colonne separate."""
    cells = [_make_cell(), _make_cell()]
    df = rows_to_dataframe(cells)
    assert len(df) == 2
    # le colonne critiche per R6 esistono nel DataFrame
    for col in ("vram_kv_cache", "vram_experts_resident", "vram_total", "accuracy_drop_vs_full"):
        assert col in df.columns
    # i valori sopravvivono al round-trip
    assert df["vram_kv_cache"].iloc[0] == 2_000
    assert df["vram_experts_resident"].iloc[0] == 4_000
    assert df["vram_total"].iloc[0] == 15_000


def test_rows_to_dataframe_accepts_dicts():
    """L'helper accetta anche dict gia' piatti (non solo CellMetrics)."""
    row = _make_cell().to_row()
    df = rows_to_dataframe([row, row])
    assert len(df) == 2
    assert df["vram_experts_resident"].iloc[1] == 4_000
