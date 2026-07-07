"""Integrazione metrics -> report: blocca il contratto CSV tra i due moduli (scritti da agent
distinti in parallelo) e verifica il PROTOCOLLO DI FALSIFICAZIONE (docs §12) su oggetti REALI.

Niente CSV sintetico fatto a mano: costruiamo CellMetrics veri -> rows_to_dataframe -> CSV ->
funzioni di report. Cosi se i nomi di colonna divergessero, qui esplode (non in produzione).

Caso-chiave (R1): una regione che regge a ctx corto ma sfora a ctx lungo NON dev'essere promossa.
"""

from __future__ import annotations

import pandas as pd

from msc.experiment.metrics import CellMetrics, VramBreakdown, rows_to_dataframe
from msc.report import curves

_GIB = 1 << 30
_FULL_ACC = 0.80
_CTX = [1024, 4096, 16384, 65536]


def _cell(model, sparsity, miss_mode, k, ctx, drop, n_eff):
    """CellMetrics realistico: accuracy = FULL - drop; VRAM_expert proporzionale a K."""
    return CellMetrics(
        model_id=model,
        policy="AGGRESSIVE-COMMIT",
        k_fraction=k,
        ctx_len=ctx,
        miss_mode=miss_mode,
        sparsity_ratio=sparsity,
        accuracy=_FULL_ACC - drop,
        accuracy_drop_vs_full=drop,
        mean_n_eff=n_eff,
        mean_entropy_norm=0.5,
        vram=VramBreakdown(
            backbone=1 * _GIB,
            kv_cache=int(ctx / 1024 * 0.1 * _GIB),   # cresce col contesto (R6)
            experts_resident=int(k * 4 * _GIB),       # il termine su cui agisce la policy
            overhead=1 << 28,
        ),
        miss_rate=min(0.9, (1.0 - k) * (ctx / 65536)),
        latency_ms_per_token=10.0,
        working_set_converged=True,
    )


def _build_grid():
    """Due modelli, precision-cascade:
       - granite (s=0.20): k=0.25 regge su TUTTE le ctx, k=0.12 collassa a ctx lunga -> PROMOSSO k=0.25
       - olmoe   (s=0.125): nessun K regge su tutte le ctx (collasso R1)         -> FALLITO
    """
    # drop[k][ctx_index]
    granite = {
        0.50: [0.005, 0.005, 0.006, 0.008],   # regge
        0.25: [0.010, 0.012, 0.015, 0.018],   # regge (worst 0.018 < 0.02)
        0.12: [0.010, 0.015, 0.025, 0.040],   # SFORA a 16k/64k -> non promosso
    }
    olmoe = {
        0.50: [0.010, 0.015, 0.030, 0.050],   # gia sfora a 16k
        0.25: [0.020, 0.030, 0.060, 0.090],
        0.12: [0.050, 0.080, 0.120, 0.200],
    }
    cells = []
    for k, drops in granite.items():
        for ctx, d in zip(_CTX, drops):
            cells.append(_cell("granite-3b", 0.20, "precision-cascade", k, ctx, d, n_eff=12.0))
    for k, drops in olmoe.items():
        for ctx, d in zip(_CTX, drops):
            cells.append(_cell("olmoe", 0.125, "precision-cascade", k, ctx, d, n_eff=28.0))
    return cells


def _csv(tmp_path):
    df = rows_to_dataframe(_build_grid())
    p = tmp_path / "metrics.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_to_row_columns_match_report_contract():
    """Le colonne che il report si aspetta esistono davvero nell'output di to_row (anti-drift)."""
    row = _build_grid()[0].to_row()
    # la colonna asse-x del report deve esistere e NON essere aggregata con kv/backbone
    assert curves._VRAM_EXPERT_COL in row
    for col in ("vram_kv_cache", "vram_backbone", "vram_overhead", "vram_total",
                "model_id", "miss_mode", "ctx_len", "accuracy", "accuracy_drop_vs_full"):
        assert col in row, col
    # R6: KV separata dagli expert
    assert "vram" not in row  # niente colonna nested


def test_verdict_promotes_only_region_holding_on_all_ctx(tmp_path):
    v = curves.verdict(_csv(tmp_path), drop_threshold=0.02)

    g = v["granite-3b::precision-cascade"]
    assert g["failed"] is False
    assert g["promoted_k_fraction"] == 0.25   # k=0.12 sfora a ctx lunga -> scartato
    assert g["max_drop_at_promoted"] < 0.02
    assert g["n_ctx_tested"] == len(_CTX)

    o = v["olmoe::precision-cascade"]
    assert o["failed"] is True                 # collasso R1: nessun K regge su tutte le ctx
    assert o["promoted_k_fraction"] is None


def test_report_figures_are_produced(tmp_path):
    csv = _csv(tmp_path)
    p1 = tmp_path / "acc_vs_vram.png"
    curves.accuracy_vs_vram_curves(csv, model_id="granite-3b", miss_mode="precision-cascade",
                                   out_path=str(p1))
    p2 = tmp_path / "isoacc.png"
    curves.isoaccuracy_gain_vs_sparsity(csv, drop_threshold=0.02, out_path=str(p2))
    curves.diagnostics(csv, out_dir=str(tmp_path / "diag"))

    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0
    assert (tmp_path / "diag" / "diag_missrate_vs_ctx.png").exists()


def test_roundtrip_csv_reload_is_stable(tmp_path):
    """Il CSV riletto da pandas conserva i tipi che il report converte (nessun NaN spurio)."""
    df = pd.read_csv(_csv(tmp_path))
    assert set(["granite-3b", "olmoe"]).issubset(set(df["model_id"]))
    assert df["accuracy_drop_vs_full"].notna().all()
    assert df[curves._VRAM_EXPERT_COL].notna().all()
