"""test_report.py — test CPU-only, deterministici e isolati per msc.report.curves.

Non dipendono da torch ne' da moduli implementati da altri agent: costruiscono un CSV
SINTETICO con le colonne attese (coerenti con experiment/metrics.py::CellMetrics, con
VramBreakdown appiattito col prefisso ``vram_``) e verificano:
  - che i PNG vengano effettivamente creati;
  - che `verdict` restituisca la struttura attesa su un caso COSTRUITO con una regione
    che regge su tutte le ctx e una che collassa a contesto lungo (protocollo §12).
"""

from __future__ import annotations

import csv

import pytest

from msc.report import (
    accuracy_vs_vram_curves,
    diagnostics,
    isoaccuracy_gain_vs_sparsity,
    verdict,
)

# Colonne del CSV: appiattimento di CellMetrics (VramBreakdown -> prefisso vram_).
_COLUMNS = [
    "model_id", "policy", "k_fraction", "ctx_len", "miss_mode", "sparsity_ratio",
    "accuracy", "accuracy_drop_vs_full",
    "mean_n_eff", "mean_entropy_norm",
    "vram_backbone", "vram_kv_cache", "vram_experts_resident", "vram_overhead", "vram_total",
    "miss_rate", "latency_ms_per_token", "working_set_converged",
]

_GIB = 1 << 30
# VRAM_expert (byte) per k_fraction: piu' aggressivo (K piccolo) -> meno VRAM.
_VRAM_EXPERT_BY_K = {1.0: 8 * _GIB, 0.5: 4 * _GIB, 0.25: 2 * _GIB}
_CTXS = [1024, 4096, 16384]


def _row(model_id, k, ctx, miss_mode, accuracy, drop, *,
         sparsity, n_eff, miss_rate=0.1):
    vram_expert = _VRAM_EXPERT_BY_K[k]
    # KV cresce con la ctx (R6): serve a far funzionare il diagnostico VRAM_kv vs expert.
    vram_kv = int(ctx / 1024) * (1 * _GIB // 4)
    return {
        "model_id": model_id,
        "policy": "AGGRESSIVE-COMMIT",
        "k_fraction": k,
        "ctx_len": ctx,
        "miss_mode": miss_mode,
        "sparsity_ratio": sparsity,
        "accuracy": accuracy,
        "accuracy_drop_vs_full": drop,
        "mean_n_eff": n_eff,
        "mean_entropy_norm": 0.7,
        "vram_backbone": 2 * _GIB,
        "vram_kv_cache": vram_kv,
        "vram_experts_resident": vram_expert,
        "vram_overhead": 1 * _GIB,
        "vram_total": 2 * _GIB + vram_kv + vram_expert + 1 * _GIB,
        "miss_rate": miss_rate,
        "latency_ms_per_token": 5.0,
        "working_set_converged": True,
    }


def _write_synthetic_csv(path) -> str:
    """Costruisce un CSV con due modelli e un caso 'regge' + un caso 'collassa a ctx lunga'.

    Modello A (precision-cascade): a K=0.25 il drop resta < 2% su TUTTE le ctx -> PROMOSSO.
    Modello B (precision-cascade): a K=0.25 regge a ctx corto/medio ma SFORA a 16k (drop 5%)
        -> deve scendere a K=0.5 per essere promosso (falsificazione del K piu' aggressivo).
    """
    rows: list[dict] = []

    # --- Modello A: regge anche al K piu' aggressivo (0.25) su tutte le ctx ---
    for ctx in _CTXS:
        rows.append(_row("model_A", 1.0, ctx, "precision-cascade", 0.90, 0.00,
                         sparsity=0.125, n_eff=6.0, miss_rate=0.05))
        rows.append(_row("model_A", 0.5, ctx, "precision-cascade", 0.895, 0.005,
                         sparsity=0.125, n_eff=6.0, miss_rate=0.10))
        rows.append(_row("model_A", 0.25, ctx, "precision-cascade", 0.885, 0.015,
                         sparsity=0.125, n_eff=6.0, miss_rate=0.15))

    # --- Modello B: a K=0.25 collassa a 16k (drop 5%); a K=0.5 regge ovunque ---
    drop_b_025 = {1024: 0.010, 4096: 0.018, 16384: 0.050}  # 16k SFORA il 2%
    for ctx in _CTXS:
        rows.append(_row("model_B", 1.0, ctx, "precision-cascade", 0.80, 0.00,
                         sparsity=0.20, n_eff=9.0, miss_rate=0.05))
        rows.append(_row("model_B", 0.5, ctx, "precision-cascade", 0.795, 0.005,
                         sparsity=0.20, n_eff=9.0, miss_rate=0.12))
        rows.append(_row("model_B", 0.25, ctx, "precision-cascade",
                         0.80 - drop_b_025[ctx], drop_b_025[ctx],
                         sparsity=0.20, n_eff=9.0, miss_rate=0.30))

    csv_path = str(path / "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #

def test_accuracy_vs_vram_curves_creates_png(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path)
    out = tmp_path / "curve_A.png"
    accuracy_vs_vram_curves(csv_path, model_id="model_A",
                            miss_mode="precision-cascade", out_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_accuracy_vs_vram_curves_empty_selection_still_writes_png(tmp_path):
    """Selezione vuota (modello inesistente) -> PNG placeholder, nessuna eccezione."""
    csv_path = _write_synthetic_csv(tmp_path)
    out = tmp_path / "curve_none.png"
    accuracy_vs_vram_curves(csv_path, model_id="model_INESISTENTE",
                            miss_mode="precision-cascade", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_isoaccuracy_gain_vs_sparsity_creates_png(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path)
    out = tmp_path / "iso.png"
    isoaccuracy_gain_vs_sparsity(csv_path, drop_threshold=0.02, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_diagnostics_creates_pngs(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path)
    out_dir = tmp_path / "diag"
    diagnostics(csv_path, out_dir=str(out_dir))
    miss_png = out_dir / "diag_missrate_vs_ctx.png"
    kv_png = out_dir / "diag_vram_kv_vs_expert.png"
    assert miss_png.exists() and miss_png.stat().st_size > 0
    # VRAM_kv vs expert e' opzionale: presente perche' le colonne ci sono nel CSV sintetico.
    assert kv_png.exists() and kv_png.stat().st_size > 0


def test_verdict_structure_and_falsification(tmp_path):
    csv_path = _write_synthetic_csv(tmp_path)
    res = verdict(csv_path, drop_threshold=0.02)

    key_a = "model_A::precision-cascade"
    key_b = "model_B::precision-cascade"
    assert key_a in res and key_b in res

    a = res[key_a]
    b = res[key_b]

    # Struttura attesa.
    expected_keys = {
        "model_id", "miss_mode", "promoted_k_fraction", "n_ctx_tested",
        "ctx_lengths", "vram_gain_fraction", "max_drop_at_promoted", "failed",
    }
    assert expected_keys.issubset(a.keys())
    assert expected_keys.issubset(b.keys())

    # Modello A: il K piu' aggressivo (0.25) regge su tutte le ctx -> promosso a 0.25.
    assert a["promoted_k_fraction"] == 0.25
    assert a["failed"] is False
    assert a["n_ctx_tested"] == 3
    assert a["ctx_lengths"] == [1024, 4096, 16384]
    assert a["max_drop_at_promoted"] < 0.02
    # Guadagno VRAM: da 8 GiB (FULL) a 2 GiB (K=0.25) -> 75% di taglio.
    assert a["vram_gain_fraction"] == pytest.approx(0.75, abs=1e-6)

    # Modello B: K=0.25 FALSIFICATO (sfora a 16k) -> il K minimo promosso scende a 0.5.
    assert b["promoted_k_fraction"] == 0.5
    assert b["failed"] is False
    # A K=0.5 il guadagno e' da 8 GiB a 4 GiB -> 50%.
    assert b["vram_gain_fraction"] == pytest.approx(0.50, abs=1e-6)
    assert b["max_drop_at_promoted"] < 0.02


def test_verdict_tiny_threshold_only_full_holds(tmp_path):
    """Soglia irrealisticamente piccola -> regge solo FULL (drop=0): nessun TAGLIO promosso.

    FULL (k=1.0) ha drop 0 per costruzione, quindi resta sotto qualsiasi soglia: il
    verdetto lo promuove ma con guadagno VRAM nullo (e' il ground truth, non un taglio).
    Nessuna cella con K aggressivo (0.5/0.25) puo' essere promossa.
    """
    csv_path = _write_synthetic_csv(tmp_path)
    res = verdict(csv_path, drop_threshold=0.0001)
    assert res  # non vuoto
    for v in res.values():
        # Solo FULL regge: nessun taglio aggressivo.
        assert v["promoted_k_fraction"] == 1.0
        assert v["failed"] is False
        assert v["vram_gain_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_verdict_no_full_in_data_can_fail(tmp_path):
    """Senza FULL e con drop sempre >= soglia su una ctx -> regione FALLITA (failed=True)."""
    import pandas as pd

    base = _write_synthetic_csv(tmp_path)
    df = pd.read_csv(base)
    # Tieni solo K aggressivi del modello B (niente FULL); a 16k il drop di K=0.25 e' 5%
    # e di K=0.5 e' 0.5%: con soglia 0.4% nessuno dei due regge su tutte le ctx.
    sub = df[(df["model_id"] == "model_B") & (df["k_fraction"] != 1.0)]
    csv_path = str(tmp_path / "metrics_nofull.csv")
    sub.to_csv(csv_path, index=False)

    res = verdict(csv_path, drop_threshold=0.004)
    key_b = "model_B::precision-cascade"
    assert key_b in res
    assert res[key_b]["failed"] is True
    assert res[key_b]["promoted_k_fraction"] is None
    assert res[key_b]["vram_gain_fraction"] is None
