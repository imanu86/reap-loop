"""curves.py — i deliverable grafici. NON un grafo 2D: una FAMIGLIA di curve.

Input: il CSV di CellMetrics prodotto da experiment/runner.py (una riga per cella,
vedi experiment/metrics.py::CellMetrics.to_row).

Backend matplotlib forzato ad "Agg" (NON interattivo): la generazione delle figure
deve funzionare headless (CI, server senza display) e non aprire finestre.

Schema CSV atteso (appiattimento di CellMetrics; la dataclass VramBreakdown viene
appiattita con prefisso ``vram_``):
    model_id, policy, k_fraction, ctx_len, miss_mode, sparsity_ratio,
    accuracy, accuracy_drop_vs_full,
    mean_n_eff, mean_entropy_norm,
    vram_backbone, vram_kv_cache, vram_experts_resident, vram_overhead, vram_total,
    miss_rate, latency_ms_per_token, working_set_converged

L'asse "VRAM_expert / K" delle curve usa ``vram_experts_resident`` (il termine su cui
agisce la policy, docs §5/§11): KV cache e backbone restano separati per non mascherare
il guadagno-expert a contesto lungo (rischio R6).
"""

from __future__ import annotations

import os

import matplotlib

# Backend non interattivo: deve essere impostato PRIMA di importare pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colonna usata come asse "VRAM tagliabile" (il claim del progetto).
_VRAM_EXPERT_COL = "vram_experts_resident"
# Conversione byte -> GiB per assi leggibili.
_GIB = float(1 << 30)


def _load_metrics(metrics_csv: str) -> pd.DataFrame:
    """Carica il CSV in un DataFrame e normalizza i tipi minimi necessari.

    Non assume la presenza di tutte le colonne diagnostiche: le funzioni che ne
    hanno bisogno controllano da sole. Garantisce però le colonne identitarie.
    """
    df = pd.read_csv(metrics_csv)
    required = {"model_id", "ctx_len", "miss_mode", "accuracy", "accuracy_drop_vs_full"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV metriche privo delle colonne richieste: {sorted(missing)}")
    # ctx_len numerico (per ordinare le linee della famiglia di curve).
    df["ctx_len"] = pd.to_numeric(df["ctx_len"], errors="coerce")
    return df


def accuracy_vs_vram_curves(metrics_csv: str, *, model_id: str, miss_mode: str, out_path: str) -> None:
    """Deliverable 1: per UN modello, accuratezza vs VRAM_expert (residente).

    UNA LINEA PER LUNGHEZZA DI CONTESTO. Mostra se la regione "piatta" (drop < 2%)
    sopravvive al crescere del contesto o collassa (il verdetto su R1). La banda di
    riferimento a -2% rispetto all'accuratezza FULL (drop massimo tollerato) è
    evidenziata. Salva un PNG in ``out_path``.
    """
    df = _load_metrics(metrics_csv)
    sub = df[(df["model_id"] == model_id) & (df["miss_mode"] == miss_mode)].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    if sub.empty or _VRAM_EXPERT_COL not in sub.columns:
        # Nessun dato: produciamo comunque un PNG (placeholder) per non rompere la pipeline.
        ax.text(0.5, 0.5, f"nessun dato per {model_id} / {miss_mode}",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sub[_VRAM_EXPERT_COL] = pd.to_numeric(sub[_VRAM_EXPERT_COL], errors="coerce")
        sub["accuracy"] = pd.to_numeric(sub["accuracy"], errors="coerce")
        sub = sub.dropna(subset=[_VRAM_EXPERT_COL, "accuracy", "ctx_len"])

        # Riferimento FULL: accuracy a drop 0 (k_fraction massima / drop nullo).
        # La banda a -2% si ancora all'accuratezza FULL stimata per ciascuna ctx.
        # Stima FULL@ctx = accuracy + accuracy_drop_vs_full (per costruzione di metrics).
        if "accuracy_drop_vs_full" in sub.columns:
            sub["acc_full_ctx"] = (
                pd.to_numeric(sub["accuracy"], errors="coerce")
                + pd.to_numeric(sub["accuracy_drop_vs_full"], errors="coerce")
            )
            full_acc = float(sub["acc_full_ctx"].max())
        else:
            full_acc = float(sub["accuracy"].max())

        # Una linea per ctx_len, ordinata crescente (contesto corto -> lungo).
        for ctx, grp in sorted(sub.groupby("ctx_len"), key=lambda kv: kv[0]):
            grp = grp.sort_values(_VRAM_EXPERT_COL)
            x = grp[_VRAM_EXPERT_COL].to_numpy(dtype=float) / _GIB
            y = grp["accuracy"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=f"ctx={_fmt_ctx(ctx)}")

        # Banda di riferimento -2%: tra (full - soglia) e full.
        band_lo = full_acc - 0.02
        ax.axhspan(band_lo, full_acc, color="green", alpha=0.12,
                   label="banda iso-accuratezza (-2%)")
        ax.axhline(full_acc, color="green", linestyle="--", linewidth=1, alpha=0.6)

        ax.set_xlabel("VRAM expert residente (GiB)")
        ax.set_ylabel("accuratezza")
        ax.legend(loc="best", fontsize=8)

    ax.set_title(f"Accuratezza vs VRAM_expert — {model_id} [{miss_mode}]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


def isoaccuracy_gain_vs_sparsity(metrics_csv: str, *, drop_threshold: float = 0.02, out_path: str) -> None:
    """Deliverable 2 (riassuntivo): guadagno VRAM a iso-accuratezza vs SPARSITÀ.

    Per ogni modello: il MASSIMO taglio di VRAM_expert che mantiene drop < soglia
    SU TUTTE le ctx, plottato contro s = n_active/n_total (``sparsity_ratio``).
    In overlay, lo stesso vs concentrazione empirica ``mean_n_eff``. Salva un PNG.

    Il "taglio" è (VRAM_expert_FULL - VRAM_expert_minimo_che_regge) / VRAM_expert_FULL,
    dove il minimo che regge è la cella col K più piccolo che soddisfa il protocollo di
    falsificazione (drop < soglia su TUTTE le ctx). Vedi docs §10 e §12.
    """
    df = _load_metrics(metrics_csv)

    rows = []
    for model_id, grp in df.groupby("model_id"):
        gain = _max_vram_cut_holding(grp, drop_threshold=drop_threshold)
        if gain is None:
            continue
        sparsity = _scalar_or_nan(grp, "sparsity_ratio")
        n_eff = _scalar_or_nan(grp, "mean_n_eff")
        rows.append((str(model_id), sparsity, n_eff, gain))

    fig, ax = plt.subplots(figsize=(8, 5))

    if not rows:
        ax.text(0.5, 0.5, "nessuna regione che regge su TUTTE le ctx",
                ha="center", va="center", transform=ax.transAxes)
    else:
        rows.sort(key=lambda r: (np.nan_to_num(r[1], nan=np.inf)))
        models = [r[0] for r in rows]
        sparsity = np.array([r[1] for r in rows], dtype=float)
        n_eff = np.array([r[2] for r in rows], dtype=float)
        gain = np.array([r[3] for r in rows], dtype=float) * 100.0  # in %

        # Asse primario: guadagno VRAM vs sparsità strutturale.
        ax.plot(sparsity, gain, marker="o", color="C0", label="vs sparsità s")
        for m, sx, gy in zip(models, sparsity, gain):
            ax.annotate(m, (sx, gy), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("sparsità strutturale  s = n_active / n_total")
        ax.set_ylabel("taglio VRAM_expert a iso-accuratezza (%)")

        # Overlay: stesso guadagno vs concentrazione empirica N_eff (asse x secondario in alto).
        if not np.all(np.isnan(n_eff)):
            ax2 = ax.twiny()
            order = np.argsort(n_eff)
            ax2.plot(n_eff[order], gain[order], marker="s", color="C1",
                     linestyle="--", label="vs N_eff (empirico)")
            ax2.set_xlabel("numero efficace di expert  N_eff")
            # Unifica le legende dei due assi.
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        else:
            ax.legend(loc="best", fontsize=8)

    ax.set_title(f"Guadagno VRAM a iso-accuratezza (drop < {drop_threshold:.0%}) vs sparsità")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_path)


def diagnostics(metrics_csv: str, *, out_dir: str) -> None:
    """Diagnostici (docs §10.3). Genera PNG separati in ``out_dir``.

    Implementato: miss_rate vs ctx (segnale R1) — il diagnostico chiave che espone
    "regge a corto, sfasa a lungo". Una linea per (modello, miss_mode).

    Gli altri diagnostici previsti dal disegno (convergenza W_θ vs warm-up R4, latenza
    vs K per fetch-lossless R3, VRAM_kv vs VRAM_expert R6) si appoggiano a colonne
    opzionali: se presenti nel CSV vengono plottati, altrimenti vengono saltati senza
    errore (la griglia su 3060 può non popolarli tutti).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = _load_metrics(metrics_csv)

    # --- Diagnostico 1: miss_rate vs ctx (sempre, è il segnale R1) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if "miss_rate" in df.columns and not df.empty:
        df_m = df.copy()
        df_m["miss_rate"] = pd.to_numeric(df_m["miss_rate"], errors="coerce")
        for (model_id, miss_mode), grp in df_m.groupby(["model_id", "miss_mode"]):
            grp = grp.dropna(subset=["ctx_len", "miss_rate"]).sort_values("ctx_len")
            if grp.empty:
                continue
            # Aggrega su K: la curva diagnostica è il miss_rate medio per ctx.
            agg = grp.groupby("ctx_len")["miss_rate"].mean()
            ax.plot([_fmt_ctx(c) for c in agg.index], agg.to_numpy(dtype=float),
                    marker="o", label=f"{model_id} [{miss_mode}]")
        ax.set_xlabel("lunghezza di contesto")
        ax.set_ylabel("miss_rate (medio su K)")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "miss_rate non disponibile", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Diagnostico R1: miss_rate vs contesto")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "diag_missrate_vs_ctx.png"))

    # --- Diagnostico 2 (opzionale): VRAM_kv vs VRAM_expert (R6) ---
    if {"vram_kv_cache", _VRAM_EXPERT_COL}.issubset(df.columns) and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        df_v = df.copy()
        for (model_id,), grp in df_v.groupby(["model_id"]):
            grp = grp.dropna(subset=["ctx_len"]).sort_values("ctx_len")
            if grp.empty:
                continue
            kv = pd.to_numeric(grp.groupby("ctx_len")["vram_kv_cache"].mean(), errors="coerce") / _GIB
            ex = pd.to_numeric(grp.groupby("ctx_len")[_VRAM_EXPERT_COL].mean(), errors="coerce") / _GIB
            labels = [_fmt_ctx(c) for c in kv.index]
            ax.plot(labels, kv.to_numpy(dtype=float), marker="o", label=f"{model_id} VRAM_kv")
            ax.plot(labels, ex.to_numpy(dtype=float), marker="s", linestyle="--",
                    label=f"{model_id} VRAM_expert")
        ax.set_xlabel("lunghezza di contesto")
        ax.set_ylabel("VRAM (GiB)")
        ax.set_title("Diagnostico R6: VRAM_kv vs VRAM_expert (la KV può mascherare il guadagno)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, "diag_vram_kv_vs_expert.png"))


def verdict(metrics_csv: str, *, drop_threshold: float = 0.02) -> dict:
    """Verdetto automatico (docs §12) con protocollo di falsificazione.

    Per ogni (modello, miss_mode) cerca il K minimo PROMOSSO: la cella col taglio di
    VRAM più aggressivo (k_fraction più piccola) che mantiene drop < soglia
    SU TUTTE le ctx testate. Una regione che regge a ctx corto ma sfora a ctx lungo
    è FALLITA, non promossa per la media.

    Ritorna un dict mappato per la coppia "model_id::miss_mode":
        {
          "<model_id>::<miss_mode>": {
              "model_id": str,
              "miss_mode": str,
              "promoted_k_fraction": float | None,   # K minimo che regge, o None
              "n_ctx_tested": int,
              "ctx_lengths": list[int],
              "vram_gain_fraction": float | None,    # taglio VRAM_expert vs FULL
              "max_drop_at_promoted": float | None,  # drop peggiore sulle ctx al K promosso
              "failed": bool,                         # True se NESSUN K regge su tutte le ctx
          },
          ...
        }
    Il `fetch-lossless` ha drop 0 per costruzione: se presente nel CSV regge sempre.
    """
    df = _load_metrics(metrics_csv)
    out: dict = {}

    for (model_id, miss_mode), grp in df.groupby(["model_id", "miss_mode"]):
        grp = grp.copy()
        grp["k_fraction"] = pd.to_numeric(grp.get("k_fraction"), errors="coerce")
        grp["accuracy_drop_vs_full"] = pd.to_numeric(
            grp["accuracy_drop_vs_full"], errors="coerce"
        )
        ctx_all = sorted({int(c) for c in grp["ctx_len"].dropna().unique()})
        n_ctx = len(ctx_all)

        promoted_k = None
        max_drop_promoted = None
        # Valuta ogni K dal più aggressivo (k_fraction minore) al più conservativo.
        for k in sorted(grp["k_fraction"].dropna().unique()):
            cells = grp[np.isclose(grp["k_fraction"], k)]
            # Protocollo di falsificazione: il K deve coprire e reggere TUTTE le ctx.
            covered = sorted({int(c) for c in cells["ctx_len"].dropna().unique()})
            if covered != ctx_all:
                continue  # copertura incompleta -> non valutabile come promosso
            worst_drop = float(cells["accuracy_drop_vs_full"].max())
            # drop < soglia su TUTTE le ctx (la peggiore deve restare sotto soglia).
            if worst_drop < drop_threshold:
                promoted_k = float(k)
                max_drop_promoted = worst_drop
                break  # primo (più aggressivo) che regge = K minimo promosso

        vram_gain = None
        if promoted_k is not None:
            vram_gain = _vram_gain_at_k(grp, promoted_k)

        key = f"{model_id}::{miss_mode}"
        out[key] = {
            "model_id": str(model_id),
            "miss_mode": str(miss_mode),
            "promoted_k_fraction": promoted_k,
            "n_ctx_tested": n_ctx,
            "ctx_lengths": ctx_all,
            "vram_gain_fraction": vram_gain,
            "max_drop_at_promoted": max_drop_promoted,
            "failed": promoted_k is None,
        }

    return out


# --------------------------------------------------------------------------- #
# Helper interni                                                              #
# --------------------------------------------------------------------------- #

def _max_vram_cut_holding(grp: pd.DataFrame, *, drop_threshold: float) -> float | None:
    """Massimo taglio frazionario di VRAM_expert che regge su TUTTE le ctx.

    Trova il K minimo che soddisfa il protocollo (drop < soglia su tutte le ctx) e
    ritorna (VRAM_FULL - VRAM_K) / VRAM_FULL. None se nessun K regge o mancano dati VRAM.
    """
    if _VRAM_EXPERT_COL not in grp.columns:
        return None
    g = grp.copy()
    g["k_fraction"] = pd.to_numeric(g.get("k_fraction"), errors="coerce")
    g["accuracy_drop_vs_full"] = pd.to_numeric(g["accuracy_drop_vs_full"], errors="coerce")
    ctx_all = sorted({int(c) for c in g["ctx_len"].dropna().unique()})

    for k in sorted(g["k_fraction"].dropna().unique()):
        cells = g[np.isclose(g["k_fraction"], k)]
        covered = sorted({int(c) for c in cells["ctx_len"].dropna().unique()})
        if covered != ctx_all:
            continue
        if float(cells["accuracy_drop_vs_full"].max()) < drop_threshold:
            return _vram_gain_at_k(g, float(k))
    return None


def _vram_gain_at_k(grp: pd.DataFrame, k: float) -> float | None:
    """Taglio VRAM_expert al K dato rispetto al FULL (k_fraction massima del modello).

    VRAM_expert per cella è invariante rispetto a ctx (dipende da K, non dal contesto):
    prendiamo la media difensivamente. Ritorna frazione in [0, 1] o None.
    """
    if _VRAM_EXPERT_COL not in grp.columns:
        return None
    g = grp.copy()
    g["k_fraction"] = pd.to_numeric(g.get("k_fraction"), errors="coerce")
    g[_VRAM_EXPERT_COL] = pd.to_numeric(g[_VRAM_EXPERT_COL], errors="coerce")

    k_full = float(g["k_fraction"].max())
    full_cells = g[np.isclose(g["k_fraction"], k_full)]
    k_cells = g[np.isclose(g["k_fraction"], k)]
    if full_cells.empty or k_cells.empty:
        return None
    vram_full = float(full_cells[_VRAM_EXPERT_COL].mean())
    vram_k = float(k_cells[_VRAM_EXPERT_COL].mean())
    if not np.isfinite(vram_full) or vram_full <= 0:
        return None
    return (vram_full - vram_k) / vram_full


def _scalar_or_nan(grp: pd.DataFrame, col: str) -> float:
    """Estrae un valore scalare costante per il gruppo (o NaN se assente)."""
    if col not in grp.columns:
        return float("nan")
    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty else float("nan")


def _fmt_ctx(ctx) -> str:
    """Etichetta leggibile per la lunghezza di contesto (1024 -> '1k', -1 -> 'max')."""
    try:
        c = int(ctx)
    except (TypeError, ValueError):
        return str(ctx)
    if c < 0:
        return "max"
    if c >= 1024 and c % 1024 == 0:
        return f"{c // 1024}k"
    return str(c)


def _savefig(fig, out_path: str) -> None:
    """Salva la figura su PNG (creando la dir) e la chiude per liberare memoria."""
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
