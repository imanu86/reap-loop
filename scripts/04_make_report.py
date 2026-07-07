"""04_make_report.py — dalla griglia ai deliverable grafici + verdetto.

CLI:
    python scripts/04_make_report.py --metrics runs/grid/metrics.csv --out report/

Produce: famiglia di curve accuratezza-vs-VRAM (una per modello/miss_mode, linea per ctx),
grafico riassuntivo iso-accuratezza-vs-sparsità, diagnostici, e il verdetto automatico (docs §12).
Vedi msc.report.curves.

Torch-free: lavora sul CSV di CellMetrics. matplotlib è forzato ad "Agg" dentro msc.report.curves,
quindi gira headless.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    p = argparse.ArgumentParser(description="Genera la famiglia di curve + il verdetto dalla griglia.")
    p.add_argument("--metrics", required=True, help="CSV di CellMetrics (output di run_grid).")
    p.add_argument("--out", default="report/", help="Directory di output per le figure + il verdetto.")
    p.add_argument("--drop-threshold", type=float, default=0.02,
                   help="Soglia di drop accuratezza per la promozione (docs §12).")
    args = p.parse_args()

    from msc.report import (
        accuracy_vs_vram_curves,
        isoaccuracy_gain_vs_sparsity,
        diagnostics,
        verdict,
    )

    # Servono i pivot (modello × miss_mode) per la famiglia di curve: li leggiamo dal CSV.
    import pandas as pd

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.metrics)

    # --- Deliverable 1: una curva accuratezza-vs-VRAM per ciascun (modello, miss_mode) ---
    pairs = df[["model_id", "miss_mode"]].drop_duplicates().itertuples(index=False)
    n_curves = 0
    for model_id, miss_mode in pairs:
        fname = f"acc_vs_vram__{_safe(str(model_id))}__{_safe(str(miss_mode))}.png"
        accuracy_vs_vram_curves(
            args.metrics, model_id=str(model_id), miss_mode=str(miss_mode),
            out_path=os.path.join(args.out, fname),
        )
        n_curves += 1

    # --- Deliverable 2 (riassuntivo): guadagno VRAM a iso-accuratezza vs sparsità ---
    isoaccuracy_gain_vs_sparsity(
        args.metrics, drop_threshold=args.drop_threshold,
        out_path=os.path.join(args.out, "isoaccuracy_gain_vs_sparsity.png"),
    )

    # --- Diagnostici (miss_rate vs ctx, ecc.) ---
    diagnostics(args.metrics, out_dir=os.path.join(args.out, "diagnostics"))

    # --- Verdetto automatico (docs §12) ---
    verdicts = verdict(args.metrics, drop_threshold=args.drop_threshold)
    verdict_path = os.path.join(args.out, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as fh:
        json.dump(verdicts, fh, ensure_ascii=False, indent=2)

    promoted = [v for v in verdicts.values() if not v.get("failed", True)]
    print(f"[report] curve accuratezza-vs-VRAM: {n_curves}")
    print(f"[report] verdetto: {len(promoted)}/{len(verdicts)} regioni PROMOSSE (drop < "
          f"{args.drop_threshold:.0%} su TUTTE le ctx)")
    for key, v in verdicts.items():
        if v.get("failed", True):
            print(f"  - {key}: FALLITA")
        else:
            print(f"  - {key}: PROMOSSA @K={v['promoted_k_fraction']:g} "
                  f"(taglio VRAM={_pct(v.get('vram_gain_fraction'))}, "
                  f"drop max={_pct(v.get('max_drop_at_promoted'))})")
    print(f"[report] figure + verdict.json scritti in {args.out}")
    return 0


def _safe(s: str) -> str:
    """Rende una stringa filesystem-safe per i nomi dei PNG."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def _pct(x) -> str:
    """Formatta una frazione in percentuale, o '-' se None."""
    return f"{x:.1%}" if isinstance(x, (int, float)) else "-"


if __name__ == "__main__":
    raise SystemExit(main())
