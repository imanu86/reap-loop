"""03_run_grid.py — esegue la griglia 4D e scrive CellMetrics.

CLI:
    python scripts/03_run_grid.py --config configs/grid/sweep.yaml --out runs/grid/

Riprendibile: salta le celle già completate in --out (le griglie sono lunghe su 3060).
Vedi msc.experiment.runner.run_grid.

L'import del modulo è torch-free (così `--help` e la pianificazione della griglia funzionano
ovunque). L'esecuzione effettiva di ogni cella richiede torch + un modello: se mancano, run_grid lo
segnala con un errore chiaro alla prima cella (gestito qui senza traceback). Con `--dry-run` si
elencano soltanto le celle pianificate (utile per stimare la cardinalità senza GPU).
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Esegue (o pianifica) la griglia 4D dell'esperimento.")
    p.add_argument("--config", required=True, help="YAML della griglia (configs/grid/sweep.yaml).")
    p.add_argument("--out", default="runs/grid/", help="Directory dei risultati per-cella + CSV.")
    p.add_argument("--dry-run", action="store_true",
                   help="Elenca le celle pianificate (con le scorciatoie §9) senza eseguirle.")
    args = p.parse_args()

    from msc.experiment.runner import load_grid_spec, iter_cells, run_grid

    spec = load_grid_spec(args.config)
    cells = list(iter_cells(spec))
    print(f"[grid] modelli={len(spec.model_ids)} K={spec.k_fractions} ctx={spec.ctx_lengths} "
          f"miss_modes={[m.value for m in spec.miss_modes]} policies={spec.policies}")
    print(f"[grid] celle pianificate (dopo scorciatoie §9): {len(cells)}")

    if args.dry_run:
        for c in cells:
            print(f"  - {c.model_id} | {c.policy} | K={c.k_fraction:g} | "
                  f"ctx={c.ctx_len} | {c.miss_mode.value}")
        return 0

    # Esecuzione reale: richiede torch + modello (gpu_seam in _execute_cell).
    try:
        executed = run_grid(spec, out_dir=args.out)
    except RuntimeError as exc:
        print(f"[grid] esecuzione non possibile: {exc}")
        return 2

    print(f"[grid] celle eseguite in questa run: {len(executed)} (le altre erano già fatte). "
          f"Risultati in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
