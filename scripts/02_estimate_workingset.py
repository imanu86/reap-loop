"""02_estimate_workingset.py — dalla traccia: working set per copertura + concentrazione.

CLI:
    python scripts/02_estimate_workingset.py --trace runs/<id>/trace.jsonl --theta 0.95

Stampa/serializza: W_θ per layer, entropia/Gini/N_eff per layer, e la curva di convergenza
(W_θ vs token di warm-up) per controllare il rischio R4. Vedi msc.workingset.estimator.

Questo step è interamente torch-free (lavora sulla traccia jsonl), quindi NON serve la GPU.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    p = argparse.ArgumentParser(description="Stima working set + concentrazione da una traccia jsonl.")
    p.add_argument("--trace", required=True, help="Traccia jsonl (output di 01_warmup_trace.py).")
    p.add_argument("--theta", type=float, default=0.95, help="Soglia di copertura cumulata.")
    p.add_argument("--k-budget", type=float, default=None,
                   help="Frazione massima di expert residenti per layer (asse K). Default: solo θ.")
    p.add_argument("--out", default=None, help="Se dato, scrive il riepilogo JSON in questo path.")
    args = p.parse_args()

    from msc.instrument.trace import TraceReader
    from msc.workingset import estimate_working_set

    reader = TraceReader(args.trace)
    est = estimate_working_set(reader, theta=args.theta, k_budget=args.k_budget)

    # Riepilogo serializzabile: W_θ per layer + concentrazione per layer + frazione committata.
    per_layer = {}
    for layer, ws in sorted(est.per_layer_working_set.items()):
        conc = est.per_layer_concentration.get(layer)
        per_layer[str(layer)] = {
            "working_set": list(ws),
            "working_set_size": len(ws),
            "n_total": conc.n_total if conc else None,
            "entropy_norm": conc.entropy_norm if conc else None,
            "gini": conc.gini if conc else None,
            "n_eff": conc.n_eff if conc else None,
        }

    summary = {
        "session_id": est.session_id,
        "model_id": est.model_id,
        "theta": est.theta,
        "committed_fraction": est.committed_fraction,
        "per_layer": per_layer,
    }

    print(f"[workingset] session={est.session_id or '?'} model={est.model_id or '?'} "
          f"theta={est.theta} committed_fraction={est.committed_fraction:.4f} "
          f"layers={len(per_layer)}")
    for layer, info in per_layer.items():
        print(f"  layer {layer}: |W_θ|={info['working_set_size']} "
              f"N_eff={info['n_eff']:.2f} H={info['entropy_norm']:.3f} Gini={info['gini']:.3f}"
              if info["n_eff"] is not None else f"  layer {layer}: (vuoto)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"[workingset] riepilogo scritto in {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
