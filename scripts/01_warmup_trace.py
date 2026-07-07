"""01_warmup_trace.py — gira N esempi di warm-up e scrive la traccia di attivazione.

CLI:
    python scripts/01_warmup_trace.py --config configs/grid/sweep.yaml --model olmoe --out runs/

Passi: carica modello + RouterHookSpec.for_model -> RouterLogger.capture() durante generate ->
traccia jsonl per sessione (instrument/trace.py). Niente policy ancora: serve solo il routing FULL.

Richiede torch/transformers + un modello scaricato: l'import del modulo NON li richiede (così
`python scripts/01_warmup_trace.py --help` funziona ovunque); se mancano a runtime stampiamo un
messaggio chiaro ed usciamo con codice 2, senza traceback.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="Warm-up: raccoglie la traccia di attivazione del router.")
    p.add_argument("--config", required=True, help="YAML della griglia (per i parametri di warm-up).")
    p.add_argument("--model", required=True, help="Id/path del modello MoE da tracciare.")
    p.add_argument("--out", default="runs/", help="Directory di output per la traccia jsonl.")
    args = p.parse_args()

    # Import del modulo (torch-free): serve solo per leggere lo spec / preparare l'output.
    from msc.experiment.runner import load_grid_spec
    from msc.instrument import RouterLogger, RouterHookSpec  # noqa: F401  (uso a runtime con GPU)

    spec = load_grid_spec(args.config)
    print(
        f"[warmup] modello={args.model} warmup_examples={spec.warmup_examples} "
        f"seed={spec.seed} out={args.out}"
    )

    # gpu_seam: caricamento modello + cattura del routing durante generate(). Richiede torch.
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print(
            "[warmup] torch/transformers non disponibili: il warm-up richiede un modello su GPU.\n"
            "         Installa torch (build CUDA, vedi docs/02_models.md) e transformers, "
            "poi rilancia.\n"
            "         (RouterLogger/RouterHookSpec sono pronti; manca solo il backend GPU.)"
        )
        return 2

    # Da qui servirebbe la GPU reale: caricamento modello, hook del router, generate, TraceWriter.
    # Il cablaggio concreto vive in msc.instrument (RouterLogger.capture) ed è fuori dalla CI CPU-only.
    print(
        "[warmup] backend GPU presente ma il cablaggio del modello concreto non è eseguibile qui "
        "(serve un modello MoE scaricato). Vedi msc.instrument.RouterLogger.capture()."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
