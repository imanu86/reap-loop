"""aggressive_commit.py — LA NOSTRA policy.

warm-up -> stima working set per-sessione (per copertura θ) -> COMMIT dei residenti -> resto
gestito dal miss_mode (asse D, deciso a livello di ResidencyManager).

Delta verificato (docs/01_positioning.md §6): MoE-Infinity stima già un working set per-sequenza ma
è LOSSLESS; HOBBIT è lossy ma per-token e validato solo a contesto corto. Il nostro contributo è la
congiunzione: commit per-sessione + degrado deliberato del complemento + misura della superficie
accuratezza-vs-VRAM in funzione di sparsità e lunghezza di contesto.
"""

from __future__ import annotations

import math

from msc.policies.base import PolicyDecision, ResidencyPolicy


def _max_experts_for_budget(n_total: int, k_budget: float) -> int:
    """Tetto di expert residenti per layer imposto dal budget K (frazione). Clamp in [1, n_total]."""
    raw = math.ceil(k_budget * n_total)
    return max(1, min(n_total, raw))


class AggressiveCommitPolicy(ResidencyPolicy):
    name = "AGGRESSIVE-COMMIT"

    def __init__(self, coverage_theta: float = 0.95) -> None:
        # coverage_theta: copertura cumulata target del working set (vedi workingset/estimator.py).
        self.coverage_theta = coverage_theta

    def decide(self, *, trace_reader, k_budget: float, model_spec) -> PolicyDecision:
        """Stima il working set dalla traccia di WARM-UP della sessione e committa.

        Il set residente = expert fino a copertura θ, troncato a k_budget se più stringente.
        Delega a msc.workingset.estimator.estimate_working_set.

        ATTENZIONE (rischio R1): il warm-up qui è tipicamente a contesto CORTO; il commit verrà poi
        valutato a contesto crescente. La divergenza miss_rate(ctx) è il segnale-chiave da osservare.
        """
        # Import a livello di funzione: 'import msc.policies...' deve funzionare senza torch e il
        # riferimento al MODULO permette il monkeypatch di estimate_working_set nei test.
        from msc.workingset import estimator as ws_estimator

        n_total = int(model_spec.n_total_experts)
        n_layers = int(model_spec.n_layers)
        max_keep = _max_experts_for_budget(n_total, k_budget)

        # La stima fa il lavoro pesante: istogramma -> curva di copertura -> W_θ, con troncamento
        # opzionale al budget. Passiamo comunque k_budget cosi l'estimator può troncare a monte.
        estimate = ws_estimator.estimate_working_set(
            trace_reader, theta=self.coverage_theta, k_budget=k_budget
        )

        # Tronchiamo difensivamente a max_keep per layer: il commit non deve mai sforare il budget K
        # anche se l'estimator restituisse un working set più ampio (es. theta alto, routing diffuso).
        per_layer_resident: dict[int, set[int]] = {}
        total_resident = 0
        for layer in range(n_layers):
            ws = estimate.per_layer_working_set.get(layer, ())
            # ws è ordinato per frequenza decrescente: i primi sono i più importanti.
            kept = tuple(ws)[:max_keep]
            resident = {int(e) for e in kept}
            # Garantisce almeno un expert residente per layer (evita layer vuoto).
            if not resident:
                resident = {0}
            per_layer_resident[layer] = resident
            total_resident += len(resident)

        # committed_fraction effettiva = media (su layer) della frazione residente, post-troncamento.
        if n_layers > 0 and n_total > 0:
            committed_fraction = total_resident / (n_layers * n_total)
        else:
            committed_fraction = 0.0

        return PolicyDecision(
            per_layer_resident=per_layer_resident,
            committed_fraction=committed_fraction,
            rationale=(
                f"AGGRESSIVE-COMMIT: working set per-sessione a θ={self.coverage_theta:.2f}, "
                f"troncato a k_budget={k_budget:.3f} (<= {max_keep}/{n_total} per layer), "
                f"frazione residente={committed_fraction:.3f}"
            ),
        )
