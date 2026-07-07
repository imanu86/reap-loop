"""full.py — policy FULL: nessuna eviction. GROUND TRUTH di accuratezza.

Tutti gli expert disponibili a precisione nativa. Su 12 GB:
  - Granite-3B/1B entrano a fp16 -> FULL nativo pulito
  - OLMoE/DeepSeek/Qwen non entrano -> FULL via CPU-offload (accelerate) o riferimento int8 uniforme
(vedi docs/02_models.md §5). K=100% equivale a questa policy.
"""

from __future__ import annotations

from msc.policies.base import PolicyDecision, ResidencyPolicy


class FullPolicy(ResidencyPolicy):
    name = "FULL"

    def decide(self, *, trace_reader, k_budget: float, model_spec) -> PolicyDecision:
        """Tutti gli expert residenti (ignora k_budget e la traccia).

        Ogni layer MoE tiene residente l'intero pool di expert -> committed_fraction = 1.0.
        Questa è la ground-truth di accuratezza (asse K = 100%).
        """
        n_total = int(model_spec.n_total_experts)
        n_layers = int(model_spec.n_layers)

        # Set completo di expert, identico per ogni layer.
        all_experts = set(range(n_total))
        per_layer_resident = {layer: set(all_experts) for layer in range(n_layers)}

        return PolicyDecision(
            per_layer_resident=per_layer_resident,
            committed_fraction=1.0,
            rationale=(
                f"FULL: tutti i {n_total} expert residenti su {n_layers} layer "
                f"(ground truth, k_budget ignorato)"
            ),
        )
