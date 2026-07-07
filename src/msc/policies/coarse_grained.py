"""coarse_grained.py — baseline COARSE: top-K expert più frequenti GLOBALMENTE.

⚠️ Posizionamento (verificato, docs/01_positioning.md §2): questa è la baseline stile **BrainStorm**
(popolarità globale), NON "stile MoE-Infinity". MoE-Infinity rifiuta esplicitamente la popolarità
globale e fa matching per-sequenza lossless -> è semmai il cugino lossless di AGGRESSIVE-COMMIT.

Differenza chiave da AGGRESSIVE: la frequenza è aggregata su un corpus globale/statico, non stimata
per-sessione. Per workload ad alta località ci aspettiamo che AGGRESSIVE batta COARSE perché il
working set del task corrente è più piccolo e più predittivo della popolarità globale.
"""

from __future__ import annotations

import json
import math

from msc.policies.base import PolicyDecision, ResidencyPolicy


def _experts_to_keep(n_total: int, k_budget: float) -> int:
    """Numero di expert residenti per layer dato il budget K (frazione).

    Tronca con ceil per non scendere sotto la copertura richiesta, clamp in [1, n_total].
    """
    raw = math.ceil(k_budget * n_total)
    return max(1, min(n_total, raw))


def _normalize_freq_map(raw_freq: dict) -> dict[int, dict[int, float]]:
    """Normalizza l'istogramma globale in {layer: {expert_id: freq}}.

    Accetta vari formati (i path JSON possono avere chiavi stringa):
      - {layer: {expert_id: freq}}
      - {layer: [freq_0, freq_1, ...]}   (lista indicizzata per expert_id)
      - {layer: [[expert_id, freq], ...]} (coppie)
    """
    out: dict[int, dict[int, float]] = {}
    for layer_key, per_layer in raw_freq.items():
        layer = int(layer_key)
        freq_map: dict[int, float] = {}
        if isinstance(per_layer, dict):
            for e_key, f in per_layer.items():
                freq_map[int(e_key)] = float(f)
        elif isinstance(per_layer, (list, tuple)):
            # Distingue lista di coppie (id, freq) da lista di sole frequenze.
            if per_layer and isinstance(per_layer[0], (list, tuple)):
                for pair in per_layer:
                    e_id, f = pair
                    freq_map[int(e_id)] = float(f)
            else:
                for e_id, f in enumerate(per_layer):
                    freq_map[int(e_id)] = float(f)
        else:
            raise ValueError(f"Formato frequenza globale non riconosciuto per layer {layer}")
        out[layer] = freq_map
    return out


class CoarseGrainedPolicy(ResidencyPolicy):
    name = "COARSE"

    def __init__(self, global_freq_path: str | None = None, global_freq: dict | None = None) -> None:
        # global_freq_path: istogramma d'uso pre-calcolato su corpus globale (statico).
        # global_freq: in alternativa, l'istogramma già in memoria (utile nei test, evita I/O).
        self._global_freq_path = global_freq_path
        self._global_freq = global_freq

    def _load_global_freq(self) -> dict[int, dict[int, float]]:
        """Carica la frequenza globale (da memoria o da file JSON). Sempre da corpus statico."""
        if self._global_freq is not None:
            return _normalize_freq_map(self._global_freq)
        if self._global_freq_path is None:
            raise ValueError(
                "COARSE richiede una frequenza globale: passa global_freq o global_freq_path"
            )
        with open(self._global_freq_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return _normalize_freq_map(raw)

    def decide(self, *, trace_reader, k_budget: float, model_spec) -> PolicyDecision:
        """Tiene i top-(k_budget) expert per frequenza GLOBALE, per layer.

        Nota: NON usa la traccia di warm-up della sessione corrente (quello è il delta di AGGRESSIVE);
        usa la frequenza globale pre-calcolata. `trace_reader` è deliberatamente ignorato.
        """
        n_total = int(model_spec.n_total_experts)
        n_layers = int(model_spec.n_layers)
        keep = _experts_to_keep(n_total, k_budget)

        global_freq = self._load_global_freq()

        per_layer_resident: dict[int, set[int]] = {}
        for layer in range(n_layers):
            freq_map = global_freq.get(layer, {})
            # Ordina per frequenza decrescente; a parità, per expert_id crescente (deterministico).
            ranked = sorted(freq_map.items(), key=lambda kv: (-kv[1], kv[0]))
            resident = {e_id for e_id, _ in ranked[:keep]}

            # Fallback: se la frequenza globale non copre abbastanza expert per il budget,
            # completa con gli id più bassi non ancora presenti (deterministico).
            if len(resident) < keep:
                for e_id in range(n_total):
                    if len(resident) >= keep:
                        break
                    resident.add(e_id)
            per_layer_resident[layer] = resident

        committed_fraction = keep / n_total

        return PolicyDecision(
            per_layer_resident=per_layer_resident,
            committed_fraction=committed_fraction,
            rationale=(
                f"COARSE: top-{keep}/{n_total} per frequenza GLOBALE (stile BrainStorm), "
                f"k_budget={k_budget:.3f}, frazione residente={committed_fraction:.3f}"
            ),
        )
