"""base.py — interfaccia comune delle policy di residenza.

Una policy MAPPA (traccia di warm-up, budget K) -> insieme di expert residenti per layer.
È deliberatamente AGNOSTICA al miss_mode (asse D): cosa fare sui non-residenti è responsabilità del
ResidencyManager/MissHandler. Così le 3 policy × 3 miss_mode si combinano ortogonalmente.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    """Output di una policy: chi resta residente, per layer."""

    per_layer_resident: dict      # layer -> set[expert_id]
    committed_fraction: float     # frazione media residente (per la contabilità VRAM / asse K)
    rationale: str                # breve descrizione (per il logging dei run)


class ResidencyPolicy(abc.ABC):
    """Interfaccia comune. `decide()` è puro (no side-effect sul modello)."""

    name: str

    @abc.abstractmethod
    def decide(self, *, trace_reader, k_budget: float, model_spec) -> PolicyDecision:
        """Decide il set residente. `k_budget` = frazione di expert tenibili (asse K)."""
        raise NotImplementedError
