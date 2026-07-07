"""msc.enforce — enforcement hard-drop del routing MoE.

Dato un insieme di expert RESIDENTI per layer, forza il router a instradare i token
SOLO verso quegli expert (gli altri vengono soppressi). L'enforcement avviene via
forward-hook che RISCRIVE l'output del router, senza patchare transformers.

Backend disponibili:
    olmoe — allenai/OLMoE-1B-7B-0924(-Instruct): 64 expert, top-8, 16 layer.
            Router = OlmoeTopKRouter su model.model.layers[i].mlp.gate, che restituisce
            la tupla (router_logits, router_scores, router_indices).
"""

from __future__ import annotations

from .olmoe import HardDropHandle, attach_hard_drop

__all__ = ["attach_hard_drop", "HardDropHandle"]
