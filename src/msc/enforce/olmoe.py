"""msc.enforce.olmoe — enforcement hard-drop per OLMoE via forward-hook sul router.

Modello target: allenai/OLMoE-1B-7B-0924(-Instruct) — 64 expert, top-8, 16 layer.

Strategia (NIENTE patch a transformers):
    Registra un forward_hook su ciascun ``model.model.layers[i].mlp.gate``
    (un OlmoeTopKRouter). Il router, dato hidden_states, restituisce la tupla
        (router_logits[n_token, 64], router_scores[n_token, 8], router_indices[n_token, 8])
    e il blocco MoE (OlmoeSparseMoeBlock.forward) usa SOLO scores e indices per
    dispatchare ai relativi expert. Riscrivendo l'output del router cambiamo dunque
    quali expert vengono effettivamente eseguiti.

    Il hook:
      1. prende router_logits [n_token, 64];
      2. mette a -inf le colonne degli expert NON residenti per quel layer;
      3. ricalcola la top-k (k = num_experts_per_tok, ma capata a |resident| se piu
         piccolo) con la STESSA pipeline OLMoE: softmax-then-topk, norm_topk_prob=False;
      4. restituisce una tupla nello STESSO formato/ordine dell'originale
         (logits mascherati, nuovi scores, nuovi indici), cosi che il blocco MoE
         instradi solo verso gli expert residenti.

    I calcoli del top-k avvengono in float32 (stabilita del softmax/argsort) e poi
    si ricasta ai dtype originali; device preservato (gestisce bf16 + cuda/offload).
"""

from __future__ import annotations

import torch
from torch import nn


# Sentinella per "nessun filtro" (layer con tutti gli expert residenti): in quel caso
# il hook diventa un no-op e restituisce l'output originale invariato, cosi da garantire
# l'identita bit-a-bit col baseline senza hook.
class HardDropHandle:
    """Handle restituito da ``attach_hard_drop``; chiama ``.remove()`` per staccare tutti i hook."""

    def __init__(self, handles: list[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles = handles
        self._removed = False

    def remove(self) -> None:
        """Stacca tutti i forward_hook registrati. Idempotente."""
        if self._removed:
            return
        for h in self._handles:
            h.remove()
        self._handles = []
        self._removed = True

    def __len__(self) -> int:
        return len(self._handles)

    def __enter__(self) -> "HardDropHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def _make_hook(resident: torch.Tensor, top_k: int):
    """Costruisce il forward_hook per un layer, dato il tensore degli indici residenti.

    Args:
        resident: LongTensor 1D con gli ID degli expert residenti per questo layer.
        top_k:    num_experts_per_tok del modello (es. 8).
    """
    # k effettivo: se i residenti sono meno di top_k, non possiamo selezionarne di piu.
    k_eff = int(min(top_k, resident.numel()))

    def hook(module: nn.Module, inputs, output):
        # output originale del router: (router_logits, router_scores, router_indices)
        router_logits, _orig_scores, _orig_indices = output
        num_experts = router_logits.shape[-1]

        # No-op se tutti gli expert sono residenti: restituiamo l'output invariato
        # (identita bit-a-bit col baseline). Il confronto e' su device/dtype del tensore.
        if resident.numel() >= num_experts:
            return output

        orig_dtype = router_logits.dtype
        device = router_logits.device

        # Maschera booleana [num_experts]: True = residente. Costruita sul device giusto.
        res_idx = resident.to(device=device, dtype=torch.long)
        keep_mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        keep_mask[res_idx] = True

        # Lavoriamo in float32 per stabilita di softmax/topk, poi ricastiamo.
        logits_f = router_logits.float()
        # Colonne dei NON residenti -> -inf (saranno ~0 dopo softmax e mai selezionate).
        neg_inf = torch.finfo(torch.float32).min
        masked_logits = logits_f.masked_fill(~keep_mask.unsqueeze(0), neg_inf)

        # Pipeline OLMoE: softmax su TUTTI i logit (mascherati), poi topk sulle probabilita.
        probs = torch.softmax(masked_logits, dim=-1)  # [n_token, num_experts]
        scores, indices = torch.topk(probs, k_eff, dim=-1)  # [n_token, k_eff]
        # norm_topk_prob=False per OLMoE-0924 => NON rinormalizziamo gli scores.

        # Ricast al formato originale: logits mascherati nel dtype del router,
        # scores nel dtype del router, indici long.
        new_logits = masked_logits.to(orig_dtype)
        new_scores = scores.to(orig_dtype)
        new_indices = indices.to(torch.long)

        return (new_logits, new_scores, new_indices)

    return hook


def attach_hard_drop(
    model: nn.Module,
    resident_by_layer: dict[int, set[int]],
) -> HardDropHandle:
    """Registra l'enforcement hard-drop sul router di OLMoE.

    Per ogni layer indicato in ``resident_by_layer`` registra un forward_hook su
    ``model.model.layers[i].mlp.gate`` che forza il routing dei token SOLO verso gli
    expert residenti di quel layer (gli altri soppressi via -inf sui logit). I layer
    non presenti nel dict restano col routing originale (nessun hook).

    Args:
        model: un OlmoeForCausalLM caricato (anche in offload/bf16/cuda).
        resident_by_layer: mappa layer_idx -> insieme di expert-ID residenti per quel layer.
            Se l'insieme contiene tutti i 64 expert, il hook su quel layer e' un no-op
            (output del router invariato, identita col baseline).

    Returns:
        HardDropHandle: oggetto con ``.remove()`` per staccare tutti i hook. Supporta
        anche l'uso come context manager (``with attach_hard_drop(...):``).

    Note:
        Il top_k effettivo per layer e' ``min(num_experts_per_tok, |resident|)``: se i
        residenti sono meno di top_k, si instrada verso tutti i residenti disponibili.
    """
    top_k = int(model.config.num_experts_per_tok)
    num_experts = int(model.config.num_experts)
    layers = model.model.layers

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx, experts in resident_by_layer.items():
        if not 0 <= layer_idx < len(layers):
            raise IndexError(f"layer_idx {layer_idx} fuori range [0, {len(layers)})")
        if len(experts) == 0:
            raise ValueError(f"layer {layer_idx}: insieme di expert residenti vuoto")
        bad = [e for e in experts if not 0 <= e < num_experts]
        if bad:
            raise ValueError(f"layer {layer_idx}: expert-ID fuori range [0, {num_experts}): {bad}")

        resident = torch.tensor(sorted(experts), dtype=torch.long)
        gate = layers[layer_idx].mlp.gate
        h = gate.register_forward_hook(_make_hook(resident, top_k))
        handles.append(h)

    return HardDropHandle(handles)
