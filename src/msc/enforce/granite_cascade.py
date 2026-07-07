"""msc.enforce.granite_cascade — confronto dei miss_mode su Granite-MoE via MODIFICA DEI PESI.

Granite-3.1 entra interamente in 12GB a fp16 (niente offload) -> possiamo modificare in-place i pesi
degli expert. Gli expert sono FUSI: block_sparse_moe.input_linear.weight [n_exp, 2*inter, hidden] e
block_sparse_moe.output_linear.weight [n_exp, hidden, inter]. L'expert e e' la slice [e].

Confronto CONTROLLATO dei miss_mode (il router NON viene toccato: stessa selezione top-k; cambia solo
la QUALITA' dell'expert non residente):
  - "hard-drop"        : pesi dei non residenti azzerati -> contributo nullo se selezionati (variante "zero").
  - "precision-cascade": pesi dei non residenti fake-quantizzati a int{nbits} (degrado graduale).

snapshot_experts/restore_experts permettono di valutare piu' celle senza ricaricare il modello.
"""

from __future__ import annotations

import torch


def fake_quant_(w: torch.Tensor, nbits: int, groupsize: int = 64) -> None:
    """Fake-quant in-place, simmetrica, a gruppi lungo l'ultima dimensione (stile int{nbits} a blocchi).

    Quantizza->dequantizza: i pesi restano fp16 ma rappresentabili su una griglia int{nbits} (la VRAM
    risparmiata e' contabilizzata analiticamente). groupsize=64 ~ blocchi GGUF.
    """
    D = w.shape[-1]
    G = groupsize if D % groupsize == 0 else D
    wf = w.detach().float()
    shp = wf.shape
    x = wf.reshape(*shp[:-1], D // G, G)
    qmax = (1 << (nbits - 1)) - 1                       # int4 -> 7, int2 -> 1, int8 -> 127
    scale = x.abs().amax(dim=-1, keepdim=True) / max(qmax, 1)
    scale = scale.clamp_min(1e-8)
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    deq = (q * scale).reshape(shp)
    w.data.copy_(deq.to(w.dtype))


def _moe(model, layer_idx: int):
    return model.model.layers[layer_idx].block_sparse_moe


def snapshot_experts(model) -> dict:
    """Copia su CPU i pesi expert (input/output linear) di tutti i layer MoE, per il restore."""
    snap = {}
    for i, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        snap[(i, "in")] = moe.input_linear.weight.detach().to("cpu", copy=True)
        snap[(i, "out")] = moe.output_linear.weight.detach().to("cpu", copy=True)
    return snap


def restore_experts(model, snap: dict) -> None:
    """Ripristina i pesi expert originali dallo snapshot (annulla qualsiasi modifica di modo)."""
    for i, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        moe.input_linear.weight.data.copy_(snap[(i, "in")].to(moe.input_linear.weight.device))
        moe.output_linear.weight.data.copy_(snap[(i, "out")].to(moe.output_linear.weight.device))


def apply_mode(model, resident_by_layer: dict[int, set[int]], mode: str, nbits: int = 4) -> int:
    """Applica il miss_mode ai pesi degli expert NON residenti, in-place. Ritorna il n. di expert toccati.

    mode: "hard-drop" (azzera) | "precision-cascade" (fake-quant int{nbits}).
    """
    assert mode in ("hard-drop", "precision-cascade")
    n_exp = int(model.config.num_local_experts)
    touched = 0
    for layer_idx, resident in resident_by_layer.items():
        moe = _moe(model, layer_idx)
        w_in = moe.input_linear.weight.data    # [n_exp, 2*inter, hidden]
        w_out = moe.output_linear.weight.data  # [n_exp, hidden, inter]
        for e in range(n_exp):
            if e in resident:
                continue
            if mode == "hard-drop":
                w_in[e].zero_()
                w_out[e].zero_()
            else:
                fake_quant_(w_in[e], nbits)
                fake_quant_(w_out[e], nbits)
            touched += 1
    return touched
