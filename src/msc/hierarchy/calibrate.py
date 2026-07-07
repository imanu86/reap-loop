"""calibrate.py — dal probe (traccia di routing) al piano di residenza per-layer.

Il calibratore osserva QUALI expert il router attiva di piu' durante un probe (warm-up) e ne ricava,
per ogni layer, il working-set da PINNARE in VRAM (i residenti che install_expert_cache non evicte
mai). Confrontiamo questo piano "informato dal probe" con una baseline CIECA (expert 0..n-1): se il
probe non battesse il cieco, la calibrazione non servirebbe.

Output di entrambe le funzioni = `resident_by_layer`, esattamente la forma che
`install_expert_cache(..., resident_by_layer=...)` si aspetta:
    mappa  layer_idx -> set di expert-id (int).

CPU-only: si lavora sulla traccia jsonl (schema `msc.instrument.trace`, campi `layer` e `topk_ids`)
con `collections.Counter`; nessun import di torch/transformers.
"""

from __future__ import annotations

import json
from collections import Counter

__all__ = ["resident_from_trace", "blind_resident"]


def resident_from_trace(trace_path: str, n_per_layer: int) -> dict[int, set[int]]:
    """Working-set calibrato dal probe: per ogni layer i top-`n_per_layer` expert per frequenza.

    Legge la traccia jsonl di routing (schema `msc.instrument.trace`: ogni riga ha almeno i campi
    `layer` e `topk_ids`), conta quante volte ciascun expert compare in `topk_ids` per quel layer, e
    seleziona i piu' frequenti con `Counter.most_common(n_per_layer)`.

    Args:
        trace_path: percorso del file jsonl prodotto dal probe (una riga per record di routing).
        n_per_layer: quanti expert pinnare per layer (il budget di residenti per-layer). Se <= 0 si
            ritorna un set vuoto per ogni layer osservato. Se un layer ha visto meno di
            `n_per_layer` expert distinti, si ritornano tutti quelli osservati.

    Returns:
        Mappa `layer_idx -> set(expert_id)` con gli expert da tenere residenti/pinnati, pronta per
        `install_expert_cache(..., resident_by_layer=...)`.

    Note:
        `most_common` e' deterministico a parita' di conteggio (mantiene l'ordine di prima
        comparsa nel Counter), quindi due probe identici producono lo stesso piano. Le righe vuote o
        prive dei campi richiesti vengono ignorate (robustezza, come TraceReader).
    """
    n = max(0, int(n_per_layer))
    hist: dict[int, Counter] = {}

    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "layer" not in obj or "topk_ids" not in obj:
                continue
            layer = int(obj["layer"])
            counter = hist.get(layer)
            if counter is None:
                counter = Counter()
                hist[layer] = counter
            for eid in obj["topk_ids"]:
                counter[int(eid)] += 1

    resident: dict[int, set[int]] = {}
    for layer, counter in hist.items():
        if n == 0:
            resident[layer] = set()
            continue
        resident[layer] = {eid for eid, _ in counter.most_common(n)}
    return resident


def blind_resident(
    layers, n_per_layer: int, n_experts: int = 64
) -> dict[int, set[int]]:
    """Baseline CIECA per il confronto probe-vs-cieco: expert 0..n-1 pinnati in OGNI layer.

    Non guarda nessuna traccia: assegna lo STESSO set fisso di `n_per_layer` expert (gli id piu'
    bassi) a ogni layer. E' il piano "senza informazione" contro cui misurare il guadagno del
    working-set calibrato da `resident_from_trace`.

    Args:
        layers: iterabile di layer-idx per cui generare il piano (es. `range(16)` per OLMoE).
        n_per_layer: quanti expert pinnare per layer. Viene clampato a [0, n_experts].
        n_experts: numero totale di expert per layer (64 per OLMoE-1B-7B). Limita superiormente
            `n_per_layer` (non si possono pinnare piu' expert di quanti esistano).

    Returns:
        Mappa `layer_idx -> set(0..n_per_layer-1)`, stessa forma di `resident_from_trace` e pronta
        per `install_expert_cache(..., resident_by_layer=...)`.
    """
    n = max(0, min(int(n_per_layer), int(n_experts)))
    blind_set = set(range(n))
    # set() nuovo per ogni layer: evita aliasing condiviso (un evict/modifica non deve propagarsi).
    return {int(layer): set(blind_set) for layer in layers}
