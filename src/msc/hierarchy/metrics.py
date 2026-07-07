"""metrics.py — metriche di CAPACITA della gerarchia (CPU, no GPU).

La tesi del progetto: con la gerarchia VRAM->RAM->SSD la VRAM-expert non limita piu' la DIMENSIONE
del modello, solo la latenza. La metrica chiave e' quindi il `capacity_ratio`: quanti GB di modello
(expert) si riescono a far girare per ogni GB di VRAM dedicata agli expert residenti. Piu' e' alto,
piu' "enorme" e' il modello che entra su una 3060.

Le altre due metriche raccontano il COSTO di quel guadagno:
  - fetch_rate  : frazione di accessi serviti da un fetch CPU->GPU (miss). Alto = piu' traffico PCIe
                  (latenza), ma l'accuratezza resta ~costante perche' si fa SEMPRE fetch (mai drop).
  - fetched_gb  : volume totale trasferito CPU->GPU durante la sessione.

Tutto CPU-only: si lavora sul dict ritornato da `CacheHandle.stats()` (vedi INTERFACCIA), nessun
import di torch/transformers.

Schema di `cache_stats` atteso (= CacheHandle.stats()):
    {
      "per_layer": { layer_idx: Stats, ... },     # Stats = {hits, misses, fetched_bytes, resident}
      "total":     Stats,                          # aggregata sui layer
      "fetch_rate": float,                         # total.misses / (total.hits + total.misses)
      "fetched_gb": float,
      "resident_experts": int,
    }
"""

from __future__ import annotations

__all__ = ["capacity_report", "estimate_vram_expert_gb"]

_BYTES_PER_GB = float(1 << 30)


def _fetch_rate_from_total(cache_stats: dict) -> float:
    """Ricalcola il fetch-rate dal blocco `total` (fallback se la chiave top-level mancasse).

    fetch_rate = misses / (hits + misses); con zero accessi (hits+misses==0) per convenzione 0.0.
    """
    total = cache_stats.get("total", {}) or {}
    hits = int(total.get("hits", 0))
    misses = int(total.get("misses", 0))
    accesses = hits + misses
    if accesses <= 0:
        return 0.0
    return misses / accesses


def capacity_report(
    cache_stats: dict, model_total_gb: float, vram_expert_gb: float
) -> dict:
    """Report di capacita' da uno stats-dict di `CacheHandle.stats()`.

    Args:
        cache_stats: dict ritornato da `CacheHandle.stats()` (vedi schema nel docstring del modulo).
        model_total_gb: dimensione totale degli expert del modello su disco/RAM, in GB (il "modello
            enorme" che vogliamo far girare).
        vram_expert_gb: VRAM effettivamente dedicata agli expert residenti, in GB (la cache GPU
            per-layer). E' il denominatore del guadagno di capacita'.

    Returns:
        dict con:
          - fetch_rate (float): preso da `cache_stats["fetch_rate"]` se presente, altrimenti
            ricalcolato da `total`.
          - fetched_gb (float): volume CPU->GPU trasferito (da `cache_stats`).
          - resident_experts (int): numero di expert residenti in VRAM (da `cache_stats`).
          - capacity_ratio (float): model_total_gb / vram_expert_gb = GB di modello per GB di
            VRAM-expert. Con vram_expert_gb <= 0 -> 0.0 (evita divisione per zero).
          - model_total_gb / vram_expert_gb (float): echo degli input, per tracciabilita'.
          - summary (str): una riga sintetica leggibile.
    """
    # fetch_rate: usa la chiave gia' calcolata dall'handle, ma resta robusto se mancasse.
    if "fetch_rate" in cache_stats and cache_stats["fetch_rate"] is not None:
        fetch_rate = float(cache_stats["fetch_rate"])
    else:
        fetch_rate = _fetch_rate_from_total(cache_stats)

    fetched_gb = float(cache_stats.get("fetched_gb", 0.0))
    resident_experts = int(cache_stats.get("resident_experts", 0))

    vram_expert_gb = float(vram_expert_gb)
    model_total_gb = float(model_total_gb)
    capacity_ratio = (model_total_gb / vram_expert_gb) if vram_expert_gb > 0 else 0.0

    summary = (
        f"capacity {capacity_ratio:.1f}x "
        f"({model_total_gb:.2f} GB model / {vram_expert_gb:.2f} GB VRAM-expert) | "
        f"fetch_rate {fetch_rate * 100:.1f}% | "
        f"fetched {fetched_gb:.2f} GB | "
        f"resident {resident_experts} experts"
    )

    return {
        "fetch_rate": fetch_rate,
        "fetched_gb": fetched_gb,
        "resident_experts": resident_experts,
        "capacity_ratio": capacity_ratio,
        "model_total_gb": model_total_gb,
        "vram_expert_gb": vram_expert_gb,
        "summary": summary,
    }


def estimate_vram_expert_gb(
    capacity_per_layer: int, n_layers: int, bytes_per_expert: float
) -> float:
    """Stima la VRAM-expert occupata dalla cache GPU per-layer, in GB.

    La cache tiene al piu' `capacity_per_layer` slice di expert residenti per layer; con `n_layers`
    layer e `bytes_per_expert` byte per slice (gate_up_proj + down_proj di UN expert), la VRAM-expert
    e':
        capacity_per_layer * n_layers * bytes_per_expert / 2**30.

    Args:
        capacity_per_layer: capienza (numero di expert) della cache GPU di ciascun layer.
        n_layers: numero di layer MoE del modello (16 per OLMoE-1B-7B).
        bytes_per_expert: byte di UNA slice di expert (gate_up_proj + down_proj), nella dtype usata.

    Returns:
        VRAM-expert stimata in GB (float). Coerente col denominatore di `capacity_report`.
    """
    total_bytes = (
        max(0, int(capacity_per_layer))
        * max(0, int(n_layers))
        * float(bytes_per_expert)
    )
    return total_bytes / _BYTES_PER_GB
