"""hierarchy/ — gerarchia VRAM->RAM->SSD per far girare MoE ENORMI su HW minimo.

L'idea: la CAPACITA e' l'obiettivo (quanti GB di modello per GB di VRAM-expert), non la latenza.
Si fa SEMPRE fetch su miss (mai si droppa davvero un expert), quindi l'accuratezza resta ~costante;
la cache GPU per-layer + i residenti pinnati riducono i fetch CPU->GPU.

Moduli:
    calibrate.py — dal probe (traccia jsonl di routing) al piano di residenza per-layer.
    metrics.py   — metriche di capacita (CPU, no GPU): fetch-rate, GB-per-GB, stima VRAM-expert.

NB: nessun import di torch/transformers a livello di pacchetto — questi due moduli sono CPU-only e
lavorano su tracce jsonl e su stats-dict, cosi i test girano senza GPU.
"""

from msc.hierarchy.calibrate import blind_resident, resident_from_trace  # noqa: F401
from msc.hierarchy.metrics import (  # noqa: F401
    capacity_report,
    estimate_vram_expert_gb,
)
