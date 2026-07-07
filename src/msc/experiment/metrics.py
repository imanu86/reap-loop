"""metrics.py — cosa registriamo per ogni cella della griglia.

CRITICO (rischio R6): VRAM_expert e VRAM_kv vanno SEPARATI, altrimenti a 64k la KV cache maschera
il guadagno-expert e il verdetto è ingannevole. Misura reale via nvidia-ml-py (non solo stima).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo per i type hint: pandas non serve a import-time
    import pandas as pd


@dataclass(frozen=True)
class VramBreakdown:
    """Scomposizione della VRAM (byte). La somma ≈ VRAM totale misurata."""

    backbone: int      # attention, embedding, norm
    kv_cache: int      # cresce con ctx_len -> riportato a parte (R6)
    experts_resident: int  # il termine su cui agisce la policy (il claim del progetto)
    overhead: int      # contesto CUDA, buffer, framing

    @property
    def total(self) -> int:
        return self.backbone + self.kv_cache + self.experts_resident + self.overhead


@dataclass(frozen=True)
class CellMetrics:
    """Riga di risultato per una cella (modello, policy, K, ctx, miss_mode)."""

    # identità della cella
    model_id: str
    policy: str
    k_fraction: float
    ctx_len: int
    miss_mode: str
    sparsity_ratio: float          # asse B strutturale

    # esito primario: accuratezza A QUESTA ctx (un punto della curva), e drop vs FULL
    accuracy: float
    accuracy_drop_vs_full: float   # FULL@stessa ctx - questa cella; il <2% si valuta qui

    # concentrazione empirica (asse B, la variabile che spiega la curva)
    mean_n_eff: float              # numero efficace di expert medio sui layer
    mean_entropy_norm: float

    # costo
    vram: VramBreakdown
    miss_rate: float               # da osservare vs ctx (segnale di R1)
    latency_ms_per_token: float    # rilevante soprattutto per fetch-lossless (R3)

    # affidabilità
    working_set_converged: bool    # R4: la stima si è stabilizzata in warm-up?

    def to_row(self) -> dict:
        """Appiattisce in una riga piatta (per pandas/CSV).

        TUTTI i campi diventano colonne scalari. In particolare la VramBreakdown viene scorporata
        in colonne SEPARATE con prefisso ``vram_`` -> ``vram_backbone``, ``vram_kv_cache``,
        ``vram_experts_resident``, ``vram_overhead`` (rischio R6: la VRAM-KV NON va sommata a
        quella degli expert, altrimenti a 64k maschera il guadagno della policy). Aggiungiamo anche
        ``vram_total`` come comodità, ma resta derivabile dai contributi separati.
        """
        row: dict = {}
        # Tutti i campi scalari della cella, nello stesso ordine di dichiarazione, tranne `vram`
        # che richiede uno scorporo dedicato.
        for f in fields(self):
            if f.name == "vram":
                continue
            row[f.name] = getattr(self, f.name)

        # VramBreakdown -> colonne separate con prefisso `vram_` (contributi NON aggregati).
        for vf in fields(self.vram):
            row[f"vram_{vf.name}"] = getattr(self.vram, vf.name)
        # Totale derivato (comodità per i grafici), separato dai contributi.
        row["vram_total"] = self.vram.total

        return row


def rows_to_dataframe(rows: list[CellMetrics | dict]) -> "pd.DataFrame":
    """Costruisce un DataFrame piatto da una lista di CellMetrics (o di dict già appiattiti).

    Helper di comodità per l'aggregazione della griglia: accetta indifferentemente oggetti
    CellMetrics (su cui chiama ``to_row``) o dict già piatti. pandas viene importato QUI dentro,
    cosi ``import msc.experiment.metrics`` resta leggero e privo di dipendenze pesanti a import-time.
    """
    import pandas as pd

    flat = [r.to_row() if isinstance(r, CellMetrics) else r for r in rows]
    return pd.DataFrame(flat)
