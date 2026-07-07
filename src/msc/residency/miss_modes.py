"""miss_modes.py — asse D: cosa fa la policy su un expert NON residente.

STATO: IMPLEMENTATO. Solo `precision-cascade` e `hard-drop` producono perdita di ACCURATEZZA;
`fetch-lossless` fissa la baseline latenza-pura (perdita 0). Vedi docs/00_architecture.md §3/§5.

Nota di design (torch-free):
  La logica di BOOKKEEPING (scelta della precisione dalle soglie, conteggio byte PCIe, decisione
  reroute/zero) è tutta torch-free e testabile con fake. Il calcolo effettivo dell'expert e il
  movimento di tensori sono il "gpu_seam": isolati dietro callable iniettabili che importano torch
  internamente con guardia (vedi `compute_fn`/`fetch_fn`). Senza callable iniettati i gpu_seam
  ricadono su un import torch guardato, così `import msc.residency.miss_modes` non richiede torch.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, Optional


class MissMode(str, enum.Enum):
    FETCH_LOSSLESS = "fetch-lossless"      # carica da RAM a piena precisione: Δacc=0, costo=latenza PCIe
    PRECISION_CASCADE = "precision-cascade"  # serve da copia low-bit in VRAM: Δacc da quant, no PCIe
    HARD_DROP = "hard-drop"                # non serve l'expert: reroute o contributo azzerato


@dataclass(frozen=True)
class MissOutcome:
    """Cosa è successo servendo un miss (per le metriche)."""

    served: bool          # False solo in hard-drop=zero
    rerouted_to: int | None
    precision_bits: int | None  # bit usati (16 per lossless, 4/2 per cascade, None se drop)
    pcie_bytes: int       # byte trasferiti RAM->VRAM (0 se non si fa fetch)


# Tipo del callable di calcolo expert (gpu_seam). Riceve i pesi/handle dell'expert, l'hidden state e
# i bit di precisione; ritorna l'output dell'expert. Tutto opaco a questo livello (può essere torch).
ComputeFn = Callable[..., object]


def _expert_nbytes(hidden_state, bits: int) -> int:
    """Stima i byte di un expert servito a `bits` di precisione (gpu_seam contabile).

    Torch-free: prova a leggere una dimensione/hint dall'hidden_state (numpy/torch/fake con attributi
    .nbytes o .shape o len) per ricavare una scala plausibile; in mancanza usa un fallback costante.
    Serve solo a popolare `pcie_bytes`/contabilità, NON a fare il movimento reale.
    """
    # fp16 = 2 byte/elemento; bits arbitrari -> bits/8 byte/elemento.
    bytes_per_elem = max(bits, 1) / 8.0
    n_elems = _infer_n_elems(hidden_state)
    return int(n_elems * bytes_per_elem)


def _infer_n_elems(hidden_state) -> int:
    """Numero di elementi 'di scala' dell'hidden state, torch-free e difensivo."""
    if hidden_state is None:
        return 1
    # numpy / torch tensor espongono .size (callable in torch, int in numpy) -> gestiamo entrambi.
    size = getattr(hidden_state, "size", None)
    if callable(size):
        try:
            return int(size())
        except Exception:  # firma diversa (es. torch.size(dim)) -> ripiega su shape
            pass
    elif isinstance(size, int):
        return size
    shape = getattr(hidden_state, "shape", None)
    if shape is not None:
        n = 1
        for d in shape:
            n *= int(d)
        return n
    try:
        return len(hidden_state)
    except TypeError:
        return 1


class MissHandler:
    """Strategia di gestione miss. Implementazioni concrete una per MissMode."""

    mode: MissMode

    def handle(self, layer: int, expert_id: int, hidden_state, gate_weight: float):
        """Restituisce (output_dell_expert, MissOutcome). Implementato nelle sottoclassi."""
        raise NotImplementedError


class FetchLossless(MissHandler):
    """Carica l'expert da RAM a piena precisione (lossless). Costo = latenza PCIe.

    Opzionale: prefetch col predittore pre-attention (riuso del SOTA, vedi docs/01_positioning.md §1)
    per ammortizzare la latenza (rischio R3).

    gpu_seam: `fetch_fn(layer, expert_id, hidden_state)` materializza l'expert da RAM->VRAM e ne
    calcola l'output. Iniettabile per i test (fake). Se assente, l'output è None ma la contabilità
    (byte PCIe a fp16) resta valida e testabile CPU-only.
    """

    mode = MissMode.FETCH_LOSSLESS

    def __init__(self, fetch_fn: Optional[ComputeFn] = None, fp16_bits: int = 16) -> None:
        self.fetch_fn = fetch_fn
        self.fp16_bits = fp16_bits

    def handle(self, layer: int, expert_id: int, hidden_state, gate_weight: float):
        # Lossless: precisione piena (fp16) -> nessuna perdita di accuratezza per costruzione.
        pcie_bytes = _expert_nbytes(hidden_state, self.fp16_bits)
        output = None
        if self.fetch_fn is not None:
            output = self.fetch_fn(layer, expert_id, hidden_state)
        outcome = MissOutcome(
            served=True,
            rerouted_to=None,
            precision_bits=self.fp16_bits,
            pcie_bytes=pcie_bytes,
        )
        return output, outcome


class PrecisionCascade(MissHandler):
    """Serve l'expert da una copia a bassa precisione (int4/int2) residente in VRAM.

    Soglia di precisione sul gating weight ‖G(x)‖ (proxy validato a 0.99 da HOBBIT,
    docs/01_positioning.md §4): expert poco importanti -> precisione più bassa o skip.

    Semantica delle soglie (stile HOBBIT, T1/T2 sull'UNIMPORTANCE cumulata `u = 1 - ‖G(x)‖`):
      - u <  t_low   (expert importante)        -> servito a `low_bits` (es. 4)
      - t_low ≤ u < t_skip (importanza media)   -> servito a `low_bits/2` (es. 2)
      - u ≥  t_skip  (expert irrilevante)       -> SKIP: contributo azzerato (served=False)

    `gate_weight` è ‖G(x)‖ normalizzato in [0,1] (peso di gating dell'expert). Più è alto, più
    l'expert è importante -> più bit gli concediamo.

    gpu_seam: `compute_fn(layer, expert_id, hidden_state, bits)` esegue l'expert quantizzato a `bits`.
    """

    mode = MissMode.PRECISION_CASCADE

    def __init__(
        self,
        low_bits: int = 4,
        t_low: float = 0.6,
        t_skip: float = 0.9,
        compute_fn: Optional[ComputeFn] = None,
    ) -> None:
        # t_low/t_skip: soglie stile HOBBIT (T1=0.6, T2=0.9) sull'unimportance cumulata.
        self.low_bits = low_bits
        self.t_low = t_low
        self.t_skip = t_skip
        self.compute_fn = compute_fn

    def _bits_for(self, gate_weight: float) -> Optional[int]:
        """Sceglie i bit di precisione (o None=skip) dall'importanza del gate. Torch-free.

        Lavora sull'unimportance u = 1 - ‖G(x)‖ confrontata con le soglie cumulate T1/T2.
        """
        # Clampa l'importanza in [0,1]: gate_weight è già una norma normalizzata.
        importance = gate_weight
        if importance < 0.0:
            importance = 0.0
        elif importance > 1.0:
            importance = 1.0
        unimportance = 1.0 - importance
        if unimportance >= self.t_skip:
            return None  # skip: troppo irrilevante per spendere bit
        if unimportance >= self.t_low:
            # precisione dimezzata (es. low_bits=4 -> 2), mai sotto 1 bit
            return max(self.low_bits // 2, 1)
        return self.low_bits

    def handle(self, layer: int, expert_id: int, hidden_state, gate_weight: float):
        bits = self._bits_for(gate_weight)
        if bits is None:
            # Skip per irrilevanza: nessun calcolo, nessun PCIe (la copia low-bit è già in VRAM).
            outcome = MissOutcome(
                served=False,
                rerouted_to=None,
                precision_bits=None,
                pcie_bytes=0,
            )
            return None, outcome
        # La copia low-bit è RESIDENTE in VRAM -> nessun trasferimento PCIe.
        output = None
        if self.compute_fn is not None:
            output = self.compute_fn(layer, expert_id, hidden_state, bits)
        outcome = MissOutcome(
            served=True,
            rerouted_to=None,
            precision_bits=bits,
            pcie_bytes=0,
        )
        return output, outcome


class HardDrop(MissHandler):
    """Non serve l'expert non residente.

    Due varianti (vedi rischio R8):
      - reroute: reinstrada sul residente più vicino (altera il path -> ri-tracciare)
      - zero: azzera il contributo dell'expert (più pulito da analizzare) [default consigliato]

    gpu_seam:
      - `compute_fn(layer, expert_id, hidden_state, bits)` (solo reroute) calcola l'output del
        residente di reroute a piena precisione.
      - `nearest_resident_fn(layer, expert_id)` (solo reroute) ritorna l'id del residente più vicino,
        o None se nessun residente è disponibile -> si ripiega su zero.
    """

    mode = MissMode.HARD_DROP

    def __init__(
        self,
        variant: str = "zero",
        compute_fn: Optional[ComputeFn] = None,
        nearest_resident_fn: Optional[Callable[[int, int], Optional[int]]] = None,
    ) -> None:
        assert variant in ("zero", "reroute")
        self.variant = variant
        self.compute_fn = compute_fn
        self.nearest_resident_fn = nearest_resident_fn

    def handle(self, layer: int, expert_id: int, hidden_state, gate_weight: float):
        if self.variant == "reroute":
            target = None
            if self.nearest_resident_fn is not None:
                target = self.nearest_resident_fn(layer, expert_id)
            if target is not None:
                # Reroute: servito dal residente più vicino, a piena precisione, nessun PCIe.
                output = None
                if self.compute_fn is not None:
                    output = self.compute_fn(layer, target, hidden_state, 16)
                outcome = MissOutcome(
                    served=True,
                    rerouted_to=target,
                    precision_bits=16,
                    pcie_bytes=0,
                )
                return output, outcome
            # Nessun residente verso cui reinstradare -> degrada a zero.
        # Variante 'zero' (o reroute fallito): contributo azzerato, expert non servito.
        outcome = MissOutcome(
            served=False,
            rerouted_to=None,
            precision_bits=None,
            pcie_bytes=0,
        )
        return None, outcome


def make_miss_handler(mode: MissMode, **kwargs) -> MissHandler:
    """Factory: costruisce il MissHandler concreto per il `mode` dato.

    Accetta `mode` come MissMode o come stringa equivalente ("fetch-lossless", ...). I kwargs sono
    inoltrati al costruttore concreto (es. low_bits/t_low/t_skip per la cascade, variant per il drop,
    fetch_fn/compute_fn per il gpu_seam).
    """
    mode = MissMode(mode)  # normalizza stringa->enum (solleva ValueError se ignoto)
    if mode is MissMode.FETCH_LOSSLESS:
        return FetchLossless(**kwargs)
    if mode is MissMode.PRECISION_CASCADE:
        return PrecisionCascade(**kwargs)
    if mode is MissMode.HARD_DROP:
        return HardDrop(**kwargs)
    raise ValueError(f"MissMode non gestito: {mode!r}")
