"""manager.py — ResidencyManager + astrazione ExpertStore.

STATO: IMPLEMENTATO.

Responsabilità:
  - tenere un set di expert RESIDENTI in VRAM a piena precisione (per layer)
  - instradare i miss al MissHandler scelto (asse D)
  - contabilizzare VRAM residente, miss rate, byte PCIe, latenza (per experiment/metrics.py)

Astrazione CRITICA (docs/02_models.md §3): i modelli differiscono nel layout degli expert.
  - ModuleList (OLMoE, DeepSeek, Qwen): expert = nn.Module separati -> spostarli è diretto
  - fused/parallel (Granite): expert impacchettati in tensori unici -> servono SLICE
ExpertStore nasconde questa differenza dietro due backend.

Nota di design (torch-free):
  Tutta la BOOKKEEPING (insieme dei residenti per layer, byte VRAM stimati, conteggi hit/miss/drop/
  reroute/pcie, scelta del dispatch) è torch-free e testabile con fake. Il movimento reale dei
  tensori (`.to(device)` per ModuleList, gather/scatter di slice per i fused) è il "gpu_seam":
  isolato nel metodo `_materialize` di ciascun backend, che importa torch internamente con guardia.
  Senza torch i backend restano usabili per la sola contabilità (i test girano CPU-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from msc.residency.miss_modes import HardDrop, MissHandler, MissMode


class ExpertStore:
    """Backend di accesso ai pesi di un expert, astrae ModuleList vs fused (docs/02_models.md §3).

    Sottoclassi concrete: ModuleListStore, FusedSliceStore. Lo store tiene la contabilità dei
    residenti per layer e la stima dei byte VRAM, indipendentemente da torch.
    """

    def __init__(self, n_layers: int, n_experts: int, bytes_per_expert_fp16: int) -> None:
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.bytes_per_expert_fp16 = bytes_per_expert_fp16
        # layer -> set[expert_id] residenti in VRAM a piena precisione.
        self._resident: dict[int, set[int]] = {}

    def is_resident(self, layer: int, expert_id: int) -> bool:
        return expert_id in self._resident.get(layer, ())

    def set_resident(self, layer: int, expert_ids: set[int]) -> None:
        """Fissa (e materializza) il set residente di un layer. Gli altri expert restano fuori
        dalla VRAM a piena precisione (gestiti poi dal miss_mode).
        """
        ids = set(int(e) for e in expert_ids)
        self._resident[layer] = ids
        self._materialize(layer, ids)

    def resident_ids(self, layer: int) -> set[int]:
        """Set residente del layer (vuoto se non ancora committato)."""
        return set(self._resident.get(layer, set()))

    def nearest_resident(self, layer: int, expert_id: int) -> Optional[int]:
        """Residente più vicino (per id) all'expert dato, per il reroute di hard-drop.

        Heuristica torch-free e deterministica: minimizza |id - expert_id|, a parità il più piccolo.
        Ritorna None se il layer non ha residenti.
        """
        residents = self._resident.get(layer)
        if not residents:
            return None
        return min(residents, key=lambda r: (abs(r - expert_id), r))

    def resident_vram_bytes(self) -> int:
        """Byte VRAM occupati dagli expert residenti a piena precisione (somma su tutti i layer)."""
        total = sum(len(ids) for ids in self._resident.values())
        return total * self.bytes_per_expert_fp16

    # --- gpu_seam -----------------------------------------------------------------
    def _materialize(self, layer: int, expert_ids: set[int]) -> None:
        """Movimento reale dei pesi in VRAM. Override nei backend concreti (importa torch lì).

        Nel backend base è un no-op contabile (utile per i test CPU-only).
        """
        return None


class ModuleListStore(ExpertStore):
    """Backend per expert come ModuleList (OLMoE, DeepSeek, Qwen): residenza = `.to(device)` del
    singolo `nn.Module`.

    `experts_by_layer`: opzionale, layer -> sequenza di moduli expert (lista/ModuleList). Se fornito,
    `_materialize` sposta i residenti su `device` e gli altri su 'cpu'. Senza, lo store fa solo
    contabilità (test CPU-only).
    """

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        bytes_per_expert_fp16: int,
        experts_by_layer: Optional[dict] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(n_layers, n_experts, bytes_per_expert_fp16)
        self.experts_by_layer = experts_by_layer
        self.device = device

    def _materialize(self, layer: int, expert_ids: set[int]) -> None:
        # gpu_seam: sposta i moduli expert. Senza handle ai moduli, no-op (sola contabilità).
        if self.experts_by_layer is None:
            return None
        modules = self.experts_by_layer.get(layer)
        if modules is None:
            return None
        # torch non è importato a livello di modulo: qui non serve neppure importarlo perché usiamo
        # solo l'API `.to()` dei moduli (anatra). Resta isolato come gpu_seam comunque.
        for eid, module in enumerate(modules):
            if module is None:
                continue
            target = self.device if eid in expert_ids else "cpu"
            to_fn = getattr(module, "to", None)
            if callable(to_fn):
                to_fn(target)
        return None


class FusedSliceStore(ExpertStore):
    """Backend per expert fusi (Granite): residenza = slice dei tensori input_linear/output_linear.

    Nota: fetch-lossless è il più scomodo qui (gather/scatter di slice). Vedi docs/02_models.md §3.

    `fused_tensors`: opzionale, layer -> oggetto con i tensori fusi `[n_expert, ...]`. Se fornito,
    `_materialize` gestisce le slice (gpu_seam, importa torch internamente). Senza, sola contabilità.
    """

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        bytes_per_expert_fp16: int,
        fused_tensors: Optional[dict] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(n_layers, n_experts, bytes_per_expert_fp16)
        self.fused_tensors = fused_tensors
        self.device = device

    def _materialize(self, layer: int, expert_ids: set[int]) -> None:
        # gpu_seam: per i fused la residenza è una MASCHERA sulle slice. Il movimento/quantizzazione
        # reale delle slice non-residenti dipende dal miss_mode (cascade quantizza, drop azzera) ed è
        # gestito a valle. Qui registriamo solo la maschera; torch importato lazy se servisse.
        if self.fused_tensors is None:
            return None
        bundle = self.fused_tensors.get(layer)
        if bundle is None:
            return None
        # Annota la maschera di residenza sul bundle (consumata dal kernel fuso a valle).
        try:
            bundle.resident_mask = sorted(expert_ids)
        except AttributeError:
            # bundle immutabile/dict: best-effort, nessun fallimento hard nel seam.
            if isinstance(bundle, dict):
                bundle["resident_mask"] = sorted(expert_ids)
        return None


@dataclass
class ResidencyStats:
    """Contabilità per (modello, K, ctx, miss_mode). Alimenta experiment/metrics.py."""

    resident_vram_bytes: int = 0
    miss_count: int = 0
    hit_count: int = 0
    pcie_bytes: int = 0
    drop_count: int = 0
    reroute_count: int = 0

    @property
    def miss_rate(self) -> float:
        total = self.miss_count + self.hit_count
        return self.miss_count / total if total else 0.0


@dataclass
class ResidencyManager:
    """Orchestratore della residenza per un'intera rete MoE.

    Costruito dalla policy (policies/) con il set di expert da committare; durante l'inferenza
    intercetta le selezioni del router e serve hit (VRAM) o miss (via MissHandler).
    """

    store: ExpertStore
    miss_handler: MissHandler
    committed: dict = field(default_factory=dict)  # layer -> set[expert_id] residenti
    stats: ResidencyStats = field(default_factory=ResidencyStats)
    # gpu_seam (hit): calcola l'output di un expert residente a piena precisione.
    # Firma: hit_compute_fn(layer, expert_id, hidden_state) -> output. None nei test CPU-only.
    hit_compute_fn: Optional[object] = None

    def __post_init__(self) -> None:
        # Per il reroute di hard-drop: collega la ricerca del residente più vicino allo store, se
        # l'handler non ne ha già uno proprio. Non altera handler di altro tipo.
        if isinstance(self.miss_handler, HardDrop) and self.miss_handler.variant == "reroute":
            if self.miss_handler.nearest_resident_fn is None:
                self.miss_handler.nearest_resident_fn = self.store.nearest_resident

    def commit(self, per_layer_experts: dict) -> None:
        """Fissa il set residente per layer e lo materializza nello store.

        `per_layer_experts`: layer -> iterable di expert_id da tenere residenti (output della policy,
        PolicyDecision.per_layer_resident). Aggiorna `committed`, materializza nello store e ricalcola
        i byte VRAM residenti nelle stats.
        """
        self.committed = {}
        for layer, experts in per_layer_experts.items():
            ids = set(int(e) for e in experts)
            self.committed[int(layer)] = ids
            self.store.set_resident(int(layer), ids)
        self.stats.resident_vram_bytes = self.store.resident_vram_bytes()

    def on_route(self, layer: int, expert_id: int, hidden_state, gate_weight: float):
        """Hit -> serve da VRAM; miss -> delega al MissHandler e aggiorna stats.

        Ritorna (output, outcome) dove:
          - su HIT: output = calcolo fp16 residente (None nei test senza gpu_seam), outcome=None;
          - su MISS: output e MissOutcome del MissHandler scelto.
        """
        if self.store.is_resident(layer, expert_id):
            self.stats.hit_count += 1
            output = None
            if self.hit_compute_fn is not None:
                output = self.hit_compute_fn(layer, expert_id, hidden_state)
            return output, None

        # MISS: delega all'handler e contabilizza l'esito.
        self.stats.miss_count += 1
        output, outcome = self.miss_handler.handle(layer, expert_id, hidden_state, gate_weight)
        self.stats.pcie_bytes += outcome.pcie_bytes
        if outcome.rerouted_to is not None:
            self.stats.reroute_count += 1
        elif not outcome.served:
            # Non servito e non reinstradato -> drop effettivo (hard-drop zero o skip cascade).
            self.stats.drop_count += 1
        return output, outcome

    @property
    def miss_mode(self) -> MissMode:
        return self.miss_handler.mode
