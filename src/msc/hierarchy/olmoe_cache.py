"""olmoe_cache.py — MOTORE slice-cache degli expert per OLMoE (il cuore della gerarchia).

Idea: gli expert FUSI di OLMoE (``OlmoeExperts``) pesano ~14 GB in fp16 e non entrano nella VRAM di
una 3060. Ma a ogni token solo top-8 dei 64 expert per layer vengono eseguiti. Teniamo quindi il
BACKING STORE canonico degli expert su CPU (o, in futuro, su SSD via ``numpy.memmap``) e in VRAM solo
una CACHE per-layer di poche slice ``(gate_up_proj[e], down_proj[e])``. Su MISS si fa fetch CPU->GPU
della slice e si fa LRU-evict di una slice non pinnata; gli expert nel working-set sono PINNATI e mai
evitti. Si fa SEMPRE fetch (mai si droppa un expert): la matematica del forward resta IDENTICA a
quella originale, cambia solo DOVE vivono i pesi -> accuratezza ~costante, capacita' enorme.

Architettura:
  - ExpertBacking (ABC): astrazione del backing store di UN layer (64 slice fp16). Implementazioni:
      * CpuParamBacking: legge le slice dal ``nn.Parameter`` su CPU (pinned se possibile). Default.
      * (futuro) MemmapBacking: leggerebbe da ``numpy.memmap`` su SSD — stessa interfaccia ``slice(e)``,
        cosi swappare il backing e' BANALE (vedi docstring di ExpertBacking).
  - ExpertSliceCache: cache GPU a capienza fissa per UN OlmoeExperts. Pin-set + LRU sui non pinnati +
    Stats(hits, misses, fetched_bytes, resident). ``get(e)`` ritorna le due slice su GPU.
  - install_expert_cache(...): muove il backbone su cuda, lascia gli expert su CPU, monkeypatcha
    ``OlmoeExperts.forward`` di ogni layer per servire le slice dalla cache. Ritorna un CacheHandle.

INTERFACCIA CONDIVISA (rispettata ESATTAMENTE):
  - Stats per-layer = dict {hits, misses, fetched_bytes, resident} (tutti int).
  - CacheHandle.stats() = {per_layer: {layer_idx: Stats}, total: Stats, fetch_rate, fetched_gb,
    resident_experts}.
  - install_expert_cache(model, capacity_per_layer, resident_by_layer, device='cuda') -> CacheHandle.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Callable

import torch
from torch import nn

__all__ = [
    "Stats",
    "ExpertBacking",
    "CpuParamBacking",
    "MemmapBacking",
    "ExpertSliceCache",
    "CacheHandle",
    "install_expert_cache",
]


# --------------------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------------------
def _new_stats() -> dict[str, int]:
    """Stats per-layer come da INTERFACCIA: tutte le chiavi int.

    hits          : accessi serviti da una slice gia' residente in VRAM.
    misses        : accessi che hanno richiesto un fetch CPU->GPU (slice non residente).
    fetched_bytes : byte totali trasferiti CPU->GPU (somma delle slice fetchate).
    resident      : numero di slice di expert attualmente residenti in VRAM per questo layer.
    """
    return {"hits": 0, "misses": 0, "fetched_bytes": 0, "resident": 0}


# Alias documentale: una "Stats" e' semplicemente un dict[str, int] con quelle 4 chiavi.
Stats = dict


# --------------------------------------------------------------------------------------------------
# Backing store (astrazione: CPU oggi, SSD/memmap domani)
# --------------------------------------------------------------------------------------------------
class ExpertBacking(abc.ABC):
    """Backing store canonico fp16 degli expert di UN layer (la copia che vivrebbe su SSD).

    Espone le slice di UN expert come due tensori CPU pronti per il transfer su GPU. La cache GPU
    legge SOLO da qui: questa e' l'astrazione che rende BANALE swappare la sorgente.

    Per passare a SSD basta scrivere una sottoclasse ``MemmapBacking`` che, in ``__init__``, fa il dump
    di ``gate_up_proj``/``down_proj`` in due ``numpy.memmap`` fp16 su disco (shape
    [num_experts, ...]) e, in ``slice(e)``, fa ``torch.from_numpy(mmap[e])``. La firma resta identica
    (``slice(e) -> (gate_up_cpu, down_cpu)``, piu' ``num_experts``/``bytes_per_expert``), quindi
    ExpertSliceCache e install_expert_cache NON cambiano di una riga. La differenza e' solo che il
    fetch diventa SSD->RAM->GPU invece di RAM(pinned)->GPU.
    """

    num_experts: int

    @abc.abstractmethod
    def slice(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Ritorna (gate_up_proj[e], down_proj[e]) come tensori CPU fp16 pronti al transfer."""
        raise NotImplementedError

    @abc.abstractmethod
    def bytes_per_expert(self) -> int:
        """Byte di UNA slice di expert (gate_up + down) nella dtype del backing (per le Stats)."""
        raise NotImplementedError


class CpuParamBacking(ExpertBacking):
    """Backing su CPU: legge le slice direttamente dai ``nn.Parameter`` dell'OlmoeExperts.

    I parametri ``gate_up_proj`` [num_experts, 2*inter, hidden] e ``down_proj`` [num_experts, hidden,
    inter] restano su CPU (mai spostati su cuda). Vengono castati a fp16 (dtype canonico del backing,
    = quella che vivrebbe su SSD) e, se possibile, ``pin_memory()`` per transfer ``non_blocking``
    piu' veloci. ``slice(e)`` e' una vista [e] -> nessuna copia extra prima del ``.to(cuda)``.
    """

    def __init__(self, experts: nn.Module, store_dtype: torch.dtype = torch.float16) -> None:
        self.num_experts = int(experts.num_experts)
        self._store_dtype = store_dtype

        # Materializziamo il backing canonico fp16 su CPU. .data per evitare di trascinare autograd.
        gate_up = experts.gate_up_proj.data.to(device="cpu", dtype=store_dtype).contiguous()
        down = experts.down_proj.data.to(device="cpu", dtype=store_dtype).contiguous()

        # Pinned memory: abilita i transfer non_blocking CPU->GPU. Best-effort (puo' fallire se la RAM
        # pinnabile e' esaurita); in quel caso si resta su memoria pageable, comunque corretto.
        self._pinned = False
        try:
            gate_up = gate_up.pin_memory()
            down = down.pin_memory()
            self._pinned = True
        except (RuntimeError, NotImplementedError):
            self._pinned = False

        self._gate_up = gate_up
        self._down = down

        # byte di una slice (gate_up[e] + down[e]) nella store_dtype.
        elem = gate_up.element_size()
        self._bytes_per_expert = int(
            (gate_up[0].numel() + down[0].numel()) * elem
        )

    @property
    def pinned(self) -> bool:
        return self._pinned

    def slice(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Viste [e]: il transfer .to(cuda, non_blocking=True) avviene nella cache.
        return self._gate_up[expert_idx], self._down[expert_idx]

    def bytes_per_expert(self) -> int:
        return self._bytes_per_expert


class MemmapBacking(ExpertBacking):
    """Backing su SSD via ``numpy.memmap``: gli expert NON stanno in RAM, vivono su disco.

    Legge le slice di UN layer da due memmap GLOBALI fp16 (condivise tra tutti i layer):
      gate_up [n_layers, n_experts, 2*inter, hidden] e down [n_layers, n_experts, hidden, inter].
    ``slice(e)`` materializza in RAM solo la slice richiesta (~MB), poi la cache la porta su GPU.
    Cosi un modello con expert piu' grandi della RAM gira: la capacita' e' limitata dal DISCO, non
    dalla RAM. E' il cuore della tesi "modello gigante su HW minimo".

    I memmap si costruiscono una volta dai safetensors con scripts/build_qwen3_memmap.py.
    """

    def __init__(self, gate_up_mmap, down_mmap, layer_idx: int, num_experts: int) -> None:
        self._gu = gate_up_mmap   # np.memmap [L, E, 2I, H] fp16
        self._dn = down_mmap      # np.memmap [L, E, H, I] fp16
        self._li = int(layer_idx)
        self.num_experts = int(num_experts)
        self.pinned = False       # le slice da memmap non sono pinnate (copia poi transfer)
        elem = int(self._gu.dtype.itemsize)
        self._bytes_per_expert = int((self._gu[0, 0].size + self._dn[0, 0].size) * elem)

    def slice(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # .copy() materializza la sola slice [layer, e] dal memmap (legge le pagine da SSD);
        # torch.from_numpy avvolge senza ulteriore copia. Il transfer su GPU avviene nella cache.
        gu = torch.from_numpy(self._gu[self._li, expert_idx].copy())
        dn = torch.from_numpy(self._dn[self._li, expert_idx].copy())
        return gu, dn

    def bytes_per_expert(self) -> int:
        return self._bytes_per_expert


# --------------------------------------------------------------------------------------------------
# Cache GPU di slice per UN layer
# --------------------------------------------------------------------------------------------------
class ExpertSliceCache:
    """Cache GPU a capienza fissa delle slice di UN OlmoeExperts.

    Tiene in VRAM al piu' ``capacity`` slice ``(gate_up[e], down[e])`` (fp16). Gli expert in ``pinned``
    sono il working-set: sempre residenti, mai evitti, e NON contano per l'LRU. Gli altri slot sono
    gestiti LRU: su miss, se non c'e' spazio, si evicte la slice non-pinnata usata meno di recente.

    ``capacity`` deve essere >= |pinned| (i pinnati devono entrare tutti). Se capacity == num_experts
    e tutti gli expert vengono toccati, dopo il warm-up non ci sono piu' miss (cache piena = lossless
    "always-resident").
    """

    def __init__(
        self,
        backing: ExpertBacking,
        capacity: int,
        pinned: set[int],
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> None:
        self.backing = backing
        self.num_experts = int(backing.num_experts)
        self.device = device
        # dtype con cui gli expert girano nel forward (= dtype del modello, es. bf16). Le slice del
        # backing sono fp16; si castano a compute_dtype al momento del fetch, sulla GPU.
        self.compute_dtype = compute_dtype

        self.pinned: set[int] = set(int(e) for e in pinned)
        if any(not 0 <= e < self.num_experts for e in self.pinned):
            raise ValueError(f"pinned contiene expert-id fuori range [0,{self.num_experts}): {self.pinned}")

        cap = int(capacity)
        if cap < len(self.pinned):
            raise ValueError(
                f"capacity ({cap}) < |pinned| ({len(self.pinned)}): i pinnati non entrano in cache"
            )
        self.capacity = min(cap, self.num_experts)

        # Slot residenti: expert_idx -> (gate_up_gpu, down_gpu). OrderedDict come coda LRU per i NON
        # pinnati (i pinnati restano nella mappa ma non vengono mai scelti come vittima).
        self._resident: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

        self.stats: dict[str, int] = _new_stats()

        # Pre-carica (pin) il working-set: questi non verranno mai fetchati di nuovo ne' evitti. Il
        # fetch iniziale dei pinnati e' setup, non un "miss" di runtime -> non lo contiamo nelle stats.
        for e in sorted(self.pinned):
            gu, dn = self._fetch_to_gpu(e, count=False)
            self._resident[e] = (gu, dn)
        self.stats["resident"] = len(self._resident)

    # ---- transfer ----
    def _fetch_to_gpu(self, expert_idx: int, count: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Copia la slice dell'expert dal backing CPU alla GPU (cast a compute_dtype).

        ``count=True`` aggiorna misses/fetched_bytes (fetch di runtime); ``count=False`` per il
        pre-load dei pinnati (setup, non un miss).
        """
        gu_cpu, dn_cpu = self.backing.slice(expert_idx)
        non_blocking = bool(getattr(self.backing, "pinned", False))
        gu = gu_cpu.to(device=self.device, dtype=self.compute_dtype, non_blocking=non_blocking)
        dn = dn_cpu.to(device=self.device, dtype=self.compute_dtype, non_blocking=non_blocking)
        if count:
            self.stats["misses"] += 1
            self.stats["fetched_bytes"] += self.backing.bytes_per_expert()
        return gu, dn

    def _evict_one(self) -> None:
        """Evicte la slice NON pinnata usata meno di recente (fronte dell'OrderedDict)."""
        for e in list(self._resident.keys()):
            if e in self.pinned:
                continue
            self._resident.pop(e)
            self.stats["resident"] = len(self._resident)
            return
        # Nessun candidato non-pinnato: significa capacity == |pinned| (cache tutta pinnata). In quel
        # caso non si dovrebbe mai arrivare qui per un non-pinnato, ma restiamo difensivi.
        raise RuntimeError("evict richiesto ma nessuna slice non-pinnata disponibile")

    # ---- API ----
    def get(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Ritorna (gate_up_gpu, down_gpu) per l'expert, fetchando da CPU su miss.

        HIT: slice gia' residente -> aggiorna l'LRU (se non pinnata) e conta un hit.
        MISS: fetch CPU->GPU, eventuale evict LRU di un non-pinnato, inserimento; conta miss+bytes.
        """
        e = int(expert_idx)
        slot = self._resident.get(e)
        if slot is not None:
            self.stats["hits"] += 1
            if e not in self.pinned:
                # touch LRU: sposta in coda (most-recently-used).
                self._resident.move_to_end(e)
            return slot

        # MISS
        gu, dn = self._fetch_to_gpu(e, count=True)
        # Fai spazio se serve (solo non-pinnati possono essere evitti).
        while len(self._resident) >= self.capacity:
            self._evict_one()
        self._resident[e] = (gu, dn)
        self._resident.move_to_end(e)  # most-recently-used in coda
        self.stats["resident"] = len(self._resident)
        return gu, dn

    def clear(self) -> None:
        """Libera tutte le slice GPU (chiamato da CacheHandle.remove())."""
        self._resident.clear()
        self.stats["resident"] = 0


# --------------------------------------------------------------------------------------------------
# Forward monkeypatch (stessa matematica dell'originale, slice dalla cache)
# --------------------------------------------------------------------------------------------------
def _make_cached_forward(experts: nn.Module, cache: ExpertSliceCache) -> Callable:
    """Costruisce il forward sostitutivo per UN OlmoeExperts, che serve le slice dalla cache.

    Replica ESATTAMENTE la matematica del forward originale di OlmoeExperts:
      - one_hot/permute per trovare gli expert colpiti e i token per expert;
      - per ogni expert: gate, up = linear(state, gate_up[e]).chunk(2); act_fn(gate)*up;
        out = linear(., down[e]); moltiplica per top_k_weights; index_add_ nel risultato.
    L'UNICA differenza: ``gate_up[e]``/``down[e]`` arrivano dalla cache GPU (fetch su miss) invece che
    dal Parameter. ``act_fn`` resta quello del modulo (silu per OLMoE).
    """
    act_fn = experts.act_fn
    num_experts = int(experts.num_experts)

    def cached_forward(
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # ---- unica differenza vs originale: pesi dalla cache (fetch su miss) ----
            gate_up_w, down_w = cache.get(int(expert_idx))

            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    return cached_forward


# --------------------------------------------------------------------------------------------------
# CacheHandle + install_expert_cache
# --------------------------------------------------------------------------------------------------
class CacheHandle:
    """Handle restituito da ``install_expert_cache``: espone le stats e ripristina i forward.

    ``stats()`` aggrega le Stats per-layer secondo l'INTERFACCIA CONDIVISA. ``remove()`` ripristina i
    forward originali di ogni OlmoeExperts e libera le cache GPU (idempotente).
    """

    def __init__(
        self,
        caches: dict[int, ExpertSliceCache],
        originals: dict[int, Callable],
        experts_modules: dict[int, nn.Module],
    ) -> None:
        self._caches = caches
        self._originals = originals
        self._experts = experts_modules
        self._removed = False

    def stats(self) -> dict:
        """Stats aggregate come da INTERFACCIA CONDIVISA.

        Ritorna {per_layer: {layer_idx: Stats}, total: Stats, fetch_rate, fetched_gb,
        resident_experts}. ``fetch_rate`` = total.misses/(total.hits+total.misses) (0.0 se 0 accessi).
        """
        per_layer: dict[int, dict[str, int]] = {}
        total = _new_stats()
        for layer_idx, cache in self._caches.items():
            s = cache.stats
            # copia difensiva (gli int sono immutabili, ma evitiamo aliasing del dict interno)
            per_layer[layer_idx] = dict(s)
            total["hits"] += s["hits"]
            total["misses"] += s["misses"]
            total["fetched_bytes"] += s["fetched_bytes"]
            total["resident"] += s["resident"]

        accesses = total["hits"] + total["misses"]
        fetch_rate = (total["misses"] / accesses) if accesses > 0 else 0.0
        fetched_gb = total["fetched_bytes"] / float(1 << 30)

        return {
            "per_layer": per_layer,
            "total": total,
            "fetch_rate": fetch_rate,
            "fetched_gb": fetched_gb,
            "resident_experts": total["resident"],
        }

    def remove(self) -> None:
        """Ripristina i forward originali e libera le cache GPU. Idempotente."""
        if self._removed:
            return
        for layer_idx, experts in self._experts.items():
            orig = self._originals.get(layer_idx)
            if orig is not None:
                # rimuovi l'override d'istanza -> torna a usare il forward di classe.
                try:
                    del experts.forward
                except AttributeError:
                    experts.forward = orig  # type: ignore[assignment]
        for cache in self._caches.values():
            cache.clear()
        self._caches = {}
        self._removed = True

    def __enter__(self) -> "CacheHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def _move_backbone_to_device(model: nn.Module, device: torch.device) -> None:
    """Sposta su ``device`` TUTTO il modello TRANNE i pesi degli OlmoeExperts.

    Embedding, attention, router (gate), norm, lm_head -> device. I Parameter ``gate_up_proj`` e
    ``down_proj`` di ogni OlmoeExperts restano su CPU (il loro backing canonico lo gestisce la cache).
    Si sposta per-parametro/buffer per non toccare gli expert.
    """
    # Set degli id() dei tensori-expert da NON spostare.
    expert_param_ids: set[int] = set()
    for layer in model.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue  # layer DENSE (es. DeepSeek-V2 layer 0): nessun expert da preservare
        expert_param_ids.add(id(experts.gate_up_proj))
        expert_param_ids.add(id(experts.down_proj))

    for _name, p in model.named_parameters(recurse=True):
        if id(p) in expert_param_ids:
            continue
        if p.device != device:
            p.data = p.data.to(device)
    for _name, b in model.named_buffers(recurse=True):
        if b.device != device:
            b.data = b.data.to(device)


def install_expert_cache(
    model: nn.Module,
    capacity_per_layer: int,
    resident_by_layer: dict[int, set[int]],
    device: str | torch.device = "cuda",
    backing_factory: "Callable[[int, nn.Module], ExpertBacking] | None" = None,
) -> CacheHandle:
    """Installa la slice-cache degli expert su un OLMoE caricato su CPU.

    Sposta il backbone (embed/attention/router/norm/lm_head) su ``device``, lascia i pesi degli
    expert su CPU (backing canonico fp16, pinned se possibile) e monkeypatcha
    ``OlmoeExperts.forward`` di OGNI layer perche' serva le slice ``(gate_up[e], down[e])`` da una
    cache GPU per-layer di capienza ``capacity_per_layer``, con gli expert in ``resident_by_layer[i]``
    PINNATI (mai evitti). Su miss fa fetch CPU->GPU della slice e LRU-evict dei non pinnati. La
    matematica del forward resta identica all'originale (act=silu, chunk(2), pesi top_k).

    Args:
        model: OlmoeForCausalLM caricato su CPU (dtype es. bfloat16). NON usare device_map/offload di
            accelerate: la gestione device e' qui.
        capacity_per_layer: capienza (numero di slice di expert) della cache GPU di ciascun layer.
            Deve essere >= max_i |resident_by_layer[i]|. Con capacity == num_experts (64) e tutti gli
            expert toccati, dopo il warm-up non ci sono piu' miss (lossless always-resident).
        resident_by_layer: mappa layer_idx -> set di expert-id da PINNARE in quel layer (working-set).
            I layer non presenti ricevono un set pinnato vuoto (cache puramente LRU). Layer presenti
            con set vuoto -> nessun pin.
        device: device GPU target (default 'cuda').

    Returns:
        CacheHandle con ``.stats()`` (INTERFACCIA CONDIVISA) e ``.remove()`` per ripristinare i forward
        e liberare la VRAM. Supporta l'uso come context manager.
    """
    device = torch.device(device)
    cap = int(capacity_per_layer)

    layers = model.model.layers
    n_layers = len(layers)

    # dtype di calcolo = dtype del modello (es. bf16). Lo desumiamo da un parametro del backbone.
    compute_dtype = model.model.embed_tokens.weight.dtype

    # 1) backbone -> device, expert -> restano su CPU.
    _move_backbone_to_device(model, device)

    caches: dict[int, ExpertSliceCache] = {}
    originals: dict[int, Callable] = {}
    experts_modules: dict[int, nn.Module] = {}

    # 2) per ogni layer: backing CPU + cache GPU + monkeypatch del forward.
    for i in range(n_layers):
        experts = getattr(layers[i].mlp, "experts", None)
        if experts is None:
            continue  # layer DENSE (no MoE, es. DeepSeek-V2 layer 0): niente slice-cache
        num_experts = int(experts.num_experts)

        pinned = set(int(e) for e in resident_by_layer.get(i, set()))
        bad = [e for e in pinned if not 0 <= e < num_experts]
        if bad:
            raise ValueError(f"layer {i}: expert-id residenti fuori range [0,{num_experts}): {bad}")
        if cap < len(pinned):
            raise ValueError(
                f"layer {i}: capacity_per_layer ({cap}) < |resident| ({len(pinned)})"
            )

        # Backing: CPU (default) o iniettato (es. MemmapBacking su SSD per modelli > RAM).
        backing = (backing_factory(i, experts) if backing_factory is not None
                   else CpuParamBacking(experts, store_dtype=torch.float16))
        cache = ExpertSliceCache(
            backing=backing,
            capacity=cap,
            pinned=pinned,
            device=device,
            compute_dtype=compute_dtype,
        )

        # salva il forward d'istanza originale (se mai esistesse) per il ripristino; di norma e' di
        # classe, quindi 'forward' non e' nel __dict__ d'istanza -> remove() fara' del experts.forward.
        originals[i] = experts.__dict__.get("forward", type(experts).forward)
        experts_modules[i] = experts

        # override d'istanza: assegna un forward bound a QUESTO experts. Non tocca la classe (gli altri
        # layer/modelli restano intatti).
        experts.forward = _make_cached_forward(experts, cache)  # type: ignore[assignment]

        caches[i] = cache

    return CacheHandle(caches=caches, originals=originals, experts_modules=experts_modules)
