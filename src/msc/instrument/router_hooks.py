"""router_hooks.py — logging della top-k degli expert, per layer e per token.

STATO: IMPLEMENTATO. Interfacce + ricette di hook verificate per modello (vedi docs/02_models.md §4).

Idea: registriamo un forward-hook sul modulo *gate/router* di OGNI layer MoE. Per uniformità tra
modelli usiamo SEMPRE l'hook sui logits (non il flag `output_router_logits`), perché DeepSeek-V2-Lite
non espone quel flag (vedi docs/02_models.md). Dai logits ricaviamo top-k id + gate weights.

Ricette verificate (modulo gate per layer i):
    OLMoE            : model.model.layers[i].mlp.gate                       (nn.Linear -> logits)
    Granite 3B/1B    : model.model.layers[i].block_sparse_moe.router.layer  (nn.Linear -> logits)
    DeepSeek-V2-Lite : model.model.layers[i].mlp.gate   (layer 0 è dense → skip)
    Qwen1.5-MoE      : model.model.layers[i].mlp.gate

VERIFICATO sul modello vero (Granite-3B, transformers 5.x): `block_sparse_moe.router` è un
GraniteMoeTopKGating che ritorna una TUPLA con i logit come ULTIMO elemento (output[0] sarebbe gli
indici, sbagliato). Hookiamo quindi il sotto-modulo `.router.layer` (l'nn.Linear, out=40), il cui
output sono i logit grezzi [n_token, 40] — uniforme a OLMoE dove `mlp.gate` è già l'nn.Linear.

Nota sulla rinormalizzazione (`norm_topk_prob`):
    - OLMoE / DeepSeek-V2-Lite / Qwen1.5-MoE: router = softmax su TUTTI i logits, poi top-k; i loro
      config riportano `norm_topk_prob=false` → i pesi della top-k NON vengono rinormalizzati.
    - Granite 3B/1B: il `GraniteMoeTopKGating` applica softmax SOLO sui top-k logits → equivale a
      "seleziona top-k dai logit grezzi e rinormalizza a somma 1". In termini della nostra
      ricostruzione: `norm_topk_prob=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from msc.instrument.trace import ActivationRecord


# Tabella verificata da config.json / model card ufficiali (vedi docs/02_models.md §1 e §4).
# Chiave: model_id HF. Valori: (gate_attr_path, first_moe_layer, topk, norm_topk_prob).
_MODEL_TABLE: dict[str, tuple[str, int, int, bool]] = {
    "allenai/OLMoE-1B-7B-0924": ("mlp.gate", 0, 8, False),
    "allenai/OLMoE-1B-7B-0924-Instruct": ("mlp.gate", 0, 8, False),  # stessa arch del base
    "ibm-granite/granite-3.1-3b-a800m-instruct": ("block_sparse_moe.router.layer", 0, 8, True),
    "ibm-granite/granite-3.1-1b-a400m-instruct": ("block_sparse_moe.router.layer", 0, 8, True),
    "deepseek-ai/DeepSeek-V2-Lite": ("mlp.gate", 1, 6, False),
    "Qwen/Qwen1.5-MoE-A2.7B": ("mlp.gate", 0, 4, False),
    # Qwen3-MoE: Qwen3MoeExperts identico a OlmoeExperts; gate = Qwen3MoeTopKRouter (mlp.gate) che
    # ritorna (router_logits, scores, indices) -> output[0] = logits [n_token, 128]. norm_topk_prob=True.
    "Qwen/Qwen3-30B-A3B": ("mlp.gate", 0, 8, True),
}


@dataclass(frozen=True)
class RouterHookSpec:
    """Come raggiungere il router di un dato modello.

    Attributes:
        gate_attr_path: attributo relativo al layer che individua il gate
            (es. "mlp.gate" oppure "block_sparse_moe.router").
        first_moe_layer: primo indice di layer che è MoE (DeepSeek: 1, gli altri: 0).
        topk: numero di expert attivi per token (per ricostruire la selezione dai logits).
        norm_topk_prob: se i gate weights vanno rinormalizzati dopo il top-k.
    """

    gate_attr_path: str
    first_moe_layer: int
    topk: int
    norm_topk_prob: bool = False

    @staticmethod
    def for_model(model_id: str) -> "RouterHookSpec":
        """Ritorna la spec verificata per uno dei modelli supportati.

        Tabella per {OLMoE, Granite-3B, Granite-1B, DeepSeek-V2-Lite, Qwen1.5-MoE}.
        """
        try:
            gate_attr_path, first_moe_layer, topk, norm_topk_prob = _MODEL_TABLE[model_id]
        except KeyError:
            supportati = ", ".join(sorted(_MODEL_TABLE))
            raise KeyError(
                f"model_id non supportato: {model_id!r}. Modelli noti: {supportati}"
            ) from None
        return RouterHookSpec(
            gate_attr_path=gate_attr_path,
            first_moe_layer=first_moe_layer,
            topk=topk,
            norm_topk_prob=norm_topk_prob,
        )


def _getattr_path(obj: object, path: str) -> object:
    """Naviga `obj.<a>.<b>.<c>` dato path="a.b.c" via getattr annidato."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _topk_from_logits(
    logits, topk: int, norm_topk_prob: bool
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Da un vettore di router_logits di UN token estrae (top-k id, top-k gate weights).

    Pipeline (uniforme ai router HF dei modelli target):
        1. softmax sui logit grezzi → probabilità su TUTTI gli expert;
        2. seleziona i top-k per probabilità decrescente (tie-break: id più piccolo);
        3. se `norm_topk_prob`, rinormalizza i k pesi a somma 1.

    Lavora su ARRAY-LIKE numpy (np.ndarray 1-D) così è testabile senza torch.
    """
    import numpy as np

    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    # softmax stabile
    shifted = arr - arr.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    # top-k per probabilità decrescente; tie-break deterministico sull'id (più piccolo prima).
    # argsort è stabile (kind="stable") quindi a parità di prob mantiene l'ordine crescente di id.
    order = np.argsort(-probs, kind="stable")
    sel = order[:topk]

    ids = tuple(int(i) for i in sel)
    weights = probs[sel]
    if norm_topk_prob:
        total = weights.sum()
        if total > 0:
            weights = weights / total
    gate_w = tuple(float(w) for w in weights)
    return ids, gate_w


class RouterLogger:
    """Attacca forward-hook ai gate e accumula la top-k per (layer, token).

    Uso previsto:
        logger = RouterLogger(model, spec, writer)
        with logger.capture(session_id=..., ctx_len=...):
            model.generate(...)          # gli hook scrivono ActivationRecord nel writer
    """

    def __init__(self, model, spec: RouterHookSpec, writer) -> None:
        self._model = model
        self._spec = spec
        self._writer = writer
        self._handles: list = []
        # Stato di sessione, impostato da capture() e letto dagli hook.
        self._session_id: str | None = None
        self._ctx_len: int | None = None
        # Contatore di forward (step) per ciascun layer: serve per il campo `step`.
        self._step_per_layer: dict[int, int] = {}

    def _iter_gate_modules(self) -> Iterable[tuple[int, object]]:
        """Naviga model.model.layers[i].<gate_attr_path> per i layer MoE.

        Restituisce coppie (layer_idx, gate_module) solo per i >= first_moe_layer.
        """
        layers = self._model.model.layers
        for layer_idx, layer in enumerate(layers):
            if layer_idx < self._spec.first_moe_layer:
                continue
            gate = _getattr_path(layer, self._spec.gate_attr_path)
            yield layer_idx, gate

    def _make_hook(self, layer_idx: int) -> Callable:
        """Costruisce l'hook che da router_logits estrae top-k id + pesi e li scrive.

        L'hook usa l'output del modulo gate come `router_logits`: tensore di forma
        [n_token, n_expert] (o [n_expert] per un singolo token). Per ogni token scrive un
        ActivationRecord nel writer.

        Nota: per i modelli a expert FUSI (Granite) l'hook sul router è l'UNICO punto
        di osservazione per-token (non esistono moduli expert separati da hookare).
        """

        def hook(module, inputs, output):
            import numpy as np

            # Alcuni router HF ritornano una tupla (logits, ...); prendiamo il primo elemento.
            logits = output[0] if isinstance(output, tuple) else output

            # Materializza un array numpy 2-D [n_token, n_expert] indipendente da torch.
            if hasattr(logits, "detach"):  # tensore torch
                logits = logits.detach().to("cpu").float().numpy()
            arr = np.asarray(logits, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            step = self._step_per_layer.get(layer_idx, 0)
            for token_pos in range(arr.shape[0]):
                ids, gate_w = _topk_from_logits(
                    arr[token_pos], self._spec.topk, self._spec.norm_topk_prob
                )
                rec = ActivationRecord(
                    session_id=self._session_id or "",
                    step=step,
                    layer=layer_idx,
                    token_pos=token_pos,
                    topk_ids=ids,
                    gate_w=gate_w,
                    ctx_len=self._ctx_len if self._ctx_len is not None else 0,
                    model_id=self._model_id(),
                )
                self._writer.write(rec)
            self._step_per_layer[layer_idx] = step + 1

        return hook

    def _model_id(self) -> str:
        """Best-effort: ricava il model_id dal config del modello, altrimenti stringa vuota."""
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            for attr in ("name_or_path", "_name_or_path"):
                val = getattr(cfg, attr, None)
                if val:
                    return str(val)
        return ""

    def capture(self, session_id: str, ctx_len: int):
        """Context manager: registra gli hook all'ingresso, li rimuove all'uscita.

        L'import di torch (register_forward_hook) avviene QUI, non a livello di modulo, così
        `import msc.instrument.router_hooks` funziona anche senza torch installato.
        """
        logger = self

        class _Capture:
            def __enter__(self):
                logger._session_id = session_id
                logger._ctx_len = ctx_len
                logger._step_per_layer = {}
                logger._handles = []
                for layer_idx, gate in logger._iter_gate_modules():
                    handle = gate.register_forward_hook(logger._make_hook(layer_idx))
                    logger._handles.append(handle)
                return logger

            def __exit__(self, exc_type, exc, tb):
                for handle in logger._handles:
                    handle.remove()
                logger._handles = []
                logger._session_id = None
                logger._ctx_len = None
                return False

        return _Capture()
