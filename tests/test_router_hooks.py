"""test_router_hooks.py — test CPU-only per l'harness di logging del router.

Verifica (senza torch):
    1. RouterHookSpec.for_model ritorna le spec verificate per i 5 modelli target,
       incluso first_moe_layer=1 per DeepSeek-V2-Lite e 0 per gli altri.
    2. RouterLogger._iter_gate_modules naviga un GRAFO FINTO (SimpleNamespace) e trova i
       moduli gate giusti, rispettando first_moe_layer.
    3. La logica top-k del hook su router_logits numpy sintetici produce id/pesi attesi
       (top-k corretto, ordinamento per probabilità decrescente, rinormalizzazione se
       norm_topk_prob).

Tutto deterministico e isolato: il writer è un semplice collector in-memory, il modello è
un SimpleNamespace. Nessuna dipendenza da torch/transformers o da altri moduli implementati
in parallelo (a parte ActivationRecord/dataclass, parte dell'interfaccia condivisa).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from msc.instrument.router_hooks import RouterHookSpec, RouterLogger, _topk_from_logits


# ---------------------------------------------------------------------------
# 1. Tabella spec verificata per modello
# ---------------------------------------------------------------------------

def test_for_model_olmoe():
    spec = RouterHookSpec.for_model("allenai/OLMoE-1B-7B-0924")
    assert spec.gate_attr_path == "mlp.gate"
    assert spec.first_moe_layer == 0
    assert spec.topk == 8
    assert spec.norm_topk_prob is False


def test_for_model_granite_3b():
    spec = RouterHookSpec.for_model("ibm-granite/granite-3.1-3b-a800m-instruct")
    # Hook sul sotto-modulo nn.Linear (.router.layer): output = logit grezzi [n_token, 40].
    # `block_sparse_moe.router` (GraniteMoeTopKGating) ritorna una tupla coi logit ULTIMI.
    assert spec.gate_attr_path == "block_sparse_moe.router.layer"
    assert spec.first_moe_layer == 0
    assert spec.topk == 8
    # Granite: softmax sui soli top-k logit → equivale a rinormalizzazione.
    assert spec.norm_topk_prob is True


def test_for_model_granite_1b():
    spec = RouterHookSpec.for_model("ibm-granite/granite-3.1-1b-a400m-instruct")
    assert spec.gate_attr_path == "block_sparse_moe.router.layer"
    assert spec.first_moe_layer == 0
    assert spec.topk == 8
    assert spec.norm_topk_prob is True


def test_for_model_deepseek_first_moe_layer_is_one():
    spec = RouterHookSpec.for_model("deepseek-ai/DeepSeek-V2-Lite")
    assert spec.gate_attr_path == "mlp.gate"
    # Il caso chiave: layer 0 è dense → primo layer MoE = 1.
    assert spec.first_moe_layer == 1
    assert spec.topk == 6
    assert spec.norm_topk_prob is False


def test_for_model_qwen():
    spec = RouterHookSpec.for_model("Qwen/Qwen1.5-MoE-A2.7B")
    assert spec.gate_attr_path == "mlp.gate"
    assert spec.first_moe_layer == 0
    assert spec.topk == 4
    assert spec.norm_topk_prob is False


def test_for_model_unknown_raises():
    with pytest.raises(KeyError):
        RouterHookSpec.for_model("not/a-real-model")


# ---------------------------------------------------------------------------
# 2. _iter_gate_modules su un grafo finto
# ---------------------------------------------------------------------------

def _fake_model_mlp_gate(n_layers: int):
    """Grafo finto stile OLMoE/DeepSeek: layer[i].mlp.gate è un modulo riconoscibile."""
    layers = []
    for i in range(n_layers):
        gate = SimpleNamespace(_marker=("gate", i))
        layer = SimpleNamespace(mlp=SimpleNamespace(gate=gate))
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return model


def _fake_model_fused_router(n_layers: int):
    """Grafo finto stile Granite: layer[i].block_sparse_moe.router."""
    layers = []
    for i in range(n_layers):
        router = SimpleNamespace(_marker=("router", i))
        layer = SimpleNamespace(block_sparse_moe=SimpleNamespace(router=router))
        layers.append(layer)
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    return model


def test_iter_gate_modules_mlp_gate_all_layers():
    model = _fake_model_mlp_gate(4)
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=0, topk=8)
    logger = RouterLogger(model, spec, writer=None)

    found = list(logger._iter_gate_modules())
    assert [idx for idx, _ in found] == [0, 1, 2, 3]
    # Ogni gate trovato è esattamente il modulo del layer corrispondente.
    for idx, gate in found:
        assert gate._marker == ("gate", idx)
        assert gate is model.model.layers[idx].mlp.gate


def test_iter_gate_modules_skips_dense_layer_deepseek():
    model = _fake_model_mlp_gate(4)
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=1, topk=6)
    logger = RouterLogger(model, spec, writer=None)

    found = list(logger._iter_gate_modules())
    # Layer 0 (dense) deve essere saltato.
    assert [idx for idx, _ in found] == [1, 2, 3]
    assert found[0][1] is model.model.layers[1].mlp.gate


def test_iter_gate_modules_fused_router_path():
    model = _fake_model_fused_router(3)
    spec = RouterHookSpec(gate_attr_path="block_sparse_moe.router", first_moe_layer=0, topk=8)
    logger = RouterLogger(model, spec, writer=None)

    found = list(logger._iter_gate_modules())
    assert [idx for idx, _ in found] == [0, 1, 2]
    for idx, router in found:
        assert router._marker == ("router", idx)


# ---------------------------------------------------------------------------
# 3. Logica top-k del hook su logits numpy sintetici
# ---------------------------------------------------------------------------

def test_topk_selects_highest_and_orders_descending():
    # 5 expert; logit grezzi crescenti → top-2 sono gli id 4 e 3, in quest'ordine.
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ids, weights = _topk_from_logits(logits, topk=2, norm_topk_prob=False)
    assert ids == (4, 3)
    # I pesi sono softmax(su tutti) dei selezionati, in ordine decrescente.
    assert weights[0] > weights[1]
    # Senza rinormalizzazione i pesi NON sommano a 1 (sono prob su tutti i 5 expert).
    assert sum(weights) < 1.0


def test_topk_weights_match_global_softmax_no_norm():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ids, weights = _topk_from_logits(logits, topk=2, norm_topk_prob=False)
    # Confronto esplicito col softmax di riferimento.
    full = np.exp(logits - logits.max())
    full = full / full.sum()
    assert ids == (4, 3)
    np.testing.assert_allclose(weights, [full[4], full[3]], rtol=1e-12, atol=1e-12)


def test_topk_renormalization_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ids, weights = _topk_from_logits(logits, topk=3, norm_topk_prob=True)
    assert ids == (4, 3, 2)
    # Con rinormalizzazione i k pesi sommano a 1.
    assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    # E sono proporzionali ai softmax globali dei selezionati.
    full = np.exp(logits - logits.max())
    full = full / full.sum()
    expected = np.array([full[4], full[3], full[2]])
    expected = expected / expected.sum()
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)


def test_topk_tie_break_smallest_id_first():
    # Logit tutti uguali → prob uguali; il tie-break deve essere deterministico (id crescente).
    logits = np.array([5.0, 5.0, 5.0, 5.0])
    ids, weights = _topk_from_logits(logits, topk=2, norm_topk_prob=False)
    assert ids == (0, 1)
    np.testing.assert_allclose(weights, [0.25, 0.25], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Hook end-to-end (senza torch): scrive un ActivationRecord per token
# ---------------------------------------------------------------------------

class _CollectingWriter:
    """Writer fittizio: accumula i record in memoria (mock dell'interfaccia TraceWriter)."""

    def __init__(self):
        self.records = []

    def write(self, rec):
        self.records.append(rec)


def test_make_hook_writes_one_record_per_token():
    model = _fake_model_mlp_gate(2)
    # config finto così _model_id() ritorna un valore.
    model.config = SimpleNamespace(name_or_path="fake/model")
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=0, topk=2)
    writer = _CollectingWriter()
    logger = RouterLogger(model, spec, writer)

    # Imposta lo stato di sessione come farebbe capture().
    logger._session_id = "sess-1"
    logger._ctx_len = 1024
    logger._step_per_layer = {}

    hook = logger._make_hook(layer_idx=3)
    # router_logits sintetici: 2 token, 5 expert.
    logits = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # token 0: top-2 = (4, 3)
            [9.0, 0.0, 0.0, 0.0, 8.0],  # token 1: top-2 = (0, 4)
        ]
    )
    hook(module=None, inputs=None, output=logits)

    assert len(writer.records) == 2
    r0, r1 = writer.records
    assert r0.session_id == "sess-1"
    assert r0.ctx_len == 1024
    assert r0.model_id == "fake/model"
    assert r0.layer == 3
    assert r0.token_pos == 0
    assert r0.topk_ids == (4, 3)
    assert r1.token_pos == 1
    assert r1.topk_ids == (0, 4)
    # Stesso step per entrambi i token dello stesso forward.
    assert r0.step == 0 and r1.step == 0


def test_make_hook_increments_step_across_forwards():
    model = _fake_model_mlp_gate(1)
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=0, topk=1)
    writer = _CollectingWriter()
    logger = RouterLogger(model, spec, writer)
    logger._session_id = "s"
    logger._ctx_len = 8
    logger._step_per_layer = {}

    hook = logger._make_hook(layer_idx=0)
    single = np.array([0.1, 0.9, 0.2])  # 1-D → 1 token
    hook(None, None, single)
    hook(None, None, single)

    assert [r.step for r in writer.records] == [0, 1]
    assert all(r.topk_ids == (1,) for r in writer.records)


def test_make_hook_handles_tuple_output():
    """Alcuni router HF ritornano (logits, ...); l'hook deve prendere il primo elemento."""
    model = _fake_model_mlp_gate(1)
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=0, topk=2)
    writer = _CollectingWriter()
    logger = RouterLogger(model, spec, writer)
    logger._session_id = "s"
    logger._ctx_len = 8
    logger._step_per_layer = {}

    hook = logger._make_hook(layer_idx=0)
    logits = np.array([[1.0, 5.0, 3.0]])
    hook(None, None, (logits, "altro"))

    assert len(writer.records) == 1
    assert writer.records[0].topk_ids == (1, 2)


# ---------------------------------------------------------------------------
# 5. Parti torch-only: skippate, non fallite
# ---------------------------------------------------------------------------

def test_capture_registers_and_removes_hooks_torch_optional():
    """Il context manager capture() usa register_forward_hook: con un gate finto che lo
    implementa possiamo testarlo anche senza torch (l'API è un duck-type)."""

    class _FakeHandle:
        def __init__(self, store, fn):
            self._store = store
            self._fn = fn
            self.removed = False

        def remove(self):
            self.removed = True

    class _FakeGate:
        def __init__(self):
            self.hooks = []

        def register_forward_hook(self, fn):
            handle = _FakeHandle(self.hooks, fn)
            self.hooks.append(handle)
            return handle

    # Grafo finto con gate che espongono register_forward_hook.
    layers = [SimpleNamespace(mlp=SimpleNamespace(gate=_FakeGate())) for _ in range(3)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    spec = RouterHookSpec(gate_attr_path="mlp.gate", first_moe_layer=1, topk=2)
    writer = _CollectingWriter()
    logger = RouterLogger(model, spec, writer)

    with logger.capture(session_id="abc", ctx_len=4096):
        # Dentro al contesto: hook registrati solo sui layer MoE (1 e 2 → first_moe_layer=1).
        assert len(logger._handles) == 2
        assert logger._session_id == "abc"
        assert logger._ctx_len == 4096

    # Fuori dal contesto: tutti gli handle rimossi e stato ripulito.
    assert logger._handles == []
    assert logger._session_id is None
    for layer in layers[1:]:
        assert layer.mlp.gate.hooks[0].removed is True
    # Layer 0 (dense) non deve aver ricevuto hook.
    assert layers[0].mlp.gate.hooks == []
