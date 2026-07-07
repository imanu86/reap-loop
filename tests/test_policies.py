"""test_policies.py — test ISOLATI e CPU-only per le tre policy di residenza.

Contratti verificati (docs/00_architecture.md §4):
  - FULL: tiene TUTTI gli expert per layer, committed_fraction == 1.0, ignora k_budget e la traccia.
  - COARSE: top-(k_budget) per FREQUENZA GLOBALE (corpus statico), NON dalla traccia di sessione.
  - AGGRESSIVE-COMMIT: stima il working set per-sessione (delegando a estimate_working_set) e
    committa, troncando a k_budget.

Isolamento: per AGGRESSIVE MONKEYPATCHIAMO estimate_working_set, così il test NON dipende
dall'implementazione reale del modulo workingset (sviluppato in parallelo da un altro agent).
Tutto deterministico, niente torch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from msc.policies import (
    AggressiveCommitPolicy,
    CoarseGrainedPolicy,
    FullPolicy,
    PolicyDecision,
)
import msc.workingset.estimator as ws_estimator


# --------------------------------------------------------------------------------------
# Fake / fixtures condivise
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeModelSpec:
    """Sostituto leggero del model_spec: espone solo i campi usati dalle policy."""

    n_total_experts: int
    top_k: int
    n_layers: int


class FakeTraceReader:
    """Trace reader fittizio. Per FULL/COARSE deve essere IGNORATO; se viene letto, esplode.

    Così il test fallisce esplicitamente se una policy che NON deve usare la sessione la legge.
    """

    def records(self, layer=None):  # pragma: no cover - non deve essere chiamato
        raise AssertionError("trace_reader non deve essere usato da questa policy")


@pytest.fixture
def spec():
    # OLMoE-like: 64 expert totali, top-8, ipotizziamo 4 layer per i test.
    return FakeModelSpec(n_total_experts=64, top_k=8, n_layers=4)


# --------------------------------------------------------------------------------------
# FULL
# --------------------------------------------------------------------------------------
def test_full_keeps_everything_ignoring_budget(spec):
    decision = FullPolicy().decide(
        trace_reader=FakeTraceReader(), k_budget=0.06, model_spec=spec
    )
    assert isinstance(decision, PolicyDecision)
    # committed_fraction == 1.0 a prescindere dal k_budget stringente.
    assert decision.committed_fraction == 1.0
    # Ogni layer tiene l'INTERO pool di expert.
    assert set(decision.per_layer_resident.keys()) == set(range(spec.n_layers))
    for layer in range(spec.n_layers):
        assert decision.per_layer_resident[layer] == set(range(spec.n_total_experts))


def test_full_name():
    assert FullPolicy().name == "FULL"


# --------------------------------------------------------------------------------------
# COARSE — frequenza GLOBALE, non la sessione
# --------------------------------------------------------------------------------------
def _make_global_freq(spec):
    """Costruisce una frequenza globale deterministica e distinta dalla 'sessione'.

    Per ogni layer, l'expert con id più alto è il più frequente, a scendere. Così il top-k globale
    è un insieme NOTO e diverso da quello che produrrebbe una traccia di sessione arbitraria.
    """
    freq = {}
    for layer in range(spec.n_layers):
        # expert e -> frequenza = e (id alti = più frequenti globalmente).
        freq[layer] = {e: float(e) for e in range(spec.n_total_experts)}
    return freq


def test_coarse_uses_global_freq_not_session(spec):
    k_budget = 0.125  # -> ceil(0.125*64) = 8 expert per layer
    global_freq = _make_global_freq(spec)

    policy = CoarseGrainedPolicy(global_freq=global_freq)
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=k_budget, model_spec=spec)

    expected_keep = math.ceil(k_budget * spec.n_total_experts)
    assert expected_keep == 8

    # I top-8 per frequenza globale sono gli id più alti: 56..63 (perché freq = id).
    expected_top = set(range(spec.n_total_experts - expected_keep, spec.n_total_experts))
    for layer in range(spec.n_layers):
        assert decision.per_layer_resident[layer] == expected_top

    # committed_fraction = keep / n_total.
    assert decision.committed_fraction == pytest.approx(expected_keep / spec.n_total_experts)
    assert decision.committed_fraction < 1.0


def test_coarse_ignores_session_trace(spec):
    # Se COARSE leggesse la traccia di sessione, FakeTraceReader.records() solleverebbe.
    global_freq = _make_global_freq(spec)
    policy = CoarseGrainedPolicy(global_freq=global_freq)
    # Non deve sollevare: la traccia è deliberatamente ignorata.
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=0.25, model_spec=spec)
    assert isinstance(decision, PolicyDecision)


def test_coarse_loads_from_json_file(tmp_path, spec):
    """COARSE legge un file JSON pre-calcolato (corpus globale/statico)."""
    global_freq = _make_global_freq(spec)
    # Su file le chiavi diventano stringhe: verifichiamo che il loader le normalizzi.
    serializable = {str(layer): {str(e): f for e, f in m.items()} for layer, m in global_freq.items()}
    freq_path = tmp_path / "global_expert_freq.json"
    freq_path.write_text(json.dumps(serializable), encoding="utf-8")

    policy = CoarseGrainedPolicy(global_freq_path=str(freq_path))
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=0.125, model_spec=spec)

    expected_top = set(range(spec.n_total_experts - 8, spec.n_total_experts))
    for layer in range(spec.n_layers):
        assert decision.per_layer_resident[layer] == expected_top


def test_coarse_requires_a_global_freq(spec):
    policy = CoarseGrainedPolicy()  # né dict né path
    with pytest.raises(ValueError):
        policy.decide(trace_reader=FakeTraceReader(), k_budget=0.5, model_spec=spec)


def test_coarse_name():
    assert CoarseGrainedPolicy(global_freq={}).name == "COARSE"


# --------------------------------------------------------------------------------------
# AGGRESSIVE-COMMIT — monkeypatch di estimate_working_set
# --------------------------------------------------------------------------------------
def _fake_estimate_factory(per_layer_working_set, committed_fraction):
    """Ritorna una fake estimate_working_set che restituisce un oggetto con i campi attesi.

    Cattura gli argomenti dell'ultima chiamata in `calls` per asserzioni sul contratto.
    """
    calls = []

    @dataclass(frozen=True)
    class FakeEstimate:
        per_layer_working_set: dict
        committed_fraction: float

    def fake(trace_reader, theta, k_budget=None):
        calls.append({"trace_reader": trace_reader, "theta": theta, "k_budget": k_budget})
        return FakeEstimate(
            per_layer_working_set=per_layer_working_set,
            committed_fraction=committed_fraction,
        )

    fake.calls = calls
    return fake


def test_aggressive_commits_estimated_working_set(monkeypatch, spec):
    # Working set "concentrato" stimato dalla sessione: pochi expert per layer.
    per_layer_ws = {
        0: (3, 7, 11),
        1: (0, 1),
        2: (63, 62, 5, 9),
        3: (10,),
    }
    fake = _fake_estimate_factory(per_layer_ws, committed_fraction=0.05)
    monkeypatch.setattr(ws_estimator, "estimate_working_set", fake)

    policy = AggressiveCommitPolicy(coverage_theta=0.95)
    k_budget = 0.5  # tetto largo: 32 expert/layer -> nessun troncamento qui
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=k_budget, model_spec=spec)

    # Ha delegato all'estimator con la traccia di sessione e theta della policy.
    assert len(fake.calls) == 1
    assert fake.calls[0]["theta"] == pytest.approx(0.95)
    assert fake.calls[0]["k_budget"] == pytest.approx(k_budget)

    # Senza troncamento, i residenti coincidono col working set stimato.
    assert decision.per_layer_resident[0] == {3, 7, 11}
    assert decision.per_layer_resident[1] == {0, 1}
    assert decision.per_layer_resident[2] == {63, 62, 5, 9}
    assert decision.per_layer_resident[3] == {10}

    # committed_fraction = totale residenti / (n_layers * n_total).
    total = 3 + 2 + 4 + 1
    assert decision.committed_fraction == pytest.approx(total / (spec.n_layers * spec.n_total_experts))


def test_aggressive_respects_k_budget_truncation(monkeypatch, spec):
    # L'estimator (per qualunque ragione) restituisce working set più larghi del budget.
    per_layer_ws = {layer: tuple(range(20)) for layer in range(spec.n_layers)}  # 20 expert/layer
    fake = _fake_estimate_factory(per_layer_ws, committed_fraction=20 / spec.n_total_experts)
    monkeypatch.setattr(ws_estimator, "estimate_working_set", fake)

    k_budget = 0.125  # ceil(0.125*64) = 8 -> tetto stringente
    max_keep = math.ceil(k_budget * spec.n_total_experts)
    assert max_keep == 8

    policy = AggressiveCommitPolicy()
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=k_budget, model_spec=spec)

    # Ogni layer è troncato a max_keep (i primi del working set, che è ordinato per frequenza desc).
    for layer in range(spec.n_layers):
        resident = decision.per_layer_resident[layer]
        assert len(resident) == max_keep
        assert resident == set(range(max_keep))

    # committed_fraction effettiva rispetta il budget (non lo sfora).
    assert decision.committed_fraction <= k_budget + 1e-9
    assert decision.committed_fraction == pytest.approx(max_keep / spec.n_total_experts)


def test_aggressive_never_exceeds_budget_even_with_huge_ws(monkeypatch, spec):
    # Working set = tutti gli expert: deve comunque essere troncato al budget.
    per_layer_ws = {layer: tuple(range(spec.n_total_experts)) for layer in range(spec.n_layers)}
    fake = _fake_estimate_factory(per_layer_ws, committed_fraction=1.0)
    monkeypatch.setattr(ws_estimator, "estimate_working_set", fake)

    k_budget = 0.06  # ceil(0.06*64) = 4
    max_keep = math.ceil(k_budget * spec.n_total_experts)
    policy = AggressiveCommitPolicy()
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=k_budget, model_spec=spec)

    for layer in range(spec.n_layers):
        assert len(decision.per_layer_resident[layer]) == max_keep
    assert decision.committed_fraction == pytest.approx(max_keep / spec.n_total_experts)
    assert decision.committed_fraction < 1.0


def test_aggressive_empty_layer_gets_at_least_one(monkeypatch, spec):
    # Se l'estimator non copre un layer (working set vuoto), la policy garantisce >=1 residente.
    per_layer_ws = {0: (1, 2)}  # layer 1,2,3 assenti
    fake = _fake_estimate_factory(per_layer_ws, committed_fraction=0.01)
    monkeypatch.setattr(ws_estimator, "estimate_working_set", fake)

    policy = AggressiveCommitPolicy()
    decision = policy.decide(trace_reader=FakeTraceReader(), k_budget=0.5, model_spec=spec)

    for layer in range(spec.n_layers):
        assert len(decision.per_layer_resident[layer]) >= 1


def test_aggressive_name():
    assert AggressiveCommitPolicy().name == "AGGRESSIVE-COMMIT"
