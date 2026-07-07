"""test_residency.py — test CPU-only del modulo residency (manager + miss_modes).

Tutto torch-free e deterministico: usa hidden_state fittizi (oggetti con .shape) e callable di
calcolo finti per il gpu_seam. Verifica:
  - commit fissa i residenti giusti per layer + contabilità VRAM;
  - on_route su residente -> hit; su non-residente -> miss con outcome dipendente dal miss_mode;
  - fetch-lossless: served=True e pcie_bytes>0;
  - precision-cascade: precision_bits in {low_bits, low_bits/2} secondo le soglie, skip se irrilevante;
  - hard-drop zero: served=False; reroute: rerouted_to valorizzato;
  - miss_rate calcolato bene.
I test che richiederebbero veri tensori torch sono guardati da pytest.importorskip("torch").
"""

from __future__ import annotations

import pytest

from msc.residency.manager import (
    FusedSliceStore,
    ModuleListStore,
    ResidencyManager,
    ResidencyStats,
)
from msc.residency.miss_modes import (
    FetchLossless,
    HardDrop,
    MissMode,
    MissOutcome,
    PrecisionCascade,
    make_miss_handler,
)


class FakeHidden:
    """Hidden state fittizio: espone solo .shape, sufficiente per la stima dei byte (torch-free)."""

    def __init__(self, shape=(1, 8)):
        self.shape = shape


# byte/expert fp16 arbitrario ma deterministico per la contabilità VRAM.
BPE = 1_000


def make_store(n_layers=2, n_experts=8):
    return ModuleListStore(n_layers=n_layers, n_experts=n_experts, bytes_per_expert_fp16=BPE)


# --------------------------------------------------------------------------------------
# commit + residenza
# --------------------------------------------------------------------------------------
def test_commit_sets_correct_residents():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {1, 3, 5}, 1: [2, 4]})

    assert store.resident_ids(0) == {1, 3, 5}
    assert store.resident_ids(1) == {2, 4}
    assert mgr.committed[0] == {1, 3, 5}
    assert mgr.committed[1] == {2, 4}

    # is_resident coerente con il set committato.
    assert store.is_resident(0, 1) is True
    assert store.is_resident(0, 2) is False
    assert store.is_resident(1, 4) is True


def test_commit_accounts_resident_vram():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {1, 3, 5}, 1: {2, 4}})
    # 5 expert residenti totali * BPE.
    assert mgr.stats.resident_vram_bytes == 5 * BPE
    assert store.resident_vram_bytes() == 5 * BPE


def test_recommit_replaces_previous_residents():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {1, 2, 3}})
    mgr.commit({0: {7}})
    assert store.resident_ids(0) == {7}
    assert mgr.stats.resident_vram_bytes == 1 * BPE


# --------------------------------------------------------------------------------------
# on_route: hit
# --------------------------------------------------------------------------------------
def test_on_route_hit_counts_and_no_outcome():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {1, 2}})

    out, outcome = mgr.on_route(0, 1, FakeHidden(), gate_weight=0.5)
    assert outcome is None            # hit: nessun MissOutcome
    assert mgr.stats.hit_count == 1
    assert mgr.stats.miss_count == 0


def test_on_route_hit_invokes_gpu_seam():
    store = make_store()
    calls = []

    def fake_hit(layer, eid, h):
        calls.append((layer, eid))
        return "expert_out"

    mgr = ResidencyManager(store=store, miss_handler=FetchLossless(), hit_compute_fn=fake_hit)
    mgr.commit({0: {3}})
    out, outcome = mgr.on_route(0, 3, FakeHidden(), gate_weight=0.9)
    assert out == "expert_out"
    assert calls == [(0, 3)]


# --------------------------------------------------------------------------------------
# on_route: miss -> fetch-lossless
# --------------------------------------------------------------------------------------
def test_miss_fetch_lossless_served_and_pcie():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {1}})

    out, outcome = mgr.on_route(0, 7, FakeHidden(shape=(2, 16)), gate_weight=0.3)
    assert isinstance(outcome, MissOutcome)
    assert outcome.served is True
    assert outcome.pcie_bytes > 0            # lossless => trasferimento RAM->VRAM
    assert outcome.precision_bits == 16
    assert mgr.stats.miss_count == 1
    assert mgr.stats.pcie_bytes == outcome.pcie_bytes
    assert mgr.stats.drop_count == 0
    assert mgr.stats.reroute_count == 0


# --------------------------------------------------------------------------------------
# on_route: miss -> precision-cascade (soglie HOBBIT)
# --------------------------------------------------------------------------------------
def test_miss_cascade_high_importance_full_low_bits():
    # gate_weight alto (importante) => unimportance bassa => low_bits pieni (4).
    store = make_store()
    handler = PrecisionCascade(low_bits=4, t_low=0.6, t_skip=0.9)
    mgr = ResidencyManager(store=store, miss_handler=handler)
    mgr.commit({0: {0}})

    out, outcome = mgr.on_route(0, 5, FakeHidden(), gate_weight=0.95)  # u=0.05 < t_low
    assert outcome.served is True
    assert outcome.precision_bits == 4
    assert outcome.pcie_bytes == 0           # copia low-bit già in VRAM


def test_miss_cascade_mid_importance_halved_bits():
    # importanza media => unimportance in [t_low, t_skip) => low_bits/2 = 2.
    store = make_store()
    handler = PrecisionCascade(low_bits=4, t_low=0.6, t_skip=0.9)
    mgr = ResidencyManager(store=store, miss_handler=handler)
    mgr.commit({0: {0}})

    out, outcome = mgr.on_route(0, 5, FakeHidden(), gate_weight=0.3)  # u=0.7 in [0.6,0.9)
    assert outcome.served is True
    assert outcome.precision_bits == 2


def test_miss_cascade_precision_bits_in_4_2():
    handler = PrecisionCascade(low_bits=4, t_low=0.6, t_skip=0.9)
    # campioniamo gate_weight su tutto il range servito (escludendo lo skip): bits sempre in {4,2}.
    for gw in (1.0, 0.8, 0.6, 0.41, 0.4, 0.2, 0.11):
        _, outcome = handler.handle(0, 1, FakeHidden(), gate_weight=gw)
        if outcome.served:
            assert outcome.precision_bits in (4, 2)


def test_miss_cascade_skip_when_irrelevant():
    # gate_weight molto basso (irrilevante) => unimportance >= t_skip => skip (served=False).
    store = make_store()
    handler = PrecisionCascade(low_bits=4, t_low=0.6, t_skip=0.9)
    mgr = ResidencyManager(store=store, miss_handler=handler)
    mgr.commit({0: {0}})

    out, outcome = mgr.on_route(0, 5, FakeHidden(), gate_weight=0.05)  # u=0.95 >= t_skip
    assert outcome.served is False
    assert outcome.precision_bits is None
    assert mgr.stats.drop_count == 1          # skip conta come drop effettivo


# --------------------------------------------------------------------------------------
# on_route: miss -> hard-drop
# --------------------------------------------------------------------------------------
def test_miss_hard_drop_zero_not_served():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=HardDrop(variant="zero"))
    mgr.commit({0: {1}})

    out, outcome = mgr.on_route(0, 7, FakeHidden(), gate_weight=0.8)
    assert outcome.served is False
    assert outcome.rerouted_to is None
    assert outcome.pcie_bytes == 0
    assert mgr.stats.drop_count == 1
    assert mgr.stats.reroute_count == 0


def test_miss_hard_drop_reroute_to_nearest_resident():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=HardDrop(variant="reroute"))
    mgr.commit({0: {2, 6}})

    # expert 7 non residente: residente più vicino per id è 6.
    out, outcome = mgr.on_route(0, 7, FakeHidden(), gate_weight=0.8)
    assert outcome.served is True
    assert outcome.rerouted_to == 6
    assert mgr.stats.reroute_count == 1
    assert mgr.stats.drop_count == 0


def test_miss_hard_drop_reroute_falls_back_to_zero_without_residents():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=HardDrop(variant="reroute"))
    mgr.commit({0: set()})  # nessun residente nel layer 0

    out, outcome = mgr.on_route(0, 3, FakeHidden(), gate_weight=0.8)
    assert outcome.served is False
    assert outcome.rerouted_to is None
    assert mgr.stats.drop_count == 1


# --------------------------------------------------------------------------------------
# miss_rate
# --------------------------------------------------------------------------------------
def test_miss_rate_computed_correctly():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=FetchLossless())
    mgr.commit({0: {0, 1}})

    h = FakeHidden()
    # 3 hit (expert 0/1 residenti) e 2 miss (expert 5/6 non residenti).
    mgr.on_route(0, 0, h, 0.5)
    mgr.on_route(0, 1, h, 0.5)
    mgr.on_route(0, 0, h, 0.5)
    mgr.on_route(0, 5, h, 0.5)
    mgr.on_route(0, 6, h, 0.5)

    assert mgr.stats.hit_count == 3
    assert mgr.stats.miss_count == 2
    assert mgr.stats.miss_rate == pytest.approx(2 / 5)


def test_miss_rate_zero_when_no_routes():
    stats = ResidencyStats()
    assert stats.miss_rate == 0.0


# --------------------------------------------------------------------------------------
# factory + miss_mode property
# --------------------------------------------------------------------------------------
def test_make_miss_handler_from_enum_and_string():
    h1 = make_miss_handler(MissMode.FETCH_LOSSLESS)
    assert isinstance(h1, FetchLossless)
    h2 = make_miss_handler("precision-cascade", low_bits=2)
    assert isinstance(h2, PrecisionCascade)
    assert h2.low_bits == 2
    h3 = make_miss_handler(MissMode.HARD_DROP, variant="reroute")
    assert isinstance(h3, HardDrop)
    assert h3.variant == "reroute"


def test_make_miss_handler_rejects_unknown():
    with pytest.raises(ValueError):
        make_miss_handler("does-not-exist")


def test_manager_miss_mode_property():
    store = make_store()
    mgr = ResidencyManager(store=store, miss_handler=make_miss_handler(MissMode.HARD_DROP))
    assert mgr.miss_mode is MissMode.HARD_DROP


# --------------------------------------------------------------------------------------
# FusedSliceStore: stessa contabilità, annota la maschera sul bundle
# --------------------------------------------------------------------------------------
def test_fused_slice_store_accounting_and_mask():
    class Bundle:
        pass

    bundles = {0: Bundle()}
    store = FusedSliceStore(
        n_layers=1, n_experts=8, bytes_per_expert_fp16=BPE, fused_tensors=bundles
    )
    store.set_resident(0, {2, 5})
    assert store.resident_ids(0) == {2, 5}
    assert store.resident_vram_bytes() == 2 * BPE
    assert bundles[0].resident_mask == [2, 5]   # maschera annotata dal gpu_seam


# --------------------------------------------------------------------------------------
# torch-only (skippato senza torch)
# --------------------------------------------------------------------------------------
def test_module_list_materialize_with_real_tensors():
    torch = pytest.importorskip("torch")

    class TinyExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

    experts = [TinyExpert() for _ in range(4)]
    store = ModuleListStore(
        n_layers=1,
        n_experts=4,
        bytes_per_expert_fp16=BPE,
        experts_by_layer={0: experts},
        device="cpu",  # CPU per il test (no CUDA garantita in CI)
    )
    store.set_resident(0, {0, 2})
    # Materializzazione no-crash: i moduli restano utilizzabili.
    x = torch.zeros(1, 4)
    assert experts[0](x).shape == (1, 4)
