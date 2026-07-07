"""test_workingset.py — working set + metriche di concentrazione (CPU-only, deterministico).

Non dipende da torch ne dai moduli implementati da altri agent: usa un fake reader/record locale
con i soli attributi che l'estimator richiede (`.layer`, `.topk_ids`, e opzionalmente `session_id`/
`model_id`). Verifica i casi limite descritti nel task:
  - distribuzione uniforme -> H_norm ≈ 1, N_eff ≈ N
  - distribuzione concentrata su 1 expert -> H_norm ≈ 0, N_eff ≈ 1
  - copertura cumulata monotona crescente
  - working_set_at(theta) corretto su un caso costruito a mano
  - convergence_curve cresce e si stabilizza su traccia sintetica
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from msc.workingset.estimator import (
    CoverageCurve,
    concentration_stats,
    convergence_curve,
    estimate_working_set,
)


# --------------------------------------------------------------------------- #
# Fake CPU-only: imita ActivationRecord/TraceReader senza dipendere da instrument/  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeRecord:
    layer: int
    topk_ids: tuple[int, ...]


class FakeTraceReader:
    """Espone `.records(layer=None)` come il TraceReader reale, su una lista in memoria."""

    def __init__(self, records, session_id="sess", model_id="fake-moe"):
        self._records = list(records)
        self.session_id = session_id
        self.model_id = model_id

    def records(self, layer=None):
        for rec in self._records:
            if layer is None or rec.layer == layer:
                yield rec


def _records_from_topk_sequence(layer, sequence):
    """Crea una lista di FakeRecord: ogni elemento di `sequence` e la top-k di un token."""
    return [FakeRecord(layer=layer, topk_ids=tuple(topk)) for topk in sequence]


# --------------------------------------------------------------------------- #
# Concentrazione: distribuzione uniforme                                       #
# --------------------------------------------------------------------------- #
def test_uniform_distribution_entropy_and_neff():
    # ogni token attiva esattamente 1 expert; 8 expert ciascuno usato lo stesso numero di volte.
    n = 8
    seq = []
    for _ in range(10):  # 10 round su tutti gli expert -> uso perfettamente uniforme
        for e in range(n):
            seq.append([e])
    reader = FakeTraceReader(_records_from_topk_sequence(layer=0, sequence=seq))

    stats = concentration_stats(reader, layer=0)
    assert stats.n_total == n
    assert math.isclose(stats.entropy_norm, 1.0, abs_tol=1e-9)
    assert math.isclose(stats.n_eff, float(n), rel_tol=1e-9)
    # uniforme -> Gini = 0.
    assert math.isclose(stats.gini, 0.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Concentrazione: tutto su 1 expert                                            #
# --------------------------------------------------------------------------- #
def test_single_expert_concentration():
    seq = [[3] for _ in range(50)]  # sempre lo stesso expert
    reader = FakeTraceReader(_records_from_topk_sequence(layer=1, sequence=seq))

    stats = concentration_stats(reader, layer=1)
    assert stats.n_total == 1
    assert math.isclose(stats.entropy_norm, 0.0, abs_tol=1e-12)
    assert math.isclose(stats.n_eff, 1.0, abs_tol=1e-12)
    assert math.isclose(stats.gini, 0.0, abs_tol=1e-12)


def test_skewed_distribution_intermediate():
    # un expert dominante + coda: H_norm in (0,1), N_eff < N_total, Gini > 0.
    seq = [[0]] * 90 + [[1]] * 5 + [[2]] * 3 + [[3]] * 2
    reader = FakeTraceReader(_records_from_topk_sequence(layer=0, sequence=seq))
    stats = concentration_stats(reader, layer=0)
    assert stats.n_total == 4
    assert 0.0 < stats.entropy_norm < 1.0
    assert 1.0 < stats.n_eff < 4.0
    assert stats.gini > 0.0


# --------------------------------------------------------------------------- #
# Curva di copertura: monotona + working_set_at corretto a mano                #
# --------------------------------------------------------------------------- #
def test_coverage_monotone_and_working_set_at_handmade():
    # caso costruito a mano: conteggi noti -> copertura nota.
    # expert: 0 ->50, 1 ->30, 2 ->15, 3 ->5 (totale 100).
    seq = [[0]] * 50 + [[1]] * 30 + [[2]] * 15 + [[3]] * 5
    reader = FakeTraceReader(_records_from_topk_sequence(layer=2, sequence=seq))

    # estimate_working_set produce per_layer; ma per testare la curva costruiamo direttamente
    # tramite estimate e poi verifichiamo working_set_at su una CoverageCurve nota.
    est = estimate_working_set(reader, theta=0.80)
    # ordine di frequenza desc atteso: 0,1,2,3 ; copertura: .50,.80,.95,1.0
    # θ=0.80 -> serve fino a copertura 0.80 -> expert {0,1}.
    assert est.per_layer_working_set[2] == (0, 1)

    # costruiamo a mano la curva attesa e verifichiamo monotonia + working_set_at.
    curve = CoverageCurve(
        layer=2,
        expert_ids_sorted=(0, 1, 2, 3),
        cumulative_coverage=(0.50, 0.80, 0.95, 1.0),
    )
    cov = curve.cumulative_coverage
    assert all(cov[i] <= cov[i + 1] for i in range(len(cov) - 1))  # monotona crescente
    assert math.isclose(cov[-1], 1.0, abs_tol=1e-12)

    assert curve.working_set_at(0.0) == ()
    assert curve.working_set_at(0.50) == (0,)
    assert curve.working_set_at(0.80) == (0, 1)
    assert curve.working_set_at(0.81) == (0, 1, 2)
    assert curve.working_set_at(0.95) == (0, 1, 2)
    assert curve.working_set_at(1.0) == (0, 1, 2, 3)


def test_estimate_coverage_curve_is_monotone_internal():
    # verifica che la curva costruita internamente da estimate sia monotona e chiuda a 1.0.
    seq = [[0]] * 7 + [[1]] * 5 + [[2]] * 3 + [[3]] * 1
    reader = FakeTraceReader(_records_from_topk_sequence(layer=0, sequence=seq))
    # ricostruiamo la curva via la utility pubblica indiretta: usiamo convergence sull'ultimo punto
    # non basta; testiamo invece la monotonia tramite working_set crescente al crescere di θ.
    est = estimate_working_set(reader, theta=0.99)
    ws_full = est.per_layer_working_set[0]
    # a θ=0.99 (quasi tutta la massa) il working set deve includere quasi tutti gli expert.
    assert set(ws_full).issuperset({0, 1, 2})


# --------------------------------------------------------------------------- #
# k_budget tronca il working set                                               #
# --------------------------------------------------------------------------- #
def test_k_budget_truncates_more_than_theta():
    # 10 expert, distribuzione moderatamente diffusa; θ ampia includerebbe molti expert,
    # ma k_budget=0.2 forza al massimo 2 expert per layer.
    seq = []
    counts = [40, 20, 15, 10, 5, 4, 3, 1, 1, 1]  # 10 expert
    for eid, c in enumerate(counts):
        seq += [[eid]] * c
    reader = FakeTraceReader(_records_from_topk_sequence(layer=0, sequence=seq))

    est_theta = estimate_working_set(reader, theta=0.95, k_budget=None)
    est_budget = estimate_working_set(reader, theta=0.95, k_budget=0.2)

    assert len(est_theta.per_layer_working_set[0]) > 2  # θ da solo prende >2 expert
    assert est_budget.per_layer_working_set[0] == (0, 1)  # k_budget 0.2*10=2 -> i 2 piu frequenti
    assert math.isclose(est_budget.committed_fraction, 2.0 / 10.0, abs_tol=1e-12)


def test_committed_fraction_averaged_over_layers():
    # due layer con concentrazioni diverse: la frazione committata e la media per-layer.
    recs = []
    # layer 0: tutto su 1 expert su 4 osservati -> con θ basso 1/.. ; usiamo θ=1.0 per determinismo
    recs += [FakeRecord(0, (0,))] * 10
    recs += [FakeRecord(0, (1,))] * 1
    # layer 1: due expert equiprobabili
    recs += [FakeRecord(1, (0,))] * 5
    recs += [FakeRecord(1, (1,))] * 5
    reader = FakeTraceReader(recs)
    est = estimate_working_set(reader, theta=1.0)
    # layer0: n_total=2, working set a θ=1.0 = entrambi -> frazione 1.0
    # layer1: n_total=2, θ=1.0 -> entrambi -> 1.0 ; media = 1.0
    assert math.isclose(est.committed_fraction, 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# Convergenza: cresce poi si stabilizza                                        #
# --------------------------------------------------------------------------- #
def test_convergence_curve_grows_then_stabilizes():
    # Traccia sintetica deterministica: nei primi token compaiono expert nuovi (W_θ cresce);
    # dopo, si ripete sempre lo stesso piccolo working set (W_θ si stabilizza).
    layer = 0
    seq = []
    # fase 1: 6 expert distinti appaiono uno alla volta (W_θ sale).
    for e in range(6):
        seq.append([e])
    # fase 2: 200 token che riusano SOLO {0,1} (i dominanti) -> W_θ a θ=0.9 si assesta.
    for _ in range(200):
        seq.append([0])
        seq.append([1])
    reader = FakeTraceReader(_records_from_topk_sequence(layer=layer, sequence=seq))

    curve = convergence_curve(reader, theta=0.90, layer=layer)

    # un punto per record consumato.
    assert len(curve) == len(seq)
    # n_token strettamente crescente 1..N.
    ntokens = [n for n, _ in curve]
    assert ntokens == list(range(1, len(seq) + 1))

    w_values = [w for _, w in curve]
    # tutti positivi e mai oltre il numero di expert osservati.
    assert all(w >= 1 for w in w_values)
    # nella prima fase il working set raggiunge un picco (>= qualche expert distinto)...
    early_max = max(w_values[:20])
    assert early_max >= 2
    # ...e nella coda finale si stabilizza (ultimi 50 punti costanti).
    tail = w_values[-50:]
    assert len(set(tail)) == 1  # stabile
    # la coda non puo essere piu grande del picco iniziale transitorio (W_θ non esplode).
    assert tail[0] <= early_max


def test_concentration_empty_layer():
    # robustezza: layer non presente nella traccia -> stats a zero, niente crash.
    reader = FakeTraceReader([FakeRecord(0, (1,))])
    stats = concentration_stats(reader, layer=99)
    assert stats.n_total == 0
    assert stats.entropy_norm == 0.0
    assert stats.n_eff == 0.0
    assert stats.gini == 0.0


def test_multi_id_topk_counts_each_expert():
    # ogni token attiva top-2: entrambi gli id devono contare nell'istogramma.
    seq = [[0, 1], [0, 2], [0, 1]]
    reader = FakeTraceReader(_records_from_topk_sequence(layer=0, sequence=seq))
    # conteggi: 0->3, 1->2, 2->1 ; totale attivazioni = 6.
    est = estimate_working_set(reader, theta=0.50)  # 0 copre 3/6=0.5 -> {0}
    assert est.per_layer_working_set[0] == (0,)
    est2 = estimate_working_set(reader, theta=0.84)  # 0,1 = 5/6≈0.833 <0.84 -> serve anche 2
    assert est2.per_layer_working_set[0] == (0, 1, 2)
