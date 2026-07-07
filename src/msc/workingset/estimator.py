"""estimator.py — working set per copertura + metriche di concentrazione.

Tutto per-layer (la concentrazione varia molto tra i layer di un MoE).

Definizioni (docs/00_architecture.md §7):
  - istogramma d'uso p_e degli expert in un layer (dalla traccia di warm-up)
  - curva di copertura C(k) = somma delle top-k frequenze ordinate desc
  - working set a soglia θ: W_θ = min{k : C(k) >= θ}
  - concentrazione: entropia normalizzata H, Gini, numero efficace N_eff = exp(H_nat)
  - convergenza: W_θ in funzione del numero di token di warm-up (deve stabilizzarsi)

Nota: nessun import di torch/transformers a livello di modulo. Lavoriamo su `trace_reader`
(`msc.instrument.trace.TraceReader`) o, equivalentemente, su un qualunque iterabile di record con
attributi `.layer` e `.topk_ids` — cosi i test possono passare un fake CPU-only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageCurve:
    """Curva di copertura cumulata per un layer: experts ordinati per frequenza desc."""

    layer: int
    expert_ids_sorted: tuple[int, ...]   # id ordinati per frequenza decrescente
    cumulative_coverage: tuple[float, ...]  # C(k), monotona crescente fino a 1.0

    def working_set_at(self, theta: float) -> tuple[int, ...]:
        """Gli expert necessari a coprire la frazione `theta` dell'uso.

        W_θ = il piu piccolo prefisso (in ordine di frequenza desc) tale che C(k) >= θ.
        Con θ <= 0 restituiamo il set vuoto; con θ >= 1 (o se nessun prefisso raggiunge θ
        per arrotondamenti) restituiamo tutti gli expert osservati.
        """
        if theta <= 0.0:
            return ()
        # primo indice k (0-based) in cui la copertura cumulata raggiunge θ.
        for k, cov in enumerate(self.cumulative_coverage):
            if cov >= theta:
                return self.expert_ids_sorted[: k + 1]
        # nessun prefisso raggiunge θ (es. θ=1.0 con float rounding) -> tutti.
        return self.expert_ids_sorted


@dataclass(frozen=True)
class ConcentrationStats:
    """Metriche che spiegano la curva accuratezza-vs-VRAM (oltre alla sparsità strutturale)."""

    layer: int
    n_total: int
    entropy_norm: float   # H / log(N): 1 = uniforme/diffuso, 0 = concentrato su 1 expert
    gini: float
    n_eff: float          # numero efficace di expert = exp(entropia naturale)


@dataclass(frozen=True)
class WorkingSetEstimate:
    """Risultato della stima per una sessione: per-layer + aggregati."""

    session_id: str
    model_id: str
    theta: float
    per_layer_working_set: dict  # layer -> tuple[expert_id, ...]
    per_layer_concentration: dict  # layer -> ConcentrationStats
    committed_fraction: float    # frazione media di expert residenti -> mappa su K


# --------------------------------------------------------------------------- #
# Utility interne                                                             #
# --------------------------------------------------------------------------- #
def _layer_histogram(records) -> dict[int, Counter]:
    """Costruisce, per ogni layer, l'istogramma di attivazione degli expert.

    Conta UNA attivazione per ciascun id presente in `topk_ids` di ciascun record (cioe ogni
    selezione top-k di ogni token contribuisce al conteggio). Accetta un qualunque iterabile di
    record con attributi `.layer` e `.topk_ids`.
    """
    hist: dict[int, Counter] = {}
    for rec in records:
        layer = rec.layer
        counter = hist.get(layer)
        if counter is None:
            counter = Counter()
            hist[layer] = counter
        for eid in rec.topk_ids:
            counter[int(eid)] += 1
    return hist


def _iter_records(trace_reader, layer: int | None = None):
    """Itera i record della traccia in modo uniforme.

    Supporta:
      - un TraceReader REALE (ha il metodo `.records(layer=...)`);
      - un iterabile semplice di record (lista/generatore di fake nei test).
    Se passiamo un iterabile semplice e `layer` e specificato, filtriamo qui.
    """
    records_method = getattr(trace_reader, "records", None)
    if callable(records_method):
        return records_method(layer)
    # fallback: iterabile diretto di record.
    if layer is None:
        return iter(trace_reader)
    return (rec for rec in trace_reader if rec.layer == layer)


def _coverage_from_counter(layer: int, counter: Counter) -> CoverageCurve:
    """Curva di copertura cumulata da un istogramma d'uso (frequenze ordinate desc)."""
    if not counter:
        return CoverageCurve(layer=layer, expert_ids_sorted=(), cumulative_coverage=())
    # ordina per frequenza desc; a parita di frequenza, per id crescente (deterministico).
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ids = np.array([eid for eid, _ in items], dtype=np.int64)
    counts = np.array([c for _, c in items], dtype=np.float64)
    total = counts.sum()
    cumulative = np.cumsum(counts) / total
    return CoverageCurve(
        layer=layer,
        expert_ids_sorted=tuple(int(x) for x in ids),
        cumulative_coverage=tuple(float(x) for x in cumulative),
    )


def _concentration_from_counter(layer: int, counter: Counter) -> ConcentrationStats:
    """Entropia normalizzata / Gini / N_eff da un istogramma d'uso degli expert."""
    n_total = len(counter)
    if n_total == 0:
        return ConcentrationStats(layer=layer, n_total=0, entropy_norm=0.0, gini=0.0, n_eff=0.0)
    counts = np.array(list(counter.values()), dtype=np.float64)
    p = counts / counts.sum()

    # entropia naturale H_nat = -Σ p ln p ; N_eff = exp(H_nat) (perplexity dell'uso).
    # con p>0 (i counter non hanno chiavi a zero) il log e ben definito.
    h_nat = float(-np.sum(p * np.log(p)))
    n_eff = float(np.exp(h_nat))

    # entropia normalizzata H/ln(N): 1 = uniforme, 0 = concentrato su 1 expert.
    # con un solo expert osservato (N=1) ln(N)=0 -> per convenzione concentrazione massima -> 0.
    if n_total == 1:
        entropy_norm = 0.0
    else:
        entropy_norm = h_nat / float(np.log(n_total))

    gini = _gini(counts)
    return ConcentrationStats(
        layer=layer,
        n_total=n_total,
        entropy_norm=entropy_norm,
        gini=gini,
        n_eff=n_eff,
    )


def _gini(counts: np.ndarray) -> float:
    """Coefficiente di Gini della distribuzione d'uso (0 = perfettamente uniforme).

    Formula standard sui valori ordinati:
        G = (Σ_i (2i - n - 1) x_i) / (n Σ_i x_i),  i = 1..n (x ordinati crescente).
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    if n == 1:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    g = float(np.sum((2.0 * index - n - 1.0) * x) / (n * total))
    # clamp difensivo contro micro-negativi da arrotondamento.
    return max(0.0, g)


# --------------------------------------------------------------------------- #
# API pubblica                                                                #
# --------------------------------------------------------------------------- #
def estimate_working_set(
    trace_reader, theta: float, k_budget: float | None = None
) -> WorkingSetEstimate:
    """Dalla traccia di warm-up: working set per copertura θ, troncato al budget K se più stringente.

    Args:
        trace_reader: msc.instrument.trace.TraceReader sulla traccia di warm-up (o iterabile di
            record con attributi `.layer`/`.topk_ids`).
        theta: soglia di copertura cumulata (es. 0.95).
        k_budget: frazione massima [0,1] di expert residenti per layer (asse K); se None, solo θ
            comanda. Il commit tiene il MIN tra (working set per θ) e (k_budget·n_total), arrotondato
            verso l'alto a >=1 quando il layer ha almeno un expert.

    Returns:
        WorkingSetEstimate con, per ogni layer: il set committato e le metriche di concentrazione;
        `committed_fraction` = media sui layer di |committed_layer| / n_total_layer.
    """
    hist = _layer_histogram(_iter_records(trace_reader))

    session_id = getattr(trace_reader, "session_id", "") or ""
    model_id = getattr(trace_reader, "model_id", "") or ""

    per_layer_ws: dict[int, tuple[int, ...]] = {}
    per_layer_conc: dict[int, ConcentrationStats] = {}
    fractions: list[float] = []

    for layer in sorted(hist):
        counter = hist[layer]
        n_total = len(counter)

        curve = _coverage_from_counter(layer, counter)
        committed = list(curve.working_set_at(theta))

        # troncamento al budget K (se piu stringente della copertura θ).
        if k_budget is not None and n_total > 0:
            # almeno 1 expert residente quando il layer e usato (k_budget>0).
            k_max = max(1, int(np.floor(k_budget * n_total))) if k_budget > 0 else 0
            if len(committed) > k_max:
                committed = committed[:k_max]

        per_layer_ws[layer] = tuple(committed)
        per_layer_conc[layer] = _concentration_from_counter(layer, counter)

        if n_total > 0:
            fractions.append(len(committed) / n_total)

    committed_fraction = float(np.mean(fractions)) if fractions else 0.0

    return WorkingSetEstimate(
        session_id=session_id,
        model_id=model_id,
        theta=theta,
        per_layer_working_set=per_layer_ws,
        per_layer_concentration=per_layer_conc,
        committed_fraction=committed_fraction,
    )


def concentration_stats(trace_reader, layer: int) -> ConcentrationStats:
    """Entropia normalizzata / Gini / N_eff dell'uso degli expert in un layer."""
    hist = _layer_histogram(_iter_records(trace_reader, layer=layer))
    counter = hist.get(layer, Counter())
    return _concentration_from_counter(layer, counter)


def convergence_curve(trace_reader, theta: float, layer: int) -> list[tuple[int, int]]:
    """W_θ in funzione del numero di token di warm-up: lista (n_token, W_θ).

    Scorriamo i record del layer in ordine, accumulando l'istogramma man mano, e a ogni passo
    registriamo (numero di token visti finora, dimensione del working set a copertura θ). Serve a
    verificare il rischio R4: se W_θ non si stabilizza, la stima del commit è inaffidabile.

    `n_token` conta i record (= attivazioni top-k di un layer per un token) consumati finora, cioe
    il numero di token di warm-up processati per quel layer.
    """
    counter: Counter = Counter()
    out: list[tuple[int, int]] = []
    n_token = 0
    for rec in _iter_records(trace_reader, layer=layer):
        n_token += 1
        for eid in rec.topk_ids:
            counter[int(eid)] += 1
        curve = _coverage_from_counter(layer, counter)
        w_theta = len(curve.working_set_at(theta))
        out.append((n_token, w_theta))
    return out
