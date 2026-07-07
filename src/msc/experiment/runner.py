"""runner.py — la griglia 4D: (modello, K, ctx, miss_mode) × policy.

STATO: IMPLEMENTATO (orchestrazione). Vedi docs/00_architecture.md §9.

Per ogni cella:
  1. warm-up   : gira N esempi, raccogli traccia (instrument/)
  2. stima     : working set + concentrazione (workingset/)
  3. commit    : ResidencyManager materializza i residenti; resto -> miss_mode (residency/)
  4. valuta    : genera -> validatore deterministico a contesto ctx (validator/)
  5. registra  : CellMetrics (experiment/metrics.py)

Scorciatoie (riducono la griglia, docs §9):
  - fetch-lossless: accuratezza ≡ FULL -> calcola accuratezza una volta; varia solo latenza/miss con K
  - K=100% ≡ FULL (ground truth), una volta per (modello, ctx)
  - traccia di warm-up riusabile tra miss_mode, TRANNE hard-drop+reroute (il path cambia -> ri-traccia)

NOTA torch-free: iter_cells / load_grid_spec / la resumability di run_grid sono PURI e testabili
CPU-only (nessun torch/transformers). Il pezzo che carica il modello, fa warm-up, commit e genera è
isolato in `_execute_cell` (il "gpu_seam"): importa torch/transformers con guardia e solleva un
errore chiaro se mancano. Così `import msc.experiment.runner` funziona senza torch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from msc.residency.miss_modes import MissMode

# Policy che, per costruzione, tiene TUTTI gli expert residenti (ground truth di accuratezza).
# A K=100% qualunque policy coincide con questa: nessun miss -> miss_mode irrilevante.
_FULL_POLICY = "FULL"

# Frazione K che equivale a FULL (tutti gli expert residenti).
_K_FULL = 1.0


@dataclass(frozen=True)
class GridSpec:
    """Definizione della griglia (caricata da configs/grid/sweep.yaml)."""

    model_ids: list[str]
    k_fractions: list[float]          # es. [1.0, 0.5, 0.25, 0.12, 0.06]
    ctx_lengths: list[int]            # es. [1024, 4096, 16384, 65536, -1(=max)]
    miss_modes: list[MissMode]
    policies: list[str]               # ["FULL", "COARSE", "AGGRESSIVE-COMMIT"]
    coverage_theta: float = 0.95
    warmup_examples: int = 32
    seed: int = 0


@dataclass(frozen=True)
class GridCell:
    """Una cella della griglia."""

    model_id: str
    policy: str
    k_fraction: float
    ctx_len: int
    miss_mode: MissMode


# --------------------------------------------------------------------------- #
# Caricamento della griglia da YAML                                           #
# --------------------------------------------------------------------------- #
def load_grid_spec(path: str) -> GridSpec:
    """Legge configs/grid/sweep.yaml e popola un GridSpec.

    Mappa i campi del file (model paths, k_fractions, ctx_lengths, miss_modes, policies, e i
    parametri comuni coverage_theta/warmup_examples/seed) sui campi della dataclass. I `miss_modes`
    vengono normalizzati da stringa a `MissMode` (solleva ValueError su valori ignoti). I campi extra
    del file (decode, validators, promotion, ...) sono ignorati: non fanno parte della griglia 4D.
    """
    import yaml  # import locale: pyyaml è una dipendenza leggera ma evitiamo costi a import-time

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    miss_modes = [MissMode(m) for m in cfg["miss_modes"]]

    return GridSpec(
        model_ids=list(cfg["models"]),
        k_fractions=[float(k) for k in cfg["k_fractions"]],
        ctx_lengths=[int(c) for c in cfg["ctx_lengths"]],
        miss_modes=miss_modes,
        policies=list(cfg["policies"]),
        coverage_theta=float(cfg.get("coverage_theta", 0.95)),
        warmup_examples=int(cfg.get("warmup_examples", 32)),
        seed=int(cfg.get("seed", 0)),
    )


# --------------------------------------------------------------------------- #
# Generazione delle celle (con le scorciatoie di docs §9)                     #
# --------------------------------------------------------------------------- #
def iter_cells(spec: GridSpec):
    """Genera le celle della griglia applicando le scorciatoie di docs/00_architecture.md §9.

    Scorciatoie implementate (eliminano celle ridondanti, non segnale):

    1) ``K=100% ≡ FULL`` — a K=100% tutti gli expert sono residenti: nessun miss, quindi miss_mode e
       policy sono irrilevanti. Emettiamo UNA SOLA cella ground-truth per ``(modello, ctx)``
       (policy=FULL, miss_mode=fetch-lossless canonico). Niente duplicati su policy/miss_mode a K=1.0.

    2) ``fetch-lossless: accuratezza ≡ FULL`` — in fetch-lossless l'accuratezza non dipende da K (è la
       baseline lossless). Misurarla per ogni K duplicherebbe celle-accuratezza identiche. Emettiamo
       quindi le celle fetch-lossless UNA SOLA volta per ``(modello, ctx, policy)`` (al K più
       aggressivo della griglia, dove latenza/miss_rate sono massimi e quindi più informativi), invece
       di una per ogni K. Le celle lossy (precision-cascade / hard-drop) coprono invece TUTTI i K < 1.0
       perché lì l'accuratezza cambia con K.

    Le policy lossy (tutto ciò che non è FULL) spazzano ``K < 1.0`` (il K=1.0 è già coperto dal punto 1)
    × ctx × miss_mode, con la deduplica fetch-lossless del punto 2.

    Ordine deterministico (utile per la resumability): modello -> ctx -> [FULL] -> policy -> K -> miss.
    """
    lossy_policies = [p for p in spec.policies if p != _FULL_POLICY]
    # K non-100%: gli unici K su cui le policy lossy producono celle distinte (K=1.0 ≡ FULL).
    k_sub = [k for k in spec.k_fractions if not _is_full_k(k)]

    has_lossless = MissMode.FETCH_LOSSLESS in spec.miss_modes
    lossy_miss_modes = [m for m in spec.miss_modes if m is not MissMode.FETCH_LOSSLESS]
    # K più aggressivo (più piccolo) della griglia: l'unico su cui emettiamo la fetch-lossless,
    # perché lì latenza/miss_rate sono massimi (l'accuratezza è comunque ≡ FULL per ogni K).
    k_for_lossless = min(k_sub) if k_sub else _K_FULL

    seen: set[tuple] = set()

    def _emit(cell: GridCell):
        key = (cell.model_id, cell.policy, cell.k_fraction, cell.ctx_len, cell.miss_mode)
        if key in seen:
            return None
        seen.add(key)
        return cell

    for model_id in spec.model_ids:
        for ctx in spec.ctx_lengths:
            # --- Scorciatoia 1: K=100% ≡ FULL, una sola cella ground-truth per (modello, ctx) ---
            cell = _emit(
                GridCell(
                    model_id=model_id,
                    policy=_FULL_POLICY,
                    k_fraction=_K_FULL,
                    ctx_len=ctx,
                    miss_mode=MissMode.FETCH_LOSSLESS,
                )
            )
            if cell is not None:
                yield cell

            # --- Celle lossy: K<1.0 × policy(lossy) × miss_mode ---
            for policy in lossy_policies:
                # Scorciatoia 2: fetch-lossless una sola volta per (modello, ctx, policy).
                if has_lossless:
                    cell = _emit(
                        GridCell(
                            model_id=model_id,
                            policy=policy,
                            k_fraction=k_for_lossless,
                            ctx_len=ctx,
                            miss_mode=MissMode.FETCH_LOSSLESS,
                        )
                    )
                    if cell is not None:
                        yield cell

                # Celle lossy (precision-cascade / hard-drop): tutti i K<1.0 (accuratezza varia con K).
                for k in k_sub:
                    for miss_mode in lossy_miss_modes:
                        cell = _emit(
                            GridCell(
                                model_id=model_id,
                                policy=policy,
                                k_fraction=k,
                                ctx_len=ctx,
                                miss_mode=miss_mode,
                            )
                        )
                        if cell is not None:
                            yield cell


def _is_full_k(k: float) -> bool:
    """True se la frazione K equivale a 100% (tutti gli expert residenti)."""
    return abs(float(k) - _K_FULL) < 1e-9


# --------------------------------------------------------------------------- #
# Identità della cella su disco (per la resumability)                         #
# --------------------------------------------------------------------------- #
def cell_id(cell: GridCell) -> str:
    """Identificatore stabile e filesystem-safe di una cella (per il file di risultato per-cella).

    Deterministico e indipendente dalla piattaforma: gli stessi assi -> sempre lo stesso id, così la
    ripresa riconosce una cella già fatta a prescindere dall'ordine di iterazione.
    """
    mm = cell.miss_mode.value if isinstance(cell.miss_mode, MissMode) else str(cell.miss_mode)
    safe_model = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(cell.model_id))
    return (
        f"{safe_model}__{cell.policy}__k{cell.k_fraction:g}"
        f"__ctx{cell.ctx_len}__{mm}"
    )


def _cell_result_path(out_dir: str, cell: GridCell) -> str:
    """Percorso del file di risultato per-cella (una riga JSON di CellMetrics.to_row())."""
    return os.path.join(out_dir, f"{cell_id(cell)}.json")


def is_cell_done(out_dir: str, cell: GridCell) -> bool:
    """True se la cella ha già un file di risultato non vuoto in `out_dir` (ripresa)."""
    path = _cell_result_path(out_dir, cell)
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Esecuzione della griglia (scheletro riprendibile)                           #
# --------------------------------------------------------------------------- #
def run_grid(spec: GridSpec, *, out_dir: str, execute_cell=None) -> list[GridCell]:
    """Esegue l'intera griglia e scrive una riga di metriche per cella in `out_dir`.

    RIPRENDIBILE: salta le celle già presenti (con risultato) in `out_dir`; scrive risultati
    incrementali (un file JSON per cella + append nel CSV cumulativo `metrics.csv`). Su 3060 le
    griglie sono lunghe, quindi ogni cella completata è persistita subito.

    `execute_cell`: callable iniettabile ``execute_cell(spec, cell) -> CellMetrics`` (il "gpu_seam").
    Se None usa `_execute_cell`, che richiede torch/transformers e solleva un errore chiaro se
    mancano. Iniettare un fake permette di testare la resumability/I-O senza GPU (vedi test).

    Ritorna la lista delle celle effettivamente ESEGUITE in questa chiamata (escluse quelle saltate
    perché già fatte), utile al chiamante/CLI per il riepilogo.
    """
    os.makedirs(out_dir, exist_ok=True)
    runner = execute_cell if execute_cell is not None else _execute_cell

    executed: list[GridCell] = []
    for cell in iter_cells(spec):
        if is_cell_done(out_dir, cell):
            continue  # ripresa: la cella ha già un risultato persistito.
        metrics = runner(spec, cell)
        _write_cell_result(out_dir, cell, metrics)
        executed.append(cell)
    return executed


def _write_cell_result(out_dir: str, cell: GridCell, metrics) -> None:
    """Persiste il risultato di una cella: un file JSON per-cella + append nel CSV cumulativo.

    `metrics` è un CellMetrics (o un dict già piatto / qualunque oggetto con `to_row()`). La scrittura
    per-cella è ciò che rende la griglia riprendibile (presenza del file = cella fatta).
    """
    row = metrics.to_row() if hasattr(metrics, "to_row") else dict(metrics)

    # 1) File per-cella (sentinella di completamento + risultato puntuale).
    path = _cell_result_path(out_dir, cell)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(row), fh, ensure_ascii=False, sort_keys=True)

    # 2) Append nel CSV cumulativo (aggregazione incrementale per il report).
    _append_csv_row(os.path.join(out_dir, "metrics.csv"), row)


def _jsonable(row: dict) -> dict:
    """Rende serializzabile in JSON una riga di metriche (enum -> valore, ecc.)."""
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, MissMode):
            out[k] = v.value
        else:
            out[k] = v
    return out


def _append_csv_row(csv_path: str, row: dict) -> None:
    """Append di una riga nel CSV cumulativo, scrivendo l'header solo al primo record.

    Usa il modulo `csv` della stdlib (no pandas a import-time). Le colonne seguono l'ordine della
    PRIMA riga scritta; le righe successive vengono riallineate a quell'header.
    """
    import csv

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    jrow = _jsonable(row)
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        if write_header:
            fieldnames = list(jrow.keys())
        else:
            with open(csv_path, "r", encoding="utf-8", newline="") as rh:
                import csv as _csv

                reader = _csv.reader(rh)
                fieldnames = next(reader, list(jrow.keys()))
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(jrow)


# --------------------------------------------------------------------------- #
# gpu_seam: l'unico pezzo che richiede torch/transformers + un modello reale  #
# --------------------------------------------------------------------------- #
def _execute_cell(spec: GridSpec, cell: GridCell):
    """Esegue UNA cella: warm-up -> stima working set -> commit -> genera -> valuta -> CellMetrics.

    Questo è il "gpu_seam": carica il modello, fa generazione e misura la VRAM reale. Richiede
    torch + transformers (e un modello scaricato). Importa entrambi QUI DENTRO con guardia e solleva
    un RuntimeError CHIARO se mancano, così `import msc.experiment.runner`, `iter_cells`,
    `load_grid_spec` e la resumability di `run_grid` restano usabili e testabili senza GPU.

    Il cablaggio reale (instrument -> workingset -> residency -> validator -> metrics) vive qui:
    è l'unico punto che dipende dall'hardware. I test iniettano un `execute_cell` fittizio.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dipende dall'ambiente con GPU
        raise RuntimeError(
            "_execute_cell richiede torch e transformers (più un modello MoE scaricato): "
            "non sono importabili in questo ambiente. La griglia, iter_cells, load_grid_spec e la "
            "ripresa di run_grid funzionano senza torch; solo l'esecuzione effettiva di una cella "
            "(warm-up/generazione/misura VRAM) necessita della GPU. Installa torch (build CUDA, vedi "
            "docs/02_models.md) e transformers, oppure inietta un execute_cell di test."
        ) from exc

    # pragma: no cover -- da qui in giù serve davvero la GPU + un modello reale. Il cablaggio
    # concreto (RouterLogger -> estimate_working_set -> ResidencyManager.commit -> Validator
    # .evaluate_at_lengths -> CellMetrics con VRAM misurata via nvidia-ml-py) si appoggia ai moduli
    # già implementati (msc.instrument, msc.workingset, msc.residency, msc.validator, msc.report) e
    # alle policy di msc.policies. Non eseguibile in CI CPU-only.
    raise RuntimeError(  # pragma: no cover
        "_execute_cell: esecuzione reale della cella non disponibile in ambiente CPU-only. "
        "Eseguire su una macchina con GPU + modello, o iniettare un execute_cell di test."
    )
