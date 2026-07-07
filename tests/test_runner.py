"""test_runner.py — orchestrazione della griglia 4D, CPU-only e SENZA torch.

Copre i tre pezzi torch-free del runner (docs/00_architecture.md §9):
  1. load_grid_spec legge DAVVERO configs/grid/sweep.yaml e popola un GridSpec.
  2. iter_cells applica le scorciatoie §9 (K=100% ≡ FULL una volta per (modello,ctx);
     fetch-lossless: accuratezza ≡ FULL -> niente celle-accuratezza duplicate su K) e NON emette
     celle ridondanti.
  3. run_grid è RIPRENDIBILE: salta una cella già presente in out_dir e scrive risultati incrementali.

Isolato: usa solo msc.experiment.runner (+ msc.residency.miss_modes per l'enum, che è torch-free) e
un `execute_cell` FITTIZIO iniettato -> nessun torch/transformers, nessun modello, nessun modulo di
altri agent. Deterministico (nessun rng, nessun tempo, nessuna rete).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from msc.experiment.runner import (
    GridCell,
    GridSpec,
    cell_id,
    is_cell_done,
    iter_cells,
    load_grid_spec,
    run_grid,
)
from msc.residency.miss_modes import MissMode

# Percorso del file di griglia REALE versionato nel repo (root = parent di tests/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SWEEP_YAML = _REPO_ROOT / "configs" / "grid" / "sweep.yaml"


# --------------------------------------------------------------------------- #
# Helper: uno spec piccolo e controllabile per testare la forma delle celle   #
# --------------------------------------------------------------------------- #
def _small_spec() -> GridSpec:
    """Griglia minima ma rappresentativa: 1 modello, K con 1.0 + due K lossy, 2 ctx, 3 miss_mode."""
    return GridSpec(
        model_ids=["m0"],
        k_fractions=[1.0, 0.5, 0.25],
        ctx_lengths=[1024, 4096],
        miss_modes=[MissMode.FETCH_LOSSLESS, MissMode.PRECISION_CASCADE, MissMode.HARD_DROP],
        policies=["FULL", "COARSE", "AGGRESSIVE-COMMIT"],
    )


# --------------------------------------------------------------------------- #
# 1. load_grid_spec legge DAVVERO il file YAML versionato                     #
# --------------------------------------------------------------------------- #
def test_load_grid_spec_reads_real_sweep_yaml():
    assert _SWEEP_YAML.exists(), f"manca il file di griglia atteso: {_SWEEP_YAML}"
    spec = load_grid_spec(str(_SWEEP_YAML))

    assert isinstance(spec, GridSpec)
    # I valori provengono DAVVERO dal file (non da default): verifichiamo i campi chiave.
    assert spec.k_fractions == [1.0, 0.5, 0.25, 0.12, 0.06]
    assert spec.ctx_lengths == [1024, 4096, 16384, 65536, -1]
    assert spec.policies == ["FULL", "COARSE", "AGGRESSIVE-COMMIT"]
    assert len(spec.model_ids) == 3  # le 3 voci `models:` del file
    # miss_modes normalizzati da stringa a enum MissMode.
    assert spec.miss_modes == [
        MissMode.FETCH_LOSSLESS,
        MissMode.PRECISION_CASCADE,
        MissMode.HARD_DROP,
    ]
    # parametri comuni letti dal file (non dai default della dataclass).
    assert spec.coverage_theta == 0.95
    assert spec.warmup_examples == 32
    assert spec.seed == 0


def test_load_grid_spec_rejects_unknown_miss_mode(tmp_path: Path):
    """Un miss_mode ignoto nel file deve sollevare ValueError (normalizzazione enum)."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "models: [m0]\n"
        "k_fractions: [1.0, 0.5]\n"
        "ctx_lengths: [1024]\n"
        "miss_modes: [fetch-lossless, teleport]\n"  # 'teleport' non esiste
        "policies: [FULL, COARSE]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_grid_spec(str(bad))


# --------------------------------------------------------------------------- #
# 2. iter_cells applica le scorciatoie §9 e non duplica celle                 #
# --------------------------------------------------------------------------- #
def _keys(cells):
    return [(c.model_id, c.policy, c.k_fraction, c.ctx_len, c.miss_mode) for c in cells]


def test_iter_cells_no_duplicate_cells():
    """Nessuna cella ridondante: tutte le chiave (modello,policy,K,ctx,miss) sono uniche."""
    spec = _small_spec()
    cells = list(iter_cells(spec))
    keys = _keys(cells)
    assert len(keys) == len(set(keys)), "iter_cells ha emesso celle duplicate"


def test_iter_cells_k100_collapses_to_single_full_per_model_ctx():
    """Scorciatoia: K=100% ≡ FULL una sola cella per (modello, ctx); niente duplicati su policy/miss."""
    spec = _small_spec()
    cells = list(iter_cells(spec))

    k1 = [c for c in cells if c.k_fraction == 1.0]
    # A K=1.0: sempre e solo policy FULL (niente COARSE/AGGRESSIVE ridondanti).
    assert all(c.policy == "FULL" for c in k1)
    # Esattamente una cella K=1.0 per (modello, ctx).
    per_model_ctx = Counter((c.model_id, c.ctx_len) for c in k1)
    assert set(per_model_ctx.values()) == {1}
    assert len(k1) == len(spec.model_ids) * len(spec.ctx_lengths)
    # La policy FULL non compare MAI a K<1.0 (sarebbe una duplicazione del ground truth).
    assert all(c.k_fraction == 1.0 for c in cells if c.policy == "FULL")


def test_iter_cells_fetch_lossless_accuracy_not_duplicated_over_k():
    """Scorciatoia: fetch-lossless (acc ≡ FULL) non duplica celle-accuratezza su K per policy lossy.

    Per ciascun (modello, ctx, policy-lossy) ci deve essere AL PIÙ una cella fetch-lossless, non una
    per ogni K (l'accuratezza è identica per ogni K -> sarebbe ridondante).
    """
    spec = _small_spec()
    cells = list(iter_cells(spec))

    fl = [c for c in cells if c.miss_mode is MissMode.FETCH_LOSSLESS]
    # FULL ground-truth (K=1.0) + una fetch-lossless per (modello,ctx,policy-lossy).
    lossy_policies = [p for p in spec.policies if p != "FULL"]
    per_combo = Counter((c.model_id, c.ctx_len, c.policy) for c in fl)
    assert max(per_combo.values()) == 1, "fetch-lossless duplicata su K per qualche (modello,ctx,policy)"

    # Le celle fetch-lossless lossy stanno a un K<1.0 (il K=1.0 è il ground-truth FULL).
    fl_lossy = [c for c in fl if c.policy in lossy_policies]
    assert fl_lossy, "attese celle fetch-lossless per le policy lossy"
    assert all(c.k_fraction != 1.0 for c in fl_lossy)


def test_iter_cells_lossy_miss_modes_span_all_sub_k():
    """Le celle lossy (cascade/drop) coprono TUTTI i K<1.0 (lì l'accuratezza varia con K)."""
    spec = _small_spec()
    cells = list(iter_cells(spec))

    k_sub = [k for k in spec.k_fractions if k != 1.0]
    lossy_policies = [p for p in spec.policies if p != "FULL"]
    for policy in lossy_policies:
        for mm in (MissMode.PRECISION_CASCADE, MissMode.HARD_DROP):
            for ctx in spec.ctx_lengths:
                ks = sorted(
                    c.k_fraction
                    for c in cells
                    if c.policy == policy and c.miss_mode is mm and c.ctx_len == ctx
                )
                assert ks == sorted(k_sub), (
                    f"{policy}/{mm.value}/ctx={ctx}: K coperti {ks} != attesi {sorted(k_sub)}"
                )
    # Nessuna cella lossy a K=1.0 (sarebbe coperta dal ground-truth FULL).
    assert all(
        c.k_fraction != 1.0 for c in cells if c.miss_mode is not MissMode.FETCH_LOSSLESS
    )


def test_iter_cells_exact_cardinality_small_spec():
    """Conteggio esatto delle celle dopo le scorciatoie, calcolato a mano sullo spec piccolo.

    Per (modello, ctx):
      - 1 cella FULL (K=1.0, fetch-lossless)
      - per ciascuna delle 2 policy lossy:
          * 1 cella fetch-lossless (al K più aggressivo)
          * |K<1.0| × |miss lossy| = 2 × 2 = 4 celle lossy
        -> 5 celle per policy lossy -> 10
      => 11 celle per (modello, ctx)
    Con 1 modello × 2 ctx -> 22.
    """
    spec = _small_spec()
    cells = list(iter_cells(spec))
    assert len(cells) == 22


def test_iter_cells_cardinality_real_sweep():
    """Cardinalità della griglia REALE dopo le scorciatoie (controllo di non-regressione).

    3 modelli × 5 ctx × [1 FULL + 2 policy-lossy × (1 lossless + 4 K × 2 miss-lossy)]
    = 3 × 5 × (1 + 2 × 9) = 3 × 5 × 19 = 285.
    """
    spec = load_grid_spec(str(_SWEEP_YAML))
    cells = list(iter_cells(spec))
    assert len(cells) == 285
    keys = _keys(cells)
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# 3. run_grid è riprendibile e scrive risultati incrementali                  #
# --------------------------------------------------------------------------- #
class _FakeMetrics:
    """Sostituto torch-free di CellMetrics: espone solo `to_row()` (ciò che usa run_grid)."""

    def __init__(self, cell: GridCell) -> None:
        self._cell = cell

    def to_row(self) -> dict:
        return {
            "model_id": self._cell.model_id,
            "policy": self._cell.policy,
            "k_fraction": self._cell.k_fraction,
            "ctx_len": self._cell.ctx_len,
            "miss_mode": self._cell.miss_mode,  # enum: run_grid lo serializza a stringa
            "accuracy": 0.9,
        }


def test_run_grid_executes_all_cells_and_writes_outputs(tmp_path: Path):
    """Senza nulla di pre-esistente, run_grid esegue ogni cella e scrive un file per cella + CSV."""
    spec = _small_spec()
    out_dir = tmp_path / "grid"

    executed_cells: list[GridCell] = []

    def fake_execute(_spec, cell):
        executed_cells.append(cell)
        return _FakeMetrics(cell)

    executed = run_grid(spec, out_dir=str(out_dir), execute_cell=fake_execute)

    all_cells = list(iter_cells(spec))
    assert len(executed) == len(all_cells)
    assert len(executed_cells) == len(all_cells)
    # Un file di risultato per cella.
    for cell in all_cells:
        path = out_dir / f"{cell_id(cell)}.json"
        assert path.exists() and path.stat().st_size > 0
    # CSV cumulativo presente con una riga di dati per cella (+ header).
    csv_path = out_dir / "metrics.csv"
    assert csv_path.exists()
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(all_cells) + 1  # header + una riga per cella
    # L'enum miss_mode è stato serializzato a stringa nel JSON per-cella.
    sample = json.loads((out_dir / f"{cell_id(all_cells[0])}.json").read_text(encoding="utf-8"))
    assert isinstance(sample["miss_mode"], str)


def test_run_grid_resumes_skipping_done_cells(tmp_path: Path):
    """RIPRENDIBILE: una cella con risultato già presente in out_dir NON viene rieseguita."""
    spec = _small_spec()
    out_dir = tmp_path / "grid"
    out_dir.mkdir(parents=True)

    all_cells = list(iter_cells(spec))
    done_cell = all_cells[0]

    # Simuliamo una cella "già fatta": scriviamo a mano il suo file di risultato non vuoto.
    done_path = out_dir / f"{cell_id(done_cell)}.json"
    done_path.write_text(json.dumps({"already": "done"}), encoding="utf-8")
    assert is_cell_done(str(out_dir), done_cell)

    executed_cells: list[GridCell] = []

    def fake_execute(_spec, cell):
        executed_cells.append(cell)
        return _FakeMetrics(cell)

    executed = run_grid(spec, out_dir=str(out_dir), execute_cell=fake_execute)

    # La cella già fatta è stata SALTATA: non rieseguita, non sovrascritta.
    assert done_cell not in executed_cells
    assert done_cell not in executed
    assert len(executed) == len(all_cells) - 1
    # Il file pre-esistente è rimasto intatto (non riscritto da run_grid).
    assert json.loads(done_path.read_text(encoding="utf-8")) == {"already": "done"}


def test_run_grid_second_run_is_noop(tmp_path: Path):
    """Idempotenza: una seconda run completa non esegue nulla (tutte le celle già fatte)."""
    spec = _small_spec()
    out_dir = tmp_path / "grid"

    def fake_execute(_spec, cell):
        return _FakeMetrics(cell)

    first = run_grid(spec, out_dir=str(out_dir), execute_cell=fake_execute)
    assert len(first) > 0
    second = run_grid(spec, out_dir=str(out_dir), execute_cell=fake_execute)
    assert second == []


def test_cell_id_is_stable_and_filesystem_safe():
    """L'id di cella è deterministico e non contiene separatori di path."""
    cell = GridCell(
        model_id="configs/models/olmoe_1b_7b.yaml",
        policy="AGGRESSIVE-COMMIT",
        k_fraction=0.06,
        ctx_len=-1,
        miss_mode=MissMode.PRECISION_CASCADE,
    )
    cid = cell_id(cell)
    assert cid == cell_id(cell)  # stabile
    assert "/" not in cid and "\\" not in cid  # filesystem-safe (no path separators)
    # assi distintivi presenti nell'id.
    assert "AGGRESSIVE-COMMIT" in cid
    assert "precision-cascade" in cid


# --------------------------------------------------------------------------- #
# 4. il gpu_seam (_execute_cell) richiede torch: deve fallire in modo CHIARO  #
# --------------------------------------------------------------------------- #
def test_execute_cell_default_requires_torch_and_errors_clearly():
    """Senza torch nell'ambiente, run_grid con il gpu_seam di default solleva un RuntimeError chiaro.

    Saltato (non fallito) se torch è installato: lì il seam tenta l'esecuzione reale, fuori scope CPU.
    """
    try:
        import torch  # noqa: F401

        pytest.skip("torch installato: il gpu_seam tenta l'esecuzione reale (fuori scope CPU-only)")
    except ImportError:
        pass

    spec = _small_spec()
    one_cell = next(iter_cells(spec))
    from msc.experiment.runner import _execute_cell

    with pytest.raises(RuntimeError) as ei:
        _execute_cell(spec, one_cell)
    msg = str(ei.value).lower()
    assert "torch" in msg and "transformers" in msg
