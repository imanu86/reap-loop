"""workingset/ — stima del working set + metriche di concentrazione empirica.

È qui che vive la *correzione concettuale* (docs/00_architecture.md §3, Asse B): la sparsità
strutturale è un proxy; il driver reale del guadagno è la CONCENTRAZIONE empirica dell'uso.
"""

from msc.workingset.estimator import (  # noqa: F401
    CoverageCurve,
    ConcentrationStats,
    WorkingSetEstimate,
    estimate_working_set,
    concentration_stats,
    convergence_curve,
)
