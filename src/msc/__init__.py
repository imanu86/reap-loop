"""msc — MoE Session Commit.

Pacchetto di ricerca per la policy "commit aggressivo per-sessione".

Moduli (vedi docs/00_architecture.md per il disegno):
    instrument/  — hook sul router + schema della traccia di attivazione
    workingset/  — stima del working set + metriche di concentrazione empirica
    policies/    — FULL · COARSE · AGGRESSIVE-COMMIT
    residency/   — gestione residenza expert in VRAM + miss handling (3 miss_mode)
    validator/   — segnale binario deterministico a contesto crescente
    experiment/  — loop della griglia 4D + metriche
    report/      — famiglia di curve + grafico riassuntivo

Stato: SCAFFOLD. Le funzioni sollevano NotImplementedError finché non implementate.
"""

__version__ = "0.0.1"
