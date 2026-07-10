# POINTER — `runs/reap/multiseed_2026-07-07/` (dati NON in questo repo)

> Il paper canonico (`docs/paper/PAPER.md`) e i ledger citano `runs/reap/multiseed_2026-07-07/`.
> Quella directory NON vive in questo repo: vive nel repo **moe-aggressive-commit**, branch
> **`reap/k91-coding-vram`** (su origin `github.com/imanu86/moe-aggressive-commit`, tip `c8a7569`;
> la dir entra col commit `7140397`). Questo file è solo il puntatore.

## Cosa contiene (67 file)

Repliche **N=3 multi-seed** su 2 pod RTX 3090 24GB (stessa GPU per ogni confronto, ordine
interleaved, env-capture per pod) che fondano:
- la **RITRATTAZIONE dell'asimmetria HOT/COLD**: rep-rate 4-gram sovrapposti — HOT
  [0.026, 0.026, 0.064] vs COLD [0.033, 0.045, 0.029], con aderenza-mask verificata su tutti e 6 i run;
- il **contrasto ordinale paired** rand/reap = **1.345× CI95 [1.270, 1.423]** (reap/full 1.009×
  [0.972, 1.035] resta [OPEN], CI attraversa 1.0);
- la correzione velocità: headline = **mask STATICA** (file-mask keep-23 **17.3 t/s** hit 0.986 vs
  full no-mask 3.6 t/s), NON la staircase (2.5 t/s, cache-poison).

## File chiave (reali sul branch)

- `SUMMARY.md` — verdetto completo: ritrattazioni, tabelle, istruzioni d'innesto nel paper
- `paired/paired_results.csv` + `paired/ppl_p_{full,reap,rand}_s{1..3}_c{0..3}.log` (36 run ppl) + `paired/env.txt`
- `abc/ppl_results.csv`, `abc/gen_a_{hot,cold}_s{1..3}.log`, `abc/mask_a_*.txt`, `abc/env.txt`

## Come ottenerlo

```bash
cd <checkout di moe-aggressive-commit>
git fetch origin reap/k91-coding-vram
git checkout origin/reap/k91-coding-vram -- runs/reap/multiseed_2026-07-07
# oppure, senza toccare il working tree:
git show origin/reap/k91-coding-vram:runs/reap/multiseed_2026-07-07/SUMMARY.md
```
