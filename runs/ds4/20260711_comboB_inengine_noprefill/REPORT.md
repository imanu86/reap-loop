# COMBO B — in-engine continuo (no re-prefill) vs two-phase — REPORT

Verdetto (3 domande):
1. In-engine elimina il doc-restart? SÌ, alla radice: <!DOCTYPE>=1 su 0/3 vs two-phase 2 su 3/3. Conferma T4: il restart è artefatto del RE-PREFILL, non del cut.
2. Alza il cap L2→L3 sul coffee? NO, lo ABBASSA L2→L1. Il re-prefill non era un difetto: RI-ANCORAVA il documento (riscrive una stesura fresca e completa). Rimosso, lo stream nudo degenera (loop 2/3 o JS corrotto).
3. Sul wide fa meglio? NO, entrambi L0: il collasso-wide è INTRINSECO ad aggressive-K + decode continuo, a monte del re-prefill.

## Tabelle (pod4 3090Ti, canonical-v2 bfa987e, greedy, n=3 ABAB)
COFFEE ctx4096 cache256 W50 K23 weighted 1200 tok:
- A two-phase: L 2,2,1 (med L2) | restart 3/3 | </html> 3/3 | 1.98 t/s | doc completo, cap L2 per 2° DOCTYPE
- B in-engine: L 1,1,0 (med L1) | restart 0/3 | </html> 1/3 | 1.98 t/s | stream nudo degenera
CYBERPUNK ctx8192 cache1024 4050 tok:
- A two-phase: L0×3 | restart 3/3 | </html> 0/3 | 24.4 t/s | 8344 char ×3 byte-identici
- B in-engine: L0×3 | restart 0/3 | </html> 0/3 | 17.8 t/s | 14570 char ×3 byte-identici

## Lettura (il vero yield)
Il doc-restart NON è la leva-qualità che T4 faceva sospettare: è un SINTOMO accoppiato a un RECOVERY UTILE. Il re-prefill [prompt+frozen] fa ricominciare il documento da capo pulito e completo (tutte le feature, JS valido, </html>), lasciando solo il primo DOCTYPE cosmetico → grader cappa a L2, ma il documento usabile c'è. L'in-engine rimuove il DOCTYPE a costo-velocità zero MA togliendo il re-ancoraggio espone lo stream alla degenerazione. => il re-prefill è una forma NATIVA (grezza) di recovery-by-restart; il rewind è la sua versione chirurgica. Il two-phase resta > in-engine. La leva 7x fase2/fase1 resta NON convertita in qualità.

Caveat: t/s pod diagnostici (RAM 72GB<81GB modello, rapporti trasferiscono non assoluti); cyberpunk cache1024 (~2x decode, qualità cache-indip.); grader euristica-JS senza node.
