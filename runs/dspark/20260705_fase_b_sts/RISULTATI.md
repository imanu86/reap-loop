# Fase B (offline) — STS fittata + scheduler R=1 simulato su dati reali

**Data**: 2026-07-05 · tutto offline/local, zero GPU, zero pod.
**Input**: `accept.csv` (585 cicli teacher-forcing del drafter DSpark vero su V4-Flash,
pod B) + curva expert-unici u(k) dalla trace reale (`runs/ds4_routing_trace_smoke`).
**Codice**: `src/msc/dspark/sts_fit.py`, `src/msc/dspark/sched_sim.py`.
**Output**: `sts_params.json` (temperature per posizione, formato pronto per il loader C),
`STS_REPORT.md`, `SCHED_SIM.md`.

## 1. STS (calibrazione sequenziale, fedele a §3.2.1 del paper)

Temperature fittate (grid search sequenziale, target = sopravvivenza cumulata del
prefisso, fit su cicli pari / **validazione su cicli dispari**):
`T = [0.85, 1.07, 0.71, 1.01, 1.44]`

- Holdout: ECE migliora su 4 posizioni su 5 (pos1 0.058→0.039; pos5 0.062→0.055).
- Il guadagno più forte dove serve di più: **chat pos1 0.118→0.059, pos3 0.060→0.030**.
- Su code/math la testa era già quasi calibrata (ECE ~0.01-0.06): la STS lima.

## 2. Scheduler R=1 con cost-model IO-aware (il twist streaming)

Modello di costo: passo = fisso (10% di T1) + draft (10% di T1 per token) +
verifica = L_routed · u(ℓ), con u(k) = 6.30·k^0.668 (fit sui punti misurati
6/10.6/16.4/24.3). Policy valutate sugli ESITI REALI dei 585 cicli.

Numeri chiave (IO per token committato, speedup vs decode liscio):

| policy | TUTTI | chat | code | math |
|---|---|---|---|---|
| fixed-5 (blocco pieno) | 1.46× | **0.92×** (peggio del liscio!) | 1.76× | 1.70× |
| **dynamic-STS** | **1.91×** | **1.71×** | **1.99×** | **1.98×** |
| oracle (limite teorico) | 1.96× | 1.79× | 2.01× | 2.00× |

Letture:
1. **Il blocco fisso su chat DISTRUGGE il guadagno** (0.92×: la verifica sprecata costa
   più di quel che rende) — è esattamente il fallimento del MTP-3/5 statico che il paper
   cita come ragione per cui la produzione DeepSeek era rimasta a MTP-1 (§5.4).
2. **Lo scheduler dinamico con confidenze STS-calibrate raggiunge il 97-99% dell'oracolo**
   in ogni dominio, senza saperne la difficoltà a priori. Su code/math ≈ **2×** sul collo
   IO — il numero promesso dalla tesi del brief ("~2× sul collo di bottiglia").
3. Il dinamico sceglie da solo blocchi corti su chat e lunghi su code — il comportamento
   load/domain-adaptive della Fig. 8 del paper, riprodotto con la nostra cost-model IO.

## 3. Caveat dichiarati (anti-confabulazione)
- Cost-model parametrica (t_fix, δ_draft al 10%): i valori veri si misurano in Fase C sul
  3060; la struttura (IO sublineare nel blocco) è misurata, i coefficienti no.
- La policy dinamica ottimizza la metrica riportata: il finding non banale non è "vince",
  ma che arriva a ridosso dell'oracolo CON confidenze calibrate e che il blocco fisso
  degrada su chat.
- I cicli sono del drafter DSpark (Strada B). Per la Strada A pura (drafter MTP-1,
  acceptance pos1 0.87/0.85/0.60) la meccanica è identica ma servono i margini per-step
  loggati sul 3060 (Fase C) per fittare la STS di quel drafter.

## 4. Prossimo passo (Fase C, on-device)
Innesti C nel ds4 (design §2.3): loader STS (`DS4_DSPARK_STS_FILE`), draft-loop con
early-stop dallo scheduler, sblocco verifier streaming (stream-map per layer) → misura
vera su 3060: byte SSD/token e t/s vs baseline. Con l'MTP-1 attuale il tetto è più basso
del DSpark pieno; se i numeri C confermano la struttura, la Strada B (porting drafter
DSpark in GGUF) ha un business case quantificato: ~2× IO su code/math.
