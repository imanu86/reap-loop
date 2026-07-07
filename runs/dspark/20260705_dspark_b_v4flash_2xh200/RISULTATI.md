# Strada B — DSpark ufficiale su DeepSeek-V4-Flash: acceptance misurata

**Data**: 2026-07-05 · **Pod**: 2×H200 secure (~1h, terminato+verificato)
**Modello**: `deepseek-ai/DeepSeek-V4-Flash-DSpark` (167GB fp8/fp4, 48 shard, convert MP=2)
**Metodo**: eval teacher-forcing (`dspark_accept_eval.py`): generazione greedy col target
(ground truth deterministica) + `forward_spec` a ogni token; acceptance = match del draft
col token greedy reale, condizionata al prefisso accettato (stessa definizione della
position-wise conditional acceptance del paper, Fig. 2). 3 prompt (code/math/chat, identici
a quelli del pod A), 200 token l'uno, 195 cicli validi per dominio.
**Fonti su disco**: `accept.csv` (585 cicli: draft, ground-truth, confidence),
`eval_pod.log`, `convert_pod.log`.

## Acceptance condizionale per posizione (drafter DSpark-5: 3 layer MoE + Markov head)

| dominio | pos1 | pos2 | pos3 | pos4 | pos5 | **τ atteso** (bonus incl.) |
|---|---|---|---|---|---|---|
| code | 0.979 | 0.948 | 0.928 | 0.899 | 0.821 | **5.18 / 6** |
| math | 0.959 | 0.936 | 0.903 | 0.880 | 0.863 | **5.00 / 6** |
| chat | 0.713 | 0.612 | 0.635 | 0.667 | 0.500 | **2.70 / 6** |

Lettura: su code/math il drafter addestrato tiene acceptance >0.82 fino a pos5 (decadimento
lento = firma semi-AR del paper); su chat crolla subito (0.71 a pos1) — l'effetto-dominio
della Tabella 1 del paper, riprodotto sul V4-Flash reale.

## Confidence head: calibrazione grezza (σ dei logit, prodotto cumulato vs esito)

| dominio | pos1 ECE | pos3 ECE | pos5 ECE | pattern |
|---|---|---|---|---|
| code | 0.014 | 0.087 | 0.151 | sottoconfidente in profondità (0.68 predetto vs 0.82 reale) |
| math | 0.011 | 0.068 | 0.180 | idem (0.70 vs 0.86) |
| chat | 0.118 | 0.261 | 0.363-0.441 | fortemente sottoconfidente da pos3 |

**Implicazione diretta per la Strada A**: il segnale di confidenza esiste ed è ottimo come
ranking, ma i valori assoluti (che servono allo scheduler per stimare τ) richiedono la
calibrazione sequenziale — esattamente la STS del paper (§3.2.1), che DeepSeek applica
post-hoc per la stessa ragione (loro ECE raw 3-8% → ~1%). Le 585 righe di `accept.csv`
sono il primo dataset di calibrazione, sul modello vero.
NB: le colonne c1..c5 del CSV sono LOGIT pre-sigmoide della confidence head (fp32).

## Confronto col baseline MTP-1 (pod A, stesso modello 2-bit ds4, stessi prompt)

| misura | code | math | chat |
|---|---|---|---|
| MTP-1 probe (ds4, acceptance top-1 per token) | 0.872 (×2 run identici) | 0.846 | (in corso) |
| DSpark pos1 (drafter addestrato) | 0.979 | 0.959 | 0.713 |

Il drafter addestrato (3 layer + KV-injection) batte nettamente la testa MTP-1 già alla
prima posizione su code/math — coerente con la "capacity advantage at position 1" del
paper (§4.3.1). ⚠️ Confronto indicativo: target a precisioni diverse (fp8/fp4 su H200 vs
2-bit imatrix su ds4) — la parte trasferibile è l'ordine di grandezza e la struttura, non
il terzo decimale.

## Cosa significa per il track
1. **Il 49% è realistico da monetizzare**: con τ≈5 su code/math a blocco-5, il regime
   streaming caricherebbe ~16-24 expert unici/layer per ~5 token accettati invece di 30
   (6×5) — la sinergia blocco×union-load ha il carburante che serve.
2. **Strada A** ha ora sia il baseline (MTP-1 ~0.85-0.87 top-1) sia il tetto (DSpark τ≈5)
   misurati sul modello vero: lo scheduler confidence-scheduled ha spazio di manovra
   ENORME su chat (τ 2.7: tagliare i draft inutili) e poco da tagliare su code/math
   (verificare blocchi pieni).
3. **Strada B integrazione in ds4**: il modulo pesa 3 layer MoE + teste — è "un secondo
   MTP più profondo". Il GGUF MTP attuale (3.8GB, 1 layer) diventerebbe ~3× — fattibile
   nel budget RAM del 3060 (28GB), da progettare dopo i numeri IO della fase C.
