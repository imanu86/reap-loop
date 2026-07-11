# MAN/MSAN gate-free score feasibility (2026-07-11)

**Mode:** OFFLINE, trace-only (no GPU/WSL/pod) · **Script:** `scripts/analyze_man_score_feasibility.py`
**Candidato:** `docs/MOE_ECOSYSTEM_SURVEY_20260711.md` §1A + §4 #4, arXiv **2606.15716**

## Formula (dal paper, fetch 2026-07-11)

Scoring unificato: `S_j(b,α,β) = (1/N_j^b) · Σ_t 1[j∈E_t] · g_{j,t}^α · ||f_{j,t}||₂^β`
(`g` = peso del gate router, `f_{j,t}` = **vettore di attivazione di OUTPUT dell'expert**
post-FFN pre-combine, `N_j` = conteggio token routati a *j*). Casi speciali del paper:

| Score | (b,α,β) | Formula | Serve `‖f‖`? |
|---|---|---|---|
| Frequency | (0,0,0) | Σ 1[j∈E_t] | no |
| **SEER** | (0,1,0) | Σ 1[j∈E_t]·g | no |
| EAN | (0,0,1) | Σ 1[j∈E_t]·‖f‖ | **sì** |
| REAP (vero, arXiv 2510.13999) | (1,1,1) | media(g·‖f‖) | **sì** |
| **MAN** | (1,0,1) | media(‖f‖) — gate-free | **sì** |
| **MSAN** | (1,0,2) | media(‖f‖²) — gate-free | **sì** |

## Scoperta chiave — il punteggio attuale del repo NON è "REAP"

`build_session_mask_canonical.py --mode weighted` (il ranking di produzione, massa-gate
cumulata per layer) è **esattamente SEER** (b=0,α=1,β=0), non REAP: nessuna dipendenza da
`‖f‖`. `--mode unit` è **esattamente Frequency** (b=0,α=0,β=0). `docs/SCALE_FRONTIER_VERDICT.md`
aveva già flaggato la collisione di nome con REAP (arXiv 2510.13999, Cerebras) — questo
conferma che non era solo una collisione di nome, il criterio effettivamente implementato è
diverso (manca del tutto l'asse activation-norm).

## Verifica empirica — nessun trace ha la colonna richiesta

MAN, MSAN, EAN e il vero REAP richiedono TUTTI `‖f_{j,t}‖₂`. Header controllato su **18
sorgenti** (2 html, 5 narrow, 11 coding) + il trace S1 di `20260710_scope_divergence_pod`:
zero eccezioni, schema ovunque `pos,layer,n,e0..e5,w0..w5`. Confermato anche leggendo
`docs/REAP_DS4_design.md` / `docs/DS4_ROUTING_RESIDENCY_TRACE.md`: `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`
cattura SOLO i pesi del gate, mai una norma di attivazione. **MAN/MSAN non sono calcolabili
dai trace esistenti, punto.**

## Cosa È misurabile ora — limite inferiore sull'asse gate-weight

Le uniche due caselle della tabella già coperte dal codice esistente sono Frequency e SEER
(`--mode unit` / `--mode weighted`). Rimuovere SOLO l'asse gate-weight (α: 1→0, SEER→Frequency,
β resta 0 in entrambi — l'asse activation-norm non è mai toccato) sposta già la keep-23 mask:

| Confronto | Jaccard medio (40 layer) |
|---|---:|
| SEER (weighted) vs Frequency (unit), per-source, media di 17 sorgenti | **0.799** |
| SEER vs Frequency, pool di tutte le sorgenti | **0.799** |

~20% della mask cambia già rimuovendo un solo asse (gate-weight) mentre l'asse
activation-norm resta a zero in entrambi i termini del confronto. È un **limite inferiore
onesto**: attivare l'asse `β` (quello che MAN davvero introduce) plausibilmente sposta la
mask almeno altrettanto, probabilmente di più — ma questo NON è misurato qui, e non può
esserlo senza nuova strumentazione.

## VERDETTO

> **A/B RUNTIME NON GIUSTIFICATO ORA — bloccato da un gap di dati, non da mancanza di
> interesse.** MAN/MSAN richiedono `‖f_{j,t}‖₂`, assente in ogni trace su disco. Non è
> implementabile come `--mode man` in `build_session_mask_canonical.py` senza prima
> estendere la cattura runtime. Il segnale parziale disponibile (Jaccard 0.799 SEER-vs-Frequency)
> conferma solo che la mask *è* sensibile a quale asse si sceglie di pesare — non dice nulla
> su exactly quanto sposterebbe MAN specificamente.

### Spec di implementazione (per sbloccare, in ordine)

1. **C-side (ds4.c, patch 0012/`DS4_SPEX_TRACE_ROUTING_WEIGHTS`)**: per ciascuno dei 6 expert
   selezionati il forward calcola GIÀ il tensore di output FFN `f_{j,t}` prima della combine
   pesata dal gate. Aggiungere una L2-norm su quel tensore già materializzato è marginale —
   **6 riduzioni norm/token/layer, zero matmul aggiuntivi, zero esperti extra valutati**
   (nessun costo di "the missing expert must be computed to be scored" — è già in memoria).
   Estendere `DS4_SPEX_TRACE_ROUTING_WEIGHTS` con 6 colonne `n0..n5` (norma nello stesso
   slot di `e0..e5`).
2. **Python-side (`build_session_mask_canonical.py`)**: banale una volta presente la colonna —
   `read_route_trace()` accumula `actnorm_sum[layer][e]` / `actnorm_count[layer][e]` dalla
   nuova colonna; `score_man = actnorm_sum/actnorm_count` (β=1), `score_msan =
   Σ(norm²)/count` (β=2); `rank_keep()` è riusato **senza modifiche** (già agnostico al
   segnale di ranking — accetta qualunque `score[layer][expert]`).
3. **Effort**: BASSO-MEDIO lato C (nuova metrica letta da un tensore già calcolato, non nuovo
   compute), BASSO lato python (estensione di un'interfaccia già generica).

Solo dopo il passo 1 questo script (o una sua estensione con `--mode man/msan`) diventa
eseguibile offline sui prossimi trace catturati, ripetendo esattamente questa stessa analisi
Jaccard sui casi mancanti (SEER vs MAN, Frequency vs MAN, ecc.) invece che solo sull'asse
gate-weight.

### Riprodurre
```
python scripts/analyze_man_score_feasibility.py \
  --reap-loop-root <reap-loop> --moe-root <moe-aggressive-commit>
```
Output: `stats.json` (formula, verifica schema, Jaccard per-source e pooled, spec),
`per_layer_seer_vs_frequency_jaccard.csv`.
