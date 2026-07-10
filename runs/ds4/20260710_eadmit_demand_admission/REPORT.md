# E-ADMIT — demand-driven admission (constant-K targeted exchange), simulazione offline

**Data:** 2026-07-10 · **Modo:** OFFLINE, solo trace su disco (no GPU / WSL / pod)
**Script:** `scripts/simulate_demand_admission.py` (riproducibile, `py_compile` clean)
**Artefatti:** `policy_summary.csv`, `sweep_C.csv`, `boundary_transitions.csv`,
`cov_curves_bestC.csv`, `stats.json`

## Il meccanismo sotto test (utente)

**Demand-driven admission**: mask a **K costante** dove un expert VIETATO
(fuori-mask) viene **AMMESSO** — sfrattando il keep meno usato (EWMA minima) —
quando la sua domanda bloccata è **forte e persistente** (CUSUM per-expert con
soglia + persistenza). Mai tornare a K0, mai re-ranking all'ingrosso. Da
distinguere nettamente dal **rotate periodico wholesale** (rotate32), già
bocciato dai dati (E-CAL regime notes; collasso pod ~gen126,
`runs/ds4/20260710_scope_divergence_pod/README.md`): qui lo si tiene solo come
**riferimento** di quanto costa/rende il re-ranking totale.

## Dati

Trace FULL-model **con pesi** (nessuna mask attiva ⇒ la domanda del router è
direttamente osservabile), schema `pos,layer,n,e0..e5,w0..w5` (pesi = softmax
non-biased dei 6 selezionati; copertura def-1 "routed mass" come E-CAL/E-PHASE):

| Fonte | Trace | Uso |
|---|---|---|
| `runs/ds4/20260710_pod_cache1024_warmup_replay/` | `route_W130.csv` (129 tok) | eval (79 tok post-warmup) |
| idem | `route_W50.csv` (49 tok) | **escluso**: 0 tok post-warmup (solo warmup) |
| `runs/reap/k91_coding_vram/trace_coding.tgz` (parsing convenzioni E1 `scripts/analyze_top_expert_mass.py`) | 10 trace coding da 255 tok | eval (205 tok post-warmup ciascuno) |
| idem | `test-mock` (52 tok) | **escluso**: 2 tok post-warmup < 30 |
| `runs/ds4/20260710_scope_divergence_pod/` ctrl (full "router libero") | — | **DICHIARATO INUTILIZZABILE**: ctrl ha testo ma NESSUN trace di routing su disco; `r1/s1_r1.csv.gz` logga solo S1 aggregato (pos,layer,pruned,total) senza id degli expert |

Totale eval: **11 trace, 2129 token post-warmup**. Fasi strutturali e confini da
segmentazione E-PHASE (`<style>`/`</style>`/`<script>`, fence ```` ``` ````;
allineamento char→tok proporzionale, errore ±5 tok). 5 trace hanno un confine
strutturale **post-warmup** (prose→code): js-debounce, sql-window, c-pointers,
git-rebase, rust-owner.

## Metodo

Per ogni trace: mask iniziale = **top-23/layer dalla massa dei primi 50 tok**
(il W50 frozen; stessa costruzione E-PHASE, sanity: css W130 frozen 0.604 qui vs
0.641 E-PHASE su finestra leggermente più larga). Poi cammino token-per-token
(decisioni causali: la mask al token t è decisa dai token ≤ t−1) e confronto:

- **(A) FROZEN** — mask fissa.
- **(B) ROTATE-32** — re-ranking completo top-23 da EWMA della massa (decay
  0.98 = `DS4_PACE_ROTATE_DECAY` del manifest pod) ogni 32 tok. Riferimento
  bocciato.
- **(C) DEMAND-ADMIT** — per ogni expert fuori-mask: CUSUM della quota di massa
  richiesta `S ← max(0, S + share − k_d)`; ammissione quando `S ≥ h` con
  persistenza ≥ `p` token richiesti nell'escursione; sfratto del keep con EWMA
  minima (cooldown anti-thrash 16 tok sul neo-ammesso); K costante. Sweep
  3×3×3: h ∈ {0.3, 0.6, 1.2}, k_d ∈ {0.01, 0.02, 0.04}, p ∈ {2, 4, 8}.
- **(D) C + confini strutturali** — per 16 tok dopo un confine E-PHASE la
  sensibilità è raddoppiata (h/2, p/2).

Metriche (regione eval): copertura istantanea def-1 (media, p10, ultimo terzo
"tardivo", per fase); churn (scambi totali, per-100-tok, GiB a 6.75 MiB/exp);
lag di recupero post-confine (vs livello pre-confine, smussato 5 tok, tolleranza
2 pt); dip/plateau di transizione ([b,b+16) e [b+16,b+48)); scambi-rimbalzo
(ammesso→ri-sfrattato entro 100 tok).

## Risultati

### Tabella pooled (11 trace, config C/D raccomandata h=1.2, k_d=0.02, p=2)

| Politica | cov media | cov p10 | **cov tardiva** | scambi/100tok | GiB totali | rimbalzi |
|---|---|---|---|---|---|---|
| (A) FROZEN | 0.510 | 0.328 | **0.485** | 0 | 0 | 0 |
| (B) ROTATE-32 | 0.641 | 0.478 | **0.662** | 706.3 | 99.9 | 7826 (**51.7%** dei 15149 ingressi) |
| (C) DEMAND-ADMIT | 0.612 | 0.489 | **0.622** | **130.2** | **18.3** | **8 (0.3%** di 2778 ammissioni) |
| (D) C + confini | 0.620 | 0.498 | 0.632 | 142.7 | 20.2 | 23 |

**C recupera 13.7 pt della copertura tardiva persa da A (48.5→62.2) = 77% del
gap A→B, con 5.4× meno churn di B** (0.86 vs 4.7 GiB/100tok) e rimbalzi
~zero (0.3% vs 51.7%: metà degli scambi wholesale di B è rumore che rientra
entro 100 tok).

### Frontiera churn↔copertura (sweep CUSUM, gain tardivo vs A in pt)

| (h, k_d, p) | gain tardivo | scambi/100tok | churn vs B | rimbalzi |
|---|---|---|---|---|
| (1.2, 0.04, 4) | +10.1 | 74.4 | 0.11 | 0 |
| **(1.2, 0.02, 2) racc.** | **+13.7** | **130.2** | **0.18** | 8 |
| (0.6, 0.02, 8) | +15.7 | 190.8 | 0.27 | 93 |
| (0.3, 0.02, 8) max-recovery | +16.4 | 221.0 | 0.31 | 193 |
| (0.3, 0.01, 2) oltre-B | +21.1 | 1113.2 | 1.58 | 15967 |

Monotona e regolare: si compra copertura con churn. Il ginocchio utile è
h≈0.6–1.2 con k_d≈0.02–0.04. Superare B in copertura è possibile solo
superandone il churn — non è il regime interessante.

### Andamento nel tempo (curve in `cov_curves_bestC.csv`)

FROZEN erode con la profondità del documento: media 0.51 → 0.485 nell'ultimo
terzo, con crolli di fase (js-debounce: prose 0.55 → code 0.33; sql-window:
0.55 → 0.33). C insegue B a 1–4 pt di distanza per tutta la traiettoria; in un
caso (api-paging) C **supera** B (0.629 vs 0.595 tardiva: l'EWMA wholesale di B
resta ancorata al passato, il CUSUM mirato no).

### Transizioni di fase (5 confini prose→code post-warmup)

| Politica | dip [b, b+16) | plateau [b+16, b+48) |
|---|---|---|
| (A) FROZEN | 0.408 | 0.363 (continua a scendere) |
| (B) ROTATE-32 | 0.578 | 0.639 |
| (C) DEMAND-ADMIT | 0.558 | 0.567 |
| (D) C + confini | **0.621** | 0.600 |

Il **lag classico è risultato non informativo** (media ~0 tok, 0/5 non
recuperati per TUTTE le politiche): al confine non c'è un gradino secco perché
il livello pre-confine è già eroso — la firma vera è il **livello** dip/plateau
qui sopra. C dimezza il buco di transizione di A e si stabilizza ~20 pt sopra;
D (boost al confine) recupera altri +6.3 pt nel dip a +10% di churn: il confine
strutturale **aggiunge**, ma è un raffinamento, non il grosso dell'effetto.

### Stabilità dei parametri tra trace

Best per-trace (vincolo churn ≤ B/3): h ∈ {0.3–1.2}, k_d ∈ {0.01–0.04}, p
quasi sempre 8 — ma il **regret** del config unico raccomandato (1.2, 0.02, 2)
vs il best di ciascun trace è **mediana 3.4 pt, max 6.0 pt** su un gain di
13.7: un solo settaggio funziona ovunque. `p` è una leva debole (con h ≥ 1.2 la
persistenza ≥ 3 tok è implicita perché share ≤ ~0.5/tok); dominano h e k_d.

## VERDETTO

**POSITIVO.** DEMAND-ADMIT (C) recupera **13.7 pt** di copertura tardiva persa
da FROZEN (0.485 → 0.622; B rotate32 0.662), chiudendo il **77%** del gap A→B
con churn **5.4× sotto** il rotate wholesale (130 vs 706 scambi/100tok; 0.86 vs
4.7 GiB/100tok) e **rimbalzi ~zero** (0.3% vs 51.7% di B). I parametri CUSUM
sono stabili tra trace (regret mediano 3.4 pt). (D) aggiunge +1.0 pt pooled e
+6.3 pt nel dip di transizione per +10% churn: opzionale, dipende da un
detector di confini live. Merita la **candidata patch 0026** e l'A/B live S3.

## Limiti (onestà)

1. **Traiettoria sana**: la simulazione su trace full misura la domanda della
   traiettoria NON mascherata. Nel runtime reale la domanda è condizionata alla
   traiettoria mascherata: questi numeri stimano il **potenziale di copertura**
   del meccanismo, non l'esito di qualità — quello lo decide **solo l'A/B live
   (S3)**.
2. **Visibilità top-6**: il trace logga solo i 6 expert selezionati dal router
   libero; la domanda fuori-mask è visibile solo quando l'expert entra nel
   top-6. Il segnale live (router_probs non-biased su tutti i 256, come letto
   dal sensore 0012 / rmass 0020) è PIÙ ricco ⇒ il CUSUM live vedrebbe la
   domanda prima, non dopo.
3. Allineamento token↔testo proporzionale (±5 tok) per fasi/confini (regola
   E-PHASE); trace corti (≤255 tok, 2129 tok eval totali) e due soli domini
   (html, coding); il controllo full di scope_divergence è inutilizzabile
   (nessun trace di routing su disco).
4. Costo churn espresso in GiB di delta-prefetch (6.75 MiB/exp, meccanismo
   0021): ~0.9 GiB/100tok raccomandato. La fattibilità di banda sul 3060 va
   confermata live (lo smoke 0021 sul pod ha già dimostrato paging per-expert
   senza WRAP full).

## Spec candidata patch 0026 — "demand-driven admission" (registrata in `patches/README.md`)

1. Base: live-tree post-0018 + 0020 + 0021 (stessa ancora delle sorelle).
2. Riusa il rmass **per-expert** della 0020 (router_probs non-biased, già
   letti per S1): nessun nuovo sensore.
3. Stato per (layer, expert fuori-mask): CUSUM `S ← max(0, S + share − k_d)`
   + contatore di persistenza; azzerato su ammissione/sfratto.
4. Trigger per-token (post-warmup, mask attiva): se `S ≥ h` e persistenza ≥ `p`
   ⇒ ammissione: sfratta il keep con EWMA minima (decay 0.98), K invariato.
5. Lo swap fisico usa il **delta-prefetch della 0021** (un expert = 6.75 MiB,
   niente WRAP full su decode).
6. Cooldown anti-thrash: il neo-ammesso non è sfrattabile per 16 tok.
7. Mai ritorno a K0; nessun re-ranking wholesale; compatibile ma alternativo a
   ROTATE (con ADMIT attivo si raccomanda `DS4_PACE_ROTATE=0`).
8. Env: `DS4_PACE_ADMIT=0` **default off**; `DS4_PACE_ADMIT_H=1.2`,
   `DS4_PACE_ADMIT_K=0.02`, `DS4_PACE_ADMIT_P=2`,
   `DS4_PACE_ADMIT_COOLDOWN=16`; telemetria: eventi `admit(expert,evicted,S)`
   nel pace jsonl.
9. Boost ai confini strutturali (variante D): NON in v1 — richiede un detector
   di confini live; riaprire dopo l'A/B.
10. A/B live S3 (gate): W50+K23 FROZEN vs FROZEN+ADMIT, n≥3, collapse-rate e
    copertura S1 come metriche; churn atteso ~1 GiB/100tok.

## Prossimo passo

A/B live S3 (pod o 3060): il verdetto qualità NON è deducibile da questa
simulazione (limite 1). Riga aggiunta alla tabella S3 di
`docs/SOTA_ROADMAP.md` accanto alla leva rewind.
