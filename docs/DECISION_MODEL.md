# DECISION MODEL — reap-loop operativo (sensori · formule · tabella decisioni)

**v0 · 2026-07-11 · fonte di verita del modello:** `scripts/decision_model.py`
(riproducibile) + `runs/ds4/20260711_emodel_decision/{REPORT.md,decision_model.json}`.

> **Perche esiste.** Fino a ieri le config si sceglievano **per tentativi**. Questo
> documento e il **modello di decisione**: dati i sensori invarianti (P2), calcola K*,
> la cadenza di correzione e le soglie — e dichiara i **pochi parametri ancora
> non-identificati** con l'esperimento esatto che li misura. Principio P4 in SOTA_ROADMAP:
> *gli esperimenti identificano parametri del modello, non enumerano config.*

## 1. I tre sensori (tutti INVARIANTI, calcolabili al freeze)

| Sensore | Formula | Fonte | Uso |
|---|---|---|---|
| **Larghezza (identity)** | `union_slope` = nuovi expert distinti /100 tok (fit cumulata union, per-layer); companion `sat_ratio` = union(all)/union(prima meta) | traccia pesata warmup (selected-6/tok, gia loggata da 0020) | stima la **classe di larghezza** -> knee -> K* |
| **Coverage(K)** | `cov(K) = massa(top-K)/tot` sui 6 selezionati | E-CAL (offline) | **solo pavimento anti-under-provisioning** (cov90 ~ K38); NON separa il collasso |
| **Hazard/token** | `lam(K,width)` (sotto) | ledger graduato (retro-grade, M1a/b, armA, pod3, T4, knee, K91) | probabilita di collasso -> useful fraction |

**Attenzione operativa (F2).** La larghezza-identita **non** e leggibile ai 50 tok del warmup
(li e piatta quasi come la massa, `union_fix` CV 0.047). Il segnale discriminante emerge solo su
una finestra **>=100-150 tok**: il sensore deve **continuare ad accumulare la cumulata-union
oltre PACE_WARMUP** (~150 tok) prima del freeze. E l'unica cosa nuova da loggare.

## 2. Formule del modello

**Velocita (E-LAT, 3060 locale — HW-dipendente, mai input del controller):**

    t_ss(K) = 74.9 ms + 258 * miss(K) * 0.952 ms         # miss(K): working-set K*40 vs cache 407
    tps(K)  = 1000 / t_ss(K)        # K12 -> 4.35 ,  K23 -> 3.12 t/s

**Hazard di collasso (2 termini, anchor-scarso, con CI):**

    lam(K, width) = DRIFT(width) + LAM_COV * max(0, exp((knee(width) - K)/SCALE) - 1)
    DRIFT = {narrow 1.0e-4, medium 2.0e-4, wide 4.0e-4}   # residuo demand-shift (E-PHASE)
    knee  = {narrow 20, medium 32, wide 48}               # bersaglio di sizing
    LAM_COV = 2.6e-4    SCALE = 22                         # termine coverage-gap, one-sided

Il `DRIFT` **cresce con la larghezza**: un task largo sposta di continuo gli expert richiesti
(E-PHASE: la mask frozen affama le fasi tardive), quindi collassa lentamente anche ben
provvisto (K91 -> lock ~2476 tok); un task stretto e ~una fase (coffee 0 collassi / 11 600 tok).
**Cioe la larghezza-identita del sensore 1 E lo stesso DRIFT del sensore 3.**

**Costi di correzione (invarianti, in token):**

    rewind ~ 56 tok   (n-gram FIRE mediana 40 + margine 16, docs/S1_REWIND_DESIGN.md)
    breath ~ 70 tok finestra * 15% relearn  (J28 breath 290->370; D6b 13-17%)

**Obiettivo (SOTA-metric):** `good-tok/s = tps(K) * E[useful | K,width,budget,mode]`, vincolo
L2+ mediano. `useful` da hazard+correzioni: `none` = frazione prima del 1o collasso; `rewind` =
budget - lam*budget*56 (fallisce se 56 >= MTTC); `breath` = sopravvivenza a hazard dimezzato
meno overhead relearn.

## 3. TABELLA DECISIONI (per classe di larghezza)

Soglie in **unita invarianti**. K* = ottimo good-tok/s sotto vincolo L2+.

| larghezza | knee | K* (con airbag rewind) | K* (senza airbag) | breath | trigger correzione |
|---|--:|---|---|---|---|
| **stretta** (coffee, JSON) | 20 | **K 12-16** — 4.0-4.3 gtps | K 12 (o >= knee 20) — 3.7 | **no** | n-gram (raro) |
| **media** (Python) | 32 | **K 16** — 3.6 gtps | **K 32 static** — 2.5 | **no** | n-gram |
| **larga** (cyberpunk, frontpage) | 48 | **K 12 + rewind** — 4.0 gtps | **K 48 static** — 1.6 | **no** | n-gram su repeat-rate |

**Regole di lettura:**
- **K piccolo vince SOLO con airbag rewind provato.** Senza airbag il modello ripiega sul
  **pavimento = knee** (stretta 20 / media 32 / larga 48): il floor no-airbag e sempre "K al knee".
- **breath mai raccomandato** in questo modello: su mask piccole/larghe le finestre di relearn
  costano piu di quanto salvano (thrash); il rewind e la correzione giusta.
- **La coverage-cov90 (~K38)** e solo un pavimento anti-under-provisioning, non un predittore.
- **Actuation:** static/frozen (mai rotate: a K23 largo rotate e static collassano uguale, MTTC
  ~1050 entrambi; rotate aggiunge churn senza guadagno). Rotate resta solo se serve churn mirato.

### Recovery-ladder (domanda utente: K12+rewind+breath conviene?)

Su task LARGO, budget 4000 tok — **il modello dice:**

| config | good-tok/s | verdetto |
|---|--:|---|
| **K12 + rewind** | **3.99** | **SI, vince (+156% vs miglior static)** |
| K48 static (no airbag) | 1.56 | fallback |
| K91 static | 1.56 | fallback |
| K12 + **breath** | 1.29 | **NO (sotto K48-static)** |
| K23 static | 0.80 | peggiore |

=> **rewind SI (decisivo), breath NO.** K12+rewind, non K12+rewind+breath.
**Ma:** l'intera colonna "con airbag" e vera solo se il rewind cattura davvero il collasso largo
in ~56 tok — non ancora misurato (esperimento #2).

## 4. Esperimenti di identificazione (i SOLI parametri non-identificabili)

Il fit dichiara tre gambe che non puo risolvere dal disco. **Non chiede di ri-enumerare config**:
chiede di piazzare tre anchor mancanti.

| # | parametro identificato | config esatta | n | costo |
|---|---|---|--:|---|
| **1** | `knee(narrow/medium)` + quale metrica identity linearizza a K* | traccia full-model **trace-on** su coffee + JSON + Python, >=150 tok, log route.csv (selected-6 + pesi) | 1/prompt (routing deterministico) | ~0 se in archivio; else pod ~$0.30 |
| **2** *(pivot)* | `CORR_REWIND_TOK` reale + `useful_frac(K12,rewind)` — il segno di tutta la strategia small-K | build patch 0022, cyberpunk K12 static + rewind da n-gram, >=2000 tok, grade L0-L3 + latenza detection + bit-equality rewind | >=3 | pod A/B ~$1-2 |
| **3** | `miss(K)` sotto K23 (E-LAT extrapola; il bug resident-hit=0 potrebbe appiattirlo) | 3060 locale static K12/K16/K23, cache-isolata, >=800 tok, avg+last t/s | 3 | locale, gratis |

**Gia identificati (non li rifacciamo):** ordinamento di larghezza (F1), hazard wide-K23
(misurato, entrambe le actuation), sopravvivenza narrow (coffee 0/11600), costi di correzione
(S1_REWIND / J28 / D6b).

## 5. Limiti onesti

- Solo **2 task con traccia** (cyberpunk, coding), entrambi larghi -> sensore validato
  sull'**ordinamento**, non sul K* assoluto (l'esp. #1 chiude questo).
- Anchor hazard = **3-4 punti** -> CI larghe (wide-K91 MTTC CI 445-189k). E un'**ipotesi tipata
  con CI**, non una curva misurata.
- `miss(K)` sotto K23 e **ottimistico** (ignora il bug resident-hit) -> il good-tok/s delle
  opzioni small-K e un **tetto** finche l'esp. #3 non lo misura.
- Onset ngram puo **ritardare** la vera perdita di coerenza (~gen126 scope vs 118-848 ngram) ->
  gli hazard wide-K23 sono semmai **sotto**-stimati.
- t/s pod sono RAM-hot, non confrontabili col 3060: trasferiscono solo **hazard (per-token) e
  ordinamento**; ogni t/s assoluto viene dalla calibrazione E-LAT 3060 (flag HW-dip).
