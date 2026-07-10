# Lever Retrospective + Pattern Analysis — 2026-07-11

**Scopo.** Revisione lungimirante (far-sighted) di *tutte le leve tirate* nel track
ds4 / REAP-LOOP, letta attraverso il decision model E-MODEL
(`scripts/decision_model.py`, output `runs/ds4/20260711_emodel_decision/decision_model.json`).
Il principio guida di questa revisione è **P4 — IDENTIFICAZIONE DI PARAMETRI, NON
ENUMERAZIONE DI CONFIG**: estensione di P1-P3 (`docs/SOTA_ROADMAP.md`). Non si
tratta di provare più valori di K/W finché uno regge; si tratta di *misurare i
parametri* (knee(width), DRIFT, LAM_COV, costo-rewind) che rendono il regime
predicibile con 2-3 punti invece che con una griglia. Ogni riga porta la fonte.

> Ground truth vincolante: in caso di conflitto vince `docs/CLAIMS_CURRENT.md`.
> Questa è una retrospettiva di metodo, non una fonte di claim.

---

## 1. Tabella leve (tirata / esito / fonte)

| # | Leva (cosa si è tirato) | Esito misurato | Verdetto | Fonte |
|---|---|---|---|---|
| L-STATIC | Mask **statica K23** 2-phase, freeze-safe | Narrow (coffee): L2 tiene (0 collassi / 11600 tok). Wide (cyberpunk/frontpage): L0, loop ~tok100-120 | Base velocità, NON risposta qualità sul wide | `t4_t5_w_sweep_local/`, armA `pod_static_ab_ctx8192/`, CLAIM-005 |
| L-ROTATE | **rotate32** (rotazione periodica) | Wide K23: collassa comunque, MTTC 1051; costo −0.36/−0.45 t/s vs static | La rotazione NON compra qualità sul wide; è destabilizzante | M1a `m1a_w50_w100_ctx8192_n3/`, T2 |
| L-STOP | rotate + **stopper anti-ripetizione (airbag M1b)** | Recupera 1/3 rollout a L2, mediana L1; 0 token utili post-return quando fira tardi | Airbag (ultima risorsa), non cura | M1b `m1b_w50_stopguard_ctx8192_n3/` |
| L-FROZEN | **Mask session-learned congelata**, no re-prefill | Narrow coffee: L1×3 chiude. Wide cyberpunk: L0 loop tok68 | Provenienza session ≫ cold; ma non salva il wide da sola | pod3 `pod3_s3_ab_frozen_vs_admit/` |
| L-ADMIT | **Demand-admit (0026)** on top del frozen | Config C fallisce il gate qualità live | Chiusa negativa per quel setpoint | pod3 S3 A/B `REPORT.md` |
| L-WSWEEP | **Warmup sweep W30…150** (freeze a boundary sicuro) | W50/W130 coffee L2; W30 borderline L1; esito flippa sul CARATTERE di taglio (`}`/`;`/`>`) | Il freeze-point è una **lotteria**; W non è una manopola pulita | T4 `t4_W0*/summary_median.csv` |
| L-WEIGHT | **Weighted vs unit mask** (T5 ABAB) | Weighted ≳ unit (marginale); doc-restart universale in fase2 su entrambi | Weighted non è LA leva | T5 `runs/…20260710_t4_t5_w_sweep_local/t5_*` |
| L-SAMPLE | **Sampling under mask** (temp0.7/top_p0.95 vs greedy) n=3 | Fenotipo INVARIATO: mediana L2=L2, loop 0/3=0/3, restart 3/3 persiste | Il sampler **non** è la leva | commit `91ea3bd` SAMPLING-UNDER-MASK |
| L-COV | **Adaptive-K coverage (cov90)** | E-CAL NEGATIVO: cov@23≈79% (73-82%) task-invariante; Kmin-cov90≈38 per html *e* 11 coding | NON discrimina larghezza; uso legittimo solo come **pavimento anti-under-provisioning** | L5 / `ecal_coverage_threshold/`, spec 0024 |
| L-CACHE | **Cache sweep 64/128/256/1024** | cache1024 ripristina throughput (14-16 t/s) ma NON la qualità cyberpunk | Leva velocità, non qualità | `pod_cache1024_*`, cache-sweep rows |
| L-PREC | **Mixed precision q2 vs q2-q4** | Entrambi L1 sul task hard; q2-q4 +19% tempo | **RETRACTED** lose-lose: il soffitto è la TAGLIA, non i bit | CLAIM-013 |
| L-DYN | **Dynamic staircase / learn-live (0014)** | Avvelena la cache; direct-descent vince | **RETRACTED**; path dinamico OFF | CLAIM-016 |
| L-K91 | **Mask statica larga K91** (coding) | Erosione lenta: coerente ~2200, text-lock 2476 | Ancora DRIFT_wide ≈ 1/2476; non entra in 12GB reali | `scope_divergence_pod/`, memory reap-k91 |
| L-KNEE | **Knee ladder cold-static** (JSON k20 / Python k32 / Frontpage) | JSON keep-20=L3; Python keep-32=L3 (keep-28 rompe); Frontpage collassa a ogni K cold | Il **ginocchio scala col task** | CLAIM-005 |
| L-BREATH | **clock_breath64** (breath periodico) | Breath dopo danno n-gram = 0 token utili post-return; breath corto/precoce ha segnale velocità positivo (2.78-2.97 t/s) | Timing-critico; tardi = inutile | L6, `pace_advanced_ab_*` |
| L-SPEX | **SPEX hidden prefetch** | recall 0.62-0.81 costante-su-cache ma t/s: RAM-served 2.5-4.5× più LENTO | Aiuta SOLO se deeply-SSD-bound | CLAIM-014 |
| L-REWIND | **S1-guided rewind (0022)** — CORREZIONE | **Progettato, NON ancora misurato.** Il decision model lo indica come modo a valore massimo | Da misurare (vedi Mossa 1) | `S1_REWIND_DESIGN.md` |

---

## 2. Pattern trasversali

**PAT-1 — Attuazione e provenienza sono le leve causali; copertura%, sampler,
cache-size e precisione sono INERTI / task-invarianti.** I separatori reali del
collasso sono: *modo di attuazione* (static ≫ rotate), *provenienza mask*
(session ≫ cold), *budget token*, *larghezza task* (knee). C'è un vero e proprio
**cimitero di "la manopola che credevamo contasse non è quella"**: coverage
(E-CAL negativo), sampler (fenotipo invariato), asimmetria HOT/COLD (retratta),
mixed-precision (retratta lose-lose), dynamic staircase (retratta). Tutte
condividono la stessa forma: la variabile appariscente è task-invariante o inerte;
il driver reale è altrove.

**PAT-2 — La larghezza è la variabile nascosta ed è leggibile per IDENTITÀ, non
per massa.** La copertura-di-massa cov@23 è piatta fra i task (CV 0.033); ma
l'*union_slope* identità (nuovi esperti reclutati / token) ha **CV 0.49 — ~15×
più discriminante** (html 107.9±35.1 vs coding 52.0±18.3; html_W50 143 vs
code_p05 40). Un task wide continua a *reclutare* esperti → alza il knee e lascia
un **DRIFT residuo** anche a K≥knee (E-PHASE: mask congelata affama le fasi
strutturali tardive, −17.6pt copertura). Questo lega il sensore-larghezza al hazard:
la stessa non-stazionarietà che alza il knee fissa il DRIFT.

**PAT-3 — Il collasso è un HAZARD (tasso/MTTC), non un verdetto; e sull'obiettivo
good-tok/s la CORREZIONE batte la PREVENZIONE.** Riformulare il singolo rollout
come hazard per-token (wide-K23 MTTC≈1051; wide-K91≈2476; narrow-K23 → ∞) e il
greedy come *distribuzione* (non riproducibile run-to-run, PAT-4) trasforma "K
regge?" in "quanto vale lam(K)?". Una volta che è un tasso, un **rewind economico
(~56 tok) che fira ogni MTTC domina il pagare throughput per alzare K**: nel
decision model **K12+rewind vince in TUTTE e tre le classi** (wide good_tps 3.99
vs 1.56 miglior static; narrow 4.30 vs 3.72; medium 3.56 vs 2.46). **Questo È P4**:
si identificano DRIFT/knee/costo-rewind e si sceglie il regime, non si enumera K.

**PAT-4 (di supporto) — Ogni verdetto è una distribuzione, non un punto.** Greedy
non riproducibile a temp0 (locale diverge tok~75; pod cyberpunk bit-identico n=3
ma coffee L1/L1/L3). Il freeze-point è una lotteria sul carattere di taglio. →
la metrica primaria è il **collapse-rate**, non l'esito singolo; ogni n=1 è sospetto.

---

## 3. Combinazioni NON testate, ordinate per valore atteso

| Rank | Combinazione | Predizione E-MODEL / razionale | Perché alto valore | Costo |
|---|---|---|---|---|
| **A** | **K12-static + S1-rewind su WIDE** (cyberpunk), n=3, ≥4000 tok | good_tps **3.99 vs 1.56** miglior static; MTTC 678, ~6 correzioni; useful 0.92 vs 0.50 | È l'UNICO esperimento che conferma/refuta la tesi centrale del decision model (correzione ≫ prevenzione = P4). Sblocca l'intera tabella | 1 run pod/locale; ma richiede il primitivo rewind (unbuilt) |
| **B** | **Two-phase W50 session-learn IN-ENGINE sul 3060 + frozen low-K, no re-prefill** (L1) | Leva velocità+qualità più grande MAI misurata (fase2/fase1 = 14.6/2.03 ≈ 7×) | Entrambe le gambe già confermate indipendentemente (session≫cold, low-K throughput); manca SOLO l'integrazione in-engine sull'HW target. Chiude il più grande buco cieco | Integrazione engine, medio |
| C | **cov90-floor static (K≈38) + S1-rewind** su wide | Testa se alzare K al knee riduce le correzioni abbastanza da battere K12+rewind | Accoppia l'unico uso legittimo di coverage (pavimento) con rewind; 2 punti per fittare LAM_COV | 1 run + rewind |
| D | **S1-slope (0020) come TRIGGER di rewind/rotate** su wide | Unico segnale con ~200 tok di preavviso; n-gram tardivo 6× | Trasforma l'anomalia aperta CLAIM-011 in una soglia identificata in unità invarianti | patch 0020, medio |
| E | **K91-static PER-SESSIONE (live, non cold) + rewind** su wide | K91 è l'ancora DRIFT (tiene 2476); rewind per superare il text-lock | Valore minore: K91 non entra in 12GB → fuori HW target | pod, alto |

**Le due a valore atteso più alto = A e B.**

---

## 4. Anomalie aperte (con ipotesi)

| # | Anomalia | Ipotesi |
|---|---|---|
| AN-1 | keep-23 **STATIC** sopravvive (L2/L3) ma keep-23 **ROTATE** collassa alla copertura IDENTICA 0.79 | La rotazione È causale: scambiare l'identità degli esperti rompe la continuità hidden/KV che la mask congelata preserva. Il costo non è coverage, è **discontinuità**. Coverage è epifenomenale |
| AN-2 | Sopravvivenza NON-monotona vs S1: survivor K91 S1 0.845 > collassato K23-rotate 0.811 | S1 assoluto (massa sui potati) è cronico ~0.75 e non-discriminante (CLAIM-011); solo lo **SLOPE/onset** porta segnale. Il trace logga solo i 6 selezionati → def-2 (full-256) non osservabile offline (identità S1=1−cov falsa ~4×) |
| AN-3 | Greedy non riproducibile temp0: locale diverge tok~75, pod cyberpunk bit-identico n=3 | Non-associatività FP/CUDA ribalta un argmax quasi-pari; **prompt-dependent margin** (cyberpunk high-margin→stabile, coffee low-margin→L1/L1/L3). Confonde ogni n=1 |
| AN-4 | **Doc-restart (doctype=2) universale in fase2** nonostante freeze a boundary sicuro | Bug di state-carry nell'handoff two-phase: la fase2 non vede il documento di fase1 come "committed" e ri-emette `<!doctype`. NON è collasso di mask |
| AN-5 | reap/full CI [0.972, 1.035] **attraversa 1.0** per il loop (CLAIM-017) | La ri-selezione live del loop aggiunge varianza che la mask statica (F1 pulito) non ha → non dire "near-lossless" secco per il loop |
| AN-6 | M1a n=3 NON riproduce l'n=1 "primo `</html>`": 0/6 emettono `</html>` | L'originale era un rollout fortunato in una distribuzione ad alta varianza; collapse-rate≈1.0 a wide-K23, la singola chiusura era rumore |

---

## 5. Buchi ciechi

1. **Il costo del rewind (`CORR_REWIND_TOK=56`) è ASSUNTO, mai misurato.** L'intero
   verdetto della tabella (low-K+rewind vince) ci poggia sopra: se il rewind costa
   ≥ MTTC, va in thrash a useful 0. I primitivi hard esistono (frontier
   snapshot/restore, `ds4_session_rewind`) ma il macro è non misurato.
2. **Tutte le ancore hazard sono 3-4 punti; wide-K91 DRIFT = una SINGOLA
   osservazione (n=1).** CI larghissimi (K91 1/lam 445-189411). Il modello nomina
   da sé le proprie gambe non identificabili.
3. **Zero dati two-phase in-engine sul 3060.** Ogni numero session-learning è pod
   (3090/3080Ti), mai sull'HW target — e la Definition-of-Done è 3060-specifica.
4. **Il sensore-larghezza non è validato su un task fuori distribuzione.** union_slope
   separa html vs coding ma sono entrambi "wide"; nessun trace warmup weighted per un
   task NARROW (coffee/json) ancora l'estremo basso dell'asse. knee(width) {20,32,48}
   sono 3 punti quasi-collineari.
5. **t/s (E-LAT) è solo local-3060, calibrazione singola; `miss_rate(K)` è un
   surrogato piecewise su 3 ancore (keep-8/23/32) con un bug resident-hit noto che
   gonfia il miss K23 a ~1.0.** La gamba throughput di good-tps è morbida.
6. **Il budget-confound non è mai stato separato dalla mask sul wide.** T1: full è
   L0 a 800 E 2000 sul cyberpunk (colpa budget/prompt). Manca un prompt wide dove il
   full sia L2+ allo STESSO budget in cui si testa la mask → non si può attribuire
   pulito il collasso wide alla mask oltre la firma-loop.
7. **Nessun set prompt avversariale/OOD:** la tesi wide poggia essenzialmente su
   1 famiglia (cyberpunk). "Un nuovo HTML + 2 code" (Fase 1) è solo parzialmente coperto.

---

## 6. Prossime 5 mosse (coerenti con P4 — identificare parametri, non enumerare)

1. **MISURA direttamente il costo-rewind (`CORR_REWIND_TOK`).** Micro-esperimento
   sul macro esistente frontier-snapshot/restore + `session_rewind`: strumenta la
   latenza di detection (S1_REWIND FIRE mediana ~40) + i token di rigenerazione. È
   il **singolo parametro** su cui ruota tutta la tabella: identificarlo prima di
   qualsiasi sweep di K. *(Identifica la costante più load-bearing dell'obiettivo.)*
2. **Fitta lam(K) a 2 ancore K (K23 vs K48) su UN prompt wide, n=3, ≥4000 tok.** NON
   è un'enumerazione di K: due punti per identificare l'ampiezza cov-gap `LAM_COV` e
   confermare/uccidere la forma esponenziale del knee. Il modello predice già K48
   static useful 0.50 vs K23 0.26 → il fit a 2 punti testa la **curvatura**, non il
   vincitore.
3. **Porta il two-phase W50 session-learn IN-ENGINE sul 3060 (L1)**, frozen low-K,
   no re-prefill. Chiude il buco cieco più grande (nessun dato session su HW target)
   e istanzia insieme il miglior separatore (provenienza) + la gamba throughput. Un
   solo delta vs SOTA_LOCAL_3060.
4. **Cabla lo S1-SLOPE come trigger di rewind/rotate (0020→0022), calibrando l'onset
   offline prima.** Risolve l'anomalia aperta AN-2/CLAIM-011 trasformando S1 da
   livello non-discriminante in una **soglia identificata in unità invarianti**
   (slope/tok, mai un assoluto). Unico segnale con 200-tok di preavviso.
5. **Aggiungi UN trace warmup weighted per un task NARROW (coffee/json)** per ancorare
   l'estremo basso dell'asse larghezza → rende union_slope un sensore identificato a
   2 lati e valida knee(width) oltre 3 punti collineari. Offline, economico: promuove
   la mappa width→knee da enumerazione a retta fittata.

> Tutte e 5 sono mosse di **identificazione di parametro** (costo-rewind;
> curvatura LAM_COV/knee; legge di velocità in-engine; soglia S1-slope; ancora bassa
> dell'asse larghezza), non "prova altri valori di K/W". Questo è il senso operativo
> di P4.

---

*Fonti primarie:* `scripts/decision_model.py` +
`runs/ds4/20260711_emodel_decision/decision_model.json`;
`docs/DS4_EXPERIMENT_LEDGER_20260710.md`; `docs/CLAIMS_CURRENT.md`;
`docs/SOTA_ROADMAP.md` (P1-P3); `docs/NEXT_STEPS_PLAN_20260710.md` (Fasi/Leve);
`docs/S1_REWIND_DESIGN.md`.
