# SOTA ROADMAP — reap-loop su hardware consumer (2026-07-10)

**Obiettivo:** SOTA sulla config utente (RTX 3060 12GB, WSL) **con leggi di
controllo e metriche portabili su qualsiasi hardware**. Il target headline resta
`>= 3.0 t/s stabili sul 3060 con qualità HTML/coding L2+ al render`
(`docs/NEXT_STEPS_PLAN_20260710.md` §Definition of done); nessun numero pod è mai
headline (`docs/HANDOFF_CODEX.md` §REGOLE 5).

> **Ground truth vincolante.** Questo documento è una roadmap, NON una fonte di
> verità sui claim: in caso di conflitto vince `docs/CLAIMS_CURRENT.md` (con la sua
> regola anti-regressione). Ogni numero qui sotto porta la sua fonte tra parentesi
> (commit `git`, path artifact, o nota di ledger).

---

## Principi (vincolanti per ogni test)

**P1 — MISURA.** n=3 per-seed + mediana per ogni verdetto; ordine alternato ABAB
(runner `--runs 3 --order abab`, commit `d0ad967`); grading funzionale L0-L3 al
render (`scripts/functional_grade.py`, portato con `0d1d269`), **mai** il
repeat_flag come proxy di qualità (il 97% dei run repeat=0 è comunque L0-L1,
`runs/ds4/20260710_retro_grade_l0l3/REPORT.md`: HTML L0=87, L1=1, L2=1 su n=89).
Controllo positivo full per ogni prompt nuovo (T1, sotto). Trace off nei bench;
manifest per ogni run; server spenti a fine; **un solo delta vs SOTA corrente**
(`docs/HANDOFF_CODEX.md` §REGOLE 3/5).

**P2 — PORTABILITÀ.** Separare rigidamente due classi di metriche:

| INVARIANTI (leggi di controllo usano SOLO queste) | HW-DIPENDENTI (output, mai input del controller) |
|---|---|
| L-level al render (L0-L3) | t/s (`avg_tps`, `first50_tps`) |
| collapse-rate per rollout | `prompt_s` / prefill wall-clock |
| curva coverage(K) per sessione | GiB paginati / slot cache fisici |
| dinamica S1 (livello all'engage, slope, onset) | TTFT assoluto |
| hit-rate cache (LRU sim) | working-set in GiB |
| working-set in #expert | |

Le **leggi di controllo usano SOLO invarianti + costanti derivate da
AUTO-CALIBRAZIONE al boot** (sotto). I t/s sono un output della roadmap, mai un
ingresso del controller. Nessuna costante assoluta cablata.

**P3 — ROLLOUT-AWARENESS.** Il greedy NON è riproducibile run-to-run: divergenza
misurata a tok~75 a config/prompt identici, temp 0
(`runs/ds4/20260710_w50_rotate32_k23_cache256_html4000/ANALYSIS.md`;
`CLAIMS_CURRENT.md` riga "Greedy NON riproducibile", **CLOSED**). Sul pod (3090)
è prompt-dependent (cyberpunk bit-identico n=3, coffee diverge a ~tok100 con grade
L1/L1/L3, `runs/ds4/20260710_pod_t1_full_positive_control/README.md`). Quindi ogni
claim è una **distribuzione**, non un punto; **collapse-rate** = frazione di
rollout che degenerano non-recuperati, è la metrica primaria (non gli esiti
singoli).

**P4 — DESIGN OVER TRIAL.** Gli esperimenti **identificano parametri del decision
model**, non enumerano config. Prima di ogni run: quale parametro non-identificato
del modello (width→K*, hazard, obiettivo) misura? Se nessuno, non si lancia. Il
modello vive in `docs/DECISION_MODEL.md` + `scripts/decision_model.py`
(`runs/ds4/20260711_emodel_decision/REPORT.md`): sensori invarianti, formule,
tabella decisioni, e la **lista minima** degli esperimenti di identificazione
(oggi: traccia routing su task stretti, efficacia rewind su K12 largo, tps(K<23)
locale). Le config si **calcolano** dal modello, non si provano a tentativi.

---

## Auto-calibrazione (il cuore "qualsiasi hardware") — `[FATTO — script + design + 3060 profile]`

Tool **"boot-probe"** COSTRUITO: `scripts/boot_probe.py` + `docs/BOOT_PROBE_DESIGN.md`
+ primo profilo di riferimento `runs/ds4/20260711_bootprobe/{profile.json,REPORT.md}`
(3060 misurato, gate P2 (a)-(b)-(c) soddisfatto — probe `--no-ds4` in 24.4 s).
Consumato da PACE via il contratto di lancio `DS4_PACE_AUTO` (`--emit-launch`
mappa il profilo su `DS4_PACE_KEEP`/`DS4_PACE_WRAP`/`--ssd-streaming-cache-experts`;
nessuna patch engine richiesta oggi, reader in-engine riservato patch 0029). Probe
~60-90 s al primo avvio su un HW nuovo:

| Sonda | Cosa misura | Cosa decide |
|---|---|---|
| (a) VRAM libera **+ footprint per-esperto** | GiB liberi post-modello **e** footprint per-esperto misurato (params/esperto × bytes/dtype; oggi 6.75 MiB/expert, riga 140) | `slot = floor(GiB_liberi / footprint_per_esperto)`, **dtype-aware**. 407/258/512 sono punti-dati, non la costante: lo slot-count è sempre derivato dal footprint misurato, mai pinnato (oggi manuale: reserve=1 riabilita la cache, `CLAIMS_CURRENT.md` "Bug reserve cache esperti") |
| (b) banda SSD→RAM→VRAM | throughput reale del path offload **+** domanda-byte-per-token del decode alla mask attiva | regime WRAP/prefetch via **confine ADIMENSIONALE**: `deeply-SSD-bound ⇔ banda_offload_misurata / domanda_decode < 1` (soglia su un **rapporto puro**, entrambi misurabili al boot) — **MAI un MB/s assoluto cablato** (un 3080-12GB / 3090 / NVMe più veloce non deve cadere sul lato sbagliato del confine). SPEX aiuta SOLO se deeply-SSD-bound (`CLAIMS_CURRENT.md` "PREFETCH SPEX-dense") |
| (c) t/s full su ~32 tok | baseline di velocità del nodo | fattore di scala **AUTO-NORMALIZZATO** (relativo alla baseline della probe dello STESSO nodo, es. first50/full di questo HW), **mai** contro un t/s di riferimento assoluto del 3060 |

**(d) coverage-target — NON è una sonda al boot.** È una calibrazione **offline
E-CAL (pre-boot)**, non una misura al boot; il gate P2 (esistenza della
boot-probe) si valuta su (a)-(b)-(c), non su (d). Dopo il verdetto E-CAL
**NEGATIVO** (§S0) la coverage non separa il collasso: la soglia coverage resta un
**invariante P2** (soglia fissa = portabile) usata SOLO come **pavimento cov90
anti-under-provisioning**; K-da-coverage deriva il pavimento dalla soglia-invariante
e **non introduce mai un K assoluto**.

**Output:** un file di profilo HW consumato da PACE (`DS4_PACE_AUTO=1`, env da
creare). Nessuna costante assoluta nel controller: tutte le soglie derivano dalla
probe o sono invarianti P2. **t/s è letto UNA sola volta al boot** (sonda c) per
coniare una costante di calibrazione auto-normalizzata; **non viene mai
ri-consumato come ingresso del controller a runtime** — così P2 ("t/s mai un
ingresso del controller") e la sonda (c) restano coerenti, senza buco logico.

---

## Definizione di SOTA (Definition of Done, due livelli)

**SOTA-3060** (`docs/NEXT_STEPS_PLAN_20260710.md` §Definition of done):
- `avg_tps >= 3.0` streaming su task HTML largo a 2000-4000 tok;
- **L2+ MEDIANO** su 3 prompt (cyberpunk / coffee / dashboard) a n=3;
- collassi non-recuperati = 0 (recupero via rewind/stopper ammesso se l'esito
  finale è L2+);
- numeri con manifest completo, trace off.

*Baseline SOTA_LOCAL_3060 corrente* **(3060-specific, NON-portabile — costanti da
NON trasferire in S5):** static K23 3.03-3.39 t/s (degenera ~tok116) /
rotate32 2.61-3.03 t/s; **entrambi L0 funzionale a <=800 tok sul cyberpunk** — ma
per budget-confound, NON per la mask (vedi T1). Costo rotate 0.36-0.45 t/s
(`CLAIMS_CURRENT.md` riga STABILITA, **OPEN**).

**SOTA-PORTABILE:** stessa logica, **zero ritarature manuali**, su un secondo HW
(pod 3080-12GB ~ spec utente, e 3090-24GB) → qualità INVARIATA, velocità che scala
con la banda misurata dalla boot-probe. La "config finale" trasferita deve usare
SOLO equivalenti *derivati* — K/pavimento-da-coverage (0024), cache-slot da
boot-probe (a), regime da boot-probe (b): le costanti fisse 3060-tuned (static K23
/ rotate32 / cache256 / cache 407-slot / reserve=1) sono **non-trasferibili e non
ammesse** in trasferimento. Test di trasferibilità esplicito nello Stage 5.

---

## Stage

Legenda: **[FATTO]** = artifact/commit verificabile in repo · **[IN CORSO]** =
lavoro aperto (mandato Codex o run non ancora committati) · **[TODO]** = non
iniziato.

### S0 — CONSOLIDAMENTO  `[IN CORSO]`
Chiusura del mandato Codex M1 + i pre-requisiti di metodo.

| Sotto-task | Stato | Evidenza |
|---|---|---|
| **M1a** — replica n=3 W50 vs W100 a ctx8192/html4000 | `[IN CORSO]` | run **non committati** `runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/` (summary_median.csv): W50 `avg_tps`=2.63 / W100=2.53; **entrambi L0 mediano** (l0l3_median=0). **0/6 rollout emettono `</html>`** → la replica NON riproduce il segnale n=1 "primo `</html>` del corpus". Da committare con REPORT prima di toccare CLAIMS. |
| **M1b** — stopper anti-ripetizione (airbag) + retry | `[IN CORSO]` | run **non committati** `runs/ds4/20260710_m1b_w50_stopguard_ctx8192_n3/`: stopper `client_stop_repeat_token_ngram` fira su 2/3 rollout; grade per-seed L2/L0/L1 (mediana L1, best L2 raggiunto SENZA retry). Numero patch riservato **0019** (`patches/README.md`). Helper `scripts/analyze_m1_html_runs.py` (non committato). |
| **M1c** — correzione artefatto prefill nel ledger | `[IN CORSO — aperta]` | i 266s di `prompt_s` del W100 ctx8192 erano **prefill a cache fredda** (ordine di esecuzione), NON warmup: il W50 prefilla lo stesso prompt in 57s e il warmup agisce DOPO `prompt done`. Correggere la nota nel ledger; d'ora in poi misure di velocità solo a stato cache appaiato (`docs/HANDOFF_CODEX.md` §M1c). Interseca il filone prefill: hygiene-fix del ledger qui, perf-gap J12 in S4. |
| **M1d** (opz.) — colonna `level` (L0-L3) nel master ledger | `[TODO]` | sostituisce il readout repeat-based nelle righe nuove (`docs/HANDOFF_CODEX.md` §M1d). |
| **E-DET** — latenza detector (fire a erosione lenta) | `[TODO]` | target di S0; oggi n-gram è sensore in ritardo (nota J47/J48, integrata in `PACE_DESIGN.md` §4 con `e78cd3d`). **Vincolo scoperto oggi (`runs/ds4/20260710_scope_divergence_pod/README.md`, sez. "Contrast with static K91"+"Takeaways"):** il sensore S1-slope compra lead SOLO nel regime **slow-erosion** (K91 static ~190 tok di anticipo), **zero lead nel regime aggressivo rotate** (S1 pinnato piatto ~0.815, collasso istantaneo ~gen126) → E-DET deve **cambiare sensore** o **restringere esplicitamente lo scope al regime lento**. La finestra-fire non è un magic-number fisso (era «<20 tok»): va derivata dallo slope di erosione / dinamica S1 (invarianti P2), così la latenza-target si adatta alla velocità di deriva. |
| **E-CAL** — soglia coverage per sizing predittivo | `[FATTO offline — NEGATIVO]` | commit **`bf3573c`**: `runs/ds4/20260710_ecal_coverage_threshold/` (`stats.json`+`coverage_by_trace.csv`+`REPORT.md`) + `scripts/calibrate_coverage_threshold.py`. **Verdetto: la coverage NON separa sopravvivenza/collasso** ("COVERAGE-AT-ENGAGE DOES NOT SEPARATE"). cov@23 ≈79% **task-invariante** (range 73-82%) su html **e tutti gli 11 prompt coding**; Kmin-cov90 ≈38 per ogni task → una regola coverage sceglie ~lo stesso K a prescindere dal task, **non discrimina la larghezza**. I separatori reali sono actuation-mode (static>>rotate), provenance (session>>cold) e token-budget, **NON la curva coverage**; controesempio a coverage identica (0.79): **keep-23 STATIC sopravvive (L3) vs keep-23 ROTATE collassa a ~gen126**. Unico uso residuo = **anti-under-provisioning**: cov90-sizing alza il fatale K23 fisso a ~38 uniforme (non testato, da A/B su pod). Questo RISOLVE **in negativo** la precondizione del gate 0024 (S2). |
| **Riconciliazione metodo mask (T5)** — weighted offline vs unit in-engine | `[IN CORSO]` | harness pronto `scripts/build_session_mask_canonical.py` (commit `561552d`); manca il run. Decide COSA calcola il relearn. |
| **Canonizzazione serie pace** | `[TODO] — bloccante` | `patches/README.md` §"Stato apply": `0015/0016-pace`+`0018` NON applicano su base pulita `0001-0014e` (dipendono da campi struct `prefill_apply`/`prefill_wait_wrap`/`prebreath_*` che vivono solo nel live-tree non committato). ⇒ rotate32 non è buildabile dalla serie canonica sul pod, e 0020/0021 restano ancorate al live-tree (md5 `771a39a8`), da ri-basare dopo la canonizzazione. |

**Gate S0:** trigger scelto (E-DET), metodo mask scelto (T5), serie patch
applicabile da base pulita (canonizzazione pace).

### S1 — FONDAMENTA  `[LARGAMENTE FATTO oggi]`

| Sotto-task | Stato | Evidenza |
|---|---|---|
| Grader L0-L3 nel runner | `[FATTO]` | `0d1d269` (port `functional_grade.py`), `d0ad967` (colonne n-runs/abab/graded). |
| n=3 / ABAB di default | `[FATTO]` | runner `--runs 3 --order abab` (`d0ad967`); usato in M1a/M1b. |
| T1 controllo positivo full | `[FATTO]` | `d7410d6`, `runs/ds4/20260710_pod_t1_full_positive_control/README.md`: full no_pace MAI degenera (13/13 repeat=0); cyberpunk L0 a 800 **E** 2000 tok per BUDGET, pagina L2 a 3498 tok. **Budget-confound dimostrato**: i test qualità sul cyberpunk richiedono ~4000 tok o il prompt compatto; il 2-bit non è indiziato. |
| Retro-grade archivio | `[FATTO]` | `8f3302a`, `runs/ds4/20260710_retro_grade_l0l3/REPORT.md` (105 output; HTML L0=87/L1=1/L2=1; `</html>` in 0/89). |
| Colonne invarianti standard nel summary (collapse_rate, coverage@engage, S1@engage/onset) | `[TODO]` | il summary attuale non le espone come colonne dedicate. |
| Controllo positivo full per `html_dashboard` | `[TODO]` | mai misurato full (T1 copre solo cyberpunk + coffee compatto). |

### S2 — MASK GIUSTA (sizing + costruzione)  `[TODO]` (harness pronto)

| Sotto-task | Stato | Evidenza |
|---|---|---|
| **T4** — W-sweep freeze-safe (W=30..150, freeze su boundary sicuri `}`/`;`) | `[TODO]`, harness `[FATTO]` | `scripts/run_w_sweep_freeze_safe.py` + `scripts/freeze_boundary.py` (+test), commit `561552d`, runbook `docs/T4_T5_RUNBOOK.md`. Domanda: la scala W diventa monotona? qual è il pavimento W vero? Riabilita/uccide la tabella W (oggi lotteria del punto di taglio, knife-edge freeze, `CLAIMS_CURRENT.md` CLAIM SESSION-LEARNING **OPEN** + nota J44). |
| **0024** — coverage-sized descent (K-da-coverage, riusa `rmass` di 0020) | `[RI-SCOPATO dopo E-CAL negativa]` | **NON più condizionato a E-CAL positiva** (S0): la premessa "K-da-coverage separa la sopravvivenza" è **morta** (E-CAL negativa). Resta SOLO la variante **anti-under-provisioning** — cov90-sizing come **pavimento K** (K23→~38 uniforme) per non sotto-dimensionare, non come predittore di collasso. Numero patch **0024** da registrare in `patches/README.md`. Domanda residua: il pavimento K-da-cov90 (~38) regge >=2000 tok meglio di K23 fisso? (A/B n=3; qualità su pod economico da R2-cache, velocità solo locale). |

**Gate S2:** una config mask che porta il task largo a **L2+ mediano** senza
interventi runtime.

### S3 — CONTROLLO RUNTIME (per la deriva residua)  `[IN CORSO]`

| Sotto-task | Stato | Evidenza |
|---|---|---|
| **0020** — S1-slope trigger (leva L2) | `[FATTO] authoring + smoke` | `58fbe98` (author) + `57a0caa` (smoke PASS), `runs/ds4/20260710_pod_smoke_0020_0021/README.md` (**solo smoke di meccanismo a trigger FORZATO** — SLOPE_WIN e soglia abbassati apposta —, NON stabilisce il regime-split): 8× `s1_trigger` → 8× `rotate(s1)`, slope non-NaN, S1 sale 0.727→0.814 (banda CLAIM-011). **REGIME SPLIT — fonte primaria `runs/ds4/20260710_scope_divergence_pod/README.md`** (sez. "Contrast with static K91"+"Takeaways"): il rotate aggressivo pinna S1 ~0.815 **piatto** e collassa a ~gen126 **senza lead**; K91 static deriva 0.845→0.895 con **onset-slope ~190 tok di anticipo** → il trigger vale per **mask larghe / slow-drift**, non per l'aggressivo. Pending canonization (S0). |
| **0021** — delta-prefetch default-on (leva L3) | `[FATTO] authoring + smoke` | stessi commit; smoke: 15× `rotate_delta`, `entered==exited` (631,268,227,… converge), 6.75 MiB/expert, **niente WRAP full su decode**. Riduce il rotate da 75-699 GiB a delta di pochi GiB (nell'ordine 0.6-4.2 GiB). Si applica DOPO 0020. |
| **M1b stopper** (airbag) | `[IN CORSO]` | vedi S0/M1b; è baseline e benchmark dell'A/B, non la cura. |
| **0022** — rewind S1-guided (correzione) DOPO gate R1 | `[FATTO] design; patch TODO` | design study `6b03787`, `docs/S1_REWIND_DESIGN.md`. Feasibility engine = **CLEAR/low-risk**: i primitivi hard esistono e girano ogni token sotto MTP (`spec_frontier_snapshot`/`restore` ds4.c:26766/26800, `ds4_session_rewind` ds4.c:30560). Posizione nella scala = CORREZIONE, dopo prevenzione (0020) e **prima** dello stopper come strategia primaria. Blocco R1: smoke di bit-equality del rewind (§5 del design). Patch 0022 scritta solo dopo verdetto M1b + canonizzazione. |
| **0025** — phase-gated relearn / piecewise-static mask (prevenzione strutturale) | `[EVIDENZA OFFLINE] design candidato; patch TODO` | E-PHASE `runs/ds4/20260710_ephase_structural_masks/REPORT.md` (verdetto POSITIVO): la mask warmup-frozen affama le fasi strutturali successive (post-warmup cov 0.564 vs 0.740 locale @K23 = **+17.6pt** di guadagno mask-per-fase, **+34.9pt** sulle fasi fuori-warmup; sess_W50 reale su W130: head 0.803 → CSS **0.498**); struttura > tempo cieco (Jaccard within 0.447 vs cross 0.289; dom-overlap terzile 0.544 replica E1 53.6%); churn al confine **12.9/23 expert/layer ≈ 3.4 GiB** una tantum, coperto da 0021. Trigger = marcatori strutturali (`</style>`, `<body>`, `<script>`, fence) via ring n-gram PACE; mini-warmup 30-50 tok → relearn pesato → freeze static (mai rotate). Patch solo dopo A/B pod (gap JS da chiudere con un trace full-document). |
| **0026** — demand-driven admission (scambio mirato a K costante, leva complementare a 0025/0022) | `[EVIDENZA OFFLINE] candidata; patch TODO` | E-ADMIT `runs/ds4/20260710_eadmit_demand_admission/REPORT.md` (verdetto POSITIVO; sim su traiettoria sana = **potenziale di copertura, NON qualita'** — quella la decide l'A/B live): CUSUM per-expert sulla domanda bloccata fuori-mask → ammissione con sfratto del keep a EWMA minima, **K costante, mai K0, mai re-rank wholesale**. Recupera **+13.7pt** di copertura tardiva persa dalla frozen (0.485→0.622; rotate32 0.662) = **77% del gap A→B** con churn **5.4× sotto** il rotate bocciato (130 vs 706 scambi/100tok; 0.86 vs 4.7 GiB/100tok a 6.75 MiB/exp) e rimbalzi ~zero (0.3% vs 51.7%). Parametri CUSUM stabili tra 11 trace (h=1.2, k_d=0.02, p=2; regret mediano 3.4pt). Copre la deriva GRADUALE senza confini strutturali, dove 0025 (fase-driven) non scatta; riusa rmass 0020 + delta-prefetch 0021, `DS4_PACE_ADMIT=0` default. Registrata in `patches/README.md`. Gate: A/B live S3 (FROZEN vs FROZEN+ADMIT, n≥3). |

**Gate S3:** collapse-rate non-recuperato ~0 su n>=5 rollout da 2000-4000 tok.

### S4 — VELOCITÀ LOCALE (solo dopo qualità assicurata)  `[TODO]`

| Sotto-task | Stato | Evidenza |
|---|---|---|
| Cache sizing da boot-probe | `[SBLOCCATO — boot-probe FATTO]` | dipendenza auto-calibrazione ora soddisfatta (`scripts/boot_probe.py`: `derived.cache.cache_slots`=394 sul 3060, dtype-aware). Resta da cablare lo slot derivato nel path di sizing S4. Effetto cache dominante: LRU sim cap258=0.34 → cap512=0.59 → cap1024=0.74 (nota J17 `EXPERIMENTS_LEDGER.md`). |
| Prefill / TTFT — prompt-derived prefetch SENZA mask apply | `[TODO]` | **gap J12** (`EXPERIMENTS_LEDGER.md`): unica cura nota del prefill 115-213 s (es. J6 114.994 s, J34 158.896 s). Serve separare il prefetch-dal-prompt dall'apply-della-mask. (Correlato: hygiene-fix del ledger M1c in S0 — il 266s W100 era prefill a cache fredda, non un vero costo prefill.) |
| SPEX hidden consumer | `[TODO] — condizionato` | SOLO se boot-probe (b) dice **deeply-SSD-bound** (rapporto adimensionale banda/domanda_decode < 1, **mai** un MB/s assoluto): converte recall→velocità solo lì (gate full baseline 0.24→0.77, 3.2x); sul caso pratico 3060 RALLENTA ~1.55x (`CLAIMS_CURRENT.md` "PREFETCH SPEX-dense", **OPEN**). |
| Promozione precisione session-learned | `[PARCHEGGIATA]` | E1 negativo per il pin statico (`5ea4c64`, `runs/ds4/20260710_e1_top_expert_mass/`): top-1 per-token ~30.5% ma identità NON stabile (cross-task overlap 2.5%, within-coding 7.8%) → pin statico cattura ~5.7% massa/layer (tetto live-sessione 16.7%) a costo 21.4% del cache 407-slot. Riprendere SOLO se S4 stalla, e solo nella variante **live per-sessione** (E2 pod A/B, ~$1-2). |
| Compressione differenziata asincrona (CQ1 sui cold) | `[MODELLATA — E-LAT]` | Probe offline E-LAT (`scripts/latency_tier_model.py`, `runs/ds4/20260710_elat_tier_latency/REPORT.md`): modello per-tier calibrato su run hit~1 e validato su 90 run (err. mediano 15.4%, steady 7.8%). Verdetto: **seconda leva S4** dopo fit-in-cache/hit-rate del tier hot, e suo abilitatore RAM. Steady 3060: status quo 3.12 t/s → tiering completo **3.7-4.1 t/s** (+20-32%); CQ1 async da solo non muove lo steady (path H2D-copy-bound, t_b 0.95 ms/expert × 258/tok) ma elimina i cliff SSD (breath 2.7 s, storm 60 s/tok). Richiamo singolo expert da SSD: 1.5-2.7 ms prefetched, **50-59 ms sincrono** (0.16-0.25 token), ~230 ms tail. CQ1 sync resta vietato nel hot path (67.6 ms/expert, J38). Capacità CQ1 misurata ×1.5-×2.1 (J35), NON ×3. Micro-bench mancanti nel REPORT §7. |

**Gate S4 = SOTA-3060 raggiunto.**

### S5 — TRASFERIBILITÀ  `[TODO]`
Replay della config finale su pod 3080-12GB e 3090-24GB via R2-cache (deploy in
minuti, `docs/POD_R2_CACHE.md`). Domande: qualità invariata? le leggi si
auto-adattano (cache/WRAP/coverage) senza toccare l'env?

**Vincolo di trasferibilità (obbligatorio).** La "config finale" replicata deve
usare SOLO gli equivalenti *derivati*: K/pavimento-da-coverage (0024, cov90),
cache-slot da boot-probe (a) sul footprint per-esperto misurato, regime da
boot-probe (b) col rapporto adimensionale. Le costanti fisse 3060-tuned — **static
K23 / rotate32 / cache256 / cache 407-slot / reserve=1** — sono **non-trasferibili
e VIETATE in S5**: portarle fisse invaliderebbe il test di trasferibilità.

**Gate S5 = SOTA-PORTABILE.**

### S6 — CONSOLIDAMENTO SCIENTIFICO  `[TODO]`
Ogni gate aggiorna `CLAIMS_CURRENT.md` (regola anti-regressione); paper §nuova
**"hardware-adaptive control laws"** (fonte canonica `docs/paper/PAPER.md`,
banner con `e78cd3d`); Scope come strumento di ispezione (divergence mode già
vivo, `7d8af92` + `runs/ds4/20260710_scope_divergence_pod/`; UX track in sessione
dedicata).

---

## Budget e logistica

- **Pod:** ~$10-15 totali stimati sull'intera roadmap. R2-cache abbatte i deploy
  (`docs/POD_R2_CACHE.md`, skip del download ~87 GB + build da sorgente). Tariffe:
  3080 ~$0.17-0.25/h; **3090 secure $0.46/h verificato** (T1, `runs/ds4/20260710_pod_t1_full_positive_control/README.md`).
- **Gate-check obbligatorio** prima di scaricare il modello: 3 pod community 3090
  hanno fallito il CUDA gate-check prima del secure che è passato al primo colpo
  (`/dev/nvidia-uvm` EIO, T1 README).
- **Locale:** coordinare con i mandati Codex via `docs/HANDOFF_CODEX.md` — un
  mandato per stage (M2=S2, M3=S3, …). Regole pod: gate-check, stop-non-terminate,
  mutazioni solo sui propri pod, credenziali mai stampate **e da ruotare**
  (R2 + RunPod/HF).

---

## Rischi aperti e mitigazioni

| Rischio | Evidenza | Mitigazione |
|---|---|---|
| Rumore 3060 ±50% | `docs/HANDOFF_CODEX.md` §REGOLE 2 | mediane n=3 (P1). |
| Knife-edge freeze | `CLAIMS_CURRENT.md` CLAIM SESSION-LEARNING **OPEN**, nota J44 | T4 lo chiude (freeze su boundary sicuri). |
| Rollout lottery | divergenza greedy tok~75 (P3) | collapse-rate come metrica primaria, non esiti singoli. |
| ctx8192 costa t/s locale | ~1 t/s osservato n=1 (nota locale) | misurare la curva ctx→t/s in S4 e scegliere il ctx minimo che completa il task. |
| Doppia fonte di verità del paper | `PAPER.md` canonico reap-loop, draft moe FROZEN (`e78cd3d`) | pointer unico; congelato moe. |
| Patch numbering | `patches/README.md` (collisioni 0011/0012/0015/0016 tra arene) | numeri nuovi da 0019: 0019 stopper Codex, 0020 s1-trigger, 0021 rotate-delta, 0022 rewind, 0023 diagnostica, 0024 coverage — registrare in `patches/README.md`. |
