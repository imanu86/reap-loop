# Intuition Archaeology — rilettura di TUTTE le sessioni Claude con la lente di oggi (2026-07-11)

**Scopo.** Riprendere ogni intuizione dell'utente emersa nelle sessioni Claude dei
track ds4 (REAP iniziale · TUTOR cold/warm · DSpark/MTP · K91/REAP-LOOP/PACE) e
**rileggerla con la lente di oggi**, classificandola in quattro cassetti:

- **(A) DA RIABILITARE** — bocciata dal *metodo dell'epoca*, non dal merito. Porta lo
  spec del **retrial minimo P4-style** (identifica un parametro, non enumera config).
- **(B) ORA FATTIBILE** — l'infrastruttura di oggi la rende *cheap* (R2, canonical
  build v2, rewind bit-exact provato, decision model, trace pesati già loggati, E-DET).
- **(C) CONFERMATA** — le scoperte di oggi le danno ragione. Si dà credito.
- **(D) RESTA CHIUSA** — con la ragione *fondamentale* (fisica/architettura), non
  contingente.

> Ground truth vincolante: `docs/CLAIMS_CURRENT.md`. Questa è archeologia di metodo,
> non una fonte di claim. Fonti-lente: `docs/DECISION_MODEL.md`,
> `docs/LEVER_RETROSPECTIVE_20260711.md`, `docs/S1_REWIND_DESIGN.md`,
> `docs/SOTA_ROADMAP.md` (P1-P4), `docs/CLAIMS_CURRENT.md`.

## La lente di oggi (2026-07-10/11), in una riga ciascuna

- **Legge dell'ancora** — ogni cambio di *membership* della mask in corsa sotto greedy
  destabilizza: staircase/rotate/admit tutti bocciati live (AN-1: costo = discontinuità
  hidden/KV, non coverage).
- **Regime-split S1** — S1 è *piatto* nel regime aggressivo (W50+K23+rotate, loop ~gen126,
  zero lead); dà lead **~190-214 tok solo in erosione lenta** (K91-family). Il negativo
  storico era regime-scoped.
- **Evaporazione della domanda sotto mask** — la mask congelata affama le fasi tardive
  (E-PHASE, −17.6pt); **K0 = unica finestra onesta** per leggere la domanda vera.
- **Budget-confound** — banco ≤800 tok INVALIDO sul wide (T1: full L0 a 800 E 2000 sul
  cyberpunk per budget, non per config; L2 solo a ~3500 tok).
- **Lotteria rollout** — greedy NON riproducibile run-to-run (divergenza tok~75); n=1 =
  un'estrazione; la metrica primaria è il **collapse-rate**, non l'esito singolo.
- **Larghezza leggibile per IDENTITÀ** — `union_slope` (nuovi expert/token) ha CV 0.49,
  ~15× la massa-coverage (CV 0.033); la larghezza si legge per identità, non per massa.
- **Hazard model + correzione ≫ prevenzione** — il collasso è un tasso (MTTC), non un
  verdetto; **K12+rewind = 3.99 good-tps** (+156% vs miglior static) in tutte le classi.
- **Rewind bit-exact provato fattibile** — i primitivi (`spec_frontier_snapshot/restore`,
  `ds4_session_rewind`) esistono e girano per-token sotto MTP; 0022 attuatore + 0027
  harness di bit-equality AUTHORED.
- **Costi per-tier E-LAT** — copy-bound; `t_ss(K)=74.9ms+258·miss(K)·0.952ms`; bug
  resident-hit=0 noto (gonfia miss K23).
- **Sostituzione-al-miss** — mai davvero provata (rotate era il cugino bocciato,
  mass-scored membership churn); retrial giustificato (archeologia già committata).
- **Breath a orologio in retrial** — clock_breath64 in coda di test; il decision model lo
  declassa (breath NO, rewind YES sull'obiettivo good-tok/s).

---

## RANKING per VALORE ATTESO — sezioni (A) + (B) combinate

| # | Cassetto | Intuizione (origine) | Perché ora | Costo |
|--:|---|---|---|---|
| **1** | A | **Rollback: "cancella un % e riparti da K0"** (D6a, K91) | Rewind bit-exact provato → K12+rewind **3.99** vs 1.56; è l'UNICO esperimento che conferma/refuta correzione≫prevenzione | 1 run pod/locale (rewind AUTHORED) |
| **2** | A | **"Prevedere la deriva PRIMA che sporchi"** (PACE, K91) | Regime-split S1: il negativo era solo aggressivo; in erosione lenta S1 dà ~200 tok di lead → è il *trigger* del rewind | offline (E-DET su trace K91 esistenti) |
| **3** | B | **"Il profilo generalizza al N+1-esimo"** (TUTOR, leave-one-out) | I trace pesati (selected-6/tok) sono GIÀ loggati (0020); coverage leave-one-out è una lettura offline gratis; ancora l'estremo NARROW dell'asse larghezza | offline ~0 |
| **4** | B | **Two-phase W50 session-learn + warmup-corto IN-ENGINE sul 3060** (K91) | Canonical build v2 sblocca; leva velocità+qualità più grande MAI misurata su HW target (fase2/fase1 ≈ 7×) | integrazione engine, media |
| **5** | B | **Layer densi come oracolo precoce del routing** (K91, relay) | `dense_oracle.py` scritto + trace hidden/routing pesati esistono → recall@23 misurabile offline | offline, media |
| **6** | A | **Fine-tuning per concentrare/rendere-predicibile il ROUTING** (TUTOR) | Asse mai misurato (solo neuron-concentration, Qwen, negativo); FT-router-only su proxy Qwen-30B; concentrazione abbassa il knee, predicibilità aiuta SPEX | training proxy, medio-alto |
| **7** | A | **Coverage-incompleta + speculative → batch dei miss** (TUTOR) | Il nucleo (rivelare la union dei miss in anticipo e batch-arli) mai isolato come esperimento; oggi SPEX è regime-dipendente → EV ridotto | media |
| **8** | B | **Prefetcher t/s REALE (non hit-rate simulato)** (K91, parcheggiata) | Canonical build lo rende misurabile end-to-end; ma SPEX RALLENTA il caso pratico 3060 → EV basso | pod/locale, basso EV |

**Le due a valore atteso più alto = #1 e #2** (la coppia rollback→rewind + il suo
trigger S1), coerenti con la Mossa 1/Combinazione A del `LEVER_RETROSPECTIVE`.

---

## (A) DA RIABILITARE — bocciate dal metodo, non dal merito

### A1 — Rollback: "reinizializzare cancellando un tot% della generazione, cercare lo sweet-spot, ripartire da K0" (D6a, track K91)
> *"possiamo reiniziare il loop cancellando un tot% della generazione? facciamo dei test su quale % è lo sweetspot di cancellamento per riprendere da K0?"*

**Verdetto dell'epoca:** diventò l'esperimento D6a, ma l'unico attuatore disponibile era
il **rollback-via-prompt** (ri-mostra "scrivi il documento"): 3/3 fail, e il re-prefill a
metà-CSS induce il document-restart (knife-edge J44). Conclusione registrata: *"il
contesto avvelenato vince → serve rewind KV"*. L'intuizione fu quindi **rimandata per
mancanza dello strumento**, non refutata nel merito.

**Rilettura con la lente di oggi:** è ESATTAMENTE la tesi centrale del decision model.
Il collasso è un **hazard** (wide-K23 MTTC ~1051), il greedy una distribuzione; un
**rewind economico (~56 tok) che fira ogni MTTC domina** il pagare throughput per alzare
K. `K12+rewind = 3.99 good-tok/s` (+156% vs miglior static) in tutte e tre le classi di
larghezza. E il rewind KV che allora "serviva e non c'era" **oggi è provato fattibile e
low-risk**: i primitivi esistono e girano per-token sotto MTP (`spec_frontier_snapshot/
restore` ds4.c:26766/26800, `ds4_session_rewind` ds4.c:30560); 0022 (attuatore) e 0027
(harness bit-equality) sono AUTHORED. Lo sweet-spot di cancellazione dell'utente = il
`DS4_PACE_REWIND_MARGIN` + il bound clean-rewind (~640 tok di slack del raw-ring).

**Retrial minimo (P4 — identifica `CORR_REWIND_TOK` + `useful_frac(K12,rewind)`):**
build patch 0022 → cyberpunk **K12 static + rewind da S1/n-gram**, ctx8192, ≥4000 tok,
n≥3 ABAB vs baseline L-STATIC/L-ROTATE. Misura: (1) `CORR_REWIND_TOK` reale (latenza
detection + token rigenerati), (2) `useful_frac`, (3) **bit-equality** del rewind (0027,
smoke bloccante prima di fidarsi di qualunque A/B). Metrica primaria = collapse-rate +
good-tok/s, non L singolo. **È l'unico esperimento che sblocca l'intera tabella del
decision model.**

### A2 — "Prevedere la deriva che attiva il respiro PRIMA che sporchi il codice" (PACE, track K91)
> *"Possiamo prevedere la deriva che attiva il respiro? prima che sporchi il codice?"*

**Verdetto dell'epoca:** NEGATIVO. Il routing non prediceva: S2-Jaccard "morto" (0% sopra
soglia durante il loop), S1 assoluto cronico ~0.75 non-discriminante, il detector n-gram
reattivo (~30-90 tok dopo l'inizio della deriva). Archiviata come "non si può anticipare".

**Rilettura con la lente di oggi:** il negativo era **regime-scoped, non di merito**. Il
**regime-split S1** dice: nel regime *aggressivo* (W50+K23+rotate) S1 è pinnato piatto
~0.815 e il loop parte subito → lì niente da prevedere (giustamente inerte). Ma nel regime
di **erosione lenta** (mask static larga, K91-family) lo **SLOPE** di S1 sale ~+0.058 su
~200 tok PRIMA del collasso — è l'unico riser pre-collasso misurato (CLAIM-011). E-DET
(EWMA-CUSUM, `scripts/tune_s1_detector.py`) lo formalizza: ARM delay ~31 tok, FIRE ~40 tok,
**~214 tok di lead** prima del text-lock K91, 0/1k falsi allarmi sul controllo aggressivo.
L'intuizione dell'utente era giusta *nel regime dove il segnale esiste* — l'epoca la testò
nel regime sbagliato.

**Retrial minimo (P4 — identifica la soglia S1-slope in unità invarianti):** replay
**offline** dei due trace S1 per-layer già registrati (collasso static-K91 onset ~pos2286
/ lock 2476; pod-r1 aggressivo come controllo avversariale) → cabla ARM/FIRE come *trigger
del rewind* di A1 (patch 0020→0022). Nessun run nuovo per calibrare: la soglia si fitta sul
disco. È la Combinazione D del `LEVER_RETROSPECTIVE` (unico segnale con ~200 tok di
preavviso). **A2 è il trigger che A1 consuma** → vanno insieme.

### A3 — Fine-tuning per concentrare / rendere-predicibile il ROUTING (TUTOR)
> *"C'è anche la leva del fine tuning che non abbiamo tirato su ds4 ma su qwen … Spex più preciso"*

**Verdetto dell'epoca:** semi-chiusa. Il tutor corresse: il FT era stato provato su Qwen e
dava risultato *piatto* — ma su un **asse diverso** (neuron-concentration DENTRO gli
expert, n_eff 0.885→0.890). La domanda vera dell'utente — il FT **concentra o rende
predicibile il ROUTING** (working-set più piccolo / router più localizzato) — restò
**mai-raccolta**.

**Rilettura con la lente di oggi:** l'asse-routing è precisamente ciò che il decision model
nomina come load-bearing: `knee(width)` e `DRIFT`. Un router che **concentra** abbassa il
knee (working-set più stretto, meno miss E-LAT); un router più **predicibile** alza il
recall SPEX. Furono misurati i neuroni, mai il routing. Metodo sbagliato → merito intatto.

**Retrial minimo (P4 — identifica Δknee e Δrecall da FT-router-only):** FT del **solo
router** su proxy Qwen-30B (economico), poi misura offline su trace: (1) `union_slope` e
knee pre/post-FT (concentrazione), (2) recall@K del predittore denso/n-gram pre/post
(predicibilità). Due numeri, nessun modello da 158B da ri-quantizzare. EV medio, costo
medio-alto (training proxy) → sotto la coppia rewind.

### A4 — Coverage-incompleta + speculative decoding → batch dei miss (TUTOR)
> *"Se comincio a produrre token con coverage incompleta sfruttando anche dspark … Ho ancora un risultato accettabile o perdo troppa qualità?"*

**Verdetto dell'epoca:** il timore-qualità era un equivoco (miss=latenza, mai errore;
spec-dec lossless per costruzione), ma il **nucleo buono** — un blocco di token draftati
rivela la *union* dei miss in anticipo e tutti insieme, batch-abili invece che pagati in
serie — non fu mai isolato come esperimento a sé.

**Rilettura con la lente di oggi:** parzialmente eroso. SPEX-dense è **regime-dipendente**
(CLAIM-014): converte recall→velocità solo se DEEPLY-SSD-bound; sul caso pratico 3060
(mask keep-23 + cache 5GB) RALLENTA ~1.55×. Il batch-dei-miss resta teoricamente valido ma
solo nel regime cold/no-mask, che non è l'operating point. **Riabilitabile solo come
sotto-caso di A5/oracolo denso**, non standalone → EV basso.

**Retrial minimo:** offline sui trace — misura la *union-size* dei miss per blocco di K
token draftati vs seriale, a diversi K di draft; procedi al runtime SOLO se DEEPLY-SSD-bound
(gate baseline <0.5 t/s). Altrimenti resta chiusa col motivo di CLAIM-014.

---

## (B) ORA FATTIBILE — l'infrastruttura di oggi le rende cheap

### B1 — "Il profilo del router generalizza al (N+1)-esimo esempio" → leave-one-out coverage (TUTOR)
> *"il 21 esimo essendo simile dovrebbe lavorare su un set già caldo"*

**Verdetto dell'epoca:** **mai-raccolta**. Il tutor la trasformò in un disegno (profilo =
union di 19 trace, coverage misurata sul 20°, pesata per load) ma non c'era il logging del
routing pesato per eseguirlo.

**Cheap oggi:** i trace pesati (**selected-6/tok + pesi**) sono GIÀ loggati da 0020, usati
dal sensore di larghezza. Il leave-one-out è una **lettura offline gratis** su archivio.
E chiude un buco cieco esplicito del `LEVER_RETROSPECTIVE` (§5.4 / Mossa 5): manca un trace
warmup weighted per un task **NARROW** (coffee/JSON) per ancorare l'estremo basso dell'asse
larghezza — `knee(width) {20,32,48}` sono 3 punti quasi-collineari. La stessa lettura
serve due scopi (valida il profilo + ancora l'asse) a costo ~0.

**Retrial minimo (P4):** su un trace NARROW pesato (coffee/JSON, ≥150 tok) misura coverage
leave-one-out (union di N−1 → coverage del N-esimo, pesata per load) e `union_slope`.
Offline. Promuove la mappa width→knee da enumerazione a **retta fittata a 2 lati**.

### B2 — Layer densi iniziali come oracolo precoce del routing (K91, relay attribuito all'utente)
> *"front-end DENSO come oracolo precoce del routing … usa le loro attivazioni per PREDIRE quali esperti instraderanno i layer MoE"*

**Verdetto dell'epoca:** testata-in-corso, marcata `[NON TESTATO]` con limiti onesti (denso
predice bene i primi layer MoE, peggio i profondi); `dense_oracle.py` (ridge+PCA) scritto,
collezione dati avviata a fine log.

**Cheap oggi:** il trace hidden binario (0007) + routing (0006) esistono, i trace **pesati**
pure → recall@23 (denso vs n-gram vs chance) è **misurabile offline** senza pod. Interseca
A2 (segnale precoce) e la gamba predicibilità di A3.

**Retrial minimo (P4 — identifica recall@K del segnale denso):** completa il training
`dense_oracle.py` sui trace esistenti; riporta recall@23 per profondità di layer MoE vs
baseline. Se il denso batte n-gram sui primi ~10 layer MoE → è un secondo trigger candidato
per il ladder di A2. Offline.

### B3 — Two-phase W50 session-learn + warmup-corto IN-ENGINE sul 3060 (K91)
> *"ma hai fatto test con token di warmup 30-40 a K0 e poi scendere a K91 subito dopo?"* + mask imparata dalla sessione

**Verdetto dell'epoca:** confermata come regime utile (fase2 14.6 vs fase1 2.03 t/s ≈ 7×;
session ≫ cold), MA ogni numero session-learning è **pod** (3090/3080Ti), MAI sull'HW target
3060 — e la Definition-of-Done è 3060-specifica. Buco cieco #3 del `LEVER_RETROSPECTIVE`.

**Cheap oggi:** la **canonical build v2** (md5 62ed2e71, binari su R2, micro-smoke GPU OK)
sblocca lo switchover; le due gambe (session≫cold, low-K throughput) sono confermate
indipendentemente → manca **solo** l'integrazione in-engine two-phase (warmup wide → freeze
low-K, no re-prefill) sull'HW target. È la Combinazione B (2° valore atteso del retrospective).

**Retrial minimo (P4):** porta il two-phase W50 in-engine sul 3060, frozen low-K, no
re-prefill; un solo delta vs SOTA_LOCAL_3060. Attenzione al knife-edge J44 (freeze su
boundary sicuro). Costo: integrazione engine, media.

### B4 — Prefetcher: t/s REALE invece dell'hit-rate simulato (K91, parcheggiata dall'utente)
> *"il mio risultato è hit-rate simulato — il t/s vero servirebbe una patch ds4"*

**Verdetto dell'epoca:** esplicitamente **parcheggiata** dall'utente (patch ds4 non scritta).

**Cheap oggi:** la patch/canonical esistono → il t/s end-to-end del prefetcher è misurabile.
MA la lente CLAIM-014 dice già l'esito atteso: sul caso pratico 3060 il prefetch RALLENTA
(~1.55×) perché il working-set è piccolo/cache-ato; aiuta SOLO se DEEPLY-SSD-bound. **EV
basso** → si misura solo se si rientra nel regime SSD-bound (gate full/no-mask 0.24 t/s).

---

## (C) CONFERMATA — le scoperte di oggi le danno ragione

| Intuizione (origine) | Cosa è diventata / conferma di oggi |
|---|---|
| **Termometro/temperatura di coerenza per giustificare la rotazione** (K91) — *"una temperatura che misuri se quello che sto facendo è coerente"* | È nato il sensore **S1** (gate-mass sui potati), citato come "il termometro dell'utente". Oggi il **regime-split** gli dà ragione con un numero: **~190-214 tok di lead** nel regime di erosione lenta (E-DET). L'utente ha inventato il sensore. |
| **Asimmetria scendere ≠ risalire la scala K** (K91) — *"se passo da k0 a k35 e via via scendo è la stessa cosa che passare al contrario?"* | Fortemente asimmetrico, come intuito → oggi è la **legge dell'ancora**: ogni cambio di membership in corsa sotto greedy destabilizza (AN-1: il costo è discontinuità hidden/KV, non coverage; C1: il contesto avvelenato vince). |
| **"Il risultato è di qualità o ha degenerato dopo?"** (scetticismo sul W=40 "pulito", K91) | Diventato **P3/P4**: budget-confound (≤800 tok invalido) + **lotteria rollout** (n=1 sospetto, collapse-rate come metrica primaria). Lo scetticismo era metodologicamente esatto. |
| **"Loop recupera da solo dopo il collasso?"** (K91) | No, confermato: serve **correzione (rewind)**, non auto-recovery — la mask K91 era già la selezione ottima per il coding, ri-selezionare torna alla stessa mask. Fonda correzione≫prevenzione. |
| **Non fissare K, fissare la COPERTURA di massa-gate** (K91) | Diventato adaptive-K coverage (CLOSED come manopola task-indipendente). Nuance di oggi: E-CAL **negativo** → la coverage NON separa il collasso, resta solo **pavimento anti-under-provisioning** (cov90 ~K38). Confermata come *manopola*, declassata come *predittore*. |
| **Copia del modello su R2/Cloudflare, deploy senza riscaricare 86GB** (K91) | Infra core: `docs/POD_R2_CACHE.md`, egress gratis, rclone multi-thread. Confermata e adottata. |
| **Curva di token-scritti per gradino K** (K91) | Costruita: il grosso del lavoro (82%) al K più stretto e veloce. Confermata; alimenta E-LAT. |
| **"Perché ai warmup W alti fa peggio?" / "perché scatta il restart?"** (K91) | Confermato meccanismo: **lotteria del freeze-point** (L-WSWEEP), l'esito flippa sul carattere di taglio (`}`/`;`/`>`); il taglio a metà dichiarazione CSS induce il doppio `<!doctype` (J44/AN-4). Non-monotono in W, come sospettato. |
| **K come manopola CONTINUA appena sotto il budget RAM** (REAP iniziale, K56/64/67) | Confermato come asse operativo: il decision model opera su K12-K48 per classe di larghezza, non su checkpoint fissi. |
| **"Pod più potente inquina il risultato IO-bound"** (DSpark) | Formalizzato in **P2**: invarianti (hazard/token, ordinamento) vs HW-dipendenti (t/s, mai input del controller). Un GPU enorme cancella il regime IO-bound oggetto della misura. Esattamente l'obiezione dell'utente. |
| **10 t/s in BURST aggregato + prompt caching + batched decode** (TUTOR) | Confermati come economia del batch (union sub-lineare degli expert vs token lineari); prefill una volta sola sul template condiviso. |
| **Il modello potato dà benefici GIÀ OGGI (prima dei 64GB)** (REAP iniziale) | Confermato: l'intero track 3060 (12GB reali) è costruito su questa scommessa. |
| **SPEX in loop "liste della spesa / fattorino"** (TUTOR) | Ri-derivazione esatta dell'architettura SPEX già pianificata (predittore hidden + union blocchi + warm-keeper). Modello mentale = sistema reale. |
| **`--simulate-used-memory` per emulare la fame di RAM** (REAP iniziale) | Adottata cross-session per il capping RAM (Stage B), invece del cgroup generico. |

---

## (D) RESTA CHIUSA — con la ragione fondamentale

| Intuizione (origine) | Perché resta chiusa (ragione fondamentale) |
|---|---|
| **Pre-caricare in VRAM i primi expert del profilo** (TUTOR) | **Tier fisico sbagliato**: la cache VRAM (1-2 GiB) è troppo piccola per il profilo (i primi 10 tok toccano ~938 expert = 6.2 GiB); bench utente hit-rate VRAM 0.44 → t/s peggiore. Il profilo vive in page-cache; la VRAM ci scorre attraverso. Non contingente. |
| **RAID SSD / partizioni sullo stesso disco in RAID** (TUTOR) | **Layer sbagliato**: un NVMe è già internamente RAID-0 sui canali NAND; due partizioni condividono controller e link PCIe. Il collo non è banda ma **latenza a QD1** (si usano ~450MB/s di ~3GB/s). Oggi E-LAT lo rafforza: **copy-bound**, resident-hit=0 → la leva vera è queue-depth (sorted+parallel reads), non più dischi. |
| **DwarfStar adattivo: decomprimere/ricomprimere gli expert a K alto** (K91) | **Fisica dell'informazione**: un 2-bit non si "decomprime" — l'informazione persa dalla quantizzazione non si recupera. |
| **Mixed-precision q2 vs q2+ultimi-layer-q4** (K91) | **RETRACTED lose-lose (CLAIM-013)**: q2 e q2-q4 entrambi L1 sul task hard → il soffitto è la **TAGLIA** del modello, non i bit; q2-q4 +19% tempo. Per alzarlo serve full-q4/≥200GB o modello più grosso. |
| **Dynamic staircase / learn-live (0014)** (K91) | **RETRACTED refutato (CLAIM-016)**: avvelena la cache, il direct-descent vince. Oggi la **legge dell'ancora** lo rafforza: ogni cambio membership in corsa destabilizza. Path dinamico OFF. |
| **Spostare il drafter MTP su CPU+RAM** (DSpark) | **NO-GO da probe decisionale**: il gate "VRAM available=0.00" che si voleva aggirare non scatta mai nella config daily (solo nelle config sintetiche a budget-GB); delta vero ~5% dentro rumore ±50%. Riattivabile SOLO con evidenza nuova che il daily sia VRAM-gated (non lo è). |
| **"Non leggerli e darli per scontati su dominio ristretto"** (TUTOR) | Non chiusa per merito: è la **re-invenzione della bias-mask REAP** già esistente (leva hard che cambia il modello, distinta dal profilo soft che cambia solo la latenza). Già in-tree, non è idea nuova. |
| **Rendere DwarfStar adattivo su precisione mista per gli expert "importanti"** (K91) | Vedi mixed-precision: il soffitto è la taglia. Chiusa con la stessa ragione fondamentale. |

---

## Le 3 riesumazioni che proporrei domani

1. **Rollback → REWIND bit-exact (A1 + trigger A2).** Riabilita "cancella un % e riparti":
   l'idea era giusta, mancava lo strumento — oggi il rewind KV è provato fattibile. Esegui
   **K12-static + S1-rewind su WIDE** (cyberpunk, ctx8192, ≥4000 tok, n≥3 ABAB), con la
   **bit-equality smoke 0027 bloccante** prima dell'A/B. Misura `CORR_REWIND_TOK` +
   `useful_frac(K12,rewind)` + collapse-rate. È l'**unico** esperimento che conferma/refuta
   correzione≫prevenzione e sblocca tutta la tabella del decision model (predetto **3.99 vs
   1.56 good-tok/s**). *Massimo valore atteso.*

2. **S1-slope come TRIGGER, calibrato OFFLINE (A2).** Riabilita "prevedere la deriva": il
   negativo era solo il regime aggressivo — nel regime di erosione lenta S1 dà ~200 tok di
   lead. Replay offline dei due trace S1 per-layer già registrati → cabla ARM/FIRE (E-DET
   EWMA-CUSUM) come trigger del rewind di (1). Zero run nuovi per calibrare; è il segnale che
   (1) consuma.

3. **Leave-one-out del profilo sui trace pesati già loggati (B1).** Riabilita "il profilo
   generalizza al N+1": mai raccolta per mancanza di logging, oggi i pesi selected-6/tok
   sono già su disco. Un solo trace NARROW pesato (coffee/JSON) chiude due cose a **costo ~0
   offline**: valida la generalizzazione del profilo E ancora l'estremo basso dell'asse
   larghezza (promuove `knee(width)` da 3 punti collineari a retta fittata a 2 lati).

> Ordine per valore atteso: (1) sblocca la tesi centrale ma richiede il primitivo rewind
> (AUTHORED); (2) è il suo trigger, offline; (3) è gratis e chiude un buco cieco. La coppia
> (1)+(2) = correzione≫prevenzione reso misurabile — la scommessa più load-bearing del
> progetto, nata da un'intuizione dell'utente bocciata dall'epoca per sola mancanza di
> strumento.
