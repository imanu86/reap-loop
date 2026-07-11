# AUDIT — verdetto rewind 0022-v3 (scoped) + reframe a due assi

**Data:** 2026-07-11 · **Tipo:** audit trasversale (no nuovo compute; lettura di
smoke a/b/c, ripivotal v3, pivotal v2, design-doc, instrumented-collapse,
pin-viability). · **Scopo:** fissare il verdetto PRECISO su 0022-v3 rewind, dire
se resta un test-rewind che vale GPU o se è chiuso, e riquadrare la strategia
velocità sui due assi che restano.

Fonti primarie lette:
`runs/ds4/20260711_smoke_0022v3/SUMMARY.md`,
`runs/ds4/20260711_ripivotal_k12_rewind_v3/REPORT.md`,
`runs/ds4/20260711_pivotal_k12_rewind/REPORT.md`,
`docs/S1_REWIND_DESIGN.md` (§3.1 scope-note vincolante),
`runs/ds4/20260711_instrumented_collapse/metrics.json`,
`docs/RESIDENCY_ROTATION.md` + `runs/ds4/20260711_pin_viability_and_gaps/REPORT.md`,
`docs/DS4_EXPERIMENT_LEDGER_20260710.md` (CLAIM-011).

---

## 1. VERDETTO PRECISAMENTE SCOPATO

**Il rewind 0022-v3 è REFUTATO PER I TASK WIDE/AGGRESSIVI, e il meccanismo è
confermato funzionante — ma lo status nel regime NARROW/slow-erosion (il suo
target di design) è UNTESTED, non refutato.** Sono due affermazioni distinte e
vanno tenute separate.

### 1a. REFUTATO-per-wide (airtight, meccanismo verificato)
- **Meccanismo sano** (`smoke_0022v3/SUMMARY.md`, gate a/b/c = PASS, build ds4.c
  md5 `d4ff85af`): il detector char-garbage arma+spara DEEP e PRE-lock (pos 209,
  ~51 tok prima della S1 v2 che sparava a 260), atterra a `to=118` (salto 91 tok,
  sotto l'onset di erosione ~pos 190), e la rigenerazione DIVERGE (138/148 pos
  diverse, emette un vero `<title>AI Cyber Shop</title>` + `<style>` fresco). Il
  negativo NON è artefatto di un meccanismo rotto.
- **Refutazione qualità** (`ripivotal_k12_rewind_v3/REPORT.md`): campagna n=3 +
  ladder K-escalation. ARM1 (K12+rewind-v3, n=3) = 0/3 `</html>`, tutti L0, 3
  seed BYTE-IDENTICI (4615 char): cicla a REWIND_MAX=2 e ri-collassa in un nuovo
  lock `1:`-repetition. Ladder K in {12,20,28,36,44,48} = 0/6 chiudono. Conferma
  n=3 al tetto K48 = 0/3, byte-identici (4268 char), onset lock 280.
  **Gran totale: 0 `</html>` in 12 run mascherate. Storico 0/23 -> resta 0.**
- La conclusione v2 "narrow mask / poisoned context wins" SOPRAVVIVE a tutti e
  tre i fix meccanicistici v3 (deep ring / char-garbage EWMA / resume-warmup).
- **Confound onesto sul tetto:** a K48 il rewind ha sparato 0 volte (arm/fire
  1/0, garbage <0.80) -> K48 = di fatto static-K48 puro, il rewind-v3 era INERTE
  lì. Il rewind ha effettivamente SPARATO solo a K12/K36/K44; K20/K28 rewind_n=0.
  Quindi "nemmeno K48+rewind chiude" è vero ma va letto come "K48-static non
  chiude", non come "rewind provato e fallito a K48".

### 1b. Status NARROW/slow-erosion = UNTESTED (coverage gap pre-registrato)
- **Ogni** rollout di rewind mai eseguito (pivotal v2 n=2/3, ripivotal v3 n=3,
  smoke n=1) è cyberpunk-WIDE / K12-static / fast-collapse. Il regime di design —
  narrow/slow-erosion K91-family, l'UNICO con lead S1 misurato — non è mai stato
  dato a nessun rewind, né v2 né v3.
- È scritto e VINCOLANTE in due punti:
  - `S1_REWIND_DESIGN.md` §3.1: *"this S1 early-warning only buys lead in the
    SLOW-EROSION regime (K91-family)... In the aggressive dynamic regime
    (W50+K23+rotate32) S1 is pinned flat ~0.815 from mask-engage... there the S1
    arm of this ladder is inert."*
  - `pivotal_k12_rewind/REPORT.md` (riga 82): *"Lo scope slow-erosion
    (K91-family) resta NON testato da questo esperimento — per design: è l'unico
    regime dove S1 ha lead misurato (CLAIM-011) e dove la scala
    prevention->correction può ancora pagare come progettata."*
- Lead misurato SOLO nel design-regime: CLAIM-011 (S1 slope +0.058 su ~200 tok,
  K91 0.73->0.81; E-DET tuning ~214-225 tok di lead prima del text-lock 2476),
  stato **OPEN**. In wide S1 è piatto (aggS1 z-onset pos 1200 = inutile,
  `instrumented_collapse/metrics.json`).

**Nota di precisione (correzione premessa):** l'"entropy-lead +57" NON è una
misura del narrow-regime. `instrumented_collapse/metrics.json` è etichettato
`run: K12-wide ... cyberpunk`: entropy z-onset pos 125 (il detector più precoce),
lead fino a +57/+68/+82 tok — MISURATO IN WIDE. Cioè entropy/char-garbage hanno
lead anche in wide (ed è esattamente ciò che il detector v3 sfrutta); è solo la
**S1** ad andare piatta in wide. Lo split pulito narrow/wide vale per il segnale
S1, non per entropy. Il design-regime resta non testato per copertura, non
perché "non warnable".

**Sintesi 1:** negativo WIDE equo, deterministico, meccanismo-verificato per
K12..48 cyberpunk/ctx8192/4000. NON è, e non pretende di essere, una refutazione
universale. Nel regime narrow/slow-erosion per cui il rewind è stato progettato:
**status ignoto — zero rollout, non un fallimento.**

---

## 2. RESTA UN TEST-REWIND CHE VALE GPU? — SÌ, ESATTAMENTE UNO (bassa priorità)

Il negativo wide è airtight; **non** vale un altro dollaro su rewind wide
K12..48 (chiuso). Ma NON è chiuso in scope: gli input scettici convergono su
**un** rescue plausibile-ma-non-testato, ed è lo stesso identico esperimento che
i due design-doc pre-registrano come mai fatto.

**Il test (unico) che vale GPU:** rewind ACCESO NEL SUO REGIME DI DESIGN —
slow-erosion, K SOPRA il ginocchio wide (famiglia K64-K91) o narrow, dove il
detector sia FIRE **sia** l'erosione sia abbastanza lenta da poter STERZARE
(prevention->correction), n>=3, orizzonte lungo (~2500 tok, dove il text-lock K91
cade a 2476). È l'unica cella dove il rewind sia (i) spara e (ii) ha finestra di
sterzata — mai realizzata: le rung dove il rewind ha sparato (K12/K36/K44)
collassano troppo in fretta; le rung a erosione più lenta (K20/K28/K48) sono
esattamente quelle dove il detector NON ha mai sparato.

Leve **scartate** come non-rescue (verificate, non valgono GPU):
- **REWIND_MAX>2:** la 0022-v3 spedita non incrementa mai il keep per-rewind
  (design §3.4 lo specifica ma il binario non lo fa); più budget oscilla soltanto
  su una mask K12 a capacità insufficiente. La ladder->K48 già refuta il
  widen-and-hold.
- **CKPT_DEPTH più profondo:** depth=8 atterra GIÀ pre-erosione (to=118 <
  onset ~190); il fallimento è il **ri-collasso post-restore**, che la profondità
  non tocca.

**Priorità: BASSA.** Tre freni onesti: (a) il design-regime **chiude già senza
rewind** (coffee K23 static L1-L2, K91 coding regge ~2200/text-lock 2476), quindi
il valore MARGINALE del rewind lì è non provato — un eventuale close potrebbe
accreditare la mask più mite, non il rewind; (b) K91 non entra in 12GB reali ->
serve un pod (costo/priorità); (c) plumbing non fixato (airbag n-gram dead-code
via de-arm nel breath-branch, saturazione CUSUM in lock stazionario, bit-exattezza
frontier §1.5 non ancora validata dallo smoke 0027) può nullare per motivi
non-scope. Confidenza dei rescue: bassa (narrow-trigger/entropy) -> media (tetto
K>knee). **Verdetto: non chiuso in assoluto, ma il singolo esperimento residuo è
low-ROI; il ramo wide è chiuso.**

---

## 3. REFRAME STRATEGICO A DUE ASSI (cosa comprare col prossimo dollaro su task wide)

Il campo di manovra per la velocità su un task WIDE è chiuso su due lati:

- **Asse A — ricomprare qualità DOPO il collasso (rewind).** FALLISCE. Non puoi
  cancellare il contesto avvelenato e ri-convergere: 0/12 close v3, il rewind è
  un *rate-improver*, non un *page-completer* (useful_frac 0.04->0.17, gtps
  1.18->3.7, ma L0 ovunque).
- **Asse B — barare abbassando K sul task wide.** FALLISCE. K48 non tiene (onset
  280, 0/3), la soglia di chiusura wide (se esiste) è **>48**; scendere sotto il
  ginocchio del task fa dominare l'hazard di copertura -> collasso certo.

=> **Su un task wide la velocità NON viene né dall'asse-qualità-post-collasso né
dall'asse-under-provisioning-di-K. Viene SOLO dall'asse RESIDENZA: 0031 pin-keep,
a un K appropriato-al-task (>= ginocchio, non sotto).** 0031 è residency-only,
output bit-identico (`docs/RESIDENCY_ROTATION.md`: scrive residenza VRAM, non
`g_reap_mask_pruned` né bias) -> non può convertire L0->L2, ma non è per quello:
pinna un sottoinsieme caldo dei keep in VRAM, alza il resident-hit, abbassa
miss(K), alza t/s **senza toccare la traiettoria**. Non barare su K, tenere il K
giusto e comprare velocità dalla residenza.

**Come il rate confermato de-riska la predizione 0031 (e cosa lascia aperto —
onesto):**
- **De-riska (il verso della predizione):** il RANKING dell'obiettivo — "il
  piccolo-K speed-dominated vale la caccia, il soffitto di rate è raggiungibile"
  — è vendicato: v2 gtps 1.18 -> v3 ~3.7 ~= i 3.99 del decision-model. Il tetto di
  rate esiste ed è raggiungibile, e il meccanismo è reale. Aggiungere 0031 a K
  appropriato è quindi **low-risk** (bit-identico, può solo aiutare la velocità)
  e ben motivato. pin-viability conferma K12 pin-viable (working-set 516;
  hot-core k90=9/12) mentre K23 è residency-starved sul 3060 (working-set 989 >
  cache).
- **NON de-riska (il numero core):** il 3.7~=3.99 è stato misurato su POD (RTX
  3090, cache 1024) il cui cache **eccede** il working-set K12 (516) -> NON è
  residency-starved: il run non esercita mai la pressione VRAM del 3060 che 0031
  esiste per alleviare. Il numero che conta per 0031 — la curva t/s = f(hit-rate)
  end-to-end sotto blocking-sync sul 3060 — resta il **gap G5**
  (`pin_viability_and_gaps/REPORT.md`), non misurato. Lo smoke gate di 0031 è
  pending (no CUDA locale). Attenzione anche al confound: i due errori ~5x
  (pod t/s troppo alto vs useful_frac del modello troppo alto) si compensano nel
  prodotto 3.7 — è il composito a coincidere, non i fattori singoli.

**Riga di reframe (una frase):** su task wide non puoi ricomprare qualità dopo il
collasso (rewind refutato) né barare abbassando K (K48 non tiene) => la velocità
viene SOLO dall'asse residenza (0031 pin-keep) a K-appropriato-al-task, e il rate
confermato (3.7~=3.99) de-riska il VERSO della predizione 0031 (rate-ceiling
raggiungibile, add bit-identico low-risk) lasciando il NUMERO core (curva
hit->t/s sul 3060, gap G5) ancora da misurare.

---

## 4. Aggiornamento a CLAIMS_CURRENT.md

Aggiunta una riga in "Negativi onesti / ritrattazioni" per rewind/0022-v3 con lo
stato scopato **REFUTED-for-wide / OPEN-narrow (untested)**, che cita questo
audit e le fonti primarie. Nessun altro claim toccato (regola anti-regressione
rispettata: F1, near-lossless-loop, asimmetria restano come sono).
