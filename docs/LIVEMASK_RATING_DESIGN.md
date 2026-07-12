# LIVEMASK — Rating / promote-prune policy (PROVVISORIO, richiede sweep)

Design del rating che guida promote/prune/block della maschera dinamica viva (patch 0035
`DS4_PACE_LIVEMASK`). **Le soglie qui sono arbitrarie**: vanno esposte come env e trovate con
uno sweep sui contatori veri (miss, copie H2D, swap, render), non a intuito.

## 0. PRINCIPIO DURO — K FISSO, non allargare

**K8 resta K8. NON si allarga** (né K, né l'ammissione, né la finestra di rating "per far
entrare più roba"). Allargare = tornare verso K0 = ammettere che l'approccio non funziona.

**Criterio di FALLIMENTO (esplicito):** se la maschera **ruota così tanto** che l'unione degli
expert nel tempo → K0-larga **e/o** la velocità → velocità-K0, allora **il meccanismo ha
fallito e va RIVISTO da capo** — NON si risponde allargando. La rotazione deve restare
**bounded** (pochi swap, working-set tight, t/s > K0). Il tasso di swap e la "distanza" dal K0
sono metriche di prima classe: se salgono verso K0, è un red flag, non un parametro da alzare.

## 1. Granularità del knock — STABILITO

- Un expert è la coppia **(layer, expert)**. In DS4-Flash ci sono **256 expert PER LAYER,
  indipendenti** (expert #5 del layer 3 ≠ expert #5 del layer 10: pesi diversi, slot di
  residenza diversi).
- Ogni layer gira **una volta per token** → un dato (layer,expert) è richiesto **0 o 1 volte
  per token**. In un token ci sono 40×6 = 240 knock-event, ma sono **240 (layer,expert)
  DISTINTI**, non lo stesso che bussa 240 volte.
- **"Bussa più volte" ha senso solo TRA token** ("richiesto in K degli ultimi W token").
- ⇒ **Il rating è per (layer, expert)**, mai per indice-solo (sarebbe 40 expert fisici diversi).

## 2. Stato del rating (0035)

- `lm_wcount[layer][expert]` = # degli ultimi W token in cui l'expert era nel top-OBSERVE_TOP
  **unbiased** → il **knock intero** (gate della promozione, confrontato con X).
- `lm_wshare[layer][expert]` = somma della share unbiased sulla finestra → la **magnitudine**
  (rank per seed + score di eviction + tiebreak).
- Finestra scorrevole **esatta** (ring subtract-then-add), W=16 default, **NON EWMA** → il
  roll-off è il decay, ed è ciò che rende il rating **fase-adattivo**.

## 3. Regole promote/prune (provvisorie — con critica)

| Regola | Attuale (0035) | Nota |
|---|---|---|
| SEED | a ~10 tok (K0), top-K per `wcount` (share tiebreak) | mask-matched-at-birth sul prompt vivo |
| PROMOTE SSD→RAM→VRAM | knock **sostenuto** ≥X (default 3) | *l'utente voleva su singolo knock → thrash misurato; sostenuto è più sicuro* |
| KEEP-ALIVE in VRAM | un residente che bussa refresha la recency, resta pinnato | ✓ |
| DEMOTE / unpin | roll-off: non usato in ~W token → `wcount`→0 → candidato eviction | *l'utente voleva 5 token → corto/churn; da tunare* |
| ANCHOR-LAW | ≤`max_swaps`/layer/scan (cap K/2), cooldown≥1, mai wholesale re-rank | strutturale dopo 0037 |

**Principio di ASIMMETRIA (chiave):** promuovere costa un load (~6.75 MiB SSD/RAM→VRAM),
depinnare è gratis (toggle di flag) ⇒ **la promozione deve chiedere più evidenza del demote**
(hyst + cooldown + rate-cap sulle promozioni). Le regole "promo-veloce + demote-veloce"
dell'utente = massima reattività ma thrash-prone; 0033 = promo-lento + demote-per-decay =
stabile ma pigro. Il sweet spot è nel mezzo e **asimmetrico**.

### Nota sul COOLDOWN (semantica + default)

Il `_COOLDOWN` è il **minimo intervallo di token tra due SWAP dello stesso layer** (guardia
anti-thrash/anti-oscillazione), **NON** un timer di "quanto un expert resta pinnato": un
residente resta eleggibile finché è top-K per rating. **MA** un cooldown alto rende la maschera
**lenta ad adattarsi** — un residente raffreddato resta eleggibile fino a `_COOLDOWN` token dopo
aver smesso di servire. Il **64 iniziale era ereditato dal default RESIDENZA di 0033** (concern
diverso) ed è **troppo sluggish** per una maschera di SELEZIONE fase-adattiva (i shift di
dominio avvengono in ~10-20 token). **Default abbassato a ~16**; è la **leva #1 di reattività**
dello sweep, in coppia con `_WINDOW` (che governa quando il knock decade a 0). Troppo basso →
thrash (→ K0, §0); troppo alto → sluggish. Negli smoke corti: 8-16.

### Principio: reattività dall'ORIZZONTE corto, stabilità dalla PROMOZIONE dura

Scelta architetturale (utente): **tenere gli orizzonti CORTI** (finestra ~10, cooldown ~8) e
ottenere la stabilità **alzando il muro in ingresso** (promozione dura), NON allungando i guard.
Motivo: un cooldown/finestra lungo rende sluggish *tutto*, incluso il demote dei freddi; orizzonte
corto + promozione stretta = **adatta in fretta MA non ammette flukes**.
- Finestra **W = 10**, cooldown **~8**.
- Promozione **dura**: `_X` alto sulla finestra (≥5-7/10), `_HYST` ampia, blocked-demand forte.
- Demote **facile**: freddo negli ultimi ~10 → candidato all'uscita.

**Sottigliezza (a favore della stabilità):** con K costante (one-in/one-out) **non si demota
senza promuovere un rimpiazzo** → se nessun candidato passa il gate duro, lo swap non avviene →
il set resta ≈ seed. Promozione dura ⇒ mask stabile, cambia solo su expert davvero forti.

**IL sweet spot dello sweep:** promozione troppo dura → mask ≈ seed ≈ K8-statico → **NON rende**;
troppo molle → thrash → K0 (§0). L'obiettivo è **la promozione più dura possibile che ANCORA
renda**. Questo, non la velocità, è ciò che lo sweep cerca.

### Il cooldown NON è la guardia — l'isteresi lo è (cooldown = METRICA)

Insight (utente): **se il rating è solido, il cooldown non scatta mai.** L'anti-thrash vive nel
RATING (banda isteresi promuovi-a-X / demota-a-X−margine, gate a domanda sostenuta, margine di
share, roll-off della finestra), **NON** nel timer. Un rating con isteresi corretta non oscilla →
il cooldown resta idle.
⇒ **Ribaltiamo il ruolo del cooldown: da leva a METRICA.** Da strumentare: **`cooldown_fire_count`**
= quante volte blocca uno swap che il rating voleva fare.
- Target **≈ 0** = rating solido.
- Scatta spesso = il rating sta thrashando → si **aggiusta il RATING** (isteresi ↑, gate ↑), NON
  si allunga il cooldown.
Il cooldown resta solo **backstop** per input patologici (domanda A-B-A-B/token); default minimale
(2-4). La sua frequenza è un **health-check del rating**, da loggare in ogni smoke.

## 4. Dinamiche aggiuntive proposte (da valutare nello sweep)

1. **Knock pesato per rank/probabilità** del router (non binario) — già `lm_wshare`.
2. **Blocked-demand > satisfied-demand:** peso maggiore ai knock di chi è ESCLUSO dalla mask
   (domanda insoddisfatta = segnale più informativo). → env `_BLOCKED_WEIGHT`.
3. **Promozione adattiva alla PRESSIONE aggregata (intuizione utente 2026-07-12 — candidato fix
   del collasso):** oltre alla temperatura per-esperto, il rating deve vedere **quanti bussano
   CONTEMPORANEAMENTE**. Segnale = **differenziale di domanda**: `pressione = (# esperti BLOCCATI
   che bussano forte nella finestra) vs (# residenti/eleggibili che si RAFFREDDANO)`. Quando la
   pressione spara (shift di fase: prosa→codice → tanti codice-esperti bussano mentre i
   prosa-esperti pinnati si raffreddano) → **promuovi PIÙ IN FRETTA**: cooldown transiente ↓, X ↓,
   più swap/layer. **Ruota più veloce, MAI allarga K** (§0). È il "respiro di fase" fatto con la
   velocità di rotazione. **Ipotesi del collasso K8:** il promote lento (X≥3 + cooldown) non regge
   il picco di pressione dello shift → i codice-esperti non entrano in tempo → word-salad. →
   env `_PRESSURE_PROMOTE` (soglia pressione), `_PRESSURE_COOLDOWN`/`_PRESSURE_X` (valori accelerati).
4. **Co-occorrenza (SPEX):** se A e B vanno spesso insieme, promuovere A **pre-scalda** B
   (il predittore SPEX lo fa cross-layer).
4b. **SWAP GROSSO → FATTORINO async (intuizione utente 2026-07-12):** quando la promozione a
   pressione (#3) promuove MOLTI esperti insieme (shift di fase), i load NON vanno fatti sincroni
   (stallo, il "0.25 t/s" del thrash). Vanno passati al **fattorino = WRAP (patch 0013**, env
   `DS4_REAP_PREFETCH_THREADS/_LOCK`, oggi dormiente), che li carica in **background sovrapposti al
   compute**. Catena: DECIDE (rating/pressione) → CARICA (fattorino async) → TIENE (cache/pin
   pinnati). La 0021 (rotate-delta-prefetch) copre già lo swap PICCOLO (1-2 esperti); il fattorino
   serve per il BATCH. **Dipende dal fix cache/pin** (sennò crescita/thrash azzera i prefetch).
   Costruire nella sessione cache/pin (lato async-loading della residenza).
5. **Decoupling selezione vs residenza:** un expert può restare **eleggibile** ma con residenza
   che scende a RAM se scelto di rado — due rating separati, non uno.
6. **Rate-cap promozioni/intervallo** (il load è il costo) + **eviction pesata**
   (recency × domanda), non LRU puro.
7. ~~Layer-depth aware (K più largo nei layer profondi)~~ — **RESPINTA (§0):** allargare K viola
   il principio. Se i layer profondi (diffusi) non tengono K8, è un **segnale di fallimento**,
   non un motivo per allargare. Al massimo: rotazione più attiva lì, K uguale.

## 5. Env da esporre per lo sweep

- **Esistenti (0035/0037):** `_K`, `_WINDOW`, `_X`, `_MAX_SWAPS`, `_COOLDOWN`, `_HYST`,
  `_OBSERVE_TOP`, `_BOOTSTRAP`, `_WEIGHTED`.
- **Da aggiungere:** `_BLOCKED_WEIGHT`, `_RANK_WEIGHT`, `_DEMOTE_HORIZON`, `_PROMO_RATE_CAP`,
  `_PHASE_BREATH`.

## 6. Piano

Le regole sono **arbitrarie ora**. Ordine corretto:
1. Il mechanism **gira e MISURA** (0036 in-place + 0035 livemask + 0037 hardening, con LOG).
2. Espone **tutte** le regole come env.
3. **Sweep per dominio** (promo stretto vs coding largo) sui contatori veri (miss, copie, swap,
   render/chiusura `</html>`, t/s) → sweet spot.

Non tunare a intuito: prima il mechanism, poi lo sweep sui numeri.

## 7. BRIDGE "pin per massa" (DECIDE → TIENE) — CONTRATTO DI COORDINAMENTO

**Lacuna diagnosticata (2026-07-12, verificata sul codice 0035):** il reap loop decide la
SELEZIONE (chi è eleggibile) ma **non comanda il pin**. Prove nel codice:
- lo scan sceglie il candidato per **frequenza** (`lm_wcount`), non per massa; la massa
  (`lm_wshare`) la usa solo per vittima + margine di gate;
- dopo lo swap fa solo `ds4_reap_mask_apply` (applica la mask); il commento è esplicito:
  *"il delta-prefetch non è presente… la residenza segue lo stream **emergentemente** via 0033
  (cold miss alla prima richiesta, poi re-pin)"* → il pin **non è comandato**, emerge dai miss;
- il canale esplicito `DS4_PACE_LIVEMASK_PIN`/`lm_pin` è uno **stub deferito** ("tier bridge not wired").

Manca l'arco **DECIDE → TIENE**: il motore ha già il segnale di massa (`lm_wshare` = Σ share =
freq×peso) ma non lo esporta come **ordine di pin** al layer di residenza.

### Il seam (FROZEN — nessuna sessione lo ridefinisce)

Segue il precedente di `g_reap_mask_pruned` (globale che ds4.c scrive e ds4_cuda.cu legge):

```c
/* ds4.c produce, ds4_cuda.cu consuma. Massa windowed dell'expert ELEGGIBILE; 0.0 se pruned. */
extern float           g_reap_pin_mass[DS4_MAX_LAYER][DS4_MAX_EXPERT];
extern volatile uint32_t g_reap_pin_epoch;   /* bump a ogni cambio della mask/pin_mass */
```

- **Attivazione:** `DS4_REAP_PIN_BY_MASS=1` (entrambi i lati gate su questo env; default 0 =
  comportamento emergente attuale → **niente si rompe da spento**).
- **Semantica:** `g_reap_pin_mass[L][e]` = `lm_wshare[L][e]` se `e` è eleggibile (mask non-pruned),
  `0.0` altrimenti. Più alto = tieni pinnato. Aggiornato **una volta per token decode** a fine
  scan (dopo che la mask si è assestata). `g_reap_pin_epoch++` a ogni variazione.

### Produttore — ds4.c / motore rating [→ sessione PRESSIONE, quizzical-bose]

1. **Rank per MASSA:** seed e scelta candidato ordinano per `lm_wshare` (massa), non `lm_wcount`.
   `lm_wcount` resta come **gate di knock sostenuto** (≥X token), ma l'ORDINE è per massa.
2. Popola `g_reap_pin_mass` da `lm_wshare` per gli eleggibili (0 per i pruned) ogni token; bump epoch.
3. La promozione a pressione aggregata (§4.3) ci sale sopra: sotto pressione ruota più in fretta,
   e il `g_reap_pin_mass` pubblicato riflette quella rotazione (nessun nuovo seam).

### Consumatore — ds4_cuda.cu / residenza [→ sessione CACHE/PIN, cool-elbakyan]

1. **Prerequisito:** fix cache-growth-pin-thrash PRIMA (senza, qualunque pin viene azzerato).
2. Cabla `lm_pin`: con `DS4_REAP_PIN_BY_MASS=1`, **pinna attivamente** il set eleggibile invece di
   aspettare il cold-miss; in eviction, butta per **primo il residente con `g_reap_pin_mass` più basso**
   (non LRU puro).
3. **Fattorino (WRAP async)** per il carico BATCH quando il motore ruota molti expert insieme (§4.4b).

### Sync point

Nome array + semantica + env sono **congelati da questo contratto**. Nessuna sessione li cambia
in autonomia: ogni modifica del seam torna all'orchestratore. Le due metà si costruiscono in
parallelo e si incastrano su questo seam.
