# PHASE-ADMISSION DYNAMIC-RESIDENCY CONTROLLER — PLAN (the endgame)

> **Una leva per due problemi.** Tieni residenti in VRAM i ~240 Expert della fase
> corrente (ENTRANO nei ~394 slot → il modello gira *come se fosse tutto in VRAM* →
> velocità) **e** ammetti gli Expert della fase quando la competenza cala (rende +
> chiude la coda `</html>`). La stessa mossa — cambiare *quali* Expert vivono nel set
> residente-di-fase — compra velocità (fit) e qualità (giusti) insieme. È la
> maschera-dinamica-dal-prompt dell'utente (I4), governata dalla rotazione
> condizionale-sulla-competenza (I6): l'endgame, non un ottimizzatore in più.
>
> **Provenienza.** Sintesi dei tre survey del 2026-07-11 (residenza/velocità,
> ammissione/qualità, Legge-dell'Ancora/rewind) *ricondotta alle fonti di verità del
> repo*: `INTUITION_REGISTER.md` (I1–I10, §2 controller di residenza viva) e
> `CLAIMS_CURRENT.md` (numeri correnti). Se un numero qui contraddice
> `CLAIMS_CURRENT.md`, vince quel file. Nessun claim OPEN è presentato come CLOSED.
> Data: 2026-07-11. Stato: **design** (read-only sul motore; niente codice-motore qui).

---

## 0. TL;DR operativo

- **Due strati separabili.** (a) **RESIDENZA** (stile 0031 pin-keep): tiene il set
  per-fase inchiodato in VRAM. **Non tocca la selezione → output bit-identico → pura
  velocità (fit-in-VRAM).** (b) **AMMISSIONE** (stile 0026 + mappa-targeting): cambia
  la mask/selezione, *targeted* 2–5 Expert/layer, **gated su competenza/entropia →
  qualità + chiusura della coda.** L'ammissione perturba la traiettoria → è lei, non
  la residenza, a rischiare la **Legge dell'Ancora**.
- **Vincolo di fit condiviso.** Il set residente = *keep-corrente + admit-di-fase*
  deve restare **≤ ~394 slot**. L'ammissione è **K-costante (uno-dentro/uno-fuori)**
  proprio per non gonfiare il budget: swappa, non cresce.
- **La dinamica è NECESSARIA, non un lusso.** Nessuna mask **statica** è insieme
  «entra» **e** «rende»: la più piccola che rende (K12, unione 516 Expert) sfora i
  ~394; solo la dinamica (unione 516, istantaneo per-fase ~240, backbone ~354) entra
  E rende. Questo è l'argomento-esistenza del controller.
- **Quattro stadi con gate misurabili sul 3060:** S0 residenza-sola (bit-exact,
  entra, veloce) → S1 ammissione statica-precalcolata su trigger-entropia → S2
  scoring/promozione/lock dinamico → S3 phase-detect a runtime. Ogni gate risponde a
  **fit? rende? chiude? t/s?**
- **Primo bersaglio: il caso PROMO** — dominio stretto, 2 fasi (prosa → JSON),
  transizione = `{`. Confine di fase leggibile ⇒ il controller più facile da chiudere.

---

## 1. Il problema unico (perché una leva sola)

Storicamente li trattavamo come due problemi:

1. **Velocità.** Sul 3060 (12 GB) il working-set wide non entra in cache → refetch da
   RAM/SSD ogni token → ~0.8–3.4 t/s *stuck* (tetto di **capacità**, non di motore).
2. **Qualità/chiusura.** A K aggressivo la traiettoria erode e non chiude il documento
   (storico `</html>` 0/23): la coda non arriva.

La correzione critica del 2026-07-11 (INTUITION_REGISTER §2; CLAIM TIMING-SEGMENTATO)
li unifica: **la velocità È fit-in-VRAM, e il fit-in-VRAM si ottiene tenendo piccolo e
giusto il set residente-di-fase — che è esattamente ciò che serve anche alla qualità.**
Tenere residenti *gli Expert giusti della fase* (~240) li fa entrare (velocità) *ed* è
la copertura che chiude la coda (qualità). Una leva — la membership del set
residente-di-fase — due effetti.

---

## 2. Ancoraggio misurato (i numeri che vincolano il design)

| Grandezza | Valore corrente | Fonte |
|---|---|---|
| **Confine-fit reale (3060)** | **~394 slot Expert** residenti (cache≈400, ~12 GB). Non-expert *flat* ~9.86–9.95 GB; KV piccola **per costruzione** (CSA/HCA, finestra 128 tok) → il 3060 è **fixed-weight-bound**, liberabili solo ~0.3–0.5 GB. | `runs/ds4/20260711_vram_split_kv/REPORT.md`; INTUITION_REGISTER §2 (KV-headroom) |
| **Prova che il FIT è la leva** | Stesso 3060, cache400: **keep-8 (working-set entra) = 23–25 t/s**; **keep-32 (working-set 1280 > 400) = 3.4 t/s STUCK**. ~7× dal solo fit. | CLAIM «TIMING SEGMENTATO» (CLAIM-008) |
| **Attivi per token** | **240** (6 top × 40 layer MoE), COSTANTE per ogni K e dominio (full compreso) = ~1.6 GB. I «989/516» sono l'**unione nel tempo**, non l'istantaneo. | INTUITION_REGISTER I7 |
| **K-min-che-rende** | **12** (L1, n=2; commit c3b49ae). K12 domain-matched RENDE. | INTUITION_REGISTER §2; c3b49ae |
| **Nessuna statica entra+rende** | K12 unione = 516 > 394; K23 unione = 989. Backbone(≥5/20) K12 = **354 Expert / 2.3 GB / hit 0.93 → ENTRA**. | INTUITION_REGISTER §2 (BACKBONE) |
| **Lead del sensore (entropia)** | Pre-collasso: entropia sale con **~+57 tok di lead** (S1 slope +0.058, 0.722→0.781). Narrow/erosione-lenta CLAIM-011 ~200–225 tok. **Lead-time da confermare (OPEN).** | CLAIMS «FEEDBACK slope-S1»; CLAIM-011 |
| **Meccanismi disponibili** | 0031 pin-keep (residenza, bit-exact); 0026 demand-admission (CUSUM 1-in/1-out, K-costante, gated); mappa-targeting (2–5 Expert/layer/fase); sensore S1/entropia; 0022 rewind-KV. | patches/ds4/ |

**Riconciliazione del «~240 vs ~354».** L'istantaneo per-token è 240; il **backbone
di fase** (residenza soft, top-N per condivisione ≥T/20) è ~354 a K12. Entrambi < 394.
Il set residente-di-fase vive quindi in una finestra: **floor = backbone (~354) ≤
residente ≤ budget (~394)** — restano ~40 slot liberi + la coda LRU rotante per il
churn e per i delta di ammissione.

---

## 3. ARCHITETTURA A DUE STRATI

Il perno di tutto il design è la distinzione **residenza ≠ selezione**, oggi
**disaccoppiate nel motore** (audit `20260711_pinning_divergence_audit`: la REAP-mask
scrive solo `-1e9` sul bias-router; la cache Expert è una LRU keep-blind).

### (a) STRATO RESIDENZA — «tieni il set-di-fase in VRAM» (stile 0031)
- **Cosa fa.** Inchioda in VRAM un sottoinsieme caldo del *keep* corrente (flag
  `pinned` sugli slot già allocati della streaming-cache; ripartisce la cache in
  «pinned eviction-immune» + «LRU rotante», **zero VRAM nuova**).
- **Invariante.** **Non scrive mai `g_reap_mask_pruned` né il bias-router ⇒ la top-k è
  byte-per-byte identica ⇒ output BIT-IDENTICO** al path non-pinnato. La residenza è
  *pura gestione di memoria*: **non può collassare la traiettoria.**
- **Cosa risolve.** SOLO velocità, via fit: il backbone-di-fase (~354 a K12) resta
  residente → resident-hit sale → niente refetch per-token → si va dai ~3.4 *stuck*
  verso il regime «entra».
- **Driver di riempimento (i due predittori, INTUITION_REGISTER §2).**
  (i) **Recency/temporale** — demand-EWMA (rmass-analog): tiene calda la fase corrente
  (hit ultimi-16-tok K23 0.97, cyberDOM 0.89); eviction = *return-aware weighted LRU*.
  (ii) **SPEX/cross-layer** — predice L+1 dallo stato nascosto di L; prende il **bordo**
  della transizione (prosa→codice) che la recency sbaglia. Si compongono: recency tiene
  la fase, SPEX prende il confine, 0031 li inchioda (seed→pin→costo-H2D ammortizzato).

### (b) STRATO AMMISSIONE — «cambia CHI è nel set quando serve» (stile 0026 + mappa)
- **Cosa fa.** Modifica la **selezione**: ammette 2–5 Expert/layer (uno-dentro/
  uno-fuori, **K-costante**), *targeted*, **gated su competenza/entropia**. Il pruned
  con domanda-bloccata forte e persistente entra; il keep meno-usato di quel layer
  esce; solo l'entrato è paginato (~6.75 MiB delta-prefetch, niente WRAP).
- **Cosa risolve.** Qualità + **chiusura della coda**: reintroduce la copertura che la
  fase nuova richiede → la traiettoria non erode → il documento chiude (`</html>`/`}`).
- **Costo/rischio.** Perturba la traiettoria (cambia la mask) → **NON bit-identico** →
  è lo strato esposto alla **Legge dell'Ancora** (§4).

### Interazione col FIT-BUDGET (il vincolo che lega i due strati)
```
set_residente(t) = backbone_di_fase  +  delta_ammessi(t)     ≤  ~394 slot
                   (floor ~354 K12)     (2–5/layer, K-costante: SWAP, non crescita)
```
La residenza **serve** qualunque selezione l'ammissione produce, veloce. Quando
l'ammissione swappa X→Y: (1) residenza pagina Y (delta-prefetch) e può pinnarlo; (2)
lo slot di X si libera. Poiché l'ammissione è **K-costante**, il conteggio residente
non cresce mai → il budget ~394 non salta. *Se* uno stadio volesse **aggiungere**
(non swappare) Expert di fase, deve stare nello slack `394 − backbone (~40 slot)` + coda
LRU. Regola dura: **mai far superare ~394 al set residente** (oltre → si ricade nel
regime refetch 3.4 t/s e la leva-velocità si spegne).

---

## 4. IL RISCHIO CENTRALE — la Legge dell'Ancora, e come l'ammissione la evita

**Legge dell'Ancora.** Cambiare la *membership della selezione* in corsa destabilizza
la traiettoria: la rotazione **a-calendario / cieca / wholesale** collassa (rotate32/
0015 = collasso wide E-CAL; lo staircase learn-live avvelena la cache — RETRACTED). La
mask è un'*àncora*: strapparla mentre generi fa deragliare. **La residenza (0031) è
immune** (non tocca la selezione). È **l'ammissione** che deve rispettare la Legge.

Quattro difese, tutte già isolate nei meccanismi del repo:

1. **Condizionale, non a-calendario (I6).** Si ammette **solo quando la competenza
   cala** (entropia sale / slope-S1), toccando **solo le traiettorie che stanno già
   fallendo**. La rotazione condizionata-sulla-competenza non collassa dove quella
   cieca sì (tocca il minimo indispensabile).
2. **Early + targeted, K-costante (0026).** Si usa il **lead** del sensore (+57 tok;
   narrow ~200–225) per ammettere **PRIMA** del collasso, e si swappa **2–5 Expert/
   layer uno-dentro/uno-fuori**, mai wholesale re-rank, mai K0. L'àncora (mask di
   warmup) resta; solo swap per-layer giustificati. (0026: recupera 13.7 pt della
   copertura tardiva a 5.4× meno churn, ~zero rimbalzi.)
3. **Promozione su segnale STRUTTURALE (il caso PROMO).** Al confine di fase noto —
   il **`{`** prosa→JSON — si ammette il set-JSON **al `{`**: trigger di *contenuto*,
   non stima rumorosa. È il verso «variant-D structural-boundary» di 0026, qui banale
   perché il confine è un token.
4. **Rewind-al-punto-di-ammissione (rete di sicurezza, 0022).** Se un'ammissione
   spinge la traiettoria nel garbage, **rewind-KV al punto di ammissione** (checkpoint
   pre-swap noto-buono) e rigenera col set nuovo. Bounded per costruzione (torni a un
   pre-ammissione sano). **⚠️ Onestà (CLAIMS_CURRENT):** il rewind è **REFUTATO su
   task WIDE** (0/12, post-collasso) e **NON testato in narrow/pre-emptive** — questa è
   l'ipotesi, non un recupero provato. Vive solo come rete di S1+, gated dietro misura.

---

## 5. PIANO A STADI CON GATE (ognuno misurabile sul 3060)

Ogni stadio dichiara **fit? rende? chiude? t/s?**. Gli stadi sono cumulativi; ognuno
è un *go/no-go* prima del successivo.

### S0 — RESIDENZA-PER-FASE SOLA (nessuna ammissione)
- **Cosa.** Pin del keep-set per-fase (0031, `DS4_PACE_PIN=1`,
  `DS4_CUDA_NO_Q8_F16_CACHE=1`). Mask **statica** per-fase, congelata. Zero swap di
  selezione.
- **Gate.**
  - **fit?** working-set ≤ ~394 residenti; resident-hit sale, `selected_direct_loads`/
    token scende (`DS4_SPEX_STATS=1`).
  - **rende?** *per costruzione = quanto rende la statica per-fase* (K12 domain-matched
    = L1). S0 **non migliora** la qualità: è il pavimento.
  - **chiude?** **NO da solo** (eredita la coda della statica). Onesto: S0 è lo stadio
    *velocità*.
  - **t/s?** target: uscire dal 3.4 *stuck*; regime «entra» ~keep-8 = 23–25 t/s come
    tetto-di-fit; per K12-realistico sul 3060 ~3–4 t/s (KV-headroom risolto). **⚠️
    smoke 0031 PENDING** (niente CUDA locale) — bit-exactness + Δt/s da misurare.
- **Perché prima.** Isola la leva-velocità *senza* rischio-Ancora (bit-identico). Se S0
  non dà fit+Δt/s, tutto il resto è discutibile.

### S1 — AMMISSIONE STATICA-PRECALCOLATA, triggerata su entropia
- **Cosa.** Dalla **mappa-targeting** (probe della fase, 2–5 Expert/layer/fase,
  precalcolata *offline*), il controller decide **QUANDO** swappare-in il set-di-fase,
  su **slope-S1/entropia** (o sul `{` nel caso PROMO). L'admit-set è **fisso**; dinamico
  è solo il *momento*. (0026 come attuatore dello swap, K-costante.)
- **Gate.** **fit?** K-costante ⇒ ≤ 394 mantenuto. **rende?** ammettere il set-di-fase
  precalcolato recupera la copertura tardiva (target: L1→L2 sul confine). **chiude?**
  *il gate chiave* — primo `</html>`/`}` sul PROMO. **t/s?** costo swap = delta-prefetch
  ~6.75 MiB/Expert, ammortizzato su fase lunga (deve restare trascurabile vs il guadagno
  di fit). **churn?** bounded (cooldown anti-thrash).
- **Rischio-Ancora coperto da:** difese 1–3 (condizionale + early-targeted + `{`).

### S2 — SCORING / PROMOZIONE / LOCK DINAMICO
- **Cosa.** Niente mappa precalcolata: **rating = EWMA di partecipazione** (sale se
  spara, **decade** se tace) → **promozione** = pin del rating alto → **espulsione** =
  il decadimento libera lo slot da solo (gli Expert-prosa decadono nel CSS/JSON).
  Residenza **soft-backbone** (top-N per condivisione ≥T/20, ricalcolato → scorre con la
  fase). Attuatori: 0031-rotate (CUSUM residenza, bit-exact) + 0026-admit (CUSUM
  selezione, gated).
- **Gate.** **fit?** backbone-soft ≤ 394. **rende?/chiude?** ≥ S1 senza mappa offline.
  **t/s?** ≥ S1. **stabilità?** churn bounded, **bit-exactness dello strato-residenza
  preservata** (il lato 0031 resta byte-identico; solo il lato 0026 perturba). Manopola:
  velocità di decadimento (lenta = stantii, veloce = re-fetch).
- **Rischio-Ancora coperto da:** difese 1–2 + il lock soft (non hard-pin: a K12 il
  backbone ≈ tutta la cache → hard-pin troppo rigido).

### S3 — PHASE-DETECT A RUNTIME
- **Cosa.** Rileva la fase **da runtime** (entropia o contenuto: il `{` per PROMO;
  stato-nascosto/SPEX per il bordo prosa→codice) e **commuta automaticamente** il set
  residente+ammesso. È la «libreria di maschere-di-fase precalcolate + rilevamento-fase»
  (INTUITION_REGISTER §2, chiarimento routing-baked-in).
- **Gate.** **detector?** precision + **lead** (fire prima del collasso, ≥ +57 tok).
  **fit? rende? chiude? t/s?** end-to-end su prompt **multi-fase** reale (prosa→JSON→…).
  **⚠️** il segnale-fase è imperfetto (granularità specializzazione forse codice-vs-prosa
  più che CSS-vs-body) → misurare i falsi-trigger.

---

## 6. Parametrizzazione (sul reale, non su costanti inventate)

- **Confine-fit `B_fit`.** Ancorare al misurato: **~394 slot** (cache≈400, 3060,
  `vram_split_kv`). Ogni stadio verifica `set_residente ≤ B_fit`. Se il motore libera i
  ~0.3–0.5 GB di KV, `B_fit` sale marginalmente (~+45–75 slot) — non cambia il regime
  (K23 resta fuori).
- **K operativo `K*`.** **K-min-che-rende = 12** è il target: `K* = 12` domain-matched
  (backbone 354 ≤ 394). Non scendere sotto senza nuova misura; non salire a K23 (989,
  non entra).
- **Slack di ammissione.** `admit_add ≤ B_fit − backbone (~40 slot)`; se K-costante
  (swap), `admit_add = 0` e il vincolo è automatico.
- **Trigger.** slope-S1 ≥ +0.05 (banda stretta, lead da confermare) **oppure** evento
  strutturale (`{`). Cooldown anti-thrash ≥ 16 tok; cap churn opzionale.

---

## 7. Il caso PROMO — primo bersaglio del controller

**Perché è il più facile.** Dominio stretto, **2 sole fasi** (prosa → JSON), e una
**transizione osservabile a token singolo: `{`**. Elimina l'incognita più dura (il
rilevamento-fase rumoroso): il confine è *dato*. Quindi:

- **S0 su PROMO:** pin del backbone-prosa, poi pin del backbone-JSON — misura fit +
  bit-exactness + Δt/s con mask statiche per-fase.
- **S1 su PROMO:** ammetti il set-JSON **al `{`** (trigger strutturale, difesa 3). Gate
  primario: **la coda JSON chiude `}`** dove la statica-prosa non chiudeva.
- **S2/S3 su PROMO:** rating dinamico (gli Expert-prosa decadono dopo il `{`) e poi
  detect automatico del `{`. Se il controller chiude il PROMO end-to-end con fit+t/s,
  si generalizza al multi-fase (HTML: prosa→CSS→body→script) col detector di S3.

Il PROMO è il banco dove **una leva chiude entrambi**: al `{`, la stessa ammissione fa
entrare gli Expert giusti (velocità mantenuta, K-costante) **e** chiude la coda
(qualità). Se funziona qui, la tesi I4/I6 è dimostrata sul caso minimo.

---

## 8. Stato onesto / cosa NON è ancora provato (anti-regressione)

- **0031 smoke = PENDING** (niente CUDA locale): fit + bit-exactness + Δt/s da misurare
  su 3060 reale prima di dichiarare S0.
- **Lead-S1 = OPEN** (+57 tok / narrow ~200–225): banda stretta, lead-time da
  confermare; il trigger di S1/S3 ci si appoggia.
- **Rewind = REFUTATO-wide / UNTESTED-narrow**: la difesa 4 è ipotesi, non recupero
  provato; non citarla come chiusa.
- **K12-rende = L1, n=2**: campione modesto; la qualità a K12 (domain-match +
  ammissione-fase) resta *la partita aperta*.
- **~25 t/s è il tetto-di-fit (keep-8), non il numero K12**: sul 3060 il K12-realistico
  è ~3–4 t/s; i ~25 t/s provano che *la leva-fit esiste*, non il t/s finale del
  controller.
- **0026 quality-gated**: la sim recupera copertura (limite-1, traiettoria-sana); la
  qualità reale è gated sull'A/B live S3, non deducibile dalla sim.

---

*Costruito da INTUITION_REGISTER §1 (I1–I10) e §2 (controller di residenza viva);
numeri da CLAIMS_CURRENT.md. Manutenzione: quando uno stadio passa/fallisce un gate,
aggiorna qui + la riga in CLAIMS_CURRENT.md (quello vince).*
