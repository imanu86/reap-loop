# Build "ds4-flash-tutto-dentro" — spec operativo (Fase 0 scoping, verificato sul sorgente)

> Output workflow `wyzmw4l05` (5 agenti, sorgente ds4 reale in WSL Ubuntu-24.04). **Read-only**: il compile+test sul
> 3060 resta sulla workstation. Ogni file:riga è VERIFICATO sul tip `7a93e3b` (branch `spex/trace-ext`) — i numeri-riga
> nei vecchi doc erano quasi tutti STALE (corretti qui).

## Headline
**PRONTO senza codice C**: **LEVA 1 (CONFIG: leva-RAM + cache + prefill)** e **LEVA 3 (SPEX)** sono **già interamente cablate e committate** nel tree ds4 (`spex/trace-ext` @ `7a93e3b`, patch 0001-0007 applicate) → servono solo i toggle e i profili di lancio. **DA COSTRUIRE**: **LEVA 2 (REAP-loop)** è PARZIALE — l'attuatore v0 (0011) va rebasato, ma i pezzi che lo rendono efficace (v2-upsert, override router CPU, sensore-live, rewind wired) **non esistono in alcun commit** (vivevano sul pod 3090 spento) e vanno riscritti. **FUORI**: LEVA 4 (spec-dec unlock) = NO-GO; l'MTP nativo resta ma è **mutuamente esclusivo con lo streaming** (hard-error) → nessun "tutto-dentro" con MTP acceso.

## ⚠️ Correzioni ai doc (stale)
- SPEX **NON** è "da cablare": è **già in-tree**. L'hook non è `ds4.c:19415` (falso) ma **`20160`/`20229`**. I file `patches/ds4/ds4_spex_predict.{c,h}` sono **superseded — non usarli**.
- env leva-RAM è **`DS4_CUDA_KEEP_MODEL_PAGES`** (non `KEEP_MODEL_PAGES`).
- Il branch dspark reale è `dspark/unlock-streaming` (non `dspark/mtp-spec-dec`).
- MTP nativo: `ds4.c:27952`/`28083` (non 27135/27264).

## Strategia di branch
Creare **`ds4-flash-unified`** da **`spex/trace-ext` @ 7a93e3b** (ha già SPEX 0001-0007 + trace 0006/0007 → evita di riapplicare 7 patch). Cherry-pick sopra: **solo** `0011-reap-runtime-mask.patch` (rebasato a mano, call-site reale `20117` non `20174`). Da dspark: **niente**. **Tutti i toggle = ENV** (nessun `#ifdef`) → un **singolo binario** copre ogni combinazione. Default senza env = ds4 stock.

## Stato leve (verificato)
| Leva | Buildable | Cosa manca |
|---|---|---|
| **L1 CONFIG** (leva-RAM+cache+prefill) | ✅ **SÌ, zero codice** | nulla in codice; solo taratura runtime (reserve clamp, MiB/expert) col GPU |
| **L2 REAP-loop** | 🟢 **PIÙ FACILE del previsto** | ✅ `42086b1`: **0011-V2 UPSERT** (v2-upsert device-range + override router CPU-bias + mtime-poll runtime-mask) e **0012** (sensore S1 gate-mass) GIÀ COMMITTATI su `reap/k91-coding-vram` (la 0011-v1 era ROTTA → sostituita). Il rischioso è FATTO. Resta: **applicare 0011-V2+0012 a unified + loop-driver (sidecar che scrive DS4_REAP_MASK_FILE per schedule/respiro) + params esposti**; rewind KV opzionale |
| **L3 SPEX** | ✅ **SÌ, già cablato** | nulla; puntare `DS4_SPEX_MARKOV_FILE` al `.spex ds4flash_d2_nextlayer`; si accende **solo con `--ssd-streaming`** |
| **L4 spec-dec unlock** | ❌ **FUORI** (NO-GO) | MTP nativo resta ma **hard-exclusive con streaming** (`ds4.c:26500`) → non combinabile col warm |

## Ordine di implementazione (gate di smoke tra gli step)
- **STEP 0** — `git checkout -b ds4-flash-unified spex/trace-ext`; build; smoke stock = baseline.
- **STEP 1** — **L1 CONFIG (ZERO codice)**: 1a warm leva-RAM; 1b + cache-forzata (`--ssd-streaming`). Gate: parte e logga il budget atteso (no "cache disabled").
- **STEP 2** — **L3 SPEX (ZERO codice)**: `DS4_SPEX_MARKOV_FILE=.spex`, smoke con profile; **gate = output byte-identico ON vs OFF** (loading-not-gating). Richiede STEP 1b.
- **STEP 3** — **L2 REAP (il build)**: 3a rebase 0011 + mask statica; 3b v2-upsert (0% violazioni); 3c override router CPU `7415`; 3d sensore-live sidecar; 3e rewind wired + `DS4_REAP_LOOP`. Gate onesto: il loop **reattivo** NON recupera (C1) → il valore è la modalità **PREVENTIVA a scalini** (D4/D5).
- **STEP 4** — L4 MTP: solo isolato, mai combinato (esclusivo con streaming).

Dipendenze: L3 richiede L1b. L4 incompatibile con L1b/L2/L3. L1a (warm puro) ortogonale.

## Profili di lancio (un solo binario, env-gated)
Una-tantum Windows: `.wslconfig [wsl2] autoMemoryReclaim=disabled` → `wsl --shutdown`.

**P1a — warm leva-RAM**: `DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_NO_DIRECT_IO=1 ./ds4 --model <m> --prefill-chunk <N>`
**P1b — warm + cache**: + `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=<R>` `./ds4 ... --ssd-streaming --ssd-streaming-cache-experts <G>GB` (validare log "per expert = Z experts"; se "cache disabled: available <= reserve" → abbassa R/G).
**P2 — warm + cache + SPEX** (⭐ il "tutto-dentro" utile): + `DS4_SPEX_MARKOV_FILE=/path/ds4flash_d2_nextlayer.spex` `DS4_SPEX_PREFETCH_NEXT_LAYER=1`. OFF per A/B: `DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1`.
**P3 — REAP preventivo** (dopo STEP 3): + `DS4_REAP_MASK_FILE=<mask>` `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1` (loop: `DS4_REAP_LOOP=on`).
**P4 — MTP** (isolato, NO streaming): `./ds4 --model <m> --mtp <path> --mtp-draft <D>`. `--ssd-streaming` non usabile (hard-error).

**Profilo TARGET = P2** (warm+cache+SPEX, le 3 leve componibili). REAP preventivo e MTP si misurano a parte. Non esiste profilo streaming+MTP.

**Profilo DAILY (REAP-loop SEMPRE ON).** Quando la LEVA 2 è costruita, il daily = **warm + cache + REAP-loop** (+ SPEX se vince l'A/B). Il toggle env resta per la MISURA/isolamento (protocollo A/B/A), ma il default è acceso. ⚠️ NB la structured-extraction task = **draft CORTI** (non un sito da 2600 tok) → la mask va **imparata-una-volta da un batch/warm-up del dominio e RIUSATA tra i draft**, non ri-appresa per-draft (il test transfer cross-sessione dice se il riuso regge — probabile SÌ su dominio stretto stabile).

## REAP-loop — parametri da ESPORRE (design 2026-07-06, NON hardcodare)
Principio utente: *"non fissiamoci con parametri statici; in fase di creazione del sistema vanno esposti"* — corretto: è la differenza tra un ESPERIMENTO (una ricetta) e un SISTEMA (un framework tunabile). Il **"termometro" dell'utente = il ROUTER stesso** (nessun modello extra): misura lo **scostamento dal working-set iniziale**.
- **S1 = gate-mass sugli expert POTATI** = "quanto il router VUOLE ciò che ho tolto" → il termometro di drift diretto.
- **S2 = stabilità del routing** (Jaccard top-k tra token) = "sta scegliendo gli STESSI expert" → segnale di MANTIENI.
- (Caveat onesto: il router mostra cosa VUOLE, pre-mask; "volere un potato" **precede** ma non **garantisce** il collasso → per questo il test lead-time (C); combinare S1+S2 è più robusto di uno solo.)

**Knob da esporre (env/CLI)** — non hardcodarli, i task diversi vogliono valori diversi:
- Schedule: keep-levels (es. 50,23) + **numero di step**, warm-up length (~150), diretto-vs-gradini.
- Respiro: cap (ogni N tok), lunghezza (M tok), keep-level del respiro (40/64), re-learn on/off.
- Trigger adattivo: sorgente-segnale (S1/S2/S3), **soglia** di drift, isteresi → respiro a soglia, non a orologio.
Esporli = ciò che permette alla **community** di estenderlo (ethos ds4-open di antirez) e al paper di essere un *framework caratterizzato*, non una singola ricetta hardcoded.

**End-state (la vera visione, utente 2026-07-06)**: lo schedule NON è una sequenza di K fissi (K0→K91→K20→K91 = numeri a mano) ma un **controller closed-loop**: `keep-level = f(termometro-drift)`. Drift basso → tieni/stringi; drift che sale → allarga **proporzionalmente** (drift piccolo = respiro leggero; grande = respiro profondo). È un problema di **controllo** (termostato con isteresi). I schedule fissi della serie-D sono i **datapoint che mappano la curva di risposta**; l'algoritmo li generalizza. Caveat onesti: (1) serve **isteresi/damping** (evitare oscillazioni respiro↔stringi); (2) il termometro deve **PRECEDERE** il collasso (test C) o il controller reagisce tardi; (3) è oltre la serie-D (schedule fissi) → il test #1 (respiro adattivo a soglia) è il **primo gradino**, il controller continuo è l'orizzonte-ricerca (e un pezzo di paper forte: *"dynamic MoE pruning come problema di controllo, keep-level pilotato da un sensore di drift del routing"*).

## STEP 3g — Control-endpoint per la GUI (fase successiva, dopo che il loop gira)
**Frontend**: ds4 è già **OpenAI-compat** (`:8000`, `/v1/chat/completions`) → per CHATTARE basta puntarci una GUI esistente (OpenWebUI / open-code), zero interfaccia da scrivere. Per l'uso reale su un task di estrazione strutturata → un client OpenAI-compat che punta al ds4-server :8000.
**Il control-endpoint aggiunge SOLO il tuning-LIVE dei knob**: espone i parametri REAP (keep-level, schedule, respiro cap/len/keep, soglia drift, isteresi, trigger-source) anche via un endpoint runtime (oltre che env-al-lancio) → una GUI/dashboard può leggerli/settarli **senza riavviare** e mostrare i **termometri** (S1/S2, hit-rate, keep-level nel tempo). È lo STEP 3g, dopo il loop base. È ciò che rende "i parametri implementati nella GUI".

## Protocollo di misura
- **Rumore ±50%** → ogni numero = **mediana-di-3** run identici (mai media). Spread >50% → +2 run.
- **Primaria**: t/s aggregate. **Secondaria**: hit-rate robusto (REAP: token dentro il working-set mascherato; SPEX: prefetch L+1 che coincidono coi selezionati veri, `DS4_SPEX_PREFETCH_PROFILE=1`).
- **Baseline P0 (stock) mediana-3 PRIMA di ogni leva**; ogni leva = **delta su P0**.
- **A/B/A a env singola**: accendi UNA leva, misura, spegni, ri-baseline (esclude drift termico/OS). Combina solo DOPO aver misurato ciascuna da sola (espone gli antagonismi).
- **SPEX correttezza-neutrale**: verificare byte-identità output ON vs OFF (se differisce = BUG).
- Comandi WSL sempre `wsl -d Ubuntu-24.04` esplicito (2 distro).

## Rischi residui (solo compile+GPU confermano)
- **L1**: reserve clamp a total/2 (su 12GB R>6 troncato, cache si spegne se free<=reserve); `KEEP_MODEL_PAGES` → rischio OOM/swap WSL (serve `autoMemoryReclaim=disabled` + RAM); verificare che `MADV/FADV_DONTNEED` sia compilato-in (altrimenti KEEP è no-op).
- **L2 (i più seri)**: v2-upsert è descritta **solo a parole** (0 sorgente), il device-range host-UVA può comportarsi diverso dal 3090; il rewind coarse tronca `checkpoint.len` ma non ricalcola stati derivati → verificare decode pulito; **il loop reattivo NON recupera** (solo il preventivo).
- **L3**: SPEX non parte senza `--ssd-streaming` ON (se la cache non si accende, SPEX è silenziosamente OFF); il recall offline (0.68-0.92) va riprodotto a runtime, altrimenti il .spex va rivalutato.
- **Antagonismi da non combinare alla cieca**: warm ↔ REAP-rewind (rigenera, spreca il warm); warm/cache/SPEX ↔ MTP (hard-error 26500).
