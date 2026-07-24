# DS4 — Analisi delle leve POST 2026-07-24 (foto "dopo oggi")

*Ri-valutazione sistematica di tutte le leve dell'`DS4_LEVER_ATLAS_20260724.md` alla luce dei finding di oggi. L'atlante è la foto "prima" (mappa neutra del codice); questo è la foto "dopo": ogni leva riceve un verdetto operativo basato su cosa abbiamo misurato/verificato oggi. Punti load-bearing ri-verificati sul sorgente `D:/ds4_work/wt-g73-open/{ds4_cuda.cu,ds4.c}`, non solo riportati dall'atlante.*

## Legenda verdetti

| Tag | Significato |
|---|---|
| `[CONFERMATA]` | I finding di oggi confermano il ruolo/valore attuale della leva |
| `[RIBALTATA]` | Oggi cambia il verdetto rispetto all'atlante (segno o utilità invertiti) |
| `[MORTA]` | Nessun effetto misurabile o percorso rotto/irraggiungibile nella config attiva |
| `[MAI-TESTATA-PROMETTENTE]` | Mai tirata ma con ipotesi d'impatto reale — candidata |
| `[CONFLITTO]` | Valore/combinazione incoerente nel preset o coppia contraddittoria |
| `[RUMORE]` | Provenance / logging / fixture / dev — non muove il t/s |

**Convenzione "valore":** `preset=` = impostata in `tests/g73_open.env.ps1`; `def=` = non nel preset, vale il default del sorgente; `rm=` = il preset la *rimuove* esplicitamente dall'ambiente (lista `$g73ForbiddenQuantServing`). Le famiglie puramente combinatorie sono compresse in una riga con conteggio.

---

## 1. Tabella maestra per sottosistema

### 1.1 Serving-path / G73 (la tensione centrale di oggi)

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_G73_OPEN` | preset=1 | =1 commuta il dispatch routed-MoE a `routed_moe_launch_impl<true>` (exact-transient-escape, cu:39696-39698) **e** auto-abilita l'SSD-wrap (cu:1516) | **`[RIBALTATA]`** — su IQ2 puro =1 fa 0.62 t/s, =0 fa 4.85 (7.8×). Il costo sta nel dispatch `<true>`, non nel wrap. Su IQ2 **spegnerla**; serve solo se il serving Q1 richiede davvero il dispatch exact-transient |
| `DS4_G73_ALLOW_Q1_SIDECAR` | def=unset | Escape hatch (cu:1477-1495): esenta le 6 var di *serving* Q1 dal rifiuto ermetico di G73; arena/promozione restano vietate | `[MAI-TESTATA-PROMETTENTE]` — unico modo di servire sidecar Q1 **restando** in G73_OPEN. Zero occorrenze in script/test |
| `DS4_G73_PAGEABLE_OVERFLOW_GB` | preset=14 | 14 GiB di arena esperti pageable (VirtualAlloc, mai VirtualLock) interlacciati fra i pinned | `[CONFERMATA]` — attiva in produzione, mai isolata; ~32% accessi su pageable dove `cudaMemcpyAsync` non è async. Sospetto costo (ipotesi G della caccia) |
| `DS4_G73_CONSERVATION_STRICT` | def=0 | Flag di conservazione stretta del path G73 | `[RUMORE]` |
| `DS4_G73_OPEN_SELFTEST` | def=0 | Selftest di avvio del path open | `[RUMORE]` |
| *(hardcoded)* dispatch `<true>` vs `<false>` | cu:39696 | Non è una env: `g_cuda_g73_open_enabled` sceglie il template. **La vera leva del t/s IQ2 è questa, pilotata solo da G73_OPEN** | `[CONFERMATA]` costante-che-dovrebbe-essere-leva |

### 1.2 Cache VRAM / residenza esperti

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_CUDA_STREAMING_EXPERT_CACHE_N` | preset=320 | Slot esperti VRAM *richiesti* (M in "N/M"); prerequisito dell'intero sottosistema | `[CONFERMATA]` capacity-wall — solo ~229 entrano (cap=`(free−reserve)/per_expert`, cu:31244); 400 rompe il seed |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` | preset=0.125 | **L'UNICO env che tocca la cap reale** (`cuda_moe_expert_cache_reserve_bytes`, cu:25948→31240), default 0.5 | `[CONFERMATA]` — verificato: è l'unica manopola su cu:31240 |
| `DS4_CUDA_STREAM_RESERVE_MB` | preset=1024 | Reserve dell'allocatore di range *streamati* (`cuda_model_stream_reserve_bytes`, cu:6190), governa l'eviction in `cuda_model_stream_alloc` | `[CONFERMATA]` — sottosistema **diverso** dalla cap cache; tocca free_b indirettamente (ipotesi D) |
| `DS4_CUDA_STREAM_RUNTIME_RESERVE_MB` | def=unset | Come sopra ma dopo che la cache è pronta (cu:6194) | `[CONFLITTO]`/`[MORTA]` per la cap — manopola **sbagliata** per la capacity cache (non tocca cu:31240) |
| `DS4_CUDA_MOE_GPU_RESIDENT_ROUTES` | preset=1 | Abilita il resolver GPU-resident (path veloce, cu:37246) | `[CONFERMATA]` prerequisito del path caldo |
| `DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC` | preset=1 | Off ⇒ `cudaStreamSynchronize(0)` per layer per token (40/token) | `[CONFERMATA]` corretta a 1; il tempo migra in `route_ready_wait_seconds` (ipotesi E) |
| `DS4_CUDA_MOE_CACHE_POLICY` | preset=`lru` | Solo `layer-top1` è riconosciuto (cu:25931-25933); ogni altro valore = "non layer-top1" = path normale | **`[CONFLITTO]`** — `lru` è **valore invalido**, no-op (identico a unset). `layer-top1` invece **spegnerebbe** il resolver GPU-resident |
| `DS4_CUDA_MOE_SPLIT_FUSED` | preset=1 | Overlap split-fused attivo | `[CONFERMATA]` |
| `DS4_CUDA_MOE_SPLIT_HIT_MISS` | preset=0 | Legacy split, mutex con fused | `[CONFERMATA]` (azzerata anche da G73) |
| `DS4_CUDA_MOE_DIRECT_CACHE_HITS` | def=off | Su layer all-hit pubblica gli slot residenti, evita ~1.6 GiB/token D2D slot→compact | `[MAI-TESTATA-PROMETTENTE]` — fragile (basta 1 miss e il layer ripaga tutto); misurare prima la frazione all-hit |
| `DS4_CUDA_EXPERT_CACHE_GROW` | def=off | Rimanda la scelta di N a decode-start | `[MAI-TESTATA-PROMETTENTE]` — con RELEASE_SCRATCH è l'unico modo di *scegliere* N invece di subirlo |
| `DS4_CUDA_RELEASE_PREFILL_SCRATCH` | def=off | Libera `g_cuda_tmp` a fine prefill (i ~2 GB `batch_*` restano) | `[MAI-TESTATA-PROMETTENTE]` (coppia con GROW) |
| `DS4_Q1_VRAM_LRU_SLOTS` / `_RESERVE_MB` | def=600 / 512 | Seconda cache LRU VRAM, **attiva di default**, zero occorrenze in test | `[CONFERMATA]` — sottrae ~512 MiB silenziosamente; sotto revisione |
| `DS4_CUDA_MOE_NO_SELECTED_LOAD` | def=off | Kill-switch: annulla cache streaming, overlap, Q1_0_SELECTED_LOAD | `[MORTA]`/kill — **non accendere** |
| `DS4_CUDA_MOE_NO_DECODE_LUT_GATE`, `_NO_DIRECT_DOWN_SUM6` | def=off | A/B kernel che **rifiutano** il resolver GPU-resident (cu:37259-37260) | `[MORTA]`/kill — non accendere sotto preset |
| `DS4_CUDA_MOE_OVERLAP_SHARED(_FULL)`, `_MIXED_DIRECT`, `_ROUTE_PACKED_COPY` | def=off | Auto-annullate da `CACHE_N≠0`; PACKED_COPY entra nella firma di config (release+ricalcolo N) | `[MORTA]` nella config attiva |
| ~26 var geometria kernel `DS4_CUDA_MOE_{TILE4,GATE_ROW*,DOWN_ROW*,NO_*,ATOMIC_DOWN,…}` | def=vari | A/B micro-kernel, `getenv` non cacheate (~1000/token); `GATE_ROW256/128`, `DOWN_ROW256/128/64` sono "leve a metà" | `[RUMORE]`/dev — non muovono il collo |

### 1.3 KV & attention

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_CUDA_KV_MANAGED` | def=off | Unico `cudaMemAdvise` del runtime: KV host-backed, letta in place, **zero migrazione** (richiede la stringa esatta `"1"` e `KV_STAGED_RING` non impostata) | `[MAI-TESTATA-PROMETTENTE]` — host-backed by design; il segno è empirico (elimina migrazione ma mette la KV su PCIe), mai misurato |
| `DS4_CUDA_KV_STAGED_RING` | def=off | Stagea la KV in host pinned al decode-start (libera VRAM → cache può tornare a 320); sticky-off irreversibile al 1° path multi-token | `[MAI-TESTATA-PROMETTENTE]` **con BUG** — illegal-memory sul restore in decode a big-ctx (in fix da Codex). `RING_BYTES=4MiB`/`SLOTS=2` sono **costanti di compilazione**, non env |
| `DS4_CUDA_NO_WINDOW_ATTENTION` / `_WINDOW_ATTENTION` | def=off | Il `NO_` è testato per primo e vince; senza kernel online e `n_comp>7936` il decode fallisce | `[RUMORE]`/trappola |
| `DS4_CUDA_NO_TOPK2048`, `_NO_TOPK_CHUNKED` | def=off | Trappole: chiudono 3 rami → insertion sort a 1 thread | `[MORTA]`/trappola — non toccare |
| `DS4_CUDA_NO_TOPK1024` | def=off | Sposta il kernel top-k senza chiudere rami (unico A/B sicuro) | `[MAI-TESTATA-PROMETTENTE]` (solo se ipotesi B sopravvive) |
| `DS4_CUDA_NO_INDEXED_HEADS8`, `_INDEXED_TWOPASS`, `_ATTENTION_OUTPUT_*`, 10× fusion-disable | def=off | Irraggiungibili a decode (gate `n_tokens>1`) o subordinati a `NO_Q8_F16_CACHE`/`--quality` | `[MORTA]` a decode |
| *(hardcoded)* `DS4_N_INDEXER_TOP_K` = 512 | ds4.c:124 | Costante confrontata con `n_comp`; il getter (`…indexer_top_k(g)`) accetta `g` e lo scarta: stub già predisposto | `[MAI-TESTATA-PROMETTENTE]` costante-che-dovrebbe-essere-leva (3 righe per renderla env; falsifica l'ipotesi B) |

### 1.4 Prefill / seed VRAM

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_METAL_PREFILL_CHUNK` | def=`min(ctx,2048)` | Fissa `prefill_cap` (ds4.c:7289-7307), dimensiona 38 tensori `batch_*` managed (~0.97 MiB/token cap) | **`[RIBALTATA]`** — =768 fixa il collo big-ctx (0.56→3.92 a ctx8192); default 2048 sovradimensiona di ~1.3 GB managed. Effetto collaterale: `raw_cap` scala, snapshot di sessione invalidati |
| `DS4_CUDA_PREFILL_MASS_OBSERVE` | preset=1 | Osservatore di massa statico, riporta il pin-target a request-end (non mid-decode) | `[CONFERMATA]` — statico by design |
| `DS4_CUDA_PREFILL_MASS_WRAP` | preset=1 | Ripubblica il set residente dopo l'osservazione | `[CONFERMATA]` — con COMPOSE soddisfa il gate cu:37253-37256 |
| `DS4_CUDA_PREFILL_TIER_COMPOSE` | preset=1 | Compone il tier di prefill; prerequisito del wrap-published | `[CONFERMATA]` — fallimento ⇒ arena avvelenata a vita (`submissions_blocked=1`) |
| `DS4_CUDA_PREFILL_TIER_ROUTER` | preset=`open` | Router aperto/chiuso del tier di prefill | `[CONFERMATA]` |
| `DS4_CUDA_PREFILL_TIER_RESERVE_SLOTS` | preset=128 | Slot riservati al pool open-router | `[CONFERMATA]` — ignorato con router `closed` (qui è `open`, ok) |
| `DS4_CUDA_PREFILL_VRAM_SEED_TOTAL` / `_PER_LAYER` / `_FLOOR_PER_LAYER` | def=0 | Semina esperti in cache VRAM a fine-prefill (partenza calda); mutuamente esclusive (cu:25855-25915) | `[CONFERMATA]` **critico** — capacity-bound: se non entra, contratto invalido ⇒ **l'intera cache esperti diventa NULL** (non solo il seed). Non nel preset attivo (il seeding va via PREFILL_MASS) |
| `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS` | def=4 | =0 elimina un `cudaDeviceSynchronize` pieno per token (dopo layer 4) | `[MAI-TESTATA-PROMETTENTE]` — il 4 è giustificato da una misura su **Metal**, mai su CUDA |

### 1.5 Tiering / promozione g133

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_EXPERT_TIERING` | preset=`enforce` | Master switch; in `enforce` il selected-load azzera la cache legacy (`[moecache]`=0) | `[CONFERMATA]` |
| `DS4_EXPERT_TIER_POLICY` | preset=`mass-lfru` | `second-touch` (default) azzererebbe clock/replacement/adaptive | `[CONFERMATA]` |
| `DS4_EXPERT_TIER_CLOCK_CALLS` | preset=430 | ~10.75 token per giro di clock | `[CONFERMATA]` |
| `DS4_EXPERT_TIER_REPLACEMENT_BUDGET` | preset=32 | Muove `replacement_budget` (=32) | `[CONFERMATA]` — **NON** tocca `g133_promote_budget` che gate il decode |
| `DS4_EXPERT_TIER_MIN_FREQUENCY` | preset=3 | Frequenza minima per residenza | **`[MORTA]`** — saltata dal ramo G133 (pur essendo loggata come attiva) |
| `DS4_EXPERT_TIER_HYSTERESIS` | preset=1.25 | Isteresi anti-thrash | **`[MORTA]`** — saltata dal ramo G133 |
| `DS4_EXPERT_TIER_ADAPTIVE_BUDGET` +`_MIN`/`_MAX`/`_STEP`/`_PRESSURE_THRESHOLD` | def=off | ~120 righe di controllo adattivo del `replacement_budget` | **`[RIBALTATA]`** — oggi: muove `replacement_budget` (=32), **non** il `g133_promote_budget` che gate il decode. Non cura il collo. Config incoerente ⇒ disattivazione silenziosa |
| `DS4_G133_TIER` | preset=1 | Attiva il ranking G133; impone ctx ≤ 1 048 575 | `[CONFERMATA]` |
| `DS4_G133_KNOCK_X` | preset=3 | Streak consecutivo per **candidatura a promozione** (cu:26397) | **`[MORTA]`** — nessun effetto misurabile oggi |
| `DS4_G133_KNOCK_Y` | preset=5 | `knock_x+knock_y` = soglia protezione-demozione (cu:26410) | **`[MORTA]`** — nessun effetto misurabile oggi |
| `DS4_G133_PROMOTE_BUDGET` | preset=8 | Budget globale di promozioni per posizione (cu:26540/26565) | **`[MORTA]`** — alzarlo NON aumenta le promozioni (capacity-bound); >62 **rompe il seed**. Il gate vero è knock-streak, ma anche knock è morto |
| `DS4_G133_DECAY` | preset=0.98 | Decadimento del calore | `[RUMORE]` — con reset a ogni richiesta degenera a inizio richiesta |
| `DS4_G133_SEED_DYNAMIC` | preset=1 | Semina lo streak; **evapora alla 1ª posizione** (sopravvive solo il calore) | `[CONFERMATA]` |
| `DS4_G133_ROTATOR_IO_TIMEOUT_S` | preset=0.05 | Timeout I/O del rotatore | `[CONFERMATA]` |
| `DS4_G133_TRANSIENT_IO_TIMEOUT_S` | preset=0.25 | Timeout transient; **vale solo con staged-KV**, altrimenti 5 s hardcoded | `[CONFERMATA]` — inerte senza KV_STAGED_RING |

### 1.6 REAP-mass dinamico (maschera del decode)

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_CUDA_REAP_MASS_OBSERVE` | def=off (rimossa `REAP_MASK_FILE`) | Osserva finestra token → ripubblica pin-target ogni GROW_INTERVAL | `[MAI-TESTATA-PROMETTENTE]` **richiede rebuild** — `g_reap_mass_observer.enabled` **rifiuta** il resolver GPU-resident (cu:37257) ⇒ tutto fallback, lento. Comporlo col path VRAM = modifica sorgente |
| `DS4_CUDA_REAP_MASS_WRAP` | def=off | Ripubblica la maschera osservata | idem `[MAI-TESTATA-PROMETTENTE]`/rebuild |
| `DS4_CUDA_REAP_MASS_WINDOW` | def=16 | Finestra di osservazione | idem (invalido ⇒ observer off, muto) |
| `DS4_CUDA_REAP_MASS_GROW_INTERVAL` | def=4 | Cadenza di ripubblicazione | idem |
| `DS4_CUDA_REAP_MASS_HYSTERESIS` | def=1.25 | Isteresi della maschera dinamica | idem |

### 1.7 Sidecar Q1 / IQ1

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_Q1_0_EXPERT_SIDECAR` (+`_BYTES`,`_SHA256`) | rm | Path GGUF sidecar Q1_0 (interruttore generale). SHA solo formato, mai calcolato | `[MAI-TESTATA-PROMETTENTE]` — serving Q1; richiede `G73_ALLOW_Q1_SIDECAR=1` per coesistere con G73_OPEN |
| `DS4_Q1_0_SELECTED_LOAD` | rm | Carica solo gli esperti selezionati dal sidecar. Semantica incoerente (`"1"` in ds4.c, non-NULL in cu) | `[MAI-TESTATA-PROMETTENTE]` |
| `DS4_Q1_0_LAYER_FIRST` / `_LAST` | rm (def 0/42) | Range layer serviti dal sidecar | `[CONFERMATA]` geometria serving |
| `DS4_Q1_0_PROMOTION_SSD_WRAP` | rm (forzata 1 da G73) | Abilita l'SSD-wrap **anche senza G73_OPEN** (cu:1517) | **`[MAI-TESTATA-PROMETTENTE]`** — chiave del disaccoppiamento: wrap à la carte **senza** il dispatch `<true>` lento (vedi Top-10 #2) |
| `DS4_Q1_0_SSD_WRAP_{QUEUE_CAPACITY,PREFILL_WAVE_*,DECODE_WAVE_*,MAX_AGE_CALLS}` | def | Tuning del ring SSD-wrap | `[RUMORE]` — decorativi: il ring reale è 2+2 slot hardcoded (cu:1508-1512); zero occorrenze nel repo |
| `DS4_Q1_0_IQ2_PINNED_{GIB,MIN_TOUCHES,DEADLINE_CALLS,MIN_MASS,MIN_WEIGHT}` | rm | Pinning IQ2 | `[MORTA]` — calcolate/loggate/scartate sotto G73; `_GIB` mai letta |
| Matrice Q1_0 a 7 booleani (`RESIDENT_ARENA`,`DUAL_ARENA`,…,`DYNAMIC_PROMOTION`) | rm | 10 combinazioni legali su 128; una morta a runtime, una no-op | `[MORTA]`/vietate sotto G73 |
| Famiglia `DS4_IQ1_S_*`, `DS4_IQ1_MIXED_*`, `DS4_NESTED_RESIDUAL_*` | rm/off | Sidecar IQ1 e nested-residual | `[MORTA]` — inerti nella config IQ2 pura; `IQ1_MIXED_GPU_PLAN` sarebbe l'unica utile (rimuove 2 D2H/layer) ma vietata da G73 |

### 1.8 Arena host / wrap

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_CUDA_DYNAMIC_ARENA_GB` | preset=30 | 4551 slot da 7 077 888 B; disattiva la registrazione zero-copy della finestra host | `[CONFERMATA]` transport frozen |
| `DS4_CUDA_ARENA_WRAP_SCHEDULE` | preset=`source-parts` | Prerequisito di FILE_QD/PART_PROFILE/UNLOCK/TRUST_CHECKSUM | `[CONFERMATA]` |
| `DS4_CUDA_ARENA_WRAP_TRUST_WORKER_CHECKSUM` | preset=1 | Fida del checksum worker (decade muto se schedule≠source-parts) | `[CONFERMATA]` |
| `DS4_CUDA_ARENA_WRAP_UNLOCK_SOURCE_RANGES` | preset=1 | Sblocca i range sorgente durante il wrap | `[CONFERMATA]` — incompatibile con TRIM_BETWEEN_PHASES |
| `DS4_CUDA_ARENA_WRAP_UNLOCK_WAVE_GIB` | preset=4 | Granularità di sblocco | `[CONFERMATA]` |
| `DS4_REAP_PREFETCH_THREADS` | preset=8 | Queue depth I/O reale nel percorso benedetto | `[CONFERMATA]` |
| `DS4_CUDA_EMBED_ROW_STAGING` | preset=1 | Staging per riga degli embedding | `[CONFERMATA]`/note |
| `DS4_CUDA_DYNAMIC_ARENA_MIN_AVAILABLE_GIB` | def=0 | 0 = nessun guard host-capacity; su Linux qualsiasi >0 **disabilita l'arena** | `[CONFLITTO]` latente cross-OS |
| `DS4_CUDA_ARENA_WRAP_{SEQUENTIAL_FILE,RANDOM_FILE,FILE_QD,SEQUENTIAL_WORKERS,PART_PROFILE,TRIM_BETWEEN_PHASES,…}` | def | Tuning/profiling del wrap, molte coppie mutex | `[RUMORE]`/dev — vedi coppie contraddittorie §3 |
| `DS4_CUDA_{COPY_MODEL,DIRECT_MODEL,WEIGHT_CACHE,KEEP_MODEL_PAGES,NO_DIRECT_IO,…}` | def | 6 strategie di residenza mutex + no-op su Windows | `[RUMORE]`/no-op Win |

### 1.9 I/O SSD

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_CUDA_MOE_IO_QD` | def=1 | Queue depth ReadFile overlapped per gli span esperti | `[MAI-TESTATA-PROMETTENTE]` — a 1 il path overlapped (`cuda_moe_fill_spans_overlapped`, ~110 righe, Windows-only) **non gira mai** (cu:34014/35834). Clampata a 4. Degrada muto se fallisce |
| `DS4_G132_CPU_LANE` / `_MAX` | def=off/3 | Lane CPU (`cores-2` worker) per dequant IQ2 in parallelo alla GPU | `[MAI-TESTATA-PROMETTENTE]` — infra completa (`cuda_g132_cpu_lane_init`, cu:40115), **mai esercitata**; su macchina overhead-bound con GPU al 29-39% |
| *(hardcoded)* chunk 1 MiB, `CreateEventW`/pread, FNV per ammissione | os_file.c / cu:29520 | Granularità I/O e checksum mai riletto | `[RUMORE]` — costanti, non env |

### 1.10 MTP / speculativo

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_MTP_BATCH_VERIFY` (+ `think:false`, `--mtp`) | def=off | Verify batch a `draft_n==2` = l'ammortizzatore dei load-esperti (DeepSpec) | `[MAI-TESTATA-PROMETTENTE]` — richiede di spegnere il think-mode; due leve insieme o non parte |
| `--mtp-draft` (1), `--mtp-margin` (3.0), `DS4_MTP_{PROBE,STRICT,SPEC_LOG,…}` | def | Ramo speculativo | `[MORTA]` col think-mode ON di default (temperature=1 annulla `temperature<=0`) |
| `DS4_MTP_SPEC_DISABLE` | def | Testata per sola presenza: `=0` la **attiva** | `[CONFLITTO]` semantica invertita |

### 1.11 Server / sessione / geometria contesto

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `think` / `model` / `reasoning_effort` (per richiesta) | ON di default | Sovrascrive temperature/top_k/top_p/min_p; via temperature=1 uccide l'MTP | `[CONFERMATA]` — rende morto ogni tuning di sampling + speculativo |
| `-c/--ctx` | server 32768 | Governa prefill_cap/raw_cap/comp_cap; con G133 rifiutato oltre 2²⁰−1 | `[CONFERMATA]` |
| `--quality` | off | Spegne 6 leve (cache F16, indexer WMMA, window crossover, cuBLAS out_a, TF32) + forza MTP strict | `[CONFERMATA]` — chi misura con `--quality` misura un altro runtime; **tenerla off** |
| `DS4_METAL_GRAPH_RAW_CAP`, `_RESUME_PREFILL_MIN`, `_GPU_BATCH_EMBED_MIN`, `--kv-*` | def | Geometria/KV-disk, nessuno script li usa | `[RUMORE]`/inattive |
| `DS4_G130_U1_ATTRIBUTION` | preset=1 | Attribuzione span g130 (mixed-Q1, validata solo `-c256/-n64`) | `[RUMORE]` — non spiega i 1776 ms per costruzione |

### 1.12 Provenance / logging / sonde

| Leva | Valore | Cosa fa | Verdetto |
|---|---|---|---|
| `DS4_METAL_{INDEXER_STAGE,DECODE_STAGE,GRAPH_TOKEN,LAYER_STAGE,Q_STAGE}_PROFILE` | def=off | **Uniche** sonde che attribuiscono il decode per stadio; ogni boundary sincronizza | `[MAI-TESTATA-PROMETTENTE]` — per **attribuire** i 1776 ms (non per misurare t/s). Costo di prova: zero |
| `DS4_CUDA_WEIGHT_CACHE_VERBOSE` | def=off | Unico contatore `CUDA evicted streamed range` (ipotesi D) | `[MAI-TESTATA-PROMETTENTE]` — a cu:4196 cambia anche un valore di ritorno: usare per contare, non cronometrare |
| `DS4_MODEL_SHA256`, `*_SIDECAR_SHA256` | preset=set / rm | Validati solo come 64 hex; **nessun hash calcolato**; abilita l'SSD-wrap gate | `[RUMORE]` — provenance, non verifica |
| `DS4_CUDA_MOE_PROFILE`, `DS4_REQUEST_PHASE_TRACE`, 24× `DUMP/TRACE`, `DS4_ORACLE_LOGITS`, `DS4_TOKEN_TIMING`, `DS4_EXPERT_RECOVERY_TRACE` | def=off | Dump/trace/telemetria | `[RUMORE]` — alcune **cambiano il sistema** (MOE_PROFILE spegne split/fused; dump spegne prefill-waves; PHASE_TRACE in forbidden-list g130) |
| `DS4_THREADS`, `DS4_LOCK_FILE` | auto / `%TEMP%\ds4.lock` | Pool CPU / lock istanza | `[RUMORE]` |

---

## 2. Top 10 leve ancora da tirare (ranked)

Priorità = (impatto atteso sul t/s o sull'attribuzione) ÷ (costo). "rebuild" = richiede modifica sorgente + ricompilazione.

| # | Leva / combo | Ipotesi d'impatto | Costo | Rebuild? |
|---|---|---|---|---|
| **1** | `DS4_G73_OPEN=0` su config IQ2 | **7.8×** documentato (0.62→4.85 t/s): il dispatch `<true>` exact-transient è il costo. Prima mossa se il target è IQ2, non Q1 | 1 run | no |
| **2** | `DS4_Q1_0_PROMOTION_SSD_WRAP=1` **con** `G73_OPEN=0` + sidecar Q1 (`EXPERT_SIDECAR`+`SELECTED_LOAD=1`+`LAYER_FIRST/LAST`) | **Disaccoppia** serving-Q1 dal dispatch lento: cu:1517 abilita il wrap senza flippare il template `<true>` (cu:39697). Se Q1 serve e resta veloce, l'intera tensione G73 si dissolve. Falsifica "Q1 richiede l'exact-transient" | 1-2 run | no (se il serving Q1 non esige `<true>`) |
| **3** | `DS4_METAL_PREFILL_CHUNK=768` @ ctx8192 | Elimina ~1.3 GB di `batch_*` managed a geometria KV invariata; fixa il collo big-ctx (0.56→3.92). **Controllo obbligatorio**: `=2048` @ ctx768 deve crollare a ~0.56 | 2 run | no |
| **4** | Le 5 sonde `DS4_METAL_*_STAGE_PROFILE` + `DS4_TOKEN_TIMING=1` @ ctx8192 | Attribuiscono i 1776 ms non-attribuiti (encode/execute/read per stadio; curva piatta=A vs crescente=B). Strumentazione già presente, **mai accesa** | 1 run | no |
| **5** | `DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` @ ctx768 vs ctx8192 | Conta le eviction VRAM (ipotesi D): se 0→centinaia, causa trovata; contromisura = abbassare `STREAM_RESERVE_MB` | 2 run | no |
| **6** | `DS4_CUDA_MOE_IO_QD=2` (o 4) | Accende ~110 righe di I/O overlapped **mai eseguite**; verificare nel log che parta (degrada muto) | 1 run | no |
| **7** | `DS4_CUDA_KV_MANAGED=1` (con `KV_STAGED_RING` **non** impostata) | Unico `cudaMemAdvise`: elimina la migrazione managed della KV. Segno incerto (KV su PCIe). Farla **dopo** #3-#5 per non contaminare l'attribuzione | 1 run | no |
| **8** | `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` | Rimuove un `cudaDeviceSynchronize` pieno/token; il 4 è tarato su Metal, mai su CUDA | 1 run | no |
| **9** | `DS4_CUDA_MOE_DIRECT_CACHE_HITS=1` | −~1.6 GiB/token D2D sui layer all-hit; misurare prima la frazione all-hit con `MOE_CACHE_STATS` | 1 run | no |
| **10** | `DS4_MTP_BATCH_VERIFY` + `think:false` + `--mtp` | Ammortizzatore load-esperti (DeepSpec): verify batch a draft=2 su macchina overhead-bound | 1 run | no |

**Fuori podio (rebuild o segno debole):** rendere `DS4_N_INDEXER_TOP_K` una env (3 righe su stub cu:10999 già predisposto — test diretto di B); `DS4_G132_CPU_LANE=1` (infra mai esercitata, compete coi worker — provare da solo); `DS4_CUDA_EXPERT_CACHE_GROW+RELEASE_PREFILL_SCRATCH` (scegliere N invece di subirlo). **Da NON tirare:** reap-mass dinamico — mutex hardcoded col path VRAM (cu:37257), comporlo è un rebuild e in ogni caso rende lento oggi.

---

## 3. Conflitti / incongruenze nel preset da sistemare

| # | Incongruenza | Verità sul sorgente | Azione |
|---|---|---|---|
| 1 | `DS4_CUDA_MOE_CACHE_POLICY = lru` | `lru` **non è un valore riconosciuto**: solo `layer-top1` (cu:25931-25933). Ogni altro valore = "non layer-top1" = path normale ⇒ **no-op identico a unset** | **Rimuovere** la riga (fuorviante). NON metterla a `layer-top1`: quello spegne il resolver GPU-resident |
| 2 | `DS4_G73_OPEN = 1` su config IQ2 | Flippa il dispatch a `<true>` (cu:39697) = decode lento; su IQ2 il serving Q1 non è nemmeno usato (nessun sidecar bound, `ALLOW_Q1_SIDECAR` non set) | **Spegnere** per il lavoro IQ2. Tenere =1 solo quando si serve davvero Q1 e si è verificato che serva il dispatch exact-transient (vedi Top-10 #2) |
| 3 | `NO_Q8_F16_CACHE=1` vs `Q8_F16_CACHE_RESERVE_MB=4096` (nell'harness, non nel preset) | **Non è VRAM sprecata**: `cuda_q8_f16_cache_reserve_bytes` è consumata solo a cu:3104, irraggiungibile perché `cuda_q8_f16_cache_allowed` ritorna 0 a cu:3135 quando NO_Q8 è set. La reserve è **inerte** | Chiarito: nessuno spreco. **Rimuovere** comunque `RESERVE_MB` dall'harness per igiene (config morta) |
| 4 | `EXPERT_TIER_MIN_FREQUENCY=3` + `EXPERT_TIER_HYSTERESIS=1.25` con `G133_TIER=1` | Il ramo G133 **salta entrambe** pur loggandole come attive | **Rimuovere** o annotare come inerti — evitano di sembrare leve tirate |
| 5 | `G133_KNOCK_X/Y`, `G133_PROMOTE_BUDGET` | Gate morto: knock-streak senza effetto misurabile; promote_budget capacity-bound (>62 rompe il seed). Il vero collo è a monte (capacity + margine hardcoded 1.0) | Lasciare ai default; **non** usarli per tuning — non muovono `promotions` |
| 6 | `STREAM_RESERVE_MB` / `STREAM_RUNTIME_RESERVE_MB` scambiate per la cap cache | Governano l'allocatore di range streamati (cu:6190), **non** la cap (cu:31240 usa solo `STREAMING_EXPERT_CACHE_RESERVE_GB`) | Per muovere N si tocca **solo** `STREAMING_EXPERT_CACHE_RESERVE_GB` (=0.125) o `_CACHE_N` |
| 7 | `Q1_0_PROMOTION_SSD_WRAP` in `$g73ForbiddenQuantServing` **ma** forzata a 1 da G73 | G73 la vieta come var ereditata (hermetic check cu:1438) poi la forza internamente (cu:1516). Contraddizione apparente ma coerente: vieta l'ereditarietà, possiede il wrap | Nessuna azione — documentare il doppio ruolo |
| 8 | `DS4_G130_U1_ATTRIBUTION=1` validata solo a `-c256/-n64` | I 10 span vivono nel path mixed-Q1; a ctx grande non attribuiscono nulla | Non fidarsi dei suoi numeri a big-ctx; usare le sonde `_STAGE_PROFILE` |

**Trappola operativa nota (non nel preset ma nel flusso):** `g7_measure.ps1:3103-3108` cancella tutte le `DS4_*` prima di lanciare il server ⇒ dot-sourcing di `g73_open.env.ps1` seguito dall'harness **non applica il preset**. Verificare sempre `prefill_chunk`/env nel log di avvio.

---

## 4. Leve da NON toccare (rumore / provenance / logging / trappole)

Elenco breve per non perderci tempo:

- **Provenance (non verificano nulla):** `DS4_MODEL_SHA256`, `DS4_*_SIDECAR_SHA256`, `DS4_MODEL_BYTES` — solo formato/gate, nessun hash calcolato.
- **Logging/dump che alterano il sistema:** `DS4_CUDA_MOE_PROFILE` (spegne split/fused), 24× `DS4_METAL_GRAPH_{DUMP,TRACE}_*` (spengono i prefill-waves), `DS4_REQUEST_PHASE_TRACE` (forbidden-list g130), `DS4_ORACLE_LOGITS`, `DS4_CPU_DUMP_*`, `DS4_METAL_MEMORY_REPORT`, `DS4_EXPERT_RECOVERY_TRACE`.
- **No-op su Windows:** `DS4_CUDA_KEEP_MODEL_PAGES`, `_NO_DIRECT_IO`, `_NO_MODEL_PREFETCH`, `_MODEL_PREFETCH_SYNC`.
- **Trappole (peggiorano o rompono):** `DS4_CUDA_NO_TOPK2048`, `_NO_TOPK_CHUNKED` (top-k a 1 thread); `DS4_CUDA_MOE_NO_SELECTED_LOAD`, `_NO_DECODE_LUT_GATE`, `_NO_DIRECT_DOWN_SUM6` (rifiutano il resolver GPU-resident); `DS4_CUDA_MOE_CACHE_POLICY=layer-top1` (spegne il resolver); `DS4_CUDA_ATTENTION_OUTPUT_PRELOAD` / `DS4_MTP_SPEC_DISABLE` (semantica invertita).
- **Inerti nella config attiva (IQ2/G73):** intera famiglia `DS4_IQ1_S_*`, `DS4_IQ1_MIXED_*`, `DS4_NESTED_RESIDUAL_*`, `DS4_Q1_0_IQ2_PINNED_*`, matrice Q1_0 a 7 booleani, `DS4_Q1_0_SSD_WRAP_*` (ring 2+2 hardcoded), `DS4_SPEX_*` (`CACHE_N≠0` disabilita la claim).
- **~26 var geometria kernel** `DS4_CUDA_MOE_{TILE4,GATE_ROW*,DOWN_ROW*,ATOMIC_DOWN,NO_*}` + **~10 fusion-disable** `DS4_METAL_DISABLE_*_FUSION` — A/B micro-kernel, non muovono il collo (1776 ms/token).
- **Tetti irrilevanti alle scale attuali:** allineamento 2048 KV-su-disco, G133 a 2²⁰ token, `--kv-*`.

---

### Nota di metodo

Punti ri-verificati sul sorgente per questo doc (non fidandosi del solo atlante): cu:1516/1517 (gate SSD-wrap), cu:1477-1495 (`ALLOW_Q1_SIDECAR`), cu:39696-39698 (dispatch `<true>` da G73_OPEN), cu:31240-31245 (cap cache), cu:25948-25950 (unico env sulla cap), cu:6190-6197 (reserve streamed-range distinto), cu:3104+3135 (Q8 reserve inerte sotto NO_Q8), cu:25931-25933 (`MOE_CACHE_POLICY` valori), cu:37246-37260 (gate fast-path: reap/prefill-observer/top1/no-lut), cu:33268+34014+35834 (`MOE_IO_QD` overlapped mai eseguito), cu:26397-26412+26540 (knock/promote gate), ds4.c:7289-7307 (`prefill_cap`). L'atlante resta la fonte per le leve non load-bearing.
