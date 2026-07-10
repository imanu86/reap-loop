# FORK SURVEY — tecniche dai fork di ds4/DwarfStar (2026-07-10)

**Scopo.** Setacciare 12 fork di `antirez/ds4` (DeepSeek-V4-Flash / "DwarfStar")
per tecniche riusabili sulle nostre leve (`docs/SOTA_ROADMAP.md`): **resident-hit
fix** (cache esperti VRAM, S4 / `CLAIMS_CURRENT.md` "Bug reserve cache esperti"),
**tiering E-LAT** (compressione differenziata hot/cold, S4 CQ1),
**prefetch** (delta-prefetch 0021 / SPEX condizionato, S3-S4), **ctx-lungo + KV**,
**boot-probe HW-adaptive** (auto-calibrazione §"qualsiasi hardware"), **mask
keep-K** (static K23 / rotate / 0024-0026), **controllo runtime / rewind** (0020 /
0022 / 0027).

> **ONESTÀ — leggere prima di citare qualunque numero.** Tutti i numeri di questo
> documento sono **CLAIM dei fork, NON verificati da noi**: HW, prompt, quant e
> metodo di misura sono quelli dichiarati dagli autori, spesso su GPU/APU diverse
> dal nostro 3060 12GB e senza il nostro n=3/ABAB (`SOTA_ROADMAP.md` P1). Nessuno
> di questi numeri è headline finché non lo replichiamo sotto il nostro protocollo
> (`docs/DS4_RUNNER_PROTOCOL.md`). In tabella e nel testo sono marcati `[CLAIM]`.
> In caso di conflitto vince `docs/CLAIMS_CURRENT.md`.

---

## 1 — Tabella fork

Ahead/behind = vs `antirez/ds4:main` come dichiarato dallo scan (verificato via
`gh api .../compare` dove indicato). "Backend" = dove vive il lavoro rilevante.

| Fork | Ahead/behind | Backend | Tecnica principale | Numeri chiave `[CLAIM]` |
|---|---|---|---|---|
| **chripell/ds4-rtx3090** (`chri-rtx3090-streaming`) | 2 / 0 | CUDA | RAM-streaming pinnato (cudaHostRegister+DMA diretta) + 3 profili ctx/cache misurati su **RTX 3090 24GB** | **ctx 535K → gen 7.23 t/s**; ctx 32K/cache 8GB → 9.04 t/s; cache 8→4GB = −17.5% gen |
| **audreyt/ds4** | 222 / 0 (30★) | Metal (M5) | Off-target (DSpark spec-decode Metal-only + robustezza server). Utili: fix eviction KV, cadence threshold-cross, RAM-residency gate | DSpark adaptive block: struct +8% / creative ~0 regressione |
| **ejpir/ds4-hip** | 179 / 67 (11★) | ROCm/HIP (Strix Halo) | Cache MoE VRAM a 4 livelli (fix thrash LRU cross-layer) + stateful KV session + WMMA layer-filter | +7% prefill con filtro layer 14-42; ngram spec snapshot/verify/rollback |
| **bonciarello/ds4-lite** | 177 / 0 | Metal (M1 Max) | Cache LRU esperti budget-GiB + **prefetch madvise/pread** + **IQ2_XXS tiering** + auto-ctx RAM-bound | madvise **+67%** decode; IQ2_XXS 48.5→21.2 GB, 5.59 t/s @cap1GiB vs Q4 3.74 |
| **rcarmo/go-ds4** (`go-ds4` + `c/streaming`) | 110 / 312 | Go + Metal | 3-tier VRAM (hot/stream/compact) **con ablation matrix su RTX 3060 12GB**; cache A/B misurata net-negativa | hot-tier statico = **2×** (0.37→0.74 t/s); FastExperts K4 vs K6 +52% |
| **chiefnoah/ds4** | 71 / 309 (7★) | ROCm/HIP (Strix Halo) | **DS4_N_USED** (curva K→qualità) + O_DIRECT align fix (**4×**) + top-K indexer block-parallel | K6 52.9→K5 58.1 (+9.8%, usabile) →K4 65.2 (degrada); SSD 1.07→4.26 GiB/s |
| **cchuter/ds4** | 48 / 169 (7★) | CUDA (multi-GPU) | **Boot-probe VRAM iterativamente debuggato** + Q8_0/Q8_K tier + per-layer mixed-quant + fix reserve-floor cache | **prefill +28.5%** sbloccando cache strozzata da reserve fisso; upfront-refuse vs OOM tardivo |
| **giannisanni/neutronstar** (`glm-local`) | 36 / — (11★) | CUDA (GLM-5.2) | Port GLM CUDA + **cross-layer prefetch** + LFU host cache + fix "cache sotto-budget diverge silenziosamente" | prefetch hit **8%→74%**; fix budget: byte-identico dove prima divergeva (Δlogit 2.08) |
| **andreaborio/ds4** | 22 / 0 | Metal (M5) | **Expert prune mask** (≈ keep-K statica) + **router-ahead prefetch** + mixed-precision streaming + RAM-guard boot | prefetch +10.6% (install mode **PERSO**); prune 40% → pass@1 74→72 |
| **mitsuhiko/ds4** (`pi-polish`) | 13 / 325 (63★) | Metal + server | Off-target (integrazione agent "Pi"). Utili: boot-probe RAM 2-soglie, watchdog anti-PID-reuse | soglie fisse >=256GB→q4 / >=128GB→q2 / else fail |
| **huihui-support/ds4** (`tp`) | 9 / 67 | CUDA (multi-GPU) | Tensor-parallel + **riordino `cuda_model_range_ptr` (bypass PRIMA del cache = hit≈0 by design)** + budget da cudaMemGetInfo | budget = VRAM_dev − 4GiB fissi; log per-tipo-hit (exact/span/registered) |
| **hawkli-1994/ds4-win** | 8 / 0 | doc-only | Design-doc porting Windows/MSVC (**zero codice**, pre-implementazione) | n/a (nessun numero misurato) |

**Il più rilevante per `ctx-lungo`:** **chripell/ds4-rtx3090** — è l'unico con
numeri end-to-end su contesti enormi (**ctx 535K @ 7.23 t/s `[CLAIM]`**) su una GPU
discreta, esattamente il regime SSD/RAM-streaming che ci interessa. Piccolo (2
commit, 2 file) ma il valore è la guida di tuning + i 3 profili misurati. Da
confermare sotto il nostro protocollo prima di trattarlo come riferimento.

---

## 2 — Sezioni per-tecnica (dettaglio + AGGANCIO alle nostre leve)

### 2A — Resident-hit fix (cache esperti VRAM)  → S4, `CLAIMS_CURRENT.md` "Bug reserve cache esperti"

Il nostro bug è: **cache esperti che sembra a hit-rate ≈0 in locale**. La causa
già isolata da noi (`CLAIMS_CURRENT.md` r.52) è che
`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` **default 16** viene capato a VRAM/2 =
6GB su una 12GB → cache esperti **disabilitata**; `reserve=1` la riabilita. I fork
convergono su **la stessa famiglia di bug** da tre angoli, e sono la porzione più
azionabile del survey.

- **huihui-support/ds4 — riordino `cuda_model_range_ptr()`** [ALTISSIMA rilevanza].
  Documenta il pattern esatto: se i bypass (`DS4_CUDA_DIRECT_MODEL` / `hmm_direct`
  / `g_model_device_owned` / `g_model_registered`) vengono valutati **PRIMA** dei
  tre tentativi di caching (fd-cache → register_mapped → device-copy), il cache
  **non si popola mai** e ogni lookup cade sul path diretto → **hit-rate ≈0 per
  costruzione, non per bug di conteggio**. La loro fix inverte l'ordine (prova
  sempre prima il cache) e aggiunge log per-tipo-hit (`exact/span/registered` + GPU
  id) dietro `DS4_CUDA_WEIGHT_CACHE_VERBOSE`. **AGGANCIO:** è il primo controllo da
  fare nel nostro binario — verificare l'ordine bypass-vs-cache in
  `cuda_model_range_ptr` e strumentare per-tipo-hit invece di un contatore
  aggregato.
- **cchuter/ds4 — fix reserve-floor cache Q8→F16** [textbook del nostro bug].
  Un `cuda_q8_f16_cache_reserve_bytes` **fisso a 4GiB** azzerava l'ammissione della
  cache su box VRAM-tight (81GB model, ~1.3GiB liberi < 4GiB floor) → 100% fallback
  al kernel scalare lento, **nessun errore**. Fix: `reserve = max(768MiB, 1% del
  totale)` invece del floor assoluto → **prefill +28.5% `[CLAIM]`** con solo ~0.88
  GiB cachati. **AGGANCIO diretto al nostro reserve-16 bug**: è esattamente la
  stessa forma. Da rubare il *pattern di fix* (`max(floor_piccolo, %_del_totale)`)
  e la disciplina di auditare **ogni** costante reserve/floor per sanità a 12GB.
  Vedi anche il loro **resolver range→device con `std::upper_bound` + containment
  half-open overflow-safe** e ritorno hit/miss esplicito: è la forma corretta di un
  lookup che conta gli hit.
- **giannisanni/neutronstar — "cache sotto-budget diverge silenziosamente"** +
  **andreaborio/ds4 (bug-cluster budget↔fallback)** + **bonciarello/rcarmo/audreyt
  (fallback silenzioso 'wrap fallito→ricopia')**. Tutti riportano lo stesso failure
  mode: sotto una soglia (working-set/token o expert-count) il codice cade su un
  path alternativo (mapped view / overflow / ricopia) **senza errore**, e in alcuni
  casi (giannisanni, andreaborio) l'ordine di accumulo diverso **cambia il
  risultato** (Δlogit fino a 2.08, greedy diverge dai primi token). **AGGANCIO
  critico:** il nostro "resident-hit≈0" potrebbe nascondere non solo un cache non
  consultato ma una **divergenza semantica**. Da replicare la loro contromisura: un
  cache-miss **non deve mai cambiare la semantica del calcolo, solo la velocità**
  (giannisanni indirizza gli overflow a whole-tensor mapped view byte-identiche;
  verificato su 256 token greedy byte-identici a budget 4GB/366-expert).
- **ejpir/ds4-hip — cache MoE 4 livelli** (resident-layer / layer-slot LRU /
  global LRU / sparse). Descrive **esattamente** il thrash che temiamo: "decode
  visita i layer in ordine ciclico fisso → LRU cross-layer va in worst-case (layer
  10 sfratta 9, 11 sfratta 10...)". Fix: pinnare gli slot ammessi per-layer +
  bypass dei layer nuovi quando pieno, invece di LRU globale. **AGGANCIO:** API di
  riferimento per la gerarchia di tiering della nostra cache (leva-hot + LRU +
  sparse), e conferma che una LRU ingenua sul nostro sweep-di-layer è sbagliata.
- **audreyt/ds4 — "freshness grace" eviction + RAM-residency gate**. (1) Uno slot
  appena popolato con 0 hit veniva sfrattato per primo come "mai utile" → aggiunge
  un termine `freshness` decadente seminato a 1.0 sulla scrittura, `max(hits,
  freshness)`. (2) `tool_memory_has_id()` controlla la tabella RAM **prima** di
  scansionare il disco. **AGGANCIO:** entrambi sono la forma del nostro fix —
  crediti di freschezza per gli esperti appena-residenti e "controlla la residency
  table veloce PRIMA di pagare lo slow-path".

### 2B — Tiering E-LAT (compressione differenziata hot/cold)  → S4 CQ1, `DYNAMIC_EXPERT_COMPRESSION_PLAN.md`

La nostra leva-2 di S4: hot=precisione alta, cold=quant aggressiva (async), per far
stare più esperti nel budget cache (steady 3060 3.12 → **3.7-4.1 t/s** modellato).

- **bonciarello/ds4-lite — IQ2_XXS (~2.13 bpw)** [argomento numerico più forte].
  Il dato da rubare non è la quant in sé ma la catena causale misurata: quant più
  aggressiva ⇒ esperti più piccoli ⇒ **più esperti nello stesso budget GiB** ⇒
  **hit-rate cache più alto**. A cap=1GiB: Q4_K 3.74 t/s vs Q2_K 4.93 vs **IQ2_XXS
  5.59 t/s**, output byte-identico `[CLAIM]`. **AGGANCIO:** è la giustificazione
  quantitativa esterna del nostro tiering (più esperti cold in cache = hit-rate su
  cui gira tutto S4), anche se loro fanno quant omogenea, non hot/cold.
- **cchuter/ds4 — Q8_0/Q8_K tier + per-layer mixed-quant**. Aggiunge un tier 8-bit
  (ancora alto della scala qualità/VRAM) e, soprattutto, **fixa il caso di quant
  MISTE per-layer nello stesso GGUF** (layer 37-42 Q4_K, resto IQ2_XXS): ogni path
  di dispatch kernel (prefill batch vs decode, sorted/tiled/fast) va controllato
  per assunzioni hardcoded sul block-type, o reinterpreta i byte del quant sbagliato
  → **garbage silenzioso, non crash**. Generalizza `routed_expert_row_bytes` da
  `QK_K(256)` fisso a `block_elems` per-tipo. **AGGANCIO:** checklist obbligatoria
  se introduciamo tiering multi-quant sulla nostra cache — è il bug-pattern da
  evitare.
- **andreaborio/ds4 — mixed-precision streaming** (`STREAMING_MIXED_PRECISION.md`).
  Stesso problema, soluzione + bug-pattern complementari: rilevare layer "off-class"
  e instradarli su un path mappato-via-view invece di forzarli nella cache uniforme;
  bug da evitare = size-class dello slab **fissata sul primo layer visto**
  (last-writer-wins) → corruzione silenziosa su mismatch. Fix: pre-seed della
  size-class al boot + `note_expert_size` freeze+reject. **AGGANCIO:** design di
  riferimento diretto per il tiering della nostra cache VRAM.
- **ejpir/ds4-hip — dispatch kernel per-quant per-tensore** (q4k/q2k/iq2 path scelti
  dal `gate_type/down_type` del GGUF). Prova che la quant mista per-expert è
  implementabile come dispatch-table per tipo-tensore. **giannisanni** aggiunge il
  kernel **dp4a IQ2_XXS routed-down** (prima solo Q2_K) — utile se il nostro cold
  tier scende a IQ2.

### 2C — Prefetch  → 0021 delta-prefetch, boot-probe (b), SPEX condizionato

- **andreaborio/ds4 — router-ahead prefetch** [più vicino a 0020+0021].
  Predice la selezione del layer L+1 girando il router di L+1 sull'attivazione
  `ffn_norm` di L (residual stream → gating consecutivi simili) e lancia
  `F_RDADVISE` (readahead avvisorio, non install forzato). +10.6% decode `[CLAIM]`,
  accuratezza top-8 = 0.79. **Controesperimento fondamentale:** la modalità
  **"install"** (streamare davvero l'expert predetto in cache mentre la GPU lavora)
  è risultata **PIÙ LENTA** (2.20→1.74 t/s), e lookahead=2 pure perso. Lezione:
  **su finestra/cache stretta il prefetch "avvisorio leggero" batte l'"aggressivo
  che occupa risorse"**. **AGGANCIO:** valida il design 0021 (delta, non WRAP full)
  e ci avverte di NON passare a install-mode aggressivo sul 3060.
- **giannisanni/neutronstar — cross-layer prefetch** (Fate-style, stessa idea):
  host-cache hit **8%→74% `[CLAIM]`**. Ma onesto: **su drive saturo è
  throughput-neutral "by physics"** (riordina i byte, non li riduce). **AGGANCIO:**
  identico al nostro vincolo boot-probe (b) — il prefetch aiuta solo se NON
  deeply-SSD-bound; buon benchmark esterno per il nostro prefetch mask-based.
- **bonciarello/ds4-lite — madvise(WILLNEED) + pread parallelo**. madvise sui
  range dei K esperti selezionati subito dopo il routing: decode **+67% `[CLAIM]`**;
  pread parallelo +15% a freddo. Onesto: **nessun guadagno se il working-set è già
  caldo** (pread→memcpy 50GB/s). **AGGANCIO:** è letteralmente route→prefetch→compute,
  e ci ricorda di **misurare separatamente hit-rate e vantaggio-da-prefetch**, mai
  confonderli.
- **giannisanni — io_uring + O_DIRECT (QD64)** con fallback page-cache, e
  **andreaborio/giannisanni MTP negativo**: la speculazione a batch NON aiuta quando
  il costo dominante è I/O per-token e l'overlap-esperti tra token è basso ("MTP paga
  solo in regime pesi-residenti, non in streaming-da-disco"). **AGGANCIO:** avviso
  prima di investire su speculative/trigger-rewind (0022) come leva di velocità.

### 2D — ctx-lungo + KV  → S5 trasferibilità, ctx→t/s curve (S4)

- **chripell/ds4-rtx3090** [il riferimento headline]. 3 profili su RTX 3090
  24GB/128GB RAM: ctx 32K/cache 8GB → 9.04 t/s; ctx 150K/cache 4GB → 7.46; **ctx
  535K/cache 4GB → 7.23 t/s** (prefill-chunk 1024, 2048 causa OOM), tutti `[CLAIM]`.
  Trade-off cache 8→4GB = −17.5% gen. Meccanismo abilitante = `cudaHostRegister`
  su tutta la mmap + `MAP_SHARED`/`O_RDWR` + cudaMemcpy diretto (bypass staging
  pread). **AGGANCIO:** baseline esterna del trade-off **cache-VRAM ↔ ctx-length ↔
  t/s** che stiamo caratterizzando; su 3060 12GB la cache residente sarà molto più
  piccola → aspettarci miss-rate e penalità **maggiori**, compensabili solo da
  prefetch/mask migliori. NB il pin dell'intero file richiede RAM ≥ modello →
  sondare `ulimit -l`/RAM libera nella boot-probe prima di abilitarlo.
- **bonciarello/ds4-lite — auto-context RAM-bound + NTK extend**. `budget =
  frazione_HW − reserve − scratch, /bytes_per_token, snap a potenza-di-2`, con
  **reserve diversa per modello resident (pesi interi) vs streaming (70% RAM per la
  page-cache esperti)**. **AGGANCIO diretto boot-probe:** è il pattern di sizing del
  ctx dalla VRAM libera reale (non da un cap hardcoded), e la spartizione
  cache-esperti ↔ KV è **identica** al nostro problema di budget VRAM su 3060. Anche:
  **KV f16→int8 componibile** validato byte-exact a ogni step — modello di processo
  per introdurre quant-KV senza rompere la correttezza.
- **ejpir/ds4-hip — stateful KV session** (session_id + revision, reset/delta): il
  client manda solo il delta, il server continua dal checkpoint KV senza re-prefill.
  **AGGANCIO:** blueprint per evitare il re-prefill in loop agentici lunghi, e lo
  split reset/delta è un pattern pulito per il nostro controllo runtime trigger/rewind
  (rewind = richiamare una `parent_revision` più vecchia). **audreyt** aggiunge il
  fix "threshold-crossing cadence invece di `% N`" (i trigger modulo smettono di
  scattare se il contatore riprende da un offset non-multiplo dopo un rewind) — da
  auditare nelle nostre cadenze (re-eval mask, telemetria, boot-probe re-check).
- **chiefnoah/ds4 — fix wrap SWA ring-cache `% raw_cap`** (scrittura batch senza
  modulo mentre le read erano modulari → uscita silenziosa a metà del chunk 2 su
  prompt lunghi). **AGGANCIO:** bug-class da testare esplicitamente nel nostro
  pipeline KV su chunk-prefill multipli.

### 2E — Boot-probe HW-adaptive  → §"qualsiasi hardware" (sonde a/b/c), gate P2

- **cchuter/ds4 — boot-probe VRAM iterativamente debuggato** [il singolo pezzo più
  trasferibile]. Un case-study reale sul failure mode che vogliamo evitare — "la
  matematica del budget dice che entra, ma OOM silenzioso all'allocazione vera".
  Lezioni concrete: (1) sondare la VRAM al boot con `cudaMemGetInfo`, non tabella
  statica; (2) applicare il margine di sicurezza **una volta sola, in un solo
  posto** (avevano un bug di doppia-sottrazione probe+engine); (3) addebitare lo
  scratch per "tier usato" non per divisione piatta, e cercare il bug di
  doppio-conteggio (per-layer AND per-tier, +64GiB a ctx 196608); (4) **rifiutare
  PRIMA di qualunque allocazione** + ri-check con probe live subito prima
  dell'alloc grande (la prima probe può diventare stale); (5) test di regressione
  che limita un **delta** (sweep ctx) non un assoluto. **AGGANCIO:** è quasi un
  capitolato per la nostra boot-probe (a). Anche il loro **compile-probe di
  capability** (JIT di un kernel reale + allowlist di generazione, non fidarsi del
  flag driver) mappa sul nostro "probe con smoke reale, non compute-capability".
- **huihui-support/ds4 — budget cache da `cudaMemGetInfo`** (multi-GPU: VRAM_dev −
  4GiB fissi; chunk arena 512MB su schede piccole vs 1792MB). Grezzo (solo
  "totale meno 4GB fissi") ma **validazione indiretta della direzione boot-probe**:
  anche i fork sentono il bisogno di auto-tuning uscendo dal caso DGX Spark.
- **andreaborio/ds4 — RAM-guard al boot** (rifiuta mapping resident > 90% RAM
  fisica con messaggio azionabile, `--ssd-streaming` suggerito) e
  **giannisanni — pinned-host fallback su OOM arena** + **O_DIRECT vs buffered per
  ratio modello/RAM** (0.87 buffered vs 0.58 O_DIRECT su SATA `[CLAIM]`).
  **AGGANCIO:** tutti pattern di degradazione graceful basata su HW reale a runtime,
  riferimento per la sonda (a)/(b). rcarmo (go-ds4) è il **controesempio**: engine
  funzionante sul nostro stesso 3060 con **zero auto-detection** (ogni soglia è una
  costante cablata sul suo box) → prova per cui vale costruire la boot-probe.
- **chiefnoah/ds4 — O_DIRECT alignment fix (4×)**. Il check "O_DIRECT solo se
  offset allineato a 4KiB" era sempre falso in produzione (le partizioni tensor del
  GGUF non partono su 4KiB) → ogni read cadeva sul path bufferizzato lento: 1.07→
  **4.26 GiB/s `[CLAIM]`** arrotondando l'offset a 4KiB + staging buffer allineato.
  **AGGANCIO:** bug-pattern da cercare nel nostro loader SSD — misallineamento
  silenzioso = 4× di throughput perso senza errori.

### 2F — Mask keep-K (baseline di riferimento)  → S2 (0024), S3 (0025/0026), T4

I fork ci danno **curve K→qualità empiriche** da usare come pavimento contro cui
validare la nostra mask session-learned.

- **chiefnoah/ds4 — `DS4_N_USED`** (clamp runtime del top-K, K=6 default): K6 52.9
  t/s (coerente) → K5 58.1 (+9.8%, **unico punto operativo usabile**) → K4 65.2
  (refusi) → K3 73.7 (allucina) → K2 82.7 (collasso da ripetizione), tutti `[CLAIM]`.
  **AGGANCIO:** è la nostra leva keep-K come override globale statico — curva
  costo(K)↔qualità per validare se il nostro mask learned **batte** questo naive
  uniform-K a parità di speedup; conferma che DeepSeek-class MoE è molto sensibile
  sotto K5-6 (calibra quanto aggressivo può essere il pruning, cfr. nostro K23).
- **rcarmo/go-ds4 — FastExperts** (top-4 vs top-6, statico): CPU +52% (1.09→1.66
  t/s), GPU CUDA 2.09 t/s `[CLAIM]`. Floor/baseline dello "static-K stupido" da
  battere.
- **andreaborio/ds4 — expert prune mask** (`DS4_EXPERT_PRUNE_MASK`, griglia
  43×N_EXPERT 0/1 letta da file, `probs[e]=-1e30` prima del top-k). Coding: drop
  ~40% esperti → pass@1 74→72 (nel rumore) `[CLAIM]`. **≈ la nostra mask keep-K ma
  STATICA (da file, non session-learned).** Vincolo importante: il gancio è nel
  **router CPU**, attivo solo su path specifici (`..._CPU_ROUTER=1`); **sul router
  GPU di default la mask è inerte** → da verificare da che router passa il nostro
  3060.
- **audreyt/ds4 — DSpark adaptive block-size** (self-tuning: block=2, escala a full
  dopo commit 100%, torna a 2 su qualunque partial, hard-cap 16). Metal-only, non
  usabile sul 3060, ma è **il nostro analogo esterno più vicino a un parametro
  session-learned feedback-tuned**: escala/de-escala un knob (block ~ keep-K) su un
  segnale accept/reject rolling dello step precedente, con soffitto e fallback
  istantaneo. Pattern riusabile come loop di controllo per keep-K.

### 2G — Controllo runtime / rewind  → 0020 / 0022 / 0027

- **rcarmo/go-ds4 — contratto di equivalenza numerica bounded** [direttamente
  riusabile per 0027]. Rifiuta il bit-identical CPU/GPU (uccide il parallelismo) e
  definisce un gate testabile: greedy argmax deve combaciare, top-10 overlap ≥8/10,
  RMSE < 0.35, max drift < 2.0, dietro env-gate. **AGGANCIO:** starting-set di
  soglie per il nostro harness di esattezza-rewind (`docs/S1_REWIND_DESIGN.md` §5,
  patch 0027) — invece di inseguire il replay bit-identico, un gate statistico
  bounded.
- **ejpir/ds4-hip — ngram speculative snapshot/verify/rollback** e **audreyt DSpark
  B2 rejection sampling**: il primitivo generico "snapshot frontier → avanza
  speculativamente → verifica → commit-o-rewind sullo stato KV" è esattamente ciò
  che 0022 vuole riusare per un prefetch di esperti con rollback su miss.
  `spec_frontier_snapshot/restore` esistono già nel nostro albero (S1_REWIND_DESIGN).
- **audreyt/ds4 — gate di onestà tipizzato** ("no fake tokens": enum
  `_NOT_READY` + reason-string invece di degradare silenziosamente quando un
  runtime non è validato) e **generation-phase gating** (non applicare
  un'intervento durante span strutturali come tool-call/JSON). **AGGANCIO:** rispecchia
  la nostra disciplina no-hype e la forma di 0025 (relearn phase-gated su marcatori
  strutturali `</style>`/`<body>`/`<script>`).

---

## 3 — Candidate adozioni (ordinate per valore/effort)

Ordine = ritorno-atteso / costo. Ogni riga cita la nostra leva e il fork sorgente.
Ricorda: i numeri sorgente sono `[CLAIM]`; ogni adozione va validata sotto n=3/ABAB.

| # | Adozione | Fork sorgente | Leva nostra | Valore | Effort | Perché |
|---|---|---|---|---|---|---|
| **1** | **Audit ordine bypass-vs-cache in `cuda_model_range_ptr` + log per-tipo-hit** | huihui-support | resident-hit fix | ALTO | BASSO | Se i bypass precedono i 3 tentativi di cache, hit≈0 è **by design** — spiegherebbe il bug senza toccare i contatori. Solo lettura + logging. |
| **2** | **Reserve/floor cache = `max(floor_piccolo, %VRAM)` invece di costante** | cchuter (+ ns/andreaborio) | resident-hit fix | ALTO | BASSO | Stessa forma del nostro bug reserve-16; +28.5% prefill `[CLAIM]` sbloccando la cache strozzata. Cambio di poche righe sull'ammissione. |
| **3** | **Invariante difensivo: cache-miss non cambia MAI la semantica (overflow→mapped view byte-identica)** | giannisanni (+ andreaborio) | resident-hit fix | ALTO | MEDIO | Il nostro hit≈0 potrebbe nascondere una **divergenza** (Δlogit), non solo un cache non usato. Test byte-exact miss-vs-hit. |
| **4** | **Boot-probe VRAM: margine una-volta-sola + refuse-upfront + re-check pre-alloc + regressione su delta** | cchuter | boot-probe (a) | ALTO | MEDIO | Capitolato pronto per la nostra sonda (a); evita l'OOM-silenzioso-tardivo, gate P2. |
| **5** | **Prefetch avvisorio leggero (madvise/RDADVISE post-routing), NON install-mode** | andreaborio + bonciarello | prefetch (0021) | MEDIO-ALTO | BASSO | Valida 0021 (delta, non WRAP); +67%/+10.6% `[CLAIM]`. Controesperimento install-mode PERSO ci risparmia una strada sbagliata. |
| **6** | **Curva K→qualità come baseline: `DS4_N_USED`/prune-mask statico da battere** | chiefnoah + andreaborio | mask keep-K (0024) | MEDIO | BASSO | Il nostro mask learned deve battere lo static-K uniforme a parità di speedup; K5-only conferma il floor. |
| **7** | **IQ2_XXS come cold-tier: più esperti/GiB = hit-rate più alto** | bonciarello | tiering E-LAT | MEDIO | ALTO | Argomento numerico per il tiering; ma kernel IQ2 CUDA da (ri)scrivere + rischio bug per-quant dispatch (cchuter). |
| **8** | **Contratto equivalenza bounded (argmax + top-K overlap + RMSE + drift) per 0027** | rcarmo | rewind (0027) | MEDIO | BASSO | Soglie pronte per l'harness di esattezza-rewind, senza inseguire il bit-identical. |
| **9** | **Auto-context RAM/VRAM-bound con reserve resident-vs-streaming** | bonciarello | ctx-lungo / boot-probe | MEDIO | MEDIO | Sizing del ctx dalla VRAM reale; spartizione cache↔KV = nostro problema di budget. |
| **10** | **O_DIRECT 4KiB-align + io_uring QD64** | chiefnoah + giannisanni | prefetch / SSD | BASSO-MEDIO | MEDIO | 4× throughput `[CLAIM]` se il nostro loader ha lo stesso misallineamento silenzioso. |

### Test minimo per validarne UNA (candidata #2 — reserve/floor scalato)

Scelgo #2 perché è alto-valore/basso-effort e attacca **direttamente** il bug
resident-hit già isolato (`CLAIMS_CURRENT.md` r.52), su HW che possediamo (nessun
pod). Protocollo minimo, no-hype:

1. **Controllo positivo (baseline attuale).** Su 3060 12GB, config SOTA_LOCAL_3060
   corrente (static K23, cache attiva via `reserve=1`), 1 prompt (cyberpunk ~4000
   tok, `T1`), n=3 ABAB, trace off, manifest. Registrare `avg_tps` **e**
   **hit-rate cache (LRU sim / contatore applicativo)** — l'invariante P2, non i t/s.
2. **Arm A (bug riprodotto).** Stessa config ma `reserve` al default che capa a
   VRAM/2 → cache disabilitata. Confermare hit-rate ≈0 (deve riprodurre il sintomo).
3. **Arm B (fix scalato).** Sostituire la costante reserve con `max(768MiB,
   0.01×VRAM_totale)` letta da `cudaMemGetInfo` al boot; ri-misurare hit-rate + t/s.
4. **Verdetto (invarianti prima).** Il fix è adottato **solo se**: (i) hit-rate
   risale **e** (ii) L-level al render **non regredisce** (deve restare ≥ baseline)
   **e** (iii) i t/s sono un output riportato, mai il criterio. Byte-check
   greedy-argmax fix-vs-baseline per escludere una divergenza (aggancio #3).
5. **Costo:** locale, zero budget pod; ~1 sessione. Se passa, promuovere in
   `CLAIMS_CURRENT.md` (regola anti-regressione) e generalizzare l'audit a **ogni**
   costante reserve/floor del path cache (huihui #1 + cchuter #4).

> **Nota di scope.** Le adozioni #1-#5 sono le uniche a **basso effort su HW che
> abbiamo**; #7 (IQ2 tiering) e #10 (io_uring) sono alto-valore ma richiedono
> kernel/loader nuovi → parcheggiare a S4 dopo che #1-#3 hanno chiarito il
> resident-hit. Tutti i fork Metal-only (audreyt, bonciarello, andreaborio,
> mitsuhiko) danno **pattern/design**, non codice CUDA riusabile 1:1.
