# Catalogo esaustivo delle leve DS4

Data del censimento: 2026-07-13. Mandato: Thread 3, setup RTX 3060 12 GB,
WSL2, RAM bake60/zero-copy e KV su disco.

## 1. Risultato esecutivo

Le leve che contano oggi non sono quelle dei kernel CUDA fini. Il percorso
osservato e' dominato da cinque budget distinti:

1. **page cache WSL/RAM**: bake60 tiene circa 40.6 GiB di pagine esperto
   selezionate; `DS4_CUDA_KEEP_MODEL_PAGES=1` e `DS4_CUDA_NO_DIRECT_IO=1`
   sono coerenti con il regime statico 0050, non con la copia separata 0051;
2. **host pinned WDDM**: la 0050 puo' registrare una finestra privata della
   mmap per DMA, ma su questo host il tetto WDDM e' 31.9453 GiB. Il budget
   runtime corrente e' 24 GiB. 28 GiB e' passato soltanto in un probe
   `cudaHostAlloc` standalone e non e' ancora provato dentro DS4;
3. **cache esperti in VRAM**: `--ssd-streaming-cache-experts`,
   `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` e le leve di residency si
   contendono i 12 GB con tensori fissi, context e staging;
4. **allocator dei tensori fissi**: il default sorgente e' 1792 MiB;
   `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256` e' soltanto configurato nei runner
   recenti. I log pod12 riportano otto OOM dell'arena anche a 256 MiB e non
   esiste un A/B che ne provi vantaggio o sufficienza sul 3060;
5. **larghezza della selezione**: bake60 statico, 154/256 esperti per layer
   scelti per massa, ha prodotto un solo HTML completo L2 (`n=1`). E' una
   fixture di meccanismo, non una validazione qualita'; le maschere K8/K23/K32
   e il predittore SPEX corrente hanno invece gia' fallito i gate disponibili.

Il risultato bake60 locale (`arm_self60b_run1`) misura 11.7 KB completi,
1.95 t/s medi e 2.45-2.52 t/s a regime sul path `pread`. Il suo
`server_env.txt` **non contiene** `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB`, perche'
il candidato 256 e' entrato nel runner dopo l'avvio: compare negli arm
V2/pod12 successivi, ma non va retroattribuito a quel render ne' promosso come
vincitore misurato.

Nel sorgente WIP di `/root/ds4-v2-work` esiste inoltre la **0050 statica**:
`DS4_CUDA_STREAM_FROM_RAM_MASKED=PATH` registra una volta i range della mmap
coperti dal complemento della mask e consente una copia DMA diretta host->VRAM.
La diagnostica ha osservato 417 copie coperte, 938.25 MiB DMA e zero errori DMA
nel micro-gate 5 GiB. Questo prova il meccanismo, non uno speedup. La coppia
ON/OFF disponibile e' bit-exact ma precede l'ultimo hardening; i suoi tempi non
sono un A/B valido. La build `build_0050i.log` e' passata per `sm_86`, mentre
exactness post-hardening e performance back-to-back restano non testate. I log
raw di queste osservazioni sono ancora soltanto sotto `/root/ds4-v2-work` e non
sono quindi repo-verificabili dal checkout corrente.

La **0051** e' invece soltanto un design (`docs/DESIGN_0051_DYNAMIC_ARENA.md`):
non esistono ancora consumer runtime per `DS4_CUDA_DYNAMIC_ARENA`,
`DS4_CUDA_DYNAMIC_ARENA_GB` o `DS4_PACE_LIVEMASK_MODE`. La 0051 sostituisce la
registrazione statica 0050 con slot `cudaHostAlloc` riassegnabili, deve allocare
dopo staging/context obbligatori e deve rilasciare le pagine sorgente gia'
copiate. Non puo' quindi ereditare alla cieca `KEEP_MODEL_PAGES=1`, la mask
statica di produzione o `DS4_CUDA_STREAM_FROM_RAM_MASKED`.

Tre assenze o distinzioni sono altrettanto importanti:

- non esiste, nei sorgenti o nelle patch censite, una leva
  `DS4_PACE_LIVEMASK_RATING_ONLY`; il design 0051 propone
  `DS4_PACE_LIVEMASK_MODE=off|rating|actuate`, ma richiede ancora codice e un
  guard esplicito che impedisca la riscrittura prematura della mask;
- sotto `patches/ds4` non esiste un file 0050: l'implementazione e' verificata
  nel sorgente WIP V2, ma non e' ancora esportata come patch;
- `DS4_CUDA_NO_WHOLE_MMAP_REGISTER`, `DS4_CUDA_SELECTED_STAGE_DEPTH`,
  `DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC` e `DS4_REAP_PREFILL_READAHEAD` sono
  soltanto nelle patch 0047/0048. Eventuali export nei runner correnti sono
  **inerti** finche' quelle patch non vengono applicate integralmente.

## 2. Perimetro, metodo e legenda

Sono stati letti in sola lettura entrambi gli alberi completi,
`/root/ds4-fullstack` e `/root/ds4-v2-work`: sorgenti C/CUDA/Metal, frontend,
test, Makefile, stato/diff/log git, script e artefatti locali. Sono stati inoltre
letti tutti i file sotto `patches/ds4`, incluse le varianti `canonical`, i due
0032, `upstream-pr497-single-token-selected-load.diff` e i sorgenti SPEX, oltre
agli handoff e ai log/config dei run nel repo.

Entrambi gli alberi hanno HEAD `da0b3f63d7cc87c1f11c3c876fb57de3e0caca50`.
`/root/ds4-fullstack` ha modifiche locali a `ds4.c` e `ds4_cuda.cu`; il V2 ha
anche `ds4_gpu.h` modificato e aggiunge la 0050. I riferimenti senza prefisso
nelle sezioni storiche sono dello snapshot fullstack (**F**); le nuove righe
marcate **V** sono del V2. Gli inserimenti V2 spostano le linee successive.

La scansione trova 677 call site diretti `getenv(...)` in F e 681 in V. La
scansione lessicale source-wide di V trova **584 token DS4 quoted**: 581 nomi
env e tre marker interni (`DS4_METAL_DECODE_STAGE_PROFILE_LAYER`,
`DS4_METAL_LAYER_STAGE_PROFILE_LAYER`, `DS4_METAL_HAS_TENSOR`). Sono inclusi
anche 136 flag long letterali nei parser/test (123 opzioni parser dirette), i
define, le soglie, i Makefile e le patch applicate o soltanto disponibili.

Legenda stato:

| Stato | Significato |
|---|---|
| OSSERVATO | Contatore, log, probe o build prova che il ramo e' stato percorso; non implica un vantaggio. |
| MISURATO | Esiste una misura comparativa valida negli artefatti del repo. |
| INFERITO | Default/effetto/interazione derivati staticamente dal codice, non da un run corrente. |
| CONFIGURATO | Il valore compare in un `server_env.txt`/meta di run, ma non isola l'effetto. |
| NON RISULTA | Nessuna occorrenza negli env catturati dei run DS4; non equivale a prova assoluta di non uso. |
| INERTE | Il nome e' esportato o conservato da un harness ma non ha consumer nello snapshot indicato. |
| PATCH-ONLY | Presente in una patch ma non nello snapshot WSL letto. |
| N/A CUDA | Leva Metal/ROCm, non attiva sul 3060 CUDA. |

Per i booleani il default e' `0`/unset salvo indicazione. I parser "truthy"
CUDA considerano attivo un valore non vuoto diverso da `0`; alcune leve in
`ds4.c` usano invece `atoi` o parser dedicati. Le variabili path aprono file
di log o input e possono aggiungere I/O sincrono.

## 3. Modello, mmap, page cache e cache CUDA

| Nome | Tipo / default | Source | Effetto | Interazioni e rischi | Stato / pertinenza 3060-WSL |
|---|---|---|---|---|---|
| `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB` | MiB interi, 1792; clamp 256..8192 | F `ds4_cuda.cu:2961-2980`; V `:3152` | Dimensione dei chunk dell'arena tensori fissi. | Chunk grandi aumentano frammentazione e picco; chunk piccoli aumentano allocazioni. | **256 non provato**: configurato negli arm V2/pod12, non attestato nel bake60 gia' avviato e non isolato in A/B; pod12 registra otto OOM dell'arena anche con questo valore. Pertinente ma INFERITO/configurato, non baseline. |
| `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` | GiB interi, 96 nello snapshot | `ds4_cuda.cu:2932-2957` | Tetto cumulativo della cache device per tensori modello. | `0` produce budget zero, non unlimited; l'OOM puo' fare fallback al path diretto salvo strict. 96 e' incoerente con 12 GB ma l'allocator reale/OOM limita prima. | NON RISULTA; da riesaminare con 8-10 GB solo in A/B. |
| `DS4_CUDA_WEIGHT_CACHE` | bool, off | `ds4_cuda.cu:1338-1375` | Forza il caching device dei range modello. | Si somma a preload/copy e compete con expert cache/context. | NON RISULTA; rischio alto su 12 GB. |
| `DS4_CUDA_WEIGHT_PRELOAD` | bool, off | `ds4_cuda.cu:1338-1375` | Precarica range modello nel cache path. | Startup e VRAM piu' alti; esclude il normale prefetch ATS/HMM. | NON RISULTA; bassa pertinenza per routed streaming. |
| `DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB` | MiB interi, 1024; clamp 64..4096 | `ds4.c:2111-2123` | Granularita' degli span di preload. | Puo' cambiare picco/fragmentation, non la selezione. | NON RISULTA; leva diagnostica allocator. |
| `DS4_CUDA_STRICT_WEIGHT_CACHE` | bool, off | `ds4_cuda.cu:3035-3128` | Rende fatale un fallimento di allocazione cache. | Elimina fallback utile; ottimo solo per validare che il path atteso sia davvero attivo. | NON RISULTA; usare solo in smoke diagnostico. |
| `DS4_CUDA_DIRECT_MODEL` | bool, off | `ds4.c:2284-2297`, `ds4_cuda.cu:1338` | Salta la cache di startup e usa registrazione/HMM o copie dirette. | Interagisce con `NO_FD_CACHE`, whole-map register e zero-copy; puo' esporre letture fini host molto lente. | NON RISULTA negli env catturati; non usare come scorciatoia V2. |
| `DS4_CUDA_COPY_MODEL` | bool, off | `ds4_cuda.cu:2699-2776`, `5523-5589` | Copia il modello in memoria device invece di ATS/prefetch. | Impossibile per l'intero modello su 12 GB; cambia il ramo di preload. | NON RISULTA; non pertinente. |
| `DS4_CUDA_COPY_MODEL_CHUNKED` | bool, off | `ds4_cuda.cu:5589` | Usa il percorso di copia a chunk. | Dipende da spazio device e chunk; non risolve la capienza. | NON RISULTA. |
| `DS4_CUDA_NO_MODEL_COPY` | bool, off | `ds4_cuda.cu:3131-3215` | Vieta il fallback di copia device del modello/range. | Puo' rendere inutilizzabile un host mapping non registrato. | NON RISULTA; utile solo per attribuzione path. |
| `DS4_CUDA_MODEL_COPY_CHUNK_MB` | MiB, 64; clamp 16..4096 | `ds4_cuda.cu:2779-2789` | Chunk delle copie modello. | Impatta staging, sync e picco. | NON RISULTA; secondaria rispetto allo stage selected. |
| `DS4_CUDA_MODEL_COPY_VERBOSE` | bool, off | `ds4_cuda.cu:3198` | Log del path copy/cache. | Rumore/log I/O, nessun effetto semantico previsto. | NON RISULTA; utile per un solo smoke. |
| `DS4_CUDA_NO_MODEL_PREFETCH` | bool, off | `ds4_cuda.cu:2699-2776` | Disabilita il prefetch ATS/HMM predefinito. | Puo' spostare page fault nel percorso critico. | NON RISULTA; sconsigliato su WSL salvo diagnosi. |
| `DS4_CUDA_MODEL_PREFETCH_SYNC` | bool, off | `ds4_cuda.cu:2762` | Rende bloccante il prefetch modello. | Startup piu' deterministico, ma niente overlap. | NON RISULTA. |
| `DS4_CUDA_KEEP_MODEL_PAGES` | bool, off | `ds4_cuda.cu:2792-2814` | Evita `madvise/fadvise(DONTNEED)` dopo le copie. | E' il fondamento del RAM bake statico e aumenta la pressione sulla page cache. In 0051 non deve impedire il reclaim delle pagine sorgente gia' duplicate negli slot pinned, altrimenti il footprint tende ad arena + 40.6 GiB. | CONFIGURATO nel bake60/0050; **non trasferire come baseline 0051**. Tenere almeno 7 GiB liberi. |
| `DS4_CUDA_NO_DIRECT_IO` | bool, off | `ds4_cuda.cu:5657-5671` | Non apre il modello con `O_DIRECT`; usa page cache. | Necessario per riuso bake caldo; i confronti cold/warm vanno bracketed. | CONFIGURATO bake60; alta pertinenza. |
| `DS4_CUDA_NO_FD_CACHE` | bool, off | `ds4_cuda.cu:1338-1375` | Bypassa il cache path da file descriptor. | Spinge verso host registration/direct copy; puo' invalidare assunzioni su staging. | NON RISULTA; diagnostica, rischio alto. |
| `DS4_CUDA_HOST_REGISTER_PLAIN` | bool, off | `ds4_cuda.cu:1202-1271`, `5523-5583` | Omessa la flag host-register read-only. | Puo' cambiare compatibilita' driver, non la topologia; una registrazione enorme puo' consumare RAM/pinned quota. | NON RISULTA. |
| `DS4_CUDA_NO_WHOLE_MMAP_REGISTER` | bool, off | patch `0047-no-whole-mmap-register.patch:121` | Impedisce `cudaHostRegister` dell'intero mmap (~81-86 GB) nel ramo 0047. | La 0050 usa un indice privato bounded e non dipende da questa leva. | PATCH-ONLY in F e V. Compare in runner V2 ma e' **INERTE** nello snapshot corrente; non attribuirle effetti. |
| `DS4_CUDA_SELECTED_STAGE_DEPTH` | intero, 4; 1..16 | patch `0047-no-whole-mmap-register.patch:15,29` | Profondita' del ring staging selected-load della 0047. | Ogni slot aggiunge host-pinned RAM/eventi; utile solo con l'intera patch. | PATCH-ONLY in F e V; non testabile come env corrente. Candidato 4 vs 8 soltanto dopo applicazione. |
| `DS4_CUDA_SELECTED_UPLOAD_EVENT` / `DS4_SELECTED_UPLOAD_EVENT` | bool, off | `ds4_cuda.cu:3615-3616` | Arma evento/wait del caricamento selected. | Fa parte dell'ordering S1; una barriera mancante corrompe pesi. | Usato internamente dalle patch overlap; non isolato nei run. |
| `DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC` | bool, off | patch `0048-prefill-overlap-s1.patch:147` | In prefill differisce sync H2D/D2D e fa un wait prima della GEMM. | Richiede fix WAR, evento e cursore stage cross-call della stessa patch; decode non toccato. | PATCH-ONLY in F e V. Storicamente **MISURATO** bit-exact: TTFT -9.7/-15.1% cold, -20.6% warm; 2.15 vs 2.19 t/s decode. Un export corrente e' INERTE. |
| `DS4_CUDA_WEIGHT_CACHE_VERBOSE` | bool, off | `ds4_cuda.cu:1249`; esteso da 0047 | Log hit/miss/cache e registrazione. | I/O di log; utile per attribuire il fast path, non per produzione. | NON RISULTA negli env catturati. |

### 0050 V2: registrazione privata masked e DMA diretto

Queste tre leve esistono soltanto nel sorgente WIP V2; non esiste ancora un
file patch 0050 sotto `patches/ds4`.

| Nome | Tipo / default | Source V | Effetto | Interazioni e rischi | Stato / pertinenza 3060-WSL |
|---|---|---|---|---|---|
| `DS4_CUDA_STREAM_FROM_RAM_MASKED` | path, unset/off | `ds4.c:12932-13124,30493`; API `ds4_gpu.h:81`; CUDA `ds4_cuda.cu:1106-1292,5961-6041` | Legge righe `layer expert` con la stessa semantica della REAP mask: le righe sono blocked e il complemento e' kept. Coalesca run contigui, registra privatamente pagine della mmap una volta allo startup e usa DMA diretto verso la VRAM quando una richiesta e' interamente coperta da un range. GEMM resta in VRAM; fallback e' `pread -> staging`. | Budget diviso tra i soli layer che hanno almeno un blocked; un layer all-kept e' indistinguibile da assente. Un run e' ammesso o saltato intero. I pin sopravvivono ai rebuild della cache ordinaria, sono drenati prima di unregister e rilasciati su cambio map/cleanup. Dopo un enqueue DMA non viene avviato un `pread` sovrapposto se evento/sync fallisce. `HOST_REGISTER_PLAIN` cambia le flag; retry plain automatico su errori unsupported/invalid. Non e' un'arena dinamica e non segue una mask cambiata durante la sessione. | **OSSERVATO**: 1035/1035 range, 4.75 GiB; 417/5997 query coperte, 938.25 MiB DMA, `dma_failed=0`. Exactness ON/OFF pre-hardening identica; post-`build_0050i` non testata. Massima pertinenza zero-copy. |
| `DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB` | intero GiB positivo, **24** | `ds4.c:12968-13025` | Limita il payload dei run registrati e ripartisce il budget per layer masked. | Conta payload, non page-union allineata; non spezza un run. Il parser accetta un prefisso numerico con suffisso e non protegge esplicitamente overflow: rischio INFERITO. Il tetto WDDM misurato e' 31.9453 GiB; 40 GiB non e' operativo. | **OSSERVATO** registration gate: 5/10/20 GiB = 120/120 range e successo; 40 GiB = 92/120, 30.67 GiB e timeout. 24 GiB = 4446/4446, 23.99 GiB in 24.442 s. **24 e' l'unico candidato runtime corrente; 28 e' standalone/non provato in DS4.** |
| `DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG` | truthy bool, off | `ds4_cuda.cu:1154-1183,5229-5306` | Conta query/byte, coverage, DMA ok/fail e cause miss; logga il primo miss, ogni 512 query e il riepilogo finale. | Solo osservabilita'; aggiunge atomiche e log. Usare nell'arm di attribuzione, non in entrambi i bracci prestazionali. | **OSSERVATO** nel micro-gate sopra; alta utilita' diagnostica, nessun beneficio runtime. |

La coppia coffee `temp=0`, 60 token, ha prodotto 214 byte ON e OFF con SHA256
`81fb5d5f83d91fae4da37bd2df98ba0b37699dfe9b0f9e48fc45c9124a9eff30`.
Era precedente all'hardening P1. I tempi OFF 196.255 s/0.26 t/s e ON
350.779 s/0.32 t/s hanno ordine, cache e termica non controllati: sono
**OSSERVATI ma non MISURATI come A/B** e non dimostrano ne' speedup ne'
regressione. Evidenza consolidata in
`docs/PUNTO_V2_ZEROCOPY_DYNAMIC_ARENA_20260713.md`.

Artefatti WSL letti: `build_0050i.log`; `stage_out/stage{5,10,20,40}gb/`
(`server_env.txt`, `server.stderr.log`, response/curl); `diag_fast_path/`;
`diag_paths/`; `bitexact/{off,on}` e `bitexact_pre_0050i/`, tutti sotto
`/root/ds4-v2-work`. Sono stati soltanto letti; nessun server o test GPU e'
stato avviato da questo censimento. Salvo il `build_0050i.log` catturato nello
snapshot della campagna controllata, questi path raw **non sono ancora copiati
nel repo e non sono repo-verificabili**. Devono essere archiviati prima di usare
i numeri come gate riproducibili.

### 0051: arena dinamica, design-only

La 0051 non e' presente nei sorgenti o nelle patch censite. I nomi proposti
`DS4_CUDA_DYNAMIC_ARENA`, `DS4_CUDA_DYNAMIC_ARENA_GB` e
`DS4_PACE_LIVEMASK_MODE` non sono leve attivabili oggi e non fanno parte dei 584
token runtime inventariati. Il design richiede:

- arena `cudaHostAlloc` dopo context, staging ed eventi obbligatori;
- 24 GiB come default iniziale; 28 GiB soltanto dopo startup/cleanup ripetuti e
  A/B end-to-end con almeno 7 GiB `MemAvailable`;
- esclusione mutua con `DS4_CUDA_STREAM_FROM_RAM_MASKED`;
- target live costruito dalla massa del prefill; mask60 solo fixture di bring-up;
- reclaim delle pagine mmap gia' copiate negli slot, senza duplicazione fisica
  `arena + bake60`.

### Cache quantizzate e matematica

| Nome/gruppo | Tipo / default | Source | Effetto, rischi e stato |
|---|---|---|---|
| `DS4_CUDA_NO_Q8_F16_CACHE` | bool, off | `ds4_cuda.cu:1561-1601`, guard anche `ds4.c:13348` | Spegne la cache F16 derivata dai Q8. CONFIGURATO in bake60 e necessario alle prove PACE/residency bit-neutrali; libera VRAM. Tenerlo ON nel protocollo attuale. |
| `DS4_CUDA_Q8_F16_CACHE_MB` | MiB, unlimited | `ds4_cuda.cu:1412-1445` | Tetto alternativo alla disabilitazione totale. NON RISULTA; puo' consentire una quota F16, ma cambia rappresentazione/percorso e consuma VRAM. |
| `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` | MiB; default 512 se VRAM >=112 GiB, altrimenti max(4096, 5% VRAM) | `ds4_cuda.cu:1412-1445` | Riserva spazio prima della cache F16. Sul 3060 il default effettivo e' 4096 MiB, molto conservativo; irrilevante quando `NO_Q8_F16_CACHE=1`. |
| `DS4_CUDA_Q8_F16_ALL` | bool, off | `ds4_cuda.cu:1566` | Estende la conversione F16 a tutti i Q8. Rischio OOM elevato, NON RISULTA. |
| `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE`, `DS4_CUDA_NO_ATTN_Q_B_F16_CACHE` | bool, off | `ds4_cuda.cu:1572,1575` | Opt-out mirati per cache attention. NON RISULTANO; utili solo se si riabilita una quota Q8->F16. |
| `DS4_CUDA_ATTN_Q_B_F32_CACHE`, `DS4_CUDA_Q8_F32_ALL`, `DS4_CUDA_Q8_F32_LARGE`, `DS4_CUDA_Q8_F32_PRELOAD` | bool, off | `ds4_cuda.cu:1595-1600`, `5702` | Cache/preload F32 sperimentali, ancora piu' costosi in VRAM. NON RISULTANO; non pertinenti a 12 GB. |
| `DS4_CUDA_NO_Q8_DP4A` | bool, off | `ds4_cuda.cu:1591` | Disabilita il kernel DP4A Q8 predefinito. Potenziale regressione prestazioni; NON RISULTA. |
| `DS4_CUDA_NO_TF32` | bool, off; `--quality` lo implica | `ds4_cuda.cu:5160,5723` | Forza matematica non-TF32. Interazione diretta con qualita'/throughput; non e' una leva di residency. |

## 4. Streaming esperti, residency e tier

| Nome | Tipo / default | Source | Effetto | Interazioni/rischi | Stato / pertinenza |
|---|---|---|---|---|---|
| `DS4_CUDA_STREAMING_EXPERT_CACHE_N` | esperti, default compile-time 512, max 23424 | `ds4_cuda.cu:39-40,3288-3305` | Override del numero slot cache. | Si sovrappone al CLI `--ssd-streaming-cache-experts`; il budget libero e la reserve possono ridurlo. | NON RISULTA come env; i run usano il CLI. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` | GiB float, 0.5 | `ds4_cuda.cu:3328-3353` | VRAM lasciata libera prima di dimensionare cache; cap a meta' VRAM. | Troppo bassa causa collasso resident/OOM, troppo alta perde slot. | CONFIGURATO a 1 GiB nel bake60. Sweep 0.5/1 e' sensato con metriche di slot e free VRAM. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE` | bool, off | `ds4.c:18471` | Profilo hit/load della cache. | Overhead/log; utile per attribuzione. | Usato in campagne precedenti, non nel bake60 env. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE` | bool, off | `ds4_cuda.cu:6395` | Log dettagliato cache. | Log voluminoso. | NON RISULTA. |
| `DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST`, `DS4_CUDA_STREAMING_EXPERT_HOTLIST` | alias truthy, off | `ds4_cuda.cu:3505-3514` | Rendono visibile al graph comune il budget CUDA configurato quando manca un override CLI/env; **non caricano un path hotlist**. | Il secondo nome sembra un path ma nel consumer CUDA e' trattato come booleano. Il file hotlist del graph comune usa invece `DS4_METAL_STREAMING_EXPERT_HOTLIST`. | NON RISULTANO; alias duplicati di abilitazione, non autorita' di contenuto. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST`, `DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP`, `DS4_METAL_STREAMING_EXPERT_HOTLIST`, `..._PROFILE` | disable off; cap 4096; path/profile unset | V `ds4.c:18228-18411,24207-24337` | Controllano l'hot seed del graph SSD comune: built-in se il path manca, file se impostato, cap automatico se il CLI preload e' zero. | Nomi legacy `METAL`, ma il path non ha guard backend Metal ed e' raggiunto dal graph CUDA. Un file stale o di altro dominio puo' peggiorare residency/startup. | **APPLICABILI CUDA 3060**, INFERITI/non isolati. |
| `DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED`, `..._K`, `..._PROFILE` | bool off; K 1, max strutturale; profile off | V `ds4.c:18203-18225,24135-24205` | Dopo il prefill semina la cache esperti con gli ID realmente selezionati nelle ultime K righe. | Puo' aggiungere readback/seed e competere con hotlist, PACE pin e futura policy 0051; richiede A/B ed exactness. | **APPLICABILI CUDA 3060** nel graph comune; unused ad alto valore informativo, non baseline. |
| `DS4_EXPERT_HOTLIST` / `DS4_EXPERT_PROFILE` | path | `ds4.c:29937-29939` | Input hotlist e output profiling generale. | File stale o di altro dominio producono residency sbagliata. | Usati in fasi di profiling, non nel bake60 catturato. |
| `DS4_EXPERT_TIERING` | enum string, off; unico valore `observe` | `ds4_cuda.cu:298-337` | Osserva domanda esperti per tiering. | Non promuove da solo senza policy; log/contatori. | CONFIGURATO negli arm tiering, non bake60. |
| `DS4_EXPERT_TIER_POLICY` | string, off; `observe_promote` | `ds4_cuda.cu:320` | Abilita promozione osservata. | Puo' competere con pin-by-mass e cache LRU. | NON RISULTA nei run catturati. |
| `DS4_EXPERT_TIER_PROMOTE_CAP` | count, 1024; max 131072 | `ds4_cuda.cu:331` | Limite promozioni/ID conservati. | Memoria metadati e churn. | NON RISULTA. |
| `DS4_EXPERT_TIER_PROMOTE_VERBOSE` | bool, off | `ds4_cuda.cu:335` | Log promozioni. | Solo diagnostica. | NON RISULTA. |
| `DS4_EXPERT_TIERING_LOG`, `DS4_EXPERT_TIERING_LOG_IDS` | path | `ds4_cuda.cu:356,373` | Log summary e ID. | I/O e file grandi. | CONFIGURATI negli arm tiering. |
| `DS4_EXPERT_TIERING_SUMMARY_EVERY` | token, 0 | `ds4_cuda.cu:564` | Cadenza summary; 0 disabilita. | Overhead proporzionale alla frequenza. | NON RISULTA. |
| `DS4_PACE_PIN` | bool, off | `ds4_cuda.cu:3935` | Residency dinamica con CUSUM. | Richiede `NO_Q8_F16_CACHE` per rappresentazione bit-neutrale; confligge con `PACE_TIER`. | Testato in campagne PACE, non bake60. |
| `DS4_PACE_PIN_ROTATE` | bool, off | `ds4_cuda.cu:3936` | Ruota il set pinned al cambio regime. | Churn/copied bytes. | NON RISULTA nei run recenti. |
| `DS4_PACE_PIN_BUDGET_MB` | MiB, 3500 | `ds4_cuda.cu:3937` | Budget del pin. | Si somma a cache fissi/context; 3.5 GiB e' molto sul 3060. | NON RISULTA. |
| `DS4_PACE_PIN_WARMUP`, `EWMA`, `CUSUM_K`, `CUSUM_H`, `COOLDOWN` | 512, 0.05, 0.05, 1.0, 128 | `ds4_cuda.cu:3939-3945` | Parametri detector/stabilita'. | Sensibilita' alta crea churn; bassa reagisce tardi. | NON RISULTANO nei recenti env. |
| `DS4_PACE_PIN_LOG` | path | `ds4_cuda.cu:3947` | Telemetria pin. | I/O. | NON RISULTA. |
| `DS4_PACE_TIER` | bool, off | `ds4_cuda.cu:4227` | Tier resident con hysteresis; supersede `PACE_PIN`. | Entrambi scrivono lo stato pinned, quindi non abilitarli insieme. | Testato/configurato in run dedicati, non bake60. |
| `DS4_PACE_TIER_WARMUP`, `X`, `Y`, `HYST`, `VRAM_SLOTS`, `DECAY`, `KNOCK`, `COOLDOWN` | 512, 3, 5, 1, 394, 0.98, 1, 64 | `ds4_cuda.cu:4228-4243` | Soglie/capacita' del tier. | `VRAM_SLOTS=394` va ricalibrato sul cache envelope effettivo; churn e ritardi. | NON RISULTANO nei recenti env. |
| `DS4_PACE_TIER_LOG` | path | `ds4_cuda.cu:4245` | Telemetria tier. | I/O. | NON RISULTA. |
| `DS4_CUDA_INPLACE_RESIDENT` / `VERIFY` | bool, off/off | `ds4_cuda.cu:3978-3979` | Riusa slot resident senza copia; verify controlla equivalenza. | Epoch/stale-slot corretti dalle patch 0034/0036a; errore qui corrompe output. | Testato bit-neutral in campagne T1-T3, non bake60. |
| `DS4_CUDA_INPLACE_RESERVE_SLOTS` | count, `capacity/8+1` | `ds4_cuda.cu:4533-4542` | Slot riservati per miss/churn. | Troppo basso rischia deadlock/stale; troppo alto riduce resident. | NON RISULTA come override. |
| `DS4_REAP_PIN_BY_MASS` | bool, off | `ds4.c:13310`, consumer `ds4_cuda.cu:4481` | Ordina il pin per massa, non cambia da solo la selezione. | Da non confondere con mask top-mass; usa publisher livemask. | CONFIGURATO nelle campagne pin-by-mass; bake60 usa mask statica. |

### Tier cold/RAM

Le otto leve `DS4_EXPERT_COLD_*` (`ds4_cuda.cu:2187-2210`) controllano
formato e fallback del tier cold: `FORMAT` e' unset (`lossless`, `native` o
formato CQ1 per abilitarlo); `NATIVE_TOKENS` e `NATIVE_LAYERS` sono interi con
default 50 e 43; `RAM_LOSSLESS`, `RAM_PREFILL`, `RAM_VERIFY`, `RAM_VERBOSE`,
`ALLOW_UNGATED` sono bool off. Default complessivo: tier non attivo, nessun
fallback RAM opt-in. `ALLOW_UNGATED` riduce le protezioni di compatibilita';
`VERIFY` costa ma va usato nei primi smoke; `RAM_PREFILL` puo' contendere la
page cache bake60. Nessuna risulta nel bake60 o nei run env recenti. Sono
pertinenti solo a un esperimento cold-tier separato, non al fast path zero-copy
masked.

## 5. Mask REAP, livemask e PACE

### Mask esterna e prefetch

| Nome | Tipo / default | Source | Effetto | Interazioni/rischi | Stato |
|---|---|---|---|---|---|
| `DS4_REAP_MASK_FILE` | path, unset | `ds4.c:7722-7748` | File `layer expert`; mtime polling e riapplicazione ogni 32 token. Gli esclusi ricevono bias hard `-1e9`. | Una mask stretta causa word-salad. | CONFIGURATO bake60 (`mask60_self.txt`) e tutte le campagne mask. |
| `DS4_REAP_MASK_SOFT_BIAS` | float, hard se unset/0; accetta solo finito <=0 | patch `0046-counterfactual-admission.patch:65` | Sostituisce `-1e9` con bias finito, per es. -2. | Bias poco negativo dissolve verso K0; troppo negativo affama. | MISURATO nel braccio B: output skeleton 518 char; patch 0049 logga i breakthrough. PATCH-ONLY nello snapshot WSL. |
| `DS4_REAP_PREFETCH` | bool; se PACE attivo default on | `ds4.c:12884` | Prefetch degli esperti selezionati/wrap. | Puo' duplicare I/O e scaldare il dominio sbagliato. | CONFIGURATO in run PACE. |
| `DS4_REAP_PREFETCH_THREADS` | intero, 8; max 16 | `ds4.c:12804` | Worker del fattorino/page-in. | Su WSL QD8 non ha aumentato banda e ha conteso il `pread`. | CONFIGURATO in campagne fattorino. |
| `DS4_REAP_PREFETCH_LOCK` | bool, off | `ds4.c:12841` | Serializza/locka il prefetch. | Meno race, meno overlap. | CONFIGURATO in alcune campagne. |
| `DS4_REAP_PREFETCH_DELTA` | bool, off | `ds4.c:12947`, patch 0043 | Prefetch solo delta tra finestre. | Dipende dalla correttezza della mask pubblicata e puo' churnare page cache. | CONFIGURATO in campagne 0043. |
| `DS4_REAP_PREFILL_READAHEAD` | bool, off | patch `0048-prefill-overlap-s1.patch:19` | Page-in dell'intero layer successivo durante prefill. | Compete con `pread` e puo' espellere le pagine bake. | Storicamente **MISURATO e refutato su WSL2**: +59% cold, +46% warm e cache churn; bit-exact. PATCH-ONLY/INERTE nel V2 corrente; solo nuovo test su NVMe nativo con 0048 completa. |
| `DS4_REAP_SENSOR_LOG` | path | `ds4.c:17453` | Log del sensore/router. | I/O, privacy/dimensione tracce. | Usato per analisi offline, non bake60. |
| `DS4_REAP_WRAP`, `DS4_PACE_WRAP` | bool/alias; con PACE default on | `ds4.c:12882-12883` | Precarica il set al wrap/cambio. | Alias storici, evitare valori discordanti. | Usati nelle campagne PACE. |

Gli alias patch-only `DS4_REAP_WRAP_THREADS` e `DS4_REAP_WRAP_LOCK`
(`0013-reap-wrap.patch:131,168`) sono stati rinominati in
`DS4_REAP_PREFETCH_THREADS/LOCK`: non usarli nei runner nuovi.

### Livemask dinamica

`DS4_PACE_LIVEMASK` e' indipendente da `DS4_PACE`: default off. I default
seguenti sono letti in `ds4.c:13177-13311`.

| Gruppo di nomi | Tipo / default | Effetto, interazioni/rischi | Stato |
|---|---|---|---|
| `BOOTSTRAP`, `WINDOW`, `K` | token 10; finestra 16 clamp 3..32; K 8 | Avvio, memoria osservazioni e larghezza hard. K8 e' qualitativamente insufficiente nei nostri run. | CONFIGURATI in molte campagne. |
| `K_ADAPTIVE`, `K_MIN`, `K_MAX` | bool 0; K; K | Varia K con strong-knock. | MISURATO offline: avg K35.8, miss 26.5%, ma trigger/attuatore risultavano scollegati nella build provata. |
| `KNOCK_THRESHOLD`, `GAIN`, `STEP_UP`, `STEP_DOWN`, `DEADBAND`, `UPDATE_EVERY`, `KNOCK_PREFETCH`, `KNOCK_MIN_HISTORY` | 0.15, 0.5, 4, 1, 2, 2, 0, 2 | Controller di widening/narrowing. Prefetch puo' aggiungere I/O prima che K sia stabile. | Configurati negli arm adaptive-K; nessun vincitore di qualita'. |
| `OBSERVE_TOP`, `X`, `MAX_SWAPS`, `COOLDOWN`, `HYST` | 16 max 16; 3; 2 max K/2; 16; 1 | Candidati, promozione/demozione e churn. | CONFIGURATI. |
| `WEIGHTED`, `BLOCKED_WEIGHT`, `DEMOTE_HORIZON`, `PROMO_RATE_CAP` | 0, 1.0, WINDOW, 0 | Rating per massa/blocked e limiti di promozione. | Configurati; massa ha battuto frequenza offline. |
| `PIN`, `DS4_REAP_PIN_BY_MASS` | 0, 0 | Pubblica set per residency. Non modifica semanticamente la mask salvo il producer. | Pin-by-mass MISURATO bit-neutral come residency. |
| `PRESSURE`, `PRESSURE_KNOCK`, `PRESSURE_COOLDOWN`, `PRESSURE_X`, `PRESSURE_MAX_SWAPS` | 0; X; 2; 1; K/2 | Corsia rapida di promozione sotto pressione. | CONFIGURATA nei run pressure, senza prova di qualita' finale. |
| `LOG` | path | Log eventi/massa; patch 0049 include breakthrough soft-mask. | Usato estesamente. |
| `SPEX_ADD`, `SPEX_CADENCE`, `SPEX_WRAP`, `SPEX_LOG`, `SPEX_PIN_LOG` | 0; 1 max 64; 0; path; path | Unione additiva delle previsioni SPEX alla mask/pin. `ADD` max 16. | MISURATO: top4 additivo, 154 load/token, hit 3.4%, 0.61 t/s; cadenza riduce costo ma non migliora precisione. |

Non e' presente alcun `RATING_ONLY`: con `LIVEMASK=1` il producer puo'
pubblicare/riscrivere la mask. Per il design bake60 + osservatore dinamico non
va abilitata finche' non esiste il guard richiesto dall'handoff.

### Controller PACE classico

Tutte le leve sono in `ds4.c:13377-13471`, salvo i limiti indicati. `DS4_PACE`
default off; le prove hanno mostrato la meccanica bit-neutrale, ma non hanno
risolto la qualita' delle mask strette.

| Gruppo | Default | Effetto / rischio | Stato |
|---|---|---|---|
| `WARMUP`, `KEEP`, `KEEP_MIN`, `KEEP_MAX`, `KEEP_STEP` | 150, 40, 24, 64, 4 | Envelope K del controller. Il massimo 64 resta molto sotto bake60 K154. | Usato/testato. |
| `NGRAM_N`, `NGRAM_WINDOW`, `DRIFT`, `RELEASE`, `HYST` | 3, 120 (max 512), 0.35, 0.15, 200 | Sensore di drift e stabilita'. | Usato/testato. |
| `BREATH_EVERY`, `BREATH_LEN`, `BREATH_KEEP` | 400 clamp 150..1200, 80, 64 | Allargamento periodico. | Usato; non salva le mask strette. |
| `PREBREATH`, `PREBREATH_DRIFT`, `TARGET`, `EVERY`, `KEEP_MAX` | 0, 0.245, derivato, 64, max(breath_keep, keep_max) | Anticipa breath. | Sperimentale, non vincente. |
| `PREBREATH_ADAPT`, `GAIN`, `POWER`, `STEP_MAX` | 0, 8, 1, 0 | Adatta il prebreath. | NON RISULTA negli env recenti. |
| `PREBREATH_RELEARN`, `RELEARN_DECAY` | 0, 0.30 | Reimpara il baseline dopo drift. | NON RISULTA. |
| `RELEARN`, `RELEARN_DECAY`, `ALPHA_NGRAM`, `ALPHA_HIT` | 1, 0.30, 0.10, 0.05 | Aggiornamento EMA/baseline. | Usato come default. |
| `TIGHTEN_LO`, `HIT_HI`, `STABLE`, `ANNEAL_WARM`, `S1` | 0.10, 0.90, 120, 300, 0 | Soglie tighten/stabilita' e sensore S1. | S1 base off. |
| `WRAP_ROTATE`, `CACHE_FLUSH`, `PREFILL_APPLY`, `PREFILL_WAIT_WRAP`, `EXCHANGE_OBSERVE` | 0, 0, 1, 1, 0 | Coordinamento mask/cache/prefill. | Tocca timing e cache; non cambiare piu' leve insieme. |
| `ROTATE`, `ROTATE_EVERY`, `ROTATE_DECAY`, `ROTATE_PRESERVE_STABLE`, `RELEARN_ON_TIGHTEN` | 0, 32, 0.98, 1, 0 | Rotazione della finestra. | Puo' aumentare churn; campagne storiche. |
| `WEIGHTED_SELECTED`, `WEIGHTED_WARMUP`, `WEIGHTED_RELEARN` | 0; eredita warmup; eredita relearn | Usa massa/pesi nel ranking. | Massa e' promettente offline, ma PACE K64 resta stretto. |
| `CACHE_FLOOR`, `CACHE_TARGET_SLOTS` | derivati dal cache planner | Vincoli di cache applicati a sessione/startup. | Interagiscono con il CLI cache e reserve; non override nei run recenti. |
| `DEBUG`, `LOG` | 0, path unset | Telemetria. | Usata in campagne; overhead log. |

### Leve PACE solo patch

| Nome/gruppo | Tipo/default | Patch | Effetto, rischio e stato |
|---|---|---|---|
| `DS4_PACE_ADAPTIVE_CF_ADMIT`, `..._WINDOW` | bool 0; 3 clamp 1..8 | 0046 | Usa media counterfactuale corta solo per i nuovi ingressi; espulsioni restano mass10. MISURATO nel braccio A, che rende ma dissolve verso K0 (unione 49.6%). |
| `DS4_PACE_ADMIT`, `H`, `KDRIFT`, `PERSIST`, `COOLDOWN`, `MAX_PER_100` | 0, 1.2, 0.02, 2, 16, 0 | 0026 | Demand admission. Puo' ampliare senza limite se cap 0; non risulta nei run recenti. |
| `DS4_PACE_ALPHA_S1`, `S1_TRIGGER`, `S1_SLOPE_WIN`, `S1_SLOPE_THR`, `S1_STABLE`, `S1_ACTION` | 0.10, 0, 64 max 256, 0.0003, 16, `rotate` (`widen` alternativo) | 0020 | Trigger da pendenza S1. Patch-only nello snapshot; nessun risultato vincente. |
| `DS4_PACE_WRAP_ROTATE_DELTA` | bool 0 | 0021 | Prefetch solo delta al rotate. I/O/churn; non risulta. |
| `DS4_PACE_REWIND*` | off; parametri completi nell'appendice patch-only | 0022 | Checkpoint e rewind su peggioramento S1. Rischio di complessita'/latenza e stato KV; non presente nello snapshot. |
| `DS4_REWIND_TEST`, `DS4_REWIND_TEST_LOG` | bool/path | 0027 | Harness di exactness. Solo test. |
| `DS4_ASYNC_PIPELINE` | bool 0 | due patch 0032 | Pipeline asincrona layer L/L+1. Due varianti concorrenti (`rebased`, `s1`); non e' baseline affidabile. |

## 6. SPEX

| Nome/gruppo | Tipo/default | Source | Effetto | Rischi e stato |
|---|---|---|---|---|
| `DS4_SPEX_FILE`, `PATH`, `HIDDEN_FILE`, `MARKOV_FILE` | path, unset | `ds4.c:16316-16325` | Sorgente modello con precedenza FILE > PATH > HIDDEN_FILE > MARKOV_FILE. | File incompatibile/stale cambia previsioni. Usati nelle campagne SPEX. |
| `DS4_SPEX_CAP`, `HIDDEN_CAP` | count; 6 e CAP | `ds4.c:17002-17005` | Numero predetti/caricati, clamp all'output cap. | Aumentare cap aumenta I/O quasi linearmente. |
| `DS4_SPEX_TAU` | float 0.5 clamp 0..1 | `ds4.c:17219` | Soglia confidence/peso del predittore. | Usato in sweep; raw predictor debole. |
| `DS4_SPEX_PREFETCH_NEXT_LAYER`, `DISABLE_PREFETCH_NEXT_LAYER` | bool; auto-on con file SPEX, opt-out esplicito | `ds4.c:17267-17269` | Prefetch layer successivo, solo con SSD streaming. | Puo' caricare falsi positivi e distruggere locality. |
| `DS4_SPEX_HIDDEN_PREFETCH` | bool 0 | `ds4.c:16861` | Predizione dal hidden state. | Richiede read/cast sincrono per token/layer nel path CPU. |
| `DS4_SPEX_HIDDEN_GPU_LOAD`, `GPU_SCORE`, `GPU_PREFETCH` | bool 0; score/load implicati dal prefetch | `ds4.c:16866-16887` | Sposta scoring/top-k/handoff su GPU. | Contende stream/cache e aggiunge sync se il path non e' realmente async. CONFIGURATI in run SPEX. |
| `GPU_PREFETCH_DRY_RUN` | bool 0 | `ds4.c:16894` | Calcola/logga senza caricare. | Ottimo per precisione offline, nessun speedup. CONFIGURATO in smoke. |
| `GPU_PREFETCH_STATS`, `..._EVERY` | bool/path implicito; 256 | `ds4.c:16928,16988` | Statistiche hit/recall/cadenza. | Overhead log. CONFIGURATI. |
| `DS4_SPEX_PREFETCH_PROFILE`, `DS4_SPEX_STATS` | bool/path | `ds4.c:16901`, `ds4_cuda.cu:897` | Profiling CPU/CUDA. | Solo diagnostica. |
| `TRACE_HIDDEN`, `TRACE_ROUTING`, `TRACE_ROUTING_WEIGHTS` | path/bool | `ds4.c:17290-17358` | Cattura hidden, routing e pesi. | `TRACE_ROUTING_WEIGHTS` fa append/read sincrono; file grandi. Usati per training/offline. |
| `TRACE_ROUTING_RESIDENCY`, `TRACE_TOKENS` | bool/path | patch 0017/0028 | Aggiunge stato cache e token alla traccia. | PATCH-ONLY nello snapshot, usati in analisi. |

Fatto misurato: SPX1 top4 weighted recall circa 19%, massa catturata 1-3%; la
corsia additiva top4 ha precisione 4-7%, 154 load/token, hit 3.4% e 0.61 t/s.
La cadenza 0045 taglia il costo, non corregge la precisione. Non e' oggi una
leva da mettere nel baseline bake60.

## 7. MTP, profili e leve generali

| Nome/gruppo | Tipo/default | Source/patch | Effetto | Rischio e stato |
|---|---|---|---|---|
| `DS4_MTP_PROBE`, `FULL_LOGITS`, `STRICT`, `TIMING`, `BATCH_VERIFY`, `CAPTURE_PREFIX1`, `EXACT_REPLAY`, `FORCE_SNAPSHOT` | bool, unset | `ds4.c:31443-31788` | Diagnostica/exactness MTP. | Logits/snapshot possono essere costosi. NON RISULTANO nei run bake60. |
| `DS4_MTP_MIN_MARGIN` | float >=0; default ereditato da `e->mtp_margin` | `ds4.c:31555-31562` | Override della soglia di accettazione/speculazione. | Cambia acceptance e potenzialmente output/timing; solo con `--mtp`. |
| `DS4_MTP_CONF_LOG`, `SPEC_LOG` | path | `ds4.c:31564,31580` | Log confidence/speculazione. | I/O. |
| `DS4_MTP_STREAMING_UNSAFE` | bool 0 | patch 0008 | Bypass sperimentale del guard MTP+streaming. | Esplicitamente unsafe; non usare. |
| `DS4_MTP_STREAMING` | bool 0 | patch 0009 | Unlock successivo con lifecycle corretto e patch 0010 model-map registration. | PATCH-ONLY nello snapshot corrente; richiede test exactness e VRAM. |
| `DS4_DIAG_CONF_LOG` | path | patch 0030 | Confidence + expert norm per token. | Telemetria, I/O; non baseline. |
| `DS4_THREADS` | intero, min(CPU online, 12), max 32 | `ds4.c:1383-1397`, define `:1315` | Thread CPU. | Troppi thread su WSL contendono I/O/page fault; non oltre 32. |
| `DS4_ROUTED_TOKEN_PARALLEL` / `NO_ROUTED_TOKEN_PARALLEL` | bool force/disable; auto-on da 64 token | `ds4.c:8510-8511` | Parallelismo routed per token. | Le due leve sono mutuamente contraddittorie; benchmark, non quality. |
| `DS4_PARALLEL_ATTN_ROWS` / `NO_PARALLEL_ATTN_ROWS` | bool force/disable; off senza force | `ds4.c:9789-9792` | Parallelismo righe attention per prefix compatibile. | CPU scheduling; bassa pertinenza CUDA decode. |
| `DS4_BATCHED_ROPE_MAX` / `NO_BATCHED_ROPE` | count 4096, range 0..65536 / bool off | `ds4.c:9799-9806` | Soglia/opt-out RoPE batch. | Prefill only, puo' cambiare memoria temporanea. |
| `DS4_NO_BATCHED_ATTN`, `DS4_BATCHED_FFN`, `DS4_PARALLEL_FFN`, `DS4_NO_SHARED_BATCH_FFN`, `DS4_PREFILL_BATCH` | bool off, off, off, off; count 128 range 1..4095 | `ds4.c:10298-10308` | Attention batch e shared FFN sono on di default; FFN batched/parallel sono opt-in; count regola il batch. | Testare una leva alla volta; possibile regressione/exactness. Non risultano nei run recenti. |
| `DS4_PREFILL_PROFILE_DETAIL`, `DECODE_PROFILE_DETAIL`, `PREFILL_PROFILE_TOKEN` | bool/token | `ds4.c:8183,8494,10018` | Profiling granulare. | Overhead e log, solo diagnosi. |
| `DS4_CPU_DUMP_LOGITS`, `CPU_DUMP_PREFILL_LOGITS`, `ORACLE_LOGITS`, `TRACE_TOP`, `TOKEN_TIMING` | path/bool | `ds4.c:25772-27103` | Dump/trace exactness e timing. | I/O enorme, possibile sincronizzazione; non produzione. |
| `DS4_MOE_REPLAY_SELECTED_IDS`, `DS4_CACHEFIX_TRACE`, `DS4_LOCK_FILE` | path/bool | `ds4.c:19825,27462`, `ds4_cuda.cu:3814` | Replay routing, tracing cachefix e lock processo. | Replay cambia input router; lock coordina processi ma non la GPU internamente. |

## 8. Flag CLI e KV su disco

Il parsing e' stato letto nei frontend, non inferito dal README. Tutte le righe
di questa sezione sono **INFERITE dal codice** salvo i flag marcati
CONFIGURATO/MISURATO. Nessun test CLI e' stato eseguito durante questo censimento.

### Opzioni comuni engine/generazione

| Flag/gruppo | Default | Source V | Effetto e interazioni | Test / rilevanza RTX-WSL |
|---|---|---|---|---|
| `--model PATH` | `ds4flash.gguf` | parser frontend sotto | Modello da aprire; identita', layout e quantizzazione governano tutti gli offset/mask. | CONFIGURATO nei runner; fondamentale. |
| `--cuda`, `--metal`, `--rocm`, `--cpu`, `--backend NAME` | CUDA sui non-Apple con GPU; Metal su Apple; CPU nelle build no-GPU | `ds4_cli.c:1422-1603`, analoghi frontend | Selezionano backend mutuamente esclusivi. Nella build ROCm `--rocm` prende il posto del ramo CUDA. | `--cuda` CONFIGURATO sul 3060; Metal/ROCm/CPU N/A per zero-copy CUDA. |
| `--threads N` | runtime: min(CPU online, 12), max 32 | `ds4.c:1383-1397`; parser frontend | Thread CPU. Interagisce con page fault, pread e WSL scheduling. | Default/config runner osservati; sweep non isolato. |
| `--quality` | off | frontend; `ds4_cuda.cu:5723` | Disabilita TF32/fast paths runtime previsti. Non annulla il `--use_fast_math` compilato da NVCC. | Va fissato uguale negli A/B; semanticamente rilevante, non leva bake. |
| `--mtp`, `--mtp-draft N`, `--mtp-margin F` | MTP off, draft 1, margin 3 | `ds4_cli.c:1393-1410,1458-1464`; analoghi | Speculazione multi-token e soglia di accettazione. `DS4_MTP_SPEC_DISABLE` puo' disabilitare l'argmax speculativo frontend. | Non CONFIGURATO nel bake60; compatibilita' streaming ancora delicata. |
| `--ssd-streaming`, `--ssd-streaming-cold` | off/off | parser comuni; `ds4.c`, `ds4_ssd.c` | Routed experts da storage/cache; `cold` attiva il tier cold. | `--ssd-streaming` CONFIGURATO; cold non bake60. Massima rilevanza. |
| `--ssd-streaming-cache-experts N\|NGB` | auto se non specificato; input >0 | `ds4_ssd.c:46-105` | Tetto cache esperti in count o GiB; auto usa l'envelope memoria e garantisce almeno un esperto. Compete con fixed weights/context/reserve. | Bake60 CONFIGURATO a 400; nessun A/B corrente del valore. |
| `--ssd-streaming-preload-experts N` | **auto hot seed gia' attivo, cap 4096**; cold lo salta | `ds4_help.c:168`; consumer comune `ds4.c:18385-18411` | Con valore CLI zero/unset usa automaticamente il budget cache, limitato dal cap; un valore esplicito sostituisce il count automatico. Non e' pin permanente ne' uno switch di mask per dominio. | Bake60 non imposta il flag ma percorre il default auto; quindi non e' una leva unused. Profilare source/count prima di cambiarlo. |
| `--prefill-chunk N` | auto Metal: 8192 Pro long prompt, 4096 altrimenti; positivo se esplicito | `ds4_help.c:170`; parser comuni | Batch/chunk prefill, temporanei e overlap. Il runner CUDA usa 512. | 512 CONFIGURATO; default Metal N/A CUDA. |
| `--warm-weights` | off | parser comuni | Warmup prima del servizio; cambia page/cache state e startup. | **NON CONFIGURATO** nel runner/env bake60 ne' nei runner 0050 catturati; se testato, separare startup da TTFT. |
| `--simulate-used-memory NGB` | off; valore >0, chunk 256 MiB | `ds4_ssd.c:107-173` | Mappa, tocca e `mlock`a RAM per simulare pressione. | Solo envelope test; alto rischio con bake/pin WSL, non produzione. |
| `--power PERCENT` | 100, range 1..100 | `ds4.c:10878-10903`; parser comuni | Duty-cycle throttle sotto 100. | NON RISULTA; invalida confronti di timing se differisce. |
| `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-attn` | path unset, scale 0; file richiesto se scale !=0 | `ds4.c:11065-11095`; parser | Modifica attivazioni per layer, non memoria. | NON RISULTA; escludere dal baseline exactness. |
| `--ctx N`, `--tokens N` | CLI 32768/50000; server 32768/393216; agent 100000/50000; eval tokens 16000 | init dei frontend | Context live e limite generazione. Aumentano KV/buffer e possono ridurre slot esperti. | Bake60 usa ctx 4096. Altissima rilevanza VRAM/RAM. |
| `--temp`, `--top-p`, `--min-p`, `--seed` | 1, 1, .05; seed non fissato salvo flag | `ds4_cli.c:1404-1409`; analoghi | Sampling e riproducibilita'. | Fissare `temp=0`/seed nei test exactness; non leve di zero-copy. |
| `--think`, `--nothink`, `--think-max` | high | parser comuni | Budget/modalita' reasoning; `think-max` richiede ctx >=393216, altrimenti ricade a high. | Cambia workload/qualita'; non isolato nelle misure bake. |

### Frontend-specifiche

| Flag/gruppo | Default | Source V | Effetto/interazioni | Test / rilevanza |
|---|---|---|---|---|
| CLI `--prompt`, `--prompt-file`, `--system` | prompt unset/REPL; system `You are a helpful assistant` | `ds4_cli.c:1393-1603` | Input e template del workload. | CONFIGURATI per run; semanticamente determinanti. |
| CLI `--dump-logits`, `--dump-logprobs`, `--dump-tokens`, `--logprobs-top-k` | path/off; top-k 20 | `ds4_cli.c:1480-1495` | Dump sincroni e diagnostica exactness. | Non produzione; I/O/sync. |
| CLI inspect/test/profile/imatrix/Metal graph/server | off/unset | `ds4_cli.c:1496-1603` | `--inspect`, `--head-test`, `--first-token-test`, `--perplexity-file`, `--expert-profile`, `--imatrix-*`, `--metal-graph-*` e `--server` avviano modalita' alternative di analisi/dataset/servizio. | Non testati ora; Metal graph N/A CUDA. |
| Server `--host`, `--port`, `--cors`, `--chdir`, `--trace` | 127.0.0.1, 8000, off, unset, off | `ds4_server.c:11514-11666` | Binding, CORS, cwd e tracing. Nessun effetto atteso sui pesi salvo I/O/working directory. | Server compilato `sm_86`; non avviato. |
| Server `--tokens`, `--tool-memory-max-ids`, `--disable-exact-dsml-tool-replay` | 393216, 100000, off | `ds4_server.c:11514-11630`; define `:7764` | Limite risposta/tool memory e opt-out replay tool esatto. Tool memory ha anche limite serializzato 512 MiB. | Non zero-copy; non testato ora. |
| Agent `--non-interactive`, `--prompt`, `--system`, `--trace`, `--chdir` | interactive, prompt/cwd unset, coding system default, trace off | `ds4_agent.c:513-654` | Modalita' agentica, workload, tracing e working directory. | Build storica riuscita; nessun test corrente. |
| Bench `--ctx-start`, `--ctx-max`, `--ctx-alloc`, `--step-incr`, `--step-mul`, `--gen-tokens` | 2048, 32768, auto >max+gen, 2048, 1, 128 | `ds4_bench.c:169-285` | Definisce sweep context e generazione. Prompt/frontier dump/CSV/profile cambiano input/output. | Harness adatto agli A/B futuri; non eseguito ora. |
| Eval budget/casi | tokens 16000, pause 350 ms, soft/hard 1024/512, rank 3 | `ds4_eval.c:1503-1642` | `--questions`, `--case-sequence`, reply budget, trace/regrade/plain e self-test extractor governano il benchmark. | Non zero-copy; non eseguito ora. |
| Distributed `--role`, `--coordinator`, `--listen`, `--layers`, `--dist-prefill-chunk`, `--dist-prefill-window`, `--dist-activation-bits`, `--dist-replay-check`, `--debug` | ruolo unset; chunk session cap (~4096); window workers+2 cap 8; activation 32 bit; replay/debug off | `ds4_distributed.c:8153-8269` | Topologia, assegnazione layer, pipeline prefill, quantizzazione attivazioni e replay. | INFERITO, non testato; N/A nel singolo 3060, possibile contesa rete/I/O. |
| Quantizer input/output/tipi | `--hf`, `--template`, `--out` richiesti salvo dry/compare; threads 8 | `gguf-tools/deepseek4-quantize.c:1740-1810` | Sceglie quant per dense/shared/routed/attention/embedding, tensor override, imatrix, expert count e overwrite. Cambia il modello, quindi capienza, qualita' e offset. | Build tool non provata in questo censimento; rilevante al bake solo a monte. |
| Test selector | tutti off; `--all` li seleziona | `tests/ds4_test.c:2192-2207` | Seleziona suite long context, tool, logprob, Metal, streaming, MTP e server. | Nessuna suite eseguita ora; stato corrente esplicitamente **NON TESTATO**. |

### KV disk server

| Flag | Default | Source V | Effetto/interazioni | Test / rilevanza |
|---|---|---|---|---|
| `--kv-disk-dir PATH` | unset/off | `ds4_server.c:11578`; `ds4_kvstore.c` | Cache di checkpoint per hash del prefix tra sessioni/restart. Non sostituisce il KV live e non accelera un prefix mai visto. | Primo livello per agenti; indipendente dai pesi/zero-copy. |
| `--kv-disk-space-mb MB` | **4096 quando abilitato** | `ds4_help.c:326`; parser `ds4_server.c:11581` | Budget disco/LRU. 8192 e' una configurazione consigliata nei runner/help, non il default. | CONFIGURATO nei setup agentici; monitorare disco/privacy. |
| `--kv-cache-min-tokens`, `--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens` | 512, 30000, 10000 | `ds4_kvstore.c:33-42,164-171` | Soglie store/load cold e checkpoint continuati. | Test embedded presenti nel server, non eseguiti ora. |
| `--kv-cache-boundary-trim-tokens`, `--kv-cache-boundary-align-tokens` | 32, 2048 | stesso | Stabilizzano i confini del prefix. | Non zero-copy; INFERITO. |
| `--kv-cache-reject-different-quant` | off | `ds4_server.c:11595` | Se attivo vieta riuso cross-quant. | Safety/compatibilita', non performance pesi. |

DS4 mantiene un solo KV live in RAM. Il KV disk evita re-prefill di prefix gia'
visti, ma non libera il context buffer CUDA e non ha relazione con la cache
esperti o con la finestra host pinned.

### Inventario completo dei flag long letti

Le liste seguenti sono l'inventario machine-checkable dei literal nei parser;
`--help` e' incluso, mentre `--use_fast_math`/`--offload-arch` sono flag di
toolchain e non compaiono qui.

```text
ds4_cli.c: --backend --cpu --ctx --cuda --dir-steering-attn --dir-steering-ffn --dir-steering-file --dump-logits --dump-logprobs --dump-tokens --expert-profile --first-token-test --head-test --help --imatrix-dataset --imatrix-max-prompts --imatrix-max-tokens --imatrix-out --inspect --logprobs-top-k --metal --metal-graph-full-test --metal-graph-generate --metal-graph-prompt-test --metal-graph-test --min-p --model --mtp --mtp-draft --mtp-margin --nothink --perplexity-file --power --prefill-chunk --prompt --prompt-file --quality --rocm --seed --server --simulate-used-memory --ssd-streaming --ssd-streaming-cache-experts --ssd-streaming-cold --ssd-streaming-preload-experts --system --temp --think --think-max --threads --tokens --top-p --warm-weights
ds4_server.c: --backend --chdir --cors --cpu --ctx --cuda --dir-steering-attn --dir-steering-ffn --dir-steering-file --disable-exact-dsml-tool-replay --help --host --kv-cache-boundary-align-tokens --kv-cache-boundary-trim-tokens --kv-cache-cold-max-tokens --kv-cache-continued-interval-tokens --kv-cache-min-tokens --kv-cache-reject-different-quant --kv-disk-dir --kv-disk-space-mb --metal --model --mtp --mtp-draft --mtp-margin --port --power --prefill-chunk --quality --rocm --simulate-used-memory --ssd-streaming --ssd-streaming-cache-experts --ssd-streaming-cold --ssd-streaming-preload-experts --threads --tokens --tool-memory-max-ids --trace --warm-weights
ds4_agent.c: --backend --chdir --cpu --ctx --cuda --dir-steering-attn --dir-steering-ffn --dir-steering-file --help --metal --min-p --model --mtp --mtp-draft --mtp-margin --non-interactive --nothink --power --prefill-chunk --prompt --quality --seed --simulate-used-memory --ssd-streaming --ssd-streaming-cache-experts --ssd-streaming-cold --ssd-streaming-preload-experts --system --temp --think --think-max --threads --tokens --top-p --trace --warm-weights
ds4_bench.c: --backend --chat-prompt-file --cpu --csv --ctx-alloc --ctx-max --ctx-start --cuda --dump-frontier-logits-dir --expert-profile --gen-tokens --help --metal --model --power --prefill-chunk --prompt-file --quality --rocm --simulate-used-memory --ssd-streaming --ssd-streaming-cache-experts --ssd-streaming-cold --ssd-streaming-preload-experts --step-incr --step-mul --system --threads --tokens --warm-weights
ds4_eval.c: --backend --case-sequence --cpu --ctx --cuda --hard-limit-reply-budget --help --metal --min-p --model --mtp --nothink --pause-ms --plain --power --prefill-chunk --quality --questions --regrade-trace --rocm --seed --self-test-extractors --simulate-used-memory --soft-limit-reply-budget --soft-limit-think-close-rank --ssd-streaming --ssd-streaming-cache-experts --ssd-streaming-cold --ssd-streaming-preload-experts --temp --think --think-max --threads --tokens --top-p --trace --warm-weights
ds4_distributed.c: --coordinator --debug --dist-activation-bits --dist-prefill-chunk --dist-prefill-window --dist-replay-check --layers --listen --role
deepseek4-quantize.c: --attention --attention-proj --attn-proj --compare-gguf --compare-tensor --dense --dry-run --embedding --experts --help --hf --imatrix --imatrix-strict --n-experts --out --output --overwrite --routed --routed-down --routed-gate --routed-up --routed-w1 --routed-w2 --routed-w3 --shared --template --tensor-type --threads
ds4_test.c: --all --help --list --local-golden-vectors --logprob-vectors --long-context --metal-kernels --metal-mpp-equivalence --metal-short-prefill --metal-ssd-streaming-cache-pressure --metal-tensor-equivalence --mtp-verify-depth --server --streaming-decode-prefill-correctness --think-tool-recovery --tool-call-quality
```

Alias short: `-h` = help in tutti i parser che lo espongono; `-m` = model,
`-n` = tokens, `-t` = threads, `-c` = ctx e `-p` = prompt. CLI e agent hanno
`-c/-h/-m/-n/-p/-t`; server ed eval `-c/-h/-m/-n/-t`; bench
`-h/-m/-n/-t`; quantizer e test soltanto `-h`. Default, effetti e stato sono
quelli delle corrispondenti opzioni long.

## 9. Define, soglie e Makefile pertinenti

| Nome/soglia | Valore | Source | Impatto |
|---|---:|---|---|
| `DS4_CUDA_ATTENTION_SCORE_CAP` / `RAW_SCORE_CAP` | 8192 / 256 | `ds4_cuda.cu:35-36` | Cap buffer attention; non leva RAM bake. |
| `DS4_CUDA_TOPK_MERGE_GROUP` | 8 | `ds4_cuda.cu:37` | Geometria top-k CUDA. |
| `DS4_CUDA_ROUTED_EXPERTS_PER_TOKEN` | 6 | `ds4_cuda.cu:38` | Invariante del modello; non confondere con K della mask. |
| `DS4_CUDA_STREAM_EXPERT_DEFAULT` / `MAX` | 512 / 23424 | `ds4_cuda.cu:39-40` | Default/cap slot esperti. |
| `DS4_CUDA_TIERING_MAX_LAYER` / `MAX_EXPERT` | 256 / 512 | `ds4_cuda.cu:136-137` | Dimensione massima metadati tier. |
| `DS4_MAX_THREADS` | 32 | `ds4.c:1315` | Cap thread CPU. |
| `DS4_PACE_NGRAM_MAX` | 512 | `ds4.c:7423` | Cap finestra n-gram PACE. |
| `DS4_LM_WIN_MAX` / `OBS_MAX` | 32 / 16 | `ds4.c:13163-13166` | Cap livemask window/top osservati. |
| `DS4_STREAM_FROM_RAM_MASKED_DEFAULT_BUDGET_GB` | 24 | V `ds4.c:12968` | Default 0050; non e' env ma puo' cambiare solo ricompilando. |
| `DS4_CF_ADMIT_WINDOW_MAX` | 8 | patch 0046 `:18` | Cap storia counterfactual admission. |
| `DS4_STREAM_SELECTED_STAGE_MAX` | 16 | patch 0047 `:15` | Cap ring staging selected. |
| `DS4_PACE_S1_RING_MAX` | 256 | patch 0020 `:49` | Cap finestra slope S1. |
| bias mask hard | `-1e9f` | `ds4.c:7697-7699` | Esclusione effettiva; causa primaria di quality collapse quando K e' stretto. |
| polling mask | ogni 32 token | `ds4.c:7722-7748` | Latenza di applicazione degli switch dinamici. |
| mlock simulate | chunk 256 MiB | `ds4_ssd.c:136` | Picco/diagnosticabilita' del test RAM. |
| host pinned WDDM | 31.9453 GiB | `docs/WSL_WDDM_PINNED_LIMIT_20260713.md` | Vincolo esterno MISURATO: meta' dei 63.8905 GiB fisici. 24 GiB e' il default runtime corrente; 28 GiB e' riuscito solo nel probe standalone e resta non provato dentro DS4. |
| KV disk default | 4096 MiB | V `ds4_help.c:326` | 8192 e' un esempio/configurazione, non il default. |
| `think-max` context minimo | 393216 token | frontend/help | Sotto la soglia la modalita' massima non e' disponibile. |
| `CUDA_ARCH` | empty; `native`; esplicito | `Makefile:19-31,91-107` | Per RTX 3060 usare `make cuda CUDA_ARCH=sm_86`; `cuda-spark` non e' target adatto. |
| `NVCCFLAGS` | `-O3 -g -lineinfo --use_fast_math` | `Makefile:25` | Math fast e lineinfo; tenere build identica negli A/B. |

### Define compilabili

| Define | Default / source | Effetto e interazioni | Test / pertinenza |
|---|---|---|---|
| `DS4_NO_GPU` | assente; aggiunto dai target CPU, `Makefile:158-176` | Esclude backend GPU. | Build CPU non verificata ora; N/A al 3060 zero-copy. |
| `DS4_ROCM_BUILD` | assente; target `strix-halo`, `Makefile:112-117` | Seleziona sorgente/HIP e semantica CLI ROCm. | N/A CUDA, non testato. |
| `DS4_DIST_TRACE` | assente; guard in `ds4_distributed.c` | Compila tracing distribuito addizionale. | Non testato; N/A singolo nodo. |
| `DS4_LM_WIN_MAX`, `DS4_LM_OBS_MAX` | 32/16 se non definiti, F/V `ds4.c` | Consentono override compile-time dei cap livemask. Aumentarli cambia memoria/stato e richiede rebuild. | Default compilato; nessun A/B. |
| `DS4_SERVER_TEST`, `DS4_SERVER_TEST_NO_MAIN`, `DS4_AGENT_TEST`, `DS4_AGENT_TEST_NO_MAIN` | assenti; test-only | Espongono harness o sopprimono main per unit test. | Non runtime; suite non eseguita ora. |
| `_GNU_SOURCE`, `__HIP_PLATFORM_AMD__` | Linux CFLAGS / ROCm CFLAGS | Feature libc e selezione HIP. | `_GNU_SOURCE` osservato nella build; HIP N/A. |
| `DS4_METAL_HAS_TENSOR` | generato internamente in `ds4_metal.m:4613` | Macro dello shader Metal, non env utente. | N/A CUDA. |

### Makefile

I Makefile principali F e V sono identici. Le variabili sono override di build
e quindi leve anche se non iniziano con `DS4_`.

| Variabile/target | Default | Effetto e interazioni | Stato corrente |
|---|---|---|---|
| `CC`, `UNAME_S` | `cc`; `uname -s` auto | Compilatore host e ramo Darwin/Linux. | INFERITO; build corrente Linux. |
| `NATIVE_CPU_FLAG` | `-mcpu=native` Darwin, `-march=native` altri | ISA host per C e via `-Xcompiler` per NVCC. Riduce portabilita' binario. | Presente nei log build. |
| `DEBUG_FLAGS` | `-g` | Simboli debug pur con `-O3`; aumenta binario, non semantica attesa. | Presente. |
| `CFLAGS` | `-O3 -ffast-math -g <native> -Wall -Wextra -std=c99`; Linux aggiunge `-D_GNU_SOURCE -fno-finite-math-only` | Ottimizzazione/matematica di tutto il C. Override totale puo' perdere flag Linux. | Build `0050` riuscita; nessun test exactness da cambio flag. |
| `OBJCFLAGS` | analogo C + `-fobjc-arc` | Build Metal Objective-C. | N/A CUDA. |
| `LDLIBS`, `METAL_LDLIBS` | `-lm -pthread`; framework Metal/Foundation su Darwin | Link host/Metal. | Link CUDA riuscito; Metal N/A. |
| `CUDA_HOME`, `NVCC` | `/usr/local/cuda`, `$CUDA_HOME/bin/nvcc` | Toolchain CUDA. | OSSERVATO nei log. |
| `CUDA_ARCH`, `NVCC_ARCH_FLAGS` | vuoto; `-arch` solo se valorizzato | Target GPU. `sm_86` e' corretto per RTX 3060; `native` dipende dal toolchain/host. | `build_0050i.log` compila/linka server `sm_86`. |
| `NVCCFLAGS` | `-O3 -g -lineinfo --use_fast_math ... -pthread` | Ottimizzazione, lineinfo e fast math CUDA. `--quality` runtime non rimuove questo flag. | OSSERVATO in build; fissare negli A/B. |
| `CUDA_LDLIBS` | math, pthread, CUDA lib path, `cudart`, `cublas` | Runtime e BLAS CUDA; path include SBSA e lib64. | Link riuscito. |
| `HIPCC`, `ROCM_ARCH`, `ROCM_CFLAGS`, `ROCM_LDLIBS` | hipcc auto/fallback; gfx1151; O3/fast-math; hipBLAS/Lt | Toolchain Strix Halo. | N/A RTX, non testato. |
| `CORE_OBJS`, `CPU_CORE_OBJS` | backend-specific object set | Confine del backend linkato. Override errato puo' mescolare CUDA/ROCm/Metal. | Default CUDA osservato. |
| `DS4_LINK`, `DS4_LINK_LIBS` | NVCC+NVCCFLAGS; CUDA libs | Linker e librerie finali; il target ROCm li sostituisce. | Link server riuscito. |
| `cuda-spark`, `cuda-generic`, `cuda CUDA_ARCH=...` | arch vuota, native, esplicita richiesta | Ricompilano tutti i cinque frontend. Per 3060 usare il terzo con `sm_86`. | Build 0050 precedenti: cinque binari riusciti; `0050i`: server riuscito. |
| `cpu`, `strix-halo`/`rocm` | target espliciti | Build CPU o ROCm, sovrascrivono backend/define/link. | Non testati ora. |
| `test` | build `ds4_test`, `ds4_agent_test`, eval extractor e q4k dot, poi esecuzione | Suite CPU/runtime ampia; puo' caricare modelli/GPU a seconda env. | **NON ESEGUITO** per mandato. |
| `cuda-regression` | esegue `tests/cuda_long_context_smoke` | Smoke CUDA; timeout controllabile da `DS4_CUDA_TOPK_REGRESSION_SEC`. | **NON ESEGUITO** per mandato. |
| `METAL_SRCS`, `ROCM_SRCS` | wildcard `metal/*.metal`, `rocm/*.cuh` | Dipendenze che forzano rebuild del backend. | N/A CUDA per Metal; ROCm non testato. |

Inventario target Makefile (i target oggetto sono regole di dipendenza, non
leve runtime):

```text
.PHONY all clean cpu cuda cuda-generic cuda-regression cuda-spark ds4 ds4.o ds4_agent.o ds4_agent_cpu.o ds4_agent_test ds4_agent_test.o ds4_bench.o ds4_bench_cpu.o ds4_cli.o ds4_cli_cpu.o ds4_cpu.o ds4_cuda.o ds4_distributed.o ds4_eval.o ds4_eval_cpu.o ds4_help.o ds4_kvstore.o ds4_metal.o ds4_rocm.o ds4_server.o ds4_server_cpu.o ds4_ssd.o ds4_test ds4_test.o ds4_web.o ds4-agent ds4-bench ds4-eval ds4-server help linenoise.o q4k-dot-test rax.o rocm strix-halo test tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
```

Soglie strutturali lette ma non esposte come env/CLI: rope original context
65536 (`DS4_DEFAULT_ROPE_ORIG_CTX`), tensor max dims 8, Metal model views 4096,
server I/O timeout 10 s e send-stall 2000 ms, web port 9333/connect 3000 ms/CDP
20000 ms, tool-memory default 100000 ID, KV fixed header 48 byte e GGUF default
alignment 32. Magic/versioni sessione e messaggi distributed sono vincoli di
protocollo/formato, non tuning: modificarli richiede compatibilita' producer-
consumer e rebuild. Sono **INFERITI**, non testati in questo censimento; solo i
timeout/server possono influire indirettamente sui run, nessuno cambia bake o
zero-copy.

### Riferimento upstream GB10/ATS

Il commit upstream [`15f42aafd`](https://github.com/antirez/ds4/commit/15f42aafd31dc1bdf8f74b37178605e52bbc2504)
aggiunge il tier DGX Spark/GB10: 121 GiB UMA, accesso host mmap via ATS,
cache HBM dei tensori non-routed e routed experts (~65 GiB) lasciati su ATS.
Introduce `DS4_CUDA_NO_HBM_CACHE`, porta il cap default della weight cache a
24 GiB e riporta circa 13.9 -> 16.13 t/s nel suo setup. Questi numeri e il cap
non sono trasferibili alla RTX 3060 discreta; nello snapshot locale la leva
`NO_HBM_CACHE` non esiste e il default `WEIGHT_CACHE_LIMIT_GB` e' ormai 96.
E' trasferibile solo il principio: proteggere un budget piccolo per tensori
fissi/hot e streammare i routed experts, senza pretendere letture fini ATS
efficienti su PCIe/WSL.

## 10. Shortlist concreta

### Fatti misurati o direttamente osservati

1. **RAM bake, fixture statica pre-0050**: `KEEP_MODEL_PAGES=1`,
   `NO_DIRECT_IO=1`, mask60 e `DS4_CUDA_NO_Q8_F16_CACHE=1` hanno prodotto un
   render L2 completo a `n=1`. Non prova il path DMA 0050, non e' un baseline
   qualita' 0051 e non e' un risultato `n>=3`.
2. **0050 statica**: il fix di lifetime e' nel V2 e i log WSL riportano copie
   DMA dirette dalla mmap registrata: 938.25 MiB, zero `dma_failed`. Questo non
   prova un vantaggio end-to-end; i raw artifact devono ancora essere copiati
   nel repo e l'exactness post-hardening manca.
3. **Pinned WSL**: 31 GiB contigui passano nel probe e il 32esimo GiB fallisce
   al tetto WDDM 31.9453 GiB. Questo non rende 31 o 28 GiB budget DS4 sicuri:
   24 GiB e' l'unico default runtime corrente, 28 GiB resta non provato.
4. **0051 design-only**: non esiste ancora un'arena dinamica attivabile. Deve
   allocare dopo le risorse pinned obbligatorie, escludere la 0050 statica e
   reclamare le pagine sorgente duplicate; `KEEP_MODEL_PAGES=1` non e' un
   default trasferibile.
5. **Patch 0048**: dove e' applicata **integralmente**, lo storico misura
   `PREFILL_DEFER_UPLOAD_SYNC=1` bit-exact e vincente; il readahead e' refutato
   su WSL2. Nel V2 corrente entrambi i nomi sono patch-only/inerti.
6. **Qualita'**: non rimettere nel baseline K8/K23/K32, soft-mask o SPEX
   additivo. Mask60 per massa e' soltanto la fixture larga piu' promettente;
   richiede ancora la matrice graduata `n>=3` e in 0051 non deve diventare una
   policy statica di produzione.
7. **KV disk**: tenerlo abilitato con budget esplicito per gli agenti. Misura
   prefix-hit e spazio separatamente; non attribuirgli miglioramenti decode.

### Leve unused o ad alto valore

1. **Chiudere 0050**: ripetere exactness ON/OFF sul `build_0050i` o successivo,
   poi A/B OFF/ON back-to-back allo stesso budget 24 GiB, con ordine alternato,
   warm state bracketed, stessi env/CLI e diagnostica attiva soltanto nel
   braccio di attribuzione. Archiviare nel repo diag, stage ed exactness raw.
2. **Implementare 0051 a gate**: prima allocator/lifecycle dopo staging/context,
   poi lookup/eventi/exactness, quindi policy live. Tenere 24 GiB default;
   valutare 28 soltanto dopo startup/cleanup ripetuti e A/B end-to-end.
3. **Attribuzione cache**: usare per un solo smoke
   `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE=1` e, se necessario, `...VERBOSE`
   per misurare slot/hit/load prima di cambiare budget.
4. **Graph comune CUDA**: profilare l'hot seed automatico e A/B
   `DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED=1` con K piccolo. Nonostante
   il prefisso `METAL`, queste leve sono nel path comune CUDA; non combinarle
   subito con PACE pin o 0051.
5. **Reserve VRAM**: sweep controllato 0.5 vs 1.0 GiB, registrando slot
   effettivi, picco VRAM, TTFT e decode. Il valore 1 e' configurato, ma il suo
   vantaggio rispetto al default 0.5 non e' isolato.
6. **Arena fissi 256**: non mantenerlo come baseline. I pod12 hanno otto OOM
   dell'arena anche con `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`; confrontarlo con
   il default solo come arm diagnostico, con env, allocazioni e OOM verificati.
7. **Weight cache budget**: prima provare con il profilo che un cap sia davvero
   binding. Il bake osservato si ferma a 7.86 GiB, quindi 8/9 GiB possono essere
   nulli; abortire se il fallback dei tensori fissi domina.
8. **In-place residency**: solo se `DS4_CUDA_INPLACE_RESIDENT` viene riattivato,
   profilare `DS4_CUDA_INPLACE_RESERVE_SLOTS` contro il default `capacity/8+1`;
   richiede verify/exactness e non va combinato subito con il nuovo pin host.
9. **Stage/overlap 0047-0048**: `SELECTED_STAGE_DEPTH` 4 vs 8 e defer sync sono
   ad alto valore, ma richiedono prima l'applicazione integrale delle patch.
   `NO_WHOLE_MMAP_REGISTER=1` esportato da solo non fa nulla nel V2 corrente.
10. **Rating-only 0051**: implementare `DS4_PACE_LIVEMASK_MODE` e il guard di
    pubblicazione prima di usare la livemask come osservatore. Finche' manca,
    non esiste un test sicuro della policy live e mask60 resta solo fixture.

## 11. Inventario sorgente verificabile

Le sezioni 11.1-11.4 conservano la mappa dei 419 nomi DS4 del core
`ds4.c`/`ds4_cuda.cu` nello snapshot F. La sezione 15 aggiunge tutti i nomi
source-wide e le tre leve 0050 V, portando la copertura al censimento completo
dei 584 token quoted. Per i gruppi di kernel `ENABLE/FORCE` significa opt-in e
`DISABLE/NO` opt-out, default unset; l'effetto e' quello espresso dal nome del
kernel. Sono leve di benchmark/regressione, con rischio di prestazioni,
precisione matematica o layout. Salvo le eccezioni gia' documentate, non
risultano negli env catturati dei run. I path di profile/trace aggiungono I/O.

Schema applicato a ogni voce dei code block: **default** unset/off salvo
override numerico tabulato; **location** e' la riga riportata; **effect** e'
l'abilitazione/disabilitazione letterale del ramo nominato; **interaction** e'
il pair `ENABLE/DISABLE`, il dispatcher automatico e l'eventuale profile/sync;
**test** e' INFERITO + NON RISULTA salvo stato esplicito nelle tabelle;
**relevance** e' alta per i nomi CUDA/REAP che toccano streaming, residency,
page cache o matematica. I nomi consumati esclusivamente da `ds4_metal.m` sono
N/A CUDA; alcuni nomi legacy `DS4_METAL_*` consumati nel graph comune di
`ds4.c` sono invece applicabili anche a CUDA e sono marcati in 4/11.4. Le
eccezioni con default non booleano, prove MISURATE o rischio specifico sono tutte tabulate
nelle sezioni 3-9 e 15.

### 11.1 CUDA: inventario completo

Quasi tutte le voci low-level sono bool presence, con il kernel ottimizzato
auto-selezionato quando le `NO_/DISABLE_` sono assenti. L'unica soglia
numerica non gia' tabulata e' `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN`:
default 2 token, valori accettati 2..4095 (`ds4_cuda.cu:12749-12758`).

```text
DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN @ ds4_cuda.cu:12752
DS4_CUDA_ATTN_Q_B_F32_CACHE @ ds4_cuda.cu:1598
DS4_CUDA_COPY_MODEL @ ds4_cuda.cu:2702
DS4_CUDA_COPY_MODEL_CHUNKED @ ds4_cuda.cu:5589
DS4_CUDA_DIRECT_MODEL @ ds4.c:2287
DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED @ ds4_cuda.cu:16540
DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED @ ds4_cuda.cu:16711
DS4_CUDA_DISABLE_QKV_RMS_FUSED @ ds4_cuda.cu:11601
DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR @ ds4_cuda.cu:12950
DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR @ ds4.c:12035
DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD @ ds4.c:18545
DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP @ ds4.c:16082
DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST @ ds4_cuda.cu:3321
DS4_CUDA_HOST_REGISTER_PLAIN @ ds4_cuda.cu:1228
DS4_CUDA_INDEXED_TWOPASS @ ds4_cuda.cu:12462
DS4_CUDA_INPLACE_RESERVE_SLOTS @ ds4_cuda.cu:4536
DS4_CUDA_INPLACE_RESIDENT @ ds4_cuda.cu:3978
DS4_CUDA_INPLACE_VERIFY @ ds4_cuda.cu:3979
DS4_CUDA_KEEP_MODEL_PAGES @ ds4_cuda.cu:2794
DS4_CUDA_MODEL_COPY_CHUNK_MB @ ds4_cuda.cu:2781
DS4_CUDA_MODEL_COPY_VERBOSE @ ds4_cuda.cu:3198
DS4_CUDA_MODEL_PREFETCH_SYNC @ ds4_cuda.cu:2762
DS4_CUDA_MOE_ATOMIC_DOWN @ ds4_cuda.cu:15828
DS4_CUDA_MOE_DOWN_ROW1024 @ ds4_cuda.cu:15849
DS4_CUDA_MOE_DOWN_ROW128 @ ds4_cuda.cu:15853
DS4_CUDA_MOE_DOWN_ROW2048 @ ds4_cuda.cu:15851
DS4_CUDA_MOE_DOWN_ROW256 @ ds4_cuda.cu:15852
DS4_CUDA_MOE_DOWN_ROW512 @ ds4_cuda.cu:15848
DS4_CUDA_MOE_DOWN_ROW64 @ ds4_cuda.cu:15854
DS4_CUDA_MOE_GATE_ROW128 @ ds4_cuda.cu:15833
DS4_CUDA_MOE_GATE_ROW2048 @ ds4_cuda.cu:15831
DS4_CUDA_MOE_GATE_ROW256 @ ds4_cuda.cu:15832
DS4_CUDA_MOE_GATE_ROW512 @ ds4_cuda.cu:15845
DS4_CUDA_MOE_NO_ATOMIC_DOWN @ ds4_cuda.cu:15827
DS4_CUDA_MOE_NO_DECODE_LUT_GATE @ ds4_cuda.cu:15843
DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6 @ ds4_cuda.cu:15862
DS4_CUDA_MOE_NO_DOWN_ROW128 @ ds4_cuda.cu:15858
DS4_CUDA_MOE_NO_DOWN_ROW2048 @ ds4_cuda.cu:15856
DS4_CUDA_MOE_NO_DOWN_ROW256 @ ds4_cuda.cu:15857
DS4_CUDA_MOE_NO_DOWN_ROW64 @ ds4_cuda.cu:15859
DS4_CUDA_MOE_NO_DOWN_TILE16 @ ds4_cuda.cu:15839
DS4_CUDA_MOE_NO_EXPERT_TILES @ ds4_cuda.cu:15822
DS4_CUDA_MOE_NO_GATE_ROW128 @ ds4_cuda.cu:15837
DS4_CUDA_MOE_NO_GATE_ROW2048 @ ds4_cuda.cu:15835
DS4_CUDA_MOE_NO_GATE_ROW256 @ ds4_cuda.cu:15836
DS4_CUDA_MOE_NO_P2 @ ds4_cuda.cu:15825
DS4_CUDA_MOE_NO_Q4_EXPERT_TILES @ ds4_cuda.cu:15820
DS4_CUDA_MOE_PROFILE @ ds4_cuda.cu:15806
DS4_CUDA_MOE_TILE4 @ ds4_cuda.cu:15823
DS4_CUDA_MOE_WRITE_GATE_UP @ ds4_cuda.cu:15824
DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE @ ds4_cuda.cu:1572
DS4_CUDA_NO_ATTN_Q_B_F16_CACHE @ ds4_cuda.cu:1575
DS4_CUDA_NO_CUBLAS_ATTENTION @ ds4_cuda.cu:12197
DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A @ ds4_cuda.cu:12761
DS4_CUDA_NO_DIRECT_IO @ ds4_cuda.cu:5657
DS4_CUDA_NO_F16_PAIR_MATMUL @ ds4_cuda.cu:11471
DS4_CUDA_NO_FD_CACHE @ ds4_cuda.cu:1366
DS4_CUDA_NO_INDEXED_HEADS8 @ ds4_cuda.cu:12461
DS4_CUDA_NO_INDEXED_TOPK_SORT @ ds4_cuda.cu:12452
DS4_CUDA_NO_INDEXER_DIRECT_ONE @ ds4_cuda.cu:10750
DS4_CUDA_NO_INDEXER_WMMA @ ds4_cuda.cu:10760
DS4_CUDA_NO_INDEXER_WMMA128 @ ds4_cuda.cu:10761
DS4_CUDA_NO_INDEXER_WMMA32 @ ds4_cuda.cu:10779
DS4_CUDA_NO_INDEXER_WMMA64 @ ds4_cuda.cu:10770
DS4_CUDA_NO_MODEL_COPY @ ds4_cuda.cu:3133
DS4_CUDA_NO_MODEL_PREFETCH @ ds4_cuda.cu:2701
DS4_CUDA_NO_ORDERED_F16_MATMUL @ ds4_cuda.cu:11327
DS4_CUDA_NO_PARALLEL_ROUTER_SELECT @ ds4_cuda.cu:13015
DS4_CUDA_NO_Q8_BATCH_WARP @ ds4_cuda.cu:11131
DS4_CUDA_NO_Q8_DP4A @ ds4_cuda.cu:1591
DS4_CUDA_NO_Q8_F16_CACHE @ ds4.c:13348
DS4_CUDA_NO_Q8_F32_CACHE @ ds4_cuda.cu:1595
DS4_CUDA_NO_TF32 @ ds4_cuda.cu:5160
DS4_CUDA_NO_TOPK_CHUNKED @ ds4_cuda.cu:10941
DS4_CUDA_NO_TOPK1024 @ ds4_cuda.cu:10866
DS4_CUDA_NO_TOPK2048 @ ds4_cuda.cu:10873
DS4_CUDA_NO_TOPK8192 @ ds4_cuda.cu:10911
DS4_CUDA_NO_WARP_ROUTER_SELECT @ ds4_cuda.cu:13014
DS4_CUDA_NO_WINDOW_ATTENTION @ ds4_cuda.cu:12135
DS4_CUDA_Q8_F16_ALL @ ds4_cuda.cu:1566
DS4_CUDA_Q8_F16_CACHE_MB @ ds4_cuda.cu:1426
DS4_CUDA_Q8_F16_CACHE_RESERVE_MB @ ds4_cuda.cu:1432
DS4_CUDA_Q8_F32_ALL @ ds4_cuda.cu:1596
DS4_CUDA_Q8_F32_LARGE @ ds4_cuda.cu:1600
DS4_CUDA_Q8_F32_PRELOAD @ ds4_cuda.cu:5702
DS4_CUDA_SELECTED_UPLOAD_EVENT @ ds4_cuda.cu:3616
DS4_CUDA_SERIAL_F16_MATMUL @ ds4_cuda.cu:11317
DS4_CUDA_SERIAL_ROUTER @ ds4_cuda.cu:11322
DS4_CUDA_STREAMING_EXPERT_CACHE_N @ ds4_cuda.cu:3291
DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE @ ds4.c:18471
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB @ ds4_cuda.cu:3339
DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE @ ds4_cuda.cu:6395
DS4_CUDA_STREAMING_EXPERT_HOTLIST @ ds4_cuda.cu:3323
DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE @ ds4.c:18560
DS4_CUDA_STRICT_WEIGHT_CACHE @ ds4_cuda.cu:3055
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB @ ds4_cuda.cu:2963
DS4_CUDA_WEIGHT_CACHE @ ds4_cuda.cu:1359
DS4_CUDA_WEIGHT_CACHE_LIMIT_GB @ ds4_cuda.cu:2934
DS4_CUDA_WEIGHT_CACHE_VERBOSE @ ds4_cuda.cu:1249
DS4_CUDA_WEIGHT_PRELOAD @ ds4_cuda.cu:1360
DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB @ ds4.c:2114
DS4_CUDA_WINDOW_ATTENTION @ ds4_cuda.cu:12181
```

### 11.2 PACE e REAP: inventario completo

Per questo blocco tipo/default/effetto sono specificati nelle sezioni 4-5.
I nomi `MAX_*` sono limiti compile-time usati dal relativo controller, non env.

```text
DS4_PACE @ ds4.c:13377
DS4_PACE_ALPHA_HIT @ ds4.c:13438
DS4_PACE_ALPHA_NGRAM @ ds4.c:13437
DS4_PACE_ANNEAL_WARM @ ds4.c:13442
DS4_PACE_BREATH_EVERY @ ds4.c:13396
DS4_PACE_BREATH_EVERY_MAX @ ds4.c:13398
DS4_PACE_BREATH_EVERY_MIN @ ds4.c:13397
DS4_PACE_BREATH_KEEP @ ds4.c:13400
DS4_PACE_BREATH_LEN @ ds4.c:13399
DS4_PACE_CACHE_FLOOR @ ds4.c:29732
DS4_PACE_CACHE_FLUSH @ ds4.c:13452
DS4_PACE_CACHE_TARGET_SLOTS @ ds4.c:29736
DS4_PACE_DEBUG @ ds4.c:13380
DS4_PACE_DRIFT @ ds4.c:13393
DS4_PACE_EXCHANGE_OBSERVE @ ds4.c:13455
DS4_PACE_HIT_HI @ ds4.c:13440
DS4_PACE_HYST @ ds4.c:13395
DS4_PACE_KEEP @ ds4.c:13385
DS4_PACE_KEEP_MAX @ ds4.c:13387
DS4_PACE_KEEP_MIN @ ds4.c:13386
DS4_PACE_KEEP_STEP @ ds4.c:13388
DS4_PACE_LIVEMASK @ ds4.c:13177
DS4_PACE_LIVEMASK_BLOCKED_WEIGHT @ ds4.c:13269
DS4_PACE_LIVEMASK_BOOTSTRAP @ ds4.c:13180
DS4_PACE_LIVEMASK_COOLDOWN @ ds4.c:13259
DS4_PACE_LIVEMASK_DEMOTE_HORIZON @ ds4.c:13271
DS4_PACE_LIVEMASK_HYST @ ds4.c:13264
DS4_PACE_LIVEMASK_K @ ds4.c:13186
DS4_PACE_LIVEMASK_K_ADAPTIVE @ ds4.c:13192
DS4_PACE_LIVEMASK_K_MAX @ ds4.c:13196
DS4_PACE_LIVEMASK_K_MIN @ ds4.c:13194
DS4_PACE_LIVEMASK_KNOCK_DEADBAND @ ds4.c:13221
DS4_PACE_LIVEMASK_KNOCK_GAIN @ ds4.c:13211
DS4_PACE_LIVEMASK_KNOCK_MIN_HISTORY @ ds4.c:13231
DS4_PACE_LIVEMASK_KNOCK_PREFETCH @ ds4.c:13229
DS4_PACE_LIVEMASK_KNOCK_STEP_DOWN @ ds4.c:13217
DS4_PACE_LIVEMASK_KNOCK_STEP_UP @ ds4.c:13214
DS4_PACE_LIVEMASK_KNOCK_THRESHOLD @ ds4.c:13205
DS4_PACE_LIVEMASK_KNOCK_UPDATE_EVERY @ ds4.c:13225
DS4_PACE_LIVEMASK_LOG @ ds4.c:13311
DS4_PACE_LIVEMASK_MAX_SWAPS @ ds4.c:13241
DS4_PACE_LIVEMASK_OBSERVE_TOP @ ds4.c:13236
DS4_PACE_LIVEMASK_PIN @ ds4.c:13266
DS4_PACE_LIVEMASK_PRESSURE @ ds4.c:13288
DS4_PACE_LIVEMASK_PRESSURE_COOLDOWN @ ds4.c:13292
DS4_PACE_LIVEMASK_PRESSURE_KNOCK @ ds4.c:13290
DS4_PACE_LIVEMASK_PRESSURE_MAX_SWAPS @ ds4.c:13296
DS4_PACE_LIVEMASK_PRESSURE_X @ ds4.c:13294
DS4_PACE_LIVEMASK_PROMO_RATE_CAP @ ds4.c:13276
DS4_PACE_LIVEMASK_SPEX_ADD @ ds4.c:17569
DS4_PACE_LIVEMASK_SPEX_CADENCE @ ds4.c:17574
DS4_PACE_LIVEMASK_SPEX_LOG @ ds4.c:17623
DS4_PACE_LIVEMASK_SPEX_PIN_LOG @ ds4_cuda.cu:4514
DS4_PACE_LIVEMASK_SPEX_WRAP @ ds4.c:17612
DS4_PACE_LIVEMASK_WEIGHTED @ ds4.c:13265
DS4_PACE_LIVEMASK_WINDOW @ ds4.c:13182
DS4_PACE_LIVEMASK_X @ ds4.c:13239
DS4_PACE_LOG @ ds4.c:13381
DS4_PACE_NGRAM_N @ ds4.c:13389
DS4_PACE_NGRAM_WINDOW @ ds4.c:13390
DS4_PACE_PIN @ ds4_cuda.cu:3935
DS4_PACE_PIN_BUDGET_MB @ ds4_cuda.cu:3937
DS4_PACE_PIN_COOLDOWN @ ds4_cuda.cu:3945
DS4_PACE_PIN_CUSUM_H @ ds4_cuda.cu:3943
DS4_PACE_PIN_CUSUM_K @ ds4_cuda.cu:3942
DS4_PACE_PIN_EWMA @ ds4_cuda.cu:3940
DS4_PACE_PIN_LOG @ ds4_cuda.cu:3947
DS4_PACE_PIN_ROTATE @ ds4_cuda.cu:3936
DS4_PACE_PIN_WARMUP @ ds4_cuda.cu:3939
DS4_PACE_PREBREATH @ ds4.c:13401
DS4_PACE_PREBREATH_ADAPT @ ds4.c:13421
DS4_PACE_PREBREATH_ADAPT_GAIN @ ds4.c:13422
DS4_PACE_PREBREATH_ADAPT_POWER @ ds4.c:13424
DS4_PACE_PREBREATH_DRIFT @ ds4.c:13402
DS4_PACE_PREBREATH_EVERY @ ds4.c:13413
DS4_PACE_PREBREATH_KEEP_MAX @ ds4.c:13415
DS4_PACE_PREBREATH_RELEARN @ ds4.c:13428
DS4_PACE_PREBREATH_RELEARN_DECAY @ ds4.c:13430
DS4_PACE_PREBREATH_STEP_MAX @ ds4.c:13426
DS4_PACE_PREBREATH_TARGET @ ds4.c:13407
DS4_PACE_PREFILL_APPLY @ ds4.c:13453
DS4_PACE_PREFILL_WAIT_WRAP @ ds4.c:13454
DS4_PACE_RELEARN @ ds4.c:13435
DS4_PACE_RELEARN_DECAY @ ds4.c:13436
DS4_PACE_RELEARN_ON_TIGHTEN @ ds4.c:13465
DS4_PACE_RELEASE @ ds4.c:13394
DS4_PACE_ROTATE @ ds4.c:13456
DS4_PACE_ROTATE_DECAY @ ds4.c:13459
DS4_PACE_ROTATE_EVERY @ ds4.c:13457
DS4_PACE_ROTATE_PRESERVE_STABLE @ ds4.c:13463
DS4_PACE_S1 @ ds4.c:13443
DS4_PACE_STABLE @ ds4.c:13441
DS4_PACE_TIER @ ds4_cuda.cu:4227
DS4_PACE_TIER_COOLDOWN @ ds4_cuda.cu:4243
DS4_PACE_TIER_DECAY @ ds4_cuda.cu:4239
DS4_PACE_TIER_HYST @ ds4_cuda.cu:4233
DS4_PACE_TIER_KNOCK @ ds4_cuda.cu:4241
DS4_PACE_TIER_LOG @ ds4_cuda.cu:4245
DS4_PACE_TIER_VRAM_SLOTS @ ds4_cuda.cu:4236
DS4_PACE_TIER_WARMUP @ ds4_cuda.cu:4228
DS4_PACE_TIER_X @ ds4_cuda.cu:4229
DS4_PACE_TIER_Y @ ds4_cuda.cu:4231
DS4_PACE_TIGHTEN_LO @ ds4.c:13439
DS4_PACE_WARMUP @ ds4.c:13384
DS4_PACE_WEIGHTED_RELEARN @ ds4.c:13471
DS4_PACE_WEIGHTED_SELECTED @ ds4.c:13467
DS4_PACE_WEIGHTED_WARMUP @ ds4.c:13469
DS4_PACE_WRAP @ ds4.c:12882
DS4_PACE_WRAP_ROTATE @ ds4.c:13451
DS4_REAP_MASK_FILE @ ds4.c:7722
DS4_REAP_PIN_BY_MASS @ ds4.c:13310
DS4_REAP_PREFETCH @ ds4.c:12884
DS4_REAP_PREFETCH_DELTA @ ds4.c:12947
DS4_REAP_PREFETCH_LOCK @ ds4.c:12841
DS4_REAP_PREFETCH_THREADS @ ds4.c:12804
DS4_REAP_SENSOR_LOG @ ds4.c:17453
DS4_REAP_WRAP @ ds4.c:12883
```

### 11.3 SPEX, expert, MTP e generali: inventario completo

Le leve generali di kernel/batching sono force/opt-out bool salvo il valore
numerico esplicito nel nome o nella sezione 7. Le leve `*_LOG`, `*_FILE`,
`*_PATH`, dump e trace sono path. Default unset; nessuna e' baseline bake60
se non indicato nelle sezioni precedenti.

```text
DS4_BATCHED_FFN @ ds4.c:10299
DS4_BATCHED_ROPE_MAX @ ds4.c:9799
DS4_CACHEFIX_TRACE @ ds4_cuda.cu:3814
DS4_CPU_DUMP_LOGITS @ ds4.c:25802
DS4_CPU_DUMP_PREFILL_LOGITS @ ds4.c:27090
DS4_DECODE_PROFILE_DETAIL @ ds4.c:8183
DS4_EXPERT_COLD_ALLOW_UNGATED @ ds4_cuda.cu:2201
DS4_EXPERT_COLD_FORMAT @ ds4_cuda.cu:2187
DS4_EXPERT_COLD_NATIVE_LAYERS @ ds4_cuda.cu:2205
DS4_EXPERT_COLD_NATIVE_TOKENS @ ds4_cuda.cu:2202
DS4_EXPERT_COLD_RAM_LOSSLESS @ ds4_cuda.cu:2197
DS4_EXPERT_COLD_RAM_PREFILL @ ds4_cuda.cu:2200
DS4_EXPERT_COLD_RAM_VERBOSE @ ds4_cuda.cu:2210
DS4_EXPERT_COLD_RAM_VERIFY @ ds4_cuda.cu:2209
DS4_EXPERT_HOTLIST @ ds4.c:29939
DS4_EXPERT_PROFILE @ ds4.c:29937
DS4_EXPERT_TIER_POLICY @ ds4_cuda.cu:320
DS4_EXPERT_TIER_PROMOTE_CAP @ ds4_cuda.cu:331
DS4_EXPERT_TIER_PROMOTE_VERBOSE @ ds4_cuda.cu:335
DS4_EXPERT_TIERING @ ds4_cuda.cu:300
DS4_EXPERT_TIERING_LOG @ ds4_cuda.cu:356
DS4_EXPERT_TIERING_LOG_IDS @ ds4_cuda.cu:373
DS4_EXPERT_TIERING_SUMMARY_EVERY @ ds4_cuda.cu:564
DS4_LOCK_FILE @ ds4.c:27462
DS4_MOE_REPLAY_SELECTED_IDS @ ds4.c:19825
DS4_MTP_BATCH_VERIFY @ ds4.c:31682
DS4_MTP_CAPTURE_PREFIX1 @ ds4.c:31783
DS4_MTP_CONF_LOG @ ds4.c:31564
DS4_MTP_EXACT_REPLAY @ ds4.c:31784
DS4_MTP_FORCE_SNAPSHOT @ ds4.c:31788
DS4_MTP_FULL_LOGITS @ ds4.c:31479
DS4_MTP_MIN_MARGIN @ ds4.c:31557
DS4_MTP_PROBE @ ds4.c:31443
DS4_MTP_SPEC_LOG @ ds4.c:31580
DS4_MTP_STRICT @ ds4.c:31555
DS4_MTP_TIMING @ ds4.c:31563
DS4_NO_BATCHED_ATTN @ ds4.c:10298
DS4_NO_BATCHED_ROPE @ ds4.c:9806
DS4_NO_PARALLEL_ATTN_ROWS @ ds4.c:9792
DS4_NO_ROUTED_TOKEN_PARALLEL @ ds4.c:8511
DS4_NO_SHARED_BATCH_FFN @ ds4.c:10301
DS4_ORACLE_LOGITS @ ds4.c:25772
DS4_PARALLEL_ATTN_ROWS @ ds4.c:9789
DS4_PARALLEL_FFN @ ds4.c:10300
DS4_PREFILL_BATCH @ ds4.c:10302
DS4_PREFILL_PROFILE_DETAIL @ ds4.c:8494
DS4_PREFILL_PROFILE_TOKEN @ ds4.c:10018
DS4_ROUTED_TOKEN_PARALLEL @ ds4.c:8510
DS4_SELECTED_UPLOAD_EVENT @ ds4_cuda.cu:3615
DS4_SPEX_CAP @ ds4.c:17002
DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER @ ds4.c:17269
DS4_SPEX_FILE @ ds4.c:16317
DS4_SPEX_HIDDEN_CAP @ ds4.c:17005
DS4_SPEX_HIDDEN_FILE @ ds4.c:16321
DS4_SPEX_HIDDEN_GPU_LOAD @ ds4.c:16866
DS4_SPEX_HIDDEN_GPU_PREFETCH @ ds4.c:16887
DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN @ ds4.c:16894
DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS @ ds4.c:16928
DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS_EVERY @ ds4.c:16988
DS4_SPEX_HIDDEN_GPU_SCORE @ ds4.c:16877
DS4_SPEX_HIDDEN_PREFETCH @ ds4.c:16861
DS4_SPEX_MARKOV_FILE @ ds4.c:16323
DS4_SPEX_PATH @ ds4.c:16319
DS4_SPEX_PREFETCH_NEXT_LAYER @ ds4.c:17267
DS4_SPEX_PREFETCH_PROFILE @ ds4.c:16901
DS4_SPEX_STATS @ ds4_cuda.cu:897
DS4_SPEX_TAU @ ds4.c:17219
DS4_SPEX_TRACE_HIDDEN @ ds4.c:17358
DS4_SPEX_TRACE_ROUTING @ ds4.c:17298
DS4_SPEX_TRACE_ROUTING_WEIGHTS @ ds4.c:17290
DS4_THREADS @ ds4.c:1390
DS4_TOKEN_TIMING @ ds4.c:27103
DS4_TRACE_TOP @ ds4.c:27072
```

### 11.4 Nomi DS4_METAL nel core graph: backend misto

Queste leve sono inventariate da `ds4.c`, non da `ds4_metal.m`. Il prefisso
storico `DS4_METAL_` **non basta** a classificarle N/A CUDA: il graph comune e'
usato anche dal backend CUDA e alcuni consumer non hanno un guard Metal.

Sono confermate applicabili al 3060 CUDA le famiglie hotlist/preload
`DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST`,
`DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP`,
`DS4_METAL_STREAMING_EXPERT_HOTLIST`, `..._PROFILE` e
`DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED`, `..._K`, `..._PROFILE`.
Restano INFERITE/non isolate e possono aggiungere startup, readback o churn.
Le voci che selezionano shader, table/view Q4, fusioni o primitive realmente
Metal restano N/A CUDA; per le altre il backend va deciso dal consumer, non dal
nome. Nessuna famiglia va promossa senza attribuzione del ramo percorso.

```text
DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD @ ds4.c:15671
DS4_METAL_DECODE_STAGE_PROFILE @ ds4.c:11742
DS4_METAL_DECODE_STAGE_PROFILE_LAYER @ ds4.c:21482
DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION @ ds4.c:15761
DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ @ ds4.c:15740
DS4_METAL_DISABLE_HC_FUSION @ ds4.c:15725
DS4_METAL_DISABLE_HC_NORM_FUSION @ ds4.c:15745
DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS @ ds4.c:16168
DS4_METAL_DISABLE_IQ2_SELECTED_SHARED_OVERLAP @ ds4.c:16106
DS4_METAL_DISABLE_KV_FUSION @ ds4.c:15730
DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO @ ds4.c:16131
DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_PRELOAD @ ds4.c:29793
DS4_METAL_DISABLE_Q4_EXPERT_TABLE @ ds4.c:16132
DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS @ ds4.c:16219
DS4_METAL_DISABLE_QKV_NORM_FUSION @ ds4.c:15735
DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION @ ds4.c:12000
DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION @ ds4.c:15756
DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION @ ds4.c:19804
DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL @ ds4.c:23808
DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL @ ds4.c:23763
DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE @ ds4.c:11998
DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST @ ds4.c:18024
DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE @ ds4.c:11741
DS4_METAL_DISABLE_STREAMING_IQ2_CPU_ROUTER @ ds4.c:16070
DS4_METAL_DISABLE_STREAMING_LAYER_BATCH @ ds4.c:11739
DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED @ ds4.c:11723
DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR @ ds4.c:11997
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE @ ds4.c:11949
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN @ ds4.c:11924
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP @ ds4.c:12687
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD @ ds4.c:11940
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE @ ds4.c:11933
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP @ ds4.c:12686
DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD @ ds4.c:11932
DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE @ ds4.c:11916
DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN @ ds4.c:11908
DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PROFILE @ ds4.c:12312
DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD @ ds4.c:15360
DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED @ ds4.c:15368
DS4_METAL_DISABLE_STREAMING_READAHEAD @ ds4.c:11718
DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_EARLY_COMMIT @ ds4.c:16124
DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD @ ds4.c:16113
DS4_METAL_DISABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY @ ds4.c:18341
DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP @ ds4.c:16105
DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP @ ds4.c:11727
DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE @ ds4.c:11731
DS4_METAL_DUMP_PREFILL_LOGITS @ ds4.c:27250
DS4_METAL_ENABLE_BATCH_HC_NORM_FUSION @ ds4.c:15751
DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO @ ds4.c:16095
DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO @ ds4.c:16094
DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS @ ds4.c:16091
DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE @ ds4.c:16093
DS4_METAL_ENABLE_Q4_EXPERT_TABLE @ ds4.c:16092
DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS @ ds4.c:16090
DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE @ ds4.c:11740
DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER @ ds4.c:16069
DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED @ ds4.c:11722
DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR @ ds4.c:12013
DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED @ ds4.c:17998
DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN @ ds4.c:11923
DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD @ ds4.c:11931
DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE @ ds4.c:11915
DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN @ ds4.c:11907
DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD @ ds4.c:15358
DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED @ ds4.c:15359
DS4_METAL_ENABLE_STREAMING_READAHEAD @ ds4.c:11717
DS4_METAL_ENABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY @ ds4.c:18340
DS4_METAL_GPU_BATCH_EMBED_MIN @ ds4.c:21367
DS4_METAL_GRAPH_DUMP_LAYER @ ds4.c:11210
DS4_METAL_GRAPH_DUMP_LOGITS @ ds4.c:25796
DS4_METAL_GRAPH_DUMP_NAME @ ds4.c:11207
DS4_METAL_GRAPH_DUMP_POS @ ds4.c:11214
DS4_METAL_GRAPH_DUMP_PREFIX @ ds4.c:11204
DS4_METAL_GRAPH_OUTPUT_ROW @ ds4.c:24635
DS4_METAL_GRAPH_PREFILL_PROFILE @ ds4.c:23837
DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE @ ds4.c:24586
DS4_METAL_GRAPH_PROMPT_TOKENS @ ds4.c:25741
DS4_METAL_GRAPH_RAW_CAP @ ds4.c:25613
DS4_METAL_GRAPH_TEACHER_FORCE @ ds4.c:21024
DS4_METAL_GRAPH_TOKEN_PROFILE @ ds4.c:23529
DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS @ ds4.c:21150
DS4_METAL_GRAPH_TRACE_CACHE @ ds4.c:25808
DS4_METAL_GRAPH_TRACE_COMP @ ds4.c:25809
DS4_METAL_GRAPH_TRACE_LAYERS @ ds4.c:21021
DS4_METAL_GRAPH_TRACE_STAGE_LAYER @ ds4.c:21025
DS4_METAL_HC_NORM_FUSION_CHECK @ ds4.c:15805
DS4_METAL_HC_NORM_FUSION_CHECK_TOL @ ds4.c:15814
DS4_METAL_INDEXER_STAGE_PROFILE @ ds4.c:19285
DS4_METAL_LAYER_STAGE_PROFILE @ ds4.c:21476
DS4_METAL_LAYER_STAGE_PROFILE_LAYER @ ds4.c:21477
DS4_METAL_MEMORY_REPORT @ ds4.c:25763
DS4_METAL_MOE_WRITE_CLAMPED_ACT @ ds4.c:11999
DS4_METAL_NO_PREFILL_KERNEL_WARMUP @ ds4.c:21404
DS4_METAL_PREFILL_CHUNK @ ds4.c:8705
DS4_METAL_PRO_Q4_CPU_ROUTER @ ds4.c:16065
DS4_METAL_PRO_Q4_CPU_ROUTER_PROFILE @ ds4.c:18265
DS4_METAL_Q_STAGE_PROFILE @ ds4.c:21551
DS4_METAL_Q4_PRO_MAP_GROUPS @ ds4.c:4145
DS4_METAL_Q4_SELECTED_OVERLAP_SHARED @ ds4.c:16075
DS4_METAL_RESUME_PREFILL_MIN @ ds4.c:25642
DS4_METAL_STREAMING_DECODE_PREFILL_MAX @ ds4.c:23765
DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP @ ds4.c:18189
DS4_METAL_STREAMING_EXPERT_HOTLIST @ ds4.c:24052
DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE @ ds4.c:24032
DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE @ ds4.c:18266
DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX @ ds4.c:11953
DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN @ ds4.c:11973
DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K @ ds4.c:18003
DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE @ ds4.c:23951
DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE @ ds4.c:15044
DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP @ ds4.c:12685
DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE @ ds4.c:15042
DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS @ ds4.c:12651
DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE @ ds4.c:15043
DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_AHEAD @ ds4.c:12693
DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP @ ds4.c:12684
DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS @ ds4.c:12649
DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE @ ds4.c:15045
DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE @ ds4.c:14793
DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_THREADS @ ds4.c:12665
DS4_METAL_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE @ ds4.c:14792
DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP @ ds4.c:12675
DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS @ ds4.c:12663
DS4_METAL_STREAMING_PREFILL_SELECTED_PROFILE @ ds4.c:12311
DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_GAP @ ds4.c:15373
DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE @ ds4.c:15457
DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE @ ds4.c:18358
```

## 12. Inventario patch-only e compatibilita'

Questi 48 nomi sono introdotti dalle patch del repo ma assenti dai 419 nomi
dello snapshot WSL letto. La riga patch e' la prima introduzione trovata; le
varianti `canonical` possono ripeterla. Default, effetto e stato sono descritti
sopra per i nomi principali. Per l'intera famiglia rewind: tutti numerici,
default 0.5/0.5/4/1/8/128/32/128/32/16/2/256/32/0/0/0 come indicato sotto,
con controller principale off.

| Nome | Tipo/default | Patch | Effetto / rischio / uso |
|---|---|---|---|
| `DS4_ASYNC_PIPELINE` | bool 0 | `0032-async-pipeline-rebased.patch:75` | Pipeline sperimentale; due varianti 0032. NON RISULTA. |
| `DS4_CUDA_NO_WHOLE_MMAP_REGISTER` | bool 0 | `0047-no-whole-mmap-register.patch:121` | Vieta whole-map pin nel ramo 0047. PATCH-ONLY e INERTE se soltanto esportato nel V2 corrente. |
| `DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC` | bool 0 | `0048-prefill-overlap-s1.patch:147` | S1 prefill; storico MISURATO vincente/bit-exact, ma PATCH-ONLY e INERTE nel V2 corrente. |
| `DS4_CUDA_SELECTED_STAGE_DEPTH` | int 4, 1..16 | `0047-no-whole-mmap-register.patch:29` | Stage ring; ipotesi 4 vs 8 soltanto con 0047 applicata integralmente. |
| `DS4_DIAG_CONF_LOG` | path | `0030-diag-token-confidence-expert-norm.patch:38` | Log confidence/norm; diagnostica. |
| `DS4_MTP_STREAMING`, `DS4_MTP_STREAMING_UNSAFE` | bool 0/0 | `0009...:45`, `0008...:10` | Unlock MTP sicuro/unsafe; non usare unsafe. |
| `DS4_PACE_ADAPTIVE_CF_ADMIT`, `DS4_PACE_ADAPTIVE_CF_ADMIT_WINDOW` | bool 0; 3 (1..8) | `0046...:130,132` | Admission counterfactuale; MISURATO ma tende a K0. |
| `DS4_PACE_ADMIT`, `DS4_PACE_ADMIT_H`, `DS4_PACE_ADMIT_KDRIFT`, `DS4_PACE_ADMIT_PERSIST`, `DS4_PACE_ADMIT_COOLDOWN`, `DS4_PACE_ADMIT_MAX_PER_100` | 0; 1.2; .02; 2; 16; 0 | `0026...:101-110` | Admission per domanda; rischio widening/churn. NON RISULTA. |
| `DS4_PACE_ALPHA_S1` | 0.10 | `0020...:90` | EMA S1; patch-only. |
| `DS4_PACE_S1_TRIGGER`, `DS4_PACE_S1_SLOPE_WIN`, `DS4_PACE_S1_SLOPE_THR`, `DS4_PACE_S1_STABLE`, `DS4_PACE_S1_ACTION` | 0; 64; .0003; 16; `rotate` | `0020...:98-107` | Trigger slope, azione `rotate`/`widen`; NON RISULTA. |
| `DS4_PACE_WRAP_ROTATE_DELTA` | bool 0 | `0021...:235` | Delta-prefetch al rotate; NON RISULTA. |
| `DS4_REAP_MASK_SOFT_BIAS` | float, hard/0 | `0046...:65` | Soft mask; MISURATO, braccio B affamato. |
| `DS4_REAP_PREFILL_READAHEAD` | bool 0 | `0048...:19` | S2 storicamente MISURATO/refutato su WSL2; PATCH-ONLY e INERTE nel V2 corrente. |
| `DS4_REAP_WRAP_LOCK`, `DS4_REAP_WRAP_THREADS` | bool 0; 8 max 16 | `0013...:168,131` | Alias obsoleti; usare i nomi PREFETCH. |
| `DS4_REWIND_TEST`, `DS4_REWIND_TEST_LOG` | bool/path | `0027...:115,125` | Harness exactness rewind. |
| `DS4_SPEX_TRACE_ROUTING_RESIDENCY`, `DS4_SPEX_TRACE_TOKENS` | path/bool | `0017...:37`, `0028...:20` | Arricchiscono trace; diagnostica. |

Famiglia rewind completa, patch `0022-pace-s1-rewind.patch`:

```text
DS4_PACE_REWIND (off) @ :132
DS4_PACE_REWIND_CKPT_DEPTH (1) @ :138
DS4_PACE_REWIND_ALPHA (0.5) @ :261
DS4_PACE_REWIND_ARM_K (0.5) @ :262
DS4_PACE_REWIND_ARM_H (4) @ :263
DS4_PACE_REWIND_FIRE_K (1) @ :264
DS4_PACE_REWIND_FIRE_H (8) @ :265
DS4_PACE_REWIND_CALWIN (128) @ :266
DS4_PACE_REWIND_BASE_LAG (32) @ :268
DS4_PACE_REWIND_BASE_WIN (128) @ :269
DS4_PACE_REWIND_EVERY (32) @ :273
DS4_PACE_REWIND_MARGIN (16) @ :275
DS4_PACE_REWIND_MAX (2) @ :276
DS4_PACE_REWIND_BACKOFF (256) @ :277
DS4_PACE_REWIND_GUARD (32) @ :278
DS4_PACE_REWIND_KEEP (0) @ :279
DS4_PACE_REWIND_GARBAGE (0) @ :288
DS4_PACE_REWIND_WARMUP (0) @ :290
```

## 13. Copertura patch e note di versione

La catena top-level censita va da 0001 a 0049, con buchi nominali 0019, 0023,
0025 e 0029 e con piu' patch che condividono numero perche' sono linee di
lavoro parallele. Le directory/file letti includono:

- SPEX 0001-0007, 0015-0017, 0028, 0044-0045;
- MTP/model-map 0008-0010;
- REAP/PACE 0011-0022, 0026-0027, 0031, 0033, 0035, 0037-0046, 0049;
- CUDA cache/pipeline 0024, 0032 (entrambe le varianti), 0034, 0036/0036a,
  0041-0042, 0047-0048;
- tutte le copie `canonical`, `ds4_spex_predict.c/.h` e
  `upstream-pr497-single-token-selected-load.diff`.

0049 non aggiunge env: estende il log con i breakthrough della soft-mask.
`upstream-pr497-single-token-selected-load.diff` non aggiunge env DS4. Le
copie canonical e le patch con numero duplicato non sono una sequenza da
applicare ciecamente: vanno selezionate in base alla baseline.

I due worktree non equivalgono alla semplice applicazione lineare di tutti i
file. Entrambi mostrano marker sorgente 0011-0014, 0035, 0037-0040 e 0043-0045;
le patch 0041-0045 sono incorporate come modifiche locali sopra lo stesso HEAD.
0046-0049 (incluse 0047/0048) restano file disponibili ma non consumer attivi
negli snapshot correnti. Il solo V2 aggiunge il WIP 0050 in `ds4.c`,
`ds4_cuda.cu` e `ds4_gpu.h`; non esiste ancora il corrispondente file patch.
La 0051 non e' un ulteriore livello sorgente: e' un design non implementato.
I nomi `DS4_CUDA_DYNAMIC_ARENA`, `DS4_CUDA_DYNAMIC_ARENA_GB` e
`DS4_PACE_LIVEMASK_MODE` non hanno consumer e non vanno esportati nei runner.

| Livello | Contenuto verificato | Stato/test |
|---|---|---|
| HEAD comune `da0b3f63...` | storia committed: canonical 0001-0008, 0011-0018, 0020, 0021, 0024, 0026, 0031, 0033, 0034, 0036/0036a; feature 0035/0037/0038; poi 0039 e 0040 (`livemask: pin-by-mass PRODUCER + mass-ranked selection`) | Git history verificata; non implica che ogni patch disponibile sia applicata. |
| Diff locale F | `ds4.c`, `ds4_cuda.cu`; lavoro equivalente 0041-0045 sopra HEAD | Sorgente presente; log/run storici come nelle sezioni precedenti. |
| Diff locale V | stesso lavoro piu' 0050; modificati `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h` | `build_0050i` server `sm_86` riuscita; DMA OSSERVATO; test funzionali finali pendenti. |
| Solo file patch | 0046-0049, 0047/0048, varianti 0032 e altri rami non selezionati | PATCH-ONLY; un env esportato senza consumer e' INERTE. |

Questa e' la distinzione operativa tra **committed**, **applicata localmente**,
**sorgente WIP** e **patch-only**. Nessuna patch e' stata applicata durante il
censimento.

## 14. Configurazione bake60 osservata

`runs/ds4/20260712_virtual_bake/arm_self60b_run1/server_env.txt` contiene:

```text
DS4_CUDA_KEEP_MODEL_PAGES=1
DS4_CUDA_NO_DIRECT_IO=1
DS4_CUDA_NO_Q8_F16_CACHE=1
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
DS4_PACE=0
DS4_REAP_MASK_FILE=masks/mask60_self.txt
```

Questo e' un singolo run `n=1`: documenta una fixture bake60 pre-0050 che rende
L2, non prova il path DMA 0050 e non valida qualita' o policy di produzione 0051.

CLI del runner: `--cuda --ssd-streaming --ssd-streaming-cache-experts 400
--prefill-chunk 512 --ctx 4096`, con KV disk gestito dal server nei setup
agentici. Startup osservato: circa 0.99 GiB di preparazione, 122.69 MiB di
context buffer e cache modello salita fino a 7.86 GiB nei log. Questi numeri
spiegano perche' reserve, arena e cache-limit non possono essere trattati come
leve indipendenti.

Il runner corrente esporta anche `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256`, ma la
cronologia e gli artefatti mostrano che fu aggiunto dopo l'avvio di bake60.
Questa distinzione deve restare nei report futuri: **runner corrente non
equivale a env effettivo di un run storico**. Il valore 256 resta configurato
ma non provato da un A/B isolato; i log pod12 riportano inoltre otto OOM
dell'arena anche con 256 MiB, quindi non e' un baseline raccomandato.

## 15. Leve source-wide fuori dal core CUDA

Questa sezione chiude il divario della vecchia scansione limitata a
`ds4.c`/`ds4_cuda.cu`. Le leve del runtime `ds4_metal.m` sotto hanno
default/effetto **INFERITI dal codice**, stato **NON TESTATO** e rilevanza
**N/A CUDA**. Questa classificazione non si estende ai nomi legacy
`DS4_METAL_*` del graph comune in `ds4.c`, trattati nella sezione 11.4 e in parte
applicabili al 3060 CUDA.

### 15.1 Runtime distribuito

| Nome/gruppo | Default | Source V | Effetto e interazioni | Test / rilevanza |
|---|---|---|---|---|
| `DS4_DIST_PREFILL_SEND_DEPTH` | 2, clamp 1..8 e numero chunk | `ds4_distributed.c:470` | Profondita' invio coordinator; piu' buffer/overlap rete. | INFERITO, non testato; N/A singolo 3060. |
| `DS4_DIST_SOCKET_BUFFER_MB` | 128, valido 0..512; 0 salta il set | `:712` | SO_SNDBUF/SO_RCVBUF. | Non testato; puo' contendere RAM, non pin CUDA. |
| `DS4_DIST_WORKER_PREFETCH_DEPTH` | 2, 1..8 | `:726` | Prefetch worker. | Non testato; I/O/rete, non zero-copy locale. |
| `DS4_DIST_WORKER_FORWARD_WINDOW` | 4, 1..64 | `:740` | Finestra forwarding attivazioni. | Non testato. |
| `DS4_DIST_DECODE_PROFILE` | presence/off | `:753` | Timing decode distribuito. | Diagnostica/log, non testato. |
| `DS4_DIST_SOCKET_TIMEOUT_SEC` | 60, 1..3600 | `:1032` | Timeout send. | Rete only. |
| `DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` | unset/nessun override, 1..3600 se set | `:1049` | Timeout recv. | Rete only. |
| `DS4_DIST_CONNECT_TRACE` | presence/off | `:1112` | Trace connessioni. | I/O log, non testato. |
| `DS4_DIST_CONNECT_BIND_HOST`, `DS4_DIST_CONNECT_BIND_IF` | unset | `:1332-1334` | Forzano indirizzo locale/interfaccia per connect; l'interfaccia dipende dal supporto OS. | Non testato; N/A bake. |
| `DS4_DIST_DISABLE_PREFILL_PIPELINE` | presence/off | `:3427` | Serializza prefill distribuito. | Riduce overlap; non testato. |
| `DS4_DIST_PREFILL_CHUNK` | session cap/CLI, positivo | `:3455` | Override chunk distribuito. | Interagisce con `--dist-prefill-chunk`; non testato. |
| `DS4_DIST_PREFILL_WINDOW` | workers+2, cap 8; env max 64 | `:3485` | Finestra chunk in flight. | Buffer/latency rete; non testato. |
| `DS4_DIST_DISABLE_PREFILL_ACK_ONLY` | presence/off | `:3692` | Fa restituire hidden a ogni chunk invece di soli ACK intermedi. | Aumenta traffico/memoria; non testato. |
| `DS4_DIST_DISABLE_WORKER_PREFETCH` | presence/off | `:7855` | Usa worker loop seriale. | Diagnostica overlap; non testato. |

### 15.2 Runtime Metal

| Nome/gruppo | Default | Source V | Effetto/interazioni | Stato |
|---|---|---|---|---|
| `DS4_METAL_*_SOURCE` per `FLASH_ATTN`, `DENSE`, `MOE`, `DSV4_HC`, `UNARY`, `DSV4_KV`, `DSV4_ROPE`, `DSV4_MISC`, `ARGSORT`, `CPY`, `CONCAT`, `GET_ROWS`, `SUM_ROWS`, `SOFTMAX`, `REPEAT`, `GLU`, `NORM`, `BIN`, `SET_ROWS` | path unset | `ds4_metal.m:3048-3066` | Sostituiscono il source shader incorporato. File stale/incompatibile cambia compilazione e puo' fallire o alterare matematica. | NON TESTATO; N/A CUDA. |
| `DS4_METAL_UNRETAINED_COMMAND_BUFFERS` | off | `:751` | Usa command buffer unretained; lifetime piu' severo. | NON TESTATO; N/A. |
| `DS4_METAL_EXACT_VIEW_CACHE_GIB`, `...MIB`, `...PROFILE` | 64 GiB; MIB se set ha precedenza; profile off | `:769-813` | Limita cache exact-view; 0 significa nessun limite. Profile aggiunge log. | NON TESTATO; N/A. |
| `DS4_METAL_MODEL_UNTRACKED` | off | `:958` | Risorse modello non tracked. | Rischio ordering/residency; N/A. |
| `DS4_METAL_NO_RESIDENCY`, `DS4_METAL_USE_QUEUE_RESIDENCY_SET` | off/off | `:1411,1442` | Opt-out residency o opt-in queue residency set. | Mutano gestione memoria Metal; N/A. |
| `DS4_METAL_MODEL_VIEW_MAX_GIB` | device `maxBufferLength`; 128 GiB per split distributed | `:1526` | Cap vista modello. | N/A CUDA. |
| `DS4_METAL_NO_MODEL_WARMUP` | off | `:1619` | Salta warmup modello normalmente attivo nel path non-streaming/resident. | Startup/page state Metal; N/A. |
| `DS4_METAL_DISABLE_HOT_PIPELINE_STATICS` | off | `:1789` | Opt-out statics pipeline hot. | Possibile compile/runtime overhead; N/A. |
| `DS4_METAL_COMPRESSOR_PAIR_NR4` | off | `:1806` | Opt-in compressor pair NR4. | Matematica/layout; N/A. |
| `DS4_METAL_DISABLE_METAL4` | off | `:1953` | Disabilita Metal4, auto solo su M5/M6/A19/A20 supportati. | N/A 3060. |
| `DS4_METAL_MODEL_WARMUP_STRIDE_MB`, `...KB` | 1 MiB; MB 1..1024; KB 1..1048576 e ha precedenza/floor pagina | `:2007-2021` | Passo touch warmup. | I/O/page fault Metal; N/A. |
| `DS4_METAL_HC_STABLE`, `DS4_METAL_NORM_RSQRT_DISABLE` | **on se unset**; valore false li disabilita | `:4617-4618` | Stabilita' HC e opt-out rsqrt nel kernel. | Eccezioni al default bool-off; N/A. |
| `DS4_METAL_KV_RAW_F32`, `DS4_METAL_ROPE_EXP2_LOG2`, `DS4_METAL_MATH_SAFE` | off | `:4619-4621` | Varianti precisione/matematica. | Qualita'/speed, N/A CUDA. |
| `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS` | 9, clamp 1..18 | `:7756` | Thread pool pread Metal. | Potenziale contesa I/O; non trasferire il default al CUDA path. |
| `DS4_METAL_STREAMING_EXPERT_PREAD_POOL` | on salvo valore esatto `0` | `:7851` | Pool persistente per pread. | Interagisce con thread count; N/A. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD` | readahead on in SSD streaming | `:7658` | Opt-out readahead. | Page-cache/I/O; N/A CUDA. |
| `DS4_METAL_STREAMING_EXPERT_SLAB_MB`, `...DISABLE...SLABS` | slab 4096 MiB; slabs on in streaming | `:8266,8277` | Dimensione/opt-out slab host. | RAM/fragmentation Metal; N/A. |
| `DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED`, `...DISABLE...` | enable off; disable vince | `:8594-8595` | Abilita/forza off DONTNEED dopo expert I/O. | Page cache; semanticamente analogo al bake ma N/A CUDA. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD` | early load on | `:10623` | Opt-out caricamento anticipato. | Overlap/I/O; N/A. |
| `DS4_METAL_ENABLE/DISABLE_STREAMING_COMPACT_ADDR`, `...HIT_VALIDATOR`, `...MASKED_ADDR`; `ENABLE_STREAMING_EXPERT_ADDR_TABLE` | enable opt-in; disable vince | `:8783-8819` | Varianti addressing/validator per expert streaming. | Layout/correttezza; N/A. |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY`, `...TIMING_SUMMARY`, `...PROFILE_SUMMARY` | summary off salvo enable; disable vince | `:7576-7578` | Riepiloghi timing/profile. | Log overhead; N/A. |
| `DS4_METAL_Q4_EXPERT_TABLE_GROUP_SIZE` | 1, quindi grouping off; 2..n_total per abilitarlo | `:11561` | Raggruppa expert table Q4. | Layout/cache; N/A. |
| `DS4_METAL_Q4_EXPERT_GROUP_SIZE` | 32, clamp al totale | `:20427` | Group size kernel Q4 expert. | Geometria kernel; N/A. |
| Famiglie `ENABLE/DISABLE_Q4_*`, `Q4_*VIEWS`, `Q4_*RESIDENCY_SET`, `Q4_*USE_RESOURCES`, `MOE_MM_ID_*`, `DISABLE_ROUTER_SELECT_FUSION` | tutte unset/off; nei pair disable prevale | `:11572-12199,20427-24944` | Selezione table/address/view/resource/fusion dei kernel Q4/MoE. Cambiano layout, residency, sync e prestazioni. | NON TESTATO; N/A CUDA. |
| Profili `ATTN_OUT_STAGE`, `FLASH_ATTN_STAGE`, `MOE_ONE_STAGE`, `MOE_STAGE`, `Q8_PREFILL`, `SELECTED`/`Q4_SELECTED` e relativi `FILTER`/`LAYER` | off; filtri/layer inerti senza profilo parent | `:6631,12922-12937,16153,17183,22811-22817,23413-25147` | Forzano timing/sync/log per stadio o layer. | Diagnostica, N/A CUDA. |
| Profili/trace streaming (`BUFFER_MLOCK`, `EARLY_LOAD`, `EVICT_DONTNEED`, `LAYER_STATS*`, `PREAD`, `SPLIT`, `MAP_TRACE`, `BATCH_SELECTED_ADDR`, `TRACE_ALLOCS`) | off/path unset | `:2392-11179,23955` | Telemetria allocazioni, map, read, split e layer. | I/O/sync; N/A CUDA. |
| `DS4_MOE_RECORD_SELECTED_HOTLIST`, `...MERGE`, `...FRESH`, `DS4_MOE_RECORD_SELECTED_IDS` | path unset; merge/fresh off; fresh prevale sul merge | `:1033-1230` | Registra hotlist o ID selezionati; merge legge il file esistente. | Offline/profile, N/A CUDA. |
| `DS4_METAL_HAS_TENSOR` | macro generata, non env | `:4613` | Marker shader interno. | Non e' una leva utente. |
| `DS4_METAL_MOE_TILE_MAX` | nessun consumer runtime | `tests/ds4_test.c:1207` | Il test salva/unset/ripristina un nome legacy. | **INERTE** in F/V; N/A CUDA. |

Tutti gli altri nomi Metal elencati nell'inventario 15.4 sono toggle o profili
delle famiglie appena descritte: default unset/off, salvo le eccezioni esplicite
qui sopra; `DISABLE` prevale sul corrispondente `ENABLE`. Non esiste evidenza di
test corrente su questo host CUDA.

### 15.3 Frontend e harness

| Nome/gruppo | Default | Source V | Effetto/interazioni | Test / rilevanza |
|---|---|---|---|---|
| `DS4_CHROME` | path unset, auto-discovery | `ds4_web.c:964` | Override eseguibile Chrome per UI web. | Non testato; nessun effetto bake/zero-copy. |
| `DS4_MTP_SPEC_DISABLE` | assente: spec argmax attivo con temp<=0 e draft>1 | `ds4_cli.c:484,1155` | Presence disabilita la corsia speculativa CLI. | Non testato; MTP only. |
| `DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY` | assente: recovery attivo | `ds4_server.c:10385` | Presence spegne recovery tool avviato dentro thinking. | Non testato; semantica server, non memoria. |
| `DS4_CUDA_TOPK_REGRESSION_SEC` | 2.0 secondi | `tests/cuda_long_context_smoke.c:72` | Soglia di fallimento top-k regression. | Harness CUDA; test non eseguito per mandato. |
| `DS4_TEST_MODEL`, `DS4_TEST_MTP` | `ds4flash.gguf`; MTP path unset, draft 4 se fornito al fast engine | `tests/ds4_test.c:12,92` | Input modello/head del test. | Test-only, non eseguito. |
| `DS4_TEST_SSD_STREAMING`, `...COLD`, `...CACHE_EXPERTS`, `...CACHE_GB`, `...PRELOAD_EXPERTS` | off/off/0/0/0 | `:73,101-108` | Configurano engine streaming del test. | Test-only; possono allocare GPU/modello, non eseguiti. |
| `DS4_TEST_LONG_PROMPT` | `tests/long_context_story_prompt.txt` | `:683` | Fixture long-context. | Test-only. |
| `DS4_TEST_VECTOR_FILE`, `DS4_TEST_LOCAL_GOLDEN_FILE` | official.vec; local-golden.vec | `:922,1199` | Fixture logprob/tensor equivalence. | Test-only. |
| `DS4_TEST_LOGPROB_AUTO_METAL` | assente: test forza `DISABLE_METAL4=1` | `:933` | Presence lascia auto Metal4. | Test Metal, N/A CUDA. |
| `DS4_TEST_MPP_EQ_CASE` | unset: tutti; lista comma/substr | `:1501` | Filtra casi tensor equivalence. | Test-only. |
| `DS4_TEST_RECOVERY_PROBE` | presence/off | `:1840` | Stampa il tool turn naturale invece della recovery. | Diagnostica test-only. |

### 15.4 Inventario esatto delle aggiunte source-wide

Questa lista riporta la prima occorrenza V dei nomi che mancavano
dall'inventario core storico. Le semantiche e lo stato sono nelle tabelle 15.1-
15.3; le tre leve 0050 sono nella sezione 3.

```text
DS4_CHROME @ ds4_web.c:964
DS4_CUDA_STREAM_FROM_RAM_MASKED @ ds4.c:12984
DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB @ ds4.c:12972
DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG @ ds4_cuda.cu:1158
DS4_CUDA_TOPK_REGRESSION_SEC @ tests/cuda_long_context_smoke.c:72
DS4_DIST_CONNECT_BIND_HOST @ ds4_distributed.c:1332
DS4_DIST_CONNECT_BIND_IF @ ds4_distributed.c:1334
DS4_DIST_CONNECT_TRACE @ ds4_distributed.c:1112
DS4_DIST_DECODE_PROFILE @ ds4_distributed.c:753
DS4_DIST_DISABLE_PREFILL_ACK_ONLY @ ds4_distributed.c:3692
DS4_DIST_DISABLE_PREFILL_PIPELINE @ ds4_distributed.c:3427
DS4_DIST_DISABLE_WORKER_PREFETCH @ ds4_distributed.c:7855
DS4_DIST_PREFILL_CHUNK @ ds4_distributed.c:3455
DS4_DIST_PREFILL_SEND_DEPTH @ ds4_distributed.c:470
DS4_DIST_PREFILL_WINDOW @ ds4_distributed.c:3485
DS4_DIST_SOCKET_BUFFER_MB @ ds4_distributed.c:712
DS4_DIST_SOCKET_RECV_TIMEOUT_SEC @ ds4_distributed.c:1049
DS4_DIST_SOCKET_TIMEOUT_SEC @ ds4_distributed.c:1032
DS4_DIST_WORKER_FORWARD_WINDOW @ ds4_distributed.c:740
DS4_DIST_WORKER_PREFETCH_DEPTH @ ds4_distributed.c:726
DS4_METAL_ARGSORT_SOURCE @ ds4_metal.m:3056
DS4_METAL_ATTN_OUT_STAGE_PROFILE @ ds4_metal.m:16153
DS4_METAL_BIN_SOURCE @ ds4_metal.m:3065
DS4_METAL_COMPRESSOR_PAIR_NR4 @ ds4_metal.m:1806
DS4_METAL_CONCAT_SOURCE @ ds4_metal.m:3058
DS4_METAL_CPY_SOURCE @ ds4_metal.m:3057
DS4_METAL_DENSE_SOURCE @ ds4_metal.m:3049
DS4_METAL_DISABLE_ATTN_OUT_IDS_CACHE @ ds4_metal.m:16113
DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT @ ds4_metal.m:16102
DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE @ ds4_metal.m:15930
DS4_METAL_DISABLE_HOT_PIPELINE_STATICS @ ds4_metal.m:1789
DS4_METAL_DISABLE_METAL4 @ ds4_metal.m:1953
DS4_METAL_DISABLE_MOE_MM_ID_PAIR_SWIGLU @ ds4_metal.m:24944
DS4_METAL_DISABLE_MOE_MM_ID_USE_RESOURCES @ ds4_metal.m:21371
DS4_METAL_DISABLE_PRO_Q4_EXPERT_ADDRESS_AUTO @ ds4_metal.m:11665
DS4_METAL_DISABLE_Q4_BATCH_EXPERT_TABLE @ ds4_metal.m:24889
DS4_METAL_DISABLE_Q4_EXACT_BOUNDARY @ ds4_metal.m:22666
DS4_METAL_DISABLE_Q4_EXACT_TENSOR_ID @ ds4_metal.m:22646
DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE @ ds4_metal.m:11666
DS4_METAL_DISABLE_Q4_GATHER_SLOTS @ ds4_metal.m:22748
DS4_METAL_DISABLE_Q4_GROUP24_EXPERT_TABLE @ ds4_metal.m:22628
DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE @ ds4_metal.m:22594
DS4_METAL_DISABLE_Q4_GROUP8_EXPERT_TABLE @ ds4_metal.m:22611
DS4_METAL_DISABLE_Q4_GROUPED_BOUNDARY @ ds4_metal.m:22576
DS4_METAL_DISABLE_Q4_GROUPED_EXPERTS @ ds4_metal.m:22557
DS4_METAL_DISABLE_Q4_TABLE_BOUNDARY @ ds4_metal.m:22745
DS4_METAL_DISABLE_ROUTER_SELECT_FUSION @ ds4_metal.m:21909
DS4_METAL_DISABLE_STREAMING_COMPACT_ADDR @ ds4_metal.m:8784
DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD @ ds4_metal.m:10623
DS4_METAL_DISABLE_STREAMING_EXPERT_EVICT_DONTNEED @ ds4_metal.m:8595
DS4_METAL_DISABLE_STREAMING_EXPERT_HIT_VALIDATOR @ ds4_metal.m:8819
DS4_METAL_DISABLE_STREAMING_EXPERT_MASKED_ADDR @ ds4_metal.m:8813
DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD @ ds4_metal.m:7658
DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS @ ds4_metal.m:8266
DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY @ ds4_metal.m:7578
DS4_METAL_DSV4_HC_SOURCE @ ds4_metal.m:3051
DS4_METAL_DSV4_KV_SOURCE @ ds4_metal.m:3053
DS4_METAL_DSV4_MISC_SOURCE @ ds4_metal.m:3055
DS4_METAL_DSV4_ROPE_SOURCE @ ds4_metal.m:3054
DS4_METAL_ENABLE_MOE_MM_ID_PAIR_SWIGLU @ ds4_metal.m:24943
DS4_METAL_ENABLE_Q4_BATCH_EXPERT_TABLE @ ds4_metal.m:24875
DS4_METAL_ENABLE_Q4_EXACT_TENSOR_ID @ ds4_metal.m:22645
DS4_METAL_ENABLE_Q4_GATHER_SLOTS @ ds4_metal.m:22747
DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE @ ds4_metal.m:22627
DS4_METAL_ENABLE_Q4_GROUP6_EXPERT_TABLE @ ds4_metal.m:22593
DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE @ ds4_metal.m:22610
DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS @ ds4_metal.m:22556
DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR @ ds4_metal.m:8783
DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE @ ds4_metal.m:8790
DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED @ ds4_metal.m:8594
DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR @ ds4_metal.m:8791
DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR @ ds4_metal.m:8792
DS4_METAL_EXACT_VIEW_CACHE_GIB @ ds4_metal.m:769
DS4_METAL_EXACT_VIEW_CACHE_MIB @ ds4_metal.m:778
DS4_METAL_EXACT_VIEW_CACHE_PROFILE @ ds4_metal.m:813
DS4_METAL_FLASH_ATTN_SOURCE @ ds4_metal.m:3048
DS4_METAL_FLASH_ATTN_STAGE_PROFILE @ ds4_metal.m:17183
DS4_METAL_FLASH_ATTN_STAGE_PROFILE_FILTER @ ds4_metal.m:6631
DS4_METAL_GET_ROWS_SOURCE @ ds4_metal.m:3059
DS4_METAL_GLU_SOURCE @ ds4_metal.m:3063
DS4_METAL_HAS_TENSOR @ ds4_metal.m:4613
DS4_METAL_HC_STABLE @ ds4_metal.m:4617
DS4_METAL_KV_RAW_F32 @ ds4_metal.m:4619
DS4_METAL_MATH_SAFE @ ds4_metal.m:4621
DS4_METAL_MODEL_UNTRACKED @ ds4_metal.m:958
DS4_METAL_MODEL_VIEW_MAX_GIB @ ds4_metal.m:1526
DS4_METAL_MODEL_WARMUP_STRIDE_KB @ ds4_metal.m:2015
DS4_METAL_MODEL_WARMUP_STRIDE_MB @ ds4_metal.m:2007
DS4_METAL_MOE_MM_ID_USE_RESOURCES @ ds4_metal.m:21370
DS4_METAL_MOE_ONE_STAGE_PROFILE @ ds4_metal.m:23413
DS4_METAL_MOE_ONE_STAGE_PROFILE_LAYER @ ds4_metal.m:23577
DS4_METAL_MOE_SOURCE @ ds4_metal.m:3050
DS4_METAL_MOE_STAGE_PROFILE @ ds4_metal.m:25147
DS4_METAL_MOE_STAGE_PROFILE_FILTER @ ds4_metal.m:23589
DS4_METAL_MOE_STAGE_PROFILE_LAYER @ ds4_metal.m:25137
DS4_METAL_MOE_TILE_MAX @ tests/ds4_test.c:1207
DS4_METAL_NO_MODEL_WARMUP @ ds4_metal.m:1619
DS4_METAL_NO_RESIDENCY @ ds4_metal.m:1411
DS4_METAL_NORM_RSQRT_DISABLE @ ds4_metal.m:4618
DS4_METAL_NORM_SOURCE @ ds4_metal.m:3064
DS4_METAL_Q4_ADDR_USE_RESOURCES @ ds4_metal.m:12199
DS4_METAL_Q4_EXPERT_GROUP_SIZE @ ds4_metal.m:20427
DS4_METAL_Q4_EXPERT_TABLE_GROUP_SIZE @ ds4_metal.m:11561
DS4_METAL_Q4_EXPERT_TABLE_PROFILE @ ds4_metal.m:12043
DS4_METAL_Q4_GROUP24_BASE_VIEWS @ ds4_metal.m:22632
DS4_METAL_Q4_GROUP24_EXACT_VIEWS @ ds4_metal.m:22631
DS4_METAL_Q4_GROUPED_CACHE_VIEWS @ ds4_metal.m:22578
DS4_METAL_Q4_SELECTED_EXACT_VIEWS @ ds4_metal.m:22836
DS4_METAL_Q4_SELECTED_PROFILE @ ds4_metal.m:22813
DS4_METAL_Q4_SELECTED_PROFILE_LAYER @ ds4_metal.m:22817
DS4_METAL_Q4_SELECTED_SHARED_EVENT @ ds4_metal.m:22832
DS4_METAL_Q4_SELECTED_TRANSIENT_VIEWS @ ds4_metal.m:22840
DS4_METAL_Q4_SELECTED_USE_BASE_VIEWS @ ds4_metal.m:22835
DS4_METAL_Q4_TABLE_BIND_ANCHORS @ ds4_metal.m:11671
DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET @ ds4_metal.m:11612
DS4_METAL_Q4_TABLE_PER_TENSOR_RESIDENCY_SET @ ds4_metal.m:11715
DS4_METAL_Q4_TABLE_QUEUE_RESIDENCY_SET @ ds4_metal.m:11572
DS4_METAL_Q4_TABLE_RESIDENCY_SET @ ds4_metal.m:11714
DS4_METAL_Q4_TABLE_USE_RESOURCES @ ds4_metal.m:12198
DS4_METAL_Q8_PREFILL_PROFILE @ ds4_metal.m:12922
DS4_METAL_Q8_PREFILL_PROFILE_FILTER @ ds4_metal.m:12937
DS4_METAL_REPEAT_SOURCE @ ds4_metal.m:3062
DS4_METAL_ROPE_EXP2_LOG2 @ ds4_metal.m:4620
DS4_METAL_SELECTED_PROFILE @ ds4_metal.m:22811
DS4_METAL_SELECTED_PROFILE_LAYER @ ds4_metal.m:22815
DS4_METAL_SET_ROWS_SOURCE @ ds4_metal.m:3066
DS4_METAL_SOFTMAX_SOURCE @ ds4_metal.m:3061
DS4_METAL_STREAMING_EXPERT_BUFFER_MLOCK_PROFILE @ ds4_metal.m:8249
DS4_METAL_STREAMING_EXPERT_EARLY_LOAD_PROFILE @ ds4_metal.m:10471
DS4_METAL_STREAMING_EXPERT_EVICT_DONTNEED_PROFILE @ ds4_metal.m:8636
DS4_METAL_STREAMING_EXPERT_LAYER_STATS @ ds4_metal.m:2793
DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA @ ds4_metal.m:2830
DS4_METAL_STREAMING_EXPERT_PREAD_POOL @ ds4_metal.m:7851
DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE @ ds4_metal.m:10417
DS4_METAL_STREAMING_EXPERT_PREAD_THREADS @ ds4_metal.m:7756
DS4_METAL_STREAMING_EXPERT_PROFILE_SUMMARY @ ds4_metal.m:7577
DS4_METAL_STREAMING_EXPERT_SLAB_MB @ ds4_metal.m:8277
DS4_METAL_STREAMING_EXPERT_SPLIT_PROFILE @ ds4_metal.m:23955
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY @ ds4_metal.m:7576
DS4_METAL_STREAMING_MAP_TRACE @ ds4_metal.m:2978
DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE @ ds4_metal.m:11179
DS4_METAL_SUM_ROWS_SOURCE @ ds4_metal.m:3060
DS4_METAL_TRACE_ALLOCS @ ds4_metal.m:2392
DS4_METAL_UNARY_SOURCE @ ds4_metal.m:3052
DS4_METAL_UNRETAINED_COMMAND_BUFFERS @ ds4_metal.m:751
DS4_METAL_USE_QUEUE_RESIDENCY_SET @ ds4_metal.m:1442
DS4_MOE_RECORD_SELECTED_HOTLIST @ ds4_metal.m:1133
DS4_MOE_RECORD_SELECTED_HOTLIST_FRESH @ ds4_metal.m:1034
DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE @ ds4_metal.m:1033
DS4_MOE_RECORD_SELECTED_IDS @ ds4_metal.m:1230
DS4_MTP_SPEC_DISABLE @ ds4_cli.c:484
DS4_SERVER_DISABLE_THINK_TOOL_RECOVERY @ ds4_server.c:10385
DS4_TEST_LOCAL_GOLDEN_FILE @ tests/ds4_test.c:1199
DS4_TEST_LOGPROB_AUTO_METAL @ tests/ds4_test.c:933
DS4_TEST_LONG_PROMPT @ tests/ds4_test.c:683
DS4_TEST_MODEL @ tests/ds4_test.c:12
DS4_TEST_MPP_EQ_CASE @ tests/ds4_test.c:1501
DS4_TEST_MTP @ tests/ds4_test.c:92
DS4_TEST_RECOVERY_PROBE @ tests/ds4_test.c:1840
DS4_TEST_SSD_STREAMING @ tests/ds4_test.c:73
DS4_TEST_SSD_STREAMING_CACHE_EXPERTS @ tests/ds4_test.c:104
DS4_TEST_SSD_STREAMING_CACHE_GB @ tests/ds4_test.c:106
DS4_TEST_SSD_STREAMING_COLD @ tests/ds4_test.c:102
DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS @ tests/ds4_test.c:108
DS4_TEST_VECTOR_FILE @ tests/ds4_test.c:922
```
