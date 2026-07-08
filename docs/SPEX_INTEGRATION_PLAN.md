# SPEX → ds4 (DwarfStar): Progetto d'integrazione (Fase 1)

> Output della Fase-1 (workflow `spex-into-ds4-design`, 2026-07-04). Design grounded: i punti
> d'aggancio sono stati verificati leggendo il sorgente ds4 (commit 80ebbc3, copia in scratchpad
> `ds4-src/`). Obiettivo: integrare SPEX (prefetch predittivo expert MoE) nel motore live ds4,
> per nascondere la latenza di load expert dietro il compute. SPEX predice il LOADING, mai il
> GATING — il router vero decide sempre → accuratezza preservata (un miss = latenza, non errore).

## 1. Architettura dell'aggancio — flusso dati

## Stato locale 2026-07-08

Il piano sotto resta valido come mappa architetturale, ma lo stato DS4 locale e'
piu' avanti e piu' specifico:

- DS4 riconosce il formato hidden `SPX1` in commit `/root/ds4` `bec221c`
  (`spex: recognize hidden SPX1 predictor`).
- DS4 ha un path sperimentale di hidden-readback prefetch in commit `818ebcf`,
  ma non va usato come default: richiede `ds4_gpu_tensor_read` del router input
  e sincronizza CPU/GPU, peggiorando TTFT sui run locali.
- DS4 ha un primo foothold GPU-side in commit `e85e256`: con
  `DS4_SPEX_HIDDEN_GPU_LOAD=1` carica gli 86 MiB di pesi SPX1 su un
  `ds4_gpu_tensor` senza attivare scoring o prefetch.
- DS4 ha il primo path layout-correct di score/topK GPU in commit `dfceee3`:
  `DS4_SPEX_HIDDEN_GPU_SCORE=1` lancia un kernel dedicato per il layout SPX1
  `[hidden][expert]`, produce score 256-wide e topK su GPU, ma non consuma
  ancora quegli ID per il prefetch.
- Primo overhead check HTML160 (J23): score/topK GPU attivo ma prefetch off ha
  chiuso a 63.559s server contro 61.004s del miglior baseline caldo; quindi il
  costo del kernel non sembra il blocker principale.
- Il launcher locale tiene quindi `DS4_SPEX_HIDDEN_PREFETCH=0`.
- Lo script `scripts/analyze_spex_hidden_trace.py` valuta offline un file SPX1
  contro trace reali `DS4_SPEX_TRACE_HIDDEN` + `DS4_SPEX_TRACE_ROUTING`.
- Prima trace locale breve (J15): top6 recall 0.5155 / weighted 0.5893;
  top12 recall 0.6368 / weighted 0.7021; top23 recall 0.7260 / weighted
  0.7776. Quindi il predittore hidden contiene segnale utile; il collo di
  bottiglia e' il runtime, non l'artifact SPX1.

Nuovo prossimo passo: consumare il topK GPU senza readback host. Il path
score/topK esiste, ma gli ID restano su device; ora bisogna alimentarli al
prefetch/residency next-layer senza reintrodurre la sincronizzazione CPU/GPU.
Qualsiasi soluzione che legge `ffn_norm` o il topK su CPU resta diagnostica.

Aggiornamento architetturale dopo la lettura del loader CUDA: il consumer non
puo' essere "zero host" nel senso forte, perche' il prefetch SSD e' guidato da
CPU metadata (`ds4_gpu_stream_expert_cache_seed_experts_async` prende
`int32_t *expert_ids`, ordina priorita' su host, calcola offset e avvia le copie).
Il design corretto e' quindi un handoff D2H asincrono minuscolo:

1. GPU score/topK produce `g->spex_hidden_topk`.
2. `cudaMemcpyAsync` copia topK (6-23 int) su pinned host buffer dedicato,
   registrando un evento sullo stream corretto.
3. Il lato CPU consuma il buffer del layer precedente solo se l'evento e'
   completato; se non e' pronto, salta il prefetch e lascia il miss fallback.
4. `seed_experts_async` resta il consumer iniziale, ma non deve mai fare una
   sincronizzazione per-token sul topK.

Questo e' diverso dal vecchio hidden-readback: non legge `ffn_norm` (4096 float)
e non blocca il decode per calcolare gli ID su CPU; legge solo il risultato
compatto della predizione, con fallback sicuro.

Aggiornamento locale J29, 2026-07-08:

- Implementato in `/root/ds4` un primo handoff topK GPU -> pinned host:
  `ds4_gpu_async_read` usa host pinned, stream CUDA non bloccante, evento di
  dipendenza dal default stream e `cudaEventQuery` nel consumer.
- Nuovi env DS4 sperimentali:
  `DS4_SPEX_HIDDEN_GPU_PREFETCH=1` abilita score/topK + readback async degli ID;
  `DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN=1` legge gli ID ma non semina cache;
  `DS4_SPEX_PREFETCH_PROFILE=1` abilita il log per-layer. Bug corretto: il
  valore `0` ora e' davvero off, non solo "env presente".
- Smoke dry-run con profile acceso ha dimostrato che il vecchio approccio
  sincrono era sbagliato: moltissimo log/serializzazione, fino a ~44s server per
  8 token nel microtest. Dopo async + profile off, stesso microtest caldo:
  prompt ~6.6s, decode ~3.6s, finish ~10.2s.
- Prefetch reale cap=6 nello stesso microtest non paga ancora: run caldo
  prompt ~7.1s, decode ~12.2s, finish ~19.3s. Quindi il bridge e' corretto come
  plumbing, ma il consumer `seed_experts_async`/cache path deve essere profilato
  prima di diventare default.
- Decisione: lasciare `DS4_SPEX_HIDDEN_GPU_PREFETCH=0` nel launcher. Usare il
  path solo per test mirati con profile, mai come setup utente.

Aggiornamento locale J30, 2026-07-08:

- Patch salvate in repo:
  `patches/ds4/0015-spex-hidden-async-topk-handoff.patch` e
  `patches/ds4/0016-spex-hidden-gpu-prefetch-stats.patch`.
- DS4 locale ha ora contatori `DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS=1`:
  `scheduled`, `schedule_failed`, `ready`, `not_ready`, `zero`, `dry`,
  `seed_calls`, `seed_ok`, `seed_failed`, `skipped_all_resident`,
  `candidate_experts`, `seed_experts`, `resident_before`, `seed_ms`.
- Aggiunto filtro missing: prima del seed, il runtime conta gli expert gia'
  residenti e semina solo i mancanti; se sono tutti residenti, salta il seed.
- A/B microtest 8 token, cap=6, 2 run:
  baseline finish server 11.654s / 9.228s; dry-run async 10.394s / 16.614s;
  prefetch reale prima del filtro 12.913s / 11.172s; prefetch reale con filtro
  13.751s / 11.180s.
- Finding meccanicistico: nel microtest con filtro,
  `scheduled=672`, `ready=624`, `not_ready=0`, `schedule_failed=0`,
  `candidate_experts=3744`, `seed_experts=3744`, `resident_before=0`,
  `seed_ms=3604.641`. Quindi il bridge topK e' pronto in tempo; il problema
  non e' readback, ne' duplicato di expert residenti. Il costo viene da 624
  seed reali di 6 expert, su quasi tutti i layer/token.
- Decisione aggiornata: non abilitare `DS4_SPEX_HIDDEN_GPU_PREFETCH` nel setup
  utente. Il prossimo test utile non e' "prefetch sempre", ma admission piu'
  selettiva: meno layer, meno frequenza, cap variabile, o seed solo quando PACE
  sta per scendere/respirare.

Micro-step codice suggerito:

- in `ds4_cuda.cu`, aggiungere una piccola pool pinned host per topK SPEX
  (`cudaMallocHost`, 2-4 buffer, 32 int bastano per cap <= 32);
- aggiungere `ds4_gpu_spex_topk_readback_async(const ds4_gpu_tensor *topk,
  uint32_t n, uint32_t layer, ...)`, che fa `cudaMemcpyAsync(...,
  cudaMemcpyDeviceToHost, stream)` e `cudaEventRecord`;
- aggiungere una query non bloccante (`cudaEventQuery`) che restituisce
  "ready/not-ready" e copia gli ID nel buffer CPU usato da
  `ds4_gpu_stream_expert_cache_seed_experts_async`;
- nel path DS4, schedule del topK a layer L dopo score/topK; consumo per
  prefetch L+1 solo se il buffer del layer precedente e' ready;
- metriche minime: scheduled, ready, skipped_not_ready, seeded, seed_failed,
  exposed_miss_delta vs `DS4_SPEX_HIDDEN_GPU_SCORE=0`.

Sorgente DS4 verificato dopo J15:

- `g->ffn_norm` e' gia' il tensor F32 device-resident da usare come feature.
- `ds4_gpu_matmul_f16_tensor` esiste gia' per pesi F16 residenti nel GGUF.
  Per SPX1 non basta: l'artifact e' memorizzato input-major `[hidden][expert]`,
  mentre la matmul generica assume output-major `[expert][hidden]`.
- Commit `/root/ds4` `dfceee3` aggiunge quindi un kernel dedicato
  `ds4_gpu_spex_hidden_score_tensor` per calcolare score F32 dal layout SPX1.
- `ds4_gpu_indexer_topk_tensor` esiste gia' e puo' trasformare scores in topK.
- Il primo micro-step pratico rimasto e' un consumer device-side del topK:
  copiarli in una struttura pinned-host di residency/prefetch via async D2H,
  senza readback bloccante per-token.

Verificato: al layer L, dopo `metal_graph_encode_decode_layer` (ds4.c:19399) e lo swap
(ds4.c:19412-19414), `g->cur_hc` contiene l'hidden di L = input di L+1. La riga **19391-19392**
fa GIÀ un readahead dei pesi NON-expert di L+1 (`metal_graph_stream_readahead_layer_decode`).
È il gemello strutturale esatto di ciò che SPEX aggiunge per gli expert. **HOOK confermato: ds4.c:19415.**

```
Layer L (for a ds4.c:19385):
  [19399] encode_decode_layer(L)  -> router L scrive g->router_selected (15566); MoE compute parte
  [19412-19414] swap: g->cur_hc = hidden(L)
  [19415] === HOOK: spex_prefetch_next_layer(g, il, il+1) ===
            (a) feature: selected_L (markov) OPPURE g->cur_hc (hidden)
            (b) score[e] per expert di L+1
            (c) top-N -> confidence head -> STS -> admission (p > tau)
            (d) table L+1 (graph_stream_expert_table_make, ds4.c:3372)
            (e) seed_experts_ASYNC(table_L1, ids, prio, n) su g_stream_selected_upload_stream (fire-and-forget)
  [il++] encode_decode_layer(L+1):
            router VERO di L+1 (15566) -> cuda_stream_selected_load (15649)
              -> cuda_stream_expert_cache_find (1935): HIT se prefetchato (0 stall) / MISS = load on-demand
```

### Punti d'aggancio (file:riga)
| # | Cosa | File:riga | Azione |
|---|------|-----------|--------|
| A1 | Chiamata predittore+prefetch L+1 | ds4.c:**19415** | inserire `spex_prefetch_next_layer(...)` |
| A2 | Idem nel loop batched | ds4.c:**19348** (dopo swap 19346-19348) | copia dell'hook (se batched attivo) |
| A3 | Nuova primitiva async | ds4_cuda.cu nuova fn + `ds4_gpu.h:124` | `..._seed_experts_async` che NON sincronizza |
| A4 | Togliere sync bloccante | ds4_cuda.cu:**2237** | variante che salta `cudaStreamSynchronize`, ritorna evento |
| A5 | Wait cross-stream nel consumer | ds4_cuda.cu:**~2011** (`load_slot`) | aggiungere `cudaStreamWaitEvent` (oggi ASSENTE) |
| A6 | Feature markov già pronta | ds4.c:**921** (`ds4_expert_profile_record`) + `prev_selected` a 788 | riusare |

**Primitiva riusabile (certa):** `ds4_gpu_stream_expert_cache_seed_experts` (ds4_cuda.cu:3246,
firma `table, expert_ids, expert_priorities, n_experts`) — è letteralmente la firma di un prefetch
predittivo. Serve solo la variante async (A3/A4/A5).

**Shape modello (confermato dalla map spec):** `DS4_N_EXPERT_USED = 6` expert routati/layer (DeepSeek-V4-Flash).
`DS4_N_EXPERT` (E) e L da confermare con grep prima di fissare gli offset del file `.spex`.

## 2. Il predittore in C

Confidence-head + STS = pochi float/layer (`a[L]`, `b[L]`, `T[L]`) + (ramo markov) matrice
transizione. Girano triviali in C. **Partire dal ramo markov** (feature già in `selected[]`, zero
dipendenza da dump POD). Ramo hidden = matmul `W_l·h_L` (richiede probe dumpato dal pod).

### Formato export `.spex` (little-endian)
```
Header (32B): magic "SPEX" | version u32=1 | predictor u32(0=markov,1=hidden) |
              n_layer u32 | n_expert u32 | n_embd u32 | topN u32 | reserved u32
Per-layer (L): a f32 | b f32 | T f32
markov: per layer/expert-cur: m u16 + (eid u16, count u32)*m  (sparse top-32/riga)
hidden: W[l] f16[E*N_embd] | b_probe[l] f32[E]
```
**Gap confermato:** l'esportatore Python→C NON esiste (`spex_loop.py` salva solo miss-rate/ECE).
Va scritto (`--export-c out.spex`, serializza `a,b,T,C` — fit a spex_loop.py:79-134, admission a 154-186).

### Firma C (`ds4_spex.h`, nuovo)
```c
typedef struct { uint32_t predictor, n_layer, n_expert, n_embd, topN;
  float *a,*b,*T; uint32_t *mk_row_ptr; uint16_t *mk_col; uint32_t *mk_cnt;
  uint16_t *probe_W; float *probe_b; } ds4_spex_params;
int  ds4_spex_load(const char *path, ds4_spex_params *out);
void ds4_spex_free(ds4_spex_params *p);
void ds4_spex_score(const ds4_spex_params*, uint32_t next_L, const int *selected_L,
                    uint32_t n_used, const float *h_L, float *scored /*[E]*/);
uint32_t ds4_spex_admit(const ds4_spex_params*, uint32_t next_L, const float *scored,
                        float tau, uint32_t cap, int32_t *out_ids, uint32_t *out_prio);
```
`score` (markov): per ogni `c` in selected_L accumula count sparse in scored[eid].
`admit`: `feat=log1p(score)`; `z=a*feat+b`; `p=sigmoid(z/T)`; tieni p>tau, ordina desc, tronca a cap.
Replica `spex_loop.py:79-186`.

## 3. Piano a stadi

**Stadio 0 — strumentazione/BASELINE (nessun prefetch nuovo).** Contatori hit/miss in
`cuda_stream_expert_cache_find` (ds4_cuda.cu:1935) + exposed-stall wrappando
`cuda_model_copy_to_device_streamed` (2152) con timer. Dump dietro `DS4_SPEX_STATS=1`. Run 3060,
prompt fisso, stampa hit_rate / miss-per-tok / stall_ms-per-tok / t/s. **Questo stadio DECIDE se il
progetto vale:** se con RAM calda (`KEEP_MODEL_PAGES=1`) lo stall è già ~0, SPEX ha poco da mordere.

**Stadio 1 — prefetch minimo (plumbing).** Predittore banale = prefetcha per L+1 gli stessi expert
di L (`router_selected`). Zero parametri/file. Implementa A3/A4/A5. Verifica: (1) output IDENTICO
alla baseline (se cambia → race → bug); (2) hit_rate su; (3) t/s ≥ baseline. Prova che copia async +
wait cross-stream funzionano. (Il profiler misura già Jaccard adiacente a ds4.c:934 — se alto, guadagna già.)

**Stadio 2 — SPEX vero.** Scrivi esportatore Python (`--export-c`). `ds4_spex_load` all'init server
(accanto al load hotlist). Hook usa `ds4_spex_score`+`ds4_spex_admit` (markov). Verifica miss-rate vs
Stadio 1 su dominio stretto; sanity vs ECE di spex_loop.py.

**Stadio 3 — STS + tuning.** Attiva `T[L]`; ramo hidden (serve dump probe pod). Tuning `tau`/`topN`/`cap`
(=slot VRAM liberi ~8.5GB / bytes-per-expert) via env `DS4_SPEX_TAU`/`DS4_SPEX_TOPN`. Sweep τ, misura
t/s + miss + perplessità invariata. Recall atteso hidden 0.93-0.99 vs markov 0.51-0.89.

## 4. Build WSL per sm_86
```bash
nvidia-smi   # la 3060 deve essere visibile (driver Windows la espone)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6      # NON 'cuda' (evita il driver Linux)
export PATH=/usr/local/cuda/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc --version
cd <ds4-src> && make cuda CUDA_ARCH=sm_86        # verificare nome var/target nel Makefile
```
Fallback pod se il toolkit WSL dà grane; ma il loop veloce è WSL locale (egress pod throttlato).

## 5. Misura paper-worthy (3060 reale)
- **t/s wall-clock** (mediana+IQR, ≥5 run, 256+ tok, scarta warm-up): baseline vs SPEX-markov vs SPEX-hidden+STS.
- **miss-rate cache** (da `g_spex_stat`): metrica meccanicistica che spiega il t/s.
- **exposed-stall** (ms/tok): tempo in cui il compute aspetta un load reattivo → SPEX deve farlo crollare.
- **Controlli (rigore paper):** RANDOM (stesso budget, isola "predizione utile" da "più cache aiuta"),
  hotlist statica, 2 domini (stretto→SPEX vince / generale→pareggia, mai peggio), con/senza `KEEP_MODEL_PAGES=1`.
  Caso paper-worthy = pagine FREDDE (SSD reale). Invarianza accuratezza: perplessità identica baseline vs SPEX.

## 6. Rischi / la domanda-chiave n.1
**DA RISOLVERE PER PRIMA:** il consumer expert (`cuda_stream_selected_load`, ds4.c:15649 → path in
ds4_cuda.cu) sincronizza con lo stream di upload via evento, o assume slot residenti? Oggi è tutto
bloccante (sync 2237) quindi non si pone; rendendolo async si crea la possibilità di una **race**
kernel-MoE vs upload-in-volo. `cudaStreamWaitEvent` è ASSENTE in ds4_cuda.cu. **Prima azione: leggere
il consumer e disegnare il protocollo evento** (record su upload stream dopo seed → wait su default
stream prima del kernel MoE di L+1). Senza questo, Stadio 1 corrompe l'output.

Altri: il margine potrebbe non esistere (RAM calda → stall ~0, lo dice Stadio 0); E/L non confermati
numericamente (grep); esportatore Python→C da scrivere; probe hidden solo su pod (markov è self-contained →
parti da lì); il router vero usa `g->ffn_norm` (dentro L+1) non `g->cur_hc` che SPEX legge → il ramo hidden
vede un'approssimazione (limita recall, ok col contratto loading-non-gating); `graph_stream_expert_table_make`
va verificata chiamabile fuori contesto senza side-effect; `cap` da `cuda_stream_expert_cache_live_budget`
(1510, fa cudaMemGetInfo) è costoso per-token → cachearlo.

## Prossimo passo concreto
Leggere il consumer expert in ds4_cuda.cu (dietro `cuda_stream_selected_load`) per chiudere la
domanda-chiave race → poi Stadio 0 (contatori) → misurare baseline sul 3060. **Se il margine c'è, Stadio 1.**

---

# Fase 1b — DOMANDA-RACE CHIUSA (2026-07-04, analisi sorgente verificata)

**Verdetto: la race È reale** (silenziosa, timing-dependent) se il prefetch va async in modo ingenuo.
Numeri di riga verificati sul sorgente (`scratchpad/ds4-src/`).

**Mappa stream CUDA:** esistono 3 stream non-default, TUTTI per I/O host→device
(`g_model_prefetch_stream` :89, `g_model_upload_stream` :90, `g_stream_selected_upload_stream` :222).
**Non esiste uno stream di compute:** tutti i kernel MoE (`ds4_cuda.cu:12424-12599`) lanciano con
`<<<grid,block>>>` senza 4° arg → **default stream (0)**; `cublasSetStream` = 0 occorrenze. Quindi
**compute su stream 0, copia expert su `g_stream_selected_upload_stream` = stream diversi, non ordinati.**

**Oggi è sicuro solo per DOPPIA barriera bloccante:** `ds4_gpu_end_commands` (:14286 = `cudaDeviceSynchronize`)
incornicia + `cudaStreamSynchronize(upload_stream)` a **:2237** dentro ogni copia. Il kernel MoE assume slot
già residenti. **`cudaStreamWaitEvent` = ZERO occorrenze in tutto ds4-src** (è la primitiva mancante). Gli
eventi esistenti servono solo a riciclare i 4 pinned staging buffer, NON a ordinare compute↔copia. L'"async"
esistente (`overlap_selected_shared`) è un worker pthread con copia bloccante, non pipelining GPU.

**Protocollo evento (la fix), per-LAYER non per-slot:**
- RECORD: in `cuda_model_copy_to_device_streamed` (:2152), variante async che a **:2237** fa
  `cudaEventRecord(g_selected_upload_done_event, g_stream_selected_upload_stream)` invece del sync e ritorna.
  Creare l'evento una volta accanto a :1749 (`cudaEventCreateWithFlags(..., cudaEventDisableTiming)`).
- WAIT: nuova API `ds4_gpu_wait_selected_upload()` = `cudaStreamWaitEvent(0, g_selected_upload_done_event, 0)`,
  inserita in `metal_graph_decode_cuda_selected_load` (`ds4.c:14304`) PRIMA del kernel MoE (`ds4.c:15691`).
- Un evento per batch-prefetch di un layer (il kernel MoE consuma tutti gli slot insieme).
- MISS fallback: il ramo miss on-demand (`ds4_cuda.cu:2966-3068`, LRU + direct-load bloccante) RESTA — corretto.

**Correzioni ai punti d'aggancio:**
- A3: la funzione da forkare è la primitiva reale `cuda_model_copy_to_device_streamed` (:2152) /
  `cuda_stream_selected_cache_begin_compact_load` (:2875), NON `seed_experts` (:3246, è il path hotlist-priorità).
- A4: giusto (parametrizza async/out_event a :2237). A5: giusto (`cudaStreamWaitEvent` su stream 0, oggi 0 usi).

**3 NODI PIÙ PROFONDI che il piano non aveva colto (il vero design dello Stadio 1):**
1. **Il `cudaDeviceSynchronize` a monte (:14286) serializza tutto.** Per prefetchare L+1 devi predirne gli
   expert PRIMA di quel sync (dal router di L o dal predittore) — altrimenti non anticipi nulla. **Questo è
   il nodo centrale, non un dettaglio.**
2. **Upload stream condiviso:** miss-on-demand e prefetch userebbero lo stesso `g_stream_selected_upload_stream`
   → si serializzano. Valuta un 2° upload stream per il prefetch, o accetta la serializzazione (ok Stadio 1).
3. **Destination cache a buffer singolo:** `g_stream_selected_cache.gate_ptr` è unica → prefetchare L+1 negli
   stessi buffer mentre il kernel di L li legge = overwrite. Serve ping-pong dei buffer destinazione, o usare
   la LRU `g_stream_expert_cache` già esistente per ospitare il prefetch senza toccare il compact di L.

**PRIMO MICRO-STEP (Stadio 1a, iso-comportamento, env-gated `DS4_SELECTED_UPLOAD_EVENT=1`):** sostituire il
sync esistente con un vero evento sullo STESSO layer (non ancora prefetch di L+1): crea evento a :1749; variante
async di copy a :2237 (record invece di sync); `cudaStreamWaitEvent(0, evt, 0)` nel consumer prima del kernel.
Verifica **token IDENTICI** con flag ON vs OFF (nessuna race introdotta) → POI sposta il record al prefetch di L+1.
Serve build (task #2) per testare. Verifica race a runtime con `compute-sanitizer --tool racecheck` (sul pod).
