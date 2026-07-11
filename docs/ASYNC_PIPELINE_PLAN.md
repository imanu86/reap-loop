# Async Prefetch Pipeline — Design Plan

> Rompe il blocking-sync per-copia e collega il predittore SPEX cross-layer come
> loader (mai gating). Staged, ogni stadio committabile/reversibile (env off =
> no-op) con **gate di verifica** (bit-exact + t/s + overlap%). Sintesi dei tre
> survey di sessione: (a) locality temporale expert (`analysis fc1ecf8`,
> `overlap.py`/`tl_out.txt`), (b) integrazione SPEX (`SPEX_INTEGRATION_PLAN.md`,
> Fase-1b race chiusa), (c) plumbing già staged (patch `0001`–`0031`). Numeri di
> riga verificati sul sorgente ds4 in `SPEX_INTEGRATION_PLAN.md`; il sorgente
> live `ds4.c`/`ds4_cuda.cu` è in WSL, i diff CUDA rilevanti sono nei patch.

Modello: DeepSeek-V4-Flash, 40 layer, **6 expert usati/layer** (240 coppie
(layer,expert)/token), ~**6.75 MiB/expert** (2-bit cold-lossless).

---

## 1. STATO ATTUALE — cosa esiste (macchina SPEX) vs cosa manca (overlap)

**Esiste già (la macchina, tutta env-gated, default byte-identica):**
- **Predittore SPEX in C** (`patches/ds4/ds4_spex_predict.{c,h}`):
  `ds4_spex_predict_topk(m, src_layer, hidden_L, K, out_ids, scratch)` →
  `score[e] = Σ_i h_L[i]·W[L][i·E+e]`, top-K = gli expert di **L+1** da
  prefetchare. Puro C, trascurabile. OFF se `DS4_SPEX_FILE` non settato.
- **Variante GPU del predittore** (`0015`/`0016`): score+topK su GPU, readback
  D2H **async** degli ID compatti su pinned host, consumo con `cudaEventQuery`
  non bloccante. Env `DS4_SPEX_HIDDEN_GPU_PREFETCH`.
- **Primitiva di seed** `ds4_gpu_stream_expert_cache_seed_experts_async(table,
  ids, prio, n)` (`0003`) — è letteralmente la firma di un prefetch predittivo.
- **Evento upload** (`0002`, `DS4_SELECTED_UPLOAD_EVENT`): sostituisce il sync
  bloccante `:2237` con `cudaEventRecord` + `cudaStreamWaitEvent(0,evt)` nel
  consumer.
- **Hook cross-layer** (`0003`, `DS4_SPEX_PREFETCH_NEXT_LAYER`): a fine layer L
  memorizza i selected di L e semina L+1.
- **Residency pin** (`0031`, `DS4_PACE_PIN`): l'LRU salta un sottoinsieme hot di
  keep (residenza eviction-immune) + rotazione CUSUM su domanda; **compone con
  SPEX** (SPEX semina → 0031 pinna → il filtro skip-all-resident smette di
  ri-seminare, uccide il seed H2D per-token, il costo J30).
- **Stats stage-0** (`0001`, `DS4_SPEX_STATS`): hit/miss cache, exposed-stall
  ms/tok.
- **Cache LRU expert** `g_stream_expert_cache` già residente e riusabile come
  destinazione del prefetch (evita un nuovo doppio-buffer).

**Manca — l'overlap async vero. Diagnosi precisa del blocking-sync:**
1. **Nessuno stream di compute.** I 3 stream non-default sono TUTTI I/O H2D
   (`g_model_prefetch_stream`, `g_model_upload_stream`,
   `g_stream_selected_upload_stream`). Tutti i kernel MoE lanciano su **default
   stream 0** (`cublasSetStream` = 0 occorrenze). Compute e copia sono già su
   stream diversi ma **non ordinati**.
2. **Doppia barriera bloccante** rende sicura quella non-ordinazione a costo
   dell'overlap: `cudaStreamSynchronize(upload_stream)` a **`ds4_cuda.cu:2237`**
   dentro **ogni** copia + `cudaDeviceSynchronize` a **`ds4.c:14286`**
   (`ds4_gpu_end_commands`) che incornicia il decode. Il kernel MoE assume gli
   slot **già residenti**.
3. **`cudaStreamWaitEvent` = ZERO occorrenze** nel sorgente: manca la primitiva
   per ordinare compute↔copia a basso costo. Gli eventi esistenti riciclano solo
   i 4 staging buffer pinned. L'`overlap_selected_shared` esistente è un worker
   pthread con **copia bloccante**, non pipelining GPU.
4. **Load reattivo, non predittivo.** Il vero router di L+1 (`ds4.c:15566`) →
   `cuda_stream_selected_load` (`15649`) → `cuda_stream_expert_cache_find`
   (`1935`): HIT = 0 stall, **MISS = load on-demand bloccante**. La copia parte
   DOPO che il router ha deciso → tempo di copia **interamente esposto**.

**→ Overlap oggi = 0 per costruzione.** La copia expert non si sovrappone MAI al
compute: parte reattiva e viene chiusa da una barriera prima del kernel.

---

## 2. DISEGNO MINIMO dell'async pipeline (il cambio-motore più piccolo che dà overlap reale)

Idea: **anticipare la copia di L+1 dentro il compute di L+1** (attention+norm),
consumando una predizione disponibile a fine-L, e sostituire la barriera con un
**evento cross-stream** — non un nuovo stream di compute, non un nuovo buffer.

**Topologia minima (riusa ciò che c'è):**
- **compute-stream** = default stream 0 (invariato — i kernel MoE restano lì).
- **copy-stream** = `g_stream_selected_upload_stream` esistente (H2D expert).
- **Ordinamento** via evento invece di sync (0002): a `:2237` `cudaEventRecord`
  sul copy-stream; nel consumer, PRIMA del kernel MoE di L+1 (`ds4.c:15691`),
  `cudaStreamWaitEvent(stream0, evt, 0)`. Il kernel aspetta **solo** la copia
  che gli serve, non un device-sync globale.

**Prefetch cross-layer (il cuore).** Hook confermato a **`ds4.c:19415`** (dopo
lo swap `g->cur_hc = hidden(L)`; è il gemello strutturale del readahead
non-expert già presente a `19391-19392`):
```
Layer L: encode_decode_layer(L)  [19399]  -> L completo, hidden(L) pronto
  [19415] HOOK: ids_L1 = predict(hidden_L | selected_L)   // L+1, mai gating
          seed_experts_async(g_stream_expert_cache, ids_L1, prio, n)  // copy-stream, fire-and-forget
  encode_decode_layer(L+1):  attention+norm  <-- QUI la copia di L+1 gira in overlap
          router VERO L+1 [15566] -> selected_load [15649] -> cache_find [1935]
          wait(evt) [prima di 15691] -> kernel MoE L+1   // HIT se il prefetch è arrivato
```
- **Prefetch di L+1 sovrapposto al compute di L+1** (finestra = attention+norm di
  L+1). La FFN di L è già finita allo swap: l'anticipo utile è la fase pre-FFN di
  L+1 — vedi §4 rischio-anticipo.
- **Doppio-buffer risolto senza nuovo buffer:** il seed va nella **LRU
  `g_stream_expert_cache`** (residente, multi-slot), NON nel compact
  `g_stream_selected_cache.gate_ptr` (buffer singolo che il kernel di L sta
  leggendo → overwrite). Ping-pong implicito via slot LRU distinti.
- **Sync solo dove serve la correttezza:** un solo `cudaStreamWaitEvent` per
  layer, prima del kernel MoE. Il ramo **miss on-demand** (`ds4_cuda.cu:2966`,
  LRU + direct-load bloccante) **resta** come fallback → un miss è latenza, mai
  errore.
- **Scheduler (dentro il budget cache):** semina i top-K per **confidenza**
  (score SPEX o count markov) fino a `cap = C_slots_liberi`; salta gli expert
  già residenti (filtro missing di `0016`); priorità = score desc.
- **Eviction prediction-aware-LRU:** l'LRU salta gli slot **pinnati/predetti**
  (`0031`): un expert seminato e ancora "caldo" nella demand-EWMA non viene
  sfrattato prima di essere consumato dal kernel di L+1.

**Contratto invariante:** il prefetch tocca solo il **loading** (residenza cache
LRU); il router vero decide sempre il gating → **token bit-identici**. Un seed
sbagliato costa una copia inutile, non un output diverso.

---

## 3. PIANO A STADI (ogni stadio: env-gated, reversibile, con GATE = bit-exact vs baseline + t/s + overlap%)

Metriche comuni (da `0001`): `exposed_stall_ms/tok`, `hit_rate`, `t/s`
mediana(≥5 run, ≥256 tok, scarta warmup). **overlap% = 1 − exposed_stall /
copy_ms** (frazione del tempo-copia nascosta dietro il compute). Gate bit-exact =
token IDENTICI flag ON vs OFF (una differenza ⇒ race ⇒ bug, si blocca).

| Stadio | Cambio-motore | Env | GATE di verifica |
|---|---|---|---|
| **S0 — baseline/overlap attuale** | Solo contatori (`0001`), nessun prefetch nuovo. Prova il blocking. | `DS4_SPEX_STATS=1` | Misura `exposed_stall_ms/tok` e `overlap% ≈ 0`. **DECIDE se vale:** se con RAM calda (`KEEP_MODEL_PAGES=1`) lo stall è già ~0, SPEX ha poco da mordere. Registra t/s baseline. |
| **S1 — stream-separation minima (no SPEX)** | Evento invece di sync (`0002`) + prefetch **reattivo** di L+1 = stessi expert di L (`0003`, predittore banale, zero file). Implementa il wait cross-stream. | `DS4_SELECTED_UPLOAD_EVENT=1 DS4_SPEX_PREFETCH_NEXT_LAYER=1` | (1) **token bit-identici** vs S0; (2) `hit_rate` ↑ e `overlap% > 0` (anche solo il prefetch reattivo si nasconde?); (3) `t/s ≥ baseline`. Prova che copia-async + wait funzionano senza race (`compute-sanitizer --tool racecheck` sul pod). |
| **S2 — wiring predittore SPEX (hidden→L+1)** | Sostituisci il predittore banale con `ds4_spex_predict_topk(hidden_L)` (o markov `selected_L`) sul copy-stream; readback async ID (`0015`/`0016`). | `DS4_SPEX_FILE=… DS4_SPEX_HIDDEN_GPU_PREFETCH=1` | (1) **bit-exact** invariato; (2) `miss-rate < S1` su dominio stretto (il predittore batte "stessi-di-L"?); (3) `t/s ≥ S1`, costo predittore <1%/layer (§4); sanity recall vs offline (`analyze_spex_hidden_trace.py`). |
| **S3 — scheduler + eviction + confidenza** | Admission per confidenza+budget (`τ`, `cap`, top-N), pin/rotazione residency (`0031`), filtro skip-all-resident che smette di ri-seminare i pinnati. | `DS4_PACE_PIN=1 DS4_SPEX_TAU=… DS4_SPEX_TOPN=…` | (1) **bit-exact**; (2) sweep τ: `t/s` max con `miss` min e **perplessità invariata**; (3) `overlap%` massimo, seed-H2D/tok ↓ (0031 uccide il re-seed). Controllo RANDOM a pari budget (isola "predizione utile" da "più cache aiuta"). |

Ogni env a 0 = no-op → il motore torna byte-identico al post-`0024`.

---

## 4. RISCHI

- **Dipendenza-dato / quanto anticipo reale c'è (rischio n.1).** Il router vero
  di L+1 usa `ffn_norm(L+1)` (dentro L+1); il predittore usa `hidden(L)` =
  `g->cur_hc` (a fine-L) → è un'**approssimazione** a monte (limita recall, OK
  col contratto loading-non-gating). La finestra di overlap reale è solo
  **attention+norm di L+1** (la FFN di L è già chiusa allo swap). Su 3060 la
  copia dei miss è grossa (~900 MiB/tok a hit 0.44, ~66 ms/tok a hit 0.5;
  `tl_out.txt`) mentre l'attention/layer è breve → **l'overlap a 1 layer di
  anticipo può non bastare a nascondere l'intera copia.** Mitiga: (i) la locality
  è forte — miss intrinseco a finestra W16 = 0.008–0.11, W64 ≤ 0.016 → una
  **residency-window** cattura gran parte, il prefetch deve coprire solo il
  delta; (ii) anticipo multi-layer (L+2/L+3) richiede hidden **predetto** →
  errore composto, da valutare solo se S2 mostra margine. S0 quantifica il tetto.
- **Correttezza / race.** La race è **reale e silenziosa** se si va async in modo
  ingenuo (fix Fase-1b): prefetch nella LRU (non nel compact singolo-buffer),
  UN evento per layer, `cudaStreamWaitEvent` prima del kernel MoE, miss-fallback
  intatto. Gate bit-exact per-stadio + racecheck. Il prefetch è **solo loading,
  mai gating**.
- **Costo predittore (<1%/layer?).** Ramo markov = accumulo sparse su `selected_L`
  (trascurabile). Ramo hidden = matmul `W_l·h_L` (86 MiB pesi): il check J23 dà
  score/topK GPU acceso ma prefetch off ≈ +2.5s/61s (~4%) — da riportare sotto
  1%/layer con kernel dedicato e cap basso; `cap` NON da
  `cuda_stream_expert_cache_live_budget` (`cudaMemGetInfo`, costoso per-token) →
  **cachearlo**.
- **Interazione con 0031 pinning e cache esistente.** Composizione voluta: SPEX
  semina → 0031 rende la residenza eviction-immune → skip-all-resident smette di
  ri-seminare (elimina il seed-H2D per-token). Rischio: un pin troppo aggressivo
  riduce gli slot liberi per il prefetch → `cap` va calcolato al netto dei
  pinnati. `g_stream_selected_upload_stream` è **condiviso** tra miss-on-demand e
  prefetch → si serializzano; per S1 è accettabile, in S3 valutare un 2° upload
  stream. Pinning è residency non selection ⇒ non tocca `g_reap_mask_pruned`
  (resta bit-exact).

---

## 5. Nota — dimensionamento cache = parametro (split VRAM)

Il numero di slot cache **dipende dallo split VRAM** (in misura, **agent
`ad98a06b`**) → lasciato come **parametro** `C_slots`, non fissato qui:
```
C_slots ≈ (split_VRAM_GB × 1024) / 6.75 MiB-per-expert
cap_prefetch = C_slots_liberi = C_slots − pinned(0031) − compact(selected)
```
Riferimenti locality per tarare `C_slots` (`tl_out.txt`, per-dominio):
LRU b6=240exp/1.6GB → hit 0.42–0.73; b12=480exp/3.2GB → 0.61–0.997;
b24=960exp/6.3GB → 0.84–0.99. La scelta operativa di `C_slots` si chiude quando
lo split VRAM è misurato — questo piano è invariante rispetto a quel numero.
