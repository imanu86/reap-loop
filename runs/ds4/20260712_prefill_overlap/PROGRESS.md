# 20260712_prefill_overlap — progress log

## RISULTATO FINALE (TTFT "prompt done", prompt coffee 64 tok, chunk 512)
| Regime | OFF | S1 | S1+S2 | Verdetto |
|---|---|---|---|---|
| COLD (drop_caches per arm) | 105.911s / 112.666s (bracket) | **95.676s (−9.7/−15.1%)** | 168.736s (+59%) | S1 vince, S2 refutato |
| WARM (page cache pieno) | 14.736s | **11.705s (−20.6%)** | 21.475s (+46%) | S1 vince, S2 refutato |

Bit-exact: TUTTI gli arm (11 confronti: OFF/S1/OFF2/S1b/S1S2/wOFF3/wS1c/wS1S2/wOFF4 sul warm-set 217 char, cOFF/cS1/cS1S2/cOFF2 sul cold-set 69 char; il cold-set è prefisso esatto del warm-set) = output byte-identico. Decode invariato con env S1 (2.15 vs 2.19 t/s warm).
Raccomandazione: **DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1 da tenere ON** (sicuro, bit-exact, −10/−21%); **DS4_REAP_PREFILL_READAHEAD OFF su WSL2** (il disco virtuale non scala con la QD: i thread rubano banda al pread critico; ri-testare su pod con NVMe reale).

Obiettivo: overlap del caricamento esperti nel PREFILL (era seriale QD1, defer_upload_sync hardcoded a 0).

## Setup
- Base: copia `/root/ds4-fullstack` → `/root/ds4-prefill-work` (md5 binario base ds4: `44c6cac648948671645677cdfc938c7b`).
- ATTENZIONE: ds4-fullstack aveva modifiche NON committate (patch 0044/0045 SPEX lane, ~1100 righe su ds4.c/ds4_cuda.cu, oltre l'HEAD `da0b3f6` = 0040 pin-by-mass). Copiate tali quali e committate come baseline WIP (`67015bb`) nel work-tree per isolare il MIO diff.
- Build: `make cuda CUDA_ARCH=sm_86`, 0 errori 0 warning (build.log, build2.log).
- GPU serializzata con `flock /tmp/ds4-gpu.lock`.

## Implementazione (patch 0046, entrambe env-gated, OFF = byte-identico)
### S1 — DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1
- `cuda_stream_selected_cache_begin_compact_load`: nuovo `prefill_defer_upload_sync = prompt_like_batch && env` → passato a `cuda_stream_expert_cache_load_slot` (era letterale 0). Decode (prompt_like_batch=0) intoccato.
- `cuda_stream_selected_upload_event_enabled()` allargato: env 0046 arma il ramo evento (senza, defer=1 cade comunque nel ramo bloccante). Verificato: in QUESTO albero il gate ha solo 2 consumatori (ramo defer + wait_if_recorded); `ds4_gpu_end_commands` è `cudaDeviceSynchronize` incondizionato → nessun rilassamento barriere decode.
- `cuda_stream_expert_cache_copy_to_compact`: nuovo param `defer_async` → il D2D slot→compatto va in `cudaMemcpyAsync` sullo STESSO stream upload (FIFO dello stream = ordering corretto senza wait host), invece del `cudaMemcpy` bloccante su stream 0.
- Dopo il loop: UN `cudaEventRecord` fresco sull'upload stream (copre H2D+D2D di tutto il batch) → il wait esistente (`ds4_gpu_wait_selected_upload` in `metal_graph_cuda_stream_prefill_batch_selected_load`, ds4.c) lo consuma PRIMA della GEMM. Pattern PASS1-issue/PASS2-wait.
- **FIX da review avversaria (WAR sui staging buffer)**: pre-0046 ogni chiamata a `cuda_model_copy_to_device_streamed` finiva con streamsync → i 4 staging buffer rotanti erano sempre liberi all'ingresso e il loop chunk saltava l'event-wait per i primi 4 chunk. Col defer, la chiamata SUCCESSIVA poteva sovrascrivere (pread) un buffer ancora sorgente di una copia H2D in volo → corruzione silenziosa dei pesi. Aggiunto `g_stream_selected_stage_defer_inflight`: settato al ritorno deferred, azzerato dopo streamsync bloccante; finché attivo, event-wait su OGNI riuso buffer (anche primi 4). Env OFF: flag mai settato, condizione identica a prima.
- Review avversaria completa (6 quesiti stream-ordering): verdetto SAFE WITH CAVEAT; caveat 1 (rilassamento barriera decode via end_commands_pipeline) NON applicabile a questo albero (funzione inesistente, verificato con grep); caveat 2 (WAR staging) FIXATO come sopra.

### S2 — DS4_REAP_PREFILL_READAHEAD=1
- Nel loop per-layer ssd_streaming di `ds4_session_eval_layer_slice` (ramo batch, n_tokens>1): prima di processare il layer `il`, fire-and-forget `ds4_reap_prefetch_batch` (macchinario 0043 invariato: page-in host multi-thread, single-flight) su TUTTI i 256 expert del layer `il+1` (bounded a layer_end). Env foldato in `ds4_reap_prefetch_delta_enabled()` così basta 1 env var.

## Run (server freddo = riavvio processo tra i run; page-cache RAM caldo, documentato)
- Prompt: coffee (Bean & Brew), temp 0, max_tokens 60, `think:false` (primo tentativo in thinking mode: decode 0.16 t/s, 22 tok in 139s → timeout; passato a non-thinking).
- Run 0 (scartato, thinking): prompt done 150.333s — primo run assoluto, page cache freddo.
- OFF (binario pre-fix WAR, ma env OFF = identico; page cache caldo): **prompt done 58.309s**.
- S1 (binario con WAR-guard ma SENZA cursore cross-call): **prompt done 68.640s** — bit-exact vs OFF: **SÌ** (217 char identici), ma PIÙ LENTO.
  - Diagnosi: la WAR-guard senza rotazione cross-call dei 4 staging buffer serializza pread(N+1) contro copy(N) (entrambi vogliono stage[0], ogni call ripartiva da indice 0) → il wait si è solo spostato, overlap zero + overhead eventi.
  - Fix: `g_stream_selected_stage_cursor` globale (solo con env 0046 ON): la rotazione dei buffer CONTINUA tra le chiamate → il wait cade sulla copia di 4 chunk prima (di norma già finita) → fino a 4 pread avanti rispetto allo stream H2D. Env OFF: indice per-call originale, byte-identico.
- Rebuild (build3.log, 0 err/0 warn), md5 ds4-server `30c5c5833b46ba9c24357aa0415eb4fc`.
- Sequenza 2 (binario col cursore, S2 su slice-loop): OFF2=33.131s → S1b=13.810s → S1S2=12.063s. TUTTI bit-exact vs OFF (217 char).
  - CONFOUND SCOPERTO: page-cache RAM (WSL=60GB, cache≈59GB) si scalda progressivamente tra i run → gli arm successivi sembrano più veloci a prescindere. Serve bracketing (OFF prima E dopo) e serie COLD (drop_caches).
- SCOPERTA percorso: per prompt ≤ prefill-chunk il prefill del server NON passa dal loop generico di `ds4_session_eval_layer_slice` ma dal ramo full-span → `metal_graph_prefill_layer_major` (loop per-layer proprio, ds4.c ~24829). L'hook S2 iniziale era sul loop sbagliato → S1S2 della sequenza 2 misurava di fatto S1-only. Il readahead pre-esistente lì (`metal_graph_stream_readahead_layer`) è NO-OP su Linux (F_RDADVISE è solo Darwin; variante madvise env-gated spenta) → conferma la diagnosi QD1. Hook S2 aggiunto ANCHE al loop layer-major.
- Rebuild finale (build4.log, 0 err/0 warn), md5 ds4-server `eb4d30cf5fc5df433d2527d75477e35a`.
- Verifica: nel run S1S2 (seq 2) ZERO righe "fattorino delta" nel server log → conferma che S2 non era mai partito (hook sul loop sbagliato).
- Sequenza 3 (bracketed, warm): wOFF3=14.736s → wS1c=11.705s (−20.6%) → wS1S2=21.475s → wOFF4=40.049s.
  - Il bracket (OFF 14.7 → 40.0) mostra che wS1S2 ha AVVELENATO la cache per i run successivi (churn da page-touch). Dato warm affidabile = coppia adiacente wOFF3→wS1c: S1 = −20.6% su warm. wS1S2 warm inaffidabile per confronti diretti ma il segno è chiaro: S2 su cache calda NUOCE.
  - wS1S2 warm PEGGIORA (+46% vs wOFF3): su cache calda il page-touch dell'INTERO layer successivo (1731 MiB × 8 thread, 354-1260ms/batch, 8/39 batch — gli altri saltati dal single-flight) è puro overhead + churn di cache (RAM 60GB < modello 86GB → pagine utili evictate). S2 ha senso solo su cache FREDDA (il regime della diagnosi originale 115-213s) → serie cold decisiva.
- Sequenza 4 (cold, drop_caches prima di ogni arm, MAXTOK=20, bracket cOFF2 in coda): cOFF=105.911s → cS1=95.676s (−9.7%) → cS1S2 → cOFF2 (in corso; cache 59GB→7GB confermato a ogni drop). Determinismo: output cold (20 tok) = prefisso esatto dell'output warm (60 tok).
- Osservazione live cS1S2 (cold): batch fattorino 1731 MiB in ~3-4.9s (~350-580 MB/s aggregate a QD8) — su WSL2 il disco virtuale NON scala con la queue-depth; i thread di readahead COMPETONO col pread del layer corrente invece di aggiungere banda.
- **cS1S2 = 168.736s (+59% vs cOFF): S2 REFUTATO su questo stack I/O (WSL2 virtual disk), sia cold che warm.** S1 resta l'unica leva vincente (−9.7% cold, −20.6% warm), bit-exact ovunque.
- Bracket cold: cOFF2 = 112.666s (spread OFF-OFF 105.9–112.7 ≈ ±3%) → il guadagno cS1 (95.7s) è FUORI dallo spread = reale. Fattorino in cS1S2: 24/38 batch (14 saltati dal single-flight), ~350-580 MB/s aggregate.
- Decode cold per-arm rumoroso (0.94/0.66/0.15 t/s): stato cache post-prefill diverso; il confronto controllato resta quello warm (S1 neutro sul decode).
- Chiusura: server killato, porta 8071 libera, /root/ds4-fullstack INTOCCATO (md5 binari invariati).
- Nota operativa: la riga "wall_clock" appesa al log può venire sovrascritta dal server (fd non-append) durante lo shutdown drain → non usarla come gate. E MAI pkill -f con pattern che compare nella propria command line (self-match, un launcher ucciso così).
- Bit-exact seq 3 (wOFF3/wS1c/wS1S2 incluso S2 attivo): TUTTI identici a OFF.
- Decode warm: wOFF3 avg 2.15 t/s vs wS1c avg 2.19 t/s → l'env S1 NON tocca il decode (rumore). wS1S2 decode 0.66 t/s = danno del churn di cache (l'hook S2 spara solo con n_tokens>1, quindi è la cache avvelenata, non readahead attivo in decode).
