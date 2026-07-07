# RECON — MTP/speculative decoding in ds4 upstream (pin `80ebbc3`)

> Track DSpark, step 1 del brief (`docs/briefs/BRIEF_DSPARK_MTP_DS4.md`).
> Tutti i riferimenti file:riga sono su **`antirez/ds4` commit `80ebbc3`** (verificati con
> `git show 80ebbc3:<file>` nel repo WSL `/root/ds4`). Nessuna riga di codice modificata.

> ⚠️ **INQUADRAMENTO (correzione brief 2026-07-05, commit `563d7d3`): tutto ciò che questo
> RECON mappa è il BASELINE del paper, non DSpark.** Lo spec-dec `--mtp` di ds4 = l'"MTP-1"
> che il paper usa come production baseline e batte del 60-85%. Il drafter ricorsivo MTP è
> AUTOREGRESSIVO puro (T_draft ∝ profondità), non il semi-AR addestrato di DSpark. La mappa
> resta il fondamento della **Strada A**: tenere questo drafter, sostituire N/margine fissi
> con la verifica confidence-scheduled.

## TL;DR (3 fatti che decidono il track)

1. **ds4 ha GIÀ lo speculative decoding del baseline MTP-1 completo**: drafter ricorsivo,
   tre verifier (batch layer-major, exact N=2, sequential fallback), gate di margine,
   strumentazione di acceptance. Non c'è niente da riscrivere: `ds4_session_eval_speculative_argmax`
   è a `ds4.c:27167`. (È il BASELINE del paper, non DSpark — vedi inquadramento sopra.)
2. **MA è esplicitamente bloccato nel regime streaming**: `ds4.c:25685`
   `"ds4: --ssd-streaming is not compatible with --mtp yet"` → engine si rifiuta di partire.
   Il 49% di risparmio IO che motiva questo track vive ESATTAMENTE nella combinazione vietata.
3. **La union-load per blocco esiste già ed è streaming-only**: il path batch CUDA
   (`ds4.c:14336` + `ds4_cuda.cu:3176`) deduplica gli expert selected di tutte le righe del
   blocco e fa UNA compact-load per layer. Il verifier MTP usa lo stesso
   `metal_graph_encode_layer_batch`, quindi **erediterebbe la union-load gratis** se il blocco
   streaming+MTP venisse rimosso. Il gap non è "implementare l'union-load": è "far convivere
   MTP con lo stream mapper".

## 1. Mappa di cosa esiste (file:riga @ 80ebbc3)

### 1.1 CLI / server / opzioni
| Cosa | Dove | Note |
|---|---|---|
| `--mtp <path>` | `ds4_cli.c:1459`, `ds4_server.c:11557` | carica il GGUF MTP come support model |
| `--mtp-draft <n>` | `ds4_cli.c:1461`, `ds4_server.c:11559` | default **1** (`ds4_cli.c:1397`), cap **16** (`ds4.c:25562`) |
| `--mtp-margin <f>` | `ds4_cli.c:1463`, `ds4_server.c:11561` | default **3.0** (`ds4_cli.c:1398`) |
| opzioni engine | `ds4.h:96,100,101` | `mtp_path`, `mtp_draft_tokens`, `mtp_margin` |
| API pubblica | `ds4.h:265,276,277` | `ds4_session_eval_speculative_argmax`, `ds4_engine_has_mtp`, `ds4_engine_mtp_draft_tokens` |
| trigger nel loop CLI | `ds4_cli.c:483-489` (generate), `ds4_cli.c:1154-1160` (chat), `ds4_server.c:10418-10424` | spec-dec usato **solo se** `temperature <= 0` (greedy) **e** `mtp_draft_tokens > 1` **e** `DS4_MTP_SPEC_DISABLE` assente |

### 1.2 Caricamento modello MTP
- `ds4.c:25683-25696`: se `mtp_path` è settato → **guardia streaming a `25685`** (die se
  `e->ssd_streaming`), poi `model_open` del GGUF MTP + `mtp_weights_bind` + `mtp_ready=true`.
- `ds4.c:3074`: struct `ds4_mtp_weights`; `ds4.c:3642` validate layout.
- `ds4.c:4436-4474` `mtp_weights_bind`: tensori `mtp.0.*`. **Il blocco MTP ha un proprio layer
  MoE completo**: router (`ffn_gate_inp`, riga 4465), bias probs (4466), **256 expert routed**
  (`ffn_gate_exps/up/down`, 4467-4469) + shared expert (4470-4472), più attention MLA e teste
  hyper-connection. È un mini-transformer DeepSeek, non una testa lineare.
- File su disco: `models\ds4\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (3,807,602,400 byte).

### 1.3 Stato GPU dedicato (allocato solo con `--mtp`)
- `ds4.c:10344-10423`: scratch speculativo nel graph; il drafter ha una **raw SWA cache
  propria** (`mtp_raw_cache`, 10422) perché scrive righe "future" speculative; il KV target
  è aggiornato solo dopo verifica (commento 10410-10412).
- `ds4.c:11151-11170`: allocazioni dentro `if (enable_mtp)` — MTP è "deliberately outside the
  normal graph footprint" (11151): senza `--mtp` il decode è bit-identico al baseline.
  `spec_logits` = **16 righe** di vocab (11169) → il verifier batch supporta blocchi fino a 16.
- `ds4.c:26075`: `metal_graph_init(..., e->mtp_ready)`; `ds4.c:26094-26101` init stato sessione.

### 1.4 Drafting
- Draft ancora (MTP-1, 1 token per ciclo): dentro `ds4_session_eval_internal`,
  `ds4.c:27109-27111` (`mtp_should_draft` = `mtp_draft_tokens > 1 || DS4_MTP_PROBE`) e
  `ds4.c:27135-27150`: dopo ogni token committato, un forward MTP
  (`metal_graph_eval_mtp_draft`, def. `ds4.c:20039`) produce `mtp_draft_token`.
- Draft ricorsivo (profondità >1): `ds4.c:27264-27287`, dentro la state machine: catena di
  `metal_graph_eval_mtp_draft_from_hc` (def. `ds4.c:19924`) con ping-pong
  `mtp_state_hc`/`mtp_next_hc`, avanzando la frontiera della raw-cache MTP.
- Il drafter è **greedy/top-1** (argmax, 27146/27282); logits completi del draft solo con
  `DS4_MTP_FULL_LOGITS` o per il margin gate.

### 1.5 State machine speculativa (`ds4.c:27160-27745`)
Struttura (commento di testa 27160-27166):
1. commit del token target normale; i suoi logits validano `draft[0]` **gratis**
   (`ds4.c:27244-27249`: se argmax target ≠ draft[0] → niente suffisso, esci con 1 token).
2. draft ricorsivo del suffisso (27264-27287).
3. **Margin gate** (27295-27341): solo per `draft_n == 2` e `!strict`: calcola
   `margin = logit_top1 - logit_top2` dell'ULTIMO draft MTP (`logits_top2`, 27297-27301);
   se `margin < soglia` (default 3.0, override `DS4_MTP_MIN_MARGIN`) → **salta il verifier**,
   decode normale di draft[0] e commit di 1 token. È una proto-confidence: scalare, non
   calibrata, un solo cut-off, senza nozione di carico.
4. Verifica, tre percorsi:
   - **Exact N=2 decode verifier** (`metal_graph_verify_decode2_exact`, def. `ds4.c:21213`):
     solo con `--quality`/`DS4_MTP_STRICT` (selettore a 27346-27347). Kernel decode esatti,
     2 token in un command stream. Per-token, non batch.
   - **Batch layer-major verifier** (default): `metal_graph_verify_suffix_tops`, def.
     `ds4.c:21107-21200`, chiamato a 27467 — tutto il suffisso (fino a 16 righe) passa nei
     kernel batch (`metal_graph_encode_layer_batch`, 21147) in un solo pass layer-major;
     legge top-1 per riga (argmax kernel per K=2 a 21170, topk per righe multiple a 21178) e
     UNA riga di logits per lo stato di continuazione. Accept = match greedy per prefisso
     (27479-27484). Partial-accept: prefix-1 capture (commento `ds4.c:13166`,
     commit `ds4.c:24129`, uso 27448-27449, 27556+) o snapshot/replay della frontiera.
   - **Sequential fallback** (27690-27745): decode uno-per-token esatto, "deliberately slow",
     solo se il micro-verifier fallisce.

### 1.6 Strumentazione acceptance già in upstream (chiave per lo step 2)
| Env | Dove | Cosa misura |
|---|---|---|
| `DS4_MTP_PROBE` | `ds4.c:27108,27112-27123` | per OGNI token committato confronta il draft del ciclo precedente col token reale: stampa `mtp probe token=.. draft=.. hit=H/T`. **È l'acceptance top-1 di MTP-1 misurata sul flusso greedy reale, senza commit speculativi.** Funziona anche con `--mtp-draft 1`. |
| `DS4_MTP_CONF_LOG` | `ds4.c:27229,27288-27292,27485-27494` | per ciclo: `drafted=.. committed=.. margin=..` → acceptance per ciclo + margine del draft |
| `DS4_MTP_SPEC_LOG` | `ds4.c:27245,27427,27680,27697` | miss/fallback events |
| `DS4_MTP_TIMING` | `ds4.c:27228` + stampe 27325-27339, 27538-27546, ecc. | draft/snapshot/verify/replay in ms per ciclo |
| Altri: `DS4_MTP_SPEC_DISABLE`, `DS4_MTP_STRICT`, `DS4_MTP_MIN_MARGIN`, `DS4_MTP_FULL_LOGITS`, `DS4_MTP_BATCH_VERIFY`, `DS4_MTP_CAPTURE_PREFIX1`, `DS4_MTP_EXACT_REPLAY`, `DS4_MTP_FORCE_SNAPSHOT` | `ds4_cli.c:484`, `ds4.c:27220-27231,27347,27448-27453` | toggle di percorso/debug |

### 1.7 Union-load degli expert nel path batch (risposta allo step 4 del brief)
**Sì, ds4 fa union-load per blocco — ma solo nel path batch streaming (oggi: prefill).**
Catena verificata:
1. `metal_graph_encode_layer_ffn_batch` (def. `ds4.c:18780`): router batch su GPU per tutte
   le righe (`ds4_gpu_router_select_batch_tensor`, 18908-18925 → `batch_router_selected`,
   `n_tokens × 6` id).
2. Subito dopo (18938): `metal_graph_cuda_stream_prefill_batch_selected_load`
   (def. `ds4.c:14336-14420`): sync, readback degli id selected di TUTTE le righe,
   e chiamata a...
3. `ds4_gpu_stream_expert_cache_prepare_selected_batch` (`ds4_cuda.cu:3176-3245`):
   costruisce `compact_ids` = **insieme deduplicato** degli expert su tutte le
   `n_tokens × n_selected` slot e lancia UNA `cuda_stream_selected_cache_begin_compact_load`
   col solo insieme unico. Questo è esattamente il meccanismo che monetizza il 49%
   (24.3 expert unici/layer per blocco-8 vs 48, misurato in `runs/ds4_routing_trace_smoke/`).
4. Gating (`metal_graph_decode_cuda_selected_slots_expected`, def. `ds4.c:13774-13800`):
   richiede `ssd_streaming && !quality && DS4_N_EXPERT_USED==6 && DS4_N_EXPERT>=128` e quant
   Q4_K o IQ2 sui tre tensori expert; più `n_tokens > 1` e assenza di
   `DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD` (14345-14352).

**Il punto architetturale**: `metal_graph_verify_suffix_tops` (21107) usa lo stesso
`encode_layer_batch` del prefill → se girasse in streaming, la union-load scatterebbe già.
Ma non può girarci, per due ragioni oggi:
- la guardia hard a `ds4.c:25685` (engine rifiuta `--mtp` + `--ssd-streaming`);
- il verifier non pilota lo stream mapper: il prefill streaming avvolge il loop layer con
  `metal_graph_stream_map_layer[_decode]` + prepare/readahead (path split_commands,
  `ds4.c:20540-20704`), mentre `verify_suffix_tops` chiama `encode_layer_batch` nudo
  (21146-21153) — in streaming i pesi del layer non sarebbero mappati.

### 1.8 Cosa NON c'è (input per la gap-analysis, step 3)
Rispetto a DSpark (`docs/references/DSpark_paper.txt`):
- **Niente confidence head** (Eq. 7 del paper): l'unico segnale è il margine top1-top2
  dell'ultimo draft (27295-27341), scalare, non calibrato, usato solo per draft_n==2.
- **Niente STS** (calibrazione sequenziale min-ECE, §3.2.1): nessuna nozione di probabilità
  di sopravvivenza del prefisso.
- **Niente scheduler** (Alg. 1): la profondità di draft è FISSA (`--mtp-draft`), non adattiva
  per token/carico/dominio. Il commento a 27441 dice esplicitamente "The production MTP depth
  is two".
- **Drafter diverso dal paper**: ds4 usa l'MTP nativo di DeepSeek-V4 (autoregressivo, stile
  MTP-1 ricorsivo), non il backbone parallelo DFlash+testa Markov. La confidence/STS/scheduler
  di DSpark sono però agnostici rispetto al drafter: si innestano sul segnale di confidenza
  del draft, qualunque esso sia.
- **Spec-dec solo greedy**: niente rejection sampling a temperatura >0 (gate a
  `ds4_cli.c:483`); accept = match argmax esatto.
- **MTP + streaming vietato** (25685) — il gap che vale il 49%.

## 2. Conseguenze operative
1. **Step 2 (misura acceptance)** si fa su pod 3090 **senza** `--ssd-streaming` (lì il
   modello sta in RAM/page-cache): `--mtp ... --mtp-draft 2 -n fisso --nothink` greedy con
   `DS4_MTP_PROBE=1` (acceptance MTP-1 per token) e `DS4_MTP_CONF_LOG=1`
   (drafted/committed/margine per ciclo). L'acceptance è proprietà del modello → trasferisce
   al 3060. I t/s del pod NON trasferiscono (RAM vs SSD-streaming) e non vanno presentati
   come velocità 3060.
2. **Step 3 (design)**: i punti d'innesto naturali sono (a) sostituire il margin gate
   27295-27341 con confidence calibrata (STS offline, macchina già in
   `src/msc/spex/spex_loop.py`), (b) rendere `draft_cap` (27209-27214) dinamico con Alg. 1
   semplificato a R=1 (siamo single-request), (c) togliere la guardia 25685 e insegnare a
   `verify_suffix_tops` a pilotare lo stream mapper come fa il prefill split_commands.
3. **Il valore streaming** non è "più t/s dal verifier" ma **meno byte da SSD per token
   accettato**: con blocco-k la union-load carica ~24.3 expert unici/layer invece di 6·k.
