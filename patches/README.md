# patches/ — serie canonica ds4 (REAP-LOOP)

**La serie canonica delle patch runtime ds4 è questa directory (`patches/ds4/`).**
Le copie in `moe-aggressive-commit/patches/ds4/` sono snapshot storici o serie parallele di altre
arene: non sono la fonte di verità. Inventario verificato il 2026-07-10; hash = `git hash-object`,
primi 8 caratteri.

## Mappa per numero

| N. | File (qui, salvo nota) | Hash | Altre copie | Stato |
|---|---|---|---|---|
| 0001 | 0001-spex-stage0-cuda-stats.patch | 4300266a | moe main/k91/dspark (identica) | canonica |
| 0002 | 0002-spex-selected-upload-event.patch | 8cd334d3 | moe main/k91/dspark (identica) | canonica |
| 0003 | 0003-spex-stage1-next-layer-prefetch.patch | 7463a782 | moe main/k91/dspark (identica) | canonica |
| 0004 | 0004-spex-markov-loader-prefetch.patch | bb099112 | moe main/k91/dspark (identica) | canonica |
| 0005 | 0005-spex-routing-trace-capture.patch | b23d655e | moe main/k91/dspark (identica) | canonica |
| 0006 | 0006-spex-routing-trace-weights.patch | be4a17a3 | moe main/k91 (assente sul branch dspark) | canonica |
| 0007 | 0007-spex-trace-hidden.patch | c17c27e1 | moe main/k91 (assente sul branch dspark) | canonica |
| 0008 | 0008-dspark-mtp-streaming-probe-unsafe.patch | 4cf57001 | moe main/k91/dspark (identica) | canonica |
| 0009 | 0009-dspark-mtp-streaming-unlock.patch | b69b9738 | moe: revisione **divergente** 57ce10ce | canonica qui; vedi regola 1 |
| 0010 | 0010-dspark-cuda-support-model-map-registration.patch | f741a526 | moe: revisione **divergente** d135f38e | canonica qui; vedi regola 1 |
| 0011 | 0011-reap-runtime-mask.patch | c57bee11 | moe k91 (identica) | canonica; **collisione** con `0011-dspark-mtp-host-statics.patch` (b9c58ada, solo moe branch `dspark/mtp-spec-dec`) — vedi regola 2 |
| 0012 | 0012-reap-sensor-s1.patch | f5e2ca3a | moe k91 (identica) | canonica; **collisione** con `0012-dspark-mtp-no-register-diagnostic.patch` (892dbb6e, solo moe branch `dspark/mtp-spec-dec`) — vedi regola 2 |
| 0013 | 0013-reap-wrap.patch | 70f0e2b6 | solo reap-loop | canonica |
| 0014 | 0014-pace-controller.patch | 7939a305 | solo reap-loop | canonica |
| 0014e | 0014e-pace-wrap-rename.patch | 75d2ceae | solo reap-loop | canonica (follow-up di 0014: rename WRAP dentro la guardia 0013) |
| 0015 | 0015-pace-raw-router-k-rotation.patch | e5943b2b | solo reap-loop | canonica, ramo PACE; **collisione a tre** — vedi regola 3 |
| 0015 | 0015-spex-hidden-async-topk-handoff.patch | ef4d9a38 | solo reap-loop | canonica, ramo SPEX-hidden; **collisione a tre** — vedi regola 3 |
| 0016 | 0016-pace-rebuild-on-tighten.patch | 73cfa4ab | solo reap-loop | canonica, ramo PACE (si applica DOPO 0015-pace); **collisione** — vedi regola 3 |
| 0016 | 0016-spex-hidden-gpu-prefetch-stats.patch | 86b4125d | solo reap-loop | canonica, ramo SPEX-hidden; **collisione** — vedi regola 3 |
| 0017 | 0017-spex-routing-trace-residency.patch | b1ce60e4 | solo reap-loop | **proposta, NON applicata** al sorgente live — vedi regola 4 |
| 0018 | 0018-pace-skip-wrap-on-rotate.patch | 5b018a3b | solo reap-loop | applicata al sorgente live ma non committata (verifica simboli 2026-07-10: `DS4_PACE_WRAP_ROTATE` e `wrap_rotate` presenti in `/root/ds4/ds4.c`, che risulta modificato non committato nel repo /root/ds4) |
| 0019 | — | — | — | **numero riservato** allo stopper anti-ripetizione (mandato Codex M1b) |
| 0020 | 0020-pace-s1-slope-trigger.patch | 089da7cd | solo reap-loop | **compiled + mechanism-smoked on pod 2026-07-10 (s1_trigger fires + rotate(s1), slope numeric non-NaN, s1 0.727→0.814); pending canonization** — S1-slope trigger (leva L2, `DS4_PACE_S1_TRIGGER`); ancorata allo snapshot live 2026-07-10 di `/root/ds4/ds4.c` (post-0018, md5 771a39a8), NON alla serie canonica — vedi "Stato apply". Smoke: `runs/ds4/20260710_pod_smoke_0020_0021/` |
| 0021 | 0021-pace-rotate-delta-prefetch.patch | 7fb78678 | solo reap-loop | **compiled + mechanism-smoked on pod 2026-07-10 (rotate_delta pages only entered experts, entered==exited, 6.75 MiB/expert, no full WRAP on decode); pending canonization** — delta-prefetch su rotate (leva L3, `DS4_PACE_WRAP_ROTATE_DELTA`); si applica DOPO la 0020 (hunk struct dipendente), stessa base live-tree. Smoke: `runs/ds4/20260710_pod_smoke_0020_0021/` |
| 0026 | — | — | — | **numero riservato/candidata** — demand-driven admission (E-ADMIT): CUSUM per-expert sulla domanda bloccata fuori-mask → ammissione con sfratto del keep a EWMA minima, K costante, mai K0, mai re-rank wholesale; riusa rmass per-expert 0020 + delta-prefetch 0021; `DS4_PACE_ADMIT=0` default (h=1.2, k_d=0.02, p=2, cooldown 16). Evidenza offline (sim su traiettoria sana = potenziale copertura, NON qualità): `runs/ds4/20260710_eadmit_demand_admission/REPORT.md`, §"Spec candidata patch 0026". Patch solo dopo A/B live S3. |
| — | ds4_spex_predict.c / ds4_spex_predict.h | 396a9331 / 2ec4f88e | moe main/k91 (identici) | supporto (loader probe `.spex`) |
| — | upstream-pr497-single-token-selected-load.diff | 51dd423f | moe main/k91/dspark (identica) | riferimento upstream |

("moe main" = branch `main` e `cascade-memory/harness` di moe-aggressive-commit, identici su `patches/`;
"k91" = branch `reap/k91-coding-vram`; "dspark" = branch `dspark/mtp-spec-dec`.)

## Regole

1. **Mai applicare 0009/0010 dalla copia moe** (57ce10ce / d135f38e): gli hunk sono identici alle
   canoniche ma i file moe sono CRLF (63 byte CR su 0009, 180 su 0010) con author
   `dspark-track <imanu86@gmail.com>`; le canoniche qui sono LF con author neutro
   (b69b9738 / f741a526, 2920 / 8057 byte). Il CRLF rischia di far fallire (o sporcare) `git apply`
   sul sorgente LF, e l'email personale non deve entrare nella history di ds4.
2. **0011/0012 sul branch moe `dspark/mtp-spec-dec` sono un'altra serie**:
   `0011-dspark-mtp-host-statics` e `0012-dspark-mtp-no-register-diagnostic` riguardano l'arena
   MTP/spec-dec, non il runtime REAP (`0011-reap-runtime-mask` / `0012-reap-sensor-s1`).
   Stessa numerazione, patch diverse: si applicano solo nell'arena dspark, mai mischiarle.
3. **Collisione 0015/0016 — tre serie con gli stessi numeri**: (a) ramo PACE
   `0015-pace-raw-router-k-rotation` + `0016-pace-rebuild-on-tighten` (la 0016 si applica dopo la
   0015-pace, vedi `docs/EXPERIMENTS_LEDGER.md` nota J50); (b) ramo SPEX-hidden
   `0015-spex-hidden-async-topk-handoff` + `0016-spex-hidden-gpu-prefetch-stats` (vedi
   `docs/SPEX_INTEGRATION_PLAN.md`, aggiornamento J30); (c) sul branch moe `reap/k91-coding-vram`
   esiste anche `0015-spex-hidden-probe-wiring.apply.py` (924007a4): è uno script Python di wiring
   per il ds4.c del worktree k91, non una patch della serie. Le prossime patch NON riusano numeri
   già presi: si riparte da 0019.
4. **0017 è una proposta non applicata**: `docs/DS4_ROUTING_RESIDENCY_TRACE.md` la dichiara "Patch
   proposal", nessun run in `runs/` usa `DS4_SPEX_TRACE_ROUTING_RESIDENCY` e il sorgente live
   (`/root/ds4/ds4.c`, verifica simboli 2026-07-10) non contiene il simbolo.
5. `moe:patches/ds4/archive/2026-07-04-stall-instrumentation-uncommitted.patch` (ad71e31b) è
   materiale storico dell'arena moe: resta in moe e non entra nella serie canonica.

## Stato apply (live-tree vs serie canonica) — input per la canonizzazione

Verifica 2026-07-10 (deploy pod T1 + lettura diretta di `/root/ds4/ds4.c`, snapshot
md5 `771a39a8`, 1314218 byte): le patch pace **0015/0016/0018 come archiviate NON
applicano su base pulita 0001-0014e** — i loro hunk assumono campi/funzioni che
esistono solo nel live-tree locale e non sono mai stati formalizzati in patch.
Per questo 0020/0021 sono ancorate al live-tree (base = snapshot nell'header di
ciascuna patch) e andranno ri-basate dopo la canonizzazione. Elenco preciso di
ciò che il live-tree ha in più rispetto alla serie canonica:

- **Campi struct `g_pace` solo live**: `prebreath_on`, `prebreath_drift`,
  `prebreath_target`, `prebreath_every`, `prebreath_keep_max`, `prebreath_adapt`,
  `prebreath_adapt_gain`, `prebreath_adapt_power`, `prebreath_step_max`,
  `prebreath_relearn`, `prebreath_relearn_decay`, `cache_flush`, `prefill_apply`,
  `prefill_wait_wrap`, `exchange_observe`, `weighted_warmup`, `weighted_relearn`,
  `weighted_read_fail`, `in_prefill`, `last_prebreath_tok`, `exchange_events`,
  `exchange_promote`, `exchange_demote`.
- **Env solo live**: `DS4_PACE_PREBREATH{,_DRIFT,_TARGET,_EVERY,_KEEP_MAX,_ADAPT,
  _ADAPT_GAIN,_ADAPT_POWER,_STEP_MAX,_RELEARN,_RELEARN_DECAY}`,
  `DS4_PACE_CACHE_FLUSH`, `DS4_PACE_PREFILL_APPLY`, `DS4_PACE_PREFILL_WAIT_WRAP`,
  `DS4_PACE_EXCHANGE_OBSERVE`, `DS4_PACE_WEIGHTED_SELECTED/_WARMUP/_RELEARN`.
- **Funzioni solo live**: `ds4_pace_note_selected_batch`,
  `ds4_pace_wants_selected_weights`, `ds4_pace_flush_expert_cache`,
  `ds4_pace_exchange_observe`, `ds4_pace_reset_for_prefill`,
  `ds4_pace_apply_prefill_mask`, `ds4_reap_prefetch_wait`. Firme divergenti:
  `ds4_pace_note_selected` ha il parametro `selected_weights`;
  `ds4_pace_apply_keep`/`_acc` hanno il parametro `why`.
- **WRAP live ≠ 0013 canonica**: env thread/lock live = `DS4_REAP_PREFETCH_THREADS`
  / `DS4_REAP_PREFETCH_LOCK` (non `DS4_REAP_WRAP_*`), banner stderr "fattorino"
  (non "WRAP"), e il live-tree NON ha il fix race "pending re-run" della 0013
  canonica (busy → "gia' in corso, salto", senza recupero della mask arrivata
  durante il pass).

## Stato apply — deploy T1 pod (2026-07-10)

Il deploy T1 su base **pulita e pinnata `80ebbc3`** (pod RunPod, build da sorgente) ha
dimostrato che **la serie NON applica pulita end-to-end**:

- **Applicati clean solo `0001-0008` e `0011-0014e`** (dopo aver strippato il CRLF raccolto
  dal checkout Windows — esattamente la trappola della regola 1).
- **`0009/0010` falliscono sulla base pulita** (context mismatch; sono dspark-MTP, inutili in
  T1 → saltati).
- **`0015/0016-pace + 0018` richiedono contesto live-tree NON canonizzato**: dipendono da
  campi struct (`prefill_apply`, `prefill_wait_wrap`) che esistono solo nel live-tree locale
  non committato, in nessuna patch canonica ⇒ la rotazione (rotate32) **non è disponibile nel
  binario pod** costruito dalla serie.
- **TODO bloccante: canonizzare la serie pace** (`0015/0016-pace`, `0018`, con i campi struct
  che oggi vivono solo nel live-tree) **prima di qualunque smoke `0020/0021` su pod** — altrimenti
  le leve L2/L3 non si possono buildare dalla serie canonica.

Fonte: `runs/ds4/20260710_pod_t1_full_positive_control/README.md` (sezione gap / Runtime).
