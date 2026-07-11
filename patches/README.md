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
| 0026 | 0026-pace-demand-admission.patch | d933dec1 | solo reap-loop | **authored 2026-07-10 vs live-tree+0020+0021, `git apply --check` OK sulla catena 0020→0021→0026 su base pulita (md5 771a39a8, byte-identica al tree autorato), pending smoke** — demand-driven admission (E-ADMIT, leva ammissione mirata): CUSUM per-expert sulla domanda bloccata fuori-mask → ammissione con sfratto del keep a EWMA minima (rmass per-expert 0020, decay = `DS4_PACE_ROTATE_DECAY` 0.98), K costante, mai K0, mai re-rank wholesale; page-in del solo expert entrato via delta-prefetch 0021 (~6.75 MiB); cooldown anti-thrash + rate-cap opzionale. `DS4_PACE_ADMIT=0` default (`_H`=1.2, `_KDRIFT`=0.02, `_PERSIST`=2, `_COOLDOWN`=16, `_MAX_PER_100`=0). Ancorata allo snapshot live 2026-07-10 (post-0018, md5 771a39a8), applica DOPO 0020+0021; rebase dopo canonizzazione serie pace — vedi "Stato apply". Evidenza offline (sim su traiettoria sana = potenziale copertura, NON qualità; verdetto positivo): `runs/ds4/20260710_eadmit_demand_admission/REPORT.md`. Gate qualità: A/B live S3. |
| 0027 | 0027-rewind-exactness-harness.patch | 1734216f | solo reap-loop | **authored 2026-07-10 vs live-tree+0020+0021+0026, `git apply --check` OK sulla catena completa 0020→0021→0026→0027 su base pulita (md5 771a39a8, blob finale a29b30f), pending GPU smoke** — harness DIAGNOSTICA di ESATTEZZA rewind (gate R1 per 0022, `docs/S1_REWIND_DESIGN.md` §5/R1). `DS4_REWIND_TEST="p,k"`: a p fa `spec_frontier_snapshot` + checkpoint stato PACE (ring n-gram, EWMA/CUSUM, `mass/rmass/bmass`, clock/`tok` = "sampler pos"); a p+k fa `spec_frontier_restore` + restore PACE + `ds4_session_rewind(p)` + replay dei k input (MTP-off, greedy) e logga i due stream di token-id (pre/post) su JSONL (`DS4_REWIND_TEST_LOG`, else stderr) per il verdetto bit-exact di `scripts/verify_rewind_exactness.py`. NON copre temp>0/RNG, MTP-on speculativo, maschera dinamica (girare mask statica, es. W50). Richiede build MTP-enabled (riusa i buffer `spec_*`); no-op se `DS4_REWIND_TEST` non è settato, fail-safe se mancano i buffer frontier. Ancorata allo snapshot live 2026-07-10 (post-0018, md5 771a39a8), applica DOPO 0020+0021+0026; rebase dopo canonizzazione serie pace. |
| 0028 | 0028-spex-trace-tokens.patch | b94be032 | solo reap-loop | **rinumerata da 0019** (contributo sessione Scope UX, non era mai stata registrata in tabella); il numero 0019 resta **riservato** allo stopper anti-ripetizione (mandato Codex M1b, vedi riga 0019) — vedi regola 6. Sidecar testo-token per lo Scope: `DS4_SPEX_TRACE_TOKENS=/path.csv` scrive una riga `pos,token_id,piece` per token generato (stessa chiave `pos` di `DS4_SPEX_TRACE_ROUTING`, cosi' un viewer unisce testo e routing senza tokenizer esterni); 3 hunk su `ds4.c` (funzione `trace_token_sidecar` + 2 call-site in `generate_raw_swa_cpu`/`generate_metal_graph_raw_swa`), stesso pattern lazy-open di 0012. Ancorata allo snapshot live 2026-07-10 (post-0018, md5 771a39a8), applica DOPO 0020+0021+0026+0027. **Apply-check reale 2026-07-10 (repo git temporanei, non il repo reap-loop) — esito in due tempi.** (1) Il file *su disco come autorato dalla sessione Scope UX* aveva terminatori CRLF (`ds4.c` e le patch sorelle 0020/0021/0026/0027 sono LF — trappola CRLF della regola 1): `git apply` di quel file grezzo FALLISCE su entrambe le catene, 3/3 hunk rejected per mismatch di riga (live-tree base 771a39a8→0020→0021→0026→0027 ok→0028 rifiutata; canonical v2 base 80ebbc3→19 patch→0027 ok→0028 rifiutata). (2) Questo repo ha `core.autocrlf=true`: `git add` normalizza CRLF→LF nell'indice automaticamente, quindi il blob **committato** (hash `b94be032f10f00b1779ce72a30dc4d2a1c32864b`) è LF-puro, non il file grezzo del punto (1). Verificato per davvero (non solo per hash): il blob committato estratto via `git cat-file -p :patches/ds4/0028-spex-trace-tokens.patch` **applica pulito su entrambe le catene**, 3/3 hunk, solo offset (+3262/+3262/+3264 righe), zero fuzz, `ds4.c` finale md5 `62ed2e71` identico su entrambe — l'ancoraggio/contesto è corretto. **Verdetto: adottabile così com'è nella history git** (il fallimento del punto 1 era solo il file grezzo pre-`git add`, mai quello che finisce in git); pending pod smoke. |
| 0022 | 0022-pace-s1-rewind.patch | 24a1f306 | solo reap-loop | **authored 2026-07-11 vs la catena canonical v2 + 0027 + 0028, `git apply --check` OK sulla catena COMPLETA (base `github.com/antirez/ds4`@80ebbc3 + `patches/ds4/canonical/` 21 patch → ds4.c md5 `1db4f799`, + 0027 + 0028 → md5 `62ed2e71` = target live-tree pace0028; + 0022 → md5 `044b32ce`), 388 righe inserite in 9 hunk su `ds4.c`, zero fuzz, LF-clean; ricostruzione del target verificata byte-identica al pod2 `canonical_build_v2` (checkpoint 1db4f799/62ed2e71). Bilanciamento graffe/parentesi OK; gcc non disponibile in locale (CPU-only) → no `-fsyntax-only`, cross-check simboli manuale. Pending pod smoke.** **v2 (2026-07-11, fix chirurgico, pending re-smoke):** il pod D smoke (`runs/ds4/20260711_podD_smoke_0022/VERDICT.md`, post-0022 md5 reale `a7aab041`) trovò `ds4_pace_rewind_snapshot_frontier` strutturalmente incapace di sparare — la guardia apriva con `if (DS4_N_LAYER > 0 && !g->spec_rewind_attn_state_kv[0]) return false;`, ma su Flash i layer 0-1 sono DENSI (`ds4_expected_layer_compress_ratio`==0 per `il<2`) e i buffer `spec_rewind_*` sono allocati solo `if (ratio != 0)` (`ds4.c:11326`) ⇒ `[0]` è sempre NULL ⇒ lo snapshot dell'onset non riusciva mai ⇒ `onset_valid` mai settato ⇒ FIRE sempre bloccato su `!g_rewind.onset_valid`. Il twin MTP `spec_frontier_snapshot` non ha guardia analoga: itera e salta i layer densi (`if (ratio==0) continue;`) dentro il loop. **Fix:** la guardia ora scansiona `il=0..DS4_N_LAYER-1` e cerca il primo `spec_rewind_attn_state_kv[il]` allocato (return false solo se nessuno lo è); `snapshot`/`restore` già iteravano saltando i densi, solo la precondizione era sbagliata; `restore` non aveva guardia analoga ed era già corretto. Patch rigenerata via repo temporaneo (`github.com/antirez/ds4`@80ebbc3 + 21 canonical + 0027 + 0028 → md5 `62ed2e71` → 0022-v1 → fix → `git format-patch`), non un edit in-place del blob v1. `git apply --check` OK sulla catena REALE completa in repo temporaneo separato: base `62ed2e71` + 0022-v2 → md5 **`a88f9dcb`**, 398 righe/10 hunk (era 388/9), zero fuzz, LF-clean.** — l'ATTUATORE di produzione del rewind S1-guidato (scala prevenzione→correzione→airbag, `docs/S1_REWIND_DESIGN.md`), rimpiazza l'harness diagnostico 0027. Chiude le due lezioni del pod2 R1: **(a)** l'hook non deve stare solo in `ds4_session_eval_internal` (muto sul path greedy CLI `generate_metal_graph_raw_swa`) → l'attuatore gira in **ENTRAMBI i loop di decode** (greedy CLI GPU + session/server); **(b)** i buffer `spec_*` sono MTP-gated → 0022 alloca i **PROPRI** `spec_rewind_*` (gate `DS4_PACE_REWIND`, indipendenti da MTP, mai clobbered dallo spec-verifier). Detector E-DET aggregate EWMA-CUSUM (α=0.50, σ auto-cal 128 tok, baseline lag32/win128; ARM k0.5/h4, FIRE k1.0/h8) su S1 aggregato PRE-bias (stesso readback di S1/0012/0026, FEED_AUDIT). ARM = checkpoint rolling (frontier `spec_rewind_*` + logits onset + blob PACE completo mass/rmass/bmass/admit + ring/clock) rinfrescato ogni EVERY tok finché sano, congelato all'ARM; FIRE (o n-gram airbag, gated su ARM confermato → regime aggressivo cade sull'airbag STOP) → restore frontier + accumulatori PACE + relearn/widen maschera a keep_max + resume da onset+1 (no re-prefill). Anti-oscillazione REWIND_MAX=2/backoff×2; JSONL `rewind`/`rewind_arm`/`rewind_skip {from,to,reason}` per il client-trim; contatore token rigenerati. `DS4_PACE_REWIND=0` default; scope regime LENTO (K91-family) — MARGIN realizzato dal freeze del checkpoint rolling all'ARM (floor = pos snapshot, no rewind sotto lo snapshot). Applica DOPO 0028; rebase dopo ulteriore canonizzazione pace. |
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
6. **0019 → 0028, rinumerazione**: il contributo della sessione Scope UX (sidecar testo-token,
   `DS4_SPEX_TRACE_TOKENS`) era stato salvato come `0019-spex-trace-tokens.patch` ma **0019 era
   già riservato** allo stopper anti-ripetizione (mandato Codex M1b — vedi riga 0019, "numero
   riservato") e non era mai stato registrato in questa tabella: file non committato, numero
   collidente. Rinominato (non copiato) in `0028-spex-trace-tokens.patch`, primo numero libero
   dopo la serie 0020-0027; corretto solo il commento `patch 0019` → `patch 0028` dentro il diff
   (riga aggiunta dal patch stesso), header `index`/`@@` invariati. 0019 resta riservato allo
   stopper M1b: non riusarlo.

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

## Serie canonica v2 (post-canonizzazione) — `patches/ds4/canonical/`

**Chiude il TODO bloccante di S0** (`docs/SOTA_ROADMAP.md` §S0 "Canonizzazione serie
pace"). La dir `patches/ds4/canonical/` è una catena **auto-contenuta e LF-pulita** che
**applica pulita end-to-end da base `80ebbc3` fino a 0026**, ricostruendo `ds4.c`
**byte-identico** al live-tree (md5 `771a39a8`) su cui 0020/0021/0026 sono ancorate.
Gli originali in `patches/ds4/*.patch` **NON sono toccati** (un pod agent li usa).
`.gitattributes` locale marca `*.patch -text` così la serie resta LF su qualunque
checkout (niente trappola CRLF della regola 1).

**Verifica reale (2026-07-10, CPU-only, `git apply --check` + apply su repo git
temporaneo da 80ebbc3):** 19/19 patch applicano clean nell'ordine sotto; `ds4.c` finale
md5 `1db4f799` == live-tree+0020+0021+0026. Stato intermedio dopo `0018-pace-canonical`:
`ds4.c` md5 `771a39a8` (= live-tree, l'ancora richiesta). Base: `ds4.c` blob `640511eb`
(md5 `bf9a0b6f`). Riproducibile: estrai `git -C <ds4> show 80ebbc3:{ds4.c,ds4_cuda.cu,ds4_gpu.h}`
in un repo pulito e applica in ordine.

### v2.1 — canonizzazione siblings (`ds4_cuda.cu` / `ds4_gpu.h`) — sblocca il build

La v2 originale (19 patch) ricostruiva `ds4.c` byte-identico al live-tree ma lasciava i
**file GPU fratelli al livello fine-0014e** (solo 0001-0008/0011): il build su pod
`80ebbc3` + 19 patch **falliva** con `ds4.c:10614: error: unknown type name
'ds4_gpu_async_read'` (log `runs/ds4/20260710_pod2_smokes/canonical_build/`), più warning
di dichiarazione implicita per `ds4_gpu_async_read_ready`, `ds4_gpu_async_read_host`,
`ds4_gpu_stream_expert_cache_filter_missing`: simboli che `ds4.c` (post-canonical, ramo
SPEX-hidden/tiering) referenzia ma che vivono solo nel `ds4_cuda.cu`/`ds4_gpu.h` **live-tree**
(avanzati oltre 0014e), mai catturati da nessuna patch.

**Delta fratelli fattorizzato** (metodo: base `80ebbc3` → apply canonical 19 → confronto
file-per-file col live-tree WSL `/root/ds4` @ `0bdad9a`): `ds4.c` già byte-identico (0 delta);
restano **solo** `ds4_gpu.h` (+35 righe, 4 hunk) e `ds4_cuda.cu` (+2084 righe nette, 19 hunk).
Fattorizzato in due patch NUOVE, non-distruttive, LF-clean (`-text`), che applicano DOPO 0011
(nessuna patch >0011 tocca i fratelli, quindi lo stato base dei fratelli è identico in ogni
posizione ≥0011) e ricostruiscono i due file **byte-identici al live-tree buildabile**
(`ds4_cuda.cu` md5 `7d57f58d`, `ds4_gpu.h` md5 `55070d97`):

- **`0014f-canonical-siblings-gpu-header.patch`** (`8b280fda`) — dichiarazioni `ds4_gpu.h`:
  `typedef struct ds4_gpu_async_read ds4_gpu_async_read;` + 9 prototipi (`ds4_gpu_async_read_*`
  alloc/free/host/ready, `ds4_gpu_tensor_read_async`, `ds4_gpu_stream_expert_cache_count_resident`/
  `_filter_missing`, `ds4_gpu_matmul_f16_weight_tensor`, `ds4_gpu_spex_hidden_score_tensor`).
- **`0014g-canonical-siblings-cuda-impl.patch`** (`62d866e6`) — definizioni `ds4_cuda.cu`:
  `struct ds4_gpu_async_read {…}` + impl dei prototipi 0014f + substrato GPU SPEX-hidden/tiering
  (controparte CUDA dei sottosistemi che `0015-pace-canonical` canonizza in `ds4.c`).

**Verifica reale (2026-07-10, CPU-only):** catena completa su base fresca `80ebbc3` →
**21/21 canonical** (sorted, incl. 0014f/0014g) + `0027` + `0028` (blob committati LF) applicano
**clean** (`git apply --check` + apply); `ds4.c` finale md5 `62ed2e71`, fratelli finali
byte-identici al live-tree. **Cross-check simboli:** il typedef `ds4_gpu_async_read` e tutti i
simboli del log d'errore risultano **dichiarati** in `ds4_gpu.h` **e definiti** in `ds4_cuda.cu`;
comm dei `ds4_gpu_*` chiamati in `ds4.c` vs dichiarati nell'header → nessun simbolo mancante (i 3
apparenti — `ds4_gpu_graph` typedef locale, `ds4_gpu_rms_norm_weight_rows_tensor` troncato,
`ds4_gpu_wrap_model_range` solo in commento — sono falsi positivi). `ds4_gpu.h` passa
`g++ -fsyntax-only`. **Stato: apply-checked + symbol-checked, build pending pod.**

### Tabella (ordine di apply = ordine tabella; hash = `git hash-object`, 8 char)

| # | ordine | File in `canonical/` | Hash | Origine |
|---|---|---|---|---|
| 1 | 0001 | 0001-spex-stage0-cuda-stats.patch | 4300266a | copia LF, ≡ originale |
| 2 | 0002 | 0002-spex-selected-upload-event.patch | 8cd334d3 | copia LF, ≡ originale |
| 3 | 0003 | 0003-spex-stage1-next-layer-prefetch.patch | 7463a782 | copia LF, ≡ originale |
| 4 | 0004 | 0004-spex-markov-loader-prefetch.patch | bb099112 | copia LF, ≡ originale |
| 5 | 0005 | 0005-spex-routing-trace-capture.patch | b23d655e | copia LF, ≡ originale |
| 6 | 0006 | 0006-spex-routing-trace-weights.patch | be4a17a3 | copia LF, ≡ originale |
| 7 | 0007 | 0007-spex-trace-hidden.patch | c17c27e1 | copia LF, ≡ originale |
| 8 | 0008 | 0008-dspark-mtp-streaming-probe-unsafe.patch | 4cf57001 | copia LF, ≡ originale |
| 9 | 0011 | 0011-reap-runtime-mask.patch | c57bee11 | copia LF, ≡ originale |
| 10 | 0012 | 0012-reap-sensor-s1.patch | f5e2ca3a | copia LF, ≡ originale |
| 11 | 0013 | 0013-reap-wrap.patch | 70f0e2b6 | copia LF, ≡ originale |
| 12 | 0014 | 0014-pace-controller.patch | 7939a305 | copia LF, ≡ originale |
| 13 | 0014e | 0014e-pace-wrap-rename.patch | 75d2ceae | copia LF, ≡ originale |
| 13+ | 0014f | **0014f-canonical-siblings-gpu-header.patch** | 8b280fda | **NUOVA v2.1** — decl `ds4_gpu.h` (sblocca `ds4.c:10614`); vedi §v2.1 |
| 13+ | 0014g | **0014g-canonical-siblings-cuda-impl.patch** | 62d866e6 | **NUOVA v2.1** — def `ds4_cuda.cu` (~2084 righe); vedi §v2.1 |
| 14 | 0015 | **0015-pace-canonical.patch** | **86d67a95** | **RIGENERATA** — vedi nota fattorizzazione |
| 15 | 0016 | **0016-pace-canonical.patch** | 73cfa4ab | RIGENERATA, **byte-identica** all'originale `0016-pace-rebuild-on-tighten` (73cfa4ab) |
| 16 | 0018 | **0018-pace-canonical.patch** | 5b018a3b | RIGENERATA, **byte-identica** all'originale `0018-pace-skip-wrap-on-rotate` (5b018a3b) |
| 17 | 0020 | 0020-pace-s1-slope-trigger.patch | 089da7cd | copia LF, ≡ originale (applica invariata sopra la v2) |
| 18 | 0021 | 0021-pace-rotate-delta-prefetch.patch | 7fb78678 | copia LF, ≡ originale (applica invariata) |
| 19 | 0026 | 0026-pace-demand-admission.patch | d933dec1 | copia LF, ≡ originale (applica invariata) |

**Fuori dalla v2 (per progetto):** `0009/0010` (dspark-MTP, arena diversa, non applicano
su base pulita), `0017` (proposta mai applicata), rami `0015/0016-spex-hidden`
(serie SPEX-hidden, regola 3). `0027` (rewind-harness, appena registrata) è ANCORA fuori
scope (mandato = base→0026): applica invariata sopra la v2 (stessa ancora live-tree) e sarà
l'incremento successivo — aggiungere una sola riga per estendere la catena.

### Fattorizzazione scelta e perché

Le tre patch pace originali sono state ri-ancorate **NON** invertendole sulla base pulita
(non applicano), ma **reverse-peel** dal live-tree: `git apply -R 0018` poi `-R 0016`
riescono **esatti** ⇒ `0018-pace-canonical` e `0016-pace-canonical` sono byte-identici agli
originali (stessi hash). `git apply -R 0015` **fallisce** (2/13 hunk in `ds4_pace_init`):
il lavoro "tiering" non committato del live-tree ha **riscritto le regioni che 0015
toccava**, quindi il rotate di 0015 **non è più separabile** dal substrato. Perciò:

- **`0015-pace-canonical` = delta `fine-0014e → live-tree` MENO (0016+0018)**: un unico
  patch che riproduce il rotate raw-router (intento originale di `0015-pace`, ~46 righe:
  `rotate_on/every/decay`, `rmass`, `ds4_pace_note_router_probs`, `ds4_pace_rotate_maybe`,
  readback `router_probs`) **INSIEME** a tutto il substrato live-only mai formalizzato
  (vedi audit sotto). Non c'è confine pulito rotate-vs-substrato: struct `g_pace` e
  `ds4_pace_init` sono condivisi riga-per-riga. Uno split *pace* `0014f` separato **non è
  recuperabile fedelmente**; il 3-patch ai confini reversibili è la fattorizzazione più pulita
  ottenibile. (NB: i numeri `0014f`/`0014g` sono ora usati per le patch **sibling** GPU
  `0014f/0014g-canonical-siblings-*` della §v2.1 — file diversi, concern diverso: fratelli
  `ds4_cuda.cu`/`ds4_gpu.h`, non lo split pace qui ipotizzato.)
- **`0016-pace-canonical`** = rebuild mask on tighten (≡ originale).
- **`0018-pace-canonical`** = skip full WRAP on rotate (≡ originale).

### AUDIT — cosa contiene il live-tree che NESSUNA patch aveva catturato

Il delta `fine-0014e → live-tree` (~1213 righe nette, 51 hunk) è **molto più largo** della
lista "campi live-only" già in §"Stato apply". Oltre al substrato pace lì elencato
(`prebreath_*`, `cache_flush`, `prefill_apply/wait_wrap`, `exchange_*`, `weighted_*`,
`in_prefill`, `last_prebreath_tok`, helper fn, variante WRAP-live), `0015-pace-canonical`
**canonizza per la prima volta** i seguenti sottosistemi **mai presenti in alcuna patch**:

- **Sottosistema SPEX-hidden GPU (~162 righe):** typedef `ds4_spex_hidden_params`,
  `metal_graph_spex_hidden_*` (load/prefetch/score/`gpu_topk_read`/stats), campi
  `spex_hidden_gpu_topk_count/layer`. È il ramo SPEX-hidden (regola 3, teoricamente arena
  separata) **fisicamente presente nel `ds4.c` live-tree** — quindi ora dentro la v2.
- **Scheduler seed/prefetch "cq1/tiering cold-sidecar" (~58 righe):** secondo typedef
  con `seed_calls/experts/ok/failed/skipped_all_resident`, `scheduled/schedule_failed`,
  `candidate_experts/resident_before/ready/not_ready/zero_candidates/dry_run/seed_seconds`.
  Combacia coi commit live-tree "tiering: add cq1 cold sidecar", "gate cq1 by prompt
  hotset", "observe pace breath exchange" (mai patchati).
- **Utility env/cache-floor:** `ds4_env_truthy`, `ds4_env_u32`,
  `ds4_streaming_cache_token_working_set_slots`, `ds4_engine_apply_pace_cache_floor`
  (`DS4_PACE_CACHE_FLOOR`, `DS4_PACE_CACHE_TARGET_SLOTS`), `ds4_spex_path_magic_is`.
- **Integrazione engine/session:** hunk in `ds4_engine_configure_streaming_auto_cache`,
  `ds4_engine_open`, `ds4_session_sync` (apply mask prefill + wiring cache-floor);
  estensioni `ds4_spex_markov_*` oltre 0004.

Solo ~46 delle ~1213 righe sono il rotate originale di 0015; il resto è substrato live-only.

### Regola di switchover

Gli **originali restano la serie operativa** finché non c'è uno **SMOKE su pod della v2**
(build da `80ebbc3` + apply `canonical/` in ordine + boot + smoke di meccanismo rotate32).
**Solo dopo lo smoke verde** la v2 diventa canonica e `0015-pace-raw-router-k-rotation`
/ `0016-pace-rebuild-on-tighten` / `0018-pace-skip-wrap-on-rotate` vengono marcati
**deprecati** (NON rimossi finché il pod agent in corso li usa). `0020/0021/0026` sono
identici in entrambe le serie.

### Rischi (dichiarati)

1. **BUILD non ancora verde su pod (mandato CPU-only):** apply-clean + symbol-checked ≠
   compila-e-linka verificato. **RISOLTO il gap noto** che faceva fallire il pod build v2
   (§v2.1): i fratelli `ds4_cuda.cu`/`ds4_gpu.h` ora sono ricostruiti **allo stato live-tree**
   (0014f/0014g, byte-identici md5 `7d57f58d`/`55070d97`), quindi `ds4_gpu_async_read` e gli
   altri simboli SPEX-hidden/GPU-prefetch sono dichiarati+definiti. Poiché la catena ricostruisce
   ora `ds4.c` **e** entrambi i fratelli byte-identici al tree live-buildabile (che ha prodotto
   il binario `ds4`), il build su pod dovrebbe passare. Residuo: la conferma è un
   **build+boot su pod** (mandato CPU-only qui: nessuna toolchain CUDA locale) — non solo
   apply/symbol-check. ⇒ **lo smoke di switchover DEVE essere build+boot su pod.**
2. **OVER-SCOPE:** `0015-pace-canonical` canonizza sottosistemi non-pace (SPEX-hidden,
   cq1/tiering) perché entangled nello snapshot live a cui le leve sono ancorate. Fedele al
   binario reale, ma mischia i rami. Un refactor "pace-minimale" futuro potrà sfoltire — ma
   solo dopo aver verificato che 0020/0021/0026 applicano ancora sull'albero sfoltito.
3. **ENTANGLEMENT irreversibile:** l'originale `0015-pace` non è più reverse-applicabile
   (tiering ha riscritto `ds4_pace_init`); un `0014f`+0015 sottile non è ricostruibile
   fedelmente. La 3-patch è il massimo di separazione onesta ottenibile.

Provenienza base pulita: live-tree WSL `/root/ds4` (remote `github.com/antirez/ds4`,
commit `80ebbc3` in history) — coerente con `runs/ds4/20260710_pod_t1_full_positive_control/README.md`.
