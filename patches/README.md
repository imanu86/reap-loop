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
