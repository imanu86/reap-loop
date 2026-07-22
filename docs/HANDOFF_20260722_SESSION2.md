# HANDOFF — sessione 2026-07-22 (pomeriggio/sera) → prossima chat

Tutto committato e pushato. Nessun lavoro a metà non salvato. La nuova chat può usare
i comandi Codex avanzati (`codex exec`, `/codex:rescue`) — CLI 0.144.6 + plugin installati.
**Regola delega**: Codex per il lavoro (read-only affidabile; il sandbox `workspace-write`
è ROTTO sulla workstation → patch = Codex autora hunk SEARCH/REPLACE stampati, Claude applica).

---

## 1. RISULTATI CHIAVE DELLA GIORNATA (ledger righe 708–723, tutte committate+pushate)

| # | Cosa | Numero |
|---|---|---|
| 1 | **G73-open trasporto**: catena F1+F2+F3+F3b+F5 | **1.65 → 4.60 t/s steady** (chiuso 8.16), clamped=0 sempre |
| 2 | **Leak RAM risolto**: force-kill durante unregister orfanava 20+GB | ricetta: MAI force-kill; `taskkill /PID` (senza /F) = uscita pulita |
| 3 | **Chat reale**: crollo progressivo a ctx≥1536 | decode 5.4→9.3s/token che SALE; a ctx 640-768 piatto ~350ms (2.87-4 t/s) |
| 4 | **KV scagionata**: MLA → KV viva ~247MiB a ctx4096 | il ladro VRAM è lo **scratch prefill `g_cuda_tmp` ~1.5GiB mai rilasciato** |
| 5 | **IQ1_S wikitext valido**: morto | routing 10.0% uniforme, collasso identico al file rotto |
| 6 | **Q1 calibrazione**: morta | imatrix +0.003 coseno (0.600→0.603); STE-adapter 0.701 resta il best |
| 7 | **Q1 trained-max** (pod): overfit | train 0.962 (capacità c'è!) / test 0.651 — muro = 256 campioni |
| 8 | **Cattura all-experts H200: FUNZIONA** | smoke 1199 righe (200 vettori × 6 route), byte esatti, manifest ok |

## 2. POD H200 `ds4-capture-h200` — ATTIVO, 4.39$/h — ⚠️ MAI SPEGNERE SENZA CHIEDERE ALL'UTENTE

- Endpoint ssh in `D:\ds4_work\pod_conn.json` → **NO**: quello è il pod piccolo; H200 in `D:\ds4_work\h200_conn.json` (ip/port; se cambiano: API RunPod via key in `C:\Users\imanu\Desktop\Runapod.txt` — MAI stampare la chiave; pattern query in questa chat).
- Stato all'handoff: **cap5 vivo (pgrep=1), sync_loop R2 vivo, GPU 0%** = server in load o in wait-loop. PRIMO ATTO della nuova chat: `ssh … 'tail /root/cap5.log; nvidia-smi'`.
- **cap5** = `/root/h200_cap5.sh` (driver corretto: UNA sessione server + loop richieste; target 80k vettori o 3h cap). Log: `/root/cap5.log`. Output: `/root/capture_L15/L15_all.{vectors.f32le,samples.jsonl,manifest.json}(.partial)`.
- **sync_loop**: rclone → `r2:ds4-models/capture_L15` ogni 10 min (esclude .partial). rclone.conf già configurato sul pod.
- Build capture sul pod: `/root/ds4-cap` branch `h7b` = commit `6a0c85e` (bit-exact col locale). Modello: `/root/models/ds4-2bit.gguf` sha VERIFICATO efc7ed60….
- **Cache device è LAZY** (spans preparati, riempita on-demand): primi token ~3.7 t/s, sale col warm. Env delle 3 build-sha OBBLIGATORIE (`DS4_EXPERT_RECOVERY_{EXECUTABLE,BUILD_MANIFEST,BUILD_INPUT_FINGERPRINT}_SHA256`), OUTPUT_PREFIX deve essere PATH COMPLETO con parent==ROOT, MAX_BYTES sempre impostato. Tutte le trappole già risolte in cap5.
- SSH del pod INSTABILE (host load ~27): pattern robusto = scp/ssh con retry ×3-5, `setsid … </dev/null`, mai heredoc annidati su ssh.
- Budget utente: aveva 22$; spesi ~7-9$. Cattura fase 1 ≈ 13$ residui. "In caso ricarico" (parole sue).
- DOPO la cattura: training per-expert sul **pod piccolo** `ds4-q1train-3090` (spento? verificare; il deploy fresco costa 30s — i resume dei pod vecchi FALLISCONO, host pieni). Trainer: `/root/q1train/q1_train_max.py` sul 3090 (o D:\ds4_work\c7_bonsai\q1_train_max.py locale). Obiettivo: test-cosine con ~2-5k campioni/expert vs il muro 0.70@256.

## 3. LOCALE (workstation 3060) — stato esatto

- **Nessun ds4_server attivo**. RAM: 15.6GB in uso, **8.9GB orfani** (colpa del pulsante Stop della UI DS4 Control che force-killa → stesso male del gate pre-fix). Sotto i 20GB critici, ma **un riavvio prima della prossima sessione grossa è consigliato**; l'audit è `D:\ds4_work\ram_audit.ps1` (OBBLIGATORIO pre-gate: NON-ATTRIBUITA deve essere ~0-2GB).
- **Chat**: UI DS4 Control = `C:\Users\imanu\Documents\Codex\2026-07-07\cia\outputs\ds4-simple-ui` (porta 8787, backend `ds4-ui-server.py` con `DS4_UI_PORT=8787 DS4_PORT=8000`). Server chat: `D:\ds4_work\g73_gate\run_chat_server.sh <ctx>` (config F5). **ctx 640-768 = 2.87-4 t/s piatti; ctx≥1536 = spirale**. Fix pendenti: F6+F7 (sotto). Il pulsante Stop della UI andrebbe rifatto con taskkill graceful.
- Gate A/B: `D:\ds4_work\g73_gate\run_g73_gate.ps1` (pin HEAD/exe/preset AGGIORNATI a 174f4af; stop-discipline graceful patchata; seed 160). Single-arm: `run_g73_single_arm.sh <pageable_gb>`.

## 4. F6+F7 — IL PACCHETTO "CHAT NUMERI GRANDI" (pronto da applicare, NON applicato)

- **F6** (rilascio scratch prefill + ledger VRAM): hunks autorati in `D:\ds4_work\g73_fix\f6.out.log`, formato `<<<HUNK n>>>/<<<FILE>>>/<<<SEARCH>>>…` — **8 hunk unici su 5 file** (ds4_gpu.h, ds4_cuda.cu, ds4.h, ds4.c, ds4_server.c; nel log sono duplicati: dedupe per numero). **7/8 applicano puliti; h6 (ds4.c) ha SEARCH ambiguo (match=2)** → disambiguare col contesto (guardare i 2 siti, scegliere quello giusto o farlo ri-autorare con più contesto). Env: `DS4_CUDA_RELEASE_PREFILL_SCRATCH=1` + ledger `[vram-ledger]` sempre attivo. NIENTE è stato scritto su wt-g73-open (applier atomico, ha abortito).
- **F7** (KV managed, ~20 righe, da autorare): instradare le 3 cache KV (`ds4.c:10717`: layer_raw_cache, layer_attn_comp_cache, layer_index_comp_cache) sull'allocatore managed ESISTENTE (`cudaMallocManaged`, ds4_cuda.cu:6642) dietro env `DS4_CUDA_KV_MANAGED=1`. Prior art: upstream ha `ds4_gpu_should_use_managed_kv_cache` (postdata il nostro fork); **ds4 issue #108 chiede esattamente questo, aperto/senza risposta**; llama.cpp `-nkvo` prova la config. Studio completo (737KB, anchor esatti, piano mapped/staged, rischi): `D:\ds4_work\g73_fix\kvram.out.log`.
- **Accettazione F6+F7**: build → chat reale ctx 8192 → `[vram-ledger]` mostra ~1+GiB liberati al decode-start → decode_ms PIATTO (slope <10%) → vram_hit resta ~30%+. Se regge: la chat "numeri grandi" dell'utente è fatta.
- Build command (workstation): `cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && "...\CMake\bin\cmake.exe" --build build --config Release --target ds4_server -j 8'` in `D:\ds4_work\wt-g73-open`; output Ninja in `build\ds4_server.exe` → **copiare in build\Release\** (il gate legge lì); aggiornare i 3 pin del gate a ogni rebuild.

## 5. REPO — tutto pushato

| Repo | Branch | Tip | Remote |
|---|---|---|---|
| `D:\ds4_work\wt-g73-open` | g133/g73-open | **174f4af** (F5) | pushato → wt-lane-a2 ✓ |
| `D:\ds4_work\wt-c7-capture` | g134/c7-capture | **6a0c85e** (POSIX gate fix) | pushato → wt-lane-a2 ✓ |
| `C:\…\reap-loop` | plan/0051-transport-gate-20260713 | **db29bec** (ledger 723 righe) | pushato → github.com/imanu86/reap-loop ✓ |

Dirty NON miei (pre-esistenti, lasciati): wt-g73-open `tests/g73_open_measured_gate.ps1`; reap-loop `runs/ds4/20260713_0051…/{RESULTS.md,phase1_transport_safety.sh}` e `reaper.log`.
Pod branch fork: `/root/ds4-win` (bake pod, EXITED) ha lineage fix3-fix5 divergente da 3950a70 — riconciliazione col nostro F1-F5 PENDENTE (stessa famiglia di problemi risolta due volte).

## 6. AGENTI / TASK IN CORSO all'handoff

- **Nessun agente Codex attivo.** Output autorati conservati in `D:\ds4_work\g73_fix\*.out.log` (tracer a1-a4, f2a, f3a, f5a/f5b, f6, kvram, captrace, h7fix, q1train, q1pilot).
- Sul POD girano (setsid, sopravvivono): **cap5** + **sync_loop**. I monitor locali sono morti con questa sessione → la nuova chat rimette un monitor (pattern: ssh con retry + tick 60-90s, autoheal se cap5_procs=0 rilanciare `setsid bash /root/h200_cap5.sh`… ATTENZIONE: rilanciare cap5 fa `rm -rf` dell'output → prima controllare i vettori parziali e il sync R2!).
- Pod piccolo `ds4-q1train-3090`: stato da verificare (era RUNNING; se idle e non serve → chiedere all'utente se spegnerlo).

## 7. PRIORITÀ CONSIGLIATE PER LA PROSSIMA CHAT (ordine dell'utente: chat grande > Q1)

1. **Monitor pod**: cap5 progredisce? (vettori, t/s warm-up, ETA, budget). Recuperare dataset da R2 quando pronto.
2. **F6**: risolvere h6 ambiguo → applicare 8/8 → **F7** autorata+applicata → build → **test chat ctx 8192** (il deliverable che l'utente vuole: "numeri grandi").
3. Training pilota sui campioni nuovi (pod 3090 fresco) → il numero che decide il Q1-companion.
4. Riavvio macchina consigliato (8.9GB orfani) prima dei gate ledger-grade.
5. Se cap5 è morto/stallato: NON spegnere l'H200 senza chiedere; diagnosticare con `/root/cap5.log` + `/root/capture_L15/server.stderr`.

## 8. LEZIONI OPERATIVE DELLA SESSIONE (costate ore — non ripeterle)

- Sandbox write Codex ROTTO (helper_unknown_error) → pattern hunk-stampati. I formati marker VARIANO tra run (con/senza titolo, git-conflict style) → parser flessibile, dedupe, routing per-file, dry-run PRIMA di scrivere, verificare che il commit includa TUTTI gli hunk prima di bundle+build remoto (l'hunk 7 dimenticato è costato un ciclo rebuild).
- SSH pod flaky: retry, setsid, niente heredoc annidati. File → scp → esegui.
- Gate/bench: OFF-baseline = controllo di contaminazione (8.1 t/s o il run si butta); browser/ChatGPT aperti falsano tutto (page-out spiral).
- I contatori: `served_transient` include serve da RAM (non solo SSD); `out_of_mask` non scende mai per design; il t/s "avg" del server è cumulativo (usare decode_ms per-token per vedere le spirali).
- locale: cmake = quello di VS (path in CMakeCache) + vcvars64; taskkill senza /F da PowerShell (git-bash storpia /PID).
