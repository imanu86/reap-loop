# HANDOFF 2026-07-24 pomeriggio — stato pre-compattazione

## ✅ BUILD FATTO — patch fp8 di Codex compilata nel binario
- **Patch applicata** a `D:/ds4_work/wt-g73-open/ds4_cuda.cu` (uncommitted; committare DOPO che il test ring valida) e **compilata OK**: `ds4_server.exe` ricompilato 2026-07-24 16:00:30 (BUILD_EXIT 0 in ~67s, solo warning C4244 benigni). ds4_cuda.cu.obj fresco.
- **NOTA build**: nvcc ha bisogno di `vcvars64.bat` (cl.exe in PATH) — ninja da solo fallisce "Cannot find cl.exe". Usato `scratchpad/build_ds4.bat` (vcvars+ninja). Build velocissimo (~1 min per un TU).
- **PROSSIMO PASSO (post-reboot)**: TEST del ring a 250k = `bash D:/ds4_work/g73_gate/test_long_ring.sh`. Successo = decode produce token senza "illegal memory access" (prima crashava a fp8_kv_quantize su x->ptr NULL), cache 320 (ring libera VRAM), t/s > 0.57. Se OK → committare la patch in ds4_cuda.cu (branch g133/g73-open).
- **REBOOT**: fatto dopo il build (RAM sporca ~30GB + app Claude al 95% CPU per sessione lunga).

## 🏆 LA VITTORIA DELLA SESSIONE: DS4_G73_OPEN=0
**HTML da 0.62 → 4.85 t/s (7.8×) su IQ2.** Il flag flippa il dispatch a `routed_moe_launch_impl<true>` (ds4_cuda.cu:39696-98) = path lento "exact-transient-escape". NON era contenuto/temperature/cache/trasporto/warmth/residenza (tutte conclusioni affrettate mie, smentite una a una). Nel preset G73_OPEN=1 → **spegnere su IQ2 puro**. Dettaglio: EXPERIMENTS_LEDGER.md add.42.

## 🔧 IL FIX RING (la patch che sta compilando)
**Root cause (Codex):** a big-ctx col ring (`DS4_CUDA_KV_STAGED_RING=1`), la KV è staged in host → `x->ptr` (device) = NULL, ma `fp8_kv_quantize_kernel` gira comunque su `x->ptr` NULL → **illegal memory access**. La patch: se il tensore è staged, copia in ring slot VRAM, quantizza, ri-appende.
**Test dopo il build** (a macchina fresca): `D:/ds4_work/g73_gate/test_long_ring.sh` (G73_OPEN=0 + ctx250k + KV_STAGED_RING=1 + seed160). Successo = decode produce token senza illegal-access, cache 320 (ring libera VRAM), t/s > 0.57.

## 📋 CODA TOP-10 (no-rebuild, dall'atlante ri-analizzato)
Doc completo: `docs/DS4_LEVER_ANALYSIS_POST_20260724.md` (277 leve, verdetti post-oggi: 35 confermate, 22 mai-testate-promettenti, 14 morte, 4 conflitti, 3 ribaltate). Top-10 da tirare (ognuna 1-2 run):
1. `DS4_G73_OPEN=0` su IQ2 — FATTO (4.85). Il costo = dispatch <true> exact-transient (cu:39697).
2. `DS4_Q1_0_PROMOTION_SSD_WRAP=1` + `G73_OPEN=0` + sidecar Q1 = **test (a)**, disaccoppia Q1-serving dal dispatch lento (cu:1517 accende il wrap senza flippare il template). Script pronto: `test_a_q1fast.sh`.
3. `DS4_METAL_PREFILL_CHUNK=768` @ctx8192 — validato (0.56→3.92).
4. 5 sonde `DS4_METAL_*_STAGE_PROFILE` + `DS4_TOKEN_TIMING=1` — attribuiscono i 1776ms, mai accese.
5. `DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` — conta eviction VRAM.
6. `DS4_CUDA_MOE_IO_QD=2/4` — accende ~110 righe I/O overlapped morte.
7. `DS4_CUDA_KV_MANAGED=1` senza ring.
8. `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` — toglie un sync/token (il 4 è tarato per Metal!).
9. `DS4_CUDA_MOE_DIRECT_CACHE_HITS=1` — −1.6 GiB/token D2D sui layer all-hit.
10. `DS4_MTP_BATCH_VERIFY` + `think:false`.
Correzioni ai miei finding: il "conflitto Q8" (NO_Q8 vs RESERVE=4096) è INERTE non spreco (reserve consumata solo a cu:3104, irraggiungibile con NO_Q8); tensione G73_OPEN vs Q1 DISACCOPPIABILE.

## ⚙️ POD — campagna GPTQ Q1
- **6144 / 8177 = 75.1%** (verificato ~15:40). Rate ~600-650/ora. **ETA ~3h al 100%.** 8-11 worker teacher_backend attivi (gcc/CPU, GPU 0% by-design). Sano, non impuntato.
- Pod `ds4-camp-v2-155412` (8740zkq3kxexjg), SSH 213.192.2.94:40127 (porta cambia → GraphQL), network volume `uuj94bbe8ri602` = /workspace (dati salvi). Campagna PID 4509. Log `/workspace/campaign_resume.log`. Harness /root ridispiegato (si azzera a restart).
- **TODO pod**: download bulk sidecar a fine campagna (o incrementale). Il rebuild ring può usare la 4090 libera del pod (ma serve installare nvcc; il codice è PORTABILE su Linux — platform layer + rami #else + CMake NOT-APPLE, ma rami Linux non testati → rifiniture attese).

## 📁 SCRIPT PRONTI (D:/ds4_work/g73_gate/)
- `test_long.sh` — base HTML G73-open (per aggiungere G73_OPEN=0 o leve)
- `test_a_q1fast.sh` — test (a) Q1-veloce (PRONTO in coda)
- `test_long_ring.sh` — ring a 250k (per testare la patch fp8 dopo build)
- `seed_probe.sh` — sonda seed parametrica (BUDGET=n)
- scratchpad/build_ds4.bat — vcvars+ninja

## 💀 LEVE MORTE (non ri-testare)
PROMOTE_BUDGET, KNOCK_X/Y, EXPERT_TIER_ADAPTIVE (muove replacement_budget non il collo), STREAM_RUNTIME_RESERVE_MB / Q8_F16_CACHE_RESERVE_MB per la cap (sottosistemi diversi; la cap = SOLO STREAMING_EXPERT_CACHE_RESERVE_GB, cu:31240), MOE_CACHE_POLICY=lru (valore invalido no-op), reap-mass dinamico (mutuo-esclusivo hardcoded col g133 cu:37257 — serve rebuild per comporli).

## ⚠️ LEZIONI OPERATIVE
- **Rebuild+test ds4 satura il SSD locale** (ogni repro rilegge 44GB del modello 81GB, non entra in RAM 64GB) → freeze PC. Codex ha aggiunto anti-freeze all'harness (Idle priority durante load). Per iterare pesante → POD.
- Misure t/s locali FRAGILI (warmth-dipendenti, modello > RAM). RAM locale NON difettosa (0 WHEA/bugcheck/kernel-power41).
- Il server tiene un lock e ci mette ~120s a de-registrare una CUDA host window da 24GB allo shutdown → VRAM orfana (non è un run attivo).
- MAI killare alla cieca i PID che usano la GPU (includono explorer/chrome/claude!).

## COMMIT FATTI OGGI
- reap-loop `121e933`: ledger add.39-42 + DS4_LEVER_ANALYSIS_POST_20260724.md
- moe-aggressive-commit `80f726b`: ds4_q1_ref.c + kv_staged_fp8_fix.patch (Codex)
- wt-g73-open `729bdcb`: harness anti-freeze
- (da committare dopo build+test: la patch fp8 applicata a ds4_cuda.cu se funziona)
