# coffee@12GB on native gen4 x16 — BANDWIDTH CONFIRMATION

**Date:** 2026-07-13
**Goal:** confirm whether the 3060/WSL result (843 MiB/tok / 2.81 GiB/s WSL = 3.4 t/s) was H2D-DMA-transfer-bound, and whether at native gen4 24 GiB/s coffee@12GB reaches ~10-15 t/s.

## Node (gen4 x16 VERIFIED — non-negotiable gate passed)
- **GPU:** NVIDIA RTX A5000 (GA102, sm_86 — matches binary), 24564 MiB, driver 570.211.01
- **Cloud:** RunPod **secure** cloud, pod `32qbfbamo634hp`, $0.270/hr
- **PCIe:** current `gen1 x16` (idle power-save), **max `gen4 x16`** — VERIFIED via `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max` → `4, 16`
- **H2D pinned bandwidth (torch, 1 GiB, best of 10):** **23.63 GiB/s** — native gen4, ~8.4x the 2.81 GiB/s WSL cap. Link is NOT the bottleneck.
- CUDA gate: `cuInit rc=0`, torch micro-alloc OK.

## Binary / model
- Binary: `ds4-server` git da0b3f6 sm_86, md5 `3bc7c5200d288e26eb95fb664b3d16f2` (WSL build 2026-07-13 12:01)
- Model: `ds4-2bit.gguf` from R2, size 86720111488 (exact), downloaded in 2m17s (~630 MB/s)
- Envelope: **12 GiB** (`DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=12`), CACHE_PROFILE=1, cache-experts=400, temp0, warm=48 meas=1600

## Arms

| arm | decode t/s | copy ms/tok | noncopy ms/tok | eff. copy GiB/s | dma MiB/tok | total MiB/tok | hit_rate | cover | grade |
|-----|-----------:|------------:|---------------:|----------------:|------------:|--------------:|---------:|------:|------:|
| coffee_k83_promo      | **4.423** | 191.7 | 2.33  | 2.43 | 478.2 | 1012.6 | 0.446 | 0.472 | 0 |
| coffee_k83_staticpin  | 4.337     | 189.9 | 8.20  | 2.71 | 526.8 | 1103.8 | 0.393 | 0.477 | 0 |
| coffee_k65_promo      | 4.917     | 161.3 | 17.50 | 3.51 | 580.7 | 973.0  | 0.469 | 0.597 | 0 |

(eff. copy GiB/s = dma_MiB_per_tok / copy_ms_per_tok — the bandwidth the streaming path actually achieves during decode.)

## VERDICT

**At native gen4 x16, coffee@12GB does NOT reach ~10+ t/s — it does ~4.4-4.9 t/s. The bottleneck was NOT the PCIe link.**

- The physical link is proven at **23.63 GiB/s** (pinned bench), yet the decode copy path only achieves **2.4-3.6 GiB/s effective** — a 7-10x gap.
- Root cause is the streaming copy path, not the link: ~450-490 VRAM-miss copies per token, avg ~2.25 MiB each (`copy_calls` ~732k-800k over the run), plus ~390k-420k `cuda sync` calls (`sync_ms` ~53s ≈ 32 ms/tok). Small pageable/synced transfers cannot exploit a 24 GiB/s link; `copy_ms/tok` (~160-190 ms) dominates ~75-85% of the ~226-231 ms/tok decode budget.
- **This REFUTES the "3.4 t/s = H2D-transfer-bound → 24 GiB/s gives 10-15 t/s" hypothesis.** Moving to a native link barely helped (3.4 → 4.4 t/s) because the copy path — not the wire — is the limiter. To reach 10+ t/s the lever is copy-path efficiency (pinned staging + larger transfer granularity + async overlap / fewer syncs), not a faster PCIe link.

## Promotion delta (promo vs static, k83)
- Speed: +0.086 t/s (+2%) — negligible; both are copy-path-bound.
- Hit-rate: 0.393 → 0.446 (+5.3 pp); DMA bytes/tok: 527 → 478 MiB (-9%).
- Promotion (pin-by-mass + livemask rating-only + pressure) improves residency quality and cuts bytes moved, but the win is masked because the copy path is the ceiling.

## Quality note
All arms grade 0: temp0 deterministic generation was truncated at the 1600-token cap mid-CSS (html not closed, no body/nav/hero/form/script reached). `style=true` only. This is a token-cap truncation artifact, not a 2-bit collapse — the visible HTML/CSS is coherent. renders=false, close_html=false for all three.

## Cost / lifecycle
- Runtime ~55 min @ $0.270/hr ≈ **$0.25**. Pod terminated at end of run.
