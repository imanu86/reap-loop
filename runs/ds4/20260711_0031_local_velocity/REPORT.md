# 0031 pin-keep LOCAL velocity on 3060 — VRAM-resident vs warm-RAM refetch

Host: RTX 3060 12GB, WSL Ubuntu-24.04, driver 596.21 (CUDA cap 13.2).
Bin: `/root/ds4_pin/ds4` (sm_86, 0031 endpoint `ds4_cuda.cu` md5 `430716f4`).
Task: cyberpunk, static mask `sessCyber_K23.txt` (~989 keep), greedy temp0, ctx 2048.
Config (vaccinations): own lock `/tmp/ds4_pin_velocity.lock` (UI:8000 untouched),
`DS4_CUDA_NO_Q8_F16_CACHE=1`, warm-RAM lever (`NO_DIRECT_IO=1`,`KEEP_MODEL_PAGES=1`),
`RESERVE_GB` sized. per-expert = 6.75 MiB (confirmed by pin_freeze log).

## FASE 1 — CUDA toolkit in WSL: DONE (no download needed)
CUDA toolkit **12.8** was ALREADY installed (`/usr/local/cuda-12.8`, `nvcc` V12.8.93),
only missing from the login PATH. `<=` driver max (13.2) and matches the toolkit the
existing binary was built with. Meta `cuda` pkg NOT touched (toolkit-only). `nvcc
--version` OK. apt repo: `cuda-wsl-ubuntu-x86_64.list` (correct wsl-ubuntu repo).

## FASE 2 — build 0031 for sm_86: DONE (RC=0, 0 warnings)
`/root/ds4` was already exactly at the canonical v2.1 endpoint (`ds4_cuda.cu` md5
`7d57f58d`, `ds4.c` `771a39a8`, `ds4_gpu.h` `55070d97`), so only 0024+0031 were needed
(both touch ONLY `ds4_cuda.cu`). Built in an isolated copy `/root/ds4_pin` (`cp -a`) to
not disturb a running job. md5 chain verified at each step:

| step | ds4_cuda.cu md5 | expected |
|---|---|---|
| base (canonical v2.1) | `7d57f58d` | `7d57f58d` OK |
| + 0024 | `c564ca7c` | `c564ca7c` OK |
| + 0031 | `430716f4` | `430716f4` OK (endpoint) |

`make ds4 CUDA_ARCH=sm_86` -> RC=0, 0 warnings, 62s. Fresh ELF (BuildID `7d00109b`), boots.

## FASE 3 — the decisive measurement

### Mechanism (from 0031 source): pin budget is a SUB-BUDGET of the already-allocated
streaming cache. Pinning adds NO VRAM; it only makes cache slots eviction-immune (LRU
skips `pinned`). So to keep the keep-set resident you must run a LARGE
`--ssd-streaming-cache-experts`, and the cache must PHYSICALLY fit in VRAM.

### VRAM reality on the 3060 (the wall)
The model's non-expert weights + KV consume **~10-11.4 GB of the 12 GB** even at ctx 2048
with a minimal cache (peakGPU 11593 MB at cache 32). That leaves only ~1.5-2 GB for the
expert cache = **~220 experts max** resident. The K23 keep-set (~989 experts, ~6.7 GB)
does NOT fit. (Matches the campaign's prior finding: "cache ~407 exp non 910".)

### A/B/C (all ctx 2048, K23 mask, greedy, -n 120) — BIT-EXACT across all (genmd5 af5571f4)

| run | cache | PIN | budget | gen t/s | resident-hit | peakGPU MB | pin_freeze | pin_rotate |
|---|---|---|---|---|---|---|---|---|
| A (baseline warm-RAM) | 32  | 0 | -    | **0.86** | 0.0000 | 11593 | - | - |
| C (big cache, no pin) | 220 | 0 | -    | 0.74 | 0.0000 | 11915 | - | - |
| B (PIN resident+rotate)| 220 | 1 | 1400 | **0.78** | **0.1307** | 11903 | 1 (pinned 192) | 22 |

(probe cache256/pin0: 0.51 t/s, hit 0.0214, peakGPU 11921 — same VRAM wall.)

### Findings
1. **Bit-exact confirmed**: A/B/C all produce identical greedy output (`af5571f4`).
   Residency != selection: PIN never changes the tokens. Invariant holds.
2. **PIN mechanism works** — and is the ONLY thing that raises residency: big-cache LRU
   alone (C) stays at hit 0.0000 (keep-blind, evicts keeps before reuse = the
   pinning-divergence 0031 targets); PIN (B) lifts resident-hit 0 -> 0.13 by making the
   hot 192 keeps eviction-immune.
3. **But no velocity jump**: B 0.78 t/s ~= A 0.86 t/s (marginally slower). Raising
   residency to 13% does not move t/s because the other ~87% of the 989-keep working-set
   still refetches H2D (~6.7 GB/token) every step — that refetch is the bottleneck.

### THE ANSWER
On the real 3060, **VRAM-residency does NOT give the jump — it is CAPPED, and the cap is
VRAM CAPACITY, not bookkeeping/blocking-sync.** The non-expert weights + KV already take
~10-11.4 GB of 12 GB, so at most ~13-22% of the K23 keep-set can be made resident; the
rest refetches every token and pins t/s at ~0.8, nowhere near the ~35-45 t/s VRAM-
bandwidth ceiling. Pin-keep cannot beat warm-RAM here because the keep-set does not fit.
The lever only pays off on hardware (or a smaller keep-set / MoE with a lighter non-expert
footprint) where the working-set actually fits VRAM alongside the model.

Note: the mission's assumed ~3.6-3.8 t/s baseline is for a much smaller/warmer working
set (e.g. code256 K23 ~256 experts that DO fit, per RESIDENT_HIT_FIX 3.12->3.7-4.1). The
cyberpunk K23 mask has ~989 keeps, so its warm-RAM baseline is 0.86 t/s (refetch-bound).

### Operational notes
- A first attempt (A0) collided with the parallel `20260711_domain_calibration` campaign
  (both launched at 13:31:24) -> combined VRAM hit the 12 GB ceiling -> WSL wedged
  (`Wsl/Service/E_UNEXPECTED`). Recovered with `wsl --terminate Ubuntu-24.04` (no host-RAM
  OOM; vmmemWSL was 2.8 GB, host had 50 GB free). That terminate also halted the campaign
  driver; A/B/C then ran on an exclusive GPU (verified: campaign START count stayed 5, no
  relaunch/collision during A/B/C). A0 discarded. GPU left free; UI:8000 never touched.
