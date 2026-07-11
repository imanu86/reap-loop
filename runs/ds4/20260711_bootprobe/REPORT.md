# BOOT-PROBE — RTX 3060 reference profile (2026-07-11)

Probe: `scripts/boot_probe.py` (design: `docs/BOOT_PROBE_DESIGN.md`). First
reference HW profile for the P2 auto-calibration gate. Artifact: `profile.json`
(schema `reap-loop/boot-probe/v1`).

Reproduce (on the target host):
```
python3 scripts/boot_probe.py --model /root/models/ds4-2bit.gguf \
    --ds4-bin /root/ds4/ds4 --out runs/ds4/20260711_bootprobe/profile.json
```
The committed `profile.json` was captured with `--no-ds4 --no-drop-caches` to
avoid disturbing a concurrent local T4 sweep — see "Contention".

## What the probe measures (four probes)

| Probe | Field | Value on the 3060 | Provenance |
|---|---|---|---|
| (a) VRAM | `gpu.vram_total_bytes` | 12 GiB (12288 MiB) | measured nvidia-smi |
| (a) footprint | `model.footprint_per_expert_bytes` | 7,077,888 B = 6.75 MiB (iq2_xxs gate/up + q2_k down) | measured GGUF |
| (a) geometry | `model.recalls_per_token` | 43 MoE layers x 6 = 258, 1.70 GiB/token demanded | measured GGUF |
| (a) split | `model.nonexpert_resident_bytes` | total 80.76 GiB / expert 72.56 / nonexpert 8.20 | measured GGUF |
| RAM | `ram.ram_available_bytes` | 60.8 GiB total, ~58.7 available | measured /proc |
| (b) bandwidth | `bandwidth.*` | idle 0.314 / 2.10 / 16.6 GiB/s (see Contention) | measured |
| (c) baseline | `baseline.*` | cold 1.07 t/s; steady ~3.12 (E-LAT) | measured / see below |
| (a)->cache | `derived.cache.cache_slots` | 394 (cross-check E1 max 407) | derived |
| (d)->K* | `derived.k_initial.k_initial` | 48 (wide knee >= cov90 floor 38) | invariant |
| (b)->WRAP | `derived.offload_regime.wrap_recommended` | off (ram_fit_ratio 4.31 >= 1) | derived |

Launch contract (`--emit-launch`):
```
export DS4_PACE=1 DS4_PACE_AUTO=1 DS4_PACE_KEEP=48 DS4_PACE_WRAP=0 DS4_PACE_WRAP_ROTATE_DELTA=0
ds4 -m /root/models/ds4-2bit.gguf --ssd-streaming --ssd-streaming-cache-experts 394
```

## Contention — an unintended robustness demo

`profile.json` was captured while a concurrent local T4 W-sweep (a separate
`ds4 --ssd-streaming-cold` process) was hammering the SSD and GPU. So the
committed `bandwidth` block is contended, ~3x degraded:

| Metric | contended (profile.json) | clean idle (measured first, no other ds4) |
|---|---|---|
| SSD cold single-stream | 0.356 GiB/s | 0.314 GiB/s |
| SSD cold threaded (4-way) | 0.670 GiB/s | 2.10 GiB/s (matches E-LAT tier-c 2.5-4.4) |
| RAM warm read | 2.67 GiB/s | 16.6 GiB/s |

The `wrap_recommended = off` verdict is identical under both, because it is gated
on the dimensionless `ram_fit_ratio` (working set 13.6 GiB at K=48 vs 58.7 GiB
available = 4.31 >= 1), immune to bandwidth noise. This is exactly the P2 intent:
the control decision rides on a pure ratio, not an absolute MB/s, so a 3x
measurement error — or a different NVMe — does not flip it.

## Baseline t/s (probe c) — measured, honestly noisy

- Cold first ds4 run (uncontended, cache populating): prefill 0.55, generation
  1.07 t/s — not steady (warm-first discipline discards it).
- Warm re-run: contaminated to 0.35 t/s by the concurrent sweep (GPU contention)
  -> discarded.
- Clean steady reference: E-LAT-calibrated 3.12 t/s @ cache256
  (`runs/ds4/20260710_elat_tier_latency/REPORT.md`). A clean warm probe-(c) run
  needs an idle GPU; the script does it automatically (`--warm-first`, default)
  when the host is free.

## Measured vs stub (declared)

Measured on this host: VRAM (nvidia-smi), per-expert footprint + geometry +
model byte-split (GGUF), RAM (/proc), SSD cold single/threaded + RAM warm
bandwidth (idle numbers above), a cold ds4 baseline decode.

Stub / not first-party here:
1. H2D (cudaMemcpy) bandwidth — no isolated CUDA micro-bench; only used inside
   the `--no-ds4` estimate fallback, which the real profile does not depend on.
2. Clean warm steady t/s — needs an idle GPU; here cited from E-LAT.
3. Engine-side `DS4_PACE_AUTO` reader (patch 0029) — designed, not authored
   (CPU-only host cannot compile ds4.c); the launcher contract delivers the full
   behavior today with zero untested C.
4. Second-HW confirmation (pod 3080/3090) — analytic for the 1 TB-RAM pod;
   empirical confirmation is S5.
