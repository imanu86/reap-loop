# Virtual-bake study (2026-07-12) — progress log

Branch: `spex-predictive-mask-study-2026-07-12`. Binary: existing
`/root/ds4-fullstack/ds4-server` (no rebuild, V1 = zero new code). Model:
`/root/models/ds4-2bit.gguf` (86,720,111,488 bytes = 80.76 GiB, matches spec).

## Concept

Static per-layer top-K-by-mass REAP mask sized so the routed-expert working
set (keep% of 256/layer) fits in WSL page-cache RAM (~60 GiB budget), turning
every expert fetch into a RAM read instead of an SSD read. Quality gate =
near-lossless per the coverage study (see below). Speed gate = watch t/s
once warm.

## Orchestrator corrections applied (in order received)

1. **RAM sequencing**: fixed ~13.3 GiB weights transit page-cache -> VRAM
   during model load/startup, BEFORE any request is served; expert-window
   warmup only happens once the server is ready and we send the warmup
   request. This ordering is already enforced by the existing server
   startup contract (no new code needed to keep the two phases apart).
   Explicit `posix_fadvise DONTNEED` on the fixed-tensor byte ranges was
   considered and **rejected** for V1: identifying exact tensor byte ranges
   safely while the server has the file mmap'd live is not "zero new code"
   and risks evicting pages the server still needs. Mitigation actually
   implemented: a hard RAM-floor kill-switch (see below) plus reliance on
   normal clean-page LRU reclaim under pressure (file-backed mmap pages are
   clean/reclaimable by the kernel on demand).
2. **RAM hard floor**: `scripts/run_bake_arm.sh` runs a background monitor
   that samples `free -m` MemAvailable every 30s into `<run>/ram_log.txt`
   and immediately `kill`s the server (via `server.pid` only) if available
   RAM drops under 7168 MiB, writing `RAM_KILL.txt`. Keep-% escalation
   (55 -> 60 -> 65) is gated on the lower level having already held, per
   instruction.
3. **trace_K91 contamination**: `runs/ds4/20260711_highK_sweetspot/traces/trace_K91/route.csv`
   was flagged byte-identical to a K91-masked-demand trace calibrated on
   coffee (touches max 89 experts/layer, median 70) — NOT true K0 cyberpunk
   demand. **Not used.** Superseded by the next point.
4. **Real K0 trace already existed**: `runs/ds4/20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz`
   (gzip; decompressed to `traces/route_k0_cyber.csv`, 159,960 data rows /
   3999 tokens x 40 layers, width 203-247 experts/layer — genuinely K0). No
   new GPU trace-capture run was needed.
5. **Mass-coverage precomputed by orchestrator** on this real trace (miss %
   of mass excluded by each keep-set): keep55 self 2.86%/held-out 5.83%,
   keep60 self 1.94%/held-out 4.36%, keep65 self 1.29%/held-out 3.31% — all
   under the ~7.5% near-lossless coldtail threshold from the REAP-50 eval.
   Verdict: bake is GO on real data. **keep60_self promoted to primary arm**
   (best margin/window-size ratio, 40.6 GiB); keep55/keep65 are the scale
   arms; family (multi-domain held-out, excludes cyberpunk) is the
   control/prediction-check arm, expected to degrade per the coverage study
   (same-domain-different-prompt passes at 3-9.5% miss; cross-domain fails;
   4-domain union at keep50 = 61.7 GiB, not practicable on this box).

## Masks built (`scripts/build_mass_mask.py`, pure stdlib, no numpy)

Per-layer **fixed-count** top-K by summed routing weight (mass), MoE layers
3..42 (40 layers), output = blocked-expert lines `"<layer> <expert>"` (format
verified against `ds4.c:7739` `fscanf(f, "%u %u", &l, &e)`).

| mask | source | keep/layer | blocked lines |
|---|---|---:|---:|
| `masks/mask55_self.txt`  | real K0 cyberpunk trace (3999 tok)        | 141 (55.1%) | 4600 |
| `masks/mask60_self.txt`  | real K0 cyberpunk trace (3999 tok)        | 154 (60.2%) | 4080 |
| `masks/mask65_self.txt`  | real K0 cyberpunk trace (3999 tok)        | 166 (64.8%) | 3600 |
| `masks/mask60_family.txt`| 5 held-out domain traces (coffee/json x2/python x2), NO cyberpunk | 154 (60.2%) | 4080 |

Family traces used (`runs/ds4/20260711_podA_narrow_traces/*/route.csv`):
`a_coffee_full` (11961 rows), `b2_json_long_full` (8241), `b_json_full`
(1481), `c2_python_long_full` (9401), `c_python_full` (4681) — 35,760 rows
total.

## Run harness (`scripts/run_bake_arm.sh`)

`run_bake_arm.sh <arm> <mask_file|NONE> <run_n> [max_tokens=4000] [ctx=4096]`.
Env per mission spec: `DS4_CUDA_KEEP_MODEL_PAGES=1 DS4_CUDA_NO_DIRECT_IO=1
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 DS4_CUDA_NO_Q8_F16_CACHE=1
DS4_PACE=0 DS4_REAP_MASK_FILE=<mask>`, `--ssd-streaming
--ssd-streaming-cache-experts 400 --prefill-chunk 512 -c 4096`, port 8081.
Prompt = historical cyberpunk request (verbatim from
`runs/ds4/20260709_..._r01/request_measured.json`), temp 0, stream, stop on
`</html>` or repetition (reuses `scripts/stream_stop_guard.py`), pre-warm
(same prompt, 40 tok, non-stream) before the measured stream. Grading via
`scripts/functional_grade.py frontpage`. GPU serialized via
`flock /tmp/ds4-gpu.lock`. Launched as a transient `systemd-run` unit (plain
`nohup ... &` dies with the invoking WSL session on this box, per prior
HANDOFF lesson) so it survives the driving shell exiting. Kill discipline:
only `kill $(cat <run>/server.pid)`, never `pkill`.

## Run table (updated live)

| run | arm | keep | esito | stop_reason | grade | chars | t/s | RAM peak |
|---|---|---|---|---|---|---|---|---|
| self60_run1 | self | 60% | IN CORSO | — | — | — | — | — |

(filled in as runs complete; see `arm_<name>_run<n>/` for full artifacts:
`RUN_META.txt`, `server.pid`, `stream_live.txt`, `response.json`,
`STOP_REASON.txt`, `ram_log.txt`, `grade.json`.)
