# Smoke test — patch 0024 (CUDA streaming-expert-cache resident-hit fix)

Date: 2026-07-11. Pod: `pod2-r2-redeploy` (RunPod `i7dk94f0y05iji`), left RUNNING.

## TL;DR verdict

**Inconclusive for the core claim — wrong hardware class.** This pod is an **NVIDIA
GeForce RTX 3090 (24 GiB)**, not the ~12 GiB card patch 0024 targets. Bug A (the
reserve/clamp that starves the cache) only bites when free VRAM after the resident model
tensors is *below* the clamped reserve — which never happens on a 24 GiB card. So the
baseline is **already healthy** here (cache fully allocated, `cap=256` on every layer),
and there is no broken baseline for the fix to visibly repair.

- 0024 **applies cleanly** to the canonical chain (`git apply --check` OK as patch #24,
  no 0022 needed) and **builds** under CUDA `sm_86`.
- The `strtod` float parse **works**: `RESERVE_GB=0.5` is accepted and the cache
  allocates (old `strtoull` would have swallowed the fractional value).
- On this card 0024 is a **non-regression**: `hit_rate`, per-layer `cap`, copy volume and
  `direct_loads` are statistically identical to baseline (deltas within run-to-run noise).
- The **hit_rate RISE** that 0024 is designed to produce is **NOT demonstrated** — it
  needs a ~12 GiB pod. **Recommendation: re-run this A/B on a 12 GiB card (e.g. RTX 3060)
  to actually validate the fix.**

## Adoption gate (from RESIDENT_HIT_FIX.md §4)

| Gate | Result on this pod |
|---|---|
| (i) hit_rate rises ~0 -> clearly >0 | **N/A / not demonstrable.** Baseline is not broken on 24 GiB (hit≈0.021, not ~0). Both arms ≈0.021. |
| (ii) cap nonzero for multiple layers in fix arm | **PASS** (trivially) — `cap=256` for every sampled layer (0,1,2,21,35,37,39). But also true in baseline. |
| (iii) greedy output byte-identical baseline vs fix | **FAIL as stated — but NOT a 0024 regression.** See control: baseline-vs-baseline (same binary, same env) is *also* not byte-identical -> inherent async-streaming nondeterminism. |

## Hit-rate before/after

| arm | binary | RESERVE_GB | hit_rate | selected_experts | cache_hits | direct_loads | gen t/s |
|---|---|---:|---:|---:|---:|---:|---:|
| BASELINE | ds4-baseline | 1 | **0.0212** | 83597 | 1634 | 6455 | 1.11 |
| CONTROL (baseline #2, same binary) | ds4-baseline | 1 | 0.0212 | 83595 | 1634 | 6453 | 1.12 |
| FIX (0024) | ds4-fixed | 0.5 | **0.0211** | 83592 | 1629 | 6450 | 0.99 |

Baseline->fix delta hit_rate = -0.0001; baseline->baseline delta (pure noise) = 0.0000 with
selected_experts jitter of +/-2-5. The fix delta is **inside** the run-to-run noise band.

## Per-layer cap before/after

`cap=256` for **every** sampled layer in **all three** runs (layers 0,1,2,21,35,37,39).
No layer at `cap=0`. Contrast the 12 GiB failure trace in `docs/RESIDENT_HIT_FIX.md` §1,
where only layer 0 had `cap>0` and layers 1-42 were `cap=0`. On this 24 GiB card the
cache allocates fully in **both** arms, so cap is not a differentiator here.

## Byte-diff quality gate (generated text, ds4: diag lines stripped)

Extracted files: `gen_baseline.txt`, `gen_baseline2.txt`, `gen_fixed.txt`.

- BASELINE vs FIX: **DIFFER** — diverges at `body { font-family: ... }`
  (baseline `system-ui ... #faf7f2`; fix `Arial ... #f5f0e8`, centered vs flexbox nav).
- **CONTROL** BASELINE vs BASELINE#2 (identical binary + env): **DIFFER** — diverges at
  the *same* point (`background:#faf7f2` vs `#f5f0e8`, different hover/hero hex).
- BASELINE#2 vs FIX: **DIFFER**.

**Interpretation:** greedy (`--temp 0`) decode is **not deterministic run-to-run** in this
async-streaming build — two identical baseline runs already diverge at the same ~65-token
mark. This is the async top-k handoff / streaming-prefetch path (patches 0015/0016/0021/
0026) tipping argmax near-ties differently each run, **not** an effect of 0024. Gate (iii)
as literally worded is **not achievable on this build/card regardless of 0024**, and the
fix-vs-baseline divergence must **not** be read as a 0024-induced token change.

## Anomaly

`hit_rate ~= 0.021` even with `cap=256` and the cache allocated: 258 experts/token are
selected against a 256-slot cap with `--ssd-streaming-cold` (no preload), so
`evictions ~= misses` (~75k each) — the cache **thrashes**. That is the E-LAT
"fit-in-cache" capacity lever (raise cap / preload), a **separate** issue from the
resident-collapse bug 0024 fixes. On a 24 GiB card cap could be raised well above 256, but
the bench recipe fixed `--ssd-streaming-cache-experts 256` for comparability.

## Build notes

- **Precondition fix (not a patch/script bug):** `/root/ds4-canon` working tree already
  had the full chain applied as *uncommitted* `git apply` diffs (`ds4_cuda.cu` @ md5
  `7d57f58d`), so `canon_build.sh`'s own `git apply` step failed apply-check #1
  (already-applied hunks). Reset to pristine `80ebbc3` with
  `git checkout -- . && git clean -qfdx` (-> `ds4_cuda.cu` md5 `5d8debb3`), after which the
  script re-applied the whole chain cleanly. `ds4.c` md5 = `62ed2e71` (expected). No
  hand-editing of source.
- BASELINE: 23-patch chain, MAKE OK. `ds4` sha256 `71341c62...`.
- FIX: 0024 applied as patch **#24**, `git apply --check` **OK**, MAKE OK.
  `ds4` sha256 `0cd2b3bf...`.

## Exact commands / env

Common env: `DS4_SPEX_STATS=1 DS4_EXPERT_TIERING=observe DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23`

```
# BASELINE
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1 /root/ds4-baseline \
  -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 256 -c 4096 --nothink --temp 0 -n 300 \
  --prompt-file /root/coffee_prompt.txt > /root/smoke0024_baseline.log 2>&1

# FIX (only diff: binary + RESERVE_GB=0.5)
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.5 /root/ds4-fixed  ...  > /root/smoke0024_fixed.log 2>&1

# CONTROL (determinism check): baseline binary + env re-run -> smoke0024_baseline2.log
```

Build (fix arm): `/root/canon_build_0024.sh` = `canon_build.sh` with 0024 appended to
`CHAIN` after 0028. CLI flags from the mission spec were accepted as-is (no adjustment).

## Files in this dir

- `smoke0024_baseline.log`, `smoke0024_fixed.log` — the two required arms (raw).
- `smoke0024_baseline2.log` — control re-run (determinism check).
- `gen_baseline.txt`, `gen_baseline2.txt`, `gen_fixed.txt` — extracted generated text.
- `SUMMARY.md` — this file.

Not git-committed (standing instruction: a central process commits; this run only writes
result files).
