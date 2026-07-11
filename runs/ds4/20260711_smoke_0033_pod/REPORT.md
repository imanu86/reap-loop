# Smoke 0033 -- tiered-hysteresis dynamic residency -- GPU gate on pod

**Date:** 2026-07-11. **Pod:** RunPod `u49vytysl0xyqi` (`ds4-smoke-0033-tier`), COMMUNITY
**RTX 3090 24 GB** (driver 580.65.06), image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
**nvcc 12.8**, 112 vCPU / 220 GB RAM. Gate-check PASS: nvidia-smi -> RTX 3090; nvcc 12.8 present.

**Pod provenance.** podD (`7qgalm9sasqnr7`) and all four other stopped pods **could not resume**
-- RunPod returned "not enough free GPUs on the host machine" for every one. Per the playbook
fallback this is a **fresh DEPLOY** (empty container disk, volumeInGb:0); the model was
re-fetched from HuggingFace (`antirez/deepseek-v4-gguf`, Xet, **86.72 GB @ ~116 MB/s, 746 s**)
-- not R2, not a resumed volume. **Pod STOPPED at end** (created by me -> stop, not terminate).

First real compile of 0033 (CPU-only authoring). This gate **transfers across GPUs**. **t/s is
NOT the deliverable** -- the pod is RAM-hot on a 24 GB card; t/s does not transfer to the 3060.
This measures **mechanism + correctness + the I/O-pattern delta**.

## Headline -- all gates

| Gate | Verdict |
|---|---|
| **1 BUILD** | **PASS** -- canonical v2.1 -> 0024 -> 0031 -> 0033 byte-exact, ds4_cuda.cu md5 95af4397, clean nvcc sm_86 |
| **2 BIT-EXACT** | **PASS** -- TIER=1 token stream byte-identical to TIER=0 (tokens + gen); residency != selection |
| **3 CONVERGENCE / NO-THRASH** | **PASS** -- VRAM set converges and holds; swaps cooldown-bounded (min gap 64, never violated); naive-LRU thrash gone |
| **4 BLOCK / RE-ENTRY** | **PASS** -- X / X+Y hysteresis blocks flukes; 61 VRAM re-entries |
| **5 ENGAGEMENT** | **PASS in 3060 regime** -- cache-256 hit x18.3 (0.0213 -> 0.3897), H2D re-copies -36 pct; net-negative at cache-512 (regime-dependent) |

**0033 is mechanically sound and READY for the local 3060 t/s measurement.**

---

## Build provenance (GATE 1)

Base github.com/antirez/ds4@80ebbc3; patches shipped as committed **LF** blobs (CRLF trap
avoided, `git apply --check` clean, **zero fuzz**). Every md5 checkpoint matched exactly:

    ds4.c        base 80ebbc3                 bf9a0b6f
      + canonical/ (21 patches, sorted)       1db4f799
      + 0027 + 0028 (token-id trace)          62ed2e71   <- DS4_SPEX_TRACE_TOKENS
    ds4_cuda.cu  base 80ebbc3                 5d8debb3
      + canonical/ (21)  = v2.1 endpoint      7d57f58d
      + 0024 (resident-hit fix)               c564ca7c
      + 0031 (pin-keep)                       430716f4
      + 0033 (tiered-hysteresis)              95af4397   <- mandate endpoint, EXACT
    make cuda CUDA_ARCH=sm_86 -j112 : BUILD_RC=0, 61 s, 0 warnings / 0 errors
    fresh ELF  ds4 md5 = 72e8a76e1287 (10.5 MB)

**GATE 1 = PASS.** Clean first nvcc build; ds4_cuda.cu endpoint md5 is 95af4397 as declared.

---

## Run harness

Frozen **K12** in-engine mask (DS4_PACE=1, warmup 50 -> freeze: KEEP=KEEP_MIN=12,
BREATH_EVERY=999999 RELEARN=0 ROTATE=0), greedy --temp 0 --nothink, --ssd-streaming,
**DS4_CUDA_NO_Q8_F16_CACHE=1** (2-bit-native, no q8/f16 crack), DS4_SPEX_STATS=1, n=300, ctx 8192.
Tier at **mandate defaults**: WARMUP=512 X=3 Y=5 HYST=1.0 VRAM_SLOTS=394 DECAY=0.98 COOLDOWN=64,
events -> DS4_PACE_TIER_LOG. Token IDs per position via DS4_SPEX_TRACE_TOKENS (0028) -> exact
token-level comparison. The frozen mask makes selection identical by construction, so any output
diff would be purely a residency bug.

**Prompt = cyberpunk (deterministic).** The 0031 smoke found the *coffee* prompt is
non-deterministic run-to-run on this streaming path (GPU float non-associativity), so it cannot
support a byte-identical gate; cyberpunk is deterministic. **Cache is the pressure knob:** 512
(moderate -- K12 working set ~480 experts nearly fits) and 256 (heavy, 3060-like). A determinism
control (two TIER=0 runs) precedes the gate.

---

## Results

### GATE 2 -- BIT-EXACT (residency != selection)

| run | tier | cache | tokens.csv md5 | gen.txt md5 | vs baseline |
|---|---|---|---|---|---|
| **t0_c512** | 0 | 512 | cbc9980901d3 | 5d36caf1e4ce | baseline |
| t0b_c512 | 0 | 512 | cbc9980901d3 | 5d36caf1e4ce | **IDENTICAL** (determinism control) |
| **t1_c512** | **1** | 512 | cbc9980901d3 | 5d36caf1e4ce | **IDENTICAL** |
| **t0_c256** | 0 | 256 | cbc9980901d3 | 5d36caf1e4ce | baseline |
| **t1_c256** | **1** | 256 | cbc9980901d3 | 5d36caf1e4ce | **IDENTICAL** |

**Every run byte-identical** -- determinism control clean, TIER=1 == TIER=0 at both cache sizes
(tokens AND generated text), even across cache sizes. Residency (seed/promote/demote/swap)
provably **never touches selection** -> **invariant 1 confirmed**. **GATE 2 = PASS.**

### GATE 3 -- CONVERGENCE / NO-THRASH (the user's proof)

Tier events (tier_seed/promote/demote/swap, 12900 decode calls):

| run | seed vram | budget | promote | demote | swap | swap min-gap | VRAM set seed->final |
|---|---|---|---|---|---|---|---|
| **t1_c512** | 121 @call512 | 394 | 469 | 196 | 175 | **64** (=cooldown) | 121 -> **394 flat** |
| **t1_c256** | 223 @call512 | 223 | 6 | 6 | 193 | **64** (=cooldown) | 223 -> **223 flat** |

- **VRAM set CONVERGES.** t1_c512 seeds small (121 -- seed fires at call 512 ~ token 12, before
  the K12 mask freezes at token 50, the transient 0031 warned of), then the **continuous loop
  grows it to the full budget 394 by ~token 40 and holds it dead-flat** (VRAM=394 at every
  decile). This is exactly 0033's advantage over 0031's static freeze -- 0031 would be stuck at
  the ~121 pre-mask set; 0033's loop repairs it. t1_c256 seeds at clamped budget 223, flat 223.
- **Churn BOUNDED.** Swaps rate-limited to ~19/decile; **min swap gap = 64 = COOLDOWN, never
  violated**; totals (175, 193) <= calls/cooldown cap (~193). Promotions front-loaded (fill), then
  settle to a low steady state. **No thrash** -- swap rate capped by cooldown, not demand noise.
- vs naive LRU (GATE 5): at cache 256 LRU thrashed to **2 pct hit**; tiered set stable -> 39 pct.

**GATE 3 = PASS** -- converges + cooldown-bounded churn = hysteresis-beats-LRU proof.

### GATE 4 -- BLOCK / RE-ENTRY

- **Block.** Promotion gated at knock >= X (SSD->RAM) and >= X+Y=8 (RAM->VRAM); one fluke
  (knock +1, decay x0.98/call) never reaches threshold -> cold experts held out of the fast tier.
- **Re-entry.** knock **decays**, does not reset on demotion -> a cooled slot that resumes
  knocking climbs back. **61 VRAM re-entries** in t1_c512 (demoted 2->1 then re-promoted 1->2).
  Blocking is reversible. **GATE 4 = PASS.**

### GATE 5 -- ENGAGEMENT (I/O-pattern delta -- the transferable finding)

SPEX cache stats (TIER on vs off), same prompt/mask, n=300:

| run | cache | hit_rate | cache_misses | copy_calls | copied MiB | wall s |
|---|---|---|---|---|---|---|
| t0_c512 | 512 | **0.8109** | 14681 | 55911 | 125800 | 98 |
| t1_c512 | 512 | 0.7473 | 19622 | 70734 | 159152 | 117 |
| **t0_c256** | 256 | **0.0213** | 75747 | 239109 | 537995 | 367 |
| **t1_c256** | 256 | **0.3897** | 47238 | 153582 | 345560 | 241 |

- **cache 256 (heavy pressure = 3060 regime):** LRU **collapses** (hit 2.1 pct). Tiering raises
  resident-hit **0.0213 -> 0.3897 (x18.3)** and cuts H2D re-copies: misses -37.6 pct, copy_calls
  -35.8 pct, bytes copied -35.8 pct; wall 367 -> 241 s (-34 pct, I/O proxy; t/s does not transfer).
  Markedly larger than 0031's static pin here (0031: 0.0201->0.1341 x6.7) -- the continuous loop
  holds a hotter set than a single freeze.
- **cache 512 (moderate):** LRU already near-optimal (0.81), so pinning is net-negative (0.75) --
  same regime-dependence 0031 found: when the cache already fits the working set, freezing slots
  eviction-immune only removes LRU's freedom. Expected; not a defect.
- direct_loads is a **constant 3956** across all runs -- the 0024 first-touch floor, NOT the
  tier-sensitive metric. Tiering moves **cache_misses / copy_calls** (H2D re-copies), down sharply
  under pressure.

**GATE 5 = PASS for the target regime** (cache << working set = the 3060).

---

## 8-line verdict

1. **BUILD OK?** YES -- v2.1 -> 0024 -> 0031 -> 0033 byte-exact, ds4_cuda.cu = 95af4397, make cuda sm_86 RC=0.
2. **BIT-EXACT (on == off)?** YES -- TIER=1 byte-identical to TIER=0 (tokens + gen), all 5 runs md5-identical, det-control clean -> residency != selection.
3. **VRAM CONVERGES + churn LIMITED (no thrash, hysteresis)?** YES -- converges (121->394 flat; 223 flat), swaps cooldown-bounded (min gap 64), naive-LRU thrash removed.
4. **BLOCK / RE-ENTRY works?** YES -- X / X+Y=8 blocks flukes; 61 VRAM re-entries (knock decays, no reset).
5. **ENGAGED?** YES in the 3060 regime -- cache-256 hit x18.3, H2D re-copies -36 pct; net-negative at cache-512 (LRU already optimal).
6. **Beats naive LRU?** YES -- under pressure LRU thrashed to 2 pct; tiering held a stable hot set at 39 pct.
7. **READY for local t/s on the 3060?** YES -- mechanism sound, correctness proven; take to the 3060 with cache sized to real free VRAM.
8. **Caveat:** t/s NOT measured (24 GB RAM-hot pod, does not transfer); frozen-mask smoke validates the transferable mechanism; dynamic-mask quality is next.

---

### Artifacts (this dir)
Per-run *_status.txt / *_run.log.stats (SPEX stats) / *_events.jsonl (tier events) / *_tokens.csv
(per-pos token IDs) / *_gen.txt (generated text); harness common_env0033.sh / run_tier.sh /
run_all.sh / run_all.log; build.log; analyze_tier.py. Runs: t0*=TIER off, t1*=TIER on,
_c512/_c256=cache size, t0b=determinism control, sanity=n=8 engine probe.
