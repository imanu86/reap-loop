# Smoke 0031 — pin-keep / residency-rotation — GPU gate on podD

**Date:** 2026-07-11. **Pod:** RunPod `7qgalm9sasqnr7` (`ds4-podD-0022-smoke`), COMMUNITY
**RTX 3090 24 GB**, image cu1290 / torch `2.8.0+cu129`, nvcc **12.9**. Gate-check PASS:
`nvidia-smi -L` -> RTX 3090; `torch.cuda.is_available()=True device_count=1`.
Pod was left warm+RUNNING by the rewind agent — **used, not stopped/terminated** (still
RUNNING at end of this work). This is the **first real compile of 0031** (no local nvcc)
and the correctness gate that transfers across GPUs. **Speed (t/s) is NOT measured here** —
the pod is RAM-hot on a 24 GB card; t/s does not transfer to the 3060. This measures
**mechanism + correctness + the I/O-pattern delta**.

## Headline — all four gates

| Gate | Verdict |
|---|---|
| **1 BUILD** | **PASS** — clean nvcc compile, fresh ELF |
| **2 ENGAGEMENT** | **PASS (engaged, real — not a false pass)** — pin_freeze fires, residency provably changes; resident-hit **rises** and H2D reloads **fall** in the high-pressure regime (the 3060 target); net-negative at low pressure (regime/param dependent) |
| **3 BIT-EXACT** | **PASS** — pin ON / ROTATE produce **byte-identical** token streams to pin OFF (residency != selection confirmed) |
| **4 ROTATION** | **PASS** — pin_rotate rate-limited (delta-call = cooldown exactly), pinned count constant, bit-identical |

**0031 is mechanically sound and READY for local 3060 speed measurement**, with a tuning
caveat (below): the mandate's default pin params are mistuned for the W50 harness.

---

## Build provenance (GATE 1)

Build tree `/root/ds4_pin` = copy of the rewind agent's `/root/ds4` (untouched original),
`ds4_cuda.cu` at the canonical-v2.1 anchor. Patches shipped as committed **LF** blobs
(git `show`, CRLF trap avoided; blob hashes `1005bbb7` / `984fab8e` verified on both ends).

    ds4_cuda.cu md5 chain (git apply --check clean, zero fuzz):
      base (canonical v2.1)         7d57f58d
      + 0024 (resident-hit fix)     c564ca7c   (expected)
      + 0031 (pin-keep)             430716f4   (expected endpoint)
    make cuda CUDA_ARCH=sm_86 -j : BUILD_RC=0, 115 s, 0 warnings / 0 errors
    fresh ELF  ds4 = md5 ee497e9dfb3e23ae40b03ac92cdbd997 (10.9 MB, ELF 64-bit)

**GATE 1 = PASS.** 0031 compiles+links cleanly on the first real nvcc build; the endpoint
`ds4_cuda.cu` md5 is `430716f4` exactly as the patch header declares.

---

## Run harness

Existing two-phase **W50 static K23** in-engine PACE (frozen mask), greedy `--temp 0
--nothink`, `--ssd-streaming`, **DS4_CUDA_NO_Q8_F16_CACHE=1** (2-bit-native serving, no
q8/f16 crack), **DS4_SPEX_STATS=1**, `DS4_EXPERT_TIERING=observe`, `n=300`, ctx 8192.
The frozen mask makes selection identical by construction, so any output diff is purely a
pinning bug. Token IDs captured per position via `DS4_SPEX_TRACE_TOKENS` (0028, present in
the tree) -> exact token-level bit-exact comparison.

**Cache size is the pressure knob.** On a 24 GB card the streaming cache would hold the
whole keep working-set (~920 expert-slots = K23 x ~40 MoE layers), so pinning could not
engage. To reproduce the 12 GB pressure regime the cache was capped with
`--ssd-streaming-cache-experts` at 512 (moderate) and 256 (heavy, 3060-like).

### Determinism prerequisite for the bit-exact gate
The **coffee** prompt is **non-deterministic run-to-run on this pod**: two pin-OFF runs
(a, a2) do **not** match (they even diverge in whether the leading fenced code block is
emitted; hit 0.6946 vs 0.8162). This is a GPU float-non-associativity / knife-edge-margin
property of the streaming path — **independent of 0031** (both runs pin OFF) — so coffee
cannot support a byte-identical gate. The **cyberpunk** prompt **is deterministic**:
cy1 == cy2 (both pin OFF) are **byte-identical** (tokens, gen text, and every SPEX stat).
All bit-exact gating below uses cyberpunk.

---

## Results (cyberpunk, deterministic; cache 512 unless noted; n=300, 12900 batches)

| run | pin | rot | warmup | budgetMB | pinned | hit_rate | misses | copy_calls | direct | bit-exact vs pin-OFF |
|---|---|---|---|---|---|---|---|---|---|---|
| **cy1 (A)** | 0 | - | - | - | - | **0.7858** | 16635 | 61773 | 3956 | baseline |
| cy2 (A') | 0 | - | - | - | - | 0.7858 | 16635 | 61773 | 3956 | **IDENTICAL** (determinism control) |
| **cyb (B)** | 1 | 0 | 512 | 3500 | 447 | 0.4481 | 42860 | 140448 | 3956 | **IDENTICAL** |
| **cyc (C)** | 1 | 1 | 512 | 3500 | 447 | 0.4945 | 39252 | 129624 | 3956 | **IDENTICAL** (96 rotations) |
| cybt (tuned) | 1 | 0 | 2200 | 1000 | 148 | 0.7149 | 22137 | 78279 | 3956 | **IDENTICAL** |
| **a256** | 0 | - | - | - | - | **0.0201** | 75840 | 239388 | 3956 | baseline @256 |
| **b256** (tuned) | 1 | 0 | 2200 | 800 | 118 | **0.1341** | 67016 | 212916 | 3956 | **IDENTICAL** |

---

## GATE 2 — ENGAGEMENT (+ I/O-pattern delta, the transferable finding)

- **pin_freeze IS emitted** in every pinned run (pinned=447 budget_slots=447 capacity=512
  per_expert_mib=6.75 at cache 512; pinned=148/118 for the tuned runs). Budget clamped
  exactly to leave capacity/8+1 LRU-free slots, as designed.
- **Residency provably changes** — the cache stats move massively vs the pin-OFF baseline
  (misses, copy_calls swing 2-3x). This is **not** the "stats invariate => false pass" case;
  engagement is real.
- **Direction of the I/O-pattern delta is regime-dependent** (this is what transfers, even
  though t/s does not):
  - **cache 512 (moderate pressure):** LRU alone is already near-optimal (hit **0.79**), so
    pinning is **net-negative** (hit -> 0.45 at default params, -> 0.71 tuned): freezing
    447/512 (or 148) slots eviction-immune removes LRU's freedom when LRU was already keeping
    the hot set.
  - **cache 256 (heavy pressure, 3060-like):** LRU **collapses** (hit **0.0201**, near-total
    thrash). Pinning **raises resident-hit 0.0201 -> 0.1341 (x6.7, +567 %)** and **cuts H2D
    reloads**: misses 75840 -> 67016 (-11.6 %), copy_calls 239388 -> 212916 (-11.1 %).
- `direct_loads` is a **constant 3956** across all runs — that is the 0024 first-touch floor,
  not the pin-sensitive metric. The metric pinning actually moves is **cache_misses /
  copy_calls** (H2D re-copies), which fall with pinning under pressure.

**Verdict: engaged, and resident-hit rises / reloads fall in the intended high-pressure
regime** (cache << working set, i.e. the 3060). GATE 2 satisfied for the target regime;
counterproductive where the cache is already large enough — see tuning caveat.

## GATE 3 — BIT-EXACT

With the deterministic cyberpunk prompt, **every pinned run is byte-identical to its pin-OFF
baseline**: cyb (B) == cy1 (A), cyc (C) == cy1 (A) (through 96 rotations), cybt == cy1,
and b256 == a256. Selection is unchanged -> **0031 invariant 1 (residency != selection)
confirmed**. Engaged **and** identical together => mechanically sound (no corruption).
*(Coffee non-determinism is a pod/prompt property, not a 0031 divergence — see above.)*

## GATE 4 — ROTATION

cyc (DS4_PACE_PIN_ROTATE=1): **96 pin_rotate events**, spaced **exactly delta-call = 128**
(= DS4_PACE_PIN_COOLDOWN) — perfectly rate-limited, ~ the 12900/128 ~ 100 cooldown cap.
Each event swaps one pin in / one out => **pinned count constant at 447**. Output
**bit-identical to A**. Rotation is sound: bounded rate, constant residency budget, zero
trajectory impact.

---

## Mechanism findings / tuning caveats for the 3060

1. **Default pin warmup is mistuned for W50.** `DS4_PACE_PIN_WARMUP=512` calls ~ **token 12**
   (43 layer-loads/token), which freezes the pinned set **before** the K23 mask stabilizes at
   **token 50** (DS4_PACE_WARMUP=50). The freeze then captures the pre-mask transient hot set,
   not the true frozen keep set. **Fix:** set pin warmup **> mask_warmup x n_layers** (~2200
   here) so the pinned set is the real keep set (cybt/b256 use 2200 and behave far better:
   0.45 -> 0.71 at cache 512).
2. **Budget must leave LRU headroom.** `DS4_PACE_PIN_BUDGET_MB=3500` -> 447/512 slots pinned,
   only 65 LRU-free — starves the non-pinned tail. Size the pinned budget to a fraction of the
   actual cache so plenty of LRU slots remain.
3. **Pinning only helps when LRU thrashes** (cache << working set). That is exactly the 3060
   regime (12 GB fits ~258-407 slots vs ~920 working set) — where cache-256 here shows the
   win. On the 3060, tune warmup (post-mask-freeze) and budget (to the real free-VRAM cache
   size) and confirm the hit-rate/t-s win directly.

## Is 0031 ready for the local 3060 speed run?

**YES.** BUILD clean (430716f4). ENGAGED and provably residency-affecting (not a false pass).
BIT-EXACT (residency != selection proven, incl. under rotation). ROTATION rate-limited and
non-perturbing. The I/O-pattern delta points the right way in the pressure regime the 3060
lives in (resident-hit up, H2D reloads down at cache 256). Take it to the 3060 for the t/s
measurement **with tuned params**: DS4_PACE_PIN_WARMUP ~ 2200 (freeze after the mask), and
DS4_PACE_PIN_BUDGET_MB sized to leave LRU headroom against the 12 GB cache.

---

### Artifacts (this dir)
`STATS_SUMMARY.txt`, per-run `*_status.txt` / `*_run.log.stats` (SPEX stats) /
`*_events.jsonl` (pin_freeze/pin_rotate) / `*_tokens.csv` (per-pos token IDs) /
`*_gen.txt` (generated text); harness `common_env.sh` / `run_pin.sh`; `lcp.py` (token-LCP).
Naming: `cy*` = cyberpunk (deterministic, gate runs), `a/a2/b/c` = coffee (non-deterministic,
determinism-caveat evidence), `*256` = cache-256 heavy-pressure regime.
