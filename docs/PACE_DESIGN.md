# PACE — Perceptive Adaptive Control Engine

In-engine, self-calibrating controller for REAP-LOOP, living inside the ds4 /
DwarfStar engine (C + CUDA). PACE replaces the external Python sidecar
(`reap/reap_loop.py`) with an in-memory control loop compiled into the engine.

Branch: `pace/controller`. Patch: `patches/0014-pace-*`.

---

## 1. Why PACE (the problem with the sidecar)

The REAP-LOOP prototype is an external sidecar. Measured overhead:

* The engine writes a **routing trace CSV to disk** (`DS4_SPEX_TRACE_ROUTING`)
  every decode token, every layer.
* The engine writes an **S1 sensor CSV to disk** (`DS4_REAP_SENSOR_LOG`).
* The engine **polls a mask file by mtime** every decode token
  (`DS4_REAP_MASK_FILE`).
* The sidecar reads those files, decides, and rewrites the mask file.

On the 3060 the SSD is already the bottleneck (streamed experts: measured
`copy_ms_per_batch ≈ 59 ms`, `hit_rate ≈ 0.83`). Adding trace-write +
mask-poll I/O contends for the same SSD → this is what made REAP slow. The
parameters were also **static** (fixed keep-K, fixed breath cadence).

PACE fixes all three:

1. **Reads routing IN MEMORY** — no trace CSV.
2. **Applies the mask DIRECTLY** — no mask-file polling.
3. **ADAPTS parameters live** — EWMA-smoothed reactive control + annealing.

PACE reuses, unchanged, the two pieces that already work:

* **Actuator** = patch 0011 `ds4_reap_mask_apply()` — writes `-1e9` into the
  router bias (`ffn_exp_probs_b`) on CPU selection *and* on GPU via device-range
  UPSERT (`ds4_gpu_model_range_update`). ID-space-stable, reversible.
* **WRAP** = patch 0013 `ds4_reap_prefetch_working_set()` — host-side threaded
  bulk page-in of the kept working set. Already dodges the blocking-sync (no
  CUDA-stream interaction). Env renamed `DS4_REAP_WRAP` → `DS4_REAP_WRAP`.

---

## 2. Hook points (all verified in ds4.c, branch ds4-flash-unified)

| What | Where | Notes |
|---|---|---|
| Per-decode-token driver tick | `metal_graph_eval_token_raw_swa_streaming` → `ds4_reap_mask_poll(model,weights)` @ **20431** | Called once per decode token. `model`, `weights`, `g`, `pos`, `token` all in scope. This becomes PACE's tick. |
| Gate-mass accumulation (learn) | `metal_graph_spex_note_selected(g, il, selected_ids, n_selected)` @ **14791** | Common to BOTH router paths (CPU-router @ 15387, GPU-router @ 16934/16952). `selected_ids[]` is a CPU array (free); weights need a `g->router_weights` readback so we accumulate with unit weight (fidelity: prototype falls back to w=1.0 too). |
| n-gram token feed | the `token` arg of `metal_graph_eval_token_raw_swa_streaming` @ **20411** | Each decode-input token is the previously-generated token re-entering. One push per decode token, same site as the tick. No detok, no disk. |
| Selection check (violations / hit) | same `metal_graph_spex_note_selected` — `selected_ids[]` vs pruned set | in-memory equivalent of the sidecar's `check_rows`. |
| Actuator | `ds4_reap_mask_apply(model,weights)` (0011) | unchanged. |
| WRAP page-in | `ds4_reap_prefetch_working_set(model,weights)` (0013) | unchanged; env rename only. |

Shape (runtime): FLASH = 43 layers × 256 experts; PRO = 61 × 384. Static
maxima `DS4_MAX_LAYER=61`, `DS4_MAX_EXPERT=384`, `DS4_MAX_EXPERT_USED=6`. PACE
sizes its accumulator `[DS4_MAX_LAYER][DS4_MAX_EXPERT]` and iterates to the
runtime `DS4_N_LAYER`/`DS4_N_EXPERT`.

---

## 3. The blocking-sync constraint (honest)

`ds4_gpu_end_commands()` = `cudaDeviceSynchronize()` (ds4_cuda.cu:2693),
called **once per decode token** (ds4.c:20558 streaming / 20603 non-streaming).
There is **no compute stream and no async token pipeline**. Consequences:

* At the point PACE runs (the mask-poll tick, before the layer loop; and the
  gate-mass hook, mid graph-encode), the host is between blocking syncs. PACE's
  control work (accumulate a few floats, scan a short n-gram window, branch) is
  **pure CPU time, ~µs, not stolen GPU overlap** (there is no overlap to steal).
  With a 120-token window and n=3 the scan is a few thousand comparisons.
* The **expensive** operations are the actuator (`cudaMemcpy` H2D per masked
  layer inside `ds4_reap_mask_apply`) and WRAP (page-in). PACE keeps those
  **rare** (only when the mask actually changes — FNV signature guard, already
  in 0013) and WRAP stays detached/host-side.
* **[OPEN]** PACE cannot hide the actuator's H2D copies behind compute because
  there is no compute stream to overlap with (see `SPEX_INTEGRATION_PLAN`
  premise). The mitigation is frequency, not concurrency: apply-on-change only.

So PACE-controller-logic is clean under the blocking-sync. What the blocking-sync
blocks is a *predictive/pipelined* controller — which we don't need, because the
only live signal that works is reactive (below).

---

## 4. The control law (reactive, NOT predictive)

Measured finding (moe-aggressive-commit ledger, control C; confirmed here in
`reap_loop.py` NgramSensor/S1Sensor docstrings): **the router sensor is dead** —
router lag-k / S1 pruned-mass does NOT anticipate degeneration. The **only**
signal that anticipates is the **textual n-gram repetition detector**. PACE
therefore drives breathing from n-gram, keeps S1 as an *optional* slow-drift
diagnostic, never as the trigger.

### State machine (mirrors the sidecar `LoopDriver`)

```
WARMUP ──(warmup tokens)──▶ DESCENT ──(direct)──▶ HOLD ⇄ BREATH
```

* **WARMUP** (`pace.warmup` tokens): accumulate gate-mass, learn nothing yet.
* **DESCENT**: apply the final keep-K directly (measured best = "direct").
* **HOLD**: mask active. Breath fires on:
  * n-gram score `≥ drift_threshold` (with hysteresis: re-arm only after score
    `≤ release_threshold` AND `hysteresis_tokens` since last breath), OR
  * clock cap `breath_every` (auto-tuned, see annealing).
* **BREATH** (`breath_len` tokens): widen to `breath_keep` (or fully unmask),
  optionally re-learn (decayed merge), then return to HOLD at final keep-K.

### Signals (all in-memory)

1. **n-gram degeneration** (quality floor): over the last `ngram_window` decode
   tokens, `score = 1 − uniq_ngrams/total_ngrams` (n=`ngram_n`, on token-ids).
   EWMA-smoothed (short raw windows are noisy).
2. **hit-rate / cold-miss** (efficiency): `in_keep_rate` = fraction of selected
   experts that were in the kept set (violations = selected-but-pruned). This is
   the in-memory `check_rows`. High hit → safe to tighten.
3. **S1 gate-mass on pruned** (slow drift, diagnostic only): EWMA of
   `pruned_mass/total_mass` from `g->router_probs`. Kept OFF by default
   (`DS4_PACE_S1=0`) — it's a monitor, not a trigger.

### Adaptation

* **Tighten keep-K** when confident: n-gram EWMA low (`≤ tighten_lo`) AND hit-rate
  high (`≥ hit_hi`) for `stable_tokens` → `keep_k -= keep_step` down to `keep_min`.
  (Faster: fewer experts routed → fewer cold misses.)
* **Widen** when degrading: n-gram EWMA high → breath (widen + optional re-learn),
  and raise `keep_k += keep_step` up to `keep_max` after the breath.
* **Auto-tune breath cadence** on the degeneration rate: track breaths-per-1k
  tokens; if breaths are frequent, shorten `breath_every` (breathe sooner);
  if rare, lengthen it. Bounds `[breath_every_min, breath_every_max]`.

### Stability

* **EWMA** on every signal (`alpha_ngram`, `alpha_hit`, `alpha_s1`) — short
  windows are noisy.
* **Annealing**: early exploration → late exploitation. An `anneal` factor
  starts high (frequent breaths, conservative keep-K) and decays with token
  count toward exploit (rarer breaths, tighter keep-K). Implemented as a
  multiplier on `breath_every` (grows with anneal progress) and on the tighten
  gate (unlocks tightening only after `anneal_warm` tokens).

### Emergent HW adaptation (no a-priori HW profile)

We do NOT profile the GPU. On slow HW a cold-miss is expensive → it shows up as
low measured t/s and as more time-per-token; the controller sees the *same*
n-gram/hit signals but, because tightening is gated on *stable high hit-rate*,
slow HW (where misses hurt and hit-rate is harder to keep high under a tight
mask) naturally settles at a **more conservative keep-K**; fast HW tolerates a
tighter mask and tightens further. The knob that makes this emergent is that
tighten is hit-rate-gated, and hit-rate degrades faster on slow HW under an
aggressive mask. **[to validate on 2 HW tiers — only 3060 available now].**

### Reactive by design

PACE corrects **after** a sliver of damage (n-gram is a trailing detector over a
short window). We keep windows SHORT so the sliver is small. This is the honest
tradeoff vs a (dead) predictive sensor.

---

## 5. Parameters (env, `DS4_PACE_*`), with sidecar-measured defaults

| Env | Default | Meaning |
|---|---|---|
| `DS4_PACE` | off | master enable |
| `DS4_PACE_WARMUP` | 150 | warmup decode tokens before learning. Practical validated sweet spot is **W≈50** (short warmup wins on completion time at equal quality — CLAIMS_CURRENT §SESSION-LEARNING); the cockpit defaults to 50. |
| `DS4_PACE_KEEP` | 40 | final keep-K (measured curve point) |
| `DS4_PACE_KEEP_MIN` | 24 | tighten floor |
| `DS4_PACE_KEEP_MAX` | 64 | widen ceiling |
| `DS4_PACE_KEEP_STEP` | 4 | tighten/widen step |
| `DS4_PACE_NGRAM_N` | 3 | n-gram order (token-ids) |
| `DS4_PACE_NGRAM_WINDOW` | 120 | sliding window (tokens) |
| `DS4_PACE_DRIFT` | 0.35 | breath trigger (n-gram EWMA) |
| `DS4_PACE_RELEASE` | 0.15 | re-arm low-water |
| `DS4_PACE_HYST` | 200 | min tokens between breaths |
| `DS4_PACE_BREATH_EVERY` | 400 | clock breath cap (auto-tuned) |
| `DS4_PACE_BREATH_LEN` | 80 | breath length |
| `DS4_PACE_BREATH_KEEP` | 64 | keep-K during breath (0 = unmask) |
| `DS4_PACE_RELEARN` | 1 | re-learn mask from breath window |
| `DS4_PACE_RELEARN_DECAY` | 0.3 | old-stats weight on re-learn |
| `DS4_PACE_ALPHA_NGRAM` | 0.10 | EWMA alpha, n-gram |
| `DS4_PACE_ALPHA_HIT` | 0.05 | EWMA alpha, hit-rate |
| `DS4_PACE_TIGHTEN_LO` | 0.10 | n-gram EWMA below → allow tighten |
| `DS4_PACE_HIT_HI` | 0.90 | hit-rate above → allow tighten |
| `DS4_PACE_STABLE` | 120 | stable tokens before a tighten step |
| `DS4_PACE_ANNEAL_WARM` | 300 | tokens before tightening unlocks |
| `DS4_PACE_S1` | 0 | enable S1 diagnostic (monitor only) |
| `DS4_PACE_WRAP` | 0 | run WRAP bulk page-in on mask change. **Off by default** — on the practical 3060 config prefetch *slows* t/s (0.82 vs 1.27, see CLAIMS_CURRENT §PREFETCH). Enable only when a probe proves a deeply SSD-bound regime. |
| `DS4_PACE_PREFILL_APPLY` | 1 | learn the first mask from prompt routing and apply it after prefill, before decode token generation. This is a dynamic prompt mask, not a static domain mask. |
| `DS4_PACE_PREFILL_WAIT_WRAP` | 1 | when WRAP is enabled, wait for the prefill-derived working set page-in before starting decode. Local 3060 smoke: 6.07 GiB touched in 445 ms, generation 2.83 t/s on a 19/24-token probe. |
| `DS4_PACE_LOG` | "" | optional JSONL event log (off SSD hot path if unset) |

Backward-compat: `DS4_REAP_WRAP` still honored as an alias for
`DS4_PACE_WRAP`/`DS4_REAP_WRAP`.

---

## 6. Incremental implementation plan (compile+test each step)

* **0014a** — module skeleton: `ds4_pace` static state, env parse, per-token
  tick wired into `ds4_reap_mask_poll`, gate-mass accumulator wired into
  `metal_graph_spex_note_selected`, n-gram ring fed from the decode `token`.
  No actuation yet (learn + observe, log signals to stderr under a debug env).
  Gate: builds `make ds4 CUDA_ARCH=sm_86`, runs, prints signals.
* **0014b** — WARMUP→DESCENT→HOLD with the 0011 actuator applied in-engine
  (learn keep-K from accumulator, apply mask directly). Gate: 0 violations in a
  static-keep smoke (equivalent to sidecar phase A).
* **0014c** — BREATH from n-gram EWMA + hysteresis + clock; re-learn. Gate:
  breaths fire on injected repetition, quality recovers.
* **0014d** — adaptation (tighten/widen), annealing, cadence auto-tune. Gate:
  keep-K moves in the right direction on a warm run.
* **0014e** — WRAP env rename + apply-on-change guard; warm smoke with measure
  discipline (warm first, measure warm, no SSD trace).

---

## 7. Measurement discipline

* Always `wsl -d Ubuntu-24.04`.
* WARM first (cold decode can stall >60 s/token on cold cache) — discard the
  cold run, measure the warm run.
* No routing trace CSV during a timed PACE run (that's the whole point).
* Compare against the same-instrumentation baseline (static keep-K, PACE off).
* Report t/s + hit-rate + breaths + final keep-K. Never fabricate numbers.

### 2026-07-08 local implementation note

Commit `/root/ds4` `c8dd670 pace-dynamic-prefill-mask` moved PACE from
decode-only learning to prompt-aware learning:

* reset any stale PACE/REAP mask at the start of a new prefill;
* accumulate selected expert ids during prompt routing;
* apply the learned keep-K mask at `tok=0`, immediately after prefill;
* optionally wait for WRAP/fattorino before decode.

This is **not dynamic expert compression**. It does not create a smaller
low-bpw copy of evicted experts and it does not recompress experts on eviction.
That remains a separate open tiering project: active experts in fast form,
evicted experts in a colder compressed form, with REAP/PACE controlling K and
residency live.
