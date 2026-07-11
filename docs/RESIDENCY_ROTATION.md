# Pin-keep / residency-rotation (patch 0031)

Date: 2026-07-11. Authoring: CPU-only Windows + WSL `/root/ds4` **read-only**.
Patch anchored on `ds4_cuda.cu` md5 `7d57f58d` (canonical v2.1 endpoint) + 0024
(md5 `c564ca7c`) → 0031 (md5 `430716f4`). Apply-check REAL, zero fuzz, LF-clean;
build + pod smoke **pending** (no CUDA toolchain locally).

---

## 0. The divergence 0031 resolves

`runs/ds4/20260711_pinning_divergence_audit/REPORT.md` (commit e5a1455) confirmed
that in ds4 today the **router-mask** (selection) and the **streaming expert
cache** (VRAM residency) are *completely decoupled*:

- The REAP mask (0011) writes only `-1e9` on the router bias `ffn_exp_probs_b`
  (selection). It never touches VRAM residency. Grepping `ds4_cuda.cu` for
  `rmass|pace_|reap_mask|keep_set|g_reap` returns **zero** hits — the cache is
  blind to the keep set.
- The per-token expert cache (`g_stream_expert_cache`,
  `cuda_stream_selected_cache_begin_compact_load`) is a generic keep-blind LRU.
  The hot-tier promote is frequency-based, also keep-blind.

Consequence: keep experts are served from the *same* cache as everybody else, so
lowering K changes **which** experts are selectable but not **how** they are
served. This is why low K never delivered speed locally — the H2D re-copy of the
same experts every token dominates (`docs/RESIDENT_HIT_FIX.md`, resident-hit ≈ 0),
and K does not touch that cost.

**0031 is the missing lever**: it binds the (post-mask) keep set to VRAM
residency, so a hot subset of keeps stays nailed in VRAM across tokens.

---

## 1. Mask vs residency — the load-bearing distinction

| | writes | perturbs trajectory? | outcome |
|---|---|---|---|
| rotate32 / 0015 | `g_reap_mask_pruned` + router bias (**selection**) | **yes** (re-ranks the mask) | AN-1 wide collapse (E-CAL) |
| demand-admission 0026 | `g_reap_mask_pruned` swap (**selection**) | yes (1 expert in/out of selectable set) | coverage change, quality-gated |
| **0031 pin-keep** | per-slot `pinned` flag in the cache (**residency**) | **no** | **bit-identical output**, only speed |

0031 never writes `g_reap_mask_pruned` nor the router bias. The top-k selection
is byte-for-byte what it was, so the generated token stream is identical whether
pinning is on, off, or rotating. Pinning is pure VRAM-residency management.

Why a *flag on existing slots* and not a separate slab (as the audit sketched):
the streaming expert cache is already the resident VRAM pool. Marking a slot
`pinned` = "the LRU victim selector may not evict this slot". Pinning therefore
adds **no** VRAM — it repartitions the already-allocated cache into a pinned
(eviction-immune) part and a rotating LRU part. This sidesteps the audit's
VRAM-pressure caveat: the cache's own `live_budget` already reserves VRAM for the
non-expert working set; `DS4_PACE_PIN_BUDGET_MB` is a **sub-budget of the cache**,
not new VRAM. We always leave ≥ `capacity/8 + 1` non-pinned slots so the LRU
always has a victim (a load never deadlocks).

---

## 2. How the hot keep set is found — cache-local demand EWMA (rmass analog)

0031 does **not** import the keep set from `ds4.c` (that would couple it to the
router path). It doesn't need to: under a **frozen static mask** the only experts
the router ever requests (`compact_ids` flowing into
`cuda_stream_selected_cache_begin_compact_load`) *are* the keep set. So the demand
stream visible inside the cache already **is** the post-mask keep set.

- Per slot: a `demand` EWMA of the request indicator (rmass-0020 analog),
  decayed once per decode call for that slot's layer and credited `+alpha` when
  the expert is requested-and-resident. `alpha = DS4_PACE_PIN_EWMA` (0.05).
- **Freeze**: after `DS4_PACE_PIN_WARMUP` decode expert-load calls (512, ≈ a
  handful of tokens over the layer stack), pin the highest-demand resident slots
  up to the budget. `pin_freeze` JSONL event records the count.
- **Rotation** (`DS4_PACE_PIN_ROTATE=1`): a 0026-style CUSUM on each non-pinned
  resident expert's demand gap vs the coldest pinned expert. When the CUSUM
  crosses `DS4_PACE_PIN_CUSUM_H` (and the `DS4_PACE_PIN_COOLDOWN` anti-thrash
  window has elapsed) and the candidate is genuinely hotter, **pin the candidate,
  unpin the coldest pinned**. K-pinned stays constant. Both experts are already
  resident, so the swap is a flag toggle — **no copy**. `pin_rotate` JSONL event
  records `{pinned_in, evicted_out}`.

The pinned count never touches the mask, so rotation cannot collapse the
trajectory (unlike rotate32, which rotated the *mask*).

---

## 3. Correctness invariants (declared)

1. **Residency, not selection.** No write to `g_reap_mask_pruned` or the router
   bias ⇒ selection unchanged ⇒ output **bit-identical** to the non-pinned
   streaming path. This is exactly what separates 0031 from rotate32/0015.
2. **Representation-neutral — pin serves 2-bit native, never q8/f16.** A pinned
   slot serves the exact bytes `load_slot` produced for that expert (native
   2-bit via the cold-lossless path when active). 0031 never promotes an expert
   to a q8/f16 representation. The q8/f16 *serving* path introduces a systematic
   precision crack (`runs/ds4/20260711_local_clean_lowK/BITEXACT.md`: ~54 diff
   lines vs ~21 baseline-noise, ≈2.6×), whereas 2-bit-native serving is clean
   (cache1024 hit≈98% within noise). To guarantee the clean path run with
   `DS4_CUDA_NO_Q8_F16_CACHE=1`. **0031 inherits, never overrides, the
   representation choice**; when `DS4_PACE_PIN=1` and that env is unset, 0031
   prints a one-shot stderr reminder.
3. **Off by default.** Every hook is a no-op unless `DS4_PACE_PIN=1`, so the
   default engine is byte-identical to post-0024.

---

## 4. Relationship to SPEX — prediction vs residency (they compose)

SPEX was born with the *same* principle as pin-keep: predict the next experts and
promote them to the faster memory tier. The two halves are separable:

- **SPEX = prediction** ("*what* to promote"). `DS4_SPEX_HIDDEN_GPU_PREFETCH=1`
  scores + topK the next-layer experts on device, reads the IDs back async, and
  the consumer `ds4_gpu_stream_expert_cache_seed_experts_async` seeds them into
  the streaming cache. Per `docs/SPEX_INTEGRATION_PLAN.md` (J30) the topK bridge
  is *ready in time*; its weakness is that seeding **re-pays the H2D cost every
  layer/token** (624 real seeds of 6 experts dominated the microtest), because
  seeded experts are then LRU-evicted before reuse. That is why the launcher
  keeps SPEX prefetch **off** — the prediction works, the residency doesn't stick.
- **0031 = residency** ("*keep* it nailed"). Eviction-immune VRAM residency for a
  hot set.

**Composition.** 0031's demand EWMA is credited by SPEX seeds too (the seed path
`cuda_stream_expert_cache_seed_one` calls the same `note_resident` hook), so a
SPEX-predicted expert accumulates demand and gets **pinned**. Once pinned, the
SPEX consumer's existing *skip-all-resident* filter (J30) stops re-seeding it —
which kills exactly the per-token seed-H2D cost that made SPEX prefetch unpayable.
So: **SPEX predicts → 0031 pins → seed cost amortized to ~zero.** No new coupling
code is required beyond the shared demand hook; the default driver is the
cache-local demand EWMA (rmass analog), which is always available even with SPEX
prefetch off.

Gap note: SPEX hidden-GPU-prefetch is real plumbing but **gated off** and not yet
a proven win; 0031 therefore keeps the cache-local demand EWMA as the **default**
pin/unpin driver, and treats SPEX seeds as an *additional* demand source rather
than a hard dependency.

---

## 5. Environment variables

| env | default | meaning |
|---|---|---|
| `DS4_PACE_PIN` | `0` | master gate; off ⇒ engine byte-identical to post-0024 |
| `DS4_PACE_PIN_BUDGET_MB` | `3500` | pinned sub-budget of the (already-allocated) cache; ≈520 experts at 6.75 MiB/expert, clamped to leave ≥ capacity/8+1 rotating slots |
| `DS4_PACE_PIN_ROTATE` | `0` | enable demand-driven residency rotation |
| `DS4_PACE_PIN_WARMUP` | `512` | decode expert-load calls before the pin freeze |
| `DS4_PACE_PIN_EWMA` | `0.05` | demand EWMA rate α |
| `DS4_PACE_PIN_CUSUM_K` | `0.05` | rotation CUSUM slack |
| `DS4_PACE_PIN_CUSUM_H` | `1.0` | rotation CUSUM threshold |
| `DS4_PACE_PIN_COOLDOWN` | `128` | anti-thrash calls between rotations |
| `DS4_PACE_PIN_LOG` | — | JSONL sink for `pin_freeze` / `pin_rotate` events |
| `DS4_CUDA_NO_Q8_F16_CACHE` | — | **set to 1** for bit-exact 2-bit-native serving (invariant 2) |

---

## 6. Smoke gate (binding, pending GPU)

Measured on a real 3060 build, K-static frozen mask, `DS4_CUDA_NO_Q8_F16_CACHE=1`:

1. **Speed**: t/s for `DS4_PACE_PIN=1` vs cache-LRU baseline (`DS4_PACE_PIN=0`)
   vs direct-RAM. Expect pin ≥ LRU and the resident-hit rate to rise / the
   `selected_direct_loads` per token to fall (`DS4_SPEX_STATS=1`).
2. **Bit-exactness**: greedy-argmax token stream with pin on must be
   **byte-identical** to pin off (invariant 1). Diff the two decode transcripts;
   any divergence is a bug, not a tuning knob.
3. **Rotation sanity**: with `DS4_PACE_PIN_ROTATE=1`, `pin_rotate` events fire at
   a bounded rate (cooldown honored), pinned count stays constant, and
   bit-exactness still holds.
4. **SPEX composition** (optional): with `DS4_SPEX_HIDDEN_GPU_PREFETCH=1`, confirm
   seed calls drop after the freeze (skip-all-resident kicks in on pinned
   experts) and t/s does not regress.
