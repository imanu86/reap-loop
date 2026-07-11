# Tiered-hysteresis dynamic residency (patch 0033)

Date: 2026-07-11. Authoring: CPU-only Windows + WSL `/root/ds4` **read-only**.
Patch anchored on `ds4_cuda.cu` md5 `7d57f58d` (canonical v2.1 endpoint) + 0024
(md5 `c564ca7c`) + 0031 (md5 `430716f4`) → 0033 (md5 `95af4397`). Apply-check
REAL, zero fuzz, LF-clean; build + pod smoke **pending** (no CUDA toolchain
locally).

**0033 is the user's objective.** 0031 pinned a *static* hot set at one warmup
freeze. 0033 makes the set **live**: it self-seeds from the first tokens' routing
and then continuously promotes/evicts experts across three residency tiers by a
hysteresis-gated score. The point is to fix the naive-LRU thrash — a bigger LRU
cache measured *worse* because it promotes on every miss and pays the management
cost. Hysteresis keeps the VRAM set small, stable and genuinely hot.

---

## 0. Why dynamic, and why hysteresis

0031's premise ("under a frozen static mask the requested experts *are* the keep
set") holds only for a static mask. The user's endgame is a **dynamic** working
set: the hot experts change with the prompt/phase, so a one-shot freeze is the
wrong shape. But the obvious fix — an LRU/frequency cache that promotes any
missed expert — **thrashes**: a single fluke request pulls a cold expert into the
fast tier, evicting a genuinely hot one, and the churn itself costs more than it
saves (`docs/RESIDENT_HIT_FIX.md`, and the K12/K23 "big cache is worse" finding).

0033's answer is **hysteresis**: an expert only climbs to a faster tier after
**sustained** demand (X, then X+Y knocks), and only falls after sustained
absence (below a margin). A fluke never promotes; a brief lull never demotes. The
VRAM set therefore stays small and stable.

---

## 1. The three tiers

Each cache slot carries a `tier` (0/1/2) beside 0031's `pinned` flag. The tiers
are physical residency classes, actuated entirely through the existing streaming
expert cache — 0033 adds **no** VRAM:

| tier | name | physical meaning | actuation |
|---|---|---|---|
| 2 | **VRAM** | hot, resident, read-fast | `pinned = 1` ⇒ the LRU victim selector skips it (eviction-immune), exactly 0031's mechanism |
| 1 | **RAM** | warm, cached, LRU-evictable | normal resident slot (not pinned, not preferred victim) |
| 0 | **SSD** | cold, **BLOCKED** | `pinned = 0` **and** preferred LRU victim ⇒ cycles out of VRAM fast, served by direct host/SSD load on the next request |

The VRAM tier is a `pinned`-flag sub-partition of the *already-allocated* cache,
so pinning adds no memory — it only forbids eviction. We always leave
`≥ capacity/8 + 1` non-VRAM slots so the LRU always has a victim (a load never
deadlocks) — the same reserve 0031 uses.

The SSD "block" is implemented in `cuda_stream_expert_cache_lru_slot`: among
non-pinned candidates it now prefers the **lower tier first** (tier 0 before tier
1), then the lower age. With tiering off every slot is tier 0, so this reduces to
0031's pure min-age selection — byte-identical when `DS4_PACE_TIER` is unset.

---

## 2. The dynamic seed (replaces 0031's freeze)

A per-slot `knock` counter is a decayed request count — the sustained-demand
proxy. During warmup 0033 only **accumulates** knocks (nothing is pinned, so the
engine behaves like the baseline LRU). After `DS4_PACE_TIER_WARMUP` decode
expert-load calls (default 512 ≈ the first ~12 tokens over a ~40-layer stack, i.e.
the user's "first 10–20 tokens" window), `cuda_tier_seed` classifies the observed
routing:

- every resident slot with `knock ≥ X` → **RAM** (tier 1), else → **SSD** (tier 0);
- then the hottest RAM slots with `knock ≥ X+Y` are promoted to **VRAM** (tier 2,
  pinned) up to the budget.

`tier_seed` JSONL event records the seeded VRAM count. After the seed the
continuous loop takes over.

---

## 3. The continuous loop — knock, promote, evict, re-enter

**Knock (score).** `cuda_tier_note_resident` credits `+DS4_PACE_TIER_KNOCK` (1.0)
whenever an expert is requested-and-resident (decode HIT/LOAD, or a SPEX seed —
same composition hook as 0031). `cuda_tier_begin_call` decays each layer's knocks
by `DS4_PACE_TIER_DECAY` (0.98) once per decode call, so `knock` tracks *recent
sustained* demand: a one-off request decays away, a repeatedly-routed expert
accumulates.

**Promotion (in `note_resident`, one step per call).** Sustained knocking climbs
the ladder, gated:

- **SSD → RAM** only after `knock ≥ X`  (`DS4_PACE_TIER_X`, default 3);
- **RAM → VRAM** only after `knock ≥ X + Y`  (`+ DS4_PACE_TIER_Y`, default 5 ⇒ 8).

So a cold expert needs X sustained knocks just to be cached, and X+Y to be nailed
in VRAM — a single fluke request cannot promote it. The VRAM tier is capped at
`DS4_PACE_TIER_VRAM_SLOTS` (**394**, the real 12 GB fit) and holds the
top-by-knock experts within budget. When the budget is full, a hotter RAM
candidate **displaces** the coldest VRAM slot — but only if it is at least
`DS4_PACE_TIER_HYST` hotter and `DS4_PACE_TIER_COOLDOWN` calls have passed since
the last swap. Both experts are already resident, so the pin swap is a **flag
toggle, no copy**. `tier_promote` / `tier_swap` JSONL events record it.

**Eviction (in `begin_call`, one step per call).** After decay, a cooled slot
descends with a hysteresis margin so the boundary does not oscillate:

- **VRAM → RAM** when `knock < (X+Y) − HYST` (frees a VRAM budget slot);
- **RAM → SSD** when `knock < X − HYST` (blocked; now a preferred LRU victim).

`tier_demote` JSONL event records the VRAM→RAM step.

**Re-entry.** `knock` **decays** rather than resetting on demotion, so a blocked
(tier-0) slot that resumes knocking climbs back — SSD → RAM → VRAM — without
starting from zero. Blocking is reversible; sustained demand always wins the slot
back.

---

## 4. Correctness invariants (declared, load-bearing)

Identical in spirit to 0031 — this is the whole reason the controller can be this
aggressive:

1. **Residency, not selection.** 0033 writes only the per-slot `tier` / `pinned`
   / `knock` management state. It never writes `g_reap_mask_pruned` nor the
   router bias, so top-k **selection is unchanged** ⇒ the generated token stream
   is **bit-identical** whether tiering is off, seeding, promoting or evicting.
   Which slot is resident/evicted only decides HIT vs H2D re-copy of the **same
   bytes**, never *which* expert is selected. This is exactly what separates
   0031/0033 from rotate32/0015 (which rotated the *mask* and collapsed).
2. **Representation-neutral — 2-bit native, never q8/f16.** Every tier serves the
   exact bytes `load_slot` produced (native 2-bit via the cold-lossless path).
   0033 never promotes an expert to a q8/f16 form. For the guaranteed bit-exact
   path (avoiding the q8/f16 serving crack, `BITEXACT.md` ≈2.6×) run with
   `DS4_CUDA_NO_Q8_F16_CACHE=1`; 0033 **inherits, never overrides**, the
   representation choice and prints a one-shot stderr reminder if it is unset.
3. **Off by default.** Every hook is a no-op unless `DS4_PACE_TIER=1`, so the
   default engine is byte-identical to post-0031 (and, with `DS4_PACE_PIN=0` too,
   to post-0024). `DS4_PACE_TIER` **supersedes** 0031's static freeze — both
   drive the same `pinned` flag, so enable one; a one-shot stderr note fires if
   both are set.

**Known approximation (shared with 0031).** `knock`/`tier` live on the cache
**slot**, not the expert, so an expert that is *physically* evicted and later
reloaded into a different slot loses its history (the slot's state belongs to
whatever now occupies it). The dynamic seed and the decay self-correct, and
because selection is untouched this is a residency **efficiency** wart, never a
correctness one. Per-expert (layer,expert)→score tracking that survives eviction
is the natural next step if the smoke shows history loss hurting the VRAM-set
stability.

---

## 5. Environment variables

| env | default | meaning |
|---|---|---|
| `DS4_PACE_TIER` | `0` | master gate; off ⇒ engine byte-identical to post-0031 |
| `DS4_PACE_TIER_WARMUP` | `512` | decode expert-load calls before the dynamic seed (~first ~12 tokens) |
| `DS4_PACE_TIER_X` | `3` | knocks for **SSD → RAM** |
| `DS4_PACE_TIER_Y` | `5` | extra knocks for **RAM → VRAM** (VRAM needs X+Y = 8) |
| `DS4_PACE_TIER_HYST` | `1.0` | demotion / displacement hysteresis margin (anti-oscillation) |
| `DS4_PACE_TIER_VRAM_SLOTS` | `394` | VRAM tier budget in slots (the real 12 GB fit), clamped to leave ≥ capacity/8+1 rotating slots |
| `DS4_PACE_TIER_DECAY` | `0.98` | per-call knock decay (enables re-entry; prevents permanent lock) |
| `DS4_PACE_TIER_KNOCK` | `1.0` | knock increment per request |
| `DS4_PACE_TIER_COOLDOWN` | `64` | anti-thrash calls between VRAM displacements |
| `DS4_PACE_TIER_LOG` | — | JSONL sink for `tier_seed` / `tier_promote` / `tier_demote` / `tier_swap` |
| `DS4_CUDA_NO_Q8_F16_CACHE` | — | **set to 1** for bit-exact 2-bit-native serving (invariant 2) |

---

## 6. Where the loop lives (call graph)

- `cuda_tier_begin_call(cache, layer)` — once per decode expert-load call
  (`cuda_stream_selected_cache_begin_compact_load`, beside 0031's
  `cuda_pin_begin_call`): decay this layer's knocks, run the **demotion** half,
  fire the seed at warmup.
- `cuda_tier_note_resident(cache, slot)` — per resident request/seed (both
  `seed_one` sites + the `begin_compact_load` hit/load site, beside 0031's
  `cuda_pin_note_resident`): credit the knock, run the **promotion** half.
- `cuda_stream_expert_cache_lru_slot(cache)` — made tier-aware (SSD before RAM,
  skip VRAM); reduces to 0031's min-age selection when tiering is off.

`tier_vram_count` stays consistent with the tier-2 population by construction:
only budget-promotion (+1), VRAM→RAM demotion (−1), and displacement (net 0)
change it, and tier-2 slots are pinned so they are never repurposed by
`load_slot`.

---

## 7. Smoke gate (binding, pending GPU)

Measured on a real build, dynamic mask, `DS4_CUDA_NO_Q8_F16_CACHE=1`:

1. **Bit-exactness.** Greedy-argmax token stream with `DS4_PACE_TIER=1` must be
   **byte-identical** to `DS4_PACE_TIER=0` (invariant 1). Any divergence is a bug,
   not a tuning knob.
2. **VRAM set stays stable / no thrash.** With `DS4_PACE_TIER_LOG`, the pinned
   VRAM set converges after the seed and churns at a **bounded** rate
   (`tier_swap`/`tier_demote` honoring the cooldown), pinned count ≤ budget, and
   the naive-LRU thrash is gone (resident-hit up, `selected_direct_loads`/token
   down vs `DS4_PACE_TIER=0`).
3. **Honest t/s.** t/s for tier vs cache-LRU baseline vs direct-RAM. Expectation
   is **hardware-gated**: on a modest **3060** (12 GB) the fitting hot set already
   near-fills VRAM, so the headroom is small and t/s should be **≥** LRU, not a
   leap; the real win is on a **≥24 GB** card where the stable hot set fits with
   room for the rotating RAM tier and the thrash it removes was larger. Report the
   number for what it is per device — no extrapolation across memory classes.
