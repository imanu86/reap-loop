# S1-guided rewind — design study (0022 candidate)

> **Status: DESIGN STUDY, no patch yet.** The real patch `0022` is authored only
> after the M1b stopper verdict and the pace-series canonization. This document
> fixes the mechanism, the env surface, the validation plan, and an
> honest feasibility verdict against the live engine tree.
>
> Engine evidence below is anchored to the local live tree
> `/root/ds4/ds4.c`, 1314218 bytes, md5 `771a39a861e9512fed2fc4528780e080`
> (the same tree patches `0020`/`0021` target — see `patches/README.md`
> "Stato apply"). Line numbers are from that file; **read-only** study, no
> build/server was run.

---

## 0. TL;DR

* **The reactive stopper is not enough.** When an n-gram/line stopper fires, the
  loop tokens are already in the KV context and in the client stream; the
  poisoned context wins. Measured: hot-widening the mask *after* the damage does
  **not** recover (finding **C1**), and re-starting via a fresh prompt restarts
  from zero (`rollback-via-prompt` 3/3 fail → "serve rewind KV",
  moe `docs/INVENTIONS_LEDGER.md`). So correction must **remove** the poisoned
  tokens from the KV, not just change the mask going forward.
* **S1 gives both the trigger and the coordinate.** The EWMA slope of the
  router's pruned-mass rises ~200 tokens **before** the collapse (CLAIM-011).
  Its onset marks *where* the trajectory started to deviate → the rewind target.
* **The scale is prevention → correction → airbag.**
  1. **Prevention** = slope-S1 → rotate/widen (patch `0020`, already smoked).
  2. **Correction** (this doc, `0022`) = if the drift escapes prevention, rewind
     the KV to `pos_onset − margin`, restore the compressor frontier + PACE
     accumulators, refresh/widen the mask, and **resume without re-prefill**
     (dodges the J44 freeze-point knife-edge: context up to the healthy point
     stays byte-identical in the KV).
  3. **Airbag** = the n-gram stopper (M1b) as last resort and the A/B benchmark.
* **Engine feasibility verdict: CLEAR and LOW-RISK.** The two hard primitives a
  KV rewind needs — a compressor-frontier snapshot/restore and a
  position-counter rewind — **already exist and are exercised every token by MTP
  speculation** (`spec_frontier_snapshot`/`spec_frontier_restore`,
  `ds4_session_rewind`). `0022` is a controller-driven *macro* over those tested
  primitives, plus a stream-retraction event. See §1.

---

## 1. Engine feasibility (live-tree study)

### 1.1 Position is the single source of truth

Decode is position-indexed. In the streaming SWA path
`metal_graph_eval_token_raw_swa_streaming` (ds4.c:21970) the raw KV write slot is
computed **directly from the position**:

```c
const uint32_t raw_row = pos % g->raw_cap;                 // ds4.c:21985
const uint32_t n_raw   = metal_graph_raw_span_for_batch(g, pos, 1);
```

and the session driver `ds4_session_eval_internal` (ds4.c:29851) passes
`pos = s->checkpoint.len`, pushing the token **after** the eval:

```c
metal_graph_eval_token_raw_swa(&s->graph, ..., token,
                               (uint32_t)s->checkpoint.len, s->logits);  // ds4.c:29915
token_vec_push(&s->checkpoint, token);                                   // ds4.c:29919
```

So "the position" is just `checkpoint.len`, and the raw ring is addressed by
`pos % raw_cap`. Lowering `checkpoint.len` and re-driving `eval` re-writes the
raw rows for the reverted positions — no explicit clear needed.

### 1.2 A rewind primitive already exists (but is incomplete on its own)

`ds4_session_rewind` (ds4.c:30560, public in `ds4.h:270`) already truncates the
committed token vector:

```c
void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
}
```

This rewinds the **logical position** and invalidates the MTP draft, but it does
**not** touch the compressor frontier or the compressed-row counters. On its own
it is only correct if followed by a re-prefill. For an in-memory (no re-prefill)
rewind it must be paired with a frontier restore (§1.4).

### 1.3 "A counter is enough" — the append-only cache model

The engine's whole KV design is append-only + position/counter addressing, so
reverting future rows never needs a copy. The MTP path documents the invariant
explicitly (ds4.c:30041):

```
/* ... after verification, rows beyond the accepted prefix must become
 * invisible. We do not copy/rollback the cache body because the next draft
 * attempt will overwrite future slots. A counter is enough. */
```

The same holds for the **compressed** KV. `compressor_decode_one` (ds4.c:8842)
emits one compressed (pooled) row every `compress_ratio` tokens and keeps a small
rolling frontier (`state_kv`/`state_score`); the emitted rows are appended and
counted by `layer_n_comp[il]` (advanced at ds4.c:17822 / 20672, whole-prefix,
capped at `layer_comp_cap`). Compression ratios per layer for FLASH
(`ds4_expected_layer_compress_ratio`, ds4.c:630): layers 0-1 dense (ratio 0,
pure raw SWA); even layers ≥2 → ratio 4 (with an indexer sub-cache); odd
layers ≥2 → ratio 128. So a compressed row for a block is *append-only*, and
lowering `layer_n_comp[il]` hides the poison blocks exactly like lowering
`checkpoint.len` hides poison raw rows. On resume the corrected tokens re-emit
those rows.

### 1.4 The frontier snapshot/restore already exists

The **only** mutable state a counter cannot rebuild is the rolling compressor
frontier (the partial-block accumulator) and the row counters. The engine
already snapshots and restores exactly that, generically, for speculative
decoding:

```c
typedef struct {                        // ds4.c:26755
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    uint32_t mtp_n_raw;
} ds4_spec_frontier;

static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s);  // ds4.c:26766
static bool spec_frontier_restore (ds4_spec_frontier *f, ds4_session *s);  // ds4.c:26800
```

`spec_frontier_snapshot` copies, per compressed layer, `layer_attn_state_kv`,
`layer_attn_state_score` (and for ratio-4 layers `layer_index_state_kv/score`)
into dedicated snapshot tensors and saves the `n_comp`/`n_index_comp`/`mtp_n_raw`
counters; `spec_frontier_restore` copies them back. This is the checkpoint
primitive a rewind needs, verbatim. It does **not** snapshot the raw ring (see
§1.5) — consistent with §1.3.

**Consequence:** a full KV rewind to position `p` is the composition
```
spec_frontier_restore(snapshot_taken_at_p)   // frontier + n_comp back to p
ds4_session_rewind(s, p)                      // checkpoint.len = p
```
followed by normal `ds4_session_eval` from `p`. Both halves already exist and
run every token under MTP.

### 1.5 SWA window and the clean-rewind bound

The logical sliding-window is the model's `sliding_window`
(`deepseek4.attention.sliding_window`, read at ds4.c:3909; default shape
`n_swa = 128`, ds4.c:196/308). The **physical** raw ring `raw_cap` is larger
(the ctx=4096 runs report `raw_kv_rows=768`); the code comments the slack
explicitly (ds4.c:14328):

```
/* This graph uses a raw ring larger than the 128-token logical SWA window, so
 * writing speculative future rows does not evict visible raw rows. */
```

The visible raw window for a token at position `q` is `[q − raw_window + 1, q]`
(`metal_graph_raw_start_for_span`, ds4.c:14314: `first_raw_pos % raw_cap`). So
after rewinding to `p` and resuming, the rows `p` needs are still **physically
present** iff we did not overwrite them, i.e. iff the rewind depth stays inside
the ring slack:

> **Clean-rewind bound:** `current_pos − p  <  raw_cap − raw_window`
> (≈ `768 − 128 = 640` tokens for the ctx=4096 config).

This is **more generous** than the naive "must land inside the 128-token
window": the compressed rows cover the whole prefix and are counter-rewindable
(§1.3–1.4), so the real limit is raw-ring slack, not the SWA window. A ~200-token
onset lead (CLAIM-011) fits comfortably. **This must still be validated
empirically** — see the bit-equality smoke in §5, Risk R1.

### 1.6 What else must be reset besides the KV

| State | Where | Rewind action |
|---|---|---|
| Logical position | `s->checkpoint.len` (ds4.c:29915) | `ds4_session_rewind(s, p)` (exists) |
| Compressor frontier + `n_comp`/`n_index_comp` | `layer_attn_state_kv/score`, `layer_index_state_kv/score` (ds4.c:11264-11296) | `spec_frontier_restore(snap_p)` (exists) |
| Raw SWA ring rows | `layer_raw_cache[il]` ring, `pos % raw_cap` | none — re-written on resume, valid within the §1.5 bound |
| `mtp_n_raw` / MTP draft | `g->mtp_n_raw`, `s->mtp_draft_valid` | in `spec_frontier` + `ds4_session_rewind` (exists) |
| Sampler RNG | `sample_top_p_min_p(..., uint64_t *rng)` (ds4.c:25401) | greedy (`sample_argmax`, ds4.c:25252) has **no state** → nothing to do for our temp=0 runs; for temp>0 either snapshot `*rng` at onset or deliberately re-seed to force divergence out of the loop |
| PACE n-gram ring + EWMAs | `g_pace` ngram ring, `ema_ngram`, `ema_hit`, `ema_s1`, `s1_ring[]` (patch 0020) | rewind the accumulators to their onset checkpoint (host-side, cheap — §3) |
| PACE gate-mass accumulators | `g_pace.mass[]`, `g_pace.rmass[]`, `bmass[]` | snapshot/restore the rows touched since onset (host-side) |
| HTTP stream position | server layer (poison tokens already emitted) | emit a `rewind` retraction event; client trims (§4) |

### 1.7 Per-token driver + hook points (unchanged from PACE)

`ds4_reap_mask_poll` (ds4.c:21993) and `ds4_pace_tick` (ds4.c:21995, decode
tokens only) already run once per decode token inside the streaming eval, with
`model`, `weights`, `g`, `pos`, `token` in scope. The S1 monitor + slope already
land there via patch `0020` (`ds4_pace_s1_update`, `ds4_pace_s1_trigger_due`).
The rewind actuator hangs off the same tick — it is a new escalation branch in
`ds4_pace_tick`, not a new hook.

### 1.8 Feasibility verdict

**CLEAR / LOW-RISK on the engine.** Every hard primitive already exists and is
tested per-token by MTP: frontier snapshot/restore (ds4.c:26766/26800),
position-counter rewind (ds4.c:30560), append-only counter addressing
(ds4.c:30041). The novel work is orchestration, not new cache machinery:
(a) take **one** frontier snapshot at the controller-chosen onset instead of
per-token; (b) a public `ds4_session_rewind_kv()` that pairs the counter rewind
with the frontier restore; (c) rewind the PACE accumulators; (d) the server-side
stream retraction. The residual risk is empirical, not architectural — the
§1.5 bound and the frontier sufficiency must be confirmed by a bit-equality
smoke (Risk R1).

---

## 2. The scale: prevention → correction → airbag

Rationale, with the findings that force this ordering:

* **C1 — the poisoned context wins.** Hot-switching to a wide mask *after* the
  loop tokens are in context does not recover (moe `docs/INVENTIONS_LEDGER.md`:
  "contesto avvelenato vince; rollback-via-prompt riparte da zero 3/3 → serve
  rewind KV"). Confirmed operationally by the breath experiments: "Breath
  K0→K23 and breath K96→K23 both produced zero useful post-return tokens ...
  degeneration happened before or during breath" (`EXPERIMENTS_LEDGER.md`, note
  J42). ⇒ correction must **delete** poison from the KV, not out-run it.
* **J44 — re-prefill is a knife-edge.** Re-showing "write the full document" at a
  mid-file cut induces a document restart when the freeze lands inside a CSS
  declaration (`EXPERIMENTS_LEDGER.md`, note J44; CLAIM-006 reopened as
  knife-edge). J44 item (3) already prescribes the cure this doc formalizes:
  "rewind or trim a small window before the first degeneration marker ... and
  continue [on the same KV]". Rewinding the KV in place keeps the healthy prefix
  **byte-identical** and never re-prefills, so it steps over the knife-edge.
* **CLAIM-011 — S1 slope is the only early signal, and it is a coordinate.** The
  absolute pruned-mass level is dead (chronic ~0.75), but its EWMA slope rises
  ~+0.058 over ~200 tokens before collapse (0.722→0.781 local; 0.73→0.81
  historical K91) — the *only* measured pre-collapse riser (`CLAIMS_CURRENT.md`
  "FEEDBACK slope-S1"; `DS4_EXPERIMENT_LEDGER_20260710.md` CLAIM-011, OPEN). Its
  onset is *where* the trajectory left the healthy manifold → the rewind target.
* **The loop is cheap to catch late but expensive to leave running.** In
  `runs/ds4/20260710_w50_rotate32_k23_cache256_html4000/ANALYSIS.md`: soft
  degradation from ~tok1300, exact repetition-lock from ~tok1357, ~2600 tokens
  (65% of budget) burned; a stopper "would have fired around tok~1400, saving
  the ~2600 tokens", and "enables a stop → rewind → retry pattern instead of
  relying on breath". This doc is that rewind → retry, targeted by S1.

**Ladder (one delta over SOTA at a time):**

```
prevention                 correction                         airbag
──────────                 ──────────                         ──────
S1 slope ≥ thr             S1 slope keeps rising OR           n-gram/line
  → rotate (0020)            n-gram airbag fires anyway         triple-repeat
  → widen  (0020)            → REWIND to pos_onset−margin       → STOP
                             → restore frontier + PACE          (M1b, last
(prevents; stays the         → widen/refresh mask                resort AND
 primary strategy)           → resume, NO re-prefill             the A/B
                           (this doc, 0022)                      benchmark)
```

Prevention stays primary. Rewind is the second line **because** the clean-rewind
bound (§1.5) is finite: if prevention fails and the deviation ages past the ring
slack, a clean in-memory rewind is no longer possible and only the airbag
remains. That is itself an argument for acting early — it does not weaken
prevention, it depends on it.

---

## 3. Mechanism

### 3.1 Onset marker + checkpoint

Reuse the patch-0020 S1 machinery. Split the single slope threshold into two:

* **ARM** (`DS4_PACE_REWIND_ARM_THR`, default = `DS4_PACE_S1_SLOPE_THR` =
  3e-4/tok): when the EWMA-S1 slope first crosses ARM while in `PACE_HOLD`, mark
  `pos_onset = tok` and take **one** checkpoint:
  * `spec_frontier_snapshot(&g_pace.rewind_frontier, s)` — the compressor
    frontier + `n_comp`/`n_index_comp`/`mtp_n_raw` (needs a dedicated snapshot
    tensor set, see below);
  * a copy of `s->logits` at onset (so resume produces the same first token);
  * the PACE accumulator checkpoint: `ema_ngram`, `ema_hit`, `ema_s1`,
    `s1_ring[]` + head/len, the n-gram ring + head/len, and the *touched rows* of
    `mass[]`/`rmass[]` since onset (or the full arrays — they are host doubles,
    a few MB; simplest is a full copy).
* **FIRE** (`DS4_PACE_S1_SLOPE_THR` / n-gram airbag): the existing trigger.
  Re-arming refreshes the onset checkpoint forward (a later, still-healthy onset
  supersedes an older one) as long as no FIRE has happened; this keeps the
  rewind shallow.

**Checkpoint memory cost.** Dominated by the per-layer frontier tensors. FLASH =
43 layers; layers 0-1 dense; ~21 ratio-128 layers (`128·head_dim·2` floats each)
+ ~20 ratio-4 layers (`16·head_dim·2` + `16·idx_head_dim·2` floats each). With
`head_dim≈128` this is ≈ 2.7 MB + ≈ 0.5 MB ≈ **~3 MB**, plus logits (~0.5 MB)
and the PACE host arrays (single-digit MB if copied whole). **Total single-digit
MB** — negligible against the multi-GB KV cache. The engine already allocates
**two** such frontier buffer sets for MTP (`spec_*` and `spec_prefix1_*`,
ds4.c:11266-11296, under `enable_mtp`); the rewind checkpoint adds **one** more
dedicated set (`spec_rewind_*`), allocated only when `DS4_PACE_REWIND=1` so
MTP-off runs pay nothing.

### 3.2 Escalation and the rewind step

Inside `ds4_pace_tick`, after the existing clock/sensor/S1 branches:

1. **rotate** (0020, `DS4_PACE_S1_ACTION=rotate`): forced K-constant rotate on
   the S1 slope. Prevention.
2. **widen** if the slope keeps rising for `N` tokens after a rotate (rides the
   standard breath path). Prevention.
3. **rewind** if the airbag (n-gram ≥ drift) fires *despite* 1–2, and a valid,
   in-bound onset checkpoint exists:
   * compute `target = clamp(pos_onset − DS4_PACE_REWIND_MARGIN,
     current_pos − (raw_cap − raw_window) + guard, current_pos)`;
   * if `target` would require crossing the §1.5 bound (onset too old) → **do not
     rewind**, emit `rewind_skip{reason:"out_of_window"}`, fall through to the
     airbag STOP;
   * else: `spec_frontier_restore(&g_pace.rewind_frontier, s)` →
     `ds4_session_rewind(s, target)` → restore the PACE accumulator checkpoint →
     set the resume mask to `DS4_PACE_REWIND_KEEP` (widen: `keep_max` or a
     coverage-based width) and refresh it from the healthy-segment stats only →
     emit a `rewind` event → resume decode from `target` with the onset logits.

Note the ladder never re-prefills. Everything up to `target` stays in the KV.

### 3.3 Relearn from the healthy segment only

The resume mask must be rebuilt from the statistics of the **healthy** segment,
not the poisoned tail. Because the PACE accumulators are rewound too (§3.1),
`mass[]`/`rmass[]` already reflect only `[0, target]` after restore, so a
standard `ds4_pace_learn_mask(mass, keep)` (ds4.c:13188-region) at the widened
`keep` uses healthy stats by construction. This is the "riavvolgere anche gli
accumulatori PACE, non solo il KV" requirement.

### 3.4 Anti-oscillation

* `DS4_PACE_REWIND_MAX` (default 2): hard cap on rewinds per generation; after
  the cap, the airbag STOP is the only remaining action.
* `DS4_PACE_REWIND_BACKOFF` (default 256 tok, doubling each use): minimum tokens
  of forward progress before another rewind may fire; prevents rewind→same-loop
  →rewind thrash.
* On the second rewind of the same run, force a wider `keep` (+1 step) and, for
  temp>0, re-seed the sampler RNG so the resume cannot retrace the identical
  loop.

---

## 4. Interaction with the HTTP stream

The server streams each token to the client the moment it is sampled (emit
happens before the next eval; see the representative decode loop
ds4.c:25715-25736: `sample_argmax → emit → eval → pos++`). So by the time a
rewind fires, the client already holds the poisoned tail. Options:

| Option | Cost | Verdict |
|---|---|---|
| Buffer server-side until "sanity confirmed", release late | adds latency to every token; needs a sanity horizon; changes the streaming contract | rejected for v1 (kills the low-latency stream) |
| Emit a `rewind` retraction event in the JSONL/SSE stream; client trims back to `target` and continues | one event; server stays streaming; client owns the trim | **chosen for v1** — simplest, no latency tax |

**v1: client-side trim on a `rewind` event.** The server appends
`{"ev":"rewind","from":<current_pos>,"to":<target>,"reason":...}` to the event
stream (same JSONL channel as the existing PACE `s1_trigger`/`rotate(s1)`
events, `DS4_PACE_LOG`). A stream-aware client (our harness / `ds4_http_bench`)
truncates its accumulated text back to the character offset corresponding to
`to` and resumes appending. Non-aware clients see the retraction as a no-op and
keep the tail (degraded, but not worse than stopper-only). The offset mapping is
maintained client-side from the per-token deltas it already records in
`stream_events_measured.jsonl`.

---

## 5. Validation plan

A/B against the existing harness; **stopper-only (M1b) is the baseline and the
benchmark**, not a straw man.

* **Arms** (n≥3 rollouts each, medians, alternated ABAB order — greedy is *not*
  run-to-run deterministic here, so n=1 is uninterpretable;
  `ANALYSIS.md` §5c, fork at tok~75 on identical config):
  1. `stopper_only` (M1b airbag) — baseline.
  2. `prevention_only` (0020 rotate/widen on S1 slope).
  3. `prevention + rewind` (this doc).
* **Prompts**: the Phase-1 set (≥3 HTML: cyberpunk, coffee-shop compact, one
  new; ≥2 code), ctx8192, 2000–4000 tokens (per the post-T1 redefinition,
  `NEXT_STEPS_PLAN_20260710.md`).
* **Primary metrics**:
  * **L-level at render** (functional grade L0–L3, `functional_grade.py` /
    `retro_grade_l0l3.py`) — did the page/​code become valid?
  * **wasted tokens** = tokens emitted inside the loop before recovery/stop —
    directly comparable to the "~2600 tokens (65%)" figure in `ANALYSIS.md`.
  * **rewinds per run**, **rewind depth**, **post-rewind L-level lift** vs the
    same rollout's pre-rewind state.
* **Secondary**: t/s cost of the rewind (frontier restore + re-decode of the
  reverted span), % runs where onset fell out of the §1.5 bound
  (`rewind_skip`), oscillation count vs `REWIND_MAX`.
* **Instrumentation smoke (Risk R1, blocking)**: bit-equality check — decode to
  `p`, snapshot; continue to `p+Δ`; `rewind_kv` to `p`; assert the next K tokens
  match a **fresh** run truncated at `p` (temp=0). This proves the frontier +
  ring-slack rewind is exact before any quality A/B is trusted.

---

## 6. Prior art and honest positioning

Verified citations (from `docs/REAP_LOOP_NOVELTY.md`, WebFetch-checked, real):

* **LoopGuard** (arXiv 2604.10044, *Breaking Self-Reinforcing Attention Loops via
  Dynamic KV Cache Intervention*) — KV intervention triggered by an
  attention/KV signal.
* **LPSR** (arXiv 2604.18567, *Latent Phase-Shift Rollback*) — inference-time
  error correction that monitors the **residual stream** and steers/rolls back
  the KV cache.

Both already do "detect onset → rewind/edit the KV". **We do not claim that as
novel.** Our angle — the defensible line — is narrower and MoE-specific:

* The trigger **and** the rewind coordinate come from the **router**: S1 =
  gate-mass falling on the **pruned** experts of a real MoE under a live bias
  mask (LoopGuard uses attention/KV, LPSR uses the residual stream — neither
  uses routing). This is the same demarcation the novelty doc marks for the
  sensor, now extended to the *actuator's target*.
* The rewind is composed **closed-loop** with the REAP bias-mask actuator inside
  an **expert-offload** engine: rewind + mask-refresh + relearn-from-healthy in
  one controller, training-free, no weight update.

**do-not-claim (respect):** not "we invented detect-onset→rewind" (LoopGuard /
LPSR), not "KV-cache steering" as a primitive (LPSR), not "REAP" as a continuous
method (arXiv 2510.13999 is one-shot static). Claim only the **router-derived
onset coordinate** + the **closed-loop composition** with the mask actuator.

---

## Appendix A — hunk skeleton (pseudocode, NOT a patch)

> Feasibility is CLEAR (§1.8), so this sketches *where* the code lands. It is
> pseudocode for the design review, not the `0022` patch (that follows the M1b
> verdict + canonization). Anchors are live-tree line numbers.

**A.1 — dedicated rewind checkpoint buffers** (alloc block near ds4.c:11266,
guarded by `DS4_PACE_REWIND=1` instead of `enable_mtp`):

```c
/* per compressed layer, one extra frontier snapshot set for rewind */
g->spec_rewind_attn_state_kv[il]    = ds4_gpu_tensor_alloc(attn_width*attn_rows*4);
g->spec_rewind_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width*attn_rows*4);
if (ratio == 4) { /* index_state kv+score, same shape as spec_index_* */ }
```

**A.2 — public rewind_kv** (next to `ds4_session_rewind`, ds4.c:30560; declare in
ds4.h near :270):

```c
/* Rewind BOTH the logical position and the KV frontier to `pos`, using a
 * frontier snapshot previously taken at `pos`. In-memory, no re-prefill.
 * Precondition (caller-checked): pos_now - pos < raw_cap - raw_window. */
int ds4_session_rewind_kv(ds4_session *s, int pos, const ds4_spec_frontier *snap) {
    if (!spec_frontier_restore((ds4_spec_frontier *)snap, s)) return 1; // ds4.c:26800
    ds4_session_rewind(s, pos);                                          // ds4.c:30560
    return 0;
}
```

**A.3 — PACE state (extend `g_pace`, patch-0020 struct):**

```c
struct { /* g_pace additions */
    int              rewind_on;         /* DS4_PACE_REWIND */
    uint32_t         rewind_margin;     /* DS4_PACE_REWIND_MARGIN (16) */
    uint32_t         rewind_max;        /* DS4_PACE_REWIND_MAX (2) */
    uint32_t         rewind_backoff;    /* DS4_PACE_REWIND_BACKOFF (256) */
    int              rewind_keep;       /* DS4_PACE_REWIND_KEEP (=keep_max) */
    double           rewind_arm_thr;    /* DS4_PACE_REWIND_ARM_THR (=s1_slope_thr) */
    /* onset checkpoint */
    int              onset_pos;         /* -1 = not armed */
    int              onset_valid;
    ds4_spec_frontier onset_frontier;   /* the counters half; tensors in g */
    float           *onset_logits;      /* [DS4_N_VOCAB] */
    /* PACE accumulator checkpoint */
    double           ck_ema_ngram, ck_ema_hit, ck_ema_s1;
    double           ck_s1_ring[DS4_PACE_S1_RING_MAX]; uint32_t ck_s1_len, ck_s1_head;
    /* n-gram ring copy, mass[]/rmass[] copy (or full re-copy) */
    uint32_t         rewinds_done; uint32_t last_rewind_tok;
} ;
```

**A.4 — arm on ARM-threshold crossing** (in `ds4_pace_s1_update`, ds4.c:13381+,
after the slope is computed):

```c
if (g_pace.rewind_on && g_pace.phase == PACE_HOLD &&
    !g_pace.onset_valid && g_pace.s1_slope >= g_pace.rewind_arm_thr) {
    g_pace.onset_pos = (int)g_pace.tok;
    spec_frontier_snapshot(&g_pace.onset_frontier, s);   /* ds4.c:26766 */
    memcpy(g_pace.onset_logits, s->logits, DS4_N_VOCAB*sizeof(float));
    ds4_pace_ckpt_accumulators();     /* copy ema_*, rings, mass[] */
    g_pace.onset_valid = 1;
    ds4_pace_emit_s1("rewind_arm");
}
/* re-arm forward while still healthy (no FIRE yet): refresh onset to a later,
 * still-rising point so the eventual rewind stays shallow. */
```

**A.5 — fire in the tick ladder** (in `ds4_pace_tick`, ds4.c:13502+, in the
`PACE_HOLD` branch, after rotate/widen, when `sensor_due`/airbag is set):

```c
if (sensor_due && g_pace.rewind_on && g_pace.onset_valid &&
    g_pace.rewinds_done < g_pace.rewind_max &&
    g_pace.tok - g_pace.last_rewind_tok >= g_pace.rewind_backoff) {
    const int slack   = (int)(g->raw_cap - g->raw_window) - REWIND_GUARD;
    const int floor_p = (int)g_pace.tok - slack;
    int target = g_pace.onset_pos - (int)g_pace.rewind_margin;
    if (target < floor_p) { ds4_pace_emit_s1("rewind_skip"); /* fall to STOP */ }
    else {
        ds4_session_rewind_kv(s, target, &g_pace.onset_frontier);   /* A.2 */
        ds4_pace_restore_accumulators();          /* mass[]/rings back to onset */
        g_pace.cur_keep = g_pace.rewind_keep;     /* widen */
        ds4_pace_learn_mask(g_pace.mass, g_pace.cur_keep);  /* healthy stats only */
        ds4_pace_apply_keep(model, weights, "rewind");      /* 0011 actuator */
        memcpy(s->logits, g_pace.onset_logits, DS4_N_VOCAB*sizeof(float));
        g_pace.rewinds_done++; g_pace.last_rewind_tok = g_pace.tok = (uint32_t)target;
        g_pace.onset_valid = 0; g_pace.rewind_backoff *= 2;
        ds4_pace_emit_s1("rewind");   /* server relays as a stream retraction */
        break;   /* resume decode from `target` */
    }
}
```

**A.6 — server stream retraction** (server layer, not ds4.c): on a `rewind`
event, emit `{"ev":"rewind","from":...,"to":...}` into the response stream; the
harness client trims its accumulated text to the `to` offset and continues.

---

## Appendix B — proposed env surface

| Env | Default | Meaning |
|---|---|---|
| `DS4_PACE_REWIND` | 0 | master enable for S1-guided rewind (correction) |
| `DS4_PACE_REWIND_ARM_THR` | =`DS4_PACE_S1_SLOPE_THR` (3e-4/tok) | slope to arm the onset checkpoint |
| `DS4_PACE_REWIND_MARGIN` | 16 | rewind to `pos_onset − margin` |
| `DS4_PACE_REWIND_MAX` | 2 | max rewinds per generation (anti-oscillation) |
| `DS4_PACE_REWIND_BACKOFF` | 256 | min forward tokens between rewinds (doubles each use) |
| `DS4_PACE_REWIND_KEEP` | =`DS4_PACE_KEEP_MAX` | keep-K width on resume (widen); 0 = coverage-based |
| `DS4_PACE_REWIND_GUARD` | 32 | safety margin under the ring-slack bound (§1.5) |

Requires `DS4_PACE=1` + `DS4_PACE_S1=1` (the monitor) + `DS4_PACE_REWIND=1`.
Off by default; the airbag stopper (M1b) is independent and stays the fallback.
