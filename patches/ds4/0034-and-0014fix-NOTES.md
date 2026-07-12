# Adversarial-audit fixes: 0034 (cuda stale-slot) + 0014-fix (pace WRAP default)

Authored in an isolated `/tmp/ds4fix_work` copy in WSL; `/root/ds4` and
`/root/ds4-pin` were not touched. Not committed — left for review.

## 0034-cuda-stale-slot-fix.patch (HIGH, correctness)

Base: `ds4_cuda.cu` after `0024` -> `0031` -> `0033`
(md5 `95af439758492f77c017c60d24a0563f`). Verified by reconstructing that
chain from the pristine pre-0024 source (md5 `7d57f58d414dffc49a19ced3e9a79dd4`)
and reproducing the documented intermediate hashes (`c564ca7c...` after 0024,
`430716f4...` after 0031) before landing on `95af4397...`.

**Bug:** 0024 stopped calling `cuda_stream_expert_cache_invalidate()` on a
single failed slot load (correctly, to avoid collapsing the whole cache), but
only cleared the local `cache_slot = -1`. The underlying
`expert_cache->slots[load_slot]` kept its *old* expert's identity with
`valid = 1`, sitting over a buffer `load_slot()` may have partially
overwritten with the *new* expert's weights before failing. A later
`cuda_stream_expert_cache_find()` for the old expert matches that slot and
serves corrupted, non-bit-exact weights for the rest of the session. Same
pattern in the twin `!copied_from_global_cache` D2D-readout-failure branch.

**Fix:** three edits, all inside `cuda_stream_selected_cache_begin_compact_load()`
and `cuda_stream_expert_cache_load_slot()`:
1. `expert_cache->slots[load_slot].valid = 0;` before `cache_slot = -1;` in the
   `load_slot()` failure else-branch.
2. `expert_cache->slots[(uint32_t)cache_slot].valid = 0;` in the twin
   `!copied_from_global_cache` D2D branch, before falling through to the
   direct load.
3. (defense in depth) `cache->slots[slot].valid = 0;` at the top of
   `cuda_stream_expert_cache_load_slot()`, before its first buffer write.

Restores the residency != selection invariant on the failure path: only the
failing expert degrades to a direct load (0024's intent), without leaving a
corrupted slot matchable by identity. Purely additional slot bookkeeping on
an already-failing path -- no new env var, no change to the success path, no
change to stock (non-failure) behavior.

Apply-check: `git apply --check` and `patch -p1 --dry-run` both clean against
the reconstructed post-0033 base; applied result md5
`3c422e3a59833bf987f214bcc8e326c8`. Confidence: high -- both failure sites
and the load_slot entry point were located by exact, unique string match (not
approximate line numbers), and brace/paren count delta is 0.

## 0014-fix-pace-wrap-default.patch (MED, perf-default)

Base: pristine `/root/ds4/ds4.c` (md5 `771a39a861e9512fed2fc4528780e080`) --
0014/0014e and later WRAP-related patches are already merged into this file
on disk, so the fix applies directly, no chain reconstruction needed.

**Bug:** `ds4_pace_init()`'s WRAP block:
```c
g_pace.wrap_on = (w && *w && strcmp(w, "0")) ? 1 : (w ? 0 : 1);
```
resolves to **1 (ON)** whenever `DS4_PACE_WRAP` / `DS4_REAP_WRAP` /
`DS4_REAP_PREFETCH` are all unset (the `w == NULL` case falls into
`(w ? 0 : 1)` -> `1`). `PACE_DESIGN.md:252` documents the default as **0
(off)**: unconditional WRAP bulk page-in on every mask change measured 0.82
vs 1.27 t/s on a practical SSD-light config, a ~35% regression eaten silently
by anyone who sets `DS4_PACE=1` without separately forcing
`DS4_PACE_WRAP=0`.

**Fix:**
```c
g_pace.wrap_on = (w && *w && strcmp(w, "0")) ? 1 : 0;
```
Default off when unset (matches the doc); only an explicit non-`"0"` value
turns WRAP on. Also reconciled the stale 0014e-era comment in the sibling
`ds4_reap_prefetch_working_set()` WRAP gate, which described the old (buggy)
"Default ON when PACE drives it" behavior.

Apply-check: `git apply --check` and `patch -p1` both clean against pristine
`ds4.c`; applied result md5 `e543f8c46c2cf73ffdd3133579a59f6e`. Confidence:
high -- single-site ternary fix, truth table verified by hand for all four
input classes (unset / empty / `"0"` / other).

## Neither fix changes stock behavior

- 0034: fires only on an already-failing load/copy path; the success path is
  untouched.
- 0014-fix: `DS4_PACE` itself defaults off, and any caller who already sets
  `DS4_PACE_WRAP` explicitly (0 or non-0) sees identical behavior before and
  after. Only the previously-undocumented "unset defaults to WRAP-on" case
  changes, to match the documented default.
