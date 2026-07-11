# Smoke 0022 — S1-guided rewind actuator (pod D) — VERDICT

> **PARTIALLY SUPERSEDED — v2 re-smoke below.** The v1 section (gate a FAIL) documents the
> snapshot-guard bug, since fixed upstream (`2586a25`). The **v2 re-smoke** with the fixed
> 0022 is **ALL PASS** — see "§ v2 re-smoke" at the bottom. v2 artifacts in `v2/`.

**Date:** 2026-07-11 (UTC ~02:00). **Pod:** RunPod `7qgalm9sasqnr7` (COMMUNITY RTX 3090
24GB, machine `6p17plgwgpsn`), image `runpod/pytorch:1.0.7-rc.138-cu1290-torch280-ubuntu2404`
(CUDA 12.9 — the cu1290 variant, driver 580.159.03; the cu12.8 breakage the mandate warns
about did NOT occur). Gate-check PASS: `torch 2.8.0+cu129 cuda_avail True device_count 1`,
nvcc 12.9. Pod 128 vCPU / 251 GB RAM (model RAM-hot → **t/s are pod-only, NOT comparable to
the 3060**).

## Headline

**Gate (a) FAIL · (b) PASS · (c) PASS (detector-level, see caveat).** The 0022 **detector**
(E-DET EWMA-CUSUM ARM/FIRE) works, but the **rewind actuator cannot fire on
DeepSeek-V4-Flash** — a snapshot-precondition bug makes the onset checkpoint never validate,
so no `rewind` event is emittable under any threshold. NOT all-pass ⇒ **no R2 binary upload**.

## Build provenance

- R2 fast-bootstrap: `ds4-src_livetree-771a39a8.tgz` (sha256 verified), `git init`, base
  `ds4.c` md5 **771a39a8** (= live-tree = canonical-v2 intermediate).
- Baseline = live-tree + **0020→0021→0026→0027→0028** (0028 applied from the LF blob
  `b94be032`, CRLF trap avoided). Final baseline `ds4.c` md5 **62ed2e71** — byte-identical to
  the certified canonical-v2 + 0027 + 0028 target (mandate acceptance md5 hit exactly). This
  path was chosen because FASE-1's R2 tarball delivers the live-tree directly and the README
  certifies `live-tree 771a39a8 → 0020→0021→0026→0027→0028 → md5 62ed2e71`, identical to the
  full canonical chain from `80ebbc3`. Build `make cuda CUDA_ARCH=sm_86 -j`: **0 warnings**.
- **0022 applies clean AFTER 0028** (declared base in its header = canonical-v2+0027+0028
  md5 62ed2e71; first attempt, no fallback needed). Post-0022 `ds4.c` md5 **a7aab041**.
- Rebuild with 0022: **0 warnings, 0 errors**, 74s.
- Applied commit: 0022 blob `61fc268413a8dcb68fe305ed107e619b758be89a`
  (origin/main `4343df6` "patches: author 0022 S1-guided rewind actuator …", touches
  `patches/ds4/0022-pace-s1-rewind.patch`). Model `ds4-2bit.gguf` sha256 verified
  (`efc7ed60…`).
- **Final binary md5:** `ds4` = `527214622275e240b00ea454442ac9e2`,
  `ds4-server` = `fa7241c54eb2750ab335bd24e8d56558`.

## Common smoke config (all gates)

Coffee prompt (819 B, `coffee_prompt.txt`), in-engine PACE **W50 static K23**:
`DS4_PACE=1 DS4_PACE_S1=1 DS4_PACE_WARMUP=50 DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23
DS4_PACE_KEEP_MAX=96 DS4_PACE_BREATH_EVERY=999999 DS4_PACE_RELEARN=0 DS4_PACE_ROTATE=0
DS4_PACE_WRAP=1 DS4_PACE_WRAP_ROTATE_DELTA=1 DS4_PACE_DEBUG=1`, greedy `--temp 0 --nothink
--ssd-streaming --ssd-streaming-cache-experts 1024 -c 8192 -n 600`. Events → `DS4_PACE_LOG`.

## Gate (a) — forced FIRE: **FAIL**

Env added: `DS4_PACE_REWIND=1 ARM_K=-20 ARM_H=0.1 FIRE_K=-20 FIRE_H=0.1` (CALWIN default 128).
The detector was fully forced — at gen-tok 129 (pos 346, right after the 128-tok calibration)
`cusum_arm=cusum_fire=20.0` (>> both thresholds) and `rewind_arm` latched. **But `rewind`
events = 0.** Run completed rc=0 in 62s, no crash, output is a complete well-formed HTML page
(`…</body></html>`).

Required by the gate: a `rewind` event with from/to/reason + visible post-rewind
regeneration. **Not produced ⇒ FAIL.**

Evidence (`a_events.jsonl`):
```
{"ev":"rewind_arm","from":346,"to":0,"reason":"s1_cusum_arm","tok":129,"keep":23,
 "s1":0.88102,"cusum_arm":20.000,"cusum_fire":20.000,"n":0,"regen":0}
```
Note `"to":0` — `g_rewind.onset_pos` never advanced past 0, i.e. the onset frontier snapshot
never succeeded.

### Root cause (source-confirmed bug in 0022)

`ds4_pace_rewind_snapshot_frontier` (ds4.c:27671) opens with:
```c
if (DS4_N_LAYER > 0 && !g->spec_rewind_attn_state_kv[0]) return false;
```
On **FLASH**, `ds4_expected_layer_compress_ratio` returns 0 for `il < 2` → **layers 0–1 are
dense**. The dedicated rewind buffers are allocated only inside `if (ratio != 0)`
(ds4.c:11326), so `spec_rewind_attn_state_kv[0]` is **permanently NULL** ⇒ the guard **always
returns false** ⇒ the onset snapshot never runs ⇒ `g_rewind.onset_valid` is never set ⇒ the
FIRE branch `if (!(fire_hi||airbag) || !g_rewind.onset_valid) return -1;` always bails. The
actuator is **structurally incapable of firing a rewind on Flash**, regardless of thresholds.

The MTP twin `spec_frontier_snapshot` (ds4.c:27555) has **no** such layer-0 guard — it loops
all layers and skips dense ones via `if (ratio==0) continue;` — and works every token. The
0022 snapshot fn added the extra guard and thereby broke itself.

**Suggested fix (NOT applied — mandate forbids forcing/patching):** drop the layer-0 guard to
match the MTP twin, or guard on the first *compressed* layer (smallest `il` with `ratio!=0`,
= 2 on Flash) instead of literal index 0. `ds4_pace_rewind_restore_frontier` has no such
guard, so restore is unaffected.

## Gate (b) — off-switch (`DS4_PACE_REWIND=0`): **PASS**

`rewind_arm=0, rewind=0, rewind_skip=0`; no `PACE REWIND on` banner (dedicated buffers not
allocated). rc=0. Clean off-switch. (Note: this is a trivial pass — even ON, the actuator
never fires per gate a.)

## Gate (c) — default E-DET thresholds: **PASS (detector-level; caveat)**

Env: `DS4_PACE_REWIND=1`, detector defaults `arm(k0.5,h4) fire(k1.0,h8) CALWIN128`.
**0 `rewind` events** — the letter of the gate ("healthy regime ⇒ 0 rewinds") is met. The
conservative production FIRE stayed silent: `cusum_fire` peaked at **0.96 ≪ h_fire=8.0**. The
default ARM tripped twice (tok 207/223, `cusum_arm` 4.44/4.78 > h_arm=4.0) — within the E-DET
tuning's benign-arm budget (~4/1k here vs ~8/1k expected). rc=0.

Evidence (`c_events.jsonl`): two `rewind_arm` with `cusum_fire` 0.961 / 0.311, `to=0`.

**Caveat:** because the actuator cannot fire under any condition (gate a bug), gate (c) does
not actually exercise the actuator's spurious-fire resistance; it only confirms the *detector*
does not reach FIRE on healthy coffee. Real actuator validation is blocked until the snapshot
guard is fixed.

## Summary table

| gate | config | rewind_arm | rewind | verdict |
|---|---|---:|---:|---|
| a | forced (K=-20,H=0.1) | 1 | **0** | **FAIL** (actuator cannot fire on Flash) |
| b | off (REWIND=0) | 0 | 0 | **PASS** |
| c | E-DET defaults | 2 | 0 | **PASS** (detector-level; actuator untestable) |

All three runs completed rc=0 with no crash and complete HTML output. The failure is a
missing capability, not an instability.

## Cost / pod

- RunPod balance **before $17.9459633467 → after $16.724298862**; **mandate spend ≈ $1.2217**
  (community 3090 billed above the $0.22/h floor + 130 GB container disk). Under the $3 cap.
- **Pod left RUNNING** (id `7qgalm9sasqnr7`) per mandate — NOT stopped, NOT terminated.
- No R2 binary upload (upload is gated on all-3-PASS; gate a FAILed).

## Files

`{a,b,c}_events.jsonl` (JSONL event streams), `{a,b,c}_gen.txt` (generated HTML),
`{a,b,c}_run.log` (raw stderr), `{a,b,c}_status.txt`, `run_{a,b,c}.sh` + `common_env.sh`
(exact invocations), `build_baseline.log` / `build_0022.log` (0-warning builds).

---

# § v2 re-smoke — fixed 0022 (snapshot-guard fix `2586a25`) — **ALL PASS**

**Date:** 2026-07-11 ~00:30 UTC, same pod (`7qgalm9sasqnr7`, still RUNNING). Fix commit
`2586a25` "patches: fix 0022 snapshot guard (Flash layer-0 is dense; scan first allocated
rewind buffer)" — new 0022 blob `24a1f306`. The guard now scans layers for the first
*allocated* `spec_rewind_attn_state_kv[il]` instead of probing literal index 0 (exactly the
diagnosed root cause; layer 0 is dense on Flash).

## Build v2

Pod tree `git reset --hard` to the post-0028 baseline (md5 **62ed2e71** re-verified), fixed
0022 v2 applied clean (first try) → `ds4.c` md5 **a88f9dcb** (`have_buf` scan present at
ds4.c:27677-27681). `make cuda CUDA_ARCH=sm_86`: **0 warnings** (`v2/build_0022v2.log`).
**Binaries: `ds4` md5 `9048e70756a8f490d45add4591f59a15`, `ds4-server` md5
`ddf6da2df4c12434e99e6693a2aaccc6`.**

Same smoke config as v1 (coffee 819B, W50 static K23, greedy temp0, `-n 600 -c 8192`,
`--ssd-streaming --ssd-streaming-cache-experts 1024`).

## Gate (a) v2 — forced FIRE: **PASS**

Forced thresholds (`ARM_K=-20 ARM_H=0.1 FIRE_K=-20 FIRE_H=0.1`), `DS4_TOKEN_TIMING=1`,
stderr timestamped (`v2/ts.py`). Events (`v2/a_events.jsonl`):

```
{"ev":"rewind_arm","from":346,"to":314,"reason":"s1_cusum_arm",...}    <- onset NOW VALID (314, was 0 in v1)
{"ev":"rewind","from":346,"to":315,"reason":"s1_cusum_fire","tok":97,"keep":96,...,"n":1,"regen":32}
{"ev":"rewind_arm","from":315,"to":314,...}                            <- re-arm after fire; backoff blocks 2nd fire
```

- `rewind` event with **from=346 / to=315 / reason=s1_cusum_fire** ✓
- post-rewind regeneration visible: 379 timed evals for ~348 final positions (the 31-token
  span 315..346 re-decoded), PACE clock rewound 129→97, keep widened 23→96 ✓
- run continues and terminates normally, rc=0, complete `</html>`, zero crash markers ✓
- anti-oscillation observed: second ARM latched but no second fire (BACKOFF=256) ✓

## Gate (b) v2 — off-switch (`DS4_PACE_REWIND=0`): **PASS**

0 rewind-family events, no `PACE REWIND on` banner, rc=0 (`v2/b_events.jsonl`).

## Gate (c) v2 — default E-DET profiles: **PASS**

0 `rewind` fires on healthy coffee; `cusum_fire` peak **1.22 ≪ h_fire=8**. Two benign
`rewind_arm` (tok 195/334) now with **real frozen onsets** (`to:410`/`to:538` — the rolling
snapshot works under defaults too). rc=0 (`v2/c_events.jsonl`).

## REWIND COST — measured (decision-model input, replaces the assumed 56 tok)

From `v2/a_run.log` (monotonic-timestamped stderr + per-eval ms) parsed with
`rewind_cost.py` (median inter-eval gap over 378 gaps = **0.180 s/tok**, pod RAM-hot):

| quantity | measured | note |
|---|---|---|
| regen tokens | **31 re-decoded** (engine counter `regen:32` = from−onset) | forced-fire depth ≈ snapshot cadence EVERY=32 (floor). Production depth ≈ EVERY + ARM→FIRE detection delay (E-DET median ~40 tok) ⇒ the 56-tok decision-model assumption is the right order; measured floor = 32. **Token count transfers; times below are pod-only.** |
| restore+rewind cycle wall | **≈ 0.31 s** | raw gap between the evals surrounding the fire = 0.494 s minus median gap 0.180 s. Includes frontier restore, PACE-blob restore, mask relearn+apply (keep 23→96), delta-prefetch, logits restore. |
| regen re-decode wall | ≈ 5.6 s (31 × 0.180 s) | pod-only t/s; scales with local rate on the 3060. |

I.e. the **fixed overhead of a rewind cycle is sub-second (~1.7 eval-slots)**; the dominant
cost is re-decoding the reverted span, which is bounded by EVERY + detection delay.

## R2 upload (all-PASS condition met)

Uploaded to `r2:ds4-models` (2026-07-11 00:27–00:32 UTC, listing verified):
- `ds4-server_sm86_livetree-a88f9dcb` (11 829 088 B) + **`.meta`** (chain, sha256s, smoke ref)
- `ds4_sm86_livetree-a88f9dcb` (10 908 856 B)

Chain in .meta: live-tree 771a39a8 (=canonical v2) + 0020+0021+0026+0027+0028(b94be032) +
0022v2(24a1f306, fix 2586a25), sm_86, cu1290 image, 0-warning build.

## Cost / pod (final, whole mandate incl. v1)

Balance **$17.9459633467 → $15.7854758447** ⇒ **total spend ≈ $2.16** (cap $3). Pod
`7qgalm9sasqnr7` left **RUNNING** per mandate.

## v2 verdict table

| gate | config | rewind_arm | rewind | verdict |
|---|---|---:|---:|---|
| a | forced (K=-20,H=0.1) | 2 | **1** (from=346,to=315,regen=32) | **PASS** |
| b | off (REWIND=0) | 0 | 0 | **PASS** |
| c | E-DET defaults | 2 (benign, real onsets) | 0 | **PASS** |

**0022 v2 è GO: actuator live su Flash, off-switch pulito, profili default silenti in regime
sano. Cost-model: ciclo fisso ~0.31 s + regen ≈ EVERY+delay token.**
