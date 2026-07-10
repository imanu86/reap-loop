# 2026-07-10 Pod smoke — patches 0020 / 0021 (mechanism, not speed)

Last step of Track D. Purpose: prove the two live-tree-anchored PACE patches
**compile** and their **mechanisms fire** on a real generation. This is a
MECHANISM smoke — the pod is a 1 TB-RAM RTX 3090 where the 86.7 GB model sits in
RAM, so any t/s here is a pod number and does **not** transfer to the 3060.

## Setup

- Pod: RunPod `nyx0ubkpva1j9c`, SECURE RTX 3090 24 GB, $0.46/h, direct SSH.
- Source: LOCAL WSL live-tree `/root/ds4` snapshot (`ds4.c` md5 `771a39a8…`,
  the exact base the patch headers are anchored to — md5 matched at snapshot
  time, Codex had NOT modified the tree). Tarred sources-only, extracted on pod
  into `/root/ds4-live` (the pod's own `/root/ds4` is the canonical-series build
  WITHOUT pace 0015+ rotation → patches cannot apply there; hence the live-tree
  transfer). Pod extract md5 confirmed `771a39a8…`.
- Patches applied in series with `git apply --check` then `git apply`:
  - `0020-pace-s1-slope-trigger.patch` — CHECK OK, applied clean (0 fuzz).
  - `0021-pace-rotate-delta-prefetch.patch` — CHECK OK, applied clean on top of 0020.
  - ds4.c md5 after 0020 = `7172f2d2…`, after 0021 = `58b991ac…`.
- Build: `PATH=/usr/local/cuda/bin:$PATH make cuda CUDA_ARCH=sm_86 -j16`
  (nvcc 12.9). **Exit 0, clean under -Wall -Wextra, no fix iteration needed.**
  `ds4` + `ds4-server` produced. See `build.log`.
- Model: `/root/models/ds4-2bit.gguf` (IQ2XXS imatrix), `--cuda --ssd-streaming
  --ssd-streaming-cold --ssd-streaming-cache-experts 1024 -c 2048 --nothink
  --temp 0`, prompt = recovered `frontpage_prompt.txt` (coffee shop, 819 B).
- Runner: `ds4` CLI (the reference recipe's tool; PACE tick runs in the shared
  decode path, identical mechanism to `ds4-server`). Baseline also confirms the
  freshly-built binary serves/generates.

## Gates

| Gate | Result |
| --- | --- |
| apply 0020 | PASS (clean) |
| apply 0021 | PASS (clean, after 0020) |
| build (cuda sm_86) | PASS (exit 0, no warnings) |
| smoke a — baseline sanity | PASS — valid `<!DOCTYPE html>` page, no crash, PACE engaged (WARMUP=16 KEEP=23) |
| smoke b — 0020 S1 trigger | PASS — trigger forced, fires + rotates, slopes numeric non-NaN |
| smoke c — 0021 rotate_delta | PASS — pages only entered experts, no full WRAP on decode |
| smoke d — off-switches | PASS — trigger=0 → 0 events; delta=0 → 0 events |

## Key numbers from the JSONL

### b) S1-slope trigger (`s1trig.jsonl`)
Env: `DS4_PACE_S1=1 DS4_PACE_S1_TRIGGER=1 DS4_PACE_S1_SLOPE_THR=0.000001
DS4_PACE_S1_STABLE=4 DS4_PACE_S1_SLOPE_WIN=16 DS4_PACE_S1_ACTION=rotate`, n=256.
(`SLOPE_WIN` lowered 64→16 so the slope ring fills inside the 256-tok budget —
a deliberate mechanism-forcing choice, alongside the near-zero threshold.)
- **8× `s1_trigger`**, each immediately followed by **`rotate(s1)`** (8×). No crash.
- Slope values numeric & sensible (non-NaN): `0.0301208`, `0.0028892`, `0.0007313`, …
- S1 pruned-mass level climbs `0.72658 → 0.80803 → 0.81393` — matches CLAIM-011's
  measured pre-collapse band (0.72→0.81 local / 0.73→0.81 historical K91).
- NaN count in JSONL: 0.

### c) rotate delta-prefetch (`delta.jsonl` / `delta.diag`)
Env: `DS4_PACE_ROTATE=1 DS4_PACE_ROTATE_EVERY=8 DS4_PACE_WRAP_ROTATE_DELTA=1`, n=128.
- **15× `rotate_delta`**, `entered==exited>0` every time (K-constant rotate):
  `entered` = 631, 268, 227, 166, 119, … (delta shrinks as the mask converges).
- bytes / expert = `4466147328 / 631 = 6.7503 MiB` → matches the header's
  ~6.75 MiB/expert (gate+up+down rows).
- **No full WRAP on decode**: the only `REAP prefetch (fattorino)` line is the
  initial prefill working-set page-in (6.07 GiB); every decode-time rotate emits
  `REAP delta-prefetch (rotate)` only (0.6–4.2 GiB, the delta), never the full
  75–699 GiB re-WRAP. The delta path returns before the WRAP block.

### d) off-switches
- A — `DS4_PACE_S1=1 DS4_PACE_S1_TRIGGER=0` (monitor on, act off), n=256
  (`offA_s1trig0.jsonl`): **0× `s1_trigger`, 0× `rotate(s1)`**. Only stock PACE
  events (`breath(ngram)`, `breath_end`, `relearn`) — defaults unchanged.
- B — `DS4_PACE_ROTATE=1 DS4_PACE_ROTATE_EVERY=8 DS4_PACE_WRAP_ROTATE_DELTA=0`,
  n=128 (`offB_delta0.jsonl`): 15× stock `rotate`, **0× `rotate_delta`**, 0
  `delta-prefetch` stderr — full WRAP stays off on rotate (0018), delta not armed.

## Files
- `build.log` — full build (exit 0).
- `baseline.{out,diag,jsonl}` — sanity generation.
- `s1trig.{out,diag,jsonl}` — smoke b.
- `delta.{out,diag,jsonl}` — smoke c.
- `offA.{out,diag}`, `offA_s1trig0.jsonl` — smoke d (trigger off).
- `offB.{out,diag}`, `offB_delta0.jsonl` — smoke d (delta off).
- (`.diag` files trimmed of the per-layer `gpu prefill layer` spam; PACE/SPEX/
  prefetch lines kept.)

## Caveat
t/s observed on the pod are NOT reported as performance (1 TB-RAM regime ≠ 3060).
This smoke validates only: patches compile and the two mechanisms fire and gate
correctly. Canonization onto the clean pace-series base is still pending.
