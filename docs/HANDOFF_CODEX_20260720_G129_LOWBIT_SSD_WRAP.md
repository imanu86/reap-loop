# DS4 Windows G129 low-bit recovery and SSD-WRAP delta handoff

Date: 2026-07-20

This document is the canonical delta after
`HANDOFF_CODEX_20260719_G129_FRESH_TASK.md`. Read that handoff first, then this
file, then the three canonical ledgers. Do not reconstruct state from chat
memory.

## 1. Final mandate remains open

Do not mark the project complete until all conditions hold together:

- server decode mean greater than 6.0 t/s;
- at least three independent clean processes;
- exactness, provenance and contamination gates pass;
- quality is at least L2, ideally median L3;
- router is full/open;
- no forbidden SSD-to-VRAM transition for the current token.

G73 remains the historical closed/request-scoped reference at 4.986667 t/s
mean, 4.98 median, n=3. It is not the final full/open control.

## 2. Authoritative repositories

- G129 runtime worktree:
  `C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2`
- Historical G73 checkout:
  `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work`
- Ledgers and planning:
  `C:\Users\imanu\source\repos\reap-loop`
- Low-bit research:
  `C:\Users\imanu\source\repos\moe-aggressive-commit`

The G129 and G73 trees contain intentional unrelated dirty state. Never reset,
stash or bulk-add them. Stage only audited files named in the relevant commit.

## 3. Authoritative models

IQ2 model:

- path: `C:\ds4-models\ds4-2bit.gguf`
- bytes: 86,720,111,488
- SHA-256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`

Q1_0 routed-expert sidecar:

- path: `C:\ds4-models\ds4-q1-layers0-42-derived.gguf`
- bytes: 39,048,344,416
- SHA-256: `05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207`

Use existing receipts. Do not repeatedly hash these large files.

## 4. G129 structural result

The official promotion confirmation safety is structural PASS n=1:

- tag:
  `g129_promotion_confirm_final_20260720T025308987Z_foreground_codex_20260720T025309391Z_e3ef656d25`
- receipt SHA-256:
  `8d3898bc37fd7552489e991c4466d9e303a303b08d144efa321d5193cb06d1f4`
- router mode: open, no static mask;
- route trace and tier rows: 16,512 / 16,512;
- Q1 routes: 9,608;
- IQ2 routes: 6,904 = 5,831 VRAM + 1,019 snapshot RAM + 54 tier RAM;
- SplitFused IQ2-only expected/observed: 6,904 / 6,904;
- promotion records: 128 JSONL lines = 64 attempts + 64 successes;
- next-call causal gates and source/destination offsets/SHA passed;
- direct or forbidden SSD-to-VRAM current-token events: zero;
- Q1 H2D bytes: 34,002,173,952;
- IQ2 promotion SSD-to-RAM bytes: 452,984,832;
- source unlock: 129 calls, zero successes, 129 not-locked;
- diagnostic throughput: about 0.181705 client t/s;
- no long quality, n>=3, performance or SOTA claim.

The immediately preceding run completed the runtime but its official receipt
failed after runtime because the PowerShell parent converted multi-line child
output from `System.Object[]` to `System.Int32`. Keep the failure and the later
PASS as separate evidence.

## 5. G73 two-turn status

The original two-turn replay established the mechanism at n=1:

- stale Arm A retained the request-1 mask and seed;
- Arm B rebuilt a different 4,551-entry mask and 320-entry seed from a
  full-conversation request-2 prefill with no KV reuse;
- official outputs were L0 because fenced text failed the raw-only contract;
- fence-stripped diagnostic grading was A=L1 and B=L2;
- this is mechanism evidence only, not n>=3 or SOTA evidence.

A later true live Arm B attempt failed before readiness and before any HTTP
request. It is a NO-RUN, not a negative result for request-epoch rebuild.

The dedicated live runner now passes a CPU mock integration suite:

- successful request sequence is `system,user` then
  `system,user,assistant,user`;
- assistant turn 1 is inserted byte-identically into request 2;
- exit-before-readiness fails closed;
- malformed turn 1 prevents request 2;
- this proves runner conversation construction only, not DS4 runtime quality or
  performance.

## 6. Low-bit representation result

The current DS4 Q1_0 sidecar already has the same nominal density as the
Bonsai binary format: 1.125 bits per weight.

- one Q1 routed expert: 3,538,944 bytes;
- weights per routed expert: 25,165,824;
- routed experts: 11,008;
- total routed weights: 277,025,390,592;
- binary payload at 1.125 bpw: about 38.96 GB;
- actual sidecar: 39.05 GB, or 36.37 GiB.

This explains the apparent Bonsai discrepancy. A 27B model at 1.125 bpw is
about 3.80 GB decimal or 3.54 GiB. DS4 contains about 10.26 times as many routed
weights. The compression density is already equivalent; quality recovery is
the missing component.

One-expert probe, block 3 expert 0:

- current Q1 versus IQ2 teacher: cosine 0.811377, NMSE 0.341667;
- binary L2 re-fit reproduces the current Q1 almost exactly;
- ternary 1.296875 bpw: cosine 0.758266, NMSE 0.425033;
- another weight-only converter is NO-GO;
- activation-aware recovery or training is required.

The current Q1 representation is not a standalone quality candidate:

- pure Q1 G112 reached about 6.76 t/s but was L0;
- mixed G113/G114 reached about 6.9-7.0 steady but was L1;
- G115 with five IQ2 and one Q1 route still produced malformed output.

The intended final design is not full Q1. Q1 or a Bonsai-style recovered binary
base is the full/open resident fallback; exact IQ2 must carry most router mass.

## 7. Route replay and capacity result

Observed G129 fallback is not rare:

- Q1 route fraction is about 58 percent;
- Q1 gate-mass fraction is about 52 percent;
- first-token Q1 mass is about 63 percent.

Replay estimates for exact-IQ2 host capacities:

| exact slots | oracle fallback mass | LFRU first pass | LFRU second pass |
|---:|---:|---:|---:|
| 606 | 24.0% | 37.1% | 31.5% |
| 910 | 18.0% | 32.1% | 24.5% |
| 1213 | 13.5% | 28.6% | 19.2% |

Exclusive replacement produces excessive I/O. Keep the low-bit base resident
and duplicate a bounded IQ2 arena. Improve prefill selection/prediction before
assuming that a larger arena alone solves quality.

## 8. Activation-aware trace implementation

G129 now has an OFF-default, input-only recovery trace for one layer/expert:

- captures input vector, request epoch, call tick, token, top-k rank, gate
  weight and Q1/IQ2 representation;
- maximum 256 samples;
- canonical 64-byte binary header and float32-le vectors;
- strict JSONL metadata, byte/range/SHA checks and atomic manifest-last commit;
- no stable teacher output boundary is captured from runtime;
- exact-IQ2 teacher outputs are reconstructed offline from captured inputs;
- OFF performs no file write, D2H copy or additional synchronization.

CPU-only certification:

- PowerShell 5.1 parse PASS;
- two positive parser samples and ten negative cases PASS;
- both Python tests PASS;
- control, promotion and trace WhatIf PASS;
- Release Ninja sm_86 build PASS;
- CTest 1/1 PASS;
- git diff --check PASS.

Build identity:

- manifest:
  `ba819fee217ba5c2eddca6894a8b012e35560aa3ed2c6a216115e7671157bf5a`
- input fingerprint:
  `2c382328f8f1327b613f449f3ebdd85f2c7a32143339f77a5bb504182de5ea40`
- executable:
  `51ff6dad1e0bab5d3a0c6e145d767e5a0dc0f8e2f99660b68aaf4f4fb8adb9a8`

This authorizes only a future structural trace n=1 after separate GPU approval.
It proves no recovery quality or performance.

## 9. Current architecture decision

The desired hierarchy is:

1. exact IQ2 hot cache in VRAM;
2. small fixed page-locked IQ2 host reserve for immediate asynchronous H2D;
3. larger pageable-resident IQ2 warm/probation reserve;
4. full low-bit resident base for every routed expert;
5. IQ2 SSD/mmap as the immutable source.

On a current-token miss:

1. use packed Q1/binary from resident RAM immediately;
2. enqueue exact IQ2 promotion asynchronously;
3. move IQ2 through SSD/mmap -> RAM -> optional VRAM;
4. make it eligible only after completion and strictly after the observation
   call;
5. never wait for SSD on the current token.

The SSD is allowed to work. It must not be on the decode critical path. At one
7,077,888-byte promotion per token and 6 t/s, average SSD demand is only about
42.5 MB/s. The main risks are serialized reads, pageable-to-pinned copies,
per-upload synchronization, churn and unused promotions.

## 10. SSD-WRAP CPU implementation result

SSD-WRAP is now implemented and has passed CPU-only build and static gates. It
has not yet passed a runtime safety and is not performance or quality evidence.

Implemented behavior:

- bounded admission queue using the existing ranking/policy;
- deduplication by layer/expert;
- source-range validation, ordering and provenance-safe coalescing;
- large prefill/rebuild waves and bounded decode micro-waves;
- explicit REQUESTED -> SSD_INFLIGHT -> RAM_READY -> RAM_COMMITTING ->
  PINNED_READY or PAGEABLE_READY -> later-call ELIGIBLE states;
- current-call Q1 continuation and no current-token SSD-to-VRAM transition;
- fixed 5.5 GiB IQ2 host budget: 834 slots, with four transition-ring slots;
- 2.0/3.5 GiB initial split: 299 pinned and 531 pageable stable slots;
- fixed pinned bounce ring for pageable-to-VRAM H2D, with no dynamic host
  registration on the hot path;
- QueryWorkingSetEx only at init, flush and release, never per token;
- fail-closed partial-read, range, provenance, stale-age and pairing gates;
- OFF creates no thread, handle, ring, allocation or write.

CPU-only gates passed PowerShell parsing, parser positives/negatives, runner
SelfTest, both Python suites, three WhatIf variants, Release `sm_86`, CTest 1/1
and diff-check. Frozen manifest, fingerprint and executable SHA-256 values are:

- `1893258c8406e8c668eaf1856527ba0b5aba9ea2169e7d53525ca7257686a66b`;
- `f80f32087ed9d7716651ea661846a4ed50082aae0746700c8fc3c5c0a75a106a`;
- `e258a4fd60c7dc4dfb98cd0ec8b168f4a06e3f3f435adbe8ef89540cd2d307e5`.

At 6 t/s, one promotion per token requires about 45.65 MB/s. The recorded
SplitFused miss count could instead imply about 712 MB/s host-to-device, so
ring wait and promotion utility remain hard runtime gates. Full details are in
`runs/ds4/20260720_lowbit_recovery/g129_ssd_wrap_cpu_implementation_report.md`.

Published DS4 branch tip is
`801a6d8fee17d1fd18fa4fe83fec3f750501fd7e`. The runtime and G129 harness
commits are respectively `16273d4a1d9b648a5878223e7d3ecd3a8d233672` and
`415ed980da69bad304d98e77b1851076a7ae06a6`.

## 11. Kernel conclusion

G129 already transfers and consumes Q1_0 in packed 18-byte/128-weight form.
Creating another binary memory layout does not reduce bytes. A DP4A/MMVQ-style
kernel remains a conditional compute optimization, but it is not a substitute
for activation-aware quality recovery and it has no measured end-to-end gain.

## 12. Next ordered actions

1. Run one separately authorized G129 control trace n=1 to obtain activation
   samples from an exact-IQ2 teacher path.
2. Train/recover binary weights for one expert and compare route-weighted expert
   output error against current Q1.
3. Run one `promotion_ssd_2_0` structural SSD-WRAP safety n=1.
4. Stop unless it proves waves, pairing, working-set accounting, next-call
   causality and zero stale/drop/failure/current-token SSD-to-VRAM.
5. Run a long quality safety immediately; stop on L0/L1.
6. Only after L2/L3 run balanced independent n>=3 processes.
7. Compare against a full/open exact-IQ2 control. Use G73 only as the historical
   closed reference.

## 13. Permanent run rules

- one DS4 server/GPU runner at a time;
- three quiescence windows before runtime;
- no claim from n=1;
- exactness, provenance and contamination are gates;
- record prompt, SHA, model/sidecar receipts, commit, executable, runner,
  harness, build fingerprint, parameters, environment, timings, output and
  L0-L3 grade;
- do not use repeat flags as graders;
- render long HTML and test form plus JavaScript behavior;
- stop L0/L1 branches early and record the negative;
- never confuse cache320 with K320 or a closed mask with full/open routing.
