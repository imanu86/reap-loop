# G64 Windows K60 versus G46 Long-Output Quality Protocol

Date: 2026-07-16

## Question

Does the K60 sparse bake preserve long-output functional quality when it runs
the complete measured G46 transport composition, whose short n=3 gate measured
4.553333 t/s versus 4.563333 t/s for the G46 full-model control?

## Fixed Arms

- `g46_full`: `C:\ds4-models\ds4-2bit.gguf`.
- `g63_k60`: `C:\ds4-models\ds4-2bit-k60-mass-full-decode.gguf`.

Both arms use the same complete G46 composition: 30 GiB DynamicArena,
source-parts WRAP with trusted worker checksum, PrefillMassWrap,
ComposePrefillMassTiering, cache 320 LRU with 0.125 GiB reserve,
GPU-resident routes, RouteNoDefaultSync, mass-LFRU tiering (`clock=430`,
replacement budget `16`, minimum frequency `3`, hysteresis `1.25`), disabled
Q8/F16 cache, embedded-row staging and eight REAP prefetch threads.

The K60 arm adds only the embedded sparse-bake authorization, payload/mask
verification and sparse-aware candidate replacement/restore path. SPEX,
external masks, layer stripes, FileQD greater than one and full-model copy are
off in both arms.

## Quality Workload

- Prompt: the existing cyberpunk single-file HTML prompt used by G46/G63.
- Context: `8192`.
- Maximum generation: `4000` tokens.
- Sampling: temperature `0`, nothink.
- Stop rule: identical fixed maximum/context in both arms; record the server
  `finish_reason`. Do not infer quality from early stop, repeat flags or token
  count alone.
- Replication: three independent processes per arm.
- Frozen order: `g46_a, k60_a, k60_b, g46_b, g46_c, k60_c`.

The 4000-token budget is required because the historical full-model positive
control showed that this prompt can remain valid but incomplete at 800 and
2000 tokens and completed at about 3498 tokens. A low grade caused only by a
smaller cutoff is not an admissible quality verdict.

## Grading

Grade every final output with `scripts/functional_grade.py frontpage` and keep
the complete machine-readable detail object.

- L0: non-parsing/non-opening or catastrophic task miss.
- L1: opens but a critical requested feature is broken.
- L2: requested features are present with minor defects.
- L3: complete, functional and clean.

Promotion requires n>=3 evidence. `repeat_flag`, n-gram diagnostics and output
hashes are diagnostics/provenance only. They never determine the L0-L3 verdict.
Different hashes within or between arms are allowed and must not fail the run.

## Safety And Contamination Gates

Before the six long rows, run a short n=1 per-arm safety at context 8192. This
checks the context allocation and complete flag composition without claiming
quality or timing. Do not continue after an allocation, provenance, routing or
contamination failure.

Every measured row must retain:

- executable, runner, harness, source and build-manifest hashes;
- one owned DS4 process and zero process-isolation conflicts;
- quiescence preflight and runtime memory/disk contamination monitoring;
- route calls positive, GPU-route errors zero and default-sync calls zero;
- cache/tiering policy and capacity exactly matching the fixed arm;
- zero snapshot backing misses, forbidden cold SSD-to-VRAM transfers, tier
  failures and tier SSD bytes;
- for K60, embedded payload/mask authorization, sparse candidate replacement,
  one successful base-mask restore and zero restore failures;
- non-empty output, token count, finish reason, content SHA-256, raw response,
  TTFT, decode t/s, WRAP time, H2D and residency telemetry.

G63 short runs reached only 0.10-0.33 GiB minimum Windows available memory.
Therefore the context-8192 safety is a hard gate. Do not disable or weaken the
memory/quiescence checks merely to obtain a row; a failed safety is a measured
capacity result and must be recorded honestly.

## Decision Rule

Report the complete six-grade distribution, median grade and all structural
details. K60 can be promoted as a quality-preserving sparse SOTA candidate only
if it shows no material L0-L3 regression against the paired G46 control. A
throughput or TTFT advantage cannot override a functional grade loss.

