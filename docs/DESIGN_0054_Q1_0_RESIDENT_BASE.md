# Design 0054: Q1_0 resident routed-expert base

Status: implementation and measurement plan, no speed or quality claim.

Date: 2026-07-17

## Measured motivation

The current native-Windows SOTA is G73 at 4.986667 server decode t/s over three
independent clean processes. Every process reported:

- 2,752 tier calls and 16,512 selected expert routes;
- 5,820 VRAM hits;
- 10,692 prefill-snapshot RAM hits;
- zero IQ2 cold/SSD routes and zero IQ2 SSD bytes.

G107 repeated the same placement shape with a strict residency classifier:
17,439 VRAM and 28,641 snapshot-RAM routes over three requests, with zero SSD
cold routes, zero IQ1 substitutions and byte-identical outputs. It transported
227,030,335,488 IQ2 bytes from the snapshot to VRAM.

Therefore an IQ1 cache that activates only after an IQ2 SSD miss cannot improve
the closed SOTA stack. The active target is the number of bytes transported
for RAM-resident selected experts.

## Representation choice

Existing GGUF Q1_0 type 41 uses 18 bytes per 128 weights, or 1.125 effective
bpw. For one complete DS4 routed expert:

- gate/up: 1,179,648 bytes;
- down: 1,179,648 bytes;
- total: 3,538,944 bytes = 3.375 MiB.

All 10,240 routed experts in layers 3 through 42 require 33.75 GiB. The current
IQ2/Q2 expert footprint is 6.75 MiB, so Q1_0 halves route H2D bytes before any
kernel or scheduling improvement.

IQ1_S remains useful as proof of the mixed-representation resolver and the
probation/promotion policy. It is not the active all-cold resident format:
46.875 GiB for the routed pool did not remain physically resident on this
64 GiB Windows host.

## First useful placement

Do not begin with the 33.75 GiB full-domain allocation. G73/G107 published a
30 GiB dynamic arena with capacity/residency for 4,551 `(layer, expert)` slots.
This is not a K4551 mask: `static32` is the VRAM replacement budget, while the
4,551 value is the measured host-arena slot count. The same slot count in
Q1_0 occupies:

`4551 * 3.375 MiB = 15,359.625 MiB = 14.9996 GiB`.

The first candidate therefore replaces the 30 GiB IQ2 host snapshot with a
15 GiB Q1_0 snapshot populated from the same runtime-observed arena entries,
while preserving:

- the 320 protected IQ2 experts in the existing 2.11 GiB VRAM cache;
- router-selected expert IDs and weights;
- the same router selection and request-scoped arena-learning protocol;
- G73 split-fused and no-default-sync route transport;
- the authoritative IQ2 GGUF for validation and later promotion.

There must not be a second 30 GiB IQ2 snapshot in this arm. A duplicated host
copy would hide the memory benefit and make the comparison uninterpretable.

## Resolver contract

For every selected expert, representation resolution is ordered:

1. protected IQ2 VRAM slot;
2. exact IQ2 RAM representation, only in explicit control arms;
3. resident Q1_0 candidate slot;
4. fail closed or use the separately declared IQ2 authoritative fallback.

The Q1_0 representation cannot change router selection, route weight or expert
ID. Q1_0 and IQ1_S have separate sidecars, binders, counters and dispatch.
Unknown type, shape, offset, residency or tensor identity aborts the Q1_0 arm.

The first candidate is intentionally approximate. Output exactness against IQ2
is not expected and must not be used as the quality gate. Quality uses the
existing L0-L3 rubric with at least three independent processes and retained
raw outputs.

## Quality-source boundary

No BF16/F16, FP16 or richer-than-IQ2 DS4 Flash routed-expert checkpoint has
been identified in the verified local inventory or the documented R2
inventory. The current authoritative model is already IQ2_XXS for gate/up and
Q2_K for down. Therefore a Q1_0 sidecar derived from that file is valid for
structural binder, kernel, residency and transport measurements, but it is not
a quality-optimal Q1_0 quantization and cannot support a general quality claim.

The first converter must record `derived_from_IQ2=true` and
`not_quality_optimal=true`. A quality-grade Q1_0 verdict remains blocked until
a provenance-bound richer checkpoint is available. This does not block the
transport A/B: it only prevents confusing an engineering smoke with the best
quality the representation might achieve.

## Full-domain placement

If the 15 GiB candidate confirms the H2D lever, extend toward all 10,240 Q1_0
experts. Do not assume that a 33.75 GiB allocation can be entirely pinned under
WDDM: G106 measured a practical combined pinned/shared boundary near 30 GiB and
an additional 11 GiB IQ1 allocation failed.

The full-domain design is consequently segmented:

- as much Q1_0 as the measured WDDM pin budget permits in stable pinned RAM;
- the small remainder in ordinary resident RAM;
- a bounded pinned DMA staging ring for pageable Q1_0 slots;
- IQ2 residual or authoritative bytes only for protected/promoted experts;
- no per-token SSD Q1_0 read in the steady-state target.

Promotion may later use Q1_0 for the current token and make IQ2 available for a
subsequent token. Same-token Q1_0-to-IQ2 switching is not part of 0054.

## Runtime patch sequence

1. Q1_0 type, dequant/dot and fail-closed dispatch.
2. Separate Q1_0 sidecar binder and compiling qwarp32 gate/up/down kernels.
3. Dedicated selected-load/file source, real kernel dispatch and counters.
4. Provenance-bound Q1_0 sidecar converter and validator.
5. Structural one-expert/layer-range smoke.
6. Fifteen-GiB runtime-observed arena preload and transport profile. The arena
   backing map and byte geometry must be Q1_0-specific, while the 320-entry
   IQ2 VRAM cache remains an independent representation and is never populated
   with Q1_0 bytes.
7. Independent-process A/B and L0-L3 grading.
8. Only after a positive lever result, segmented full-domain residency.

Research patch provenance:

- step 1: `imanu86/moe-aggressive-commit` commit `9251e5e`;
- step 2: `imanu86/moe-aggressive-commit` commit `2c32417`;
- blocker fixes and normalized patch chain: commit `0871821`;
- step 3 file provenance, selected-load and qwarp32 dispatch: commit `31a02a0`;
- step 4 synthetic converter/validator protocol: commit `48fcab4`.

The native Windows integration through step 3 is committed on the dedicated
branch: the corrected foundation is
[`6a44578`](https://github.com/imanu86/ds4-win/commit/6a44578), and the
fail-closed file source plus selected dispatch is
[`12e78e0`](https://github.com/imanu86/ds4-win/commit/12e78e0). Its direct
`pread` selected loader is deliberately a structural smoke path, not a
performance candidate: the steady-state target must serve Q1_0 from resident
host slots. No real Q1_0 routed-expert runtime or quality claim exists yet.

The standalone sm_86 geometry benchmark is committed as
[`a58bab0`](https://github.com/imanu86/ds4-win/commit/a58bab0). It validated
512 block pairs with zero dot/dequant error, detected the old invalid byte-base
formula by at least `0.04296875`, and passed Compute Sanitizer with zero errors.
Its roughly `0.01 ms` kernel timing is kernel-only and is not a DS4 throughput
claim.

## Measurement gates

### Structural smoke

- opt-in environment only; env-off IQ1_S and G73 unchanged;
- sidecar identity, type, shape, offsets and hashes validated;
- selected route count equals Q1_0 dispatch count;
- loads, slots and failures internally consistent;
- no direct SSD-to-VRAM expert transition;
- output retained, but no speed or quality verdict from `n=1`.

### Performance A/B

- three independent clean processes per arm;
- same prompt, max tokens, G73 stack and model provenance;
- control is the G73 30 GiB IQ2 snapshot; candidate is a Q1_0 snapshot built
  from the same observed arena entries and capped at the equivalent 4,551
  slots;
- report server decode, prefill/TTFT, end-to-end time, route bytes, H2D submit
  count, H2D wait, GPU utilization and host residency;
- any outlier triggers a repeated three-process arm, not selective deletion.

### Quality

- at least three retained outputs;
- L0-L3 grading by document validity and task completion;
- no verdict from repetition flags or token-level heuristics;
- no claim that Q1_0 is lossless.

## Stop conditions

Stop and document the arm if any of these occur:

- Q1_0 route geometry differs from the verified 128-weight / 18-byte layout;
- sidecar identity cannot be bound to the main DS4 checkpoint;
- Windows trims/page-outs invalidate physical residency;
- route H2D bytes do not fall approximately with representation size;
- decode does not improve enough to justify quality loss;
- malformed output appears in a quality arm.

Negative results remain part of the ledger. A failed 15 GiB candidate does not
invalidate Q1_0 as a full-domain storage format; it invalidates the tested
compute/transport composition.
