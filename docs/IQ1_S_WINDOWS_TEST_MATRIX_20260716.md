# IQ1_S Windows Test Matrix (2026-07-16)

## Purpose

This document is the fail-closed test matrix for introducing IQ1_S routed
experts into native-Windows DS4. It separates measured evidence from planned
work and prevents structural or contaminated runs from entering the SOTA
ledger.

The first implementation uses IQ1_S only as a routed-expert sidecar. Router,
embedding, attention, norms, shared experts, and every non-routed tensor remain
from the existing 2-bit primary model.

Implementation commits:

- [69faac4 - validate IQ1_S transport on a real expert](https://github.com/imanu86/ds4-win/commit/69faac4)
- [c9d91bc - add the fail-closed IQ1_S expert sidecar](https://github.com/imanu86/ds4-win/commit/c9d91bc)

## Claim Rules

- `MEASURED` means that the stated artifact and numerical result exist.
- `PLANNED` means that a runner or protocol exists, but no result is claimed.
- An `n=1` run may establish only structural safety, runtime use, or an exact
  deterministic regression for that one invocation.
- A contaminated run may establish deterministic output exactness when its
  expected and observed hashes match. Its TTFT, throughput, latency, cache,
  miss-rate, and SOTA fields are invalid.
- No quality verdict may be inferred from `repeat_flag`, output hashes, token
  count, throughput, or automatic text heuristics.
- Quality requires at least `n>=3` per arm and recorded L0-L3 grading of every
  preserved output.
- No IQ1_S result is lossless, quality-equivalent, or SOTA until the relevant
  quality and clean-performance gates pass independently.

## Evidence Matrix

| ID | Status | Scope | Protocol | Evidence / acceptance gate | Claim allowed |
|---|---|---|---|---|---|
| G75 | MEASURED | Real IQ1_S expert component | One real layer-3 routed expert, `n=3`; 200 timing iterations per repetition; RTX 3060 | Numerical validation PASS; real size and H2D/kernel measurements below | Component correctness and transport reduction only |
| G75b | MEASURED | IQ1_S runtime patch with sidecar disabled | G73 prompt and settings; `n=1`; system quiescence skipped | Exit 0; sidecar not observed; expected output SHA equals observed output SHA | Env-off exactness for this deterministic invocation only |
| G76 | PLANNED | First end-to-end IQ1_S sidecar safety | `n=1`, temp 0, nothink; short hello prompt; sidecar enabled | Provenance, runtime markers, nonzero route counters, coherent counters, zero failures, minimal output-health predicate | Structural safety only; no quality or SOTA claim |
| G77 | PLANNED | Matched quality A/B | Main 2-bit vs IQ1_S sidecar; same build, prompt and settings; `n>=3` per arm; temp 0, nothink | Quiescent host, preserved raw outputs, recorded L0-L3 grade for every sample | Prompt-scoped quality comparison only after grading |
| D1 | PLANNED | Dynamic-tier env-off regression | Mixed-tier code present but disabled; clean `n>=3` exactness run against frozen baseline | Expected output hashes and runtime counters match; no IQ1 path observed | Regression safety, not performance |
| D2 | PLANNED | IQ1_S cache in ordinary RAM | Sidecar cold-read baseline vs persistent IQ1 host-cache A/B | Same outputs; measured SSD reads, host-cache hit rate, bytes moved, latency and memory | Transport effect only |
| D3 | PLANNED | Mixed hot/cold execution | Hot experts execute as 2-bit VRAM hits; cold experts execute as IQ1_S for the current token | Same selected IDs and gate weights; zero silent fallback; mixed-kernel counters consistent | Structural mixed-format safety only |
| D4 | PLANNED | Deferred 2-bit promotion | `SSD IQ1_S -> RAM IQ1_S -> current-token IQ1_S -> pinned RAM 2-bit -> next-token VRAM 2-bit` | `forbidden_cold_ssd_to_vram=0`; no format change during current token; promotion counters reconcile | Tier-transition correctness only |
| D5 | PLANNED | Mixed-tier quality | Static sidecar vs dynamic tier vs main 2-bit, matched prompt suite | `n>=3` per arm and recorded L0-L3 grades; no verdict from hashes/repetition | Quality conclusion limited to measured suite |
| D6 | PLANNED | Clean dynamic-tier performance | Candidate must first pass D1-D5; clean host; matched provenance and settings | TTFT, prefill, decode t/s, SSD/RAM/H2D bytes, hit/miss and promotion rates | SOTA comparison only after all gates pass |
| D7 | PLANNED | SPEX-assisted prefetch/promotion | Add prediction only after transport and mixed tier are stable | A/B against D6; prediction precision/recall, avoided misses, wasted bytes, cancellation rate | Incremental SPEX value only |

## Measured Results

### G75: Real-Expert IQ1_S Component Gate

Source artifact: `persadian/DeepSeek-V4-Flash-IQ1_S-XL`, revision
`7e641d4869031039314f0ae48c79a4f2a7862230`, full-file SHA-256
`b049d1eb34c068f19ab007b33c22a7d758b578bf2b10d9276e79654f85d35047`.
The sampled payload is layer 3, expert 0, gate + up + down.

| Metric | Measured value |
|---|---:|
| Repetitions | 3 |
| Current expert bytes | 7,077,888 |
| IQ1_S expert bytes | 4,915,200 |
| Byte reduction | 30.55% |
| Current pinned H2D mean | 0.270876667 ms |
| IQ1_S pinned H2D mean | 0.188821000 ms |
| H2D time reduction | 30.30% |
| IQ1_S x Q8_K kernel mean | 0.689211667 ms |
| Q8_K maximum absolute error | 6.55651093e-7 |
| Validation | PASS |

Interpretation: the real IQ1_S expert reduces transferred bytes and measured
pinned H2D time by about 30%. The CUDA dot-product agrees with the reference
within the preregistered tolerance. This is not an end-to-end decode result and
does not establish model quality.

### G75b: Env-Off Exactness Regression

G75b executed the G73 cyberpunk HTML prompt with the IQ1_S implementation
compiled but the sidecar disabled.

| Field | Measured value |
|---|---|
| Gate kind | `structural-safety` |
| Repetitions | 1 |
| Server exit | 0 |
| Sidecar runtime observed | false |
| Sidecar route calls | 0 |
| Expected content SHA-256 | `31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8` |
| Observed content SHA-256 | `31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8` |
| Completion tokens | 64 |
| System quiescence | skipped / contaminated |
| Quality eligible | false |
| SOTA eligible | false |

Interpretation: env-off behavior is bit-exact for this deterministic `n=1`
invocation. The recorded 0.943499 t/s is deliberately excluded because the run
was contaminated and cannot be compared with G73 or any SOTA result.

## Planned End-to-End Gates

### G76: Structural Sidecar Safety

G76 is the first real end-to-end use of the routed-expert sidecar. Expected
sidecar provenance:

- Path: `D:\ds4-models\DeepSeek-V4-Flash-IQ1_S-XL.gguf`
- Bytes: `61,540,805,344`
- SHA-256: `b049d1eb34c068f19ab007b33c22a7d758b578bf2b10d9276e79654f85d35047`
- Main-model SHA-256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`

Required gates:

1. Hash and size are measured by the general harness, not merely declared by
   the wrapper.
2. The runtime validates checkpoint identity, tensor layout and routed-expert
   coverage before inference.
3. Exactly one final IQ1_S summary is present.
4. Route calls and selected-load counters are nonzero and internally coherent.
5. Selected-load or I/O failure is fatal for IQ1_S; no main-model-offset or
   host-pointer fallback is permitted.
6. `failures=0`, exit code is zero, and the short output passes a minimal health
   predicate without being treated as a quality grade.
7. The result itself records `quality_eligible=false` and
   `sota_eligible=false`.

### G77: Quality A/B

G77 compares the existing main 2-bit routed experts with the IQ1_S sidecar.
Both arms use the same binary, source fingerprint, primary model, prompt,
sampling (`temp=0`, nothink), context, token budget and runtime settings. The
only intended difference is the routed-expert source.

- Prompt: complete single-file cyberpunk AI programming-shop HTML page.
- Repetitions: at least 3 per arm.
- Maximum generation: 2,048 tokens; context: 4,096.
- Host quiescence is mandatory.
- Every raw output is retained and receives a recorded L0-L3 grade.
- The runner may preserve throughput descriptively, but it declares no
  performance winner.
- The result remains `pending-recorded-human-l0-l3-grading` until all grades
  exist.

## Dynamic Tier Target

The dynamic path to test after G76/G77 is:

```text
SSD IQ1_S
  -> persistent IQ1_S cache in ordinary RAM
  -> cold expert executes as IQ1_S for the current token
  -> qualifying expert's 2-bit representation loads into existing pinned RAM
  -> 2-bit expert becomes eligible for VRAM publication from the next token
```

Invariants:

- IQ1_S never occupies the existing pinned 2-bit arena.
- The VRAM expert cache contains only 2-bit experts.
- A cold SSD expert cannot jump directly into VRAM.
- Promotion cannot change an expert's format during the current token.
- Selected expert IDs and router weights do not change.
- Any IQ1_S selected-load failure is fail-closed.
- Layers 0-2 retain their Q2_K down-projection handling.
- SPEX is introduced only after the non-predictive mixed tier passes exactness,
  quality and transport gates.

## Decision Order

1. Close the G76 structural gate.
2. Run and grade G77 before making quality claims.
3. Add ordinary-RAM IQ1_S caching and measure transport in D2.
4. Add mixed current-token execution and deferred 2-bit promotion in D3-D4.
5. Re-run env-off exactness and mixed-tier quality gates.
6. Run clean performance measurements only after correctness and quality pass.
7. Add SPEX prediction as a separately measured optimization, never as a
   prerequisite for basic transport correctness.
