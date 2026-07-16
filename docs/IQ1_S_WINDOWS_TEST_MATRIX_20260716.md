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
- [6054076 - native-Windows mixed-route structural checkpoint](https://github.com/imanu86/ds4-win/commit/6054076)

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
| G74c | MEASURED | Env-off frozen G74 control exactness | `g77_env_off_g74_control_exact`; `n=1`; IQ1/mixed path disabled | Exact historical output SHA matches; no IQ1/mixed telemetry; result SHA recorded below | Regression exactness for this deterministic invocation only; no SOTA claim |
| G76 | PLANNED | First end-to-end IQ1_S sidecar safety | `n=1`, temp 0, nothink; short hello prompt; sidecar enabled | Provenance, runtime markers, nonzero route counters, coherent counters, zero failures, minimal output-health predicate | Structural safety only; no quality or SOTA claim |
| G76b | MEASURED | Native-Windows mixed-route structural smokes | `n=1` layer-3-only and all-layer mixed-route runs; short hello prompt | Mixed calls, hot-main, cold-IQ1, selected-load and failure counters recorded below | Structural mixed-calculation safety only; no quality or SOTA claim |
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

### G74c: Env-Off Frozen G74 Control Exactness

The `g77_env_off_g74_control_exact` control kept IQ1 and mixed routing disabled
and checked the frozen historical G74 output exactly.

| Field | Measured value |
|---|---|
| Gate kind | `benchmark` with `n=1` |
| Repetitions | 1 |
| Tag | `g77_env_off_g74_control_exact` |
| Historical output SHA-256 | `31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8` |
| IQ1 / mixed telemetry | none |
| Observed decode | 4.89 t/s |
| Result SHA-256 | `42a57333dbce70cf20df8651fe1de81e7da9109e0e90b5459289d53682aef4f5` |
| Quality eligible | false |
| SOTA eligible | false |

Interpretation: this is an env-off exactness control for one deterministic
invocation. The observed decode rate is descriptive only because `n=1`; it does
not support a SOTA claim.

### G76b: Native-Windows Mixed-Route Structural Smokes

Commit `6054076` adds native-Windows mixed-route checkpoint evidence. These
runs prove mixed calculation and fail-closed counter behavior only. They are
not quality-eligible, not SOTA-eligible, and do not replace the G77 `n>=3`
quality A/B with recorded L0-L3 grading.

| Tag | Scope | Output | Mixed calls | Hot main | Cold IQ1 | Sidecar selected loads | Failures | Result SHA-256 | Claim allowed |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `g77_mixed_iq1_cold1_layer3_n1` | Layer-3-only `n=1` structural smoke | `Hello!` | 2 | 10 | 2 | 2 | 0 | n/a | Structural mixed-calculation safety only |
| `g77_mixed_iq1_cold1_all_layers_n1` | All-layer 0..42 `n=1` structural smoke | `Hello! How can I assist you today?` | 387 | 1935 | 387 | 387 | 0 | `9de63ea52caf541b1868bbe20f53e2f0bd610ddc0d020facd4a8f582c6d0f00e` | Structural mixed-calculation safety only |

Honesty limitation: commit `6054076` fixture zeroes cold primary weight but
still launches and loads six primary routes, then one IQ1 route. It proves
mixed calculation, not transport reduction. The immediate next patch physically
compacts to five primary routes plus one IQ1 route and one join.

The packed-copy candidate refusal for unequal G74 gate/down sizes is
pre-existing. The historical G74 summary was already stopped, so this is not an
IQ1 regression.

### G86-G95: Cache, GPU Planner, and Quality Checkpoint

Commits `9ff7bc4` and `c8f8678` on
`imanu86/ds4-win:port/windows-dynamic-arena-0051` add the opt-in GPU
cold-route planner and its fail-closed runner contract. The planner preserves
the CPU fixture's first-minimum tie break, queues the five hot IQ2 routes, and
stages the one IQ1_S cold route without a second router readback.

| Gate | Protocol | Total t/s | Server decode t/s | TTFT | Cache/transport | Exactness and eligibility |
|---|---|---:|---:|---:|---|---|
| G86 control | clean benchmark `n=3`, arena 20 GiB | 3.460 | 5.417 | 6.965 s | IQ1_S disabled | Baseline for this prompt/setup; no L0-L3 grade |
| G86 IQ1_S | clean benchmark `n=3`, IQ1 RAM cache 8 GiB | 2.194 | 3.430 | 10.505 s | 87.71% RAM hits; 41.112 GiB SSD avoided | Performance/transport eligible; no L0-L3 grade |
| G89 | structural profile `n=1`, IQ1 VRAM cache 1/layer | invalid | invalid | invalid | 53/640 hits; 587 misses | Exact structural smoke only |
| G90 | structural profile `n=1`, IQ1 VRAM cache 2/layer | invalid | invalid | invalid | 78/640 hits; 562 misses | Exact structural smoke only |
| G91 | structural profile `n=1`, explicit main sync removed | invalid | invalid | invalid | main-sync approximately zero; stall moves to cold submit | Exact structural smoke only |
| G93 | structural profile `n=1`, GPU planner | invalid | invalid | invalid | planner 640/640; wait 0.085 ms; router D2H 6.302 ms; metadata 3.624 ms; failures 0 | Output SHA `c7c8e02137fd31de53dc88a5645b3c6a92ab98d844e42ddcc00c52257d63823d`, identical to G87/G91 |
| G94 planner off | clean benchmark `n=3`, arena 20 GiB, IQ1 RAM cache 4 GiB | 1.543 | 2.073 | 10.588 s | 54.03% RAM hits; planner disabled | Every repeat SHA `e856d9ea88cd1c04f38cecee8b2ecb185f382a35b62495f3bc309fc339c1c004`; clean short performance gate |
| G94 planner on | clean benchmark `n=3`, otherwise identical | 1.630 | 2.223 | 10.458 s | 54.03% RAM hits; planner 10240/10240; wait 1.363 ms; failures 0 | Same repeat SHA as planner-off; decode `+7.23%`, harness `+5.68%`; no L0-L3 grade |

G89-G93 were not used for a speed verdict. In particular, G92/G93 ran while
Windows `ScheduledDefrag` kept physical disk `D:` above 90 percent busy. Their
SSD, H2D, submit, TTFT, throughput, and total-latency measurements are
contaminated and excluded from SOTA. G93 proves only output exactness, planner
counter reconciliation, and elimination of most router/metadata readback.

G94 closes the clean short planner A/B: the GPU planner is faster on this
declared surface and preserves deterministic output. The cache-8 attempt was
aborted when the runtime monitor measured less than 0.5 GiB available Windows
memory; it contributes no timing result. The 64-token G86/G94 outputs still
have no recorded human L0-L3 grading and do not establish quality equivalence
or lossless IQ1_S behavior.

The first G95 protocol (`max_tokens=2048`, context 4096) was stopped during the
control arm at generation 100 after observing 0.47 t/s and only 276/320
resident expert-cache slots. It is an incomplete protocol probe, not a quality
or performance result. The bounded replacement uses `max_tokens=768`, context
1024, `stop="</html>"`, cache 4 GiB, and `n=3` per arm. It is pending after
Windows retained the terminated DS4 process and its memory inside driver
cleanup. The current-build full G74 environment-off exactness rerun also
remains pending because its 30-GiB arena requires at least 32 GiB available
host memory.

Deferred authoritative 2-bit promotion and next-token publication remain
unimplemented. Therefore no current run closes the final residency invariant
`forbidden_cold_ssd_to_vram=0` end to end.

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
- Maximum generation: 768 tokens; context: 1,024; stop on `</html>`.
- Host quiescence is mandatory.
- Every raw output is retained and receives a recorded L0-L3 grade.
- The runner may preserve throughput descriptively, but it declares no
  performance winner.
- The result remains `pending-recorded-human-l0-l3-grading` until all grades
  exist.

## IQ1_S Capacity Arithmetic

- Layers 3-42: `40 * 256 = 10,240` eligible routed experts.
- Measured IQ1_S footprint: 4,915,200 bytes per expert.
- Eligible pool: 50,331,648,000 bytes = 46.875 GiB.
- Hypothetical 43-layer pool at the same footprint: 54,106,521,600 bytes =
  50.391 GiB.
- Real complete sidecar: 61,540,805,344 bytes = 57.310 GiB, including
  non-routed tensors, metadata, and layer-specific layout.
- Saving against the corresponding 67.5-GiB IQ2 pool: 20.625 GiB (`30.56%`).

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
