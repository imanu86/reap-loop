# Activation-aware Q1 recovery POC — layer 3, expert 0

Date: 2026-07-21  
Decision: **NO-GO for scaling the current recovery recipe.**

The first real-trace attempt fits the 12 training samples but fails to generalize to the five request-disjoint held-out samples. Relative to the untouched Q1 baseline, final held-out gate-weighted cosine falls by `0.075105` and held-out gate-weighted NMSE rises by `0.128638` (`+18.41%`). This is overfitting.

## Result

Primary metrics are the scaffold's gate-weighted metrics: expert outputs are multiplied by each route gate before flattening and computing cosine/NMSE.

| Candidate | Fit cosine (12) | Fit NMSE | Held-out cosine (5) | Held-out NMSE |
|---|---:|---:|---:|---:|
| Q1 baseline | 0.603065416 | 0.658783665 | 0.549981518 | 0.698566821 |
| Per-block scale refit | 0.697048191 | 0.518605551 | 0.474891512 | 0.827191646 |
| Scale refit + greedy sign flips | 0.697053184 | 0.518599095 | 0.474876305 | 0.827204474 |

The final recovery improves fit cosine by `0.093988` and lowers fit NMSE by `0.140185` (`-21.28%`). On held-out data it does the opposite. The sign-flip phase adds only `-0.00000646` fit NMSE after scale refit and makes held-out NMSE another `+0.00001283` worse; nearly all of the apparent fit gain and held-out loss comes from scale refitting.

For completeness, unweighted output metrics show the same behavior:

| Candidate | Fit cosine | Fit NMSE | Held-out cosine | Held-out NMSE |
|---|---:|---:|---:|---:|
| Q1 baseline | 0.580003125 | 0.674165753 | 0.555464231 | 0.692594587 |
| Final recovered | 0.658255824 | 0.566930380 | 0.477489685 | 0.826136022 |

The documented `0.811` expectation is a **weight-domain** result, not what the nonlinear expert produces on these activations. This decode independently reproduces the prior weight-domain baseline almost exactly: cosine `0.8113771893`, NMSE `0.3416670567`. The real activation-domain gate-weighted baseline is only `0.603065` on fit and `0.549982` held out.

## Decode and geometry

The decoder reuses `tools/ds4_q1_ref.c` from the research repository. That file is the existing converter's CPU reference implementation pinned to llama.cpp commit `635cdd5fcc5bdeb8ec2e108bb2a40acf62d9039b`; no new IQ2 implementation was introduced.

Actual source tensor types were:

- gate: IQ2_XXS, GGML type 16, shape `(2048, 4096)` after expert extraction;
- up: IQ2_XXS, type 16, shape `(2048, 4096)`;
- down: Q2_K, type 10, shape `(4096, 2048)`;
- Q1 baseline: type 41 sign + fp16 scale, group size 128, for all three tensors.

Each matrix contains `8,388,608` weights. Total expert geometry is exactly `25,165,824` weights.

Decoded artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `teacher_l3e0_fp32.npz` | 100,663,830 | `b21aa4bfa9e7e8c63964fd04d73b492566b3b85cb34e73aa4bd8ada7496c7ab8` |
| `q1_baseline_l3e0_fp32.npz` | 100,663,830 | `a3c7969b395d176fbbd1c73777504670f5c20ac89215795335a627a129c18295` |
| `recovered_l3e0_fp32.npz` | 100,663,830 | `162748fc920c19999014315e722d4e2c9d8aa1e350734bd52146eba10803ada1` |

Each NPZ contains `gate.npy`, `up.npy`, and `down.npy` as little-endian float32, C-order arrays. The independent artifact audit validated ZIP CRCs, NPY headers/shapes, finiteness, Q1 block magnitudes, total geometry, and the weight-domain metrics. Its machine-readable receipt is `verification_receipt.json`.

Input provenance was checked against the capture and model receipts:

- teacher SHA-256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`;
- Q1 sidecar SHA-256: `05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207`;
- Q1 sidecar manifest SHA-256: `e760ed23f6b72c63851c6629b78376e7e722ea24c13112260a1f01264321e768`;
- activation binary SHA-256: `234e0c7597e1c7dfee8c68b44c4d88493336235a5554369dce3b6a0e1a29f4fa`;
- activation JSONL SHA-256: `acc6649846e4c32951f167be3ef7dc7fa8396231762a406e90419b9ba3788779`.

## Trace adapter and split

The captured schema is `ds4_expert_recovery_manifest_v1`, whereas the unchanged scaffold reader requires `ds4_activation_trace_manifest_v1`. `trace_adapter.js` is a narrow, fail-closed adapter. Before writing NPY files it validates source hashes and byte sizes, the complete 64-byte binary header, every vector offset and shape, sample ordering, layer/expert identity, provenance, request-epoch bounds, and every gate weight's float32 bit pattern.

The source does not capture peer top-k expert IDs, so the adapted `top_k` is the semantically known selected expert only, with shape `(17, 1)`. Because the scaffold schema requires one constant request epoch while the real file spans epochs 1–8, the adapter uses cohort ID 0 and preserves all original per-row epochs in `source_request_epoch.npy`. The scaffold's validation was not weakened.

The deterministic split is request-disjoint:

- fit indices: `2,3,4,7,8,9,10,11,12,13,14,15` from request epochs `3,5,6,7`;
- held-out indices: `0,1,5,6,16` from request epochs `1,2,4,8`.

No request epoch appears in both sets.

## Recovery method

Forward evaluation is the scaffold expression, in float32:

`down(silu(gate(x)) * up(x))`

For each row and each 128-weight block, the Q1 signs are fixed and its nonnegative scale is refit by a closed-form least-squares projection of that block's output contribution over the 12 fit samples:

`scale = max(0, sum_s(z_s * t_s) / sum_s(z_s^2))`

Here `z_s` is the signed-block dot product and `t_s` is the matching teacher-block contribution. Gate/up use the captured expert inputs. Down is refit after gate/up, using recovered hidden activations for `z` and teacher hidden activations for `t`.

All three tensors contain `65,536` blocks. Refit scale ratios versus baseline were:

| Tensor | Zero scales | Mean ratio | Maximum ratio |
|---|---:|---:|---:|
| gate | 38 | 1.00447 | 2.69273 |
| up | 32 | 1.00228 | 2.49465 |
| down | 920 | 1.16054 | 5.61024 |

The 920 zeroed down blocks and 5.61× maximum ratio are warning signs for a 12-sample fit.

The bounded sign-flip search follows the scaffold's deterministic row-major behavior: at most eight candidates and two accepted flips per tensor, selected only by lower fit gate-weighted NMSE. It accepted two flips in each tensor after evaluating 2 gate, 3 up, and 2 down candidates. Their tiny fit-only gain did not generalize.

## Resources

Measured on the local CPU runner with 12 OpenMP threads:

- strict trace adaptation: `0.054 s`;
- decode plus writing teacher/Q1 NPZ files: `1.684 s`;
- recovery, intermediate evaluation, and bounded flips: `0.225 s`;
- native runner wall time: `1.979 s`;
- adapter + native runner: approximately `2.033 s`;
- peak process working set: `409,210,880 bytes` (`390.25 MiB`).

These are warm-local-storage POC timings and exclude compilation and the separate full artifact scan. They should not be treated as cold-disk throughput.

## Scaling extrapolation

Seventeen samples are insufficient for a scaling decision beyond rejecting this unregularized recipe. A credible next screening run would need at least **320 routed hits per expert** (`256 fit / 64 held out`), with **640** (`512 / 128`) preferred, request-disjoint and spread across prompts, token positions, gate weights, and prefill/decode regimes. These counts are engineering minima, not a statistical guarantee; rare experts may require substantially more global trace traffic.

Using the measured kernel and assuming roughly linear sample scaling with the same tiny sign-flip bound:

- 320 samples: about `4.2 s` recovery per expert;
- 640 samples: about `8.5 s` recovery per expert;
- adding the current fixed decode/materialization cost gives roughly `6–10 s` per expert;
- all `40 × 256 = 10,240` routed layer/expert pairs would therefore be a lower-bound `17–29 h` wall time on one similar 12-thread host (`~200–350 CPU-core-hours`).

That lower bound excludes trace capture, skew in expert routing, cold I/O, repeated validation splits, and any useful expansion of the sign search. Greedy compute scales approximately with evaluated sign candidates; a materially broader search can dominate the scale-refit cost. Streaming one expert at a time keeps RAM near the measured few hundred MiB. Storing 640 raw 4096-float activations separately for every pair would be about 100 GiB, so a global indexed routed trace should be reused instead of duplicating inputs.

## Decision

**NO-GO.** The held-out signal does not justify scaling this method across experts. The weight decode and real-trace evaluation are sound, but the current closed-form unregularized block refit learns the 12 fit samples and materially damages unseen requests. A future POC should first add scale-ratio regularization/clipping, request-level cross-validation, and validation-based early stopping; only a variant that improves multiple request-disjoint held-out splits should receive a larger trace or model-wide run.

No model file was written or modified, and no commit was created.
