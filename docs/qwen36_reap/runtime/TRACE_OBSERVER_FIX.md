# Trace observer numerical gate failure: minimal source fix

Status: parent-authorized CPU build and static ABI PASS (pwsh-111/112), numerical gate PENDING. New DLL `018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef` staged ONLY in bin-candidate and build/gate-candidate. Helpers/common/baseline remain unchanged. This worker did not execute the fix or use GPU; no claim that the numerical drift is resolved. Prior candidate/manifests remain archived. Current full patch SHA256 `8d7e3e8c51de3a6e274ccfcc4d0b89f8eae789eac934024a0612d89b1c8c5755`.

## Evidence (parent GPU job109, read-only review)

`coordinator/numeric_gate_03/off_web_dom_utf8.comparison.json` reports original vs patched OFF bit-identical full 8 x 248320 logits on the FIRST fixture only. `trace_web_dom_utf8.comparison.json` FAILS declared atol=rtol=1e-5: maxabs 0.8196802139282227, all 8 greedy tokens unchanged, roughly all logits differ. This is a failed numerical observer gate, not an accepted roundoff effect. No threshold changes. Second fixture/all-kept were not reached; no mask quality or throughput claim. Parent reports 8080 valid aligned route rows (202 tokens/layer, 195 prefill + 7 decode), but these rows must not select masks while observer identity fails.

## Concrete mechanism found in source

- Current callback requested an early stop at `ffn_moe_topk` VIEW, then a later stop at an added CONT of weights.
- `ggml-backend.cpp:2076-2100` cuts backend graph views at each requested callback, computes and synchronizes before delivering callback(false); it includes VIEW/RESHAPE nodes, not only arithmetic nodes.
- CUDA routing fusion (`ggml-cuda.cu:3177-3328, 3876-3931`) requires the full SOFTMAX/RESHAPE/ARGSORT/VIEW/GET_ROWS/RESHAPE/SUM_ROWS/CLAMP/DIV/RESHAPE chain, optionally SCALE. Stopping at top-k VIEW necessarily prevents that fusion. Stopping at DIV would also truncate it, so observing the normalized producer before the final reshape is NOT the proposed boundary.
- The trace-only CONT additionally changes topology, allocations and consumer source. Remove it even though its copy is mathematically an identity.
- These are real observer-induced execution changes. Their contribution to the measured 0.82 logit drift is NOT yet causally isolated or proven resolved.

## Minimal two-file revision

1. `src/llama-graph.cpp`: no new op or changed consumer. Annotate the EXISTING final [1,8,N] reshape/scale as `ffn_moe_weights_effective` only when tracing.
2. `src/llama-reap.cpp`: reset per-layer pending ID pointers every ubatch. On callback ask(top-k), save the actual tensor pointer and return false for our observer. Ask true only at final weights; after backend synchronization, read both actual IDs and final actual weights together. Missing pairs/duplicate data/shape/IDs/membership/finite/sum checks still fail closed. Previous callbacks remain chained; any caller-requested intermediate stop remains their behavior. The gate helper disallows such a caller callback.
3. No reconstruction from logits, no mask or weights change, no modifications to GGML/CUDA, and no widening of numerical tolerances.

## Lifetime and materialization checks

IDs are NOT dead after GET_ROWS: `llama-graph.cpp:2231,2249,2357` passes the same selected-expert IDs to subsequent up/gate/down expert matmuls. Weights are expanded into the graph before these consumers (`2149-2150`). Allocator counts children/views (`ggml-alloc.c:741-754`) and frees only after both become zero (`800-817`), so ID backing storage remains live at the final weights boundary. Backend split copies do not remove the need for the IDs before those later consumers/copies.

CUDA fusion explicitly declares IDs VIEW and final weights reshape/scale as the TWO outputs (`ggml-cuda.cu:3912,3924-3930`). Its kernel directly writes `ids[k]` and final `weights[idx]` (`topk-moe.cu:245,270`). Reading these actual outputs at the final callback therefore does not depend on an elided intermediate DIV being materialized. Scheduler callback(false) still runs for the final reshape even when the CUDA implementation fused/skipped internal graph nodes.

## Source/static review result

Seven Python reference/static checks PASS (including no added weight op and deferred-ID stop contract); these do not execute C++ or establish numerical equality. Independent read-only review found no source correctness blocker on the supported gate path. Gate109 header confirms routed_scale=0.0 sentinel, expected_weight_sum=1.0 (no SCALE op).

Two nonblocking caveats were retained, not silently generalized away: a nonunit SCALE node would receive the effective label instead of its old scaled label, so an exact-name external observer could differ (gate helper uses no upstream callback, and this model has unit effective scale). Existing CUDA fusion parser at ggml-cuda.cu:3324 lacks a node_idx<n_nodes check after final reshape. The scheduler graph view aliases the full original backing node array and later expert nodes exist here, so review found no actual allocation overread on this path; broader GGML robustness cleanup is out of scope and no CUDA source was edited.

Built revised source SHA256: llama-graph.cpp `f9a40f3c6f3f56947646e30b4e250e08bb8983dc3c36c927231dc7180afb53f7`; llama-reap.cpp `50a1860594c32af73ac6e9fe7980ac07efe1a1f8ebe976ee518cce3e88d7582c`. New gate DLL is `018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef`; prior failing trace DLL `5f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566` is archived. ABI: 263 matching exports, no consumer import gaps. Only inherited out_ids C4244 warning; no warning from llama-reap.cpp. Full patch reverse-apply check PASS. Artifact evidence: observer-fix-build.json.

## Required next gates (parent-owned)

After source review and separately authorized build: rerun original vs new OFF, then trace vs OFF on the same first fixture using existing exact tolerances/full logits, then second fixture and all-kept identity. Preserve batch/layer/token alignment checks and 1-token-prefill coverage. If drift persists, retain both failures and investigate other callback-induced split/capture/allocation effects; do not reinterpret token agreement as numerical identity. Only then permit calibration masks. No GPU, server or daily process action by this worker.
