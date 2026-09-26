# Isolated runtime handoff

## Controls and data contract

All controls are process-local environment variables, read once. Absent/empty `QWEN36_REAP_MASK` and `QWEN36_REAP_TRACE` means OFF. Never change a running server's settings to use this patch.

- `QWEN36_REAP_MODEL_SHA256=671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`: parent-hashed original 20,419,565,568-byte Q4 model. Runtime validates format/mask equality but honestly labels hash `launcher_asserted`; it does not rehash 20 GB.
- `QWEN36_REAP_TRACE=<fresh JSONL path>`: exclusive-create only. One trace context at a time. Existing files are never truncated. Startup warmup batches are excluded, and trace is flushed after every complete validated batch. Failed setup can leave a new partial file; select a new path rather than overwrite it.
- `QWEN36_REAP_MASK=<manifest JSON>`: schema_version=1, model_sha256, expert_count=256, top_k=8, layers={"0":[IDs],...,"39":[IDs]}. Each layer requires 8..256 unique integer IDs in 0..255. Duplicate object keys are rejected. Extra provenance metadata is allowed.

Header `record_type=header` includes schema_version=1, architecture=qwen35moe, modeltag, model_sha256, expert_count=256, top_k=8, layer_count=40, routed_scale and expected_weight_sum. Data records use ONLY `record_type=route`: batch_id (global monotonic ubatch ID), decode_call_id (global monotonic llama decode call ID), layer, phase=unknown, token_position, token_index, token_id, seq_ids, ids[8], weights[8]. No footer or batchend record. `record_type=error` means reject the trace. Parent's mask builder accepts the contract and rejects incomplete layer/token coverage.

Phase is deliberately unknown: a one-token prefill is not decode. Parent may annotate using actual prompt token count and request byte ranges after flush. Position is the first actual ubatch position plane, not a guessed token counter. Text-token capture only, default single-sequence context; MTP, 41-layer variants, multi-sequence contexts, pipeline graphs and pruned last-layer graphs are rejected. No multimodal calibration claim; web/DOM/MCP text fixtures are the intended first gate.

## Implementation details

Files changed in lab: src/CMakeLists.txt, llama-context.cpp/h, llama-graph.cpp, new llama-reap.cpp/h. Donor and M3 overlay unchanged.

Masking creates a separate GGML input bias (0/-infinity) and a fresh add tensor before softmax. Only nonidentity layers use masked logits for expert ranking; this prevents excluded zero-probability experts tying underflowed retained probabilities. Full-pool identity adds no input/bias/ranking operations. Existing K=8, gather, normalization/clamp and routed scaling remain. Shared experts and original model weights are untouched. No compacted GGUF is generated, so masks alone do not reduce the on-disk/host model.

Capture chains cb_eval without replacing the caller's request semantics. It reads strided actual top-k IDs and an explicit CONT of final normalized/scaled weights that the expert path actually consumes. One backend read per requested tensor, followed by stride-aware host unpacking. Validates all 40 layers before emitting any route rows: E/K, unique/in-range IDs, membership in mask, finite/nonnegative weights, positive normalized/scaled sum (1e-4 tolerance). Trace synchronizes scheduler execution and inhibits some optimizations; performance measurements must be untraced.

An independent read-only review found three P2 edge cases (failed-batch state, trace create race, failed-constructor guard), all fixed. Recovery after metadata/compute exceptions returns GGML_STATUS_FAILED; cleanup itself is guarded against allocation exceptions. Static review is not runtime correctness evidence.

## Memory / candidate configuration (not executed)

Header-only audit in gguf-memory-layout.json:

- Routed: 18,119,393,280 aligned bytes, 120 tensors, 32,212,254,720 parameters.
- Nonrouted: 2,289,183,232 aligned bytes, 613 tensors, 2,448,355,968 parameters.
- 32 slots per layer demand-cache payload: about 2,264,924,160 bytes. Payload plus ALL nonrouted weights is about 4.24 GiB, excluding KV/recurrent state, workspace, staging, alignment adjustments and OS/display pressure. This is a candidate budget, not proof of 12 GB fit.

Source-verified CLI candidate for parent-owned isolated launch:

```
-m D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf
-ngl 99 --cpu-moe --moe-expert-cache 32 --moe-expert-cache-inserts 8
-np 1 -c 2048 -b 128 -ub 128
```

Process-only environment candidate: LLAMA_MOE_DEMAND_GPU=1, LLAMA_MOE_CACHE_BATCH=1, LLAMA_MOE_ELASTIC=0. Parent must scrub inherited experimental env knobs and explicitly record the final environment; no server/live settings were edited. `--cpu-moe` matches only routed `(up|down|gate|gate_up)_(ch|)exps`, not shared/router tensors. `--moe-expert-cache` is slots/layer, not MB. Static demand has N slots, fallback has N+1; elastic uses batch3/min24 and ignores initial N. Use original Q4/no MTP. No inference of speed or quality follows from this budget.

## Validation and build state

- Source assembly: 3007 copied files SHA256 verified; all 3 M3 overlay hashes and live llama.dll match build manifest. Full per-file ledger source-provenance.json. CUDA base comparison finds only ggml/src/ggml-cuda/mmvq.cu different; NO import from the experimental CUDA base.
- Six Python reference/static tests PASS, including two synthetic numeric identity fixtures, excluded-dominant and underflow-tie cases, bad ID/finite/keep<8 rejection. They do NOT execute C++ or GGML and do NOT satisfy the two real web/DOM/MCP runtime identity gates.
- Packaged unified patch reverse `git apply --check` passed before build; patch-manifest.json records exact hashes.
- Parent granted isolated CPU-build window after review. build-runtime.cmd builds all llama and llama-common objects against four manifest-verified original GGML import libraries and exact original M3 DLLs (CUDA12.6/SM86). No nvcc or GPU execution is necessary. No stale llama object reuse despite internal context layout change. Build artifacts are under lab/build; baseline/candidate runtime directories are separate. Build completed successfully (225 targets, job pwsh-95, exit0). Candidate llama.dll SHA256 `5f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566`; baseline `34857a71a89787637a2f844eee702bcda7b9af8f7c8f157b9e90f0bd89b44791`. Static ABI gate passed (job pwsh-96): 263 exports each, no added/removed exports or unresolved consumer imports. Only candidate llama.dll replaced; all other candidate DLL/EXE files remain exact baseline. Build warnings: inherited C4297 llama.cpp, C4244 graph out_ids/model/quant, C4834 sampler; none from llama-reap.cpp. Source snapshot has no Git metadata, so CMake's version probes print nonfatal Git warnings. See lab/evidence/build-runtime.log and abi-review.json.
- MUST remain pending until parent executes: binary baseline vs rebuilt no-mask/no-trace on two fixed fixtures; no-trace vs trace full logits and greedy tokens; all-kept identity; trace/metadata complete; malformed masks fail closed; graph reuse and single-token prefill; then reduced mask vs random-matched pool quality/performance. Default bitwise identity target; any logit tolerance must be predeclared, not retrofitted to pass.
- No live server stop/settings changes, model weight edits, GPU inference, commits or pushes by this worker.
