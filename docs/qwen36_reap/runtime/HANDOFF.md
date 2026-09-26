# Isolated runtime handoff

CURRENT REVISION: deferred observer fix, CPU build/static ABI PASS; numerical re-gate pending. Candidate llama.dll `018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef`; full source patch `8d7e3e8c51de3a6e274ccfcc4d0b89f8eae789eac934024a0612d89b1c8c5755`. Parent gate109 proved OLD candidate OFF bit-identical on first fixture only, but OLD trace failed maxabs0.8196802139 despite equal greedy tokens. That failure is not waived. See TRACE_OBSERVER_FIX.md and observer-fix-build.json. No mask/calibration acceptance until re-gate.

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

Capture chains cb_eval without replacing the caller's request semantics. It remembers strided actual top-k ID tensors without requesting an intermediate callback stop, then reads those IDs and the EXISTING final normalized-weight reshape/scaled tensor together after the final-weight callback. No added CONT or consumer rewiring remains; the full CUDA routing fusion can reach its normal two materialized outputs. One backend read per requested tensor, followed by stride-aware host unpacking. Validates all 40 layers before emitting any route rows: E/K, unique/in-range IDs, membership in mask, finite/nonnegative weights, positive normalized/scaled sum (1e-4 tolerance). Trace synchronizes scheduler execution and inhibits some optimizations; performance measurements must be untraced.

An independent read-only review found three P2 edge cases (failed-batch state, trace create race, failed-constructor guard), all fixed. Recovery after metadata/compute exceptions returns GGML_STATUS_FAILED; cleanup itself is guarded against allocation exceptions. Static review is not runtime correctness evidence.

## Memory / candidate configuration (parent executions only)

Header-only audit in gguf-memory-layout.json:

- Routed: 18,119,393,280 aligned bytes, 120 tensors, 32,212,254,720 parameters.
- Nonrouted: 2,289,183,232 aligned bytes, 613 tensors, 2,448,355,968 parameters.
- 32 slots per layer demand-cache payload: about 2,264,924,160 bytes. Payload plus ALL nonrouted weights is about 4.24 GiB, excluding KV/recurrent state, workspace, staging, alignment adjustments and OS/display pressure. This is a candidate budget, not proof of 12 GB fit.

Source-verified CLI candidate for parent-owned isolated launch:

```
-m D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf
-ngl 99 --cpu-moe --no-mmap --moe-expert-cache 32 --moe-expert-cache-inserts 8
-np 1 -c 2048 -b 128 -ub 128
```

`--no-mmap` is MANDATORY for this demand-GPU configuration: routed host weights must be pinned. The original M3 guard at src/llama-moecache.cpp:515 explicitly requires it. Parent gate106 hit this baseline guard before numerical results; this was not a REAP numerical failure. The earlier CLI proposal omitted this required flag and is superseded by the command above. No benchmark or identity claim follows from the failed attempt.

Process-only environment candidate: LLAMA_MOE_DEMAND_GPU=1, LLAMA_MOE_CACHE_BATCH=1, LLAMA_MOE_ELASTIC=0. Parent must scrub inherited experimental env knobs and explicitly record the final environment; no server/live settings were edited. `--cpu-moe` matches only routed `(up|down|gate|gate_up)_(ch|)exps`, not shared/router tensors. `--moe-expert-cache` is slots/layer, not MB. Static demand has N slots, fallback has N+1; elastic uses batch3/min24 and ignores initial N. Use original Q4/no MTP. No inference of speed or quality follows from this budget.

## Validation and build state

- Source assembly: 3007 copied files SHA256 verified; all 3 M3 overlay hashes and live llama.dll match build manifest. Full per-file ledger source-provenance.json. CUDA base comparison finds only ggml/src/ggml-cuda/mmvq.cu different; NO import from the experimental CUDA base.
- Seven Python reference/static tests PASS (including the deferred-observer fusion-boundary check), including two synthetic numeric identity fixtures, excluded-dominant and underflow-tie cases, bad ID/finite/keep<8 rejection. They do NOT execute C++ or GGML and do NOT satisfy the two real web/DOM/MCP runtime identity gates.
- Packaged unified patch reverse `git apply --check` passed before build; patch-manifest.json records exact hashes.
- Parent granted isolated CPU-build window after review. build-runtime.cmd builds all llama and llama-common objects against four manifest-verified original GGML import libraries and exact original M3 DLLs (CUDA12.6/SM86). No nvcc or GPU execution is necessary. No stale llama object reuse despite internal context layout change. Build artifacts are under lab/build; baseline/candidate runtime directories are separate. Build completed successfully (225 targets, job pwsh-95, exit0). Candidate llama.dll SHA256 `5f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566`; baseline `34857a71a89787637a2f844eee702bcda7b9af8f7c8f157b9e90f0bd89b44791`. Static ABI gate passed (job pwsh-96): 263 exports each, no added/removed exports or unresolved consumer imports. Only candidate llama.dll replaced; all other candidate DLL/EXE files remain exact baseline. Build warnings: inherited C4297 llama.cpp, C4244 graph out_ids/model/quant, C4834 sampler; none from llama-reap.cpp. Source snapshot has no Git metadata, so CMake's version probes print nonfatal Git warnings. See lab/evidence/build-runtime.log and abi-review.json.
- Parent GPU gate113 (numeric_gate_04) now PASSED with observer-fix DLL018c: original vs OFF, trace and all-kept bit-identical on both fixtures (8x248320 full logits each, maxabs0); single-token prefill plus decode also bit-identical. All40layers aligned to actual inputs; keep7 rejected by the specific runtime guard. Prior gate109 drift remains documented. Reduced-mask exclusion, agent quality, random-matched controls and performance are separate gates; these numerical tests do not establish100/200token/s.
- No live server stop/settings changes, model weight edits, GPU inference, commits or pushes by this worker.

## Frozen numerical helper (compiled, never executed here)

Parent-owned `docs/qwen36_reap/gate/native_gate.cpp` was compiled as C++17 target `reap-native-gate`, linked against freshly rebuilt llama-common/llama, max 4 jobs. Initial job pwsh-99 failed because metadata referenced obsolete `common_params.flash_attn`; gate owner corrected it to `flash_attn_type` and renewed the freeze. Retry pwsh-100 passed (2 targets, no helper compiler warnings). See `lab/evidence/build-helper.log` and `build-helper-retry1.log`; CMake's nonfatal source-snapshot Git warnings remain.

Staging job pwsh-101 passed static import checks (no unresolved named imports) and created:

- `D:\ds4_work\qwen36_reap_lab\build\gate-baseline\reap-native-gate.exe`
- `D:\ds4_work\qwen36_reap_lab\build\gate-candidate\reap-native-gate.exe`

Current arms use identical helper SHA256 `97c826a2efd65628459ba8fbd65c8523e9ad2d1477ce38d4d728f0917a28d4b4`, identical fresh llama-common.dll SHA256 `55a4924a199d245d9c6f99ec17899a8ddfacd740b26f637af41ae0e7d4fddaf3`, and original unchanged GGML DLLs. Only llama.dll differs between arms (original vs patched hashes above). Exact `bin-baseline` remains untouched; helper arms are separate. Current helper source SHA256 is `70c54033444565866874fcdc9316b5c8b1bad53cddbcbb27e0bf130bad486009`.

Parent reported gate job102 failed BEFORE MODEL LOAD because LLAMA_EXAMPLE_COMMON did not expose --no-warmup. No numerical data resulted. Gate owner changed only the parser mode/comment to LLAMA_EXAMPLE_COMPLETION and renewed the freeze. Rebuild pwsh-103 PASS (2 targets); refresh/import-check pwsh-105 PASS. Both helper executables were replaced; all other arm files were hash-verified unchanged. Prior helper `bcd19c9709afb26eef7e245e030471402be64aead214c60cc30647275029e87a` and manifest are archived at `lab/evidence/helper-before-bcd19c9709afb26e`. Retry build log: `lab/evidence/build-helper-retry2-completion.log`. This worker did not execute either helper revision.

Parent then reported gate106 hit the original baseline demand-GPU pinned-weight guard before numerical results. The gate owner added only `--no-mmap` to the helper unary allowlist, and parent supplies the flag explicitly in both arms. Rebuild pwsh-107 and refresh/import-check pwsh-108 PASS; previous `f65c269ec310233ee05040d50b6683990b1c34ed6ab177bccdb9cfded6a564b3` helper and manifest are archived at `lab/evidence/helper-before-f65c269ec310233e`. Build log: `lab/evidence/build-helper-retry3-no-mmap.log`. No runtime source or non-helper arm files changed; no execution by this worker.

`helper-build-manifest.json` lists all staged files, hashes, build-command fingerprint and limitations; duplicate manifest and dumpbin imports/exports are under `lab/evidence`. Neither helper was invoked by this worker, even with --help; the parent-owned failed attempt is recorded above. Full F32-logit/greedy-token identity, phase joins and GPU gates are parent-owned and still unproven by this worker. Runtime source and candidate DLL hashes stayed frozen during helper integration.
