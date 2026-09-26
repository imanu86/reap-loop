# Qwen3.6 REAP native numerical gate (source-only handoff)

Author scope: **code and synthetic/static tests only; no build/test execution, GPU/inference, or model-byte reads by this author**. Parent/runtime child owns build/run status and evidence (reported separately; do not infer it from this handoff). Only this new `docs/qwen36_reap/gate/` subtree was written. No source/CMake/build-file changes, commits, push, or dirty-ledger edits.

This is a **raw-prompt numerical equivalence gate, NOT real-agent quality**. It neither renders chat templates nor executes browser/MCP actions. The two UTF-8 fixtures contain fake observations/actions and no credentials. Passing them is not evidence of task completion quality, mask quality, throughput, or correctness on other prompts. Reduced-mask experiments still require held-out agent-quality checks.

## Files and exact source API audit

- `native_gate.cpp`: standalone C++17 executable using the real `common_params`, common CLI parser and common initialization; not a server/REST client.
- `compare_logits.py`: Python standard-library-only streaming F32LE comparator. No NumPy.
- `test_compare_logits.py`: synthetic-file unit tests plus source-contract checks. Source-string tests do **not** substitute for compilation or a real DLL/API test.
- `fixtures/web_dom_utf8.txt`, `fixtures/recovery_utf8.txt`: fixed raw prompts (including final newline).

Read-only source inspected: `D:\ds4_work\qwen36_reap_lab\source`:

| Contract | Real declaration/implementation inspected |
|---|---|
| Common params, cache slots/inserts, context, callback, warmup | `common/common.h:448-598` |
| Parse common arguments | `common/arg.h:125`, `common_params_parse(..., LLAMA_EXAMPLE_COMPLETION)`; completion inherits common options at `common/arg.cpp:1445-1457` |
| Initialization ownership | `common/common.h:931-950`, `common_init_from_params`, returned object's `model()` / `context()` |
| Tokenization | `common/common.h:1050-1060`, `common_tokenize(vocab, bytes, true, false)` |
| Batch fields | public `include/llama.h:246-271`; allocation/free at `:964-970` |
| Complete last output row | public `include/llama.h:1037-1049`, `llama_get_logits_ith(ctx, -1)` |
| Model/vocab metadata | public `include/llama.h:589-623`, including `llama_model_n_layer_nextn` |
| Why binary reread is necessary | `common/arg.cpp:1818-1826` strips trailing LF from `-f`; helper restores exact binary file bytes **after** common parsing |
| Why explicit no-warmup is mandatory | `common/common.cpp:1511-1545` can perform encode/decode during common init, then clear memory; gate forbids this |
| Actual link target | `common/CMakeLists.txt:54,131-133`: **`llama-common`**, not a target named `common`; vendor JSON includes transitive |

Parser regression correction: parent's first real gate attempt rejected `--no-warmup` before model loading because the initial helper selected `LLAMA_EXAMPLE_COMMON`. The corrected helper selects **`LLAMA_EXAMPLE_COMPLETION`**. `common/arg.cpp:1983-1990` explicitly exposes warmup/no-warmup for completion, not common. All allowlisted flags were audited: thread/context/predict/batch/KV/cache/cpu-moe/ngl options inherit COMMON defaults, model/offline explicitly include COMMON, and `-f` only excludes SERVER. Explicit `--no-warmup` and the post-parse `!p.warmup` check remain mandatory; the fix does not hide/remove the flag. Two regression tests pin this parser scope and statically audit every allowlisted option against the exact isolated `common/arg.cpp` (the source audit skips explicitly when that source path is unavailable). There are now 21 synthetic/static tests; this author has not executed them.

No llama/private structs are invented or copied. Uses the public `llama_batch`, explicit positions and sequence 0. No private REAP header/import is required. The existing common callback defaults must be null; helper does not replace the runtime's internal trace callback. All calls are single-sequence, target-only, with no draft/MTP. Model metadata must be `qwen35moe`, 40 layers, 256 experts, top-8; `llama_model_n_layer_nextn(model)` must be zero.

## Parent/runtime integration proposal (NOT executed here)

Runtime child `08cb8dd6` owns build integration; parent owns GPU tests. This snippet is a **proposal only**, not a new or modified CMake file:

```cmake
# In a parent-authorized integration location with the existing targets available:
add_executable(qwen36-native-gate "C:/Users/imanu/source/repos/reap-loop/docs/qwen36_reap/gate/native_gate.cpp")
target_compile_features(qwen36-native-gate PRIVATE cxx_std_17)
target_link_libraries(qwen36-native-gate PRIVATE llama-common llama)
```

The `llama-common` target exports common includes and `vendor::nlohmann`; `llama` exports the public API/ggml requirements. Match the runtime build's compiler, CRT and common export/import defines. The parent selected **fresh-built matched-header llama-common.dll on both arms**, original baseline `llama.dll` versus rebuilt candidate `llama.dll`, and unchanged original GGML DLLs. Do not independently rebuild/swap the common or GGML dependencies between arms. Do not overwrite `D:\ds4_work\bin_q36m3`.

Gate executable deployment must use parent-owned isolated arm directories and a recorded DLL search-path strategy. Record actual loaded module paths and hashes, not just PATH intent. Parent must verify ABI/export compatibility before loading each arm; successful compilation alone does not establish baseline DLL compatibility. Initial baseline llama DLL SHA256 known from runtime design: `34857a71a89787637a2f844eee702bcda7b9af8f7c8f157b9e90f0bd89b44791`.

Native source freeze was sent to parent before compilation. No build/test command was executed by this author. Suggested parent-only static test command, from repository root:

```powershell
python -B -m unittest discover -s docs/qwen36_reap/gate -p 'test_*.py' -v
```

Tests use temporary synthetic files inside the gate folder and remove them; `-B` avoids bytecode. Run after the parent's build window as requested. Test coverage includes strict tolerances, relative term, full-vocabulary non-top changes, finite checks, truncation/trailing bytes, shape/token/settings mismatch, top1 ties, teacher forcing, divergence suppression, chunk boundaries, signed-zero bit differences, exclusive reports and exactly-one-token prefill metadata.

## Required invocation and fixed settings

Parent/launcher must independently hash the actual local GGUF and verify:

- Model: `D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf`
- SHA256: `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`

`--gate-model-sha256` is an explicit **launcher assertion**, not independent helper verification. Metadata states `launcher_asserted_not_hashed_by_helper`; do not describe this as the helper hashing weights. No download is supported; common offline is enabled and model must already exist. Parent should remove inherited `LLAMA_ARG_*` variables before running (common CLI honors environment). Launch each sample in a fresh process: cache/REAP state is process-global. Preserve the same cache, hardware, backend, thread and precision setup across arms.

Example, **not run** (all options and values are separate argv elements; no `--opt=value`):

```powershell
# Parent sets these only within its isolated launch environment, never globally:
$env:LLAMA_MOE_DEMAND_GPU = '1'
$env:LLAMA_MOE_CACHE_BATCH = '1'
$env:LLAMA_MOE_ELASTIC = '0'
# Keep LLAMA_MOE_PHASE_CACHE/massa/freeze-related flags identical and record them.
# Set REAP flags separately for each arm below.
& .\qwen36-native-gate.exe `
  -m D:\models\qwen36moe\Qwen3.6-35B-A3B-Q4_K_M.gguf `
  -f C:\Users\imanu\source\repos\reap-loop\docs\qwen36_reap\gate\fixtures\web_dom_utf8.txt `
  --no-warmup --no-escape --offline --no-mmap `
  --cpu-moe --moe-expert-cache 32 --moe-expert-cache-inserts 8 -ngl 99 `
  -c 4096 -b 512 -ub 128 -n 8 `
  --gate-model-sha256 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7 `
  --gate-run-id original-web-greedy --gate-out D:\PARENT_EVIDENCE\original-web-greedy
```

Cache 32 means **slots per host-resident MoE layer**, not MB; inserts 8 is uploads/layer/decode step. Fit is disabled, context defaults 4096, batch/ubatch defaults 512/128, generation defaults 8 and is bounded 1..16. No sampling knobs: argmax of the complete unmodified row, lowest token ID wins exact ties. EOG is recorded but does not terminate this fixed-count probe. Step 0 exports last-prefill logits; steps 1..N-1 export one-token incremental decode logits. There are N selected token IDs and N rows; the last selected token is not unnecessarily decoded because that would yield an extra unused row.

The helper intentionally allowlists the supported common options (model/file, token/context/batch limits, cpu-moe/cache, ngl, thread and KV precision options, offline/no-escape/no-warmup/no-mmap). It rejects unsupported options rather than silently pretending to implement chat/speculation/sampling. No prompt cache, server, REST probabilities, template, grammar or tool execution.

Host-weight configuration follow-up: the parent reported baseline demand-GPU initialization aborting with a requirement for pinned host weights / `--no-mmap`. This is a configuration guard, **not a REAP numerical failure**. The unary allowlist now accepts explicit `--no-mmap`; `common/arg.cpp:2728-2736` maps false to `LLAMA_LOAD_MODE_NONE`, equivalent to the `--load-mode none` mapping at `:2757`. Defaults and invariants are unchanged; the parent must pass the same explicit flag on both arms. The helper records it in argv. A dedicated no-mmap opt-in regression test and the exhaustive upstream option audit cover this flag; there are now 22 synthetic/static tests, not executed by this author.

### One-token prefill, without a hidden BOS

Replace `-f ...` with `--gate-token-id 0` (or another parent-selected valid vocabulary ID). These modes are mutually exclusive. ID syntax is decimal nonnegative and checked against actual `n_vocab` **after loading**. It is used directly, without tokenization or added BOS; prompt length is exactly one. Step 0 is still `last_prefill`, never inferred to be decode from its size. Single-token probes supplement, not replace, the two raw fixtures.

### Teacher forcing and divergent greedy paths

First collect baseline greedy metadata. For each comparison arm use `--gate-teacher D:\PARENT_EVIDENCE\original-web-greedy.json` and the same prompt and `-n`. The helper checks complete metadata, model identity, raw prompt and prompt token IDs, teacher length and every token's vocabulary bounds. It records both actual selected/forced `token_ids` and independent `greedy_token_ids`. Feeding the same teacher tokens ensures the same input prefix at every measured step, even if the candidate would greedily diverge.

Also retain independent greedy runs to test exact generated token equality. On a first greedy divergence at step k, the comparator compares step k (its **input** prefix is still identical), reports the divergence, and marks all k+1 onward logit comparisons **incomparable** with null metrics. Later token reconvergence does not restore prefix equality. It still validates full file shape, finite values and declared top1 on all rows. Teacher forcing does not excuse a top1 mismatch: numerical gate still fails it.

## Arm matrix and callback isolation

For **each of the two fixtures**:

1. **Original**: baseline DLL, no `QWEN36_REAP_MASK` or `QWEN36_REAP_TRACE`.
2. **Rebuilt flags OFF**: rebuilt DLL, no mask/trace. Compare against original.
3. **Capture-only**: rebuilt DLL, no mask, new exclusive `QWEN36_REAP_TRACE`, launcher-verified `QWEN36_REAP_MODEL_SHA256`. Compare against rebuilt OFF and original.
4. **All-kept**: rebuilt DLL, runtime-contract-valid manifest with every layer 0..39 retaining all IDs 0..255, same hash. First run without trace, compare against rebuilt OFF. Optional separate all-kept+trace arm can test the combined path.

No reduced masks before all gates pass. Fresh output prefixes, fresh process and explicit `--no-warmup` for every arm. This isolates callback capture from common initialization warmup. `common_params.cb_eval` is not overwritten. Runtime trace records remain phase `unknown`; the helper's `decode_calls` records explicit semantic prefill/decode, one-based helper call ordinal, position ranges and seq_id=0 for an external join. `steps` additionally map each exported row to its helper call ordinal. Verify runtime decode-call IDs/positions against these records before labeling trace phases; do not guess by ubatch size. With warmup forbidden, a one-token prefill remains semantically prefill. Numerical equality alone does not prove route trace completeness/invariants: parent/runtime's separate route audit must cover all 40 layers, IDs, weights, graph reuse and sequence handling.

## Artifact contract and overwrite protection

`--gate-out PREFIX` creates **two new files** using OS exclusive creation (`_O_EXCL`/`O_EXCL`), never truncating existing files:

- `PREFIX.logits.f32`: row-major `[N, n_vocab]`, complete vocabulary, IEEE754 binary32 encoded explicitly little-endian. Exactly `N * n_vocab * 4` bytes. No header, no top-k truncation or probabilities.
- `PREFIX.json`: schema version 1, `complete:true` only after successful logits close; shape, dtype, endian, model identity/provenance, prompt bytes/text/token IDs, selected/greedy token IDs, steps/decode-call map, effective settings, argv and relevant REAP/cache environment.

A failed run can leave empty/partial reserved files; never reuse that prefix. Empty/partial JSON is not a completed run. No deletion/rollback of existing files occurs, including when only the second file collides. On decode error, nonfinite logits, I/O failure or invalid model the helper exits 2. For successful artifacts metadata includes full raw prompt and local paths; use only approved synthetic/non-secret prompts. Parent records executable/DLL/fixture/manifest hashes in a separate evidence manifest. The helper does not independently verify all runtime module identities.

## Comparator policy — fixed BEFORE real tests

```powershell
python -B docs/qwen36_reap/gate/compare_logits.py `
  D:\PARENT_EVIDENCE\original-web-greedy D:\PARENT_EVIDENCE\rebuilt-off-web-teacher `
  --output D:\PARENT_EVIDENCE\original-vs-off-web.json
```

- Constants: **atol=1e-5, rtol=1e-5**. Per value pass criterion: `abs(reference - candidate) <= atol + rtol * abs(reference)`.
- No threshold CLI override, auto-relaxation or retry-to-pass. Any future tolerance change requires explicit predeclared owner decision and new evidence; never reinterpret an existing failure as a pass.
- Metrics: maximum absolute difference and `maxrel = abs(a-b)/max(abs(a),abs(b))`, with zero/zero defined 0. This symmetric diagnostic is not the asymmetric combined acceptance formula. Nonfinite rows have null aggregate error metrics and fail, even identical NaN/Inf bytes.
- Separate `bit_identical`/`bit_differences` diagnostics distinguish true F32 equality from tolerance passes. Signed zeros can numerically pass but are not bit-identical. Nonfinite identical bytes still fail. Aggregate `bit_identical` is false if any row is not comparable.
- Exact shape/byte-size and raw-prompt/token-prefix/effective-settings/cache-environment checks; all tokens in vocabulary; no partial/trailing bytes; identical selected IDs and identical top1 required. REAP-specific environment necessarily differs between arms and is recorded, not equated.
- Full rows streamed in 16,384-float blocks; no full logit matrix is loaded. Metadata is bounded to 8 MiB. Duplicate JSON keys and nonfinite JSON constants rejected.
- Exit 0 = numerical pass, 1 = numerical/top1/token/finite failure, 2 = malformed/incompatible artifact or I/O failure. Optional report is exclusively created with mode `x`; stdout otherwise. Comparator never overwrites a report.

The parent must retain per-arm evidence and report actual outcomes separately. No numerical result or performance claim is implied by this implementation handoff.
