# DS4 coding bake self-contained specification

Status: implementation specification, docs-only. This document converts the
read-only audit into the contract for future coordinated implementation. It is
not a measurement report and it does not authorize GPU runs, WSL runs, DS4
worktree edits, or commits.

## Scope

Build two families of DS4 coding bakes for RTX 3060 12 GB VRAM and 64 GB host
RAM:

1. `K8`: historical keep-8 per routed layer.
2. Coding `K60`, `K62.5`, and `K65`: per-layer keep counts learned from
   uncensored K0 coding traces.

The output must be self-contained. A run must not depend on an external REAP
mask file that can be omitted, stale, or mismatched. Every selection, remap,
source trace, tensor hash, and runtime invariant must be captured in the bake
bundle manifest.

## Historical meanings

`K8` means exactly eight retained experts per maskable routed layer. The G28
artifact described 42 layers, but the GGUF measured in this campaign contains
43 routed layers (`0..42`). The current router mask applies only to layers
`3..42`; layers `0..2` are hash-routed and remain full. The historical mask
therefore cannot be reused as a capacity claim. K8 must not be reinterpreted as
8 percent.

Coding high-K names are percentages of the 256 experts in each maskable layer:

| Name | Keep per layer | Definition |
|---|---:|---|
| `K60` | 154 | `ceil(0.60 * 256)` |
| `K62.5` | 160 | `ceil(0.625 * 256)` |
| `K65` | 166 | `ceil(0.65 * 256)` |

These keep counts apply independently per maskable layer. They are not global
top-N expert counts.

## Learn source

The selection source must be K0 uncensored routing on the coding learn split:

- no REAP mask;
- no static domain mask;
- no previous bake selection;
- router weights recorded for every selected expert;
- trace rows include at least `pos, layer, n, e0..e5, w0..w5`;
- source trace SHA-256 recorded in the bundle manifest.

Expert mass is the sum of router weights per `(original_layer,
original_expert_id)` across all learn traces. Ranking is per layer by descending
mass, with deterministic tie-break by ascending original expert id. If a layer
has sparse evidence, zero-mass experts may fill the tail by expert id only
after all positive-mass experts for that layer.

Global top-mass selection may be computed as a diagnostic only. It must not be
the default bake policy because a global budget can concentrate on a subset of
layers and leave other layers under-populated. DS4 routing requires a valid
expert set at every routed layer.

## Bundle layout

A physical bake is a directory or archive with the following logical contents:

```text
ds4-coding-bake-<kind>/
  model.gguf
  remap.json
  manifest.json
  source_traces/
    learn_manifest.json
    trace_sha256.txt
  reports/
    capacity_preflight.json
    selection_summary.csv
    tensor_hashes.csv
```

The pod is a producer only. It may learn mass, build selections, rewrite compact
payloads, and emit manifests, but Windows is the serving target and the final
serving artifact must be assembled and validated on NTFS.

Primary packaging option: `model.gguf` is an NTFS sparse GGUF reconstructed at
the original GGUF offsets. Retained expert extents are materialized at their
original offsets. Excluded expert extents are left as sparse holes. Dense,
shared, router, metadata, embedding, output, and any required non-routed tensors
remain at their original offsets. Original expert ids and original tensor names
therefore remain valid; the bake constrains selection instead of renumbering
the model.

The portable artifact produced by the pod is a compact pack plus assembler:

```text
ds4-coding-bake-<kind>-pack/
  payload.pack
  extents.csv
  remap.json
  manifest.json
  assemble_sparse_ntfs.ps1
  assemble_sparse_ntfs.py
```

The Windows assembler creates a sparse `model.gguf`, sets sparse-file mode,
writes only retained extents, verifies extent checksums, appends or embeds the
DS4BAKE trailer/manifest, and revalidates the final file. The pack must be the
transfer unit; copying a sparse GGUF through R2 or ordinary archive tooling can
materialize holes and destroy the storage/capacity property.

Alternative packaging option: a physically remapped GGUF stores retained experts
densely and rewrites all relevant metadata so baked expert slots are compact.
That can minimize file holes and may be easier to distribute, but it is riskier
for DS4 correctness because every router, tensor lookup, telemetry path, and
debug trace must translate between baked slot id and original expert id. It is
not the primary option for the Windows pivot.

`remap.json` maps every baked expert slot back to immutable model identity:

```json
{
  "schema": "ds4_coding_bake_remap_v1",
  "bake_kind": "K60",
  "n_expert_original": 256,
  "layers": {
    "3": [
      {"baked_expert": 0, "original_expert": 17},
      {"baked_expert": 1, "original_expert": 42}
    ]
  }
}
```

`manifest.json` is the launch contract. It records:

- source full-model GGUF path, size, and SHA-256;
- bake tool commit and command line;
- source trace set and split assignment;
- keep count per layer and the exact retained original expert ids;
- GGUF tensor names, offsets, byte sizes, quant types, and hashes;
- tier plan: VRAM protected, RAM pinned, RAM pageable;
- expected zero-SSD counters;
- compatible DS4 executable/build fingerprint;
- required runtime flags and environment;
- fail-closed conditions.

For the sparse-offset option, `manifest.json` also records:

- `DS4BAKE` trailer offset, length, schema version, and checksums;
- every retained extent: original offset, length, component, layer, original
  expert id, pack offset, and checksum;
- every excluded expert extent: original offset, length, and expected sparse
  hole status;
- full-file logical size and physical allocated size after Windows assembly;
- the exact mask that must be applied before routing can select experts.

The DS4BAKE trailer contains the audit manifest followed by a fixed binary
retained-expert bitset and a fixed footer. The bitset is `43 * 32 = 1376` bytes
for this model: one bit per original expert id, where one means retained. The
footer records version, layer/expert dimensions, original GGUF logical size,
manifest length, and CRC32 values for both manifest and bitset. This keeps the
native loader small and fail-closed; it does not need a JSON parser to install
the mask. The portable pack retains SHA-256 validation for its payload and
manifest. The trailer is versioned and ignorable by stock GGUF readers that do
not opt into DS4BAKE.

## Expert identity invariants

The original expert id is never semantic decoration. It remains the authoritative
identity for routing, scoring, logging, and quality analysis.

Required invariants:

- Router logits and route traces are interpreted in original expert-id space.
- Every baked tensor records `(original_layer, original_expert_id, component)`.
- Runtime telemetry reports both original expert ids and baked slot ids.
- A routed original expert id may execute only if it has a valid remap entry in
  the current bake snapshot.
- Missing remap, checksum mismatch, layer mismatch, or tensor-size mismatch is a
  hard failure, not a fallback to SSD.
- K8 hash-layer overrides must be represented in the remap, not hidden in a
  one-off script.

## Virtual mask versus physical bake

A virtual mask uses `DS4_REAP_MASK_FILE` or equivalent bias rows to prevent
routing to unselected experts while the full original model still exists
outside the mask. It is useful for mechanism studies and learning curves, but
it is not a self-contained bake.

A physical bake rewrites the model payload and metadata so the runtime cannot
silently recover by consulting the forgotten full model. For this task, only the
physical bake can satisfy the zero-SSD target. Virtual-mask results can seed
candidate selections but cannot be reported as final K8/K60/K62.5/K65 bake
evidence.

For the Windows sparse-offset bake, the physical payload still keeps original
offsets and original expert ids, but excluded experts are holes rather than
valid payload. That makes the loader mask mandatory: the DS4 Windows loader must
read DS4BAKE, validate it, install the retained-expert mask before any routing
decision can execute, and refuse generation if the mask cannot be installed.
The sparse file is physical evidence only when the loader is fail-closed.

## Zero-SSD fail-closed contract

The baked runtime must fail closed during inference:

- no routed-expert read from SSD after startup validation;
- no on-demand load from the full original GGUF;
- no fallback to `pread -> pinned stage -> H2D` for missing baked experts;
- no mmap fault path that masks a missing pinned/pageable tier entry as success;
- explicit nonzero exit or request failure on any required expert miss.
- no route to an excluded expert hole;
- no zero-filled sparse-hole read accepted as a valid expert tensor.

Allowed storage activity is limited to startup open/validation and manifest
loading. The measured decode phase must report zero SSD bytes for routed expert
payload. If the platform cannot distinguish startup validation from decode
traffic, the runner must bracket counters around the measured inference window.

Windows loader requirements:

- detect DS4BAKE trailer before model execution;
- validate trailer checksum, manifest checksum, logical GGUF size, and retained
  extent checksums;
- verify sparse-hole extents for excluded experts when the filesystem exposes
  allocated-range queries;
- apply the retained-expert mask before prefill or decode routing;
- treat any selected expert without a retained extent as fatal;
- report original expert ids in telemetry, even when serving from sparse GGUF
  extents.

Known risks:

- R2 transfer or archive extraction can materialize holes; use compact pack plus
  Windows assembler, then verify allocated ranges on NTFS.
- Extra trailer bytes may confuse tools that assume strict GGUF EOF; DS4BAKE
  must be versioned and loader-gated.
- Extent checksum coverage must include all retained expert components, not only
  the pack as a whole.
- If mask installation fails open, reading an excluded sparse extent may return
  zero-filled bytes; the loader must make that impossible.
- Some Windows copy tools preserve logical length but not sparse allocation
  state; assembly verification must record both logical and allocated size.

## Capacity plan

The pod preflight measured the exact GGUF layout, not an estimate:

- source file: `86,720,111,488` bytes;
- routed payload: `77,913,391,104` bytes across 43 routed layers;
- non-routed/header payload: `8,806,720,384` bytes;
- one `(gate, up, down)` expert triplet in one layer: `7,077,888` bytes;
- layers `0..2` remain full under the current hash-router implementation.

With those three layers full and the mask applied to layers `3..42`, the exact
physical payload plans are:

| Bake | Keep on layers 3..42 | Payload GiB |
|---|---:|---:|
| K8 | 8 | 15.374 |
| K23 | 23 | 19.329 |
| K60 | 154 | 53.870 |
| K62.5 | 160 | 55.452 |
| K65 | 166 | 57.034 |

Thus K8 is not a proven VRAM-only artifact on a 12 GB card. It cannot fit as
currently defined, before KV/cache/runtime buffers. Reducing the hash-routed
layers or other fixed tensors would be a separate implementation and test.

Measured capacity report must include:

- baked routed payload bytes;
- dense/shared/non-routed model bytes;
- router, embedding, norm, output, and metadata bytes;
- KV cache bytes at the configured context;
- CUDA buffers, graph buffers, staging buffers, and stream cache bytes;
- VRAM dedicated and shared bytes sampled before startup, after load, after
  warmup, during decode, and after cleanup;
- host pinned bytes, pageable resident bytes, process private bytes, working
  set, commit charge, and page-cache behavior;
- disk read/write bytes bracketed around decode;
- expert cache admissions, hits, misses, evictions, and direct tier hits.

Promotion criteria:

- K8 VRAM-only is accepted only if the complete operational working set fits in
  VRAM with zero routed expert SSD reads and no expert evictions during measured
  decode.
- K60/K62.5/K65 zero-SSD is accepted only if all selected routed experts are
  served from the declared VRAM/RAM tiers, with zero SSD fallback and no
  snapshot/remap misses.
- If K60 does not fit or fails quality, K62.5/K65 are not automatic upgrades;
  each requires the same capacity and quality evidence.

## Coding protocol

Use ten distinct coding prompts split into six learn prompts and four held-out
eval prompts. The split is fixed per campaign and recorded in
`learn_manifest.json`.

Recommended learn prompts:

1. C: implement and test a bounded parser with pointer/error handling.
2. C++: refactor to RAII and fix iterator/container lifetime issues.
3. Python: transform CSV/JSON data and include unit tests.
4. JavaScript/TypeScript: implement debounce, async API pagination, and tests.
5. SQL: solve window-function and aggregation queries with edge cases.
6. Shell/PowerShell: write a robust file-processing pipeline with quoting and
   error handling.

Recommended eval prompts:

1. HTML/CSS/JS: single-file app with form behavior and validation.
2. Rust: ownership-safe parser or iterator adapter with tests.
3. Go: concurrent worker pool with cancellation and tests.
4. Docker/Git/debug workflow: diagnose and fix a multi-step build/runtime issue.

Every arm requires `n >= 3` valid measured runs on the eval split. No verdict
may be drawn from `n = 1`. A run is invalid if the server fails warmup, the
manifest does not match the bake, the request stops before measurable output
for non-model reasons, or the zero-SSD counters are unavailable.

Grade with L0-L3:

- L0: does not parse, does not run, or catastrophically misses the task.
- L1: runs/opens but key feature or test behavior is broken.
- L2: main functionality works with minor defects, omissions, or polish issues.
- L3: complete, clean, and passes the task-specific checks.

Report all individual grades plus median grade, stop reason, token count,
throughput segments, and zero-SSD counters. The L0-L3 grade is quality evidence;
capacity and zero-SSD evidence are separate gates and must not be inferred from
quality alone.

## Implementation sequence

1. Freeze the coding learn/eval prompt manifest.
2. Run uncensored K0 learn tracing under coordinated GPU ownership.
3. Build per-layer keep lists for K60, K62.5, K65, and historical K8.
4. On the pod, produce compact pack payloads, extents, remap, and manifest.
5. On Windows, assemble the NTFS sparse GGUF at original offsets and embed the
   DS4BAKE trailer.
6. Generate remap and manifest before writing or serving any baked GGUF.
7. Add ds4-win runtime validation that refuses mismatched remap, manifest, or tensor
   hashes.
8. Add fail-closed mask installation before routing and fatal handling for any
   selected expert without a retained extent.
9. Add zero-SSD fail-closed routing for missing baked experts.
10. Run capacity preflight without quality claims.
11. Run `n>=3` held-out eval with L0-L3 grading and counter bracketing.
12. Promote only the smallest bake that passes both capacity and quality gates.

## Non-goals

- Do not promote virtual-mask runs as physical-bake evidence.
- Do not use global top-mass as the production selection policy.
- Do not use K8 as a quality fallback unless it passes the same eval protocol.
- Do not infer fit from byte arithmetic alone.
- Do not allow SSD fallback to preserve apparent correctness.
