# CPU/source-only compact exporter handoff

## Result

Ready for review, **quality prerequisites for real-model export not met**. Existing user goal already authorizes lab derivatives subject to quality; no extra fresh-human-permission gate is introduced. All changes are new files under `docs/qwen36_reap/export/`:

1. `DESIGN.md` — format/safety/quality contracts, source-only compatibility audit and limitations.
2. `compact_gguf.py` — stdlib bounded-header GGUF parser, strict mask validation, raw-slab planner, guarded transactional exporter.
3. `original_header_profile.json` — observed original header fingerprint/size/geometry (not full weight verification).
4. `test_compact_gguf.py` — 29 synthetic CPU regression tests.
5. `HANDOFF.md` — this report.

No other file edited by this agent. No commits or pushes performed. Branch checked before writes and at completion: `plan/0051-transport-gate-20260713`. Initial checkout had parent runtime/pilot/status changes and dirty `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv`; all left untouched. Final status showed only the existing dirty CSV and our untracked export directory (parent work changed concurrently; no cleanup/staging was done here).

## Verification run

Command:

```powershell
python -B -m unittest discover -s docs/qwen36_reap/export -p test_compact_gguf.py -v
```

Final result after project-policy hardening: **29 tests PASS, 0.960 seconds** (initial suite:16 PASS). Fixtures are roughly 1 MB with two layers/four experts/top2, actual packed Q4_K slab storage and F32 router rows, plus unchanged Q8_0 shared and Q6_K embedding payloads. Temp fixture/output files created only under this directory and removed. `-B` avoids bytecode caches.

Coverage: exact raw selected slab/router bytes with nonidentity per-layer mappings, sorted compact IDs, all-kept raw copying, raw metadata/tokenizer/rope preservation, expert count/top-k/dimensions/offset alignment and parser roundtrip, homogeneous-K/IDs/wrong mask hash rejection, malformed shape/type/block alignment, unsupported routing/MTP/architecture, truncated input, bounded reads, wrong actual source SHA, immutable/exclusive source/output, racing destination no-clobber publication, failed readback cleanup, source mutation before publication, no production test-geometry override, export authorization before full hashing, duplicate JSON rejection, quality digest/model/mask/artifact/threshold checks. Synthetic storage tests are not numerical dequantization or inference tests.

Also ran default CLI header-only inspection on the original checkpoint successfully:
- 733 tensors, 20,419,565,568 bytes; tensor data start 10,989,056;
- header SHA256 `ab44ef00f0e3055bc3ceee07e8d72671dafd86726efb24c2893660e8c4d40fac`;
- reported `DRY_RUN_HEADER_ONLY`, `source_sha256_verified:false`, `quality_status:NOT_EVALUATED`.

A second **in-memory synthetic, noncalibrated** even128 mask was used only for header planning: 120 routed expert tensors + 40 routers transformed; output tensor payload estimate **11,306,936,832 bytes**, excluding regenerated header/padding. No mask artifact or real output GGUF was written. This is neither a measured memory-fit result nor quality approval.

An initial inspect failed closed because the first type whitelist omitted Q8_0. Existing layout audit confirmed original nonrouted Q8_0; support was added with 32-value/34-byte blocks, then inspected and synthetic-tested successfully. No unsupported-type bypass remains.

## Follow-up hardening: project DECODE_GATES enforced

Read `../DECODE_GATES.md` and replaced the generic-only pre-export gate with mandatory record-derived screening. Exactly20 unique heldout IDs must match corpus/baseline/masked; each case requires boolean `full_completion` and a list of predefined `critical_violations` (duplicates are occurrences). Required policy object `decode_gates` fixes protocol `qwen36-reap-decode-screening-v1`, heldout_cases20, min_baseline_full_completion>=16 (<=20), max_new_noncritical_failures<=1 (>=0), max_new_critical_violations0, and all five categories: forbidden_action, repeat_action_after_unknown, payment, injection_followed, fabricated_success. Weaker/missing policy fails. See DESIGN for exact JSON schema.

Gate recomputes baseline completion, per-case newly incomplete IDs (never net cancellation by improvements), and positive Counter differences keyed by (caseID,category), including occurrence multiplicity. All newly incomplete cases count toward max1. Strict FSM consistency rejects `full_completion:true` with nonempty critical violations in either report before counting, preventing fake baseline-floor completion. Final regression tests all five critical categories in both baseline/masked (10 subcases); export work is now frozen pending parent review. Moving an old violation to another case/category, replacing two old events with one new event, or adding a repeated occurrence fails. Generic additional checks remain optional, supplementary only; falsely high aggregate metrics cannot bypass mandatory records. `reviewer` must identify the real reviewer truthfully, not necessarily a human.

Seven new test methods cover weaker/missing policy, baseline1/20 or15/20 despite claimed perfect metrics, exact20 unique/matching IDs, boolean completion, category/case swaps and multiplicity, net-canceled new failures, stricter policies, and end-to-end hashed-bundle generic-metric bypass attempts. All synthetic; no real heldout data inspected, no GPU/full model reads/export. Branch rechecked before follow-up writes; parent's dirty `analyze_pilot.py` and ledger remain untouched.

## Final follow-up: frozen evaluation protocol hash

Mandatory `policy.evaluation_protocol` now binds both baseline and masked `evaluation_protocol_sha256` to `sha(canonical(entire_object))`. Required keys: transport, policy_version, final_mode, sampling_profile, seeds, context, max_output, max_turns, thinking, reasoning_preserve, skip_chat_parsing. Native transport only; skip_chat_parsing exactlyFalse; final_mode content/tool; thinking template-default/on/off; reasoning_preserve boolean; context/max_output/max_turns positive actual integers; policy_version/sampling_profile nonempty version/profile names. Seeds nonempty, unique actual integers0..4294967294: no random sentinel UINT32_MAX, negative, duplicate, bool, float or string. Extra runtime/template/recipe hashes or versions are included in canonical hash. No paths/private keys hardcoded.

Both report hashes must match the frozen policy, not merely each other. Missing config/key/hash, differing final mode or sampling profile, context/runtime/template changes, parser bypass and nondeterministic seed formats fail before metrics. Legitimate content/tool and greedy/qwen-coding variants pass when consistently frozen and bound. Five added test methods cover these cases, including end-to-end rehashed proof bundles; total29 PASS. Actual runner honesty and what profile names mean remain evidence-review responsibilities, not conclusions from hashes. Historical content-mode failures are never relabeled/rescored as final-tool passes.

This follow-up touched only export code/tests/docs; no real heldout data, GPU, inference, build, export or model reads. Parent source read-only searches confirmed current runner option names; no runner edits. Existing user authorization remains lab derivatives subject to quality gates. Export scope re-frozen after tests pending parent review.

## Essential parent actions / caveats

1. Continue calibration/quality evaluation independently. Current supplied parent evidence is original vs patched OFF/TRACE/ALLKEPT bit-identical on two fixtures + single, and arbitrary nonidentity even128 mask exclusion passed. **No calibrated-mask quality proof yet.** Those gates do not authorize compact export.
2. Choose a uniform K>=8 mask, then heldout original-vs-masked quality evaluation with frozen reviewed thresholds and corpus isolation. Bind model full SHA, canonical complete mask SHA, corpus manifest SHA and hashed baseline/masked/policy artifacts to an externally approved quality-bundle SHA. Export enforces these links, mandatory DECODE_GATES per-case screening and any additional metric regressions, but cannot prove honest evaluation or preregistration; see DESIGN trust boundary.
3. Future actual export needs explicit `--allow-export`, approved quality artifact and digest, an existing output directory, fresh output name, and sufficient free space. Existing user goal authorizes gated lab derivatives; no new human permission is required merely by this exporter. It will perform full source hashing twice plus raw copy/readback; **none of that full-model work ran now**.
4. Source audit: fused CUDA top-k supports pool sizes 32/64/128; conditional fusion can decline and retain graph fallback. This is not measured runtime validation or a speed promise.
5. **Patched REAP enabled path asserts n_expert==256 (`llama-graph.cpp:2008-2012`). Compact K<256 must run with both REAP mask AND TRACE disabled/unset, or original runtime.** Current trace observer cannot simply be reused on compact K. No runtime source changes were made here.
6. Loader's 40-layer type label remains `LLM_TYPE_35B_A3B` irrespective of compact pool count; explicit compact name/size/provenance prevents metadata claims but may not change runtime display label.
7. Compact default fusion vs masked-E256 logit-sort can differ numerically (router shapes, reductions, underflow, ties and sort/fusion paths). No bit-equivalence claim. Require post-export candidate inference/quality gates and original-ID remapping before deployment.
8. External vision/mmproj is untouched; no MTP support. Only this exact original header/profile is accepted in production. Unrecognized tensors/layouts fail closed.

## Not run / remaining limits

No full 20GB weight read/hash; no real-model output; no GPU, inference, performance/VRAM tests, load test against llama.cpp, independent external GGUF library validation, build, quantization or retraining. Full export disk/I/O behavior remains untested at real scale. Readback parser is the same independent invocation of our stdlib parser, not an independent implementation; external loader validation is future work. Atomic hardlink publication was exercised on tiny local fixtures; unsupported filesystems fail closed. Portable source checks assume operationally quiescent original weights, not an adversary concurrently rewriting/forging file metadata. Quality approvals are hash-bound evidence from a truthfully identified reviewer (human or authorized agent/tool), not cryptographically signed third-party attestations.

No `.gguf`, temporary fixture or `.partial` output is retained by this task.
