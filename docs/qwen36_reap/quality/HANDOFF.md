# Round6 CPU-only quality assembler handoff

## Completed / frozen

New files ONLY in `docs/qwen36_reap/quality/`:
- `assemble_evidence.py`: offline strict evidence adapter/assembler, default DRAFT.
- `test_assemble_evidence.py`: invented20-case CPU tests.
- `DESIGN.md`: exact explicit input contracts, trust boundary and approval/publication design.
- `HANDOFF.md`: this result.

Branch checked before writes: `plan/0051-transport-gate-20260713`. No existing files edited. Parent changed mask-planning files concurrently; their files and dirty ledger CSV were untouched. No commit/push, GPU, inference, network, model/header/weights reads or builds. No actual heldout corpus/results or real smoke transcripts were read. Only runner/exporter Python source inspected read-only.

## What it enforces

1. Explicit identity-only heldout corpus manifest, production-schema uniform mask, frozen policy and run index. Corpus20 unique IDs, role×seed×trial complete; `evaluation_protocol.trials:[0]` explicitly required for normal single-trial-per-seed usage. More frozen trials supported without implicit defaults.
2. Completed/eligible native run descriptors, parser bypass false, original model/canonical mask/corpus/protocol SHA bindings, matching runner manifest config/dataset/context/runner identity, hash-pinned raw transcript and reviewed annotation files. Missing loaded-model/mask/protocol provenance is not inferred. Calibration/diagnostic/incomplete/subset evidence cannot approve.
3. Every case and every recorded action index/content must have explicit reviews, with explicit case/action/final critical lists including explicit empty lists. Reviewer nonempty/truthful (may be authorized agent). Annotation files bind transcript raw SHA and each full case canonical SHA. No auto-critical classification or auto-reviewer. Raw failed cases cannot be upgraded/rescored into successes.
4. Native transcript structure and request sampling/seed/output/thinking/final-tool presence crosschecked. AND completion across all replicas; critical Counter aggregation preserves multiplicity. Additional positive difference check by `(seed,trial,caseID,category)` prevents cross-replica cancellation. No net cancellation of new failed cases; mandatory project baseline>=16/20 and max1 new failure enforced by actual exporter routines.
5. Default proof is `decision:draft` and rejected by exporter regardless of apparent passing metrics. Approval requires explicit `allow-approve` and reviewer, then calls **actual current exporter `quality_gate`** on staged hash-bound proof before publishing pass. Additional generic exporter metric checks cannot be bypassed. Report metrics are derived from records only.
6. NEW directory only, exclusive/no-clobber hardlink publication, `quality.json` last; failure cleanup removes only own created files. No raw transcript/requests/responses/reasoning/private errors/local raw paths copied to bundle. Explicit identity-only corpus rejects embedded private payload. Provenance hashes every raw transcript/manifest/review/index/mask file plus assembler/exporter source.

## Tests run

```powershell
python -B -m unittest discover -s docs/qwen36_reap/quality -p test_assemble_evidence.py -v
```

**21 tests PASS, 7.892 seconds.** Then extended the existing trial-schedule test with explicit frozen `[0,1]` but only trial0 present; targeted re-run **1 PASS, 0.113 seconds**. No behavior/source change after that full suite, only the extra assertion.

Coverage includes legitimate synthetic bundle roundtrip through existing exporter gate, default draft rejection, missing reviewer/allow flag, missing seed/trial runs, duplicate cases/runs, missing reviews/case/action/final lists, action indices/content hashes, raw hash tampering, ineligible/parser bypass/incomplete/model/corpus/protocol mismatch, calibration hidden in corpus or raw record, runner config mismatch, baseline floor/new-failure no-net-cancel, per-replica critical anti-cancellation, raw historical failure cannot be upgraded, aggregate AND/Counter behavior, actual additional exporter gate rejection, exclusive destination, publication failure cleanup, all-input immutability and absence of private transcript sentinel content in emitted reports.

All test fixtures/pass bundles were synthetic and removed. No retained `quality.json` pass artifact, raw transcript, real GGUF or temporary directory is delivered.

## Parent integration prerequisites / no guessing

Current native runner transcripts do not contain all required provenance (loaded model/mask SHA, full frozen protocol SHA, run completion/eligibility, parser/reasoning flags). The explicit index descriptor must be produced from reliable wrapper/review evidence; if unknown, assembler errors, not a made-up identity. Original runner manifests are also required and hash-pinned. Parent's current calibration-only wrapper cannot produce an export-approved bundle merely by passing flags; actual heldout/native/same-policy collection and explicit reviews remain future work.

Minimal future input recipe: frozen policy with seeds plus trials `[0]`, exact identity-only20-case heldout manifest, mask, all completed baseline/masked runs' original manifests+transcripts, reviewed annotation file for each run, and explicit linked run descriptors. Annotation skeletons are deliberately NOT auto-filled with zero violations. A responsible human/authorized agent must perform and truthfully attest review. Existing goal already authorizes gated lab derivatives; no extra new human-approval requirement is invented.

Published bundle consists of `corpus.json`, `policy.json`, `baseline.json`, `masked.json`, `quality.json`. Pass proof digest returned is compatible with exporter's `approved-quality-sha256`; not a command to export now. Calibrated-mask quality status remains unapproved, and historical failures remain historical.

## Limits

Hash linkage and structural validation do not prove actual evaluation honesty, heldout isolation, correct semantic scoring, loaded runtime/mask/source identity or chronological preregistration. No signature or Simulator replay claimed. Reviewer must inspect private raw material, including unparsed turns covered by explicit case-level annotations; public outputs do not reproduce it. Only known current native sampling recipes greedy/qwen-coding are supported. Additional metrics beyond derived completion_count/rate and critical occurrence counts require a reviewed adapter change; absence never silently approved. Inputs/output paths must be quiescent; hardlink-capable filesystem required. Graceful cleanup is tested; crash/power-loss can leave partial files without approval marker. Real heldout/model export/inference integration remains deliberately NOT RUN.
