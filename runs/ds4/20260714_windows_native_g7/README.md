# Windows-native G7 result snapshot

This directory imports the structured `*_result.json` artifacts produced by
`g7_measure.ps1` in the native-Windows DS4 port. It exists so the canonical
REAP experiment ledger can compare the Windows work with the older WSL and pod
runs without depending on a user-local AppData checkout.

## Provenance

- Source repository: `https://github.com/imanu86/ds4-win`
- Source branch: `port/windows-dynamic-arena-0051`
- Latest source commit represented at snapshot time: `2de3aa7`
- Snapshot date: 2026-07-14
- Imported files: 142 `g7_runs/*_result.json` artifacts and one failed-safety JSON
- Imported bytes: 1,543,738

Each JSON retains its own source HEAD, executable hash, harness hash, prompt,
effective `DS4_*` environment, GPU identity, memory preflight and runtime
telemetry where the harness version captured them. Older schema revisions have
blank fields rather than inferred values.

Only structured result/failure JSON is mirrored here. Raw response bodies, stderr,
stdout and high-frequency telemetry remain in the source repository or local
run directory. G22's selected complete evidence bundle is committed in the
source repository at `2de3aa7`.

## Evidence rules

- A safety `n=1` row is mechanism evidence, not a sustained verdict.
- Failed safety gates remain first-class negative rows; they are not discarded
  because a successful result JSON was never written.
- `repeats>=3` inside one server is labelled `same_server_process`; it is not
  silently treated as three independent process launches.
- G19A, G19B and G20 are also aggregated from their six counterbalanced
  independent result JSONs into explicit `measured_independent_n3` arm rows.
- Speed rows without L0-L3 grading remain `speed_only`.
- Expected hash, cache state and replication scope must accompany every t/s
  comparison.

Regenerate the canonical ledger with:

```powershell
python scripts\build_ds4_experiment_ledger.py
```
