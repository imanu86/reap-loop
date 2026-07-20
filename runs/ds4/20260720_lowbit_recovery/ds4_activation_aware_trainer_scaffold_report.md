# DS4 activation-aware trainer scaffold

Scope: CPU-only. No GPU, server, DS4 build, download, GGUF/model write, ledger write, or commit. Work files are under `work\activation_aware_trainer`.

## Implemented

- `activation_reader.py`: fail-closed adapter for canonical activation traces.
  - Manifest schema: `ds4_activation_trace_manifest_v1`.
  - Required arrays: `input_vectors`, `route_gate`, `top_k`, `token`, `call`, `request_epoch`.
  - Validates schema, layer, expert, request epoch, file SHA256, shape, dtype, per-file byte budget, finite values, and expert presence in each top-k row.
  - Isolated from GGUF/model layout so the final G129 manifest can map into the same contract.
- `activation_trainer.py`: streaming probe/trainer scaffold.
  - Loads already-decoded teacher arrays from a small `.npz` with `gate`, `up`, `down`.
  - Implements DS4-style expert forward: `down(silu(gate(x)) * up(x))`.
  - Computes output and gate-weighted cosine/NMSE.
  - Provides binary g128 sign+scale and ternary g128 top-k+scale candidates.
  - Provides deterministic coordinate updates for global tensor scale and optional bounded greedy sign flips.
  - Writes delta checkpoints only: scales/sign-flip metadata/threshold-like state, never model tensor copies.
- `test_activation_trainer.py`: synthetic-only tests.
  - Positive manifest/batching path.
  - Negative SHA, shape, epoch, expert, budget, and partial-file cases.
  - Known optimum for binary constant block.
  - Bounded streaming metrics/update on small dimensions.
  - Delta checkpoint confirms no dense model tensor copy.

## Verification

- `python -m unittest test_activation_trainer.py -v`: PASS, 6 tests.
- `python -m py_compile activation_reader.py activation_trainer.py test_activation_trainer.py`: PASS.
- No real activation trace consumed; no full 25M-weight training executed.

## Scaling estimates

Measured/authoritative expert geometry from prior inventory: `25,165,824` weights/expert across gate/up/down.

| Scope | Decoded teacher FP32 | Candidate FP32 | Teacher+candidate RAM | Eval cost per trace sample |
|---:|---:|---:|---:|---:|
| 1 expert | 96 MiB | 96 MiB | ~192 MiB plus batches/temps | ~50.3M MACs for teacher+candidate |
| 64 experts concurrent | ~6.0 GiB | ~6.0 GiB | ~12.0 GiB plus batches/temps | ~3.22B MACs/sample |
| 11,008 experts concurrent | ~1.01 TiB | ~1.01 TiB | ~2.02 TiB plus batches/temps | ~554B MACs/sample |

Inference: train/evaluate experts serially or in small CPU batches. Full concurrent decoded training is NO-GO on 64 GiB RAM. RTX 3060 12 GiB is also NO-GO for 64 decoded expert teacher+candidate residency; one expert is feasible in memory, but GPU use is explicitly out of scope for this task.

Delta checkpoint scale budget if FP16 scales are eventually used: each expert has `196,608` g128 scale slots across gate/up/down, or `393,216 B/expert`; `64` experts are ~24 MiB; `11,008` experts are ~4.03 GiB before optional sparse sign/threshold deltas. Current JSON checkpoints are for small probes only, not production storage.

## GO/NO-GO

- GO: reader/trainer scaffold is ready for a bounded real-trace smoke test once G129 emits the final manifest and SHA contract.
- GO: synthetic fixtures verify fail-closed behavior and deterministic update mechanics.
- NO-GO: any quality claim. No activation trace was available, so no recovery result exists.
- NO-GO: full 25M-weight/expert training or 64/11,008 expert sweeps before defining trace count, per-expert CPU budget, checkpoint binary format, and acceptance thresholds.
