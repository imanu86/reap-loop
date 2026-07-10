# POD2 build & smoke — 2026-07-10

Pod: RunPod **community RTX 3090** `o0gd30ojfacz96` (machine 3n0g1lqe8wy5), image
`runpod/pytorch:1.0.7-cu1290-torch280-ubuntu2404` (cu129, matches driver), sm_86, 128 vcpu / 251 GB RAM,
$0.22/h. CUDA gate-check PASSED first try. Bootstrap from R2 (model + source tarball) + HF (MTP model).

## Verdicts

| Job | Result |
|---|---|
| 1. Build livetree chain base(771a39a8)+0020+0021+0026+0027 (MTP-capable), `make cuda sm_86` | **OK** — 0 warnings, ds4.c md5 3de71ef4 |
| 2. Smoke 0026 demand-admission | **PASS** — 42 admits (layer/expert/evicted/cusum/keep), 6.75 MiB delta page-in per entered expert, off-switch clean |
| 3. Smoke R1 (0027 rewind-exactness) | **PASS** — bit-identical rewind at depth 50/200/300 (first_div=-1); verify_rewind_exactness.py 1/1 PASS |
| 4. Build-test canonical v2 (19 canon patches + 0027) on clean 80ebbc3 | **BUILD FAIL** (expected) — 20 patches apply clean, single missing sibling decl `ds4_gpu_async_read`. Switchover NOT unblocked |
| 5. R2 upload livetree binary + MANIFEST | **DONE** — `*_livetree-62ed2e71-pace0028` (+.meta) + MANIFEST.txt. Canonical binary not uploaded (build failed) |
| 6. Smoke 0028 token sidecar (addendum) | **PASS** — 160 tokens 1:1 with routing, valid CSV quoting/UTF-8 |
| 7. Scope token-exact scene (addendum) | **DONE** — committed to scope `data/20260710_token_exact/` |

## Key engine findings (reported, not fixed)

1. **R1 needs a residency workaround**: `--mtp` forbids `--ssd-streaming`; non-streaming OOMs the 81 GB
   model on 24 GB. Ran with `DS4_CUDA_DIRECT_MODEL=1` (host-mapped, ~6.5 GB VRAM, PCIe-bound but bit-exact).
2. **R1 harness only fires on the session-eval decode path**: the 0027 hook lives in
   `ds4_session_eval_internal`, but greedy + `--mtp-draft ≤1` routes the CLI to
   `generate_metal_graph_raw_swa` (no hook) → `DS4_REWIND_TEST` is a silent no-op. The session-eval path
   requires `temperature>0` OR `mtp_draft>1`; since draft>1 re-enables the checkpoint-clobbering speculation
   that 0027 forbids, the only valid config is **temp>0 + `--mtp-draft 1`** (harness compares argmax
   internally, so temp>0 doesn't change the verdict). See r1_0027/VERDICT.md.

See each subdir's VERDICT.md.
