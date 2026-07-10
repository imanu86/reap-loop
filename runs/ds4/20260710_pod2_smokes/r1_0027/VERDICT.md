# Smoke R1 (0027 rewind-exactness harness) — POD2 2026-07-10

Pod: RunPod community RTX 3090 `o0gd30ojfacz96`, cu129, sm_86, 128 vcpu / 251 GB RAM.
Binary: livetree base(771a39a8) + 0020+0021+0026+0027, `make cuda sm_86`, 0 warnings (ds4.c md5 3de71ef4).
Model: ds4-2bit.gguf (efc7ed60…) + MTP support model ds4-mtp.gguf (3,807,602,400 B, from HF antirez/deepseek-v4-gguf).

## VERDICT: PASS at all tested depths (bit-identical)

| p (snap) | k (depth) | first_div | verdict |
|---:|---:|---:|---|
| 219 | 50  | -1 | PASS (sanity) |
| 400 | 50  | -1 | PASS |
| 400 | 200 | -1 | PASS (verify_rewind_exactness.py: 1/1 PASS, exit 0) |
| 400 | 300 | -1 | PASS (sweep) |

Bit-identical rewind confirmed at depth up to 200 (and 300 sweep). No divergence — proves the
S1_REWIND_DESIGN Risk-R1 claim: in-memory rewind (spec_frontier_restore + PACE checkpoint restore +
ds4_session_rewind) + greedy resume regenerates token-ids exactly, far past the 1–2 tokens MTP exercises.

## Two engine findings required to run this smoke (report, not fix)

1. **`--mtp` needs the model resident; incompatible with `--ssd-streaming`.** Non-streaming residency
   OOMs the 81 GB model on 24 GB VRAM (loads ~16 GiB then `model arena alloc failed … out of memory`).
   Worked around with `DS4_CUDA_DIRECT_MODEL=1` (weights stay host-mapped, GPU reads over PCIe; ~6.5 GB VRAM).
   Slower (~1.3 t/s) but bit-exact — fine for R1 (correctness, not speed).

2. **The 0027 hook lives in `ds4_session_eval_internal`, but the CLI greedy path bypasses it.** With
   greedy temp=0 AND `--mtp-draft ≤ 1`, ds4_cli dispatches to `ds4_engine_generate_argmax` →
   `generate_metal_graph_raw_swa` (a self-contained graph decode loop calling `metal_graph_eval_token_raw_swa`
   directly), which never calls `ds4_rewind_test_hook` → **DS4_REWIND_TEST is a silent no-op**. The
   session-eval path (which carries the hook) is only taken when `temperature>0` OR `mtp_draft_tokens>1`
   OR distributed-coordinator (ds4_cli.c ~922). Since draft>1 would enable the N≥2 speculative-verify that
   clobbers the checkpoint (the exact thing 0027 forbids), the only config that BOTH fires the hook AND
   keeps speculation off is **temp>0 + `--mtp-draft 1`**. Used temp 0.7 seed 42; the harness records
   `sample_argmax(logits)` internally for its pre/post comparison and replays the identical recorded input
   tokens, so temp>0 does not affect the bit-exactness verdict (RNG only picks the emitted token, not logits).
   Recommendation for 0027: either also instrument `generate_metal_graph_raw_swa`, or document that the
   harness requires the session-eval decode path (temp>0 / server) to fire.
