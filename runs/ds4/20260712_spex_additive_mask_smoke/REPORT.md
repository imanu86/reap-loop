# 0044 SPEX additive lane smoke - 2026-07-12

Verdict: mechanism PASS, performance FAIL, quality NOT GRADED.

- Build: `make cuda CUDA_ARCH=sm_86`, no compiler errors or warnings.
- Device: local RTX 3060 12GB.
- Base: integrated DS4 full-stack through 0043.
- Prompt: historical Cyberpunk HTML/CSS/JS request.
- Runtime: weighted W50 -> fixed K23, cache256, wrap, no breath/relearn/rotate.
- Treatment: SPX1 cap27, four duplicate-free additions per layer, mandatory
  synchronous WRAP, mask pin and VRAM pin, immediate per-layer lease consume.

The final five post-K23 tokens produced 200 add events covering target layers
3..42. All 772 expert additions completed WRAP; all 772 mask leases were
consumed. VRAM logged 772 promotions and 768 releases. WRAP averaged 16.263 ms
per layer (p95 34.448 ms, max 109.686 ms).

Performance was not viable: the post-K23 chunk measured 0.61 t/s and the final
expert-cache hit rate was 3.40%. The treatment loaded roughly 154 provisional
layer-experts per generated token into a global cache of 256 slots. No 800-token
or n>=3 quality matrix was launched because this mechanism gate already failed
the performance prerequisite. This is not an L0-L3 quality verdict.

Raw local evidence before repo copy:

- `work/spex_additive_async2_stderr.log`
- `work/spex_additive_async_mask.jsonl`
- `work/spex_additive_async_pin.jsonl`
- `work/spex_additive_async2_response.json`
