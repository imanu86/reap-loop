# 2026-07-10 Pod Cache1024 Direct K23 HTML800

Purpose: isolate whether the old cache1024/RAM-hot pod regime restores the
quality that local cache256 direct K23 failed to reproduce.

## Setup

- Pod: RunPod `zcdysuk9q0c08d`, RTX 3090 24GB, terminated after artifact pull.
- Model: `/root/models/ds4-2bit.gguf`.
- Runtime binary: copied from local WSL `/root/ds4/ds4-server`.
- Runtime SHA256: `0f1f0c7db632c736202b53714795352974fe5f10b39de813106358cd10b6168f`.
- Runner commit: `504023d` (`test: add direct k23 cache1024 variants`).
- Prompt: runner `html` prompt, cyberpunk landing page, 78 prompt tokens.
- Common schedule: W50 full/K0 warmup, then fixed K23, no breath, no prebreath,
  no rotation, no hidden SPEX, `DS4_EXPERT_TIERING=observe` without ID log.
- Cache: `--ssd-streaming-cache-experts 1024` and
  `DS4_PACE_CACHE_TARGET_SLOTS=1024`.

Command shape:

```bash
python3 scripts/run_ds4_exchange_matrix.py \
  --suite quick --prompts html \
  --variants local_k23_cache1024,local_k23_weighted_warmup_cache1024 \
  --runs 1 --warmups 0 --max-tokens 800 --timeout 1200 \
  --port 8014 --out-dir runs/ds4/20260710_pod_cache1024_html800 \
  --model /root/models/ds4-2bit.gguf --ctx 3072 \
  --server-max-tokens 1100 --prefill-chunk 128 --stream
```

## Results

| Variant | Wall | Avg t/s | Last chunk t/s | Prefetch | Quality |
| --- | ---: | ---: | ---: | --- | --- |
| `local_k23_cache1024` | 94.576 s | 14.12 | 24.79 | 6.07 GiB / 219 ms | Incomplete, repeat loop in CSS comments; no `</html>`, no `<form>`, no `<script>`. |
| `local_k23_weighted_warmup_cache1024` | 79.577 s | 16.37 | 24.71 | 6.07 GiB / 63 ms | Incomplete, repeats `Stai attento` comments; no `</html>`, no `<form>`, no `<script>`. |

Finding: cache1024 restores the high-throughput regime but does not by itself
restore quality on this cyberpunk HTML prompt. The old positive result needs the
exact compact prompt/session-learning recipe, not only a bigger cache.
