# 2026-07-10 Pod Cache1024 Warmup Replay

Purpose: replay the old two-phase session-learning recipe recovered from Claude
artifacts, because direct K23 cache1024 was fast but still degenerated.

## Setup

- Pod: RunPod `zcdysuk9q0c08d`, RTX 3090 24GB, terminated after artifact pull.
- Model: `/root/models/ds4-2bit.gguf`.
- Runtime binary: copied from local WSL `/root/ds4/ds4`.
- Runtime SHA256: `8746a87386b9727bb8dfa57abc8266994867bd42afcb28894a339ec9d79e2d74`.
- Prompt: recovered `frontpage_prompt.txt`, compact coffee-shop page, 819 bytes.
- Cache: `--ssd-streaming-cache-experts 1024`.
- Build mask: recovered `build_session_mask.py`, rank by cumulative gate mass.

Recipe per W:

```bash
DS4_SPEX_TRACE_ROUTING=route_W.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 \
  ds4 -m ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 1024 -c 2048 --nothink --temp 0 \
  -n W --prompt-file frontpage_prompt.txt > tw_W.txt 2> p1_W.diag

python3 build_session_mask.py route_W.csv sess_W.txt 23
cat frontpage_prompt.txt tw_W.txt > p2prompt_W.txt

DS4_REAP_MASK_FILE=sess_W.txt \
  ds4 -m ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 1024 -c 3072 --nothink --temp 0 \
  -n $((1000-W)) --prompt-file p2prompt_W.txt > trest_W.txt 2> p2_W.diag
```

## Results

| W | Phase 1 gen | Phase 2 gen | Output checks | Finding |
| ---: | ---: | ---: | --- | --- |
| 50 | 2.03 t/s | 14.60 t/s | `doctype=2`, `</html>=1`, `<form>=1`, `<script>=1`, `alert=2`, `repeat=0` | Reproduces the old useful regime: fast and functionally complete-ish, though not perfectly clean because it restarts once. |
| 130 | 2.30 t/s | 16.24 t/s | `doctype=1`, `</html>=0`, `<form>=1`, `<script>=1`, `alert=0`, `repeat=1` | Fast but fails quality, looping on `document.addEventListener("DOM"...`. |

Finding: the old positive behavior is real for the compact prompt at W50 under
cache1024, but it is sensitive to the freeze point/prompt/build. W130 did not
replicate as clean in this run. This supports keeping the cache1024 old claims
with explicit caveats rather than treating them as a general K23 quality result.
