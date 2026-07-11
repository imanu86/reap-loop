# 0033 tiered-hysteresis smoke — frozen mask (PACE warmup->freeze), greedy, 2-bit native
export DS4_PACE=1 DS4_PACE_S1=1
export DS4_PACE_WARMUP=50 DS4_PACE_KEEP=${KEEP:-12} DS4_PACE_KEEP_MIN=${KEEP:-12} DS4_PACE_KEEP_MAX=96
export DS4_PACE_BREATH_EVERY=999999 DS4_PACE_RELEARN=0 DS4_PACE_ROTATE=0
export DS4_PACE_WRAP=1 DS4_PACE_WRAP_ROTATE_DELTA=1 DS4_PACE_DEBUG=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_SPEX_STATS=1
MODEL=/root/models/ds4-2bit.gguf
PROMPT=${PROMPT:-/root/stage0033/cyberpunk_prompt.txt}
CACHE=${CACHE:-256}
NTOK=${NTOK:-300}
BIN=/root/ds4_0033/ds4
D=/root/smoke_0033
