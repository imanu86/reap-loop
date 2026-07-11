# 0031 smoke — W50 static K23 in-engine PACE (frozen mask), greedy, 2-bit native serving
export DS4_PACE=1 DS4_PACE_S1=1
export DS4_PACE_WARMUP=50 DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23 DS4_PACE_KEEP_MAX=96
export DS4_PACE_BREATH_EVERY=999999 DS4_PACE_RELEARN=0 DS4_PACE_ROTATE=0
export DS4_PACE_WRAP=1 DS4_PACE_WRAP_ROTATE_DELTA=1 DS4_PACE_DEBUG=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_SPEX_STATS=1
export DS4_EXPERT_TIERING=observe
MODEL=/root/models/ds4-2bit.gguf
PROMPT=${PROMPT:-/root/coffee_prompt.txt}
CACHE=${CACHE:-512}
NTOK=${NTOK:-600}
BIN=/root/ds4_pin/ds4
D=/root/smoke_0031
CLI="-m $MODEL --prompt-file $PROMPT -n $NTOK -c 8192 --temp 0 --nothink --ssd-streaming --ssd-streaming-cache-experts $CACHE"
