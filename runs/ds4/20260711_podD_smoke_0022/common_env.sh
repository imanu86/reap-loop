# W50 static K23 in-engine PACE + S1 monitor
export DS4_PACE=1 DS4_PACE_S1=1
export DS4_PACE_WARMUP=50 DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23 DS4_PACE_KEEP_MAX=96
export DS4_PACE_BREATH_EVERY=999999 DS4_PACE_RELEARN=0 DS4_PACE_ROTATE=0
export DS4_PACE_WRAP=1 DS4_PACE_WRAP_ROTATE_DELTA=1 DS4_PACE_DEBUG=1
MODEL=/root/models/ds4-2bit.gguf
PROMPT=/root/coffee_prompt.txt
CLI="-m $MODEL --prompt-file $PROMPT -n 600 -c 8192 --temp 0 --nothink --ssd-streaming --ssd-streaming-cache-experts 1024"
