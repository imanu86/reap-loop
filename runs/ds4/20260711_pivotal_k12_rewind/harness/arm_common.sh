ARGS="--binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf \
 --prompt-file /root/pivotal/cyberpunk_prompt.txt \
 --w-values 50 --total 4050 --keep-k 12 --mask-mode weighted --n-expert 256 \
 --cache 256 --ctx-p1 2048 --ctx-p2 8192 --temp 0 --timeout 3600"
PACE_BASE="\"DS4_PACE\":\"1\",\"DS4_PACE_S1\":\"1\",\"DS4_PACE_WARMUP\":\"50\",\"DS4_PACE_KEEP\":\"12\",\"DS4_PACE_KEEP_MIN\":\"12\",\"DS4_PACE_KEEP_MAX\":\"12\",\"DS4_PACE_ROTATE\":\"0\",\"DS4_PACE_RELEARN\":\"0\",\"DS4_PACE_BREATH_EVERY\":\"999999\",\"DS4_PACE_DRIFT\":\"2.0\",\"DS4_PACE_WRAP\":\"1\",\"DS4_PACE_WRAP_ROTATE_DELTA\":\"1\",\"DS4_PACE_DEBUG\":\"1\",\"DS4_PACE_WEIGHTED_SELECTED\":\"1\",\"DS4_PACE_LOG\":\"{rundir}/pace_events.jsonl\",\"DS4_PACE_REWIND\":\"1\",\"DS4_SPEX_TRACE_TOKENS\":\"{rundir}/tokens.csv\""
