#!/bin/bash
# PHASE A: warm t/s probes for the cyberpunk-domain masks (cache32, cyberpunk-wide, -n300).
# One discarded warmup (cold prime), then K23/K48/K64 back-to-back (warm pages).
set -u
RUN=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_domain_calibration
MASKS=$RUN/masks
PROMPT=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/prompt_cyberpunk_wide.txt
OUT=$RUN/sweep
R=$RUN/run_one.sh
mkdir -p "$OUT"; : > "$OUT/progress.log"
bash "$R" warmup      "$MASKS/sessCyber_K48.txt" 300 4096 "$PROMPT" "$OUT" 0   # discard (cold prime)
bash "$R" probe_K23   "$MASKS/sessCyber_K23.txt" 300 4096 "$PROMPT" "$OUT" 0
bash "$R" probe_K48   "$MASKS/sessCyber_K48.txt" 300 4096 "$PROMPT" "$OUT" 0
bash "$R" probe_K64   "$MASKS/sessCyber_K64.txt" 300 4096 "$PROMPT" "$OUT" 0
echo "PHASE_A_COMPLETE $(date -Is)" >> "$OUT/progress.log"
echo DONE > "$OUT/phaseA.done"
