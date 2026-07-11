#!/bin/bash
# Phase 2 (run after the sweep): masked route traces (K48/64/91 @ -n 300, cyberpunk-wide),
# narrow coffee contrast (K64 + K0 @ -n 1500), and K0 FULL local (cyberpunk t/s + render).
# Uses run_gen.sh. Runs are back-to-back so model pages stay warm from the sweep.
set -u
RUN=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot
G="$RUN/run_gen.sh"
CYBER="$RUN/prompt_cyberpunk_wide.txt"
COFFEE="$RUN/prompt_coffee.txt"

# --- masked route traces (short, weighted, matches existing K12-38 trace format) ---
bash "$G" trace_K48 48 300 4096 "$CYBER" "$RUN/traces" 1
bash "$G" trace_K64 64 300 4096 "$CYBER" "$RUN/traces" 1
bash "$G" trace_K91 91 300 4096 "$CYBER" "$RUN/traces" 1

# --- narrow coffee contrast ---
bash "$G" coffee_K64 64 1500 4096 "$COFFEE" "$RUN/coffee" 0
bash "$G" coffee_K0   0 1500 4096 "$COFFEE" "$RUN/coffee" 0

# --- K0 FULL local (the 81GB>60GB no-fit regime): cyberpunk t/s + render ---
bash "$G" k0_cyberpunk 0 1500 6144 "$CYBER" "$RUN/k0" 0

echo "[phase2] PHASE2_COMPLETE $(date -Is)" >> "$RUN/phase2.log"
