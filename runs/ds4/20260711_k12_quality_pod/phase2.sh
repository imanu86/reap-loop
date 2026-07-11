#!/bin/bash
# PHASE 2 (bisection, run only if K12 collapses but K48 renders): K16, K23 domain-matched.
# Find the minimum K that renders. Same config: cache32, -n5500, ctx8192, greedy temp0.
set -u
RUN=/root/k12q
R=$RUN/run_one.sh
P=$RUN/prompt_cyberpunk_wide.txt
O=$RUN/out
M=$RUN/masks
echo "==== PHASE2 START $(date -Is) ====" >> "$O/progress.log"
bash "$R" K16_r1 "$M/sessCyber_K16.txt" 5500 8192 "$P" "$O" 1
bash "$R" K23_r1 "$M/sessCyber_K23.txt" 5500 8192 "$P" "$O" 1
echo "PHASE2_DONE $(date -Is)" >> "$O/progress.log"
echo DONE > "$O/phase2.done"
