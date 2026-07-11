#!/bin/bash
# PHASE 1: K48 positive-control (cross-hardware anchor) + K12 x2 (decisive, n=2).
# Warmth order: largest keep (K48) first -> K12 uses subset -> warm. cache32, -n5500, ctx8192.
set -u
RUN=/root/k12q
R=$RUN/run_one.sh
P=$RUN/prompt_cyberpunk_wide.txt
O=$RUN/out
M=$RUN/masks
echo "==== PHASE1 START $(date -Is) ====" >> "$O/progress.log"
bash "$R" K48_ctrl "$M/sessCyber_K48.txt" 5500 8192 "$P" "$O" 1
bash "$R" K12_r1   "$M/sessCyber_K12.txt" 5500 8192 "$P" "$O" 1
bash "$R" K12_r2   "$M/sessCyber_K12.txt" 5500 8192 "$P" "$O" 0
echo "PHASE1_DONE $(date -Is)" >> "$O/progress.log"
echo DONE > "$O/phase1.done"
