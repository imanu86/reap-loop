#!/bin/bash
# 0033 core smoke: cyberpunk (deterministic vehicle), frozen K12, mandate-default tier params.
# run_tier.sh <label> <tier> [cache] [keep] [prompt] [twarm]
R=/root/smoke_0033/run_tier.sh
CY=/root/stage0033/cyberpunk_prompt.txt
echo "===== cache 512 (convergence primary) ====="
bash $R t0_c512  0 512 12 $CY 512   # baseline / determinism control A
bash $R t0b_c512 0 512 12 $CY 512   # determinism control B (== A ?)
bash $R t1_c512  1 512 12 $CY 512   # TIER on -> bit-exact vs A + convergence
echo "===== cache 256 (engagement, 3060-like heavy pressure) ====="
bash $R t0_c256  0 256 12 $CY 512   # LRU baseline
bash $R t1_c256  1 256 12 $CY 512   # TIER on -> bit-exact + engagement win
echo ALL_DONE_$(date -u +%H:%M:%S)
