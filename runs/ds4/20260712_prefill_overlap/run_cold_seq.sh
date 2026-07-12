#!/bin/bash
# Cold-cache A/B: drop_caches before each arm, short decode (MAXTOK=20).
set -u
OUTDIR=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_prefill_overlap
export MAXTOK=20
export COLD=1
bash "$OUTDIR/run_test.sh" cOFF
bash "$OUTDIR/run_test.sh" cS1 DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1
bash "$OUTDIR/run_test.sh" cS1S2 DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1 DS4_REAP_PREFILL_READAHEAD=1
bash "$OUTDIR/run_test.sh" cOFF2
echo "COLD_SEQUENCE_DONE"
