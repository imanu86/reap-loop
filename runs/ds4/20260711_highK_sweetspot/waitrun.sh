#!/bin/bash
# sleep $1 seconds, then print phase-2 status. Used for spaced background polling.
sleep "${1:-180}"
bash /mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_highK_sweetspot/p2status.sh 2>&1 | grep -vE "^[[:space:]]*$"
