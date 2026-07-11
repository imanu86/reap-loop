#!/bin/bash
S=/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_local_clean_lowK
rm -f "$S/curve/curve_trace_progress.log"
nohup bash "$S/curve_trace.sh" > "$S/curve/curve_trace.stdout.log" 2>&1 &
echo "launched pid=$!"
