#!/bin/bash
echo "BUFFCACHE_MB=$(free -m | sed -n 's/^Mem: *[0-9]* *[0-9]* *[0-9]* *[0-9]* *\([0-9]*\).*/\1/p')"
echo "--- jobs ---"
pgrep -af profile_breakdown || echo "no driver"
pgrep -af nsys || echo "no nsys"
pgrep -a ds4 || echo "no ds4"
echo "--- progress tail ---"
tail -6 /mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_token_breakdown/progress.log
echo "--- sqlite files ---"
ls -la /mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_token_breakdown/NSYS_n40/trace.sqlite /mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260711_token_breakdown/NSYS_n120/trace.sqlite 2>/dev/null || echo "no sqlite yet"
