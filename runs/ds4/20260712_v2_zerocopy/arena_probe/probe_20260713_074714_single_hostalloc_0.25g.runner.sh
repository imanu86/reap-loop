#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_86 \
  -o /tmp/cuda_pinned_arena_probe '/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_v2_zerocopy/tools/cuda_pinned_arena_probe.cu'
set +e
/tmp/cuda_pinned_arena_probe \
  --mode 'single' --api 'hostalloc' --target-gib '0.25' --step-gib '0.25' \
  --reserve-gib '8' &
probe_pid=$!
echo $probe_pid > '/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_v2_zerocopy/arena_probe/probe_20260713_074714_single_hostalloc_0.25g.pid'
wait $probe_pid
probe_rc=$?
echo $probe_rc > '/mnt/c/Users/imanu/source/repos/reap-loop/runs/ds4/20260712_v2_zerocopy/arena_probe/probe_20260713_074714_single_hostalloc_0.25g.rc'
exit $probe_rc