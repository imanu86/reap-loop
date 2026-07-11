#!/bin/bash
# Kill the current -n 5500 quality ds4 run (matches "-c 8192 --nothink", unique to qual runs;
# probes use -c 4096). Runs from a file so the invocation's own cmdline never self-matches.
pkill -f -- "-c 8192 --nothink"
sleep 2
pgrep -af "ds4 -m" | grep -v pgrep || echo "no ds4 running now"
