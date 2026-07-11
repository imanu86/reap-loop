#!/bin/bash
# Kill the k0_cyberpunk full run (matches "-c 6144", unique to it). File-based to avoid self-match.
pkill -f -- "-c 6144"
sleep 2
pgrep -af "ds4 -m" | grep -v pgrep || echo "no ds4 running now"
