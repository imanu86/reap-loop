#!/usr/bin/env bash
# Wait for an R2 model transfer, verify it, then run selected quality arms.
set -euo pipefail

MODEL=${MODEL:-/root/models/ds4-2bit.gguf}
MODEL_EXPECTED_BYTES=${MODEL_EXPECTED_BYTES:-86720111488}
MODEL_SHA_FILE=${MODEL_SHA_FILE:-${MODEL}.sha256}
RUNNER=${RUNNER:-/root/reap-loop/runs/ds4/20260712_pod12_bake/scripts/run_windows_bake_quality_ab_pod.sh}

while pgrep -x rclone >/dev/null; do
    sleep 5
done

actual_bytes=$(stat -c %s "$MODEL")
if [[ "$actual_bytes" != "$MODEL_EXPECTED_BYTES" ]]; then
    echo "model size mismatch: expected=$MODEL_EXPECTED_BYTES actual=$actual_bytes" >&2
    exit 3
fi

(cd "$(dirname "$MODEL")" && sha256sum -c "$(basename "$MODEL_SHA_FILE")")
exec bash "$RUNNER"
