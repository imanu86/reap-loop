#!/usr/bin/env bash
# Stream a DS4 sparse-bake pack directly to object storage with fail-closed receipts.
set -euo pipefail

: "${MODEL:?set MODEL}"
: "${MASK:?set MASK}"
: "${DEST:?set DEST, for example r2:bucket/path/model.ds4pack}"
: "${SOURCE_SHA256:?set SOURCE_SHA256}"
: "${OUT:?set OUT}"

PYTHON=${PYTHON:-python3}
PACKER=${PACKER:-/root/ds4_windows_sparse_bake.py}
RCLONE=${RCLONE:-rclone}
RCLONE_CONFIG=${RCLONE_CONFIG:-/root/r2-codex.conf}
CHUNK_SIZE=${CHUNK_SIZE:-256M}

test -s "$MODEL"
test -s "$MASK"
test -s "$PACKER"
test -s "$RCLONE_CONFIG"
mkdir -p "$OUT"

STATUS="$OUT/pack_status.json"
LOG="$OUT/pack_stream.log"
MANIFEST="$OUT/run_manifest.txt"
DONE="$OUT/DONE"
REMOTE_DIR=${DEST%/*}
REMOTE_NAME=${DEST##*/}

rm -f "$STATUS" "$DONE"
if "$RCLONE" --config "$RCLONE_CONFIG" lsf "$REMOTE_DIR" --files-only \
        2>/dev/null | grep -Fqx "$REMOTE_NAME"; then
    echo "destination already exists: $DEST" >&2
    exit 8
fi

{
    date -u +%FT%TZ
    printf 'model=%s\nmask=%s\ndest=%s\nsource_sha256=%s\n' \
        "$MODEL" "$MASK" "$DEST" "$SOURCE_SHA256"
    stat -c 'model_size=%s model_mtime=%y' "$MODEL"
    sha256sum "$MASK" "$PACKER" "$0"
    "$PYTHON" --version
    "$RCLONE" version | head -n 2
} > "$MANIFEST"

set +e
"$PYTHON" "$PACKER" pack-stream \
    --model "$MODEL" \
    --mask "$MASK" \
    --source-sha256 "$SOURCE_SHA256" \
    --status-out "$STATUS" \
    2>>"$LOG" |
    "$RCLONE" --config "$RCLONE_CONFIG" rcat "$DEST" \
        --s3-chunk-size "$CHUNK_SIZE" \
        --stats 30s --stats-one-line --log-level INFO \
        2>>"$LOG"
pipe_status=("${PIPESTATUS[@]}")
set -e

printf 'packer_exit=%s\nrclone_exit=%s\n' \
    "${pipe_status[0]}" "${pipe_status[1]}" >> "$MANIFEST"
if [[ "${pipe_status[0]}" != 0 || "${pipe_status[1]}" != 0 ]]; then
    echo "pack/upload pipeline failed; see $LOG" >&2
    exit 9
fi
test -s "$STATUS"

expected_bytes=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["total_bytes"])' \
    "$STATUS")
remote_bytes=$("$RCLONE" --config "$RCLONE_CONFIG" size --json "$DEST" |
    "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["bytes"])')
printf 'expected_bytes=%s\nremote_bytes=%s\n' \
    "$expected_bytes" "$remote_bytes" >> "$MANIFEST"
if [[ "$expected_bytes" != "$remote_bytes" ]]; then
    echo "remote size mismatch: expected=$expected_bytes remote=$remote_bytes" >&2
    exit 10
fi

"$RCLONE" --config "$RCLONE_CONFIG" lsjson "$DEST" > "$OUT/remote_object.json"
date -u +%FT%TZ > "$DONE"
echo "complete: $DEST ($remote_bytes bytes)"
