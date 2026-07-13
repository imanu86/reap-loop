#!/usr/bin/env bash
# Phase-1 transport safety/fault matrix. This harness is intentionally not
# run during patch authoring because it requires a Linux CUDA/model runtime.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "usage: $0 BIN MODEL COMPUTE_MASK PROMPT_FILE OUTROOT [NTOK=64]" >&2
    exit 2
fi

BIN=$1
MODEL=$2
COMPUTE_MASK=$3
PROMPT_FILE=$4
OUTROOT=$5
NTOK=${6:-64}
CTX=${CTX:-2048}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-2400}

for required in "$BIN" "$MODEL" "$COMPUTE_MASK" "$PROMPT_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 3
    fi
done
if [[ ! "$NTOK" =~ ^[1-9][0-9]*$ ]]; then
    echo "NTOK must be a positive integer" >&2
    exit 3
fi

mkdir -p "$OUTROOT"

depths=(1 2 8)
faults=(
    baseline
    stage_event_create
    stage_event_record
    upload_event_create
    upload_event_record
    upload_event_wait
)

declare -A fault_env=(
    [stage_event_create]=DS4_CUDA_FAULT_STAGE_EVENT_CREATE
    [stage_event_record]=DS4_CUDA_FAULT_STAGE_EVENT_RECORD
    [upload_event_create]=DS4_CUDA_FAULT_UPLOAD_EVENT_CREATE
    [upload_event_record]=DS4_CUDA_FAULT_UPLOAD_EVENT_RECORD
    [upload_event_wait]=DS4_CUDA_FAULT_UPLOAD_EVENT_WAIT
)
declare -A fault_counter=(
    [stage_event_create]=stage_event_create_failures
    [stage_event_record]=stage_event_record_failures
    [upload_event_create]=upload_event_create_failures
    [upload_event_record]=upload_event_record_failures
    [upload_event_wait]=upload_event_wait_failures
)

counter_names=(
    stage_event_create_failures
    stage_event_record_failures
    upload_event_create_failures
    upload_event_record_failures
    upload_event_wait_failures
)

diag_field() {
    local line=$1
    local field=$2
    sed -n "s/.*[[:space:]]${field}=\([0-9][0-9]*\).*/\1/p" <<<"$line"
}

require_eq() {
    local line=$1
    local field=$2
    local expected=$3
    local got
    got=$(diag_field "$line" "$field")
    if [[ -z "$got" || "$got" != "$expected" ]]; then
        echo "expected ${field}=${expected}, got ${got:-missing}" >&2
        return 1
    fi
}

require_gt_zero() {
    local line=$1
    local field=$2
    local got
    got=$(diag_field "$line" "$field")
    if [[ -z "$got" || "$got" -le 0 ]]; then
        echo "expected ${field}>0, got ${got:-missing}" >&2
        return 1
    fi
}

printf 'depth\tfault\tstdout_sha256\tdiag\n' > "$OUTROOT/summary.tsv"
reference_hash=

for depth in "${depths[@]}"; do
    for fault in "${faults[@]}"; do
        arm="depth${depth}_${fault}"
        run="$OUTROOT/$arm"
        mkdir -p "$run"

        envset=(
            "CUDA_VISIBLE_DEVICES=0"
            "DS4_ASYNC_PIPELINE=1"
            "DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1"
            "DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1"
            "DS4_CUDA_SELECTED_STAGE_DEPTH=$depth"
            "DS4_CUDA_TRANSPORT_DIAG=1"
            "DS4_CUDA_TRANSPORT_DEBUG=1"
            "DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1"
            "DS4_CUDA_NO_DIRECT_IO=1"
            "DS4_CUDA_KEEP_MODEL_PAGES=1"
            "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1"
            "DS4_CUDA_NO_Q8_F16_CACHE=1"
            "DS4_CUDA_WEIGHT_CACHE_VERBOSE=1"
            "DS4_REAP_MASK_FILE=$COMPUTE_MASK"
            "DS4_PACE=0"
            "HOME=/root"
            "LANG=C.UTF-8"
            "LC_ALL=C.UTF-8"
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            "LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64"
            "TMPDIR=/tmp"
        )
        if [[ "$fault" != baseline ]]; then
            envset+=("${fault_env[$fault]}=1")
        fi

        printf '%s\n' "${envset[@]}" | LC_ALL=C sort > "$run/env.txt"
        printf '%q ' "$BIN" -m "$MODEL" --cuda --ssd-streaming \
            --ssd-streaming-cache-experts 6 --prefill-chunk 512 \
            --prompt-file "$PROMPT_FILE" -n "$NTOK" -c "$CTX" \
            --temp 0 --nothink > "$run/argv.txt"
        printf '\n' >> "$run/argv.txt"

        if ! env -i "${envset[@]}" timeout "$TIMEOUT_SECONDS" \
            "$BIN" -m "$MODEL" --cuda --ssd-streaming \
            --ssd-streaming-cache-experts 6 --prefill-chunk 512 \
            --prompt-file "$PROMPT_FILE" -n "$NTOK" -c "$CTX" \
            --temp 0 --nothink \
            > "$run/stdout.txt" 2> "$run/stderr.txt"; then
            echo "arm failed: $arm" >&2
            exit 10
        fi

        grep -F "CUDA streaming selected stage ring depth=$depth" \
            "$run/stderr.txt" >/dev/null
        diag=$(grep -F "CUDA transport safety final" "$run/stderr.txt" | tail -n 1)
        if [[ -z "$diag" ]]; then
            echo "missing final transport diagnostic: $arm" >&2
            exit 11
        fi

        require_gt_zero "$diag" stage_chunks
        require_eq "$diag" war_wait_failures 0
        require_eq "$diag" fallback_sync_failures 0
        require_eq "$diag" staged_h2d_failures 0
        require_eq "$diag" stale_slot_reads 0

        if [[ "$fault" == baseline ]]; then
            require_eq "$diag" sync_stage_mode 0
            require_gt_zero "$diag" cursor_advances
            require_gt_zero "$diag" slot_reuses
            require_gt_zero "$diag" war_waits
            for counter in "${counter_names[@]}"; do
                require_eq "$diag" "$counter" 0
            done
        else
            grep -F "CUDA transport fault injected: $fault" \
                "$run/stderr.txt" >/dev/null
            grep -F "CUDA transport safe synchronous staging mode latched" \
                "$run/stderr.txt" >/dev/null
            require_eq "$diag" sync_stage_mode 1
            require_gt_zero "$diag" fallback_syncs
            for counter in "${counter_names[@]}"; do
                expected=0
                if [[ "$counter" == "${fault_counter[$fault]}" ]]; then
                    expected=1
                fi
                require_eq "$diag" "$counter" "$expected"
            done
        fi

        output_hash=$(sha256sum "$run/stdout.txt" | awk '{print $1}')
        if [[ -z "$reference_hash" ]]; then
            reference_hash=$output_hash
        elif [[ "$output_hash" != "$reference_hash" ]]; then
            echo "output mismatch: $arm has $output_hash, expected $reference_hash" >&2
            exit 12
        fi
        printf '%s\t%s\t%s\t%s\n' \
            "$depth" "$fault" "$output_hash" "$diag" >> "$OUTROOT/summary.tsv"
    done
done

echo "PASS: depth 1/2/8, cache 6, all one-shot event faults, exact stdout"
