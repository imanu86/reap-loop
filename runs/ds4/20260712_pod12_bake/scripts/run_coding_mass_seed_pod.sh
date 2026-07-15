#!/usr/bin/env bash
# Build coding-wide fixed-K masks from uncensored K0 routing on a Linux pod.
set -euo pipefail

REPO=${REPO:-/root/reap-loop}
BIN=${BIN:-/root/ds4-fullstack/ds4-server}
MODEL=${MODEL:-/root/models/ds4-2bit.gguf}
OUT=${OUT:-$REPO/runs/ds4/20260712_pod12_bake/coding_mass_prefill_seed_20260715}
PORT=${PORT:-18082}
PROMPT_SPLIT=${PROMPT_SPLIT:-learn}
PROMPT_IDS=${PROMPT_IDS:-}
MAX_TOKENS=${MAX_TOKENS:-1}
CONTEXT=${CONTEXT:-2048}
BUILD_MASKS=${BUILD_MASKS:-1}
REAP_COMMIT=${REAP_COMMIT:-}
MASK_BUILDER=$REPO/runs/ds4/20260712_virtual_bake/scripts/build_mass_mask.py
ROUTE_CSV="$OUT/${PROMPT_SPLIT}_route.csv"

if [[ "$PROMPT_SPLIT" != "learn" && "$PROMPT_SPLIT" != "eval" ]]; then
    echo "PROMPT_SPLIT must be learn or eval" >&2
    exit 2
fi
mkdir -p "$OUT/$PROMPT_SPLIT" "$OUT/masks"
test -x "$BIN"
test -s "$MODEL"
if [[ "$BUILD_MASKS" == 1 ]]; then test -f "$MASK_BUILDER"; fi

python3 - "$OUT/prompts.json" <<'PY'
import json
import sys

prompts = {
    "learn": [
        {"id": "c_ring", "prompt": "Implementa in C11 un ring buffer bounded thread-safe con API init, push, pop e destroy. Usa pthread mutex e condition variable, gestisci shutdown e commenta solo le invarianti non ovvie. Restituisci un file C completo."},
        {"id": "cpp_pool", "prompt": "Scrivi in C++20 un piccolo thread pool RAII con coda bounded, submit che restituisce std::future, stop ordinato e gestione eccezioni. Restituisci header e demo compilabile."},
        {"id": "python_async", "prompt": "Scrivi un modulo Python 3.12 che scarica URL concorrenti con asyncio, limite di concorrenza, timeout, retry esponenziale e risultati tipizzati. Includi test essenziali senza dipendenze esterne."},
        {"id": "javascript_ui", "prompt": "Crea un modulo JavaScript ES2022 per un form dinamico: validazione accessibile, submit fetch abortibile, retry, stato loading e rendering sicuro degli errori. Fornisci anche il markup minimo necessario."},
        {"id": "html_cyberpunk", "prompt": "Crea una landing page HTML CSS JS single-file per un negozio di programmazione AI in stile cyberpunk. Deve avere nav, hero, servizi, modulo contatti e popup JavaScript di conferma. Restituisci solo HTML completo e valido."},
        {"id": "sql_events", "prompt": "Progetta in PostgreSQL uno schema per eventi append-only multi-tenant. Includi DDL, indici, query keyset pagination, deduplicazione idempotente e una strategia di retention partizionata."}
    ],
    "eval": [
        {"id": "powershell_logs", "prompt": "Scrivi uno script PowerShell robusto che ruota log per dimensione e data, comprime gli archivi, conserva 14 giorni e supporta -WhatIf."},
        {"id": "rust_channel", "prompt": "Implementa in Rust stabile un canale bounded MPMC con chiusura pulita e test concorrenti, senza crate esterni."},
        {"id": "go_workers", "prompt": "Scrivi in Go un servizio HTTP con worker pool bounded, context cancellation, graceful shutdown, metriche base e test httptest."},
        {"id": "container_stack", "prompt": "Fornisci Dockerfile multi-stage e compose per una API Python con PostgreSQL, healthcheck, utente non-root, migrazioni e configurazione tramite secret."}
    ]
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)
PY

if pgrep -x ds4-reaploop-sm86 >/dev/null || pgrep -x ds4-server >/dev/null; then
    echo "another DS4 process is running on the pod" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

export DS4_SPEX_TRACE_ROUTING="$ROUTE_CSV"
export DS4_SPEX_TRACE_ROUTING_WEIGHTS=1
export DS4_SPEX_TRACE_PREFILL_ROUTING=1
export DS4_CUDA_NO_DIRECT_IO=1
export DS4_CUDA_KEEP_MODEL_PAGES=1
export DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256
export DS4_PACE=0
unset DS4_REAP_MASK_FILE

env | LC_ALL=C sort > "$OUT/server_env.txt"
{
    printf 'binary_sha256='; sha256sum "$BIN" | awk '{print $1}'
    printf 'model_size='; stat -c %s "$MODEL"
    printf 'prompt_split=%s\n' "$PROMPT_SPLIT"
    printf 'prompt_ids=%s\n' "$PROMPT_IDS"
    printf 'max_tokens=%s\n' "$MAX_TOKENS"
    printf 'context=%s\n' "$CONTEXT"
    printf 'build_masks=%s\n' "$BUILD_MASKS"
    printf 'started_utc='; date -u +%FT%TZ
} > "$OUT/manifest.txt"

if [[ -n "$REAP_COMMIT" ]]; then
    printf 'reap_commit=%s\n' "$REAP_COMMIT" >> "$OUT/manifest.txt"
elif git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    printf 'reap_commit=' >> "$OUT/manifest.txt"
    git -C "$REPO" rev-parse HEAD >> "$OUT/manifest.txt"
else
    printf 'reap_commit=unknown\n' >> "$OUT/manifest.txt"
fi

"$BIN" -m "$MODEL" --cuda --ssd-streaming \
    --ssd-streaming-cache-experts 1024 --prefill-chunk 512 \
    -c "$CONTEXT" -n "$((MAX_TOKENS + 128))" --host 127.0.0.1 --port "$PORT" --cors \
    >"$OUT/server.stdout.log" 2>"$OUT/server.stderr.log" &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"

ready=0
for _ in $(seq 1 240); do
    if curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
done
if [[ "$ready" != 1 ]]; then
    echo "server did not become ready" >&2
    exit 1
fi

python3 - "$OUT/prompts.json" "$OUT/$PROMPT_SPLIT" "$PORT" "$PROMPT_SPLIT" "$PROMPT_IDS" "$MAX_TOKENS" <<'PY'
import json
import pathlib
import sys
import time
import urllib.request

prompts_path, out_dir, port, split, prompt_ids, max_tokens = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
items = json.load(open(prompts_path, encoding="utf-8"))[split]
if prompt_ids:
    selected = set(prompt_ids.split(","))
    items = [item for item in items if item["id"] in selected]
    missing = selected - {item["id"] for item in items}
    if missing:
        raise SystemExit(f"unknown prompt ids: {sorted(missing)}")
url = f"http://127.0.0.1:{port}/v1/chat/completions"
for item in items:
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Rispondi direttamente, senza ragionamento visibile."},
            {"role": "user", "content": item["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "think": False,
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = response.read()
    elapsed = time.monotonic() - t0
    (out_dir / f"{item['id']}.json").write_bytes(body)
    (out_dir / f"{item['id']}.elapsed_s").write_text(f"{elapsed:.6f}\n")
    print(f"{split} {item['id']}: {elapsed:.3f}s", flush=True)
PY

cleanup
SERVER_PID=""
trap - EXIT

test -s "$ROUTE_CSV"
if [[ "$PROMPT_SPLIT" == "learn" && "$BUILD_MASKS" == 1 ]]; then
    for spec in 154:k60 160:k62_5 166:k65; do
        keep=${spec%%:*}
        name=${spec##*:}
        python3 "$MASK_BUILDER" --keep "$keep" \
            --out "$OUT/masks/${name}_coding.txt" "$ROUTE_CSV" \
            >"$OUT/masks/${name}.build.stdout" 2>"$OUT/masks/${name}.build.stderr"
    done
fi

python3 - "$OUT" <<'PY'
import csv
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
rows = 0
positions = set()
layers = set()
route_csv = next(out.glob("*_route.csv"))
with route_csv.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        rows += 1
        positions.add(int(row["pos"]))
        layers.add(int(row["layer"]))
summary = {
    "route_rows": rows,
    "positions": len(positions),
    "layers": sorted(layers),
    "masks": {
        p.name: sum(1 for _ in p.open(encoding="utf-8"))
        for p in sorted((out / "masks").glob("*_coding.txt"))
    },
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

date -u +%FT%TZ > "$OUT/${PROMPT_SPLIT^^}_DONE"
