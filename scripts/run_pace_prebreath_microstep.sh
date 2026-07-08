#!/usr/bin/env bash
# PACE prebreath micro-step harness for ds4/RunPod.
# Runs one warm discard, then the requested prebreath step matrix on two prompts.
set -euo pipefail

BIN="${BIN:-/root/ds4/ds4}"
MODEL="${MODEL:-/root/models/ds4-2bit.gguf}"
OUT="${OUT:-/workspace/pace_prebreath_microstep_$(date -u +%Y%m%d_%H%M%S)}"
N="${N:-160}"
CTX="${CTX:-2048}"
CACHE_EXPERTS="${CACHE_EXPERTS:-2048}"
PREFILL_CHUNK="${PREFILL_CHUNK:-512}"
RUNS="${RUNS:-1}"
INCLUDE_EVERY32="${INCLUDE_EVERY32:-0}"

mkdir -p "$OUT/prompts" "$OUT/logs" "$OUT/outputs" "$OUT/events" "$OUT/metrics"

cat > "$OUT/prompts/short_html.txt" <<'PROMPT'
Create a single-file HTML landing section for a tiny espresso timer app. Include accessible markup, CSS, and a small inline JavaScript timer. Keep it compact and valid.
PROMPT

cat > "$OUT/prompts/medium_coding.txt" <<'PROMPT'
Review this C-style pseudocode for a GPU-backed MoE inference loop. Explain the likely correctness risks, then propose a minimal patch plan that preserves routing semantics while reducing repeated expert-load stalls.

for token in decode:
    selected = router(hidden)
    for expert in selected:
        if not cache.contains(expert):
            load_from_ssd(expert)
        hidden += expert(hidden)
    if repetition_score(window) > threshold:
        widen_mask()
    else if stable_hit_rate > 0.9:
        tighten_mask()

Focus on invariants, cache behavior, and failure modes. Do not invent benchmark numbers.
PROMPT

COMMON=(
  -m "$MODEL"
  --cuda
  --ssd-streaming
  --ssd-streaming-cache-experts "$CACHE_EXPERTS"
  --prefill-chunk "$PREFILL_CHUNK"
  -c "$CTX"
  --nothink
  --temp 0
)

BASE_ENV=(
  DS4_CUDA_NO_DIRECT_IO=1
  DS4_CUDA_KEEP_MODEL_PAGES=1
  DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1
  DS4_SPEX_STATS=1
  DS4_MTP_SPEC_DISABLE=1
  DS4_PACE=1
  DS4_PACE_WARMUP=50
  DS4_PACE_KEEP=64
  DS4_PACE_KEEP_MIN=64
  DS4_PACE_KEEP_MAX=96
  DS4_PACE_PREFILL_APPLY=0
  DS4_PACE_BREATH_KEEP=96
  DS4_PACE_DRIFT=0.25
  DS4_PACE_BREATH_EVERY=400
  DS4_PACE_WRAP=0
  DS4_PACE_CACHE_FLUSH=0
  DS4_PACE_RELEARN=0
  DS4_PACE_DEBUG=1
)

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT/progress.log"
}

count_tokens() {
  local name="$1" prompt="$2" dump="$OUT/metrics/tokens_${name}.txt"
  "$BIN" "${COMMON[@]}" --dump-tokens --prompt-file "$prompt" > "$dump" 2> "$OUT/logs/tokens_${name}.log"
  python3 - "$dump" <<'PY'
import ast, sys
first = open(sys.argv[1], encoding="utf-8", errors="replace").readline()
print(len(ast.literal_eval(first)))
PY
}

run_one() {
  local prompt_name="$1" prompt_file="$2" variant="$3" rep="$4"; shift 4
  local stem="${prompt_name}_${variant}_r${rep}"
  local stdout="$OUT/outputs/${stem}.txt"
  local stderr="$OUT/logs/${stem}.log"
  local events="$OUT/events/${stem}.jsonl"
  local metric="$OUT/metrics/${stem}.json"
  log "run $stem"
  env "${BASE_ENV[@]}" "$@" DS4_PACE_LOG="$events" \
    python3 - "$BIN" "$MODEL" "$prompt_file" "$stdout" "$stderr" "$metric" "$N" "$CTX" "$CACHE_EXPERTS" "$PREFILL_CHUNK" <<'PY'
import json, os, selectors, subprocess, sys, time

bin_path, model, prompt, stdout_path, stderr_path, metric_path = sys.argv[1:7]
n, ctx, cache, prefill_chunk = sys.argv[7:11]
cmd = [
    bin_path, "-m", model, "--cuda", "--ssd-streaming",
    "--ssd-streaming-cache-experts", cache,
    "--prefill-chunk", prefill_chunk,
    "-c", ctx, "--nothink", "--temp", "0", "-n", n,
    "--prompt-file", prompt,
]
start = time.monotonic()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sel = selectors.DefaultSelector()
sel.register(p.stdout, selectors.EVENT_READ, "stdout")
sel.register(p.stderr, selectors.EVENT_READ, "stderr")
first_stdout = None
chunks = []
with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
    while sel.get_map():
        for key, _ in sel.select(timeout=0.2):
            data = key.fileobj.read1(4096)
            if not data:
                sel.unregister(key.fileobj)
                continue
            now = time.monotonic()
            if key.data == "stdout":
                if first_stdout is None:
                    first_stdout = now
                chunks.append({"t": now - start, "bytes": len(data)})
                out.write(data)
                out.flush()
            else:
                err.write(data)
                err.flush()
rc = p.wait()
end = time.monotonic()
with open(metric_path, "w", encoding="utf-8") as f:
    json.dump({
        "rc": rc,
        "wall_s": end - start,
        "ttft_s": None if first_stdout is None else first_stdout - start,
        "stdout_chunks": chunks,
    }, f, indent=2)
sys.exit(rc)
PY
}

summarize() {
  python3 - "$OUT" <<'PY'
import csv, glob, json, os, re, sys
out = sys.argv[1]
token_counts = {}
for path in glob.glob(os.path.join(out, "metrics", "tokens_*.txt")):
    name = os.path.basename(path)[7:-4]
    first = open(path, encoding="utf-8", errors="replace").readline()
    token_counts[name] = first.count(",") + (1 if first.strip().startswith("[") and first.strip() != "[]" else 0)

rows = []
for metric_path in sorted(glob.glob(os.path.join(out, "metrics", "*_r*.json"))):
    stem = os.path.basename(metric_path)[:-5]
    parts = stem.split("_")
    prompt = "_".join(parts[:2])
    variant = "_".join(parts[2:-1])
    rep = parts[-1]
    m = json.load(open(metric_path, encoding="utf-8"))
    log_path = os.path.join(out, "logs", stem + ".log")
    text_path = os.path.join(out, "outputs", stem + ".txt")
    ev_path = os.path.join(out, "events", stem + ".jsonl")
    log = open(log_path, encoding="utf-8", errors="replace").read() if os.path.exists(log_path) else ""
    gen = open(text_path, encoding="utf-8", errors="replace").read() if os.path.exists(text_path) else ""
    prefill_tps = gen_tps = None
    mt = re.search(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s", log)
    if mt:
        prefill_tps = float(mt.group(1))
        gen_tps = float(mt.group(2))
    events = []
    if os.path.exists(ev_path):
        for line in open(ev_path, encoding="utf-8", errors="replace"):
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    prompt_tokens = token_counts.get(prompt)
    prefill_s = (prompt_tokens / prefill_tps) if prompt_tokens and prefill_tps else None
    chunks = m.get("stdout_chunks") or []
    rows.append({
        "prompt": prompt,
        "variant": variant,
        "rep": rep.lstrip("r"),
        "rc": m.get("rc"),
        "prompt_tokens": prompt_tokens,
        "prefill_s": None if prefill_s is None else f"{prefill_s:.3f}",
        "prefill_tps": prefill_tps,
        "ttft_process_s": None if m.get("ttft_s") is None else f"{m['ttft_s']:.3f}",
        "wall_s": f"{m.get('wall_s', 0):.3f}",
        "gen_tps_avg": gen_tps,
        "chunks": len(chunks),
        "prebreath_events": sum(1 for e in events if e.get("ev") == "prebreath"),
        "breath_events": sum(1 for e in events if str(e.get("ev", "")).startswith("breath")),
        "tighten_events": sum(1 for e in events if e.get("ev") == "tighten"),
        "s_init_count": gen.count("S_INIT"),
        "html_tag_balance": (gen.lower().count("<html") - gen.lower().count("</html>")) if prompt == "short_html" else "",
        "loop_hint": int(bool(re.search(r"(.{20,120})\1\1", gen, re.S))),
    })

fields = [
    "prompt", "variant", "rep", "rc", "prompt_tokens", "prefill_s",
    "prefill_tps", "ttft_process_s", "wall_s", "gen_tps_avg", "chunks",
    "prebreath_events", "breath_events", "tighten_events", "s_init_count",
    "html_tag_balance", "loop_hint",
]
with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(os.path.join(out, "summary.csv"))
PY
}

log "OUT=$OUT"
log "BIN=$BIN MODEL=$MODEL N=$N CTX=$CTX CACHE_EXPERTS=$CACHE_EXPERTS"

for prompt_name in short_html medium_coding; do
  count=$(count_tokens "$prompt_name" "$OUT/prompts/${prompt_name}.txt")
  log "prompt $prompt_name tokens=$count"
done

log "warm discard"
env "${BASE_ENV[@]}" DS4_PACE_PREBREATH=0 "$BIN" "${COMMON[@]}" -n 32 \
  --prompt-file "$OUT/prompts/short_html.txt" \
  > "$OUT/outputs/warm_discard.txt" 2> "$OUT/logs/warm_discard.log" || true

for prompt_name in short_html medium_coding; do
  prompt_file="$OUT/prompts/${prompt_name}.txt"
  for rep in $(seq 1 "$RUNS"); do
    run_one "$prompt_name" "$prompt_file" prebreath_off "$rep" \
      DS4_PACE_PREBREATH=0 DS4_PACE_KEEP_STEP=4 DS4_PACE_PREBREATH_EVERY=64
    run_one "$prompt_name" "$prompt_file" step4_every64 "$rep" \
      DS4_PACE_PREBREATH=1 DS4_PACE_KEEP_STEP=4 DS4_PACE_PREBREATH_EVERY=64
    run_one "$prompt_name" "$prompt_file" step2_every64 "$rep" \
      DS4_PACE_PREBREATH=1 DS4_PACE_KEEP_STEP=2 DS4_PACE_PREBREATH_EVERY=64
    run_one "$prompt_name" "$prompt_file" step1_every64 "$rep" \
      DS4_PACE_PREBREATH=1 DS4_PACE_KEEP_STEP=1 DS4_PACE_PREBREATH_EVERY=64
    if [ "$INCLUDE_EVERY32" = "1" ]; then
      run_one "$prompt_name" "$prompt_file" step1_every32 "$rep" \
        DS4_PACE_PREBREATH=1 DS4_PACE_KEEP_STEP=1 DS4_PACE_PREBREATH_EVERY=32
    fi
  done
done

summarize | tee "$OUT/SUMMARY_PATH.txt"
log "done"
