#!/usr/bin/env python3
"""Static-mask warm timing driver for the keep-8 fit reproduction (3060, local).

Runs ENTIRELY inside WSL. Launches one ds4-server with a static REAP bias-mask,
does a warmup request (discard) then a streamed measured request while recording
per-token receive timestamps, and computes segmented t/s (TTFT / 1-64 / 65-256 /
257+). Optionally enables a routing trace for enforcement (distinct experts/layer).

Coexists with the UI: distinct DS4_LOCK_FILE + port 8014; never pkills anything
other than a ds4-server bound to this exact port.
"""
import argparse, json, os, subprocess, sys, time, urllib.request, urllib.error, signal, gzip, collections, re

REPO = "/mnt/c/Users/imanu/source/repos/reap-loop"
MODEL = "/root/models/ds4-2bit.gguf"

COFFEE_PROMPT = (
    "Write a COMPLETE and COMPACT single-file HTML page for a coffee shop. "
    "Output ONLY the HTML, nothing else. Keep the CSS SHORT (about 10-15 "
    "rules max) — prioritize a COMPLETE, working page over elaborate "
    "styling. The page MUST be fully closed with </html> and MUST contain "
    "all of these:\n"
    "1. A <nav> with three links: Home, Menu, Contact.\n"
    "2. A hero <section> with <h1>Bean & Brew</h1> and a one-line subheading.\n"
    "3. A <button id=\"order\">Order Now</button> wired in <script> with "
    "addEventListener that shows alert(\"Thank you for your order!\").\n"
    "4. A <form action=\"/submit\"> with a name text input, an email input, "
    "a submit button, and an onsubmit handler that calls preventDefault and "
    "shows a confirmation.\n"
    "5. Minimal embedded CSS in <style> and the JS in <script>.\n"
    "Write the entire compact HTML document now and finish it.\n"
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_my_server(port):
    subprocess.run(["bash", "-lc", f"pkill -f 'ds4-server.*--port {port}' 2>/dev/null || true"],
                   check=False)
    time.sleep(1)


def start_server(env, out_dir, port, cache, ctx, prefill_chunk=512, server_max_tokens=2048):
    env_prefix = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(env.items()))
    cmd = (
        f"cd /root/ds4 && {env_prefix} /root/ds4/ds4-server "
        f"-m {MODEL} --cuda --ssd-streaming "
        f"--ssd-streaming-cache-experts {cache} "
        f"--prefill-chunk {prefill_chunk} "
        f"-c {ctx} -n {server_max_tokens} "
        f"--host 127.0.0.1 --port {port} --cors"
    )
    stdout = open(os.path.join(out_dir, "server.stdout.log"), "wb")
    stderr = open(os.path.join(out_dir, "server.stderr.log"), "wb")
    json.dump(env, open(os.path.join(out_dir, "server_env.json"), "w"), indent=2)
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=stdout, stderr=stderr)
    return p


def wait_models(port, timeout_s):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def build_body(prompt, max_tokens, stream):
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "think": False,
        "thinking": {"type": "disabled"},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def post_nonstream(port, prompt, max_tokens, timeout):
    body = build_body(prompt, max_tokens, False)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return time.monotonic() - t0, resp


def post_stream(port, prompt, max_tokens, timeout, events_path):
    body = build_body(prompt, max_tokens, True)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    tokens = []          # list of (recv_monotonic, text)
    usage = None
    full = []
    t0 = time.monotonic()
    ev = open(events_path, "w")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            now = time.monotonic()
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    tokens.append((now - t0, piece))
                    full.append(piece)
                    ev.write(json.dumps({"t": now - t0, "text": piece}) + "\n")
    ev.close()
    wall = time.monotonic() - t0
    return wall, tokens, usage, "".join(full)


def segment_tps(tokens):
    """tokens: list of (t_recv_rel, text). Returns dict of segment rates."""
    n = len(tokens)
    if n == 0:
        return {"n": 0}
    recv = [t for t, _ in tokens]  # recv[0] is first token time (== TTFT)
    ttft = recv[0]

    def rate(lo, hi):
        # tokens indexed 1..n; recv list is 0-based (recv[i] = token i+1)
        hi = min(hi, n)
        if hi <= lo:
            return None
        span = recv[hi - 1] - recv[lo - 1]
        if span <= 0:
            return None
        return (hi - lo) / span

    return {
        "n": n,
        "ttft_s": round(ttft, 3),
        "seg_1_64_tps": rate(1, 64) and round(rate(1, 64), 3),
        "seg_65_256_tps": rate(64, 256) and round(rate(64, 256), 3),
        "seg_257plus_tps": rate(256, n) and round(rate(256, n), 3),
        "overall_decode_tps": (recv[-1] - recv[0]) > 0 and round((n - 1) / (recv[-1] - recv[0]), 3) or None,
        "wall_last_token_s": round(recv[-1], 3),
    }


def grade_html(text):
    """Coarse L0-L3 render grade for the coffee page."""
    t = text
    tl = t.lower()
    has_doctype = "<!doctype" in tl or "<html" in tl
    has_body = "<body" in tl
    has_close = "</html>" in tl
    has_nav = "<nav" in tl
    has_h1 = "bean & brew" in tl or "<h1" in tl
    has_button = 'id="order"' in tl or "order now" in tl
    has_form = "<form" in tl
    has_script = "<script" in tl
    has_alert = "alert(" in tl
    # repetition / loop detection: any 40-char substring repeated >=4x adjacently
    looped = False
    for L in (20, 40, 60):
        for i in range(0, max(0, len(t) - L * 4), L):
            chunk = t[i:i + L]
            if chunk.strip() and t[i:i + L * 4] == chunk * 4:
                looped = True
                break
        if looped:
            break
    # level
    if looped or not has_doctype:
        level = 0
    elif has_doctype and not has_body:
        level = 0
    elif has_body and not has_close:
        level = 1
    elif has_close and not (has_nav and has_h1 and has_button and has_form):
        level = 2
    else:
        level = 3
    return {
        "l0l3": level, "doctype": has_doctype, "body": has_body, "close_html": has_close,
        "nav": has_nav, "h1": has_h1, "button": has_button, "form": has_form,
        "script": has_script, "alert": has_alert, "looped": looped, "chars": len(t),
    }


def enforcement_from_trace(trace_path, keep_json_path):
    """Count distinct experts/layer used in the routing trace; check subset of keep."""
    if not os.path.exists(trace_path):
        return {"trace": "missing"}
    keep = None
    if keep_json_path and os.path.exists(keep_json_path):
        kj = json.load(open(keep_json_path))
        keep = {int(k): set(v) for k, v in kj["keep"].items()}
    used = collections.defaultdict(set)
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        rdr = f.read().splitlines()
    # detect header
    start = 0
    if rdr and rdr[0].startswith("pos"):
        start = 1
    for line in rdr[start:]:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            L = int(parts[1]); n = int(parts[2])
            es = [int(parts[3 + i]) for i in range(n)]
        except Exception:
            continue
        used[L].update(es)
    per_layer = {L: len(s) for L, s in sorted(used.items())}
    max_distinct = max(per_layer.values()) if per_layer else None
    violations = 0
    if keep is not None:
        for L, s in used.items():
            violations += len(s - keep.get(L, set()))
    return {"layers": len(per_layer), "max_distinct_per_layer": max_distinct,
            "distinct_per_layer_sample": dict(list(per_layer.items())[:5]),
            "violations_outside_keep": violations if keep is not None else "n/a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--keep-json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8014)
    ap.add_argument("--reserve", default="1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup-tokens", type=int, default=96)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--load-timeout", type=int, default=900)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trace_path = f"/dev/shm/route_k{args.k}_coffee.csv" if args.trace else ""

    env = {
        "DS4_LOCK_FILE": f"/tmp/ds4_keep8repro_{args.port}.lock",
        "DS4_CUDA_NO_DIRECT_IO": "1",
        "DS4_CUDA_KEEP_MODEL_PAGES": "1",
        "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB": str(args.reserve),
        "DS4_CUDA_NO_Q8_F16_CACHE": "1",
        "DS4_PACE": "0",
        "DS4_REAP_MASK_FILE": args.mask,
        "DS4_SPEX_STATS": "1",
    }
    if args.trace:
        env["DS4_SPEX_TRACE_ROUTING"] = trace_path
        env["DS4_SPEX_TRACE_ROUTING_WEIGHTS"] = "0"

    kill_my_server(args.port)
    log(f"K={args.k} cache={args.cache} reserve={args.reserve} trace={args.trace}: starting server")
    t_load0 = time.monotonic()
    p = start_server(env, args.out, args.port, args.cache, args.ctx)
    ok = wait_models(args.port, args.load_timeout)
    load_s = time.monotonic() - t_load0
    if not ok:
        log(f"SERVER FAILED TO COME UP in {load_s:.0f}s")
        kill_my_server(args.port)
        json.dump({"error": "server_no_models", "load_s": load_s},
                  open(os.path.join(args.out, "result.json"), "w"), indent=2)
        return 2
    log(f"server up in {load_s:.0f}s; warmup ({args.warmup_tokens} tok)")

    try:
        w_wall, _ = post_nonstream(args.port, COFFEE_PROMPT, args.warmup_tokens, 900)
        log(f"warmup done in {w_wall:.1f}s; measured stream ({args.max_tokens} tok)")
        m_wall, tokens, usage, full = post_stream(
            args.port, COFFEE_PROMPT, args.max_tokens, 1800,
            os.path.join(args.out, "stream_events_measured.jsonl"))
    finally:
        pass

    open(os.path.join(args.out, "content_measured.txt"), "w").write(full)
    seg = segment_tps(tokens)
    grade = grade_html(full)
    enf = enforcement_from_trace(trace_path, args.keep_json) if args.trace else {"trace": "off"}
    if args.trace and os.path.exists(trace_path):
        subprocess.run(["bash", "-lc", f"gzip -c {trace_path} > {os.path.join(args.out, 'route_measured.csv.gz')} && rm -f {trace_path}"], check=False)

    result = {
        "k": args.k, "cache": args.cache, "reserve": args.reserve,
        "load_s": round(load_s, 1), "warmup_wall_s": round(w_wall, 1),
        "measured_wall_s": round(m_wall, 1), "usage": usage,
        "segments": seg, "grade": grade, "enforcement": enf,
    }
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2)
    log("RESULT: " + json.dumps({"k": args.k, "seg": seg, "grade": {kk: grade[kk] for kk in ("l0l3", "close_html", "looped", "chars")}, "enf": enf}))
    kill_my_server(args.port)
    log(f"K={args.k} done; server torn down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
