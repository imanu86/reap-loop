#!/usr/bin/env python3
"""podC E-DET shadow: one ds4-server per run with S1 sensor ON (patch 0012).

Launches ds4-server with a STATIC PACE mask (W50, KEEP=K, ROTATE=0), waits for
the HTTP port, sends ONE greedy chat completion, saves content + response, then
kills the server. The S1 per-(token,layer) sensor CSV is written by the engine
to <run_dir>/s1.csv (DS4_REAP_SENSOR_LOG). No routing trace (sensor is
independent; keeps generation faster).

Usage: podrun.py <run_dir> <prompt:html|html_coffee> <keep> <ctx> <max_tokens> <server_max_tokens>
"""
import json, os, socket, subprocess, sys, time, urllib.request, urllib.error, signal

MODEL = "/root/models/ds4-2bit.gguf"
SERVER = "/root/bin/ds4-server"
PORT = 8014
CACHE_EXPERTS = 256
PREFILL_CHUNK = 512

PROMPTS = {
    "html": (
        "Crea una landing page HTML/CSS/JS single-file per un negozio di "
        "programmazione AI in stile cyberpunk. Deve avere un modulo contatti "
        "e un popup JS che dice richiesta inviata. Codice valido e compatto."
    ),
    "html_coffee": (
        "Write a COMPLETE and COMPACT single-file HTML page for a coffee shop. "
        "Output ONLY the HTML, nothing else. Keep the CSS SHORT (about 10-15 "
        "rules max) — prioritize a COMPLETE, working page over elaborate "
        "styling. The page MUST be fully closed with </html> and MUST contain "
        "all of these:\n"
        "1. A <nav> with three links: Home, Menu, Contact.\n"
        "2. A hero <section> with <h1>Bean & Brew</h1> and a one-line "
        "subheading.\n"
        "3. A <button id=\"order\">Order Now</button> wired in <script> with "
        "addEventListener that shows alert(\"Thank you for your order!\").\n"
        "4. A <form action=\"/submit\"> with a name text input, an email input, "
        "a submit button, and an onsubmit handler that calls preventDefault and "
        "shows a confirmation.\n"
        "5. Minimal embedded CSS in <style> and the JS in <script>.\n"
        "Write the entire compact HTML document now and finish it.\n"
    ),
}


def build_env(run_dir, keep):
    e = dict(os.environ)
    e.update({
        "DS4_CUDA_NO_DIRECT_IO": "1",
        "DS4_CUDA_KEEP_MODEL_PAGES": "1",
        "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB": "0.25",
        "DS4_PACE": "1",
        "DS4_PACE_WARMUP": "50",
        "DS4_PACE_KEEP": str(keep),
        "DS4_PACE_KEEP_MIN": str(keep),
        "DS4_PACE_KEEP_MAX": "96",
        "DS4_PACE_KEEP_STEP": "0",
        "DS4_PACE_BREATH_EVERY": "999999",
        "DS4_PACE_RELEARN": "0",
        "DS4_PACE_WRAP": "1",
        "DS4_PACE_ROTATE": "0",
        "DS4_PACE_DEBUG": "1",
        "DS4_SPEX_STATS": "1",
        "DS4_CUDA_NO_Q8_F16_CACHE": "1",
        "DS4_REAP_PREFETCH_THREADS": "16",
        "DS4_REAP_PREFETCH_LOCK": "1",
        "DS4_SPEX_TRACE_ROUTING": "",
        "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "0",
        "DS4_REAP_SENSOR_LOG": os.path.join(run_dir, "s1.csv"),
    })
    return e


def wait_port(port, logf, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(3)
    return False


def main():
    run_dir, prompt_name, keep, ctx, max_tokens, smax = sys.argv[1:7]
    keep = int(keep); ctx = int(ctx); max_tokens = int(max_tokens); smax = int(smax)
    os.makedirs(run_dir, exist_ok=True)
    env = build_env(run_dir, keep)
    slog = open(os.path.join(run_dir, "server.log"), "wb")
    cmd = [SERVER, "-m", MODEL, "--cuda", "--ssd-streaming",
           "--ssd-streaming-cache-experts", str(CACHE_EXPERTS),
           "--prefill-chunk", str(PREFILL_CHUNK),
           "-c", str(ctx), "-n", str(smax),
           "--host", "127.0.0.1", "--port", str(PORT), "--cors"]
    json.dump({"cmd": cmd, "keep": keep, "ctx": ctx, "max_tokens": max_tokens,
               "prompt": prompt_name}, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=slog)
    try:
        if not wait_port(PORT, slog, timeout=600):
            print("SERVER_TIMEOUT"); proc.terminate(); return 2
        time.sleep(2)
        body = {"model": "deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
                    {"role": "user", "content": PROMPTS[prompt_name]}],
                "max_tokens": max_tokens, "temperature": 0,
                "stream": False, "think": False, "thinking": {"type": "disabled"}}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                     data=data, headers={"Content-Type": "application/json"})
        t0 = time.time()
        raw = urllib.request.urlopen(req, timeout=5400).read()
        wall = time.time() - t0
        open(os.path.join(run_dir, "response.json"), "wb").write(raw)
        try:
            resp = json.loads(raw.decode("utf-8", errors="replace"))
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = resp.get("usage", {})
        except Exception as ex:
            content = ""; usage = {"parse_error": str(ex)}
        open(os.path.join(run_dir, "content.txt"), "w", encoding="utf-8", errors="replace").write(content)
        n = usage.get("completion_tokens", "?")
        tps = (n / wall) if isinstance(n, int) and wall > 0 else None
        summ = {"wall_s": round(wall, 1), "completion_tokens": n,
                "tps": round(tps, 2) if tps else None, "content_chars": len(content),
                "usage": usage}
        json.dump(summ, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
        print("DONE", json.dumps(summ))
        return 0
    finally:
        try:
            proc.send_signal(signal.SIGTERM); proc.wait(timeout=20)
        except Exception:
            proc.kill()
        slog.close()


if __name__ == "__main__":
    sys.exit(main())
