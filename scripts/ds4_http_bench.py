#!/usr/bin/env python3
"""Run a tiny OpenAI-compatible DS4 HTTP benchmark and save artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def post_json(url: str, body: dict, timeout: int) -> tuple[float, dict | str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return time.perf_counter() - t0, {"http_error": exc.code, "body": parsed}
    except Exception as exc:  # pragma: no cover - command-line diagnostic path.
        return time.perf_counter() - t0, {"error": type(exc).__name__, "message": str(exc)}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return time.perf_counter() - t0, parsed


def tail_file(path: pathlib.Path, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="ds4_http")
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--system", default="Rispondi in modo diretto, utile e senza ragionamento visibile.")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--server-log")
    ap.add_argument("--log-tail-lines", type=int, default=500)
    args = ap.parse_args()
    if bool(args.prompt) == bool(args.prompt_file):
        ap.error("pass exactly one of --prompt or --prompt-file")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    prompt = args.prompt
    if args.prompt_file:
        prompt = pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
        "think": False,
        "thinking": {"type": "disabled"},
    }

    results = []
    total = args.warmups + args.runs
    for idx in range(total):
        phase = "warmup" if idx < args.warmups else "measured"
        run_id = idx + 1
        (out_dir / f"request_{run_id:02d}_{phase}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        wall_s, response = post_json(args.url, body, args.timeout)
        (out_dir / f"response_{run_id:02d}_{phase}.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        usage = response.get("usage") if isinstance(response, dict) else None
        content = None
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
        results.append(
            {
                "run": run_id,
                "phase": phase,
                "wall_s": wall_s,
                "usage": usage,
                "content_chars": len(content or ""),
                "content_prefix": (content or "")[:120],
            }
        )
        print(f"{phase} run={run_id} wall_s={wall_s:.3f} usage={usage}", flush=True)

    server_log_tail = []
    if args.server_log:
        server_log_tail = tail_file(pathlib.Path(args.server_log), args.log_tail_lines)
        (out_dir / "server_log_tail.txt").write_text(
            "\n".join(server_log_tail) + ("\n" if server_log_tail else ""),
            encoding="utf-8",
        )

    meta = {
        "label": args.label,
        "started_utc": started,
        "finished_utc": utc_now(),
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "warmups": args.warmups,
        "runs": args.runs,
        "results": results,
        "server_log": args.server_log,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
