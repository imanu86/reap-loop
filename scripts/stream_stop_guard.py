#!/usr/bin/env python3
"""Stream one OpenAI-compatible request and stop on completion or repetition."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


TOKEN_RE = re.compile(r"\S+")


def repeated_ngram(text: str, n: int, window: int, count: int):
    tokens = TOKEN_RE.findall(text)[-window:]
    need = n * count
    if n < 1 or count < 2 or len(tokens) < need:
        return None
    tail = tokens[-need:]
    phrase = tail[-n:]
    if all(tail[i * n : (i + 1) * n] == phrase for i in range(count)):
        return {"kind": "token_ngram", "sample": " ".join(phrase)[:160]}
    return None


def repeated_line_block(text: str, count: int):
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < count:
        return None
    for block_len in range(1, min(8, len(lines) // count) + 1):
        tail = lines[-block_len * count :]
        block = tail[-block_len:]
        sample = "\n".join(block)
        if len(sample.strip()) < 12:
            continue
        if all(tail[i * block_len : (i + 1) * block_len] == block for i in range(count)):
            return {"kind": "line_block", "sample": sample[:160]}
    return None


def stop_reason(text: str, args):
    close_idx = text.lower().find("</html>") if args.stop_html_close else -1
    if close_idx >= 0:
        return {
            "reason": "client_stop_html_close",
            "trim_to_chars": close_idx + len("</html>"),
        }
    if not args.stop_repeat:
        return None
    repeat = repeated_line_block(text, args.repeat_count)
    if repeat is None:
        repeat = repeated_ngram(
            text, args.repeat_ngram, args.repeat_window, args.repeat_count
        )
    if repeat:
        return {"reason": f"client_stop_repeat_{repeat['kind']}", **repeat}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--stop-html-close", action="store_true")
    parser.add_argument("--stop-repeat", action="store_true")
    parser.add_argument("--repeat-ngram", type=int, default=3)
    parser.add_argument("--repeat-window", type=int, default=120)
    parser.add_argument("--repeat-count", type=int, default=3)
    args = parser.parse_args()

    body = args.request.read_bytes()
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    usage = None
    finish_reason = None
    stopped = None
    started = time.perf_counter()
    event_count = 0

    with args.events.open("w", encoding="utf-8") as events:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                event_count += 1
                choices = obj.get("choices") or []
                delta = ""
                if choices:
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                usage = obj.get("usage") or usage
                if delta:
                    chunks.append(delta)
                    content = "".join(chunks)
                    stopped = stop_reason(content, args)
                    if stopped and stopped.get("trim_to_chars") is not None:
                        chunks = [content[: int(stopped["trim_to_chars"])]]
                events.write(
                    json.dumps(
                        {
                            "event": event_count,
                            "t_s": round(time.perf_counter() - started, 6),
                            "delta": delta,
                            "content_chars": sum(map(len, chunks)),
                            "finish_reason": finish_reason,
                            "client_stop": stopped,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                events.flush()
                if stopped:
                    finish_reason = stopped["reason"]
                    break

    result = {
        "stream": True,
        "stream_events": event_count,
        "elapsed_s": round(time.perf_counter() - started, 6),
        "usage": usage,
        "client_stop": stopped,
        "choices": [
            {"message": {"role": "assistant", "content": "".join(chunks)},
             "finish_reason": finish_reason}
        ],
    }
    args.response.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"finish_reason": finish_reason, "client_stop": stopped}))


if __name__ == "__main__":
    main()
