#!/usr/bin/env python3
"""Summarize Windows bake quality-gate runner outputs as deterministic CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "suite/root",
    "arm",
    "run",
    "GPU",
    "model",
    "ctx",
    "cache_experts",
    "prefill_chunk",
    "server_max_tokens",
    "request_max_tokens",
    "temperature",
    "think",
    "stream",
    "grade",
    "finish_reason",
    "elapsed_s",
    "stream_events",
    "tok/s",
    "completion_tokens",
    "content_chars",
    "prompt_chars",
    "prompt_sha256",
    "runner_sha256",
    "binary_sha256",
    "mask_sha256",
]


class ArtifactError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    require_file(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSON: {exc}") from exc


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ArtifactError(f"missing required artifact: {path}")


def require_mapping(value: Any, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: expected JSON object")
    return value


def require_number(value: Any, name: str, path: Path) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{path}: expected numeric {name}")
    return value


def require_int(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{path}: expected integer {name}")
    return value


def extract_sha256(path: Path) -> str:
    require_file(path)
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) < 1 or len(parts[0]) != 64:
        raise ArtifactError(f"{path}: expected sha256sum format")
    digest = parts[0].lower()
    if any(ch not in "0123456789abcdef" for ch in digest):
        raise ArtifactError(f"{path}: invalid SHA-256 digest")
    return digest


def extract_gpu(path: Path) -> str:
    require_file(path)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if "," not in stripped:
            continue
        gpu = stripped.split(",", 1)[0].strip()
        if gpu:
            return gpu
    raise ArtifactError(f"{path}: could not find nvidia-smi CSV GPU line")


def prompt_from_request(request: dict[str, Any], path: Path) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ArtifactError(f"{path}: missing messages array")
    user_messages = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1 or not isinstance(user_messages[0], str):
        raise ArtifactError(f"{path}: expected exactly one string user prompt")
    return user_messages[0]


def parse_server_argv(path: Path) -> dict[str, str]:
    require_file(path)
    try:
        argv = shlex.split(path.read_text(encoding="utf-8"), posix=True)
    except ValueError as exc:
        raise ArtifactError(f"{path}: invalid shell argv: {exc}") from exc
    if not argv:
        raise ArtifactError(f"{path}: empty server argv")

    def value(*names: str) -> str:
        found = []
        for index, item in enumerate(argv):
            if item in names:
                if index + 1 >= len(argv):
                    raise ArtifactError(f"{path}: option {item} is missing its value")
                found.append(argv[index + 1])
        if len(found) != 1:
            raise ArtifactError(f"{path}: expected one of {names}, found {len(found)}")
        return found[0]

    return {
        "ctx": value("-c", "--ctx"),
        "cache_experts": value("--ssd-streaming-cache-experts"),
        "prefill_chunk": value("--prefill-chunk"),
        "server_max_tokens": value("-n", "--max-tokens"),
    }


def response_content(response: dict[str, Any], path: Path) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ArtifactError(f"{path}: expected exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ArtifactError(f"{path}: choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ArtifactError(f"{path}: missing choice message content")
    return message["content"]


def finish_reason(response: dict[str, Any], path: Path) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ArtifactError(f"{path}: expected exactly one choice")
    reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    if not isinstance(reason, str) or reason == "":
        raise ArtifactError(f"{path}: missing finish_reason")
    return reason


def grade_level(grade: dict[str, Any], path: Path) -> str:
    level = require_int(grade.get("level"), "level", path)
    if level < 0 or level > 3:
        raise ArtifactError(f"{path}: grade level must be L0-L3")
    return f"L{level}"


def summary_entry(summary: dict[str, Any], arm: str, run: int, path: Path) -> dict[str, Any]:
    rows = summary.get(arm)
    if not isinstance(rows, list):
        raise ArtifactError(f"{path}: missing summary rows for arm {arm}")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("run") == run
    ]
    if len(matches) != 1:
        raise ArtifactError(f"{path}: expected one summary row for {arm} run {run}")
    return matches[0]


def compare_summary(
    row: dict[str, Any],
    grade: str,
    response: dict[str, Any],
    response_path: Path,
    summary_path: Path,
) -> None:
    summary_grade = row.get("grade")
    if not isinstance(summary_grade, dict):
        raise ArtifactError(f"{summary_path}: summary grade must be an object")
    summary_level = grade_level(summary_grade, summary_path)
    if summary_level != grade:
        raise ArtifactError(f"{summary_path}: grade disagrees with grade artifact")

    if row.get("finish_reason") != finish_reason(response, response_path):
        raise ArtifactError(
            f"{summary_path}: finish_reason disagrees with {response_path}"
        )

    for key in ("elapsed_s", "usage"):
        if row.get(key) != response.get(key):
            raise ArtifactError(
                f"{summary_path}: {key} disagrees with {response_path}"
            )


def summarize_run(root: Path, summary: dict[str, Any], arm_dir: Path, run: int) -> dict[str, str]:
    suffix = f"r{run:02d}"
    request_path = arm_dir / f"request_{suffix}.json"
    response_path = arm_dir / f"response_{suffix}.json"
    grade_path = arm_dir / f"grade_{suffix}.json"
    events_path = arm_dir / f"events_{suffix}.jsonl"
    content_path = arm_dir / f"content_{suffix}.txt"
    summary_path = root / "summary.json"

    request = require_mapping(load_json(request_path), request_path)
    response = require_mapping(load_json(response_path), response_path)
    grade = require_mapping(load_json(grade_path), grade_path)
    require_file(events_path)
    require_file(content_path)

    arm = arm_dir.name
    summary_row = summary_entry(summary, arm, run, summary_path)
    grade = grade_level(grade, grade_path)
    compare_summary(summary_row, grade, response, response_path, summary_path)

    prompt = prompt_from_request(request, request_path)
    server = parse_server_argv(arm_dir / "server_argv.txt")
    temperature = require_number(request.get("temperature"), "temperature", request_path)
    model = request.get("model")
    if not isinstance(model, str) or not model:
        raise ArtifactError(f"{request_path}: missing model")
    request_max_tokens = require_int(request.get("max_tokens"), "max_tokens", request_path)
    stream = request.get("stream")
    if not isinstance(stream, bool):
        raise ArtifactError(f"{request_path}: stream must be boolean")
    think_value = request.get("think")
    if not isinstance(think_value, bool):
        thinking = request.get("thinking")
        if not isinstance(thinking, dict) or not isinstance(thinking.get("type"), str):
            raise ArtifactError(f"{request_path}: missing think/thinking setting")
        think_text = thinking["type"]
    else:
        think_text = str(think_value).lower()
    elapsed = require_number(response.get("elapsed_s"), "elapsed_s", response_path)
    if elapsed <= 0:
        raise ArtifactError(f"{response_path}: elapsed_s must be positive")
    stream_events = require_int(response.get("stream_events"), "stream_events", response_path)
    if stream_events < 0:
        raise ArtifactError(f"{response_path}: stream_events must be non-negative")

    usage = response.get("usage")
    completion_tokens = ""
    if usage is not None:
        if not isinstance(usage, dict):
            raise ArtifactError(f"{response_path}: usage must be an object or null")
        completion_tokens = str(
            require_int(usage.get("completion_tokens"), "completion_tokens", response_path)
        )

    content = content_path.read_text(encoding="utf-8")
    response_text = response_content(response, response_path)
    if content not in (response_text, response_text + "\n", response_text + "\r\n"):
        raise ArtifactError(f"{content_path}: content disagrees with {response_path}")

    mask_path = arm_dir / "mask.sha256"
    if arm == "k0":
        if mask_path.exists():
            raise ArtifactError(f"{mask_path}: K0 arm must not have a mask hash")
        mask_sha256 = ""
    else:
        mask_sha256 = extract_sha256(mask_path)

    return {
        "suite/root": str(root),
        "arm": arm,
        "run": str(run),
        "GPU": extract_gpu(arm_dir / "hardware.txt"),
        "model": model,
        "ctx": server["ctx"],
        "cache_experts": server["cache_experts"],
        "prefill_chunk": server["prefill_chunk"],
        "server_max_tokens": server["server_max_tokens"],
        "request_max_tokens": str(request_max_tokens),
        "temperature": f"{temperature:g}",
        "think": think_text,
        "stream": str(stream).lower(),
        "grade": grade,
        "finish_reason": finish_reason(response, response_path),
        "elapsed_s": f"{elapsed:g}",
        "stream_events": str(stream_events),
        "tok/s": f"{stream_events / elapsed:.6g}",
        "completion_tokens": completion_tokens,
        "content_chars": str(len(response_text)),
        "prompt_chars": str(len(prompt)),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "runner_sha256": extract_sha256(root / "runner.sha256"),
        "binary_sha256": extract_sha256(arm_dir / "binary.sha256"),
        "mask_sha256": mask_sha256,
    }


def summarize_root(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    summary_path = root / "summary.json"
    summary = require_mapping(load_json(summary_path), summary_path)
    rows: list[dict[str, str]] = []
    for arm in sorted(summary):
        if not isinstance(arm, str):
            raise ArtifactError(f"{summary_path}: arm names must be strings")
        arm_dir = root / arm
        if not arm_dir.is_dir():
            raise ArtifactError(f"missing arm directory: {arm_dir}")
        summary_rows = summary[arm]
        if not isinstance(summary_rows, list):
            raise ArtifactError(f"{summary_path}: rows for arm {arm} must be a list")
        runs = []
        for row in summary_rows:
            if not isinstance(row, dict):
                raise ArtifactError(f"{summary_path}: summary row for {arm} is not an object")
            runs.append(require_int(row.get("run"), "run", summary_path))
        for run in sorted(runs):
            rows.append(summarize_run(root, summary, arm_dir, run))
    return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit CSV rows for one or more Windows bake quality-gate roots."
    )
    parser.add_argument("roots", nargs="+", type=Path, help="runner output root(s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        rows = []
        for root in sorted(args.roots, key=lambda path: str(path)):
            rows.extend(summarize_root(root))
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
