#!/usr/bin/env python3
"""Single DS4 run wrapper for W100 direct K23 rotate32 cache256 HTML2000.

This intentionally lives inside the run directory so the source tree remains
untouched. It reuses prompt/request/parsing helpers from scripts/run_ds4_exchange_matrix.py
without calling that script's main(), because main() stops all ds4-server
processes during cleanup.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace


RUN_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[2]
MATRIX_PATH = REPO_ROOT / "scripts" / "run_ds4_exchange_matrix.py"
WSL_EXE = os.environ.get("WSL_EXE") or r"C:\Windows\System32\wsl.exe"


def load_matrix():
    spec = importlib.util.spec_from_file_location("run_ds4_exchange_matrix", MATRIX_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MATRIX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matrix = load_matrix()


def wsl_bash(script: str, *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [WSL_EXE, "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc", script]
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check)


def wait_models(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - readiness diagnostic.
            last = exc
        time.sleep(1)
    raise TimeoutError(f"server did not become ready on {url}: {last}")


def build_env(run_dir: pathlib.Path) -> dict[str, str]:
    env = dict(matrix.BASE_ENV)
    env.update(
        {
            "DS4_PACE_WARMUP": "100",
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_RELEARN": "0",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_CACHE_TARGET_SLOTS": "256",
            "DS4_PACE_EXCHANGE_OBSERVE": "0",
            "DS4_PACE_WEIGHTED_SELECTED": "0",
            "DS4_SPEX_HIDDEN_PREFETCH": "0",
            "DS4_SPEX_HIDDEN_GPU_LOAD": "0",
            "DS4_SPEX_HIDDEN_GPU_SCORE": "0",
            "DS4_SPEX_HIDDEN_GPU_PREFETCH": "0",
            "DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN": "0",
            "DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS": "0",
            "DS4_SPEX_TRACE_ROUTING": "",
            "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "0",
            "DS4_EXPERT_TIERING": "observe",
            "DS4_EXPERT_TIERING_LOG": "",
            "DS4_EXPERT_TIERING_LOG_IDS": "0",
            "DS4_LOCK_FILE": matrix.wsl_path(run_dir / "ds4.lock"),
            "DS4_PACE_LOG": matrix.wsl_path(run_dir / "pace_events.jsonl"),
        }
    )
    return env


def env_delta(env: dict[str, str]) -> dict[str, dict[str, str | None]]:
    keys = sorted(set(matrix.BASE_ENV) | set(env))
    return {
        key: {"base": matrix.BASE_ENV.get(key), "effective": env.get(key)}
        for key in keys
        if matrix.BASE_ENV.get(key) != env.get(key)
    }


def write_json(path: pathlib.Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def start_server(*, run_dir: pathlib.Path, env: dict[str, str], port: int, args: argparse.Namespace) -> subprocess.Popen:
    env_prefix = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(env.items()))
    server_pid_file = matrix.wsl_path(run_dir / "server.pid.wsl")
    server_cmd = (
        "cd /root/ds4 && "
        f"{env_prefix} /root/ds4/ds4-server "
        f"-m {args.model} --cuda --ssd-streaming "
        f"--ssd-streaming-cache-experts {args.cache_experts} "
        f"--prefill-chunk {args.prefill_chunk} "
        f"-c {args.ctx} -n {args.server_max_tokens} "
        f"--host 127.0.0.1 --port {port} --cors"
    )
    wrapped = f"({server_cmd}) & server_pid=$!; echo $server_pid > {json.dumps(server_pid_file)}; wait $server_pid"
    stdout = (run_dir / "server.stdout.log").open("wb")
    stderr = (run_dir / "server.stderr.log").open("wb")
    launcher = [WSL_EXE, "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc", wrapped]
    return subprocess.Popen(launcher, stdout=stdout, stderr=stderr)


def stop_server(proc: subprocess.Popen, run_dir: pathlib.Path) -> None:
    pid_file = run_dir / "server.pid.wsl"
    if pid_file.exists():
        pid = pid_file.read_text(encoding="utf-8", errors="replace").strip()
        if pid.isdigit():
            wsl_bash(f"kill -TERM {pid} 2>/dev/null || true", timeout=8, check=False)
            time.sleep(1)
            wsl_bash(f"kill -KILL {pid} 2>/dev/null || true", timeout=8, check=False)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


def write_manifest(*, run_dir: pathlib.Path, env: dict[str, str], port: int, args: argparse.Namespace) -> None:
    prompt = matrix.PROMPTS["html"]
    manifest = {
        "runner_id": "html_w100_rotate32_k23_cache256_r01",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "profile": "SOTA_LOCAL_3060_TIMED",
        "variant": "w100_rotate32_k23_cache256",
        "variant_rationale": "Direct K0/full-router warmup for 100 tokens, then fixed K23 with rotate32 enabled; no breath, prebreath, relearn, routing trace, or hidden SPEX; cache256.",
        "prompt": {
            "name": "html",
            "sha256_16": matrix.prompt_hash(prompt),
            "chars": len(prompt),
        },
        "server": {
            "model": args.model,
            "ctx": args.ctx,
            "server_max_tokens": args.server_max_tokens,
            "cache_experts": args.cache_experts,
            "prefill_chunk": args.prefill_chunk,
            "port": port,
            "request_max_tokens": args.max_tokens,
            "warmups": 0,
            "pace_warmup_tokens": 100,
        },
        "request": {
            "temperature": 0,
            "stream": True,
            "think": False,
            "thinking": {"type": "disabled"},
        },
        "trace": {
            "routing_csv": None,
            "weights": False,
        },
        "source_artifacts": {
            "runner_stdout": "runner.stdout.log",
            "runner_stderr": "runner.stderr.log",
            "server_stdout": "html_w100_rotate32_k23_cache256_r01/server.stdout.log",
            "server_stderr": "html_w100_rotate32_k23_cache256_r01/server.stderr.log",
            "server_pid_wsl": "html_w100_rotate32_k23_cache256_r01/server.pid.wsl",
            "request_measured": "html_w100_rotate32_k23_cache256_r01/request_measured.json",
            "response_measured": "html_w100_rotate32_k23_cache256_r01/response_measured.json",
            "stream_events_measured": "html_w100_rotate32_k23_cache256_r01/stream_events_measured.jsonl",
            "content_measured": "html_w100_rotate32_k23_cache256_r01/content_measured.txt",
            "server_env": "html_w100_rotate32_k23_cache256_r01/server_env.json",
            "pace_events": "html_w100_rotate32_k23_cache256_r01/pace_events.jsonl",
        },
        "env_delta_from_profile": env_delta(env),
        "env_effective": env,
    }
    write_json(run_dir / "runner_manifest.json", manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8015)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--model", default="/root/models/ds4-2bit.gguf")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--server-max-tokens", type=int, default=2048)
    ap.add_argument("--cache-experts", type=int, default=256)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    args = ap.parse_args()

    run_dir = RUN_ROOT / "html_w100_rotate32_k23_cache256_r01"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = RUN_ROOT / "status.json"
    command_line = " ".join([sys.executable, str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])
    write_json(
        status_path,
        {
            "state": "starting",
            "updated_utc": dt.datetime.now(dt.UTC).isoformat(),
            "command": command_line,
            "port": args.port,
            "runner_pid": os.getpid(),
        },
    )

    env = build_env(run_dir)
    write_json(run_dir / "server_env.json", env)
    write_manifest(run_dir=run_dir, env=env, port=args.port, args=args)
    write_json(
        RUN_ROOT / "matrix_config.json",
        {
            "suite": "single",
            "prompts": "html",
            "variants": "w100_rotate32_k23_cache256",
            "runs": 1,
            "warmups": 0,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "port": args.port,
            "out_dir": str(RUN_ROOT),
            "model": args.model,
            "ctx": args.ctx,
            "server_max_tokens": args.server_max_tokens,
            "cache_experts": args.cache_experts,
            "prefill_chunk": args.prefill_chunk,
            "stream": True,
            "no_stop_existing": True,
            "profile": "SOTA_LOCAL_3060_TIMED",
            "base_env": matrix.BASE_ENV,
            "effective_env": env,
        },
    )

    server_proc = start_server(run_dir=run_dir, env=env, port=args.port, args=args)
    try:
        write_json(
            status_path,
            {
                "state": "waiting_for_server",
                "updated_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": command_line,
                "port": args.port,
                "runner_pid": os.getpid(),
                "server_launcher_pid": server_proc.pid,
            },
        )
        wait_models(args.port, timeout_s=120)
        server_pid = (run_dir / "server.pid.wsl").read_text(encoding="utf-8", errors="replace").strip() if (run_dir / "server.pid.wsl").exists() else None
        write_json(
            status_path,
            {
                "state": "request_running",
                "updated_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": command_line,
                "port": args.port,
                "runner_pid": os.getpid(),
                "server_launcher_pid": server_proc.pid,
                "server_pid_wsl": server_pid,
            },
        )
        wall_s, response = matrix.run_request(
            port=args.port,
            prompt_name="html",
            prompt=matrix.PROMPTS["html"],
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            out_path=run_dir,
            phase="measured",
            stream=True,
        )
        usage = response.get("usage") if isinstance(response, dict) else None
        content = matrix.response_content(response)
        log_metrics = matrix.parse_server_log(run_dir / "server.stderr.log")
        q = matrix.quality_flags("html", content)
        row = {
            "stem": "html_w100_rotate32_k23_cache256_r01",
            "evidence_type": "measured",
            "source_artifacts": "server.stderr.log,response_measured.json,stream_events_measured.jsonl,pace_events.jsonl",
            "profile": "SOTA_LOCAL_3060_TIMED",
            "prompt": "html",
            "variant": "w100_rotate32_k23_cache256",
            "variant_rationale": "Direct K0/full-router warmup for 100 tokens, then fixed K23 with rotate32; cache256.",
            "run": 1,
            "wall_s": round(wall_s, 3),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "stream_events": response.get("stream_events") if isinstance(response, dict) else None,
            "stream_content_events": response.get("stream_content_events") if isinstance(response, dict) else None,
            **log_metrics,
            **matrix.trace_stats(run_dir),
            **q,
        }
        fieldnames = list(row.keys())
        with (RUN_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        write_json(RUN_ROOT / "summary.json", [row])
        write_json(
            status_path,
            {
                "state": "completed",
                "updated_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": command_line,
                "port": args.port,
                "runner_pid": os.getpid(),
                "server_launcher_pid": server_proc.pid,
                "server_pid_wsl": server_pid,
                "wall_s": round(wall_s, 3),
                "summary": row,
            },
        )
        print(f"completed wall_s={wall_s:.3f}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve failure artifact.
        write_json(
            status_path,
            {
                "state": "failed",
                "updated_utc": dt.datetime.now(dt.UTC).isoformat(),
                "command": command_line,
                "port": args.port,
                "runner_pid": os.getpid(),
                "server_launcher_pid": server_proc.pid,
                "error": repr(exc),
            },
        )
        raise
    finally:
        stop_server(server_proc, run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
