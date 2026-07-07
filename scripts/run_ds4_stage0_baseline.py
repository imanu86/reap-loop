"""Run reproducible ds4 Stage-0 SPEX baseline measurements through WSL.

The script does not implement SPEX. It exercises the instrumented ds4 CUDA
baseline with DS4_SPEX_STATS=1, stores raw logs, and extracts a compact CSV.

Example:
    python scripts/run_ds4_stage0_baseline.py --mode warm --runs 5 --tokens 256
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL = "/mnt/d/models/ds4/DeepSeek-V4-Flash-IQ2XXS-imatrix.gguf"
DEFAULT_PROMPT = (
    "Write a concise technical explanation of cache locality in MoE inference. "
    "Mention expert residency, misses, and latency."
)

GEN_RE = re.compile(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s")
STATS_RE = re.compile(r"ds4:\s*SPEX stats:\s*(.*)")


def main() -> None:
    args = parse_args()
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"{started}_{args.mode}_r{args.runs}_n{args.tokens}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "started": started,
        "mode": args.mode,
        "runs": args.runs,
        "tokens": args.tokens,
        "ctx": args.ctx,
        "model": args.model,
        "ds4_dir": args.ds4_dir,
        "cache_experts": args.cache_experts,
        "prefill_chunk": args.prefill_chunk,
        "hotlist": args.hotlist,
        "drop_caches": args.drop_caches,
        "prompt": args.prompt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.build:
        build_cmd = f"cd {shlex.quote(args.ds4_dir)} && make cuda CUDA_ARCH=sm_86"
        run_wsl(build_cmd, out_dir / "build.log", timeout=args.build_timeout)

    rows = []
    for idx in range(args.runs):
        command = ds4_command(args, run_idx=idx)
        log_path = out_dir / f"run_{idx + 1:02d}.log"
        if args.dry_run:
            log_path.write_text(command + "\n", encoding="utf-8")
            continue
        result = run_wsl(command, log_path, timeout=args.timeout)
        row = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
        row.update(
            {
                "run": idx + 1,
                "mode": args.mode,
                "exit_code": result.returncode,
                "log": str(log_path),
            }
        )
        rows.append(row)
        write_summary(out_dir / "summary.csv", rows)
        (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"dry-run commands written to {out_dir}")
    else:
        print(f"stage0 baseline written to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cold", "warm"], required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ds4-dir", default="/root/ds4")
    parser.add_argument("--out", default="runs/ds4_stage0_baseline")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--cache-experts", default=None)
    parser.add_argument("--prefill-chunk", type=int, default=None)
    parser.add_argument("--hotlist", action="store_true", help="allow ds4 static hotlist preload")
    parser.add_argument("--drop-caches", action="store_true", help="drop WSL Linux page cache before each run")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--build-timeout", type=int, default=1200)
    return parser.parse_args()


def ds4_command(args: argparse.Namespace, *, run_idx: int) -> str:
    exports = ["export DS4_SPEX_STATS=1"]
    if args.mode == "warm":
        exports.append("export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1")
    else:
        exports.append("unset DS4_CUDA_NO_DIRECT_IO DS4_CUDA_KEEP_MODEL_PAGES")

    pre = []
    if args.mode == "cold" and args.drop_caches:
        pre.append("sync")
        pre.append("echo 3 > /proc/sys/vm/drop_caches || true")

    ds4_args = [
        "./ds4",
        "-m",
        args.model,
        "--cuda",
        "--ssd-streaming",
        "-c",
        str(args.ctx),
        "-n",
        str(args.tokens),
        "--temp",
        "0",
        "--seed",
        str(run_idx + 1),
        "-p",
        args.prompt,
    ]
    if not args.hotlist:
        ds4_args.append("--ssd-streaming-cold")
    if args.cache_experts:
        ds4_args.extend(["--ssd-streaming-cache-experts", args.cache_experts])
    if args.prefill_chunk:
        ds4_args.extend(["--prefill-chunk", str(args.prefill_chunk)])

    quoted = " ".join(shlex.quote(part) for part in ds4_args)
    parts = [f"cd {shlex.quote(args.ds4_dir)}", *exports, *pre, quoted]
    return " && ".join(parts) + " 2>&1"


def run_wsl(command: str, log_path: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.run(
            ["wsl", "-e", "bash", "-lc", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    return proc


def parse_log(text: str) -> dict[str, str | float | int]:
    row: dict[str, str | float | int] = {}
    gen = GEN_RE.search(text)
    if gen:
        row["prefill_tps"] = float(gen.group(1))
        row["generation_tps"] = float(gen.group(2))

    stats = STATS_RE.search(text)
    if stats:
        for item in stats.group(1).split():
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            value = value.rstrip(",")
            row[key] = parse_value(value)
    return row


def parse_value(value: str) -> str | float | int:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def write_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
