#!/usr/bin/env python3
"""Analyze M1 DS4 HTML runs without using repeat/ngram as quality verdicts.

The functional verdict is always the L0-L3 grade. The n-gram scan is only a
diagnostic onset estimate for loops/tails, as required by docs/HANDOFF_CODEX.md.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"\S+")


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def as_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values):
    nums = [v for v in (as_float(x) for x in values) if v is not None]
    if not nums:
        return ""
    return round(statistics.median(nums), 4)


def load_jsonl(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def line_block_repeat(lines: list[str], repeats: int = 3) -> tuple[int, str] | None:
    if len(lines) < repeats:
        return None
    max_block = min(8, len(lines) // repeats)
    for block_len in range(1, max_block + 1):
        need = block_len * repeats
        tail = lines[-need:]
        block = tail[-block_len:]
        sample = "\n".join(block).strip()
        if len(sample) < 12:
            continue
        if all(tail[i * block_len : (i + 1) * block_len] == block for i in range(repeats)):
            return block_len, sample[:160]
    return None


def ngram_repeat(tokens: list[str], n: int = 3, window: int = 120, repeats: int = 3) -> tuple[str, int] | None:
    tail = tokens[-window:]
    if len(tail) < n * repeats:
        return None
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(0, len(tail) - n + 1):
        gram = tuple(tail[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= repeats:
            return " ".join(gram)[:160], counts[gram]
    return None


def diagnostic_onset(events: list[dict]) -> dict[str, str]:
    content = ""
    html_close_event = ""
    for event in events:
        content += event.get("delta") or ""
        event_index = clean(event.get("event_index"))
        if not html_close_event and "</html>" in content.lower():
            html_close_event = event_index

        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        line_hit = line_block_repeat(lines)
        if line_hit is not None:
            return {
                "loop_onset_event_est": event_index,
                "coherent_until_event_est": clean(int(event_index) - 1) if event_index.isdigit() else "",
                "loop_kind": "line_block_repeat",
                "loop_sample": line_hit[1],
                "html_close_event": html_close_event,
            }

        tokens = TOKEN_RE.findall(content)
        gram_hit = ngram_repeat(tokens)
        if gram_hit is not None:
            return {
                "loop_onset_event_est": event_index,
                "coherent_until_event_est": clean(int(event_index) - 1) if event_index.isdigit() else "",
                "loop_kind": "ngram3_window120_repeat3",
                "loop_sample": gram_hit[0],
                "html_close_event": html_close_event,
            }
    return {
        "loop_onset_event_est": "",
        "coherent_until_event_est": "",
        "loop_kind": "",
        "loop_sample": "",
        "html_close_event": html_close_event,
    }


def analyze_root(root: Path) -> list[dict[str, str]]:
    summary = root / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"missing summary.csv: {summary}")
    rows: list[dict[str, str]] = []
    with summary.open(newline="", encoding="utf-8", errors="replace") as handle:
        for src in csv.DictReader(handle):
            stem = src["stem"]
            run_dir = root / stem
            phase = src.get("final_attempt_phase") or "measured"
            content_path = run_dir / f"content_{phase}.txt"
            if not content_path.exists():
                content_path = run_dir / "content_measured.txt"
            content = content_path.read_text(encoding="utf-8", errors="replace") if content_path.exists() else ""
            stream_path = run_dir / f"stream_events_{phase}.jsonl"
            if not stream_path.exists():
                stream_path = run_dir / "stream_events_measured.jsonl"
            events = load_jsonl(stream_path)
            server_log = (run_dir / "server.stderr.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "server.stderr.log").exists() else ""
            onset = diagnostic_onset(events)
            row = {
                "stem": stem,
                "variant": clean(src.get("variant")),
                "run_index": clean(src.get("run_index") or src.get("run")),
                "final_attempt_phase": clean(phase),
                "stream_status": "stream_failed" if "final stream failed" in server_log else "ok",
                "completion_tokens": clean(src.get("completion_tokens")),
                "stream_events": clean(src.get("stream_events")),
                "content_chars": clean(len(content)),
                "emits_html_close": clean(int("</html>" in content.lower())),
                "html_close_event": onset["html_close_event"],
                "l0l3": clean(src.get("l0l3")),
                "client_stop_reason": clean(src.get("client_stop_reason")),
                "first_client_stop_reason": clean(src.get("first_client_stop_reason")),
                "retry_attempts": clean(src.get("retry_attempts")),
                "avg_tps": clean(src.get("avg_tps")),
                "prompt_s": clean(src.get("prompt_s")),
                "finish_s": clean(src.get("finish_s")),
                "loop_onset_event_est": onset["loop_onset_event_est"],
                "coherent_until_event_est": onset["coherent_until_event_est"],
                "loop_kind": onset["loop_kind"],
                "loop_sample": onset["loop_sample"],
            }
            rows.append(row)
    return rows


def write_outputs(root: Path, rows: list[dict[str, str]]) -> None:
    out_csv = root / "m1_analysis.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)

    lines = [
        f"# M1 Analysis - {root.name}",
        "",
        "Quality verdicts below are L0-L3 only. Loop/onset fields are diagnostics from n=3/window120 or repeated line blocks.",
        "",
        "## Per Seed",
        "",
    ]
    cols = [
        "stem",
        "stream_status",
        "completion_tokens",
        "stream_events",
        "emits_html_close",
        "l0l3",
        "client_stop_reason",
        "retry_attempts",
        "avg_tps",
        "prompt_s",
        "loop_onset_event_est",
        "coherent_until_event_est",
        "loop_kind",
    ]
    lines.extend(markdown_table(rows, cols))
    lines.extend(["", "## By Variant", ""])
    summary_rows = []
    for variant, grp in groups.items():
        summary_rows.append(
            {
                "variant": variant,
                "runs": clean(len(grp)),
                "l0l3_values": ",".join(row["l0l3"] for row in grp),
                "html_close_runs": clean(sum(1 for row in grp if row["emits_html_close"] == "1")),
                "client_stop_runs": clean(sum(1 for row in grp if row["client_stop_reason"])),
                "stream_failed_runs": clean(sum(1 for row in grp if row["stream_status"] != "ok")),
                "avg_tps_median": clean(median(row["avg_tps"] for row in grp)),
                "prompt_s_median": clean(median(row["prompt_s"] for row in grp)),
            }
        )
    lines.extend(markdown_table(summary_rows, list(summary_rows[0].keys())))
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, str]], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for col in cols:
            value = clean(row.get(col, "")).replace("|", "/")
            if len(value) > 120:
                value = value[:117] + "..."
            vals.append(value)
        out.append("| " + " | ".join(vals) + " |")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: analyze_m1_html_runs.py RUN_ROOT [RUN_ROOT ...]")
    for arg in argv:
        root = Path(arg)
        if not root.is_absolute():
            root = ROOT / root
        rows = analyze_root(root)
        if rows:
            write_outputs(root, rows)
            print(root / "ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
