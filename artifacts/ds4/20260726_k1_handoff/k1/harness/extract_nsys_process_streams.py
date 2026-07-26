#!/usr/bin/env python3
"""Extract target stdout/stderr captured in an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    args = parser.parse_args()

    streams: dict[str, list[str]] = {"stdout": [], "stderr": []}
    with sqlite3.connect(args.sqlite) as connection:
        rows = connection.execute(
            """
            SELECT filename.value, content.value
              FROM ProcessStreams AS process_stream
              JOIN StringIds AS filename
                ON filename.id = process_stream.filenameId
              JOIN StringIds AS content
                ON content.id = process_stream.contentId
             ORDER BY process_stream.globalPid, process_stream.filenameId
            """
        )
        for filename, content in rows:
            basename = Path(filename).name.lower()
            for stream_name in streams:
                if basename.startswith(f"{stream_name}_"):
                    streams[stream_name].append(content)
                    break

    args.stdout.write_text("".join(streams["stdout"]), encoding="utf-8")
    args.stderr.write_text("".join(streams["stderr"]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
