#!/usr/bin/env python3
"""Merge DS4 full-decode routing trace CSV files reproducibly.

The merger treats inputs as bytes: it requires every input to have the exact
same header line, emits that header once, then appends each input body in the
CLI order. It does not infer cutoffs, deduplicate rows, or rewrite CSV data.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile


VERSION = "1"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def split_header(data, path):
    if not data:
        raise ValueError(f"empty input file: {path}")
    newline_at = data.find(b"\n")
    if newline_at == -1:
        return data, b""
    return data[:newline_at + 1], data[newline_at + 1:]


def count_lines(data):
    if not data:
        return 0
    return len(data.splitlines())


def path_key(path):
    resolved = str(pathlib.Path(path).resolve())
    if os.name == "nt":
        return os.path.normcase(os.path.abspath(resolved))
    return os.path.abspath(resolved)


def same_existing_file(left, right):
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def reject_collisions(inputs, output, manifest):
    input_keys = {}
    for raw in inputs:
        key = path_key(raw)
        if key in input_keys:
            raise ValueError(f"duplicate input path: {raw}")
        for previous in input_keys.values():
            if same_existing_file(raw, previous):
                raise ValueError(f"duplicate input file: {raw}")
        input_keys[key] = raw

    output_key = path_key(output)
    if output_key in input_keys:
        raise ValueError(f"output collides with input: {output}")
    for raw in input_keys.values():
        if same_existing_file(output, raw):
            raise ValueError(f"output collides with input: {output}")

    if manifest is not None:
        manifest_key = path_key(manifest)
        if manifest_key in input_keys:
            raise ValueError(f"manifest collides with input: {manifest}")
        if manifest_key == output_key:
            raise ValueError(f"manifest collides with output: {manifest}")
        for raw in input_keys.values():
            if same_existing_file(manifest, raw):
                raise ValueError(f"manifest collides with input: {manifest}")
        if same_existing_file(manifest, output):
            raise ValueError(f"manifest collides with output: {manifest}")


def load_input(path, expected_header):
    data = path.read_bytes()
    header, body = split_header(data, path)
    if expected_header is not None and header != expected_header:
        raise ValueError(f"header mismatch: {path}")
    if not body:
        raise ValueError(f"input has no data rows: {path}")
    if not body.endswith(b"\n"):
        raise ValueError(f"input body does not end with LF: {path}")
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "header_sha256": sha256_bytes(header),
        "line_count": count_lines(data),
        "data_line_count": count_lines(body),
        "header": header,
        "body": body,
    }


def write_bytes_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def write_json_atomic(path, payload):
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_atomic(path, data)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Merge DS4 full-decode routing trace CSVs without rewriting rows. "
            "All input headers must be byte-identical."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="Input CSV files, merged in the order provided.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="Output CSV path. Must not be one of the inputs.",
    )
    parser.add_argument(
        "-m",
        "--manifest",
        required=True,
        type=pathlib.Path,
        help="Manifest JSON path. Written atomically.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        reject_collisions(args.inputs, args.output, args.manifest)

        loaded = []
        expected_header = None
        for path in args.inputs:
            item = load_input(path, expected_header)
            if expected_header is None:
                expected_header = item["header"]
            loaded.append(item)

        output_bytes = expected_header + b"".join(item["body"] for item in loaded)
        write_bytes_atomic(args.output, output_bytes)

        output_stat = args.output.stat()
        manifest = {
            "tool": "merge_full_decode_traces.py",
            "version": VERSION,
            "created_at_utc": _dt.datetime.now(_dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "cwd": str(pathlib.Path.cwd()),
            "argv": [sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            "policy": {
                "merge_order": "cli_input_order",
                "header": "byte_identical_required",
                "dedupe": False,
                "cutoff": False,
                "csv_rewrite": False,
            },
            "inputs": [
                {key: value for key, value in item.items() if key not in ("header", "body")}
                for item in loaded
            ],
            "output": {
                "path": str(args.output),
                "resolved_path": str(args.output.resolve()),
                "size_bytes": output_stat.st_size,
                "sha256": sha256_bytes(output_bytes),
                "line_count": count_lines(output_bytes),
                "data_line_count": sum(item["data_line_count"] for item in loaded),
            },
        }
        write_json_atomic(args.manifest, manifest)

        print(
            "merged {inputs} inputs -> {output} ({rows} data lines, sha256={sha})".format(
                inputs=len(loaded),
                output=args.output,
                rows=manifest["output"]["data_line_count"],
                sha=manifest["output"]["sha256"],
            )
        )
        print(f"manifest: {args.manifest}")
        return 0
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
