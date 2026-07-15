#!/usr/bin/env python3
"""Archive one routing trace CSV as deterministic gzip with a verification manifest."""

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import os
import pathlib
import sys
import tempfile


VERSION = "1"
CHUNK_SIZE = 1024 * 1024
GZIP_COMPRESSLEVEL = 9
GZIP_FILENAME = ""
GZIP_MTIME = 0


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


def reject_collisions(input_path, output_path, manifest_path):
    input_key = path_key(input_path)
    output_key = path_key(output_path)
    manifest_key = path_key(manifest_path)

    if output_key == input_key or same_existing_file(output_path, input_path):
        raise ValueError(f"output collides with input: {output_path}")
    if manifest_key == input_key or same_existing_file(manifest_path, input_path):
        raise ValueError(f"manifest collides with input: {manifest_path}")
    if manifest_key == output_key or same_existing_file(manifest_path, output_path):
        raise ValueError(f"manifest collides with output: {manifest_path}")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    if manifest_path.exists():
        raise ValueError(f"manifest already exists: {manifest_path}")


def make_temp_path(final_path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(final_path.parent),
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle


def cleanup_temp(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def line_count_from_bytes(byte_count, newline_count, last_byte):
    if byte_count == 0:
        return 0
    if last_byte == b"\n":
        return newline_count
    return newline_count + 1


def copy_to_deterministic_gzip(input_path, temp_output_path):
    raw_sha = hashlib.sha256()
    compressed_sha = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    last_byte = None
    header = bytearray()
    saw_header_newline = False

    with input_path.open("rb") as src, temp_output_path.open("wb") as raw_out:
        with gzip.GzipFile(
            filename=GZIP_FILENAME,
            mode="wb",
            fileobj=raw_out,
            compresslevel=GZIP_COMPRESSLEVEL,
            mtime=GZIP_MTIME,
        ) as gz:
            while True:
                chunk = src.read(CHUNK_SIZE)
                if not chunk:
                    break

                raw_sha.update(chunk)
                byte_count += len(chunk)
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
                if not saw_header_newline:
                    newline_at = chunk.find(b"\n")
                    if newline_at == -1:
                        header.extend(chunk)
                    else:
                        header.extend(chunk[: newline_at + 1])
                        saw_header_newline = True
                gz.write(chunk)

        raw_out.flush()
        os.fsync(raw_out.fileno())

    line_count = line_count_from_bytes(byte_count, newline_count, last_byte)
    data_row_count = max(0, line_count - 1)
    if byte_count == 0:
        raise ValueError(f"empty input file: {input_path}")
    if data_row_count == 0:
        raise ValueError(f"input has no data rows: {input_path}")

    compressed_size = 0
    with temp_output_path.open("rb") as compressed:
        while True:
            chunk = compressed.read(CHUNK_SIZE)
            if not chunk:
                break
            compressed_sha.update(chunk)
            compressed_size += len(chunk)

    return {
        "size_bytes": byte_count,
        "sha256": raw_sha.hexdigest(),
        "line_count": line_count,
        "data_row_count": data_row_count,
        "header_size_bytes": len(header),
        "header_sha256": hashlib.sha256(bytes(header)).hexdigest(),
        "compressed_size_bytes": compressed_size,
        "compressed_sha256": compressed_sha.hexdigest(),
    }


def verify_roundtrip(temp_output_path, expected_size, expected_sha256):
    raw_sha = hashlib.sha256()
    byte_count = 0

    with temp_output_path.open("rb") as raw_in:
        with gzip.GzipFile(mode="rb", fileobj=raw_in) as gz:
            while True:
                chunk = gz.read(CHUNK_SIZE)
                if not chunk:
                    break
                raw_sha.update(chunk)
                byte_count += len(chunk)

    actual_sha256 = raw_sha.hexdigest()
    if byte_count != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(
            "roundtrip verification failed: "
            f"size {byte_count} != {expected_size} or sha256 {actual_sha256} != {expected_sha256}"
        )
    return {"size_bytes": byte_count, "sha256": actual_sha256}


def write_manifest_temp(manifest_path, payload):
    with make_temp_path(manifest_path) as tmp:
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        return pathlib.Path(tmp.name)


def build_manifest(args, stats, roundtrip):
    return {
        "tool": "archive_routing_trace.py",
        "version": VERSION,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "cwd": str(pathlib.Path.cwd()),
        "argv": [sys.argv[0], *(args.argv if hasattr(args, "argv") else sys.argv[1:])],
        "policy": {
            "archive": "single_csv_to_deterministic_gzip",
            "streaming": True,
            "csv_rewrite": False,
            "overwrite": False,
            "publish": "write_temp_verify_roundtrip_then_replace_with_output_rollback_on_manifest_failure",
        },
        "input": {
            "path": str(args.input),
            "resolved_path": str(args.input.resolve()),
            "size_bytes": stats["size_bytes"],
            "sha256": stats["sha256"],
            "line_count": stats["line_count"],
            "data_row_count": stats["data_row_count"],
            "header_size_bytes": stats["header_size_bytes"],
            "header_sha256": stats["header_sha256"],
        },
        "output": {
            "path": str(args.output),
            "resolved_path": str(args.output.resolve()),
            "format": "gzip",
            "size_bytes": stats["compressed_size_bytes"],
            "sha256": stats["compressed_sha256"],
        },
        "gzip": {
            "compresslevel": GZIP_COMPRESSLEVEL,
            "mtime": GZIP_MTIME,
            "filename": GZIP_FILENAME,
        },
        "roundtrip": {
            "verified": True,
            "decompressed_size_bytes": roundtrip["size_bytes"],
            "decompressed_sha256": roundtrip["sha256"],
        },
        "reconstruct": {
            "method": "gzip_decompress_output_bytes",
            "expected_raw_size_bytes": stats["size_bytes"],
            "expected_raw_sha256": stats["sha256"],
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Archive a routing trace CSV to deterministic CSV.gz plus manifest."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Input CSV path.")
    parser.add_argument(
        "--output", required=True, type=pathlib.Path, help="Output CSV.gz path."
    )
    parser.add_argument(
        "--manifest", required=True, type=pathlib.Path, help="Manifest JSON path."
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]

    temp_output = None
    temp_manifest = None
    try:
        reject_collisions(args.input, args.output, args.manifest)
        if not args.input.is_file():
            raise ValueError(f"input is not a file: {args.input}")

        with make_temp_path(args.output) as tmp:
            temp_output = pathlib.Path(tmp.name)
        stats = copy_to_deterministic_gzip(args.input, temp_output)
        roundtrip = verify_roundtrip(
            temp_output,
            expected_size=stats["size_bytes"],
            expected_sha256=stats["sha256"],
        )

        manifest = build_manifest(args, stats, roundtrip)
        temp_manifest = write_manifest_temp(args.manifest, manifest)

        os.replace(temp_output, args.output)
        temp_output = None
        try:
            os.replace(temp_manifest, args.manifest)
            temp_manifest = None
        except Exception:
            cleanup_temp(args.output)
            raise

        print(
            "archived {input} -> {output} ({rows} data rows, raw sha256={raw}, gzip sha256={gz})".format(
                input=args.input,
                output=args.output,
                rows=stats["data_row_count"],
                raw=stats["sha256"],
                gz=stats["compressed_sha256"],
            )
        )
        print(f"manifest: {args.manifest}")
        return 0
    except Exception as exc:
        if temp_output is not None:
            cleanup_temp(temp_output)
        if temp_manifest is not None:
            cleanup_temp(temp_manifest)
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
