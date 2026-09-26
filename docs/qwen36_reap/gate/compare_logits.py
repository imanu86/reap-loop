#!/usr/bin/env python3
"""Stdlib-only, bounded-memory full-F32 gate. See README for the fixed policy."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

ATOL = 1e-5
RTOL = 1e-5
BLOCK_FLOATS = 16384
MODEL_SHA = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_metadata(prefix):
    path = Path(str(prefix) + ".json")
    check(path.stat().st_size <= 8 * 1024 * 1024, "metadata exceeds 8 MiB")
    with path.open(encoding="utf-8") as handle:
        m = json.load(handle, object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))
    check(isinstance(m, dict), "metadata must be an object")
    check(type(m.get("schema_version")) is int and m["schema_version"] == 1 and m.get("complete") is True,
          "incomplete/unsupported metadata")
    check(m.get("dtype") == "float32" and m.get("endianness") == "little", "expected little-endian float32")
    check(m.get("model_sha256") == MODEL_SHA, "wrong model identity")
    check(m.get("architecture") == "qwen35moe" and m.get("layer_count") == 40 and
          m.get("expert_count") == 256 and m.get("top_k") == 8, "wrong model architecture/E/K/layers")
    shape = m.get("shape")
    check(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), "invalid shape")
    steps, vocab = shape
    check(steps <= 16, "gate step count exceeds 16")
    for key, length in (("token_ids", steps), ("greedy_token_ids", steps), ("prompt_token_ids", None)):
        ids = m.get(key)
        check(isinstance(ids, list) and len(ids) > 0 and (length is None or len(ids) == length), "invalid " + key + " length")
        check(all(type(t) is int and 0 <= t < vocab for t in ids), "invalid " + key + " IDs")
    check(m.get("mode") in ("greedy", "teacher_forcing"), "unknown generation mode")
    if m["mode"] == "greedy":
        check(m["token_ids"] == m["greedy_token_ids"], "greedy mode did not choose argmax")
    check(m.get("warmup") is False and m.get("target_only") is True and m.get("fixed_count_including_eog") is True,
          "unsupported warmup/target/fixed-count policy")
    check(m.get("parse_special") is False, "special parsing must be disabled")
    check(m.get("prompt_mode") in ("raw_utf8", "single_token"), "invalid prompt mode")
    prompt = m.get("prompt_utf8")
    check(isinstance(prompt, str) and m.get("prompt_bytes") == len(prompt.encode("utf-8")), "raw prompt byte count mismatch")
    if m["prompt_mode"] == "single_token":
        check(len(m["prompt_token_ids"]) == 1 and prompt == "" and m.get("add_special") is False,
              "single-token prefill must be exactly one ID without BOS")
    else:
        check(bool(prompt) and m.get("add_special") is True, "invalid raw prompt tokenization policy")
    check(isinstance(m.get("settings"), dict), "missing effective settings")
    check(isinstance(m.get("environment"), dict), "missing environment snapshot")
    check(isinstance(m.get("steps"), list) and len(m["steps"]) == steps, "invalid step metadata count")
    for i, row in enumerate(m["steps"]):
        check(isinstance(row, dict) and row.get("step") == i and
              row.get("phase") == ("last_prefill" if i == 0 else "incremental_decode") and
              row.get("prefix_length") == len(m["prompt_token_ids"]) + i and
              row.get("logits_offset_bytes") == i * vocab * 4 and
              row.get("token_id") == m["token_ids"][i] and
              row.get("greedy_token_id") == m["greedy_token_ids"][i], "inconsistent step metadata")
    data_path = Path(str(prefix) + ".logits.f32")
    check(data_path.stat().st_size == steps * vocab * 4, "logits byte size != full declared shape (truncated/trailing data)")
    return m, data_path


def floats(handle, count):
    raw = handle.read(count * 4)
    check(len(raw) == count * 4, "logits truncated while streaming")
    return ((f[0], bits[0]) for f, bits in zip(struct.iter_unpack("<f", raw), struct.iter_unpack("<I", raw)))


def compare(reference, candidate):
    a, pa = load_metadata(reference)
    b, pb = load_metadata(candidate)
    check(a["shape"] == b["shape"], "shape mismatch")
    for key in ("model_sha256", "architecture", "layer_count", "expert_count", "top_k", "prompt_mode",
                "prompt_utf8", "prompt_token_ids", "add_special", "parse_special", "settings"):
        check(a[key] == b[key], "incompatible " + key)
    baseline_env = {k: v for k, v in a["environment"].items() if not k.startswith("QWEN36_REAP_")}
    candidate_env = {k: v for k, v in b["environment"].items() if not k.startswith("QWEN36_REAP_")}
    check(baseline_env == candidate_env, "incompatible cache environment")
    nsteps, vocab = a["shape"]
    prefix_equal = True
    report = {"schema_version": 1, "reference": str(reference), "candidate": str(candidate),
              "policy": {"atol": ATOL, "rtol": RTOL, "acceptance": "abs(a-b) <= atol + rtol*abs(reference)",
                         "maxrel": "abs(a-b)/max(abs(a),abs(b)); zero/zero=0",
                         "require_identical_tokens_and_top1": True, "all_logits_finite": True},
              "shape": a["shape"], "steps": [], "first_token_divergence": None,
              "tokens_equal": a["token_ids"] == b["token_ids"], "pass": True}
    with pa.open("rb") as fa, pb.open("rb") as fb:
        for step in range(nsteps):
            comparable = prefix_equal
            maxabs = maxrel = 0.0
            violations = nonfinite_a = nonfinite_b = bit_differences = 0
            tops = [None, None]
            best = [-math.inf, -math.inf]
            for start in range(0, vocab, BLOCK_FLOATS):
                count = min(BLOCK_FLOATS, vocab - start)
                for offset, ((x, bits_x), (y, bits_y)) in enumerate(zip(floats(fa, count), floats(fb, count))):
                    bit_differences += bits_x != bits_y
                    finite_x, finite_y = math.isfinite(x), math.isfinite(y)
                    nonfinite_a += not finite_x
                    nonfinite_b += not finite_y
                    for side, (value, finite) in enumerate(((x, finite_x), (y, finite_y))):
                        if finite and (tops[side] is None or value > best[side]):
                            best[side] = value
                            tops[side] = start + offset  # lowest ID wins a tie
                    if comparable and finite_x and finite_y:
                        delta = abs(x - y)
                        denom = max(abs(x), abs(y))
                        maxabs = max(maxabs, delta)
                        maxrel = max(maxrel, delta / denom if denom else 0.0)
                        violations += delta > ATOL + RTOL * abs(x)
            finite = nonfinite_a == 0 and nonfinite_b == 0
            top_metadata_ok = tops == [a["greedy_token_ids"][step], b["greedy_token_ids"][step]]
            top1_equal = tops[0] == tops[1]
            token_equal = a["token_ids"][step] == b["token_ids"][step]
            if not token_equal and report["first_token_divergence"] is None:
                report["first_token_divergence"] = step
            row_ok = comparable and finite and top_metadata_ok and top1_equal and token_equal and violations == 0
            report["steps"].append({"step": step, "comparable_prefix": comparable,
                                    "maxabs": maxabs if comparable and finite else None,
                                    "maxrel": maxrel if comparable and finite else None,
                                    "bit_identical": bit_differences == 0 if comparable else None,
                                    "bit_differences": bit_differences if comparable else None,
                                    "tolerance_violations": violations if comparable else None,
                                    "nonfinite_reference": nonfinite_a, "nonfinite_candidate": nonfinite_b,
                                    "top1_reference": tops[0], "top1_candidate": tops[1],
                                    "top1_equal": top1_equal, "top1_metadata_valid": top_metadata_ok,
                                    "token_equal": token_equal, "pass": row_ok})
            report["pass"] &= row_ok
            prefix_equal &= token_equal
        check(fa.read(1) == b"" and fb.read(1) == b"", "trailing data appeared while streaming")
    measured = [r for r in report["steps"] if r["maxabs"] is not None]
    report["maxabs"] = max((r["maxabs"] for r in measured), default=None)
    report["maxrel"] = max((r["maxrel"] for r in measured), default=None)
    report["compared_steps"] = sum(r["comparable_prefix"] for r in report["steps"])
    report["bit_identical"] = all(r["bit_identical"] is True for r in report["steps"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="prefix (without .json/.logits.f32)")
    parser.add_argument("candidate", help="prefix (without .json/.logits.f32)")
    parser.add_argument("--output", help="optional new JSON report; exclusive create, never overwrite")
    args = parser.parse_args(argv)
    try:
        result = compare(args.reference, args.candidate)
        code = 0 if result["pass"] else 1
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        result, code = {"pass": False, "error": str(error)}, 2
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        try:
            with open(args.output, "x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except OSError as error:
            print("exclusive report write failed: " + str(error), file=sys.stderr)
            return 2
    else:
        print(text, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
