#!/usr/bin/env python3
"""Verdict PASS/FAIL for the rewind-exactness harness (patch 0027, R1 gate).

The ds4 patch `0027-rewind-exactness-harness.patch` arms an in-engine probe with
``DS4_REWIND_TEST="p,k"``. It decodes greedily, snapshots the compressor frontier
and PACE controller state at position ``p``, continues to ``p+k``, then restores +
``ds4_session_rewind(p)`` and REPLAYS the same ``k`` inputs. It logs one JSONL
record per test to ``DS4_REWIND_TEST_LOG`` (or stderr)::

    {"ev":"rewind_test","snap_pos":400,"k":200,"replay_ok":1,
     "first_div":-1,"pre":[<k ids>],"post":[<k ids>]}

The rewind is EXACT iff, for every record, the greedy resume (``post``) reproduces
the pre-rewind stream (``pre``) token-for-token. This script reads the JSONL and
returns that verdict, reporting the first divergent index for any FAIL.

Usage:
    verify_rewind_exactness.py LOG.jsonl [LOG2.jsonl ...]
    verify_rewind_exactness.py -            # read JSONL from stdin
    verify_rewind_exactness.py --selftest   # run built-in unit test

Exit code: 0 = all records PASS, 1 = at least one FAIL, 2 = usage / no records.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

REWIND_EV = "rewind_test"


@dataclass
class RecordVerdict:
    snap_pos: Optional[int]
    k: Optional[int]
    replay_ok: bool
    n_pre: int
    n_post: int
    first_div: int          # -1 == no divergence
    passed: bool
    reason: str             # short human-readable status

    def summary(self) -> str:
        head = f"snap_pos={self.snap_pos} k={self.k}"
        if self.passed:
            return f"PASS  {head}  ({self.n_pre} tokens bit-identical)"
        return f"FAIL  {head}  {self.reason}"


def verdict_for(rec: dict) -> RecordVerdict:
    """Compute PASS/FAIL for a single decoded rewind_test record."""
    snap_pos = rec.get("snap_pos")
    k = rec.get("k")
    replay_ok = bool(rec.get("replay_ok", 0))
    pre = rec.get("pre")
    post = rec.get("post")

    if not isinstance(pre, list) or not isinstance(post, list):
        return RecordVerdict(snap_pos, k, replay_ok, 0, 0, 0, False,
                             "record missing 'pre'/'post' token arrays")
    n_pre, n_post = len(pre), len(post)

    if not replay_ok:
        return RecordVerdict(snap_pos, k, replay_ok, n_pre, n_post, 0, False,
                             "engine reported replay_ok=0 (replay eval failed)")
    if n_pre == 0 or n_post == 0:
        return RecordVerdict(snap_pos, k, replay_ok, n_pre, n_post, 0, False,
                             "empty token stream")
    if n_pre != n_post:
        return RecordVerdict(snap_pos, k, replay_ok, n_pre, n_post,
                             min(n_pre, n_post), False,
                             f"length mismatch: pre={n_pre} post={n_post}")
    if isinstance(k, int) and k > 0 and n_pre != k:
        # not fatal, but worth surfacing: the window was truncated
        pass

    first_div = -1
    for i in range(n_pre):
        if pre[i] != post[i]:
            first_div = i
            break

    if first_div < 0:
        return RecordVerdict(snap_pos, k, replay_ok, n_pre, n_post, -1, True,
                             "identical")
    reason = (f"first divergence at window offset {first_div} "
              f"(pos {(_as_int(snap_pos) or 0) + first_div}): "
              f"pre={pre[first_div]} != post={post[first_div]}")
    return RecordVerdict(snap_pos, k, replay_ok, n_pre, n_post, first_div,
                         False, reason)


def _as_int(v) -> Optional[int]:
    return v if isinstance(v, int) else None


def iter_records(text: str) -> Iterable[Tuple[int, dict]]:
    """Yield (line_no, obj) for each JSON object line; skip blanks/garbage."""
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line[0] != "{":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield line_no, obj


def collect_verdicts(text: str) -> List[RecordVerdict]:
    out: List[RecordVerdict] = []
    for _line_no, obj in iter_records(text):
        if obj.get("ev") == REWIND_EV:
            out.append(verdict_for(obj))
    return out


def run(texts: Sequence[str], out=sys.stdout) -> int:
    verdicts: List[RecordVerdict] = []
    for text in texts:
        verdicts.extend(collect_verdicts(text))

    if not verdicts:
        print("no rewind_test records found "
              "(is DS4_REWIND_TEST set and did the run reach p+k?)", file=out)
        return 2

    n_pass = 0
    for v in verdicts:
        print(v.summary(), file=out)
        if v.passed:
            n_pass += 1

    n_fail = len(verdicts) - n_pass
    print(f"\n{n_pass}/{len(verdicts)} rewind tests PASS, {n_fail} FAIL", file=out)
    return 0 if n_fail == 0 else 1


def _read_paths(paths: Sequence[str]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        if p == "-":
            texts.append(sys.stdin.read())
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                texts.append(fh.read())
    return texts


# --------------------------------------------------------------------------- #
# Built-in unit test on synthetic JSONL (no engine / GPU needed).
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    pass_line = json.dumps({
        "ev": "rewind_test", "snap_pos": 400, "k": 4, "replay_ok": 1,
        "first_div": -1, "pre": [10, 11, 12, 13], "post": [10, 11, 12, 13],
    })
    fail_div = json.dumps({
        "ev": "rewind_test", "snap_pos": 400, "k": 4, "replay_ok": 1,
        "first_div": 2, "pre": [10, 11, 12, 13], "post": [10, 11, 99, 13],
    })
    fail_replay = json.dumps({
        "ev": "rewind_test", "snap_pos": 400, "k": 4, "replay_ok": 0,
        "first_div": -1, "pre": [10, 11], "post": [10, 11],
    })
    fail_len = json.dumps({
        "ev": "rewind_test", "snap_pos": 5, "k": 3, "replay_ok": 1,
        "first_div": -1, "pre": [1, 2, 3], "post": [1, 2],
    })
    noise = 'not json\n{"ev":"s1_trigger","tok":401}\n\n'

    # single PASS record -> exit 0
    assert run([pass_line], out=_Null()) == 0

    # verdict-level checks
    vp = collect_verdicts(pass_line)
    assert len(vp) == 1 and vp[0].passed and vp[0].first_div == -1

    vd = collect_verdicts(fail_div)[0]
    assert not vd.passed and vd.first_div == 2, vd

    vr = collect_verdicts(fail_replay)[0]
    assert not vr.passed and "replay_ok=0" in vr.reason, vr

    vl = collect_verdicts(fail_len)[0]
    assert not vl.passed and "length mismatch" in vl.reason, vl

    # mixed stream (noise + PASS + FAIL) -> exit 1, noise ignored
    mixed = noise + pass_line + "\n" + fail_div
    verds = collect_verdicts(mixed)
    assert len(verds) == 2, verds
    assert run([mixed], out=_Null()) == 1

    # empty / no records -> exit 2
    assert run([noise], out=_Null()) == 2

    print("selftest OK")
    return 0


class _Null:
    def write(self, *_a, **_k):  # noqa: D401 - swallow test output
        return 0


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    if args[0] == "--selftest":
        return _selftest()
    try:
        texts = _read_paths(args)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return run(texts)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
