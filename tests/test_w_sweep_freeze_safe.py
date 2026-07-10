"""Unit tests for the pure helpers of scripts/run_w_sweep_freeze_safe.py (T4).

String/logic only: diag t/s parsing, output checks, command construction, and
the monotonicity verdict. No binary, no server, no WSL, no GPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_w_sweep_freeze_safe.py"
SPEC = importlib.util.spec_from_file_location("run_w_sweep_freeze_safe", SCRIPT)
assert SPEC and SPEC.loader
hs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hs
SPEC.loader.exec_module(hs)


def test_parse_tps_reads_last_line() -> None:
    diag = (
        "ds4: gpu prefill layer 43/43\n"
        "ds4: prefill: 5.44 t/s, generation: 2.03 t/s\n"
        "ds4: prefill: 7.11 t/s, generation: 14.60 t/s\n"
    )
    got = hs.parse_tps(diag)
    assert got == {"prefill": 7.11, "generation": 14.60}


def test_parse_tps_missing() -> None:
    assert hs.parse_tps("no timing here") == {"prefill": None, "generation": None}


def test_output_checks_counts() -> None:
    html = (
        "<!DOCTYPE html><html><head><style>body{}</style></head>"
        "<body><form action='/x'></form>"
        "<script>document.querySelector('button').addEventListener('click',"
        "()=>alert('hi'))</script></body></html>"
    )
    c = hs.output_checks(html)
    assert c["doctype"] == 1
    assert c["html_close"] == 1
    assert c["form"] == 1
    assert c["script"] == 1
    assert c["alert_in_script"] == 1
    assert c["repeat"] == 0


def test_output_checks_detects_repeat() -> None:
    loop = "document.addEventListener(\"DOM\");" * 5
    assert hs.output_checks(loop)["repeat"] == 1


def _fake_args(**kw):
    base = dict(binary="ds4", model="m.gguf", cache=1024, ctx_p1=2048, ctx_p2=3072,
                temp=0.0, headroom=16, total=1000, seed_base=0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_phase1_cmd_has_trace_and_headroom() -> None:
    args = _fake_args()
    env, cmd = hs.phase1_cmd(args, 50, "prompt.txt", "route.csv", seed=0)
    assert env["DS4_SPEX_TRACE_ROUTING"] == "route.csv"
    assert env["DS4_SPEX_TRACE_ROUTING_WEIGHTS"] == "1"
    # -n is W + headroom in phase 1
    assert cmd[cmd.index("-n") + 1] == "66"
    assert "--seed" not in cmd  # temp 0 -> deterministic, no seed flag
    assert cmd[cmd.index("--ssd-streaming-cache-experts") + 1] == "1024"


def test_phase2_cmd_masks_and_budgets_rest() -> None:
    args = _fake_args()
    env, cmd = hs.phase2_cmd(args, 50, "p2.txt", "sess.txt", seed=0)
    assert env["DS4_REAP_MASK_FILE"] == "sess.txt"
    assert cmd[cmd.index("-n") + 1] == "950"  # total - W
    assert cmd[cmd.index("-c") + 1] == "3072"


def test_seed_passed_only_when_sampling() -> None:
    args = _fake_args(temp=0.7)
    _, cmd = hs.phase1_cmd(args, 30, "p.txt", "r.csv", seed=2)
    assert cmd[cmd.index("--seed") + 1] == "2"


def test_verdict_monotone() -> None:
    rising = hs.verdict_monotone({30: 0, 50: 1, 70: 2, 90: 3})
    assert rising["monotone_non_decreasing"] is True
    assert rising["level_spread"] == 3

    lottery = hs.verdict_monotone({30: 3, 50: 1, 70: 3, 90: 0})
    assert lottery["monotone_non_decreasing"] is False


def test_default_w_values() -> None:
    assert hs.default_w_values() == [30, 50, 70, 90, 110, 130, 150]


def test_w_run_dir_layout() -> None:
    d = hs.w_run_dir("/out", 50, 2)
    assert d.parts[-2:] == ("W050", "r02")
