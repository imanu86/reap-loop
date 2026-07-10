"""Unit tests for the additive DS4 matrix-runner upgrades (n=3 / ABAB / grading).

These cover only the pure, torch-free, network-free helpers added to
``scripts/run_ds4_exchange_matrix.py``:

  * ``median`` — robust median over mixed numeric/empty values.
  * ``order_jobs`` — sequential (legacy) vs ABAB job interleaving.
  * ``alert_in_script`` — popup detection restricted to <script> blocks
    (the fix for the ``has_popup`` prompt-echo false positive).
  * ``compute_median_summary`` — per-(prompt, variant) medians + majority flags.
  * the exact ``html_coffee`` prompt recovery.

The module is loaded straight from its file path (``scripts`` is not a package),
so no server, WSL, or network is touched.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RUNNER_PATH = _ROOT / "scripts" / "run_ds4_exchange_matrix.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("ds4_exchange_matrix_runner", _RUNNER_PATH)
    assert spec and spec.loader, f"cannot load runner from {_RUNNER_PATH}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------- #
# median                                                                       #
# --------------------------------------------------------------------------- #
def test_median_odd_and_even():
    assert runner.median([3, 1, 2]) == 2
    assert runner.median([1, 2, 3, 4]) == 2.5


def test_median_skips_empty_and_none_and_coerces_strings():
    assert runner.median([None, "", "5", 5]) == 5
    assert runner.median([]) is None
    assert runner.median([None, ""]) is None


# --------------------------------------------------------------------------- #
# order_jobs                                                                   #
# --------------------------------------------------------------------------- #
def test_order_jobs_sequential_is_legacy_block_order():
    jobs = runner.order_jobs(["A", "B"], ["p"], 3, "sequential")
    assert jobs == [
        ("A", "p", 1),
        ("A", "p", 2),
        ("A", "p", 3),
        ("B", "p", 1),
        ("B", "p", 2),
        ("B", "p", 3),
    ]


def test_order_jobs_abab_interleaves_variants_across_runs():
    jobs = runner.order_jobs(["A", "B"], ["p"], 3, "abab")
    assert jobs == [
        ("A", "p", 1),
        ("B", "p", 1),
        ("A", "p", 2),
        ("B", "p", 2),
        ("A", "p", 3),
        ("B", "p", 3),
    ]


def test_order_jobs_default_single_run_matches_legacy_single_pass():
    # runs=1 must reproduce the pre-upgrade one-pass-per-variant behavior.
    seq = runner.order_jobs(["A", "B"], ["h", "c"], 1, "sequential")
    assert seq == [("A", "h", 1), ("A", "c", 1), ("B", "h", 1), ("B", "c", 1)]


# --------------------------------------------------------------------------- #
# alert_in_script (3 test strings)                                            #
# --------------------------------------------------------------------------- #
def test_alert_in_script_true_inside_script():
    content = (
        "<html><body><button id='o'>Buy</button>"
        "<script>document.getElementById('o')"
        ".addEventListener('click',()=>alert('done'))</script></body></html>"
    )
    assert runner.alert_in_script(content) == 1


def test_alert_in_script_false_when_only_in_prose_or_comment():
    # The classic has_popup false positive: 'popup'/'alert('/echo outside any
    # <script> block must NOT count as a real feature.
    content = (
        "<!-- popup: mostra alert(...) -->"
        "<p>Un popup con richiesta inviata</p>"
    )
    assert runner.alert_in_script(content) == 0
    # ...while the deprecated has_popup still fires on the echo.
    assert runner.quality_flags("html", content)["has_popup"] == 1


def test_alert_in_script_false_when_script_has_no_popup_api():
    content = "<html><script>console.log('ready')</script></html>"
    assert runner.alert_in_script(content) == 0


def test_alert_in_script_detects_confirm_and_showmodal():
    assert runner.alert_in_script("<script>confirm('sure?')</script>") == 1
    assert runner.alert_in_script("<script>dlg.showModal()</script>") == 1


def test_quality_flags_alert_in_script_divergence_from_has_popup():
    q = runner.quality_flags("html", "<p>alert( in prose only</p>")
    assert q["has_popup"] == 1
    assert q["alert_in_script"] == 0


# --------------------------------------------------------------------------- #
# compute_median_summary                                                       #
# --------------------------------------------------------------------------- #
def _row(**over):
    base = {
        "prompt": "html",
        "variant": "v",
        "variant_rationale": "r",
        "avg_tps": None,
        "first50_tps": None,
        "prompt_s": None,
        "l0l3": "",
        "doctype": 0,
        "has_popup": 0,
        "alert_in_script": 0,
        "repeat_flag": 0,
    }
    base.update(over)
    return base


def test_compute_median_summary_medians_and_majority_flags():
    rows = [
        _row(avg_tps=2.0, first50_tps=3.0, prompt_s=10.0, l0l3="2", repeat_flag=1, doctype=1),
        _row(avg_tps=3.0, first50_tps=4.0, prompt_s=20.0, l0l3="3", repeat_flag=0, doctype=1),
        _row(avg_tps=4.0, first50_tps=5.0, prompt_s=30.0, l0l3="2", repeat_flag=0, doctype=0),
    ]
    out = runner.compute_median_summary(rows)
    assert len(out) == 1
    rec = out[0]
    assert rec["run_count"] == 3
    assert rec["avg_tps_median"] == 3.0
    assert rec["first50_tps_median"] == 4.0
    assert rec["prompt_s_median"] == 20.0
    assert rec["l0l3_median"] == 2  # median of [2, 3, 2]
    # majority (>= half): repeat_flag 1/3 -> 0 ; doctype 2/3 -> 1
    assert rec["repeat_flag"] == 0
    assert rec["doctype"] == 1


def test_compute_median_summary_majority_is_at_least_half():
    rows = [_row(repeat_flag=1), _row(repeat_flag=1), _row(repeat_flag=0)]
    assert runner.compute_median_summary(rows)[0]["repeat_flag"] == 1  # 2/3
    rows2 = [_row(alert_in_script=1), _row(alert_in_script=0)]
    assert runner.compute_median_summary(rows2)[0]["alert_in_script"] == 1  # 1/2 == half


def test_compute_median_summary_groups_per_prompt_variant():
    rows = [
        _row(variant="a", avg_tps=2.0),
        _row(variant="b", avg_tps=4.0),
        _row(prompt="code", variant="a", avg_tps=6.0),
    ]
    out = runner.compute_median_summary(rows)
    keys = {(r["prompt"], r["variant"]) for r in out}
    assert keys == {("html", "a"), ("html", "b"), ("code", "a")}


# --------------------------------------------------------------------------- #
# prompt registry                                                              #
# --------------------------------------------------------------------------- #
def test_new_prompts_registered_without_touching_html():
    assert set(["html", "code", "code_mini", "html_coffee", "html_dashboard"]).issubset(
        set(runner.PROMPTS)
    )
    # existing html prompt is unchanged (cyberpunk landing page).
    assert "cyberpunk" in runner.PROMPTS["html"]


def test_html_coffee_prompt_matches_recovered_pod_file_exactly():
    pod_file = (
        _ROOT
        / "runs"
        / "ds4"
        / "20260710_pod_cache1024_warmup_replay"
        / "frontpage_prompt.txt"
    )
    if not pod_file.exists():
        pytest.skip(f"recovered prompt file absent: {pod_file}")
    expected = pod_file.read_text(encoding="utf-8")
    assert runner.PROMPTS["html_coffee"] == expected


def test_html_dashboard_prompt_is_italian_medium_html():
    p = runner.PROMPTS["html_dashboard"]
    assert "<canvas>" in p and "<script>" in p and "</html>" in p
    assert "filtro" in p and "tabella" in p


if __name__ == "__main__":
    # Inline runner so the pure checks pass without pytest installed.
    fns = [name for name in dir() if name.startswith("test_")]
    failures = 0
    g = dict(globals())
    for name in fns:
        fn = g[name]
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - manual harness.
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
