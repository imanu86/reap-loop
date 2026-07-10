"""Unit tests for scripts/freeze_boundary.find_safe_freeze_point (T4 prep).

Covers the pathological cut points documented in ledger note J44:
cutting mid ``font-family: ...``, mid value, inside a ``<style>``/``<meta>`` tag,
inside a quoted string, at an ``&entity;`` ``;``, at a JS ``a > b`` ``>``, and
inside a comment. Uses ``tokenizer_len_fn=len`` so the token budget is a *char*
budget and the cut position can be asserted exactly. CPU-only, no server/WSL.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_boundary.py"
SPEC = importlib.util.spec_from_file_location("freeze_boundary", SCRIPT)
assert SPEC and SPEC.loader
freeze_boundary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze_boundary
SPEC.loader.exec_module(freeze_boundary)

find = freeze_boundary.find_safe_freeze_point
SAFE_CHARS = set("};>")


def _ends_safely(fp) -> bool:
    """Frozen text ends on a boundary char (ignoring trailing whitespace)."""
    stripped = fp.frozen_text.rstrip()
    return stripped == "" or stripped[-1] in SAFE_CHARS


def test_frozen_is_always_a_prefix() -> None:
    text = "h1 { color: red; }\nbody { font-family: Arial; }\n"
    fp = find(text, 20, tokenizer_len_fn=len)
    assert text.startswith(fp.frozen_text)
    assert fp.cut_index == len(fp.frozen_text)


def test_cut_mid_font_family_backs_up_to_rule_close() -> None:
    # J44 canonical pathology: budget lands inside "font-family: Arial".
    text = "h1 { color: red; }\nbody { font-family: Arial, sans-serif; margin: 0; }\n"
    target = text.index("Arial") + 2  # mid the value
    fp = find(text, target, tokenizer_len_fn=len)
    assert "font-family" not in fp.frozen_text  # never mid declaration
    assert fp.boundary == "}"
    assert fp.frozen_text == "h1 { color: red; }\n"
    assert fp.within_target and _ends_safely(fp)


def test_cut_mid_value_backs_up() -> None:
    text = "a { text-decoration: none; }\n.x { color: #ff0000; background: blue; }\n"
    target = text.index("#ff0000") + 3  # mid the color value
    fp = find(text, target, tokenizer_len_fn=len)
    assert "#ff0000" not in fp.frozen_text
    assert fp.boundary in SAFE_CHARS and _ends_safely(fp)


def test_cut_inside_incomplete_tag_backs_up_to_prev_tag_close() -> None:
    # Real W50 phase-1 tail: it ends mid-'<style' (an open, unfinished tag).
    text = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '    <meta charset="UTF-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "    <title>Bean & Brew</title>\n"
        "    <style"
    )
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert "<style" not in fp.frozen_text  # dangling open tag dropped
    assert fp.boundary == ">"
    assert fp.frozen_text.rstrip().endswith("</title>")


def test_cut_mid_attribute_value_is_never_chosen() -> None:
    text = '<div class="hero" id="main">\n<meta content="width=devi'
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert "width=devi" not in fp.frozen_text  # never mid attribute/string
    assert fp.boundary == ">"
    assert fp.frozen_text == '<div class="hero" id="main">\n'


def test_semicolon_inside_string_is_not_a_boundary() -> None:
    text = 'div { content: "a;b;c"; }'
    # Budget lands right after the first in-string ';' — must NOT cut there.
    target = text.index(";") + 1
    fp = find(text, target, tokenizer_len_fn=len)
    assert '"a;b;c"' in fp.frozen_text  # the string was not severed
    assert fp.boundary == ";"           # cut at the real terminator instead
    assert fp.within_target is False    # real ';' is beyond the tiny budget


def test_gt_inside_javascript_is_not_a_boundary() -> None:
    text = "<script>\nif (a > b) { x(); }\n</script>"
    target = text.index("a > b") + 4  # just past the comparison '>'
    fp = find(text, target, tokenizer_len_fn=len)
    assert fp.frozen_text == "<script>\n"  # only the tag '>' counts
    assert "if (a" not in fp.frozen_text


def test_entity_semicolon_is_not_a_boundary() -> None:
    plain = "<p>a; b"
    fp_plain = find(plain, len(plain), tokenizer_len_fn=len)
    assert fp_plain.boundary == ";"  # a real statement/text ';'

    entity = "<p>Tom &amp; Jerry"
    fp_entity = find(entity, len(entity), tokenizer_len_fn=len)
    assert fp_entity.boundary == ">"          # only the <p> tag close
    assert "&amp;" not in fp_entity.frozen_text


def test_blank_line_boundary() -> None:
    text = "line one\n\nline two after blank"
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert fp.boundary == "blankline"
    assert fp.frozen_text == "line one\n"


def test_comment_internals_are_ignored() -> None:
    text = "a{color:red} /* b; c } d */ x"
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert fp.boundary == "}"
    assert "b;" not in fp.frozen_text   # the '}' and ';' inside /* */ don't count
    assert fp.frozen_text.startswith("a{color:red}")


def test_fully_safe_text_returned_whole() -> None:
    text = "a{}\nb{}\n"
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert fp.frozen_text == text
    assert fp.boundary == "}" and fp.within_target


def test_no_safe_boundary_returns_whole_text() -> None:
    text = "color: red without any terminator"
    fp = find(text, len(text), tokenizer_len_fn=len)
    assert fp.boundary == "none"
    assert fp.frozen_text == text


def test_fallback_when_nothing_fits_target() -> None:
    text = "aaaaaaaaaa{}\n"  # first safe boundary is well past a tiny budget
    fp = find(text, 3, tokenizer_len_fn=len)
    assert fp.within_target is False
    assert fp.boundary == "}"          # still a safe cut, just over budget
    assert _ends_safely(fp)


def test_default_estimator_runs_without_tokenizer() -> None:
    text = "h1 { color: red; }\n" * 10
    fp = find(text, 20)  # ~4 chars/token default -> ~80 chars
    assert text.startswith(fp.frozen_text)
    assert fp.boundary in SAFE_CHARS
    assert _ends_safely(fp)
