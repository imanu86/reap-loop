"""Safe freeze-point finder for offline two-phase session-learning (T4).

Background (ledger note J44 + CLAIMS row SESSION-LEARNING). The old cache1024
W-sweep looked like a monotone quality-vs-W table, but it was a *lottery of the
cut point*: the phase-2 prompt re-prefills ``[instruction] + [partial HTML]`` and
when W truncated the prefix *inside* a CSS declaration / an HTML attribute / a
string, the model emitted a fresh ``<!DOCTYPE html>`` (document-restart
attractor) instead of continuing. W=50/130 happened to fall right after ``}`` /
``</tag>`` (clean); W=80/110/150 fell mid-``font-family: ...`` (restart).

This module removes the lottery: given the phase-1 generated text and a target
token budget ``W``, it returns the *safe* cut point at or before ``W`` tokens,
defined as the position right after a structural boundary::

    }   end of a CSS rule / JS block
    ;   end of a CSS declaration / JS statement   (not an &entity; terminator)
    >   end of an HTML tag                          (never mid-attribute)
    \\n  the newline that closes a line before a blank line

and NEVER inside a CSS declaration, an HTML tag/attribute, a quoted string, or a
comment (CSS ``/* */``, HTML ``<!-- -->``, JS ``//``).

Public API::

    find_safe_freeze_point(text, target_tok, tokenizer_len_fn=None) -> FreezePoint

``tokenizer_len_fn`` maps a string prefix to its token length; pass the real
tokenizer for exactness. When omitted, a ``len/4`` estimate is used (adequate
because the harness generates phase-1 with ``-n W`` so the text is already ~W
tokens and we only need the last safe boundary at or before the end).

Pure module, no I/O, no deps — importable and unit-tested on strings.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FreezePoint:
    """Result of :func:`find_safe_freeze_point`.

    Attributes:
        cut_index:      char length of the frozen prefix (``text[:cut_index]``).
        frozen_text:    the phase-1 text trimmed to the safe boundary.
        n_tokens:       token length of ``frozen_text`` per the tokenizer fn.
        boundary:       kind of boundary chosen: ``}`` ``;`` ``>`` ``blankline``
                        or ``none`` (no safe boundary found; text returned whole).
        within_target:  True if the chosen cut is at or below ``target_tok``.
    """

    cut_index: int
    frozen_text: str
    n_tokens: int
    boundary: str
    within_target: bool


def _default_token_len(s: str) -> int:
    """Rough GPT-ish estimate (~4 chars/token). Monotone in ``len(s)``."""
    return max(0, round(len(s) / 4))


def _extend(text: str, k: int) -> int:
    """Extend a cut end through trailing horizontal whitespace + one newline.

    So the frozen prefix ends cleanly on a line break, which is what the phase-2
    re-prefill wants (no dangling ``    `` indentation on the last line).
    """
    n = len(text)
    j = k
    while j < n and text[j] in " \t":
        j += 1
    if j < n and text[j] == "\n":
        j += 1
    return j


def _is_entity_semicolon(text: str, i: int) -> bool:
    """True if ``text[i] == ';'`` terminates an HTML entity like ``&amp;``.

    Such a ``;`` is inside display text, not a CSS/JS statement terminator, so it
    is not a structural boundary.
    """
    j = i - 1
    while j >= 0 and (text[j].isalnum() or text[j] == "#"):
        j -= 1
    return j >= 0 and j < i - 1 and text[j] == "&"


def _safe_boundaries(text: str) -> list[tuple[int, str]]:
    """Single left-to-right scan collecting ``(cut_end, kind)`` safe boundaries.

    Tracks quoted strings, HTML tags, and comments so that ``}`` ``;`` ``>`` are
    only accepted when they are real structural terminators.
    """
    ends: list[tuple[int, str]] = []
    in_str: str | None = None       # active quote char, or None
    escaped = False
    in_tag = False                  # inside <...>
    comment: str | None = None      # None | 'css' | 'html' | 'line'
    i = 0
    n = len(text)
    while i < n:
        c = text[i]

        # --- inside a comment: only look for its terminator ---
        if comment == "css":
            if text[i:i + 2] == "*/":
                i += 2
                comment = None
                continue
            i += 1
            continue
        if comment == "html":
            if text[i:i + 3] == "-->":
                i += 3
                comment = None
                continue
            i += 1
            continue
        if comment == "line":
            if c == "\n":
                comment = None
            i += 1
            continue

        # --- inside a quoted string: only look for the matching quote ---
        if in_str is not None:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == in_str:
                in_str = None
            i += 1
            continue

        # --- comment openers (only outside strings) ---
        if text[i:i + 4] == "<!--":
            comment = "html"
            in_tag = False
            i += 4
            continue
        if text[i:i + 2] == "/*":
            comment = "css"
            i += 2
            continue
        if text[i:i + 2] == "//" and not in_tag:
            comment = "line"
            i += 2
            continue

        # --- string openers ---
        if c in ("\"", "'", "`"):
            in_str = c
            i += 1
            continue

        # --- tag start: '<' followed by a name char, '/', or '!' ---
        if c == "<" and i + 1 < n and (text[i + 1].isalpha() or text[i + 1] in "/!"):
            in_tag = True
            i += 1
            continue

        # --- boundary characters ---
        if c == ">" and in_tag:
            in_tag = False
            ends.append((_extend(text, i + 1), ">"))
            i += 1
            continue
        if c == ";" and not in_tag:
            if not _is_entity_semicolon(text, i):
                ends.append((_extend(text, i + 1), ";"))
            i += 1
            continue
        if c == "}" and not in_tag:
            ends.append((_extend(text, i + 1), "}"))
            i += 1
            continue

        # --- blank-line boundary: this newline is followed by an empty line ---
        if c == "\n" and not in_tag:
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "\n":
                ends.append((i + 1, "blankline"))

        i += 1

    # de-dup by cut end, keep the first (most specific) kind, sort ascending
    seen: dict[int, str] = {}
    for end, kind in ends:
        seen.setdefault(end, kind)
    return sorted(seen.items())


def find_safe_freeze_point(text, target_tok, tokenizer_len_fn=None) -> FreezePoint:
    """Return the safe freeze point at or before ``target_tok`` tokens.

    Args:
        text:              phase-1 generated text (model output, no prompt).
        target_tok:        target token budget W.
        tokenizer_len_fn:  ``str -> int`` token counter; default ``len/4``.

    Returns:
        A :class:`FreezePoint`. If no safe boundary fits within ``target_tok``,
        falls back to the *smallest* safe boundary (``within_target=False``) so
        the cut is still structurally safe. If the text has no safe boundary at
        all, returns the whole text with ``boundary='none'``.
    """
    len_fn = tokenizer_len_fn or _default_token_len
    cands = _safe_boundaries(text)
    if not cands:
        return FreezePoint(len(text), text, len_fn(text), "none",
                           len_fn(text) <= target_tok)

    best: tuple[int, str] | None = None
    for end, kind in cands:
        if len_fn(text[:end]) <= target_tok:
            best = (end, kind)  # keep the largest fitting boundary
    if best is not None:
        end, kind = best
        return FreezePoint(end, text[:end], len_fn(text[:end]), kind, True)

    # nothing fits within target: degrade to the smallest safe boundary
    end, kind = cands[0]
    return FreezePoint(end, text[:end], len_fn(text[:end]), kind, False)
