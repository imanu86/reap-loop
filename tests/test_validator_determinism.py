"""test_validator_determinism.py — GATE del repo: il validatore DEVE essere deterministico.

Senza determinismo, il "drop < 2%" è rumore (rischio R5). Questi test esercitano le classi REALI
(PythonUnitTestValidator + ContextFiller) con un generate_fn e un tokenizer fittizi e deterministici,
senza GPU/torch.
"""

from __future__ import annotations

import json
import os

import pytest

from msc.validator.context_filler import ContextFiller, FillStrategy
from msc.validator.python_unit_tests import PythonUnitTestValidator


# --- fixtures / fake deterministici --------------------------------------------------------------

class WhitespaceTok:
    """Tokenizer fittizio: 1 token = 1 parola separata da spazio. Niente model_max_length."""

    def encode(self, text: str):
        return text.split()

    def decode(self, ids) -> str:
        return " ".join(ids)


_CORRECT_ADD = "```python\ndef add(a, b):\n    return a + b\n```"
_BUGGY_ADD = "```python\ndef add(a, b):\n    return a - b\n```"


@pytest.fixture
def dataset(tmp_path):
    """Mini-dataset jsonl: due problemi con test nascosti."""
    rows = [
        {"item_id": "add", "prompt": "Write add(a,b).", "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"},
        {"item_id": "mul", "prompt": "Write mul(a,b).", "tests": "assert mul(2, 3) == 6"},
    ]
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


# --- 1. stesso input -> stesso verdetto ----------------------------------------------------------

def test_same_input_same_verdict(dataset):
    val = PythonUnitTestValidator(dataset)
    add_item = val.items()[0]

    good = {val.verify(add_item, _CORRECT_ADD) for _ in range(20)}
    bad = {val.verify(add_item, _BUGGY_ADD) for _ in range(20)}

    assert good == {True}   # 20 run, sempre True
    assert bad == {False}   # 20 run, sempre False


# --- 2. sandbox: isolamento, timeout, nessun side-effect -----------------------------------------

def test_sandbox_timeout_returns_false_fast(dataset):
    val = PythonUnitTestValidator(dataset, exec_timeout_s=2.0)
    add_item = val.items()[0]
    loop = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    import time
    t0 = time.monotonic()
    verdict = val.verify(add_item, loop)
    elapsed = time.monotonic() - t0
    assert verdict is False
    assert elapsed < 15.0   # ucciso al timeout, non appeso


def test_sandbox_exception_is_false(dataset):
    val = PythonUnitTestValidator(dataset)
    add_item = val.items()[0]
    boom = "```python\ndef add(a, b):\n    raise ValueError('boom')\n```"
    assert val.verify(add_item, boom) is False


def test_sandbox_side_effects_are_confined(dataset, tmp_path, monkeypatch):
    """Il codice generato gira in una cwd temporanea: nessun file finisce nella cwd del test."""
    monkeypatch.chdir(tmp_path)
    val = PythonUnitTestValidator(dataset)
    add_item = val.items()[0]
    writes = (
        "```python\n"
        "def add(a, b):\n"
        "    with open('SIDE_EFFECT.txt', 'w') as f:\n"
        "        f.write('x')\n"
        "    return a + b\n"
        "```"
    )
    # add funziona -> True, ma il file scritto resta confinato nel workdir usa-e-getta del sandbox.
    assert val.verify(add_item, writes) is True
    assert not (tmp_path / "SIDE_EFFECT.txt").exists()


def test_empty_output_is_false(dataset):
    val = PythonUnitTestValidator(dataset)
    add_item = val.items()[0]
    assert val.verify(add_item, "") is False
    assert val.verify(add_item, "nessun codice qui") is False


# --- 3. context filler riproducibile -------------------------------------------------------------

def test_context_filler_reproducible_same_seed():
    f = ContextFiller(WhitespaceTok(), FillStrategy.PADDING_DISTRACTORS, seed=42)
    p1, n1 = f.fill("base prompt", 50)
    p2, n2 = f.fill("base prompt", 50)
    assert p1 == p2 and n1 == n2


def test_context_filler_needle_deterministic_and_present():
    f = ContextFiller(WhitespaceTok(), FillStrategy.RELEVANT_HAYSTACK, seed=7)
    a = f.fill("Q?", 40, needle="SECRET-1234", needle_depth=0.5)
    b = f.fill("Q?", 40, needle="SECRET-1234", needle_depth=0.5)
    assert a == b
    assert "SECRET-1234" in a[0]


def test_context_filler_different_seed_differs():
    d1 = ContextFiller(WhitespaceTok(), FillStrategy.PADDING_DISTRACTORS, seed=1).fill("x", 60)[0]
    d2 = ContextFiller(WhitespaceTok(), FillStrategy.PADDING_DISTRACTORS, seed=2).fill("x", 60)[0]
    assert d1 != d2   # 60 token di rumore da seed diversi: collisione praticamente impossibile


# --- 4. evaluate_at_lengths -> una curva (un punto per lunghezza) ---------------------------------

def test_evaluate_returns_one_point_per_length(dataset):
    val = PythonUnitTestValidator(dataset)
    filler = ContextFiller(WhitespaceTok(), FillStrategy.PADDING_DISTRACTORS, seed=0)

    def gen(_prompt):  # generate_fn fittizio: risolve sempre solo 'add'
        return _CORRECT_ADD

    res = val.evaluate_at_lengths(generate_fn=gen, ctx_lengths=[20, 40, 60], filler=filler)

    assert [r.ctx_len for r in res] == [20, 40, 60]   # un punto per ctx, in ordine
    assert all(r.n_items == 2 for r in res)
    assert all(r.n_correct == 1 for r in res)          # solo 'add' passa, 'mul' no: deterministico


def test_evaluate_is_repeatable(dataset):
    val = PythonUnitTestValidator(dataset)
    filler = ContextFiller(WhitespaceTok(), FillStrategy.PADDING_DISTRACTORS, seed=0)

    def gen(_prompt):
        return _CORRECT_ADD

    r1 = val.evaluate_at_lengths(generate_fn=gen, ctx_lengths=[30], filler=filler)
    r2 = val.evaluate_at_lengths(generate_fn=gen, ctx_lengths=[30], filler=filler)
    assert (r1[0].ctx_len, r1[0].n_items, r1[0].n_correct) == (r2[0].ctx_len, r2[0].n_items, r2[0].n_correct)
