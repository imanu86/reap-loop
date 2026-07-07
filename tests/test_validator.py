"""test_validator.py — test CPU-only, isolato e deterministico del modulo validator.

Copre (vedi docs/00_architecture.md §8 e il task):
  - PythonUnitTestValidator: funzione corretta -> True, buggy -> False, loop infinito -> False
    entro timeout e senza side-effect; estrazione codice da output recintato/nudo.
  - ContextFiller: riproducibile (stesso seed -> stesso prompt e stessa lunghezza), needle inserito.
  - ClosedFormValidator / NeedleInHaystackValidator: verify e curva un-punto-per-ctx.
  - evaluate_at_lengths: ritorna UN ContextLengthResult per ciascuna ctx.

Tutto con FAKE generate_fn e FAKE tokenizer (split su spazi). Niente torch, niente rete.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from msc.validator import (
    ClosedFormValidator,
    ContextFiller,
    ContextLengthResult,
    FillStrategy,
    NeedleInHaystackValidator,
    PythonUnitTestValidator,
)


# --------------------------------------------------------------------------- #
# Fake tokenizer: conteggio token = numero di "parole" separate da spazi.      #
# --------------------------------------------------------------------------- #
class WhitespaceTokenizer:
    """Tokenizer fittizio: 1 token = 1 parola separata da whitespace. Deterministico, CPU-only."""

    model_max_length = 100_000  # context window finta, per testare la troncatura

    def encode(self, text: str) -> list[int]:
        # mappa ogni parola a un id arbitrario ma stabile (len soltanto conta per il filler)
        toks = text.split()
        return [hash(t) & 0xFFFF for t in toks]

    def decode(self, ids) -> str:
        # non usato dal filler corrente, ma parte del contratto del tokenizer
        return " ".join(str(i) for i in ids)


# --------------------------------------------------------------------------- #
# Dataset jsonl di esempio (scritto in un file temporaneo per ogni test).      #
# --------------------------------------------------------------------------- #
_TESTS_ADD = (
    "assert add(2, 3) == 5\n"
    "assert add(-1, 1) == 0\n"
    "assert add(0, 0) == 0\n"
)


def _write_dataset(tmp_path) -> str:
    rows = [
        {"item_id": "p0", "prompt": "Scrivi add(a, b) che ritorna la somma.", "tests": _TESTS_ADD},
    ]
    path = os.path.join(str(tmp_path), "ds.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


# --------------------------------------------------------------------------- #
# PythonUnitTestValidator                                                       #
# --------------------------------------------------------------------------- #
def test_items_loaded_from_jsonl(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    items = v.items()
    assert len(items) == 1
    assert items[0].item_id == "p0"
    assert "tests" in items[0].payload


def test_correct_function_passes(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert v.verify(item, good) is True


def test_buggy_function_fails(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    buggy = "```python\ndef add(a, b):\n    return a - b\n```"
    assert v.verify(item, buggy) is False


def test_bare_code_without_fence(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    bare = "def add(a, b):\n    return a + b\n"
    assert v.verify(item, bare) is True


def test_empty_output_fails(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    assert v.verify(item, "") is False


def test_exception_in_code_fails(tmp_path):
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    boom = "```python\ndef add(a, b):\n    raise ValueError('boom')\n```"
    assert v.verify(item, boom) is False


def test_determinism_same_verdict(tmp_path):
    """GATE: stesso (item, output) -> stesso verdetto, ripetuto N volte."""
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    item = v.items()[0]
    good = "def add(a, b):\n    return a + b\n"
    verdicts = {v.verify(item, good) for _ in range(8)}
    assert verdicts == {True}


def test_infinite_loop_times_out_without_sideeffects(tmp_path):
    """Loop infinito -> False entro il timeout, senza creare file nel cwd del test."""
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path), exec_timeout_s=2.0)
    item = v.items()[0]
    looping = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    cwd_before = set(os.listdir("."))
    start = time.monotonic()
    result = v.verify(item, looping)
    elapsed = time.monotonic() - start
    assert result is False
    # ucciso entro un margine ragionevole del timeout (no hang indefinito)
    assert elapsed < 15.0
    # nessun side-effect sul cwd del processo di test
    assert set(os.listdir(".")) == cwd_before


def test_evaluate_at_lengths_one_point_per_ctx(tmp_path):
    """evaluate_at_lengths ritorna UN ContextLengthResult per ciascuna ctx richiesta."""
    v = PythonUnitTestValidator(dataset_path=_write_dataset(tmp_path))
    tok = WhitespaceTokenizer()
    filler = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=0)

    def fake_generate(prompt: str) -> str:
        # generate_fn fittizio: ritorna sempre la soluzione corretta (indip. dal contesto)
        return "def add(a, b):\n    return a + b\n"

    ctx_lengths = [50, 200, 800]
    curve = v.evaluate_at_lengths(generate_fn=fake_generate, ctx_lengths=ctx_lengths, filler=filler)
    assert len(curve) == len(ctx_lengths)
    assert [r.ctx_len for r in curve] == ctx_lengths
    assert all(isinstance(r, ContextLengthResult) for r in curve)
    assert all(r.accuracy == 1.0 for r in curve)


# --------------------------------------------------------------------------- #
# ContextFiller                                                                 #
# --------------------------------------------------------------------------- #
def test_filler_reaches_target_length():
    tok = WhitespaceTokenizer()
    filler = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=7)
    prompt, eff = filler.fill("domanda base qui", target_tokens=300)
    # lunghezza effettiva vicina al target (il filler non deve superarlo)
    assert eff <= 300
    assert eff >= 250  # ragionevolmente vicino
    assert len(tok.encode(prompt)) == eff


def test_filler_reproducible_same_seed():
    tok = WhitespaceTokenizer()
    f1 = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=42)
    f2 = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=42)
    p1, l1 = f1.fill("task fisso", target_tokens=200)
    p2, l2 = f2.fill("task fisso", target_tokens=200)
    assert p1 == p2
    assert l1 == l2


def test_filler_different_seed_differs():
    tok = WhitespaceTokenizer()
    f1 = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=1)
    f2 = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=2)
    p1, _ = f1.fill("task fisso", target_tokens=200)
    p2, _ = f2.fill("task fisso", target_tokens=200)
    assert p1 != p2


def test_filler_inserts_needle():
    tok = WhitespaceTokenizer()
    filler = ContextFiller(tok, FillStrategy.RELEVANT_HAYSTACK, seed=3)
    needle = "il codice e ABC123"
    prompt, _ = filler.fill("dov'e il codice?", target_tokens=150, needle=needle, needle_depth=0.5)
    assert needle in prompt


def test_filler_respects_max_ctx():
    tok = WhitespaceTokenizer()
    tok.model_max_length = 60
    filler = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=0)
    _, eff = filler.fill("base", target_tokens=10_000)
    assert eff <= 60


# --------------------------------------------------------------------------- #
# ClosedFormValidator                                                           #
# --------------------------------------------------------------------------- #
def test_closed_form_verify_exact_and_embedded():
    v = ClosedFormValidator()
    items = {it.item_id: it for it in v.items()}
    add_item = items["cf-00"]  # risposta attesa "4"
    assert v.verify(add_item, "4") is True
    assert v.verify(add_item, "La risposta e' 4.") is True
    assert v.verify(add_item, "42") is False  # non deve matchare come sottostringa


def test_closed_form_evaluate_one_point_per_ctx():
    v = ClosedFormValidator()
    tok = WhitespaceTokenizer()
    filler = ContextFiller(tok, FillStrategy.PADDING_DISTRACTORS, seed=0)

    # generate_fn fittizio: risponde correttamente solo a cf-00 (contiene "4") per testare aggregazione
    answers = {row["prompt"]: row["answer"] for row in ClosedFormValidator._DATASET}

    def fake_generate(prompt: str) -> str:
        # il prompt riempito inizia col prompt base: trova quale domanda è
        for q, a in answers.items():
            if prompt.startswith(q):
                return a
        return "non lo so"

    ctx_lengths = [40, 120]
    curve = v.evaluate_at_lengths(generate_fn=fake_generate, ctx_lengths=ctx_lengths, filler=filler)
    assert len(curve) == 2
    assert all(r.accuracy == 1.0 for r in curve)


# --------------------------------------------------------------------------- #
# NeedleInHaystackValidator                                                     #
# --------------------------------------------------------------------------- #
def test_needle_items_deterministic():
    v1 = NeedleInHaystackValidator(needle_depths=(0.1, 0.5, 0.9))
    v2 = NeedleInHaystackValidator(needle_depths=(0.1, 0.5, 0.9))
    items1 = v1.items()
    items2 = v2.items()
    assert [i.item_id for i in items1] == [i.item_id for i in items2]
    assert [i.payload["answer"] for i in items1] == [i.payload["answer"] for i in items2]
    # un item per (base, profondità)
    assert len(items1) == len(NeedleInHaystackValidator._MAGIC_CODES) * 3


def test_needle_verify_recovers_value():
    v = NeedleInHaystackValidator()
    item = v.items()[0]
    value = item.payload["answer"]
    assert v.verify(item, f"Il codice e' {value}") is True
    assert v.verify(item, "Non l'ho trovato") is False


def test_needle_evaluate_one_point_per_ctx_and_recovery():
    v = NeedleInHaystackValidator(needle_depths=(0.2, 0.8))
    tok = WhitespaceTokenizer()
    filler = ContextFiller(tok, FillStrategy.RELEVANT_HAYSTACK, seed=11)

    def perfect_reader(prompt: str) -> str:
        # legge il needle dal contesto (oracolo): estrae il codice dopo "segreto e' "
        marker = "segreto e' "
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            return tail.split(".", 1)[0]
        return ""

    ctx_lengths = [60, 300]
    curve = v.evaluate_at_lengths(generate_fn=perfect_reader, ctx_lengths=ctx_lengths, filler=filler)
    assert len(curve) == 2
    # un lettore perfetto recupera sempre -> accuratezza 1.0 a ogni ctx
    assert all(r.accuracy == 1.0 for r in curve)

    def blind_reader(prompt: str) -> str:
        return "boh"

    curve_blind = v.evaluate_at_lengths(generate_fn=blind_reader, ctx_lengths=ctx_lengths, filler=filler)
    assert all(r.accuracy == 0.0 for r in curve_blind)
