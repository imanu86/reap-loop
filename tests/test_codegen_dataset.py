"""test_codegen_dataset.py — verifica CPU-only del dataset data/codegen_problems.jsonl.

Obiettivo (vedi task):
  - per OGNI problema esiste una soluzione di RIFERIMENTO corretta e una SBAGLIATA;
  - caricando il dataset con PythonUnitTestValidator e racchiudendo la soluzione in un blocco
    ```python ...```, verify(item, correct) deve essere True e verify(item, wrong) False per TUTTI.

Tutto deterministico, niente torch/GPU/rete: il validatore esegue il codice in un subprocess
isolato (`-I -S`), quindi soluzioni e test usano SOLO builtin/stdlib senza import di terze parti.
"""

from __future__ import annotations

import os

import pytest

from msc.validator import PythonUnitTestValidator

# Percorso del dataset reale prodotto dal task (non un file temporaneo).
_DATASET_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "codegen_problems.jsonl")
)


# --------------------------------------------------------------------------- #
# Soluzioni di riferimento: per ogni item_id una versione CORRETTA e una SBAGLIATA.
# Le stringhe sono il corpo "nudo"; vengono racchiuse in un blocco recintato dal test.
# --------------------------------------------------------------------------- #

_CORRECT: dict[str, str] = {
    "cg-00-reverse-words": (
        "def reverse_words(s):\n"
        "    return ' '.join(reversed(s.split()))\n"
    ),
    "cg-01-is-palindrome": (
        "def is_palindrome(s):\n"
        "    cleaned = [c.lower() for c in s if c.isalnum()]\n"
        "    return cleaned == cleaned[::-1]\n"
    ),
    "cg-02-two-sum": (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for j, x in enumerate(nums):\n"
        "        need = target - x\n"
        "        if need in seen:\n"
        "            return (seen[need], j)\n"
        "        if x not in seen:\n"
        "            seen[x] = j\n"
        "    return None\n"
    ),
    "cg-03-flatten": (
        "def flatten(nested):\n"
        "    out = []\n"
        "    for el in nested:\n"
        "        if isinstance(el, list):\n"
        "            out.extend(flatten(el))\n"
        "        else:\n"
        "            out.append(el)\n"
        "    return out\n"
    ),
    "cg-04-most-frequent": (
        "def most_frequent(items):\n"
        "    if not items:\n"
        "        return None\n"
        "    counts = {}\n"
        "    best = None\n"
        "    best_count = 0\n"
        "    for x in items:\n"
        "        key = (type(x).__name__, x)\n"
        "        counts[key] = counts.get(key, 0) + 1\n"
        "        if counts[key] > best_count:\n"
        "            best_count = counts[key]\n"
        "            best = x\n"
        "    return best\n"
    ),
    "cg-05-run-length-encode": (
        "def rle_encode(s):\n"
        "    if not s:\n"
        "        return ''\n"
        "    out = []\n"
        "    prev = s[0]\n"
        "    count = 1\n"
        "    for c in s[1:]:\n"
        "        if c == prev:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append(prev + str(count))\n"
        "            prev = c\n"
        "            count = 1\n"
        "    out.append(prev + str(count))\n"
        "    return ''.join(out)\n"
    ),
    "cg-06-gcd-list": (
        "def gcd_list(nums):\n"
        "    def g(a, b):\n"
        "        while b:\n"
        "            a, b = b, a % b\n"
        "        return a\n"
        "    result = 0\n"
        "    for x in nums:\n"
        "        result = g(result, x)\n"
        "    return result\n"
    ),
    "cg-07-balanced-brackets": (
        "def is_balanced(s):\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    opening = set(pairs.values())\n"
        "    stack = []\n"
        "    for c in s:\n"
        "        if c in opening:\n"
        "            stack.append(c)\n"
        "        elif c in pairs:\n"
        "            if not stack or stack.pop() != pairs[c]:\n"
        "                return False\n"
        "    return not stack\n"
    ),
    "cg-08-merge-intervals": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals)\n"
        "    merged = [ordered[0]]\n"
        "    for start, end in ordered[1:]:\n"
        "        last_start, last_end = merged[-1]\n"
        "        if start <= last_end:\n"
        "            merged[-1] = (last_start, max(last_end, end))\n"
        "        else:\n"
        "            merged.append((start, end))\n"
        "    return merged\n"
    ),
    "cg-09-int-to-roman": (
        "def int_to_roman(n):\n"
        "    table = [\n"
        "        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),\n"
        "        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),\n"
        "        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),\n"
        "    ]\n"
        "    out = []\n"
        "    for value, sym in table:\n"
        "        while n >= value:\n"
        "            out.append(sym)\n"
        "            n -= value\n"
        "    return ''.join(out)\n"
    ),
}

_WRONG: dict[str, str] = {
    # Dimentica di invertire l'ordine delle parole.
    "cg-00-reverse-words": (
        "def reverse_words(s):\n"
        "    return ' '.join(s.split())\n"
    ),
    # Non ignora maiuscole/minuscole.
    "cg-01-is-palindrome": (
        "def is_palindrome(s):\n"
        "    cleaned = [c for c in s if c.isalnum()]\n"
        "    return cleaned == cleaned[::-1]\n"
    ),
    # Permette di riusare lo stesso elemento (i == j) e non gestisce i duplicati.
    "cg-02-two-sum": (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(len(nums)):\n"
        "            if nums[i] + nums[j] == target:\n"
        "                return (i, j)\n"
        "    return None\n"
    ),
    # Appiattisce solo un livello.
    "cg-03-flatten": (
        "def flatten(nested):\n"
        "    out = []\n"
        "    for el in nested:\n"
        "        if isinstance(el, list):\n"
        "            out.extend(el)\n"
        "        else:\n"
        "            out.append(el)\n"
        "    return out\n"
    ),
    # In caso di parita sceglie l'ultimo che raggiunge il massimo, non il primo.
    "cg-04-most-frequent": (
        "def most_frequent(items):\n"
        "    if not items:\n"
        "        return None\n"
        "    counts = {}\n"
        "    best = None\n"
        "    best_count = -1\n"
        "    for x in items:\n"
        "        key = (type(x).__name__, x)\n"
        "        counts[key] = counts.get(key, 0) + 1\n"
        "        if counts[key] >= best_count:\n"
        "            best_count = counts[key]\n"
        "            best = x\n"
        "    return best\n"
    ),
    # Omette il numero quando il conteggio e' 1.
    "cg-05-run-length-encode": (
        "def rle_encode(s):\n"
        "    if not s:\n"
        "        return ''\n"
        "    out = []\n"
        "    prev = s[0]\n"
        "    count = 1\n"
        "    for c in s[1:]:\n"
        "        if c == prev:\n"
        "            count += 1\n"
        "        else:\n"
        "            out.append(prev + (str(count) if count > 1 else ''))\n"
        "            prev = c\n"
        "            count = 1\n"
        "    out.append(prev + (str(count) if count > 1 else ''))\n"
        "    return ''.join(out)\n"
    ),
    # gcd(0, 0) gestito male: ritorna 1 invece di 0 quando tutti sono 0.
    "cg-06-gcd-list": (
        "def gcd_list(nums):\n"
        "    def g(a, b):\n"
        "        while b:\n"
        "            a, b = b, a % b\n"
        "        return a\n"
        "    result = 1\n"
        "    for x in nums:\n"
        "        result = g(result, x)\n"
        "    return result\n"
    ),
    # Non verifica il tipo di parentesi alla chiusura: '([)]' passa erroneamente.
    "cg-07-balanced-brackets": (
        "def is_balanced(s):\n"
        "    opening = set('([{')\n"
        "    closing = set(')]}')\n"
        "    depth = 0\n"
        "    for c in s:\n"
        "        if c in opening:\n"
        "            depth += 1\n"
        "        elif c in closing:\n"
        "            depth -= 1\n"
        "            if depth < 0:\n"
        "                return False\n"
        "    return depth == 0\n"
    ),
    # Fonde solo in caso di sovrapposizione STRETTA (<), non quando si toccano (==).
    "cg-08-merge-intervals": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals)\n"
        "    merged = [ordered[0]]\n"
        "    for start, end in ordered[1:]:\n"
        "        last_start, last_end = merged[-1]\n"
        "        if start < last_end:\n"
        "            merged[-1] = (last_start, max(last_end, end))\n"
        "        else:\n"
        "            merged.append((start, end))\n"
        "    return merged\n"
    ),
    # Niente notazione sottrattiva: 4 -> 'IIII' invece di 'IV'.
    "cg-09-int-to-roman": (
        "def int_to_roman(n):\n"
        "    table = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'),\n"
        "             (10, 'X'), (5, 'V'), (1, 'I')]\n"
        "    out = []\n"
        "    for value, sym in table:\n"
        "        while n >= value:\n"
        "            out.append(sym)\n"
        "            n -= value\n"
        "    return ''.join(out)\n"
    ),
}


def _fence(code: str) -> str:
    """Racchiude il codice in un blocco recintato ```python ... ``` come farebbe il modello."""
    return "```python\n" + code + "```"


@pytest.fixture(scope="module")
def validator() -> PythonUnitTestValidator:
    assert os.path.exists(_DATASET_PATH), f"dataset mancante: {_DATASET_PATH}"
    return PythonUnitTestValidator(dataset_path=_DATASET_PATH)


def test_dataset_has_ten_problems(validator):
    items = validator.items()
    assert len(items) == 10
    # item_id unici e ordine deterministico (quello del file).
    ids = [it.item_id for it in items]
    assert len(set(ids)) == 10


def test_every_item_has_reference_solutions(validator):
    """Sanity: ogni problema del dataset ha una soluzione corretta e una sbagliata di riferimento."""
    ids = {it.item_id for it in validator.items()}
    assert ids == set(_CORRECT.keys())
    assert ids == set(_WRONG.keys())


@pytest.mark.parametrize("item_id", sorted(_CORRECT.keys()))
def test_correct_solution_passes(validator, item_id):
    item = next(it for it in validator.items() if it.item_id == item_id)
    assert validator.verify(item, _fence(_CORRECT[item_id])) is True


@pytest.mark.parametrize("item_id", sorted(_WRONG.keys()))
def test_wrong_solution_fails(validator, item_id):
    item = next(it for it in validator.items() if it.item_id == item_id)
    assert validator.verify(item, _fence(_WRONG[item_id])) is False
