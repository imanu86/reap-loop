"""Scoring: exact-match (primary for the fictitious-universe synthetic set) plus
an abstention detector and an optional anonymous LLM-judge hook.

Abstention is scored as a SEPARATE signal (spec §1: "Abstention = task separato"):
an abstention is NEITHER correct NOR folded into recovery cost. `correct` is
strictly the exact/token match of the gold fictitious answer.
"""
from __future__ import annotations
import re
import unicodedata

_ABSTENTION_PATTERNS = [
    r"\bi\s+don'?t\s+know\b", r"\bidk\b", r"\bnot\s+sure\b", r"\bno\s+idea\b",
    r"\bunknown\b", r"\bcannot\s+(find|determine|tell)\b", r"\bcan'?t\s+(find|tell)\b",
    r"\bno\s+information\b", r"\bnot\s+(mentioned|provided|stated|available)\b",
    r"\bn/?a\b", r"\bnon\s+lo\s+so\b", r"\bnon\s+so\b", r"\bnessuna\s+informazione\b",
]
_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = s.strip("\"'`.,;:!?()[]{} \t\n")
    s = re.sub(r"\s+", " ", s)
    return s


def is_abstention(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(_ABSTENTION_RE.search(t))


def _tokens(s: str) -> list:
    return re.findall(r"[a-z0-9][a-z0-9\-]*", normalize(s))


def exact_match(pred: str, gold: str) -> bool:
    """True if the gold answer is present. Short-answer format usually yields a
    clean equality; we also accept gold as a standalone token inside a sentence
    ('the code is qff-9032') to avoid penalizing verbosity, not partial matches."""
    gp, gg = normalize(pred), normalize(gold)
    if not gg:
        return False
    if gp == gg:
        return True
    # token-level containment (gold may be multi-token, e.g. an id with spaces)
    gold_toks = _tokens(gold)
    pred_toks = _tokens(pred)
    if not gold_toks:
        return False
    n = len(gold_toks)
    for i in range(len(pred_toks) - n + 1):
        if pred_toks[i:i + n] == gold_toks:
            return True
    return False


def score_item(pred: str, gold: str) -> dict:
    """Returns {correct, abstained}. `correct` never counts an abstention."""
    abstained = is_abstention(pred)
    correct = (not abstained) and exact_match(pred, gold)
    return {"correct": bool(correct), "abstained": bool(abstained)}


# --------------------------------------------------------------------------- #
# Optional anonymous LLM-judge (open-ended only; NOT used for synthetic Step 1).
# Rubric is pre-registered here. The judge never learns which arm produced the
# answer (anonymous A/B), mirroring the user's blind-eval protocol.
# --------------------------------------------------------------------------- #
JUDGE_RUBRIC = (
    "You are grading whether a candidate answer correctly recovers a specific "
    "fact from a fictitious universe. You are given QUESTION, GOLD (the correct "
    "fact), and ANSWER. Reply with exactly 'YES' if ANSWER states the same fact "
    "as GOLD (wording may differ), otherwise 'NO'. Do not consider style, only "
    "factual match. Do not reward hedging or refusals."
)


def judge(question: str, gold: str, pred: str, chat_fn, base_url=None, model=None) -> bool:
    user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nANSWER:\n{pred}\n\nVerdict (YES/NO):"
    res = chat_fn(
        [{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": user}],
        base_url=base_url, model=model, max_tokens=4, temperature=0.0, logprobs=False,
    )
    return res.text.strip().upper().startswith("YES")
