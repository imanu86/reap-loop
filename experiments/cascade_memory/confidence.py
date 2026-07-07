"""Confidence sensor (FLARE-style), PRE-REGISTERED.

Primary signal (frozen before any AUROC was inspected — config.SENSOR_PRIMARY):
    mean_logprob = mean per-token log-probability of the generated answer span.
    Higher (closer to 0) => more confident.

Secondary signals (min_logprob, mean_prob) are EXPLORATORY: logged, but NOT used
for the Gate-1 decision. Selecting a signal post-hoc to maximize AUROC would be
p-hacking (spec §4). If the server returns no logprobs, a schema/format fallback
is used (analog of the deterministic `confidence_low` gate) and each item records
which sensor path produced its score, so analysis can keep the axis consistent.
"""
from __future__ import annotations
import math
import numpy as np

import config
from scorer import is_abstention


def _schema_format_confidence(text: str, expected_format_re=None) -> float:
    """Deterministic fallback when no logprobs are available. Returns a
    pseudo-log-prob on the same 'higher=more confident' axis:
        abstention / empty      -> very low  (-4.0)
        matches expected format -> high      (-0.2)
        otherwise               -> mid-low   (-2.0)
    Coarse by construction; flagged via `used='schema_format'` so it is never
    silently mixed with real logprob scores in the headline AUROC."""
    t = (text or "").strip()
    if not t or is_abstention(t):
        return -4.0
    if expected_format_re is not None and expected_format_re.search(t):
        return -0.2
    return -2.0


def score(result, expected_format_re=None) -> dict:
    """Compute confidence signals for one LLMResult.
    Returns {primary, primary_name, used, signals{...}}."""
    signals = {}
    lps = result.token_logprobs
    if lps:
        arr = np.asarray(lps, dtype=float)
        signals["mean_logprob"] = float(arr.mean())
        signals["min_logprob"] = float(arr.min())
        signals["mean_prob"] = float(np.exp(arr).mean())
        used = "logprob"
    else:
        conf = _schema_format_confidence(result.text, expected_format_re)
        signals["schema_format"] = conf
        signals["mean_logprob"] = conf   # keep primary axis populated
        used = "schema_format"

    primary = signals.get(config.SENSOR_PRIMARY, signals["mean_logprob"])
    return {
        "primary": primary,
        "primary_name": config.SENSOR_PRIMARY,
        "used": used,
        "signals": signals,
    }
