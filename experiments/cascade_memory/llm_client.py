"""Model-agnostic LLM adapter over an OpenAI-compatible endpoint.

Uses only the Python stdlib (urllib) so the harness has ZERO new hard deps
beyond numpy/sklearn. Two backends:

  * HTTPBackend  — real /v1/chat/completions (llama.cpp, vLLM, LM Studio, remote
                   OpenAI-compatible APIs). Extracts `usage` (cost) and per-token
                   logprobs (FLARE-style sensor) where the server returns them.
  * MockBackend  — offline, deterministic. `echo` mode proves the plumbing (Step 0
                   smoke test) with NO network. `oracle_decay` mode fabricates a
                   correctness/confidence signal that decays with turn-distance so
                   the analysis code (ROC, bootstrap) can be exercised offline.
                   A mock AUROC is SYNTHETIC and is never a real Gate-1 result.

Return type is LLMResult; token counts always come from the server (real backend)
so the primary cost axis stays reproducible and non-gameable.
"""
from __future__ import annotations
import json
import time
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class LLMResult:
    text: str
    usage: dict                      # {prompt_tokens, completion_tokens, ...}
    token_logprobs: Optional[list]   # list[float] over the answer span, or None
    finish_reason: str = ""
    wall_clock_s: float = 0.0
    backend: str = "http"
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Real HTTP backend
# --------------------------------------------------------------------------- #
def _http_chat(messages, base_url, model, timeout, **overrides) -> LLMResult:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": messages}
    for k, v in config.GEN_DEFAULTS.items():
        body[k] = v
    body.update(overrides)  # explicit call-site overrides win
    # drop mock-only keys if any slipped through
    body.pop("mock_meta", None)

    data = json.dumps(body).encode("utf-8")
    # Non-default User-Agent: some proxies (e.g. RunPod's Cloudflare front) 403 the
    # stdlib "Python-urllib/x" UA. A browser/curl-like UA passes.
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0

    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    usage = payload.get("usage") or {}
    token_logprobs = _extract_logprobs(choice)
    return LLMResult(
        text=text,
        usage=usage,
        token_logprobs=token_logprobs,
        finish_reason=choice.get("finish_reason", ""),
        wall_clock_s=dt,
        backend="http",
        raw=payload,
    )


def _extract_logprobs(choice: dict) -> Optional[list]:
    """OpenAI chat-completions logprob shape:
       choice.logprobs.content = [ {token, logprob, top_logprobs:[...]}, ... ]
    Returns the flat list of per-token logprobs over the generated span, or None
    if the server did not return logprobs (→ sensor falls back to schema/format)."""
    lp = choice.get("logprobs")
    if not lp:
        return None
    content = lp.get("content")
    if not content:
        # some servers use the legacy completions shape {token_logprobs:[...]}
        tl = lp.get("token_logprobs")
        return [x for x in tl if x is not None] if tl else None
    out = []
    for tok in content:
        v = tok.get("logprob")
        if v is not None:
            out.append(float(v))
    return out or None


# --------------------------------------------------------------------------- #
# Mock backend (offline, deterministic)
# --------------------------------------------------------------------------- #
def _stable_unit(*parts) -> float:
    """Deterministic float in [0,1) from a string key (stdlib random is process-
    salted for hash(); we use md5 for reproducibility across runs)."""
    h = hashlib.md5("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _mock_chat(messages, mode="echo", mock_meta=None, **overrides) -> LLMResult:
    prompt_text = "\n".join(m.get("content", "") for m in messages)
    prefill = sum(_approx_tokens(m.get("content", "")) for m in messages)
    mock_meta = mock_meta or {}

    if mode == "echo":
        text = "MOCK-0000"
        decode = _approx_tokens(text)
        # flat, mid logprobs — deliberately non-discriminative
        lps = [-0.7] * decode
        usage = {"prompt_tokens": prefill, "completion_tokens": decode}
        return LLMResult(text, usage, lps, "stop", 0.0, "mock/echo")

    if mode == "oracle_decay":
        # Synthetic correctness that decays with turn-distance, and a confidence
        # signal correlated with it. Uses the GOLD answer + distance directly →
        # this is a plumbing/analysis fixture, NOT a model measurement.
        gold = str(mock_meta.get("answer", "UNKNOWN"))
        distance = float(mock_meta.get("distance", 20))
        key = mock_meta.get("item_id", prompt_text[:32])
        p_correct = min(0.97, max(0.03, 1.0 - distance / 100.0))
        roll = _stable_unit("correct", key)
        correct = roll < p_correct
        if correct:
            text = gold
            base_lp = -0.15 - 0.9 * _stable_unit("hi", key)     # confident
        else:
            text = "I don't know"
            base_lp = -2.2 - 1.5 * _stable_unit("lo", key)      # unconfident
        decode = _approx_tokens(text)
        lps = [base_lp + 0.2 * (_stable_unit("j", key, j) - 0.5) for j in range(decode)]
        usage = {"prompt_tokens": prefill, "completion_tokens": decode}
        return LLMResult(text, usage, lps, "stop", 0.0, "mock/oracle_decay")

    if mode == "oracle_context":
        # Correctness tied to whether the gold fact is actually PRESENT in the
        # context the model sees (native window OR recovered notes). This is the
        # right fixture for the cascade: each rung either surfaces the fact or not.
        gold = str(mock_meta.get("answer", "UNKNOWN"))
        key = mock_meta.get("item_id", prompt_text[:32])
        present = gold.lower() in prompt_text.lower()
        if present:
            text = gold
            base_lp = -0.12 - 0.6 * _stable_unit("hi", key)
        else:
            text = "I don't know"
            base_lp = -2.3 - 1.4 * _stable_unit("lo", key)
        decode = _approx_tokens(text)
        lps = [base_lp + 0.15 * (_stable_unit("j", key, j) - 0.5) for j in range(decode)]
        usage = {"prompt_tokens": prefill, "completion_tokens": decode}
        return LLMResult(text, usage, lps, "stop", 0.0, "mock/oracle_context")

    raise ValueError(f"unknown mock mode: {mode}")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def chat(messages, base_url=None, model=None, timeout=None, mock_meta=None,
         **overrides) -> LLMResult:
    """Single chat completion. If base_url starts with 'mock', the offline mock
    backend is used (mode after the colon, e.g. 'mock:oracle_decay')."""
    base_url = base_url if base_url is not None else config.BASE_URL
    model = model if model is not None else config.MODEL
    timeout = timeout if timeout is not None else config.REQUEST_TIMEOUT_S

    if base_url.startswith("mock"):
        mode = base_url.split(":", 1)[1] if ":" in base_url else "echo"
        return _mock_chat(messages, mode=mode, mock_meta=mock_meta, **overrides)
    return _http_chat(messages, base_url, model, timeout, **overrides)
