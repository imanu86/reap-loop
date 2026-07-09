#!/usr/bin/env python3
"""Run local DS4 HTTP matrix tests for PACE breath/exchange policies.

The script is intentionally Windows/WSL friendly: it starts one ds4-server per
variant, runs optional warmups plus measured requests through the OpenAI-like
HTTP API, stores raw logs/responses, and writes a compact CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


ROOT = pathlib.Path(__file__).resolve().parents[1]


def running_in_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        return "microsoft" in pathlib.Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


IN_WSL = running_in_wsl()
WSL_EXE = os.environ.get("WSL_EXE") or (r"C:\Windows\System32\wsl.exe" if os.name == "nt" else "wsl")
PROFILE_NAME = "SOTA_LOCAL_3060_TIMED"


PROMPTS = {
    "html": (
        "Crea una landing page HTML/CSS/JS single-file per un negozio di "
        "programmazione AI in stile cyberpunk. Deve avere un modulo contatti "
        "e un popup JS che dice richiesta inviata. Codice valido e compatto."
    ),
    "code": (
        "Review this C-style pseudocode for a GPU-backed MoE inference loop. "
        "Explain correctness risks and propose a minimal patch plan that "
        "preserves routing semantics while reducing repeated expert-load stalls.\n\n"
        "for token in decode:\n"
        "    selected = router(hidden)\n"
        "    for expert in selected:\n"
        "        if not cache.contains(expert):\n"
        "            load_from_ssd(expert)\n"
        "        hidden += expert(hidden)\n"
        "    if repetition_score(window) > threshold:\n"
        "        widen_mask()\n"
        "    else if stable_hit_rate > 0.9:\n"
        "        tighten_mask()\n"
    ),
    "code_mini": (
        "Review this MoE decode loop: selected = router(hidden); missing "
        "experts are loaded from SSD, then repetition_score may widen the mask. "
        "List the main correctness risks and a minimal patch plan."
    ),
}


BASE_ENV = {
    "DS4_CUDA_NO_DIRECT_IO": "1",
    "DS4_CUDA_KEEP_MODEL_PAGES": "1",
    "DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB": "0.25",
    "DS4_PACE": "1",
    "DS4_PACE_WARMUP": "50",
    "DS4_PACE_KEEP": "23",
    "DS4_PACE_KEEP_MIN": "23",
    "DS4_PACE_KEEP_MAX": "96",
    "DS4_PACE_KEEP_STEP": "8",
    "DS4_PACE_BREATH_EVERY": "999999",
    "DS4_PACE_BREATH_KEEP": "96",
    "DS4_PACE_BREATH_LEN": "80",
    "DS4_PACE_RELEARN": "0",
    "DS4_PACE_DRIFT": "0.25",
    "DS4_PACE_PREBREATH": "0",
    "DS4_PACE_PREBREATH_DRIFT": "0.18",
    "DS4_PACE_PREBREATH_EVERY": "64",
    "DS4_PACE_PREBREATH_KEEP_MAX": "96",
    "DS4_PACE_WRAP": "1",
    "DS4_PACE_DEBUG": "1",
    "DS4_PACE_CACHE_FLOOR": "1",
    "DS4_PACE_CACHE_TARGET_SLOTS": "258",
    "DS4_PACE_CACHE_FLUSH": "0",
    "DS4_PACE_PREFILL_APPLY": "0",
    "DS4_PACE_PREFILL_WAIT_WRAP": "0",
    "DS4_PACE_EXCHANGE_OBSERVE": "0",
    "DS4_PACE_ROTATE": "0",
    "DS4_PACE_ROTATE_EVERY": "32",
    "DS4_PACE_ROTATE_DECAY": "0.98",
    "DS4_SPEX_STATS": "1",
    "DS4_SPEX_HIDDEN_PREFETCH": "0",
    "DS4_SPEX_HIDDEN_GPU_LOAD": "0",
    "DS4_SPEX_HIDDEN_GPU_SCORE": "0",
    "DS4_SPEX_HIDDEN_GPU_PREFETCH": "0",
    "DS4_SPEX_HIDDEN_GPU_PREFETCH_DRY_RUN": "0",
    "DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS": "0",
    "DS4_SPEX_HIDDEN_GPU_PREFETCH_STATS_EVERY": "256",
    "DS4_SPEX_HIDDEN_CAP": "6",
    "DS4_SPEX_PREFETCH_PROFILE": "0",
    "DS4_SPEX_TRACE_ROUTING": "",
    "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "0",
    "DS4_EXPERT_TIERING": "observe",
    "DS4_EXPERT_TIERING_LOG": "",
    "DS4_EXPERT_TIERING_LOG_IDS": "0",
    "DS4_CUDA_NO_Q8_F16_CACHE": "1",
    "DS4_REAP_PREFETCH_THREADS": "16",
    "DS4_REAP_PREFETCH_LOCK": "1",
}

CONTAMINATION_CONTROLS = [
    "Base profile mirrors the local RTX 3060 UI launcher knobs, with file-heavy diagnostics off unless the variant explicitly enables them.",
    "DS4_PACE_EXCHANGE_OBSERVE=0: exchange delta logging is diagnostic, not part of this timing A/B.",
    "DS4_EXPERT_TIERING_LOG is empty and DS4_EXPERT_TIERING_LOG_IDS=0: avoid per-expert residency file churn during timed runs.",
    "SPEX hidden GPU load/score/prefetch are disabled: hidden SPEX is not the variable under test.",
    "Routing CSV is disabled in the control and enabled only in the trace_on variant.",
]


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    rationale: str = ""
    cache_experts: int | None = None


def effective_cache_experts(variant: Variant, args: argparse.Namespace) -> int:
    return variant.cache_experts if variant.cache_experts is not None else args.cache_experts


QUICK_VARIANTS = [
    Variant(
        "sota_trace_off",
        {},
        "Control: current local RTX 3060 SOTA timing profile with routing CSV disabled.",
    ),
    Variant(
        "sota_trace_on",
        {
            "DS4_SPEX_TRACE_ROUTING": "__RUN_DIR__/routing.csv",
            "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "1",
        },
        "Same profile as control, but writes routing+weight CSV for Scope replay; measures trace overhead and behavior drift.",
    ),
    Variant("no_pace", {"DS4_PACE": "0"}, "Quality/perf control with PACE disabled."),
    Variant("direct_k23", {"DS4_PACE_PREBREATH": "0"}, "Direct K23 baseline after the 50-token warmup."),
    Variant(
        "k23_static_no_breath",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
        },
        "Control for rotation: 50-token full warmup, then fixed K23 for the whole decode; no prebreath and no n-gram/clock breath.",
    ),
    Variant(
        "local_k23_cache64",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
            "DS4_PACE_CACHE_TARGET_SLOTS": "64",
        },
        "Local cache sweep: fixed K23 after 50-token K0 warmup, no breath/prebreath/rotation, server expert cache and PACE target both set to 64.",
        cache_experts=64,
    ),
    Variant(
        "local_k23_cache128",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
            "DS4_PACE_CACHE_TARGET_SLOTS": "128",
        },
        "Local cache sweep: fixed K23 after 50-token K0 warmup, no breath/prebreath/rotation, server expert cache and PACE target both set to 128.",
        cache_experts=128,
    ),
    Variant(
        "local_k23_cache258",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
            "DS4_PACE_CACHE_TARGET_SLOTS": "258",
        },
        "Local cache sweep: fixed K23 after 50-token K0 warmup, no breath/prebreath/rotation, server expert cache and PACE target both set to 258.",
        cache_experts=258,
    ),
    Variant(
        "k23_rotate_every16",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "16",
            "DS4_PACE_ROTATE_DECAY": "0.98",
        },
        "Rotation test: keep K fixed at 23, but refresh the mask every 16 decode tokens from raw router probabilities.",
    ),
    Variant(
        "k23_rotate_every32",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "32",
            "DS4_PACE_ROTATE_DECAY": "0.98",
        },
        "Lower-overhead rotation test: fixed K23, raw-router mask refresh every 32 decode tokens.",
    ),
    Variant(
        "pod_k23_static_no_breath_128",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
            "DS4_PACE_CACHE_TARGET_SLOTS": "128",
        },
        "RunPod 12GB control: fixed K23, no breath, reduced streaming cache target to 128 slots to avoid q8_0 OOM on RTX 4070 Ti pods.",
    ),
    Variant(
        "pod_k23_rotate_every16_128",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "16",
            "DS4_PACE_ROTATE_DECAY": "0.98",
            "DS4_PACE_CACHE_TARGET_SLOTS": "128",
        },
        "RunPod 12GB rotation: fixed K23, raw-router mask refresh every 16 decode tokens, cache target 128 slots to avoid q8_0 OOM.",
    ),
    Variant(
        "pod_k23_rotate_every32_128",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "32",
            "DS4_PACE_ROTATE_DECAY": "0.98",
            "DS4_PACE_CACHE_TARGET_SLOTS": "128",
        },
        "RunPod 12GB lower-overhead rotation: fixed K23, raw-router mask refresh every 32 decode tokens, cache target 128 slots.",
    ),
    Variant(
        "pod_k23_static_no_breath_64",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "0",
            "DS4_PACE_CACHE_TARGET_SLOTS": "64",
        },
        "RunPod 12GB fallback control: fixed K23, no breath, cache target 64 slots after 128-slot q8_0 OOM on RTX 4070 Ti pods.",
    ),
    Variant(
        "pod_k23_rotate_every16_64",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "16",
            "DS4_PACE_ROTATE_DECAY": "0.98",
            "DS4_PACE_CACHE_TARGET_SLOTS": "64",
        },
        "RunPod 12GB fallback rotation: fixed K23, raw-router refresh every 16 decode tokens, cache target 64 slots.",
    ),
    Variant(
        "pod_k23_rotate_every32_64",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_EVERY": "999999",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_ROTATE": "1",
            "DS4_PACE_ROTATE_EVERY": "32",
            "DS4_PACE_ROTATE_DECAY": "0.98",
            "DS4_PACE_CACHE_TARGET_SLOTS": "64",
        },
        "RunPod 12GB fallback lower-overhead rotation: fixed K23, raw-router refresh every 32 decode tokens, cache target 64 slots.",
    ),
    Variant(
        "breath_k96_return",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "0.25",
        },
        "K23 after the 50-token warmup; when n-gram drift triggers a breath, widen to K96 and record the measured return keep.",
    ),
    Variant(
        "breath_k0_return",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_KEEP": "0",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "0.25",
        },
        "K23 after the 50-token warmup; when n-gram drift triggers a breath, release the mask (K0/full) and record the measured return keep.",
    ),
    Variant(
        "breath_k96_return_k23",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_KEEP": "96",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "0.25",
        },
        "Exact return test: K23 after warmup, K96 during breath, and no sensor widening of g_pace.keep, so breath_end should return to K23.",
    ),
    Variant(
        "breath_k0_return_k23",
        {
            "DS4_PACE_KEEP": "23",
            "DS4_PACE_KEEP_MIN": "23",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_KEEP_STEP": "0",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_KEEP": "0",
            "DS4_PACE_BREATH_LEN": "80",
            "DS4_PACE_DRIFT": "0.25",
        },
        "Exact return test: K23 after warmup, full K0/unmask during breath, and no sensor widening of g_pace.keep, so breath_end should return to K23.",
    ),
    Variant(
        "keep32_direct",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "0",
        },
    ),
    Variant(
        "pre_step8_every32",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "8",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
    ),
    Variant(
        "pre_step4_every32",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "4",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
    ),
    Variant(
        "pre_step1_every16",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "1",
            "DS4_PACE_PREBREATH_EVERY": "16",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
    ),
    Variant(
        "clock_breath64",
        {
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_DRIFT": "1.0",
            "DS4_PACE_BREATH_EVERY": "96",
            "DS4_PACE_BREATH_KEEP": "64",
            "DS4_PACE_BREATH_LEN": "48",
        },
    ),
]


LONG_EXTRA_VARIANTS = [
    Variant(
        "direct_k40",
        {
            "DS4_PACE_KEEP": "40",
            "DS4_PACE_KEEP_MIN": "40",
            "DS4_PACE_PREBREATH": "0",
        },
    ),
    Variant(
        "direct_k48",
        {
            "DS4_PACE_KEEP": "48",
            "DS4_PACE_KEEP_MIN": "48",
            "DS4_PACE_PREBREATH": "0",
        },
    ),
    Variant(
        "direct_k56",
        {
            "DS4_PACE_KEEP": "56",
            "DS4_PACE_KEEP_MIN": "56",
            "DS4_PACE_PREBREATH": "0",
        },
    ),
    Variant(
        "direct_k96",
        {
            "DS4_PACE_KEEP": "96",
            "DS4_PACE_KEEP_MIN": "96",
            "DS4_PACE_KEEP_MAX": "96",
            "DS4_PACE_PREBREATH": "0",
            "DS4_PACE_BREATH_KEEP": "96",
        },
    ),
    Variant(
        "pre_step8_every16",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "8",
            "DS4_PACE_PREBREATH_EVERY": "16",
            "DS4_PACE_PREBREATH_KEEP_MAX": "96",
        },
    ),
    Variant(
        "pre_step4_every16",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "4",
            "DS4_PACE_PREBREATH_EVERY": "16",
            "DS4_PACE_PREBREATH_KEEP_MAX": "96",
        },
    ),
    Variant(
        "k32_pre_step4_every32",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "4",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
    ),
    Variant(
        "k32_pre_step2_every32",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "2",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
        "K32 plus smaller prebreath steps: test whether gentler reloads reduce breath stalls without widening too early.",
    ),
    Variant(
        "k32_pre_step2_every16",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "2",
            "DS4_PACE_PREBREATH_EVERY": "16",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
        "K32 plus frequent micro-prebreath: test one smaller reload before the ngram breath becomes urgent.",
    ),
    Variant(
        "k32_pre_step1_every16",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "1",
            "DS4_PACE_PREBREATH_EVERY": "16",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
        "K32 plus one-expert-at-a-time prebreath: smallest practical step for the long-breath hypothesis.",
    ),
    Variant(
        "k32_pre_step8_every32",
        {
            "DS4_PACE_KEEP": "32",
            "DS4_PACE_KEEP_MIN": "32",
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "8",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
        },
    ),
    Variant(
        "direct_k64",
        {
            "DS4_PACE_KEEP": "64",
            "DS4_PACE_KEEP_MIN": "64",
            "DS4_PACE_PREBREATH": "0",
        },
    ),
    Variant(
        "pre_step8_nowrap",
        {
            "DS4_PACE_PREBREATH": "1",
            "DS4_PACE_KEEP_STEP": "8",
            "DS4_PACE_PREBREATH_EVERY": "32",
            "DS4_PACE_PREBREATH_KEEP_MAX": "64",
            "DS4_PACE_WRAP": "0",
        },
    ),
]


def now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check)


def wsl_bash(script: str, *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if os.name == "posix":
        return run(["bash", "-lc", script], timeout=timeout, check=check)
    return run([WSL_EXE, "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc", script], timeout=timeout, check=check)


def wsl_path(path: pathlib.Path) -> str:
    resolved = str(path.resolve())
    drive, rest = os.path.splitdrive(resolved)
    if not drive:
        return resolved.replace(os.sep, "/")
    drive_letter = drive.rstrip(":").lower()
    return f"/mnt/{drive_letter}{rest.replace(os.sep, '/')}"


def build_env(variant: Variant, run_dir: pathlib.Path) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(variant.env)
    for key, value in list(env.items()):
        if value.startswith("__RUN_DIR__/"):
            env[key] = wsl_path(run_dir / value.removeprefix("__RUN_DIR__/"))
    env.update(
        {
            "DS4_PACE_LOG": f"/root/ds4_matrix_{variant.name}.jsonl",
            "DS4_EXPERT_TIERING_LOG": env.get("DS4_EXPERT_TIERING_LOG", ""),
        }
    )
    return env


def env_delta(env: dict[str, str]) -> dict[str, dict[str, str | None]]:
    keys = sorted(set(BASE_ENV) | set(env))
    return {
        key: {"base": BASE_ENV.get(key), "effective": env.get(key)}
        for key in keys
        if BASE_ENV.get(key) != env.get(key)
    }


def trace_stats(run_dir: pathlib.Path) -> dict:
    trace = run_dir / "routing.csv"
    if not trace.exists():
        return {"trace_enabled": 0, "trace_bytes": 0, "trace_rows": 0}
    try:
        with trace.open("r", encoding="utf-8", errors="replace") as fh:
            rows = max(0, sum(1 for _ in fh) - 1)
    except OSError:
        rows = None
    return {"trace_enabled": 1, "trace_bytes": trace.stat().st_size, "trace_rows": rows}


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_manifest(
    *,
    run_dir: pathlib.Path,
    stem: str,
    variant: Variant,
    prompt_name: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    prompt = PROMPTS[prompt_name]
    manifest = {
        "runner_id": stem,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "profile": PROFILE_NAME,
        "variant": variant.name,
        "variant_rationale": variant.rationale,
        "prompt": {
            "name": prompt_name,
            "sha256_16": prompt_hash(prompt),
            "chars": len(prompt),
        },
        "contamination_controls": CONTAMINATION_CONTROLS,
        "evidence_policy": {
            "measured": "Use only values copied from DS4 logs, HTTP usage, generated files, or direct file stats as benchmark data.",
            "derived": "Allowed only when calculated directly from measured fields kept in this run directory.",
            "hypothesis": "May be written in notes, but must not be promoted to benchmark data without a follow-up measurement.",
        },
        "server": {
            "model": args.model,
            "ctx": args.ctx,
            "server_max_tokens": args.server_max_tokens,
            "cache_experts": effective_cache_experts(variant, args),
            "cache_experts_default": args.cache_experts,
            "prefill_chunk": args.prefill_chunk,
            "port": args.port,
            "request_max_tokens": args.max_tokens,
            "warmups": args.warmups,
            "warmup_tokens": args.warmup_tokens,
        },
        "trace": {
            "routing_csv": env.get("DS4_SPEX_TRACE_ROUTING") or None,
            "weights": env.get("DS4_SPEX_TRACE_ROUTING_WEIGHTS") == "1",
            "scope_replay_note": "Open Scope with this CSV only for diagnostic replay; do not compare directly with trace_off timing unless trace overhead is accepted.",
        },
        "source_artifacts": {
            "server_stderr": "server.stderr.log",
            "server_stdout": "server.stdout.log",
            "request_measured": "request_measured.json",
            "response_measured": "response_measured.json",
            "content_measured": "content_measured.txt",
            "server_env": "server_env.json",
        },
        "env_delta_from_profile": env_delta(env),
        "env_effective": env,
    }
    (run_dir / "runner_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stop_ds4() -> None:
    wsl_bash("pkill -TERM ds4-server 2>/dev/null || true; sleep 1", timeout=10, check=False)


def wait_models(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - diagnostic path.
            last = exc
        time.sleep(1)
    raise TimeoutError(f"server did not become ready on {url}: {last}")


def post_json(url: str, body: dict, timeout: int) -> tuple[float, dict | str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return time.perf_counter() - t0, {"http_error": exc.code, "body": parsed}
    except Exception as exc:  # noqa: BLE001 - command-line diagnostic path.
        return time.perf_counter() - t0, {"error": type(exc).__name__, "message": str(exc)}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return time.perf_counter() - t0, parsed


def post_stream(url: str, body: dict, timeout: int, events_path: pathlib.Path) -> tuple[float, dict | str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    events = []
    content_parts = []
    finish_reason = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    events.append(
                        {
                            "event_index": len(events) + 1,
                            "t_s": round(time.perf_counter() - t0, 6),
                            "raw": payload,
                            "parse_error": True,
                        }
                    )
                    continue
                delta = ""
                choices = obj.get("choices") or []
                if choices:
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta_obj = choice.get("delta") or {}
                    delta = delta_obj.get("content") or ""
                    if not delta:
                        message = choice.get("message") or {}
                        delta = message.get("content") or ""
                if obj.get("usage") is not None:
                    usage = obj.get("usage")
                if delta:
                    content_parts.append(delta)
                events.append(
                    {
                        "event_index": len(events) + 1,
                        "t_s": round(time.perf_counter() - t0, 6),
                        "delta": delta,
                        "delta_chars": len(delta),
                        "content_chars": sum(len(part) for part in content_parts),
                        "finish_reason": finish_reason,
                        "usage": usage,
                    }
                )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return time.perf_counter() - t0, {"http_error": exc.code, "body": parsed}
    except Exception as exc:  # noqa: BLE001 - command-line diagnostic path.
        return time.perf_counter() - t0, {"error": type(exc).__name__, "message": str(exc)}

    with events_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    content = "".join(content_parts)
    return (
        time.perf_counter() - t0,
        {
            "stream": True,
            "stream_events": len(events),
            "stream_content_events": sum(1 for event in events if event.get("delta")),
            "usage": usage,
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        },
    )


def response_content(response: dict | str) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def parse_server_log(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def empty_metrics() -> dict:
        return {
            "prompt_s": None,
            "finish_s": None,
            "first50_tps": None,
            "last_chunk_tps": None,
            "avg_tps": None,
            "exchange_events": 0,
            "exchange_promote": 0,
            "exchange_demote": 0,
            "prebreaths": 0,
            "breaths": 0,
            "tightens": 0,
            "pace_learned": 0,
            "pace_descents": 0,
            "pace_prebreaths": 0,
            "pace_breaths": 0,
            "pace_breath_ends": 0,
            "pace_tightens": 0,
            "pace_rotates": 0,
            "first_learned_tok": None,
            "first_descent_tok": None,
            "first_descent_keep": None,
            "first_prebreath_tok": None,
            "first_prebreath_keep": None,
            "first_breath_tok": None,
            "first_breath_keep": None,
            "first_breath_end_tok": None,
            "first_breath_end_keep": None,
            "first_rotate_tok": None,
            "first_rotate_keep": None,
            "last_pace_tok": None,
            "last_pace_keep": None,
            "prefetch_count": 0,
            "prefetch_gib": 0.0,
            "prefetch_ms": 0.0,
        }

    current = empty_metrics()
    last_finished = current.copy()
    for line in text.splitlines():
        if "PACE prefill_reset" in line:
            current = empty_metrics()
        m = re.search(r"prompt done ([0-9.]+)s", line)
        if m:
            current["prompt_s"] = float(m.group(1))
        m = re.search(r"gen=50 decoding chunk=([0-9.]+) t/s", line)
        if m and current["first50_tps"] is None:
            current["first50_tps"] = float(m.group(1))
        m = re.search(r"decoding chunk=([0-9.]+) t/s avg=([0-9.]+) t/s", line)
        if m:
            current["last_chunk_tps"] = float(m.group(1))
            current["avg_tps"] = float(m.group(2))
        m = re.search(r"finish=[^ ]+ ([0-9.]+)s", line)
        if m:
            current["finish_s"] = float(m.group(1))
            last_finished = current.copy()
        m = re.search(r"PACE exchange ([^ ]+) .* promote=(\d+) demote=(\d+)", line)
        if m:
            current["exchange_events"] += 1
            if m.group(1).startswith("prebreath"):
                current["prebreaths"] += 1
            if m.group(1).startswith("breath"):
                current["breaths"] += 1
            if m.group(1).startswith("tighten"):
                current["tightens"] += 1
            current["exchange_promote"] += int(m.group(2))
            current["exchange_demote"] += int(m.group(3))
        m = re.search(
            r"PACE (learned|descent|prebreath|breath\([^)]*\)|breath_end|tighten|rotate)\s+"
            r"tok=(\d+)\s+phase=\d+\s+keep=(\d+)",
            line,
        )
        if m:
            event = m.group(1)
            tok = int(m.group(2))
            keep = int(m.group(3))
            current["last_pace_tok"] = tok
            current["last_pace_keep"] = keep
            if event == "learned":
                current["pace_learned"] += 1
                if current["first_learned_tok"] is None:
                    current["first_learned_tok"] = tok
            elif event == "descent":
                current["pace_descents"] += 1
                if current["first_descent_tok"] is None:
                    current["first_descent_tok"] = tok
                    current["first_descent_keep"] = keep
            elif event == "prebreath":
                current["pace_prebreaths"] += 1
                if current["first_prebreath_tok"] is None:
                    current["first_prebreath_tok"] = tok
                    current["first_prebreath_keep"] = keep
            elif event.startswith("breath"):
                if event == "breath_end":
                    current["pace_breath_ends"] += 1
                    if current["first_breath_end_tok"] is None:
                        current["first_breath_end_tok"] = tok
                        current["first_breath_end_keep"] = keep
                else:
                    current["pace_breaths"] += 1
                    if current["first_breath_tok"] is None:
                        current["first_breath_tok"] = tok
                        current["first_breath_keep"] = keep
            elif event == "tighten":
                current["pace_tightens"] += 1
            elif event == "rotate":
                current["pace_rotates"] += 1
                if current["first_rotate_tok"] is None:
                    current["first_rotate_tok"] = tok
                    current["first_rotate_keep"] = keep
        m = re.search(r"REAP prefetch .* ([0-9.]+) GiB touched in ([0-9.]+) ms", line)
        if m:
            current["prefetch_count"] += 1
            current["prefetch_gib"] += float(m.group(1))
            current["prefetch_ms"] += float(m.group(2))
    last_finished["prefetch_gib"] = round(last_finished["prefetch_gib"], 3)
    last_finished["prefetch_ms"] = round(last_finished["prefetch_ms"], 3)
    return last_finished


def quality_flags(prompt_name: str, content: str) -> dict:
    lower = content.lower()
    repeated = bool(re.search(r"(.{24,160})\1\1", content, re.S))
    html_balance = ""
    if prompt_name == "html":
        html_balance = lower.count("<html") - lower.count("</html>")
    return {
        "content_chars": len(content),
        "prefix": content[:120].replace("\n", "\\n"),
        "s_init_count": content.count("S_INIT"),
        "html_balance": html_balance,
        "doctype": int("<!doctype html" in lower),
        "has_popup": int("alert(" in lower or "popup" in lower or "richiesta inviata" in lower),
        "repeat_flag": int(repeated),
    }


def start_server(
    variant: Variant,
    env: dict[str, str],
    out_dir: pathlib.Path,
    port: int,
    args: argparse.Namespace,
) -> subprocess.Popen:
    cache_experts = effective_cache_experts(variant, args)
    env_prefix = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(env.items()))
    trace_path = env.get("DS4_SPEX_TRACE_ROUTING")
    trace_rm = f"rm -f {json.dumps(trace_path)} && " if trace_path else ""
    cmd = (
        "cd /root/ds4 && "
        f"rm -f /root/ds4_matrix_{variant.name}.jsonl && "
        f"{trace_rm}"
        f"{env_prefix} /root/ds4/ds4-server "
        f"-m {args.model} --cuda --ssd-streaming "
        f"--ssd-streaming-cache-experts {cache_experts} "
        f"--prefill-chunk {args.prefill_chunk} "
        f"-c {args.ctx} -n {args.server_max_tokens} "
        f"--host 127.0.0.1 --port {port} --cors"
    )
    stdout = (out_dir / "server.stdout.log").open("wb")
    stderr = (out_dir / "server.stderr.log").open("wb")
    (out_dir / "server_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    launcher = ["bash", "-lc", cmd] if os.name == "posix" else [WSL_EXE, "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-lc", cmd]
    return subprocess.Popen(
        launcher,
        stdout=stdout,
        stderr=stderr,
    )


def run_request(
    *,
    port: int,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    out_path: pathlib.Path,
    phase: str,
    stream: bool = False,
) -> tuple[float, dict | str]:
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "think": False,
        "thinking": {"type": "disabled"},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    (out_path / f"request_{phase}.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if stream:
        wall_s, response = post_stream(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            body,
            timeout,
            out_path / f"stream_events_{phase}.jsonl",
        )
    else:
        wall_s, response = post_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            body,
            timeout,
        )
    (out_path / f"response_{phase}.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    content = response_content(response)
    (out_path / f"content_{phase}.txt").write_text(content, encoding="utf-8")
    return wall_s, response


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["quick", "long"], default="quick")
    ap.add_argument("--prompts", default="html")
    ap.add_argument("--variants", help="Comma-separated variant names; default: whole suite")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--warmup-tokens", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--port", type=int, default=8014)
    ap.add_argument("--out-dir")
    ap.add_argument("--model", default="/root/models/ds4-2bit.gguf")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--server-max-tokens", type=int, default=2048)
    ap.add_argument("--cache-experts", type=int, default=258)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--stream", action="store_true", help="Capture measured requests as SSE stream events with timestamps.")
    ap.add_argument("--no-stop-existing", action="store_true")
    args = ap.parse_args()

    prompt_names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for name in prompt_names:
        if name not in PROMPTS:
            raise SystemExit(f"unknown prompt {name!r}; choose from {', '.join(PROMPTS)}")

    variants = QUICK_VARIANTS + (LONG_EXTRA_VARIANTS if args.suite == "long" else [])
    if args.variants:
        wanted_order = [v.strip() for v in args.variants.split(",") if v.strip()]
        wanted = set(wanted_order)
        known = {v.name for v in variants}
        missing = sorted(wanted - known)
        if missing:
            raise SystemExit(f"unknown variants: {', '.join(missing)}; known: {', '.join(sorted(known))}")
        by_name = {v.name: v for v in variants}
        variants = [by_name[name] for name in wanted_order]
    out_root = pathlib.Path(args.out_dir) if args.out_dir else ROOT / "runs" / "ds4" / f"{now_stamp()}_exchange_matrix"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "matrix_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "profile": PROFILE_NAME,
                "base_env": BASE_ENV,
                "contamination_controls": CONTAMINATION_CONTROLS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = []

    if not args.no_stop_existing:
        stop_ds4()

    for variant in variants:
        for prompt_name in prompt_names:
            for run_idx in range(1, args.runs + 1):
                stem = f"{prompt_name}_{variant.name}_r{run_idx:02d}"
                run_dir = out_root / stem
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[matrix] start {stem}", flush=True)
                env = build_env(variant, run_dir)
                write_manifest(
                    run_dir=run_dir,
                    stem=stem,
                    variant=variant,
                    prompt_name=prompt_name,
                    args=args,
                    env=env,
                )
                proc = start_server(variant, env, run_dir, args.port, args)
                try:
                    wait_models(args.port, timeout_s=90)
                    for warm_idx in range(1, args.warmups + 1):
                        warm_wall, _ = run_request(
                            port=args.port,
                            prompt_name=prompt_name,
                            prompt=PROMPTS[prompt_name],
                            max_tokens=args.warmup_tokens,
                            timeout=args.timeout,
                            out_path=run_dir,
                            phase=f"warmup_{warm_idx:02d}",
                            stream=False,
                        )
                        print(f"[matrix] warmup {stem} {warm_wall:.3f}s", flush=True)
                    wall_s, response = run_request(
                        port=args.port,
                        prompt_name=prompt_name,
                        prompt=PROMPTS[prompt_name],
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        out_path=run_dir,
                        phase="measured",
                        stream=args.stream,
                    )
                    usage = response.get("usage") if isinstance(response, dict) else None
                    content = response_content(response)
                    log_metrics = parse_server_log(run_dir / "server.stderr.log")
                    q = quality_flags(prompt_name, content)
                    completion_tokens = (usage or {}).get("completion_tokens")
                    post_breath_end_tokens = None
                    if completion_tokens is not None and log_metrics.get("first_breath_end_tok") is not None:
                        post_breath_end_tokens = max(0, int(completion_tokens) - int(log_metrics["first_breath_end_tok"]))
                    row = {
                        "stem": stem,
                        "evidence_type": "measured",
                        "source_artifacts": "server.stderr.log,response_measured.json,routing.csv(if enabled)",
                        "profile": PROFILE_NAME,
                        "prompt": prompt_name,
                        "variant": variant.name,
                        "variant_rationale": variant.rationale,
                        "run": run_idx,
                        "wall_s": round(wall_s, 3),
                        "prompt_tokens": (usage or {}).get("prompt_tokens"),
                        "completion_tokens": completion_tokens,
                        "stream_events": response.get("stream_events") if isinstance(response, dict) else None,
                        "stream_content_events": response.get("stream_content_events") if isinstance(response, dict) else None,
                        "derived_tokens_after_breath_end_to_finish": post_breath_end_tokens,
                        **log_metrics,
                        **trace_stats(run_dir),
                        **q,
                    }
                    rows.append(row)
                    print(
                        "[matrix] done "
                        f"{stem} wall={wall_s:.1f}s avg={log_metrics.get('avg_tps')} "
                        f"pace_pre={log_metrics.get('pace_prebreaths')} pace_br={log_metrics.get('pace_breaths')} "
                        f"pace_rot={log_metrics.get('pace_rotates')} promote={log_metrics.get('exchange_promote')}",
                        flush=True,
                    )
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    stop_ds4()
                    pace_src = f"/root/ds4_matrix_{variant.name}.jsonl"
                    pace_dst = run_dir / "pace_events.jsonl"
                    wsl_bash(
                        f"test -f {pace_src} && cp {pace_src} "
                        f"{json.dumps(wsl_path(pace_dst))} || true",
                        timeout=10,
                        check=False,
                    )

    if rows:
        fieldnames = list(rows[0].keys())
        summary = out_root / "summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (out_root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary)

    # Restore normal UI server if the local script exists.
    start_script = pathlib.Path(
        "/mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/outputs/ds4-simple-ui/start-ds4-3060.sh"
        if IN_WSL
        else r"C:\Users\imanu\Documents\Codex\2026-07-07\cia\outputs\ds4-simple-ui\start-ds4-3060.sh"
    )
    if start_script.exists():
        logs = start_script.parent
        with (logs / "ds4-server.stdout.log").open("wb") as out, (logs / "ds4-server.stderr.log").open("wb") as err:
            restore_cmd = "bash /mnt/c/Users/imanu/Documents/Codex/2026-07-07/cia/outputs/ds4-simple-ui/start-ds4-3060.sh"
            launcher = ["bash", "-lc", restore_cmd] if IN_WSL else [
                WSL_EXE,
                "-d",
                "Ubuntu-24.04",
                "-u",
                "root",
                "--",
                "bash",
                "-lc",
                restore_cmd,
            ]
            subprocess.Popen(
                launcher,
                stdout=out,
                stderr=err,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
