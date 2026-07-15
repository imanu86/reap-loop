#!/usr/bin/env python3
"""Build a DS4/REAP experiment ledger from run artifacts and legacy notes.

The output is intentionally conservative: measured runner rows keep normalized
numeric fields, while older Claude/research notes are preserved as textual rows
with their evidence level. This prevents historical data from disappearing
without pretending every old note is directly benchmark-comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_G7_SNAPSHOT_DATE = "20260714"

FIELDS = [
    "row_id",
    "source_kind",
    "evidence_level",
    "benchmark_usable",
    "date",
    "suite",
    "run_id",
    "category",
    "experiment",
    "variant",
    "profile",
    "prompt_name",
    "prompt_sha16",
    "prompt_chars",
    "prompt_excerpt",
    "system_prompt",
    "hardware",
    "gpu",
    "model",
    "model_bytes",
    "ds4_base",
    "ds4_build",
    "patches",
    "reap_loop_commit",
    "runner_commit",
    "ctx",
    "prefill_chunk",
    "server_cache_experts",
    "pace_cache_target",
    "request_max_tokens",
    "server_max_tokens",
    "temperature",
    "stream",
    "think",
    "trace_routing",
    "trace_weights",
    "trace_residency",
    "expert_tiering",
    "tiering_log_ids",
    "pace_warmup",
    "pace_keep",
    "pace_keep_min",
    "pace_keep_max",
    "pace_keep_step",
    "pace_breath_every",
    "pace_breath_keep",
    "pace_breath_len",
    "pace_prebreath",
    "pace_prebreath_every",
    "pace_prebreath_keep_max",
    "pace_relearn",
    "pace_relearn_on_tighten",
    "pace_rotate",
    "pace_rotate_every",
    "pace_wrap",
    "pace_wrap_rotate",
    "hidden_spex",
    "completion_tokens",
    "wall_s",
    "prompt_s",
    "finish_s",
    "first50_tps",
    "avg_tps",
    "last_chunk_tps",
    "prefetch_count",
    "prefetch_gib",
    "prefetch_ms",
    "tier_miss_eviction",
    "trace_rows",
    "trace_bytes",
    "content_chars",
    "doctype",
    "html_balance",
    "has_form",
    "has_script",
    "has_popup",
    "repeat_flag",
    "l0l3",
    "client_stop_enabled",
    "first_client_stop_reason",
    "client_stop_reason",
    "retry_attempts",
    "final_attempt_phase",
    "coherent_until_token_est",
    "useful_tokens_after_return_est",
    "pace_events",
    "quality_signal",
    "metrics_text",
    "setup_text",
    "env_effective_json",
    "request_settings_json",
    "runtime_platform",
    "cache_state",
    "source_head",
    "executable_sha256",
    "harness_sha256",
    "repeats",
    "replication_scope",
    "warmup",
    "server_decode_mean_tps",
    "server_decode_min_tps",
    "server_decode_max_tps",
    "client_completion_mean_tps",
    "client_completion_min_tps",
    "client_completion_max_tps",
    "server_prefill_ttft_s",
    "outputs_identical",
    "expected_hash_match",
    "output_sha256",
    "dynamic_arena_gib",
    "dynamic_arena_window",
    "dynamic_arena_min_hits",
    "dynamic_arena_grow_interval",
    "dynamic_arena_carry",
    "dynamic_arena_resident",
    "dynamic_arena_resident_gib",
    "dynamic_arena_hit_rate",
    "dynamic_arena_fatal",
    "q8_f16_cache_mib",
    "q8_f16_reserve_mib",
    "moe_io_qd",
    "resident_expert_cache",
    "spex_stage",
    "spex_cap",
    "spex_recall",
    "standby_before_gib",
    "wddm_shared_peak_gib",
    "wddm_dedicated_peak_gib",
    "process_read_gib",
    "gpu_util_median",
    "vram_peak_mib",
    "result_text",
    "verdict",
    "source_artifacts",
    "dynamic_arena_final_hits",
    "dynamic_arena_final_misses",
    "prefill_mass_wrap_result",
    "prefill_mass_wrap_reason",
    "prefill_mass_wrap_candidate_entries",
    "prefill_mass_wrap_loads",
    "prefill_mass_wrap_workers",
    "prefill_mass_wrap_seconds",
    "prefill_mass_wrap_snapshot_before",
    "prefill_mass_wrap_snapshot_after",
    "prefill_mass_wrap_resident_before",
    "prefill_mass_wrap_resident_after",
    "prefill_mass_decode_candidate_hits",
    "reap_mass_wrap_publications",
    "reap_mass_wrap_failures",
    "reap_mass_wrap_entrants",
    "reap_mass_wrap_victims",
    "reap_mass_wrap_seconds",
    "reap_mass_wrap_capacity_entries",
    "reap_mass_wrap_last_result",
    "reap_mass_wrap_last_reason",
    "reap_mass_wrap_last_snapshot_after",
    "ds4_cuda_sha256",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return {}


def text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def split_md_row(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|") or "---" in line:
        return []
    return [cell.strip().replace("<br>", " ") for cell in line.strip("|").split("|")]


def base_row() -> dict[str, str]:
    return {field: "" for field in FIELDS}


def detect_gpu(log: str) -> str:
    for pat in [
        r"initialized on ([^(,\n]+(?:\([^)]*\))?)",
        r"CUDA backend initialized on ([^\n]+)",
        r"GPU:\s*([^\n]+)",
    ]:
        m = re.search(pat, log, re.I)
        if m:
            return m.group(1).strip()
    return ""


def derive_quality(row: dict[str, str]) -> str:
    bits: list[str] = []
    if row.get("l0l3") != "":
        bits.append(f"L{row['l0l3']}")
    if row.get("repeat_flag") != "":
        bits.append(f"repeat_proxy={row['repeat_flag']}")
    if row.get("client_stop_reason"):
        bits.append(f"client_stop={row['client_stop_reason']}")
    if row.get("coherent_until_token_est"):
        bits.append(f"coherent_until~{row['coherent_until_token_est']}")
    if row.get("doctype"):
        bits.append(f"doctype={row['doctype']}")
    for key, label in [("has_form", "form"), ("has_script", "script"), ("has_popup", "popup")]:
        if row.get(key) not in ("", "0", "False", "false"):
            bits.append(label)
    if row.get("verdict"):
        bits.append(row["verdict"])
    return "; ".join(bits)


def parse_requested_coherence(repo: Path) -> dict[tuple[str, str], tuple[str, str]]:
    path = repo / "runs/ds4/20260709_requested_breath_rotation_RESULTS.md"
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    for line in text(path).splitlines():
        cells = split_md_row(line)
        if len(cells) < 10 or not cells[0].isdigit():
            continue
        cache = cells[0]
        test = cells[1]
        coherent = cells[7]
        useful_after = cells[8]
        key = ""
        if "rotate32" in test:
            key = "rotate32"
        elif "static" in test:
            key = "static"
        elif "K0" in test:
            key = "breath_k0"
        elif "K96" in test:
            key = "breath_k96"
        if key:
            mapping[(cache, key)] = (coherent, useful_after)
    return mapping


def requested_key(variant: str) -> str:
    if "rotate32" in variant:
        return "rotate32"
    if "breath_k0" in variant:
        return "breath_k0"
    if "breath_k96" in variant:
        return "breath_k96"
    if "k23_static" in variant or ("k23_cache" in variant and "rotate" not in variant):
        return "static"
    return ""


def parse_summary_rows(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    coherence = parse_requested_coherence(repo)
    for summary in sorted((repo / "runs/ds4").rglob("summary.csv")):
        suite = summary.parent.name
        with summary.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for idx, src in enumerate(reader, start=1):
                stem = src.get("stem") or src.get("run") or f"row{idx}"
                leaf = summary.parent / stem if (summary.parent / stem).is_dir() else summary.parent
                manifest = read_json(leaf / "runner_manifest.json")
                if not manifest:
                    manifest = read_json(summary.parent / "runner_manifest.json")
                env = read_json(leaf / "server_env.json")
                request = read_json(leaf / "request_measured.json")
                response = read_json(leaf / "response_measured.json")
                quality_notes = read_json(leaf / "quality_notes.json")
                usage = as_dict(response.get("usage"))
                content = text(leaf / "content_measured.txt")
                log = text(leaf / "server.stderr.log") + "\n" + text(leaf / "server.stdout.log")

                prompt_obj = manifest.get("prompt", {}) if isinstance(manifest.get("prompt"), dict) else {}
                messages = request.get("messages", []) if isinstance(request.get("messages"), list) else []
                system_prompt = ""
                user_prompt = ""
                for msg in messages:
                    if msg.get("role") == "system":
                        system_prompt = clean(msg.get("content", ""))
                    if msg.get("role") == "user" and not user_prompt:
                        user_prompt = clean(msg.get("content", ""))
                if not user_prompt:
                    user_prompt = clean(prompt_obj.get("name") or src.get("prompt"))

                row = base_row()
                row.update(
                    {
                        "row_id": f"RUN-{suite}-{stem}",
                        "source_kind": "runner_summary",
                        "evidence_level": "measured",
                        "benchmark_usable": "yes",
                        "date": suite[:8] if suite[:8].isdigit() else "",
                        "suite": suite,
                        "run_id": stem,
                        "category": "ds4_runtime",
                        "experiment": clean(src.get("variant_rationale") or manifest.get("variant_rationale")),
                        "variant": clean(src.get("variant") or manifest.get("variant")),
                        "profile": clean(src.get("profile") or manifest.get("profile")),
                        "prompt_name": clean(src.get("prompt") or prompt_obj.get("name")),
                        "prompt_sha16": clean(prompt_obj.get("sha256_16") or short_hash(user_prompt)),
                        "prompt_chars": clean(prompt_obj.get("chars") or len(user_prompt)),
                        "prompt_excerpt": user_prompt[:240],
                        "system_prompt": system_prompt[:180],
                        "gpu": detect_gpu(log),
                        "model": clean(manifest.get("server", {}).get("model") or request.get("model")),
                        "ctx": clean(manifest.get("server", {}).get("ctx")),
                        "prefill_chunk": clean(manifest.get("server", {}).get("prefill_chunk")),
                        "server_cache_experts": clean(manifest.get("server", {}).get("cache_experts")),
                        "pace_cache_target": clean(env.get("DS4_PACE_CACHE_TARGET_SLOTS")),
                        "request_max_tokens": clean(request.get("max_tokens") or manifest.get("server", {}).get("request_max_tokens")),
                        "server_max_tokens": clean(manifest.get("server", {}).get("server_max_tokens")),
                        "temperature": clean(request.get("temperature")),
                        "stream": clean(request.get("stream")),
                        "think": clean(request.get("think") if "think" in request else request.get("thinking", {}).get("type")),
                        "trace_routing": clean(env.get("DS4_SPEX_TRACE_ROUTING")),
                        "trace_weights": clean(env.get("DS4_SPEX_TRACE_ROUTING_WEIGHTS")),
                        "trace_residency": clean(env.get("DS4_SPEX_TRACE_ROUTING_RESIDENCY")),
                        "expert_tiering": clean(env.get("DS4_EXPERT_TIERING")),
                        "tiering_log_ids": clean(env.get("DS4_EXPERT_TIERING_LOG_IDS")),
                        "pace_warmup": clean(env.get("DS4_PACE_WARMUP")),
                        "pace_keep": clean(env.get("DS4_PACE_KEEP")),
                        "pace_keep_min": clean(env.get("DS4_PACE_KEEP_MIN")),
                        "pace_keep_max": clean(env.get("DS4_PACE_KEEP_MAX")),
                        "pace_keep_step": clean(env.get("DS4_PACE_KEEP_STEP")),
                        "pace_breath_every": clean(env.get("DS4_PACE_BREATH_EVERY")),
                        "pace_breath_keep": clean(env.get("DS4_PACE_BREATH_KEEP")),
                        "pace_breath_len": clean(env.get("DS4_PACE_BREATH_LEN")),
                        "pace_prebreath": clean(env.get("DS4_PACE_PREBREATH")),
                        "pace_prebreath_every": clean(env.get("DS4_PACE_PREBREATH_EVERY")),
                        "pace_prebreath_keep_max": clean(env.get("DS4_PACE_PREBREATH_KEEP_MAX")),
                        "pace_relearn": clean(env.get("DS4_PACE_RELEARN")),
                        "pace_relearn_on_tighten": clean(env.get("DS4_PACE_RELEARN_ON_TIGHTEN")),
                        "pace_rotate": clean(env.get("DS4_PACE_ROTATE")),
                        "pace_rotate_every": clean(env.get("DS4_PACE_ROTATE_EVERY")),
                        "pace_wrap": clean(env.get("DS4_PACE_WRAP")),
                        "pace_wrap_rotate": clean(env.get("DS4_PACE_WRAP_ROTATE")),
                        "hidden_spex": clean(
                            {
                                "prefetch": env.get("DS4_SPEX_HIDDEN_PREFETCH"),
                                "gpu_load": env.get("DS4_SPEX_HIDDEN_GPU_LOAD"),
                                "gpu_score": env.get("DS4_SPEX_HIDDEN_GPU_SCORE"),
                                "gpu_prefetch": env.get("DS4_SPEX_HIDDEN_GPU_PREFETCH"),
                            }
                        ),
                        "completion_tokens": clean(src.get("completion_tokens") or usage.get("completion_tokens")),
                        "wall_s": clean(src.get("wall_s")),
                        "prompt_s": clean(src.get("prompt_s")),
                        "finish_s": clean(src.get("finish_s")),
                        "first50_tps": clean(src.get("first50_tps")),
                        "avg_tps": clean(src.get("avg_tps")),
                        "last_chunk_tps": clean(src.get("last_chunk_tps") or src.get("last_tps")),
                        "prefetch_count": clean(src.get("prefetch_count")),
                        "prefetch_gib": clean(src.get("prefetch_gib")),
                        "prefetch_ms": clean(src.get("prefetch_ms")),
                        "trace_rows": clean(src.get("trace_rows")),
                        "trace_bytes": clean(src.get("trace_bytes")),
                        "content_chars": clean(src.get("content_chars") or len(content)),
                        "doctype": clean(src.get("doctype") or content.count("<!DOCTYPE")),
                        "html_balance": clean(src.get("html_balance")),
                        "has_form": clean(1 if "<form" in content.lower() else 0),
                        "has_script": clean(1 if "<script" in content.lower() else 0),
                        "has_popup": clean(src.get("has_popup") or (1 if "alert(" in content or "richiesta inviata" in content.lower() else 0)),
                        "repeat_flag": clean(src.get("repeat_flag")),
                        "l0l3": clean(src.get("l0l3")),
                        "client_stop_enabled": clean(src.get("client_stop_enabled")),
                        "first_client_stop_reason": clean(src.get("first_client_stop_reason")),
                        "client_stop_reason": clean(src.get("client_stop_reason")),
                        "retry_attempts": clean(src.get("retry_attempts")),
                        "final_attempt_phase": clean(src.get("final_attempt_phase")),
                        "coherent_until_token_est": clean(quality_notes.get("coherent_until_token_est")),
                        "result_text": clean(quality_notes.get("result_text") or quality_notes.get("failure_mode")),
                        "verdict": clean(quality_notes.get("verdict")),
                        "pace_events": clean(
                            {
                                "learned": src.get("pace_learned"),
                                "descents": src.get("pace_descents"),
                                "prebreaths": src.get("pace_prebreaths"),
                                "breaths": src.get("pace_breaths"),
                                "breath_ends": src.get("pace_breath_ends"),
                                "tightens": src.get("pace_tightens"),
                                "tighten_relearns": src.get("pace_tighten_relearns"),
                                "rotates": src.get("pace_rotates"),
                                "first_descent_tok": src.get("first_descent_tok"),
                                "first_prebreath_tok": src.get("first_prebreath_tok"),
                                "first_breath_tok": src.get("first_breath_tok"),
                                "first_rotate_tok": src.get("first_rotate_tok"),
                            }
                        ),
                        "source_artifacts": rel(summary),
                        "env_effective_json": clean(env),
                        "request_settings_json": clean(request),
                    }
                )
                if manifest.get("server", {}).get("cache_experts_default"):
                    row["setup_text"] = f"cache_default={manifest['server']['cache_experts_default']}; env_delta={clean(manifest.get('env_delta_from_profile'))}"
                tier = re.search(r"misses=(\d+).*?evictions=(\d+)", log)
                if tier:
                    row["tier_miss_eviction"] = f"{tier.group(1)}/{tier.group(2)}"
                cache_key = clean(row["server_cache_experts"])
                rk = requested_key(row["variant"])
                if "requested4" in suite and (cache_key, rk) in coherence:
                    row["coherent_until_token_est"], row["useful_tokens_after_return_est"] = coherence[(cache_key, rk)]
                row["quality_signal"] = derive_quality(row)
                rows.append(row)
    return rows


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def decimal(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def gib(value: Any) -> str:
    numeric = number(value)
    return f"{numeric / (1024 ** 3):.6f}" if numeric is not None else ""


def windows_prompt_name(tag: str, prompt: str) -> str:
    lower = f"{tag} {prompt}".lower()
    if "cyber" in lower or "landing page" in lower:
        return "cyber_html"
    if "hi9" in lower or "hi12" in lower or prompt.strip().lower().startswith("hi"):
        return "hi"
    return "windows_g7"


def parse_windows_g7_rows(repo: Path) -> list[dict[str, str]]:
    root = repo / "runs/ds4/20260714_windows_native_g7/results"
    extra_roots = [repo / "runs/ds4/20260714_windows_native_g27/results"]
    rows: list[dict[str, str]] = []
    result_paths = sorted(root.glob("*_result.json"))
    for extra_root in extra_roots:
        if extra_root.exists():
            result_paths.extend(sorted(extra_root.glob("*_result.json")))
    for path in result_paths:
        src = read_json(path)
        if not src or not src.get("tag"):
            continue

        tag = clean(src.get("tag"))
        artifact_id = path.stem.removesuffix("_result")
        results = src.get("results") if isinstance(src.get("results"), list) else []
        server_runs = src.get("server_runs") if isinstance(src.get("server_runs"), list) else []
        repeats = int(number(src.get("repeats")) or len(results) or 0)
        warmup = bool(src.get("warmup"))
        output_hashes = sorted(
            {
                clean(item.get("content_sha256"))
                for item in results
                if isinstance(item, dict) and item.get("content_sha256")
            }
        )
        expected_hash = clean(src.get("expected_content_sha256")).lower()
        expected_match = bool(expected_hash and output_hashes) and all(
            value.lower() == expected_hash for value in output_hashes
        )
        outputs_identical = len(output_hashes) == 1
        worktree_dirty = bool(src.get("worktree_dirty"))
        arena_fatal_value = number(src.get("dynamic_arena_final_fatal"))
        arena_fatal = int(arena_fatal_value) if arena_fatal_value is not None else None

        if repeats >= 3:
            evidence = "measured_same_process_n3"
            benchmark_usable = (
                "speed_only"
                if outputs_identical and arena_fatal in (None, 0)
                else "diagnostic_only"
            )
            replication_scope = "same_server_process"
        else:
            evidence = f"measured_safety_n{max(repeats, 1)}"
            benchmark_usable = "mechanism_only"
            replication_scope = "single_server_process"

        preflight = as_dict(src.get("memory_preflight"))
        preflight_before = as_dict(preflight.get("before"))
        purge = as_dict(preflight.get("standby_purge"))
        if warmup:
            cache_state = "same_process_primed"
        elif purge.get("status") == "completed":
            cache_state = "new_process_standby_purged"
        else:
            cache_state = "new_process_uncontrolled"

        started = clean(preflight.get("started_utc"))
        date = (
            started[:10].replace("-", "")
            if started
            else WINDOWS_G7_SNAPSHOT_DATE
        )
        prompt = clean(src.get("prompt"))
        prompt_hash = clean(src.get("prompt_sha256")) or short_hash(prompt)
        gpu = as_dict(src.get("gpu_identity"))
        runtime = as_dict(src.get("runtime_telemetry"))
        env = as_dict(src.get("effective_ds4_environment"))
        result_tps = [
            float(value)
            for value in (number(item.get("tokens_per_second")) for item in results if isinstance(item, dict))
            if value is not None
        ]
        result_seconds = [
            float(value)
            for value in (number(item.get("seconds")) for item in results if isinstance(item, dict))
            if value is not None
        ]
        completion_tokens = [
            int(value)
            for value in (number(item.get("completion_tokens")) for item in results if isinstance(item, dict))
            if value is not None
        ]
        first_content = clean(results[0].get("content")) if results and isinstance(results[0], dict) else ""
        server_mean = number(src.get("server_decode_mean_tokens_per_second"))
        client_mean = number(src.get("mean_tokens_per_second"))
        arena_resident_bytes = number(src.get("dynamic_arena_observer_resident_bytes"))
        arena_resident_gib = (
            f"{arena_resident_bytes / (1024 ** 3):.6f}" if arena_resident_bytes is not None else ""
        )
        arena_gib = number(src.get("dynamic_arena_gib_requested"))
        arena_window_value = number(src.get("dynamic_arena_observed_window_requested"))
        arena_window = int(arena_window_value) if arena_window_value is not None else None
        arena_min_hits_value = number(src.get("dynamic_arena_observed_min_hits_requested"))
        arena_min_hits = int(arena_min_hits_value) if arena_min_hits_value is not None else None
        profile_parts: list[str] = []
        if arena_gib is not None:
            profile_parts.append(f"arena={arena_gib:g}GiB")
        if arena_window is not None:
            profile_parts.append(f"W={arena_window}")
        if arena_min_hits is not None:
            profile_parts.append(f"min_hits={arena_min_hits}")
        carry = clean(src.get("dynamic_arena_carry_requested"))
        if carry:
            profile_parts.append(f"carry={carry}")
        expert_cache = number(src.get("expert_cache_requested"))
        if expert_cache is not None:
            profile_parts.append(f"expert_cache={int(expert_cache)}")
        spex_stage = clean(src.get("spex_stage_requested"))
        if spex_stage:
            profile_parts.append(f"spex={spex_stage}")
        profile = "; ".join(profile_parts)
        exact_text = "true" if expected_match else ("false" if expected_hash else "not_requested")

        row = base_row()
        row.update(
            {
                "row_id": f"WIN-G7-{artifact_id}",
                "source_kind": "windows_g7_result",
                "evidence_level": evidence,
                "benchmark_usable": benchmark_usable,
                "date": date,
                "suite": path.parent.parent.name,
                "run_id": artifact_id,
                "category": "windows_native_performance",
                "experiment": tag,
                "variant": tag,
                "profile": profile,
                "prompt_name": windows_prompt_name(tag, prompt),
                "prompt_sha16": prompt_hash[:16],
                "prompt_chars": str(len(prompt)),
                "prompt_excerpt": prompt[:160],
                "hardware": "native Windows / WDDM",
                "gpu": clean(gpu.get("name")),
                "model": Path(clean(src.get("model"))).name if src.get("model") else "",
                "model_bytes": clean(src.get("model_bytes")),
                "ds4_base": "hawkli-1994/ds4-win",
                "ds4_build": clean(src.get("head")),
                "runner_commit": clean(src.get("head")),
                "ctx": clean(src.get("context_observed") or src.get("context_requested")),
                "prefill_chunk": clean(src.get("prefill_chunk_observed")),
                "server_cache_experts": clean(src.get("expert_cache_requested")),
                "request_max_tokens": clean(src.get("requested_max_tokens")),
                "server_max_tokens": clean(src.get("requested_max_tokens")),
                "temperature": "0",
                "stream": "0",
                "think": "0",
                "hidden_spex": clean(src.get("spex_dry_run_requested")),
                "completion_tokens": clean(completion_tokens[0] if completion_tokens and len(set(completion_tokens)) == 1 else mean([float(v) for v in completion_tokens])),
                "wall_s": clean(mean(result_seconds)),
                "prompt_s": clean(src.get("server_prefill_ttft_mean_seconds")),
                "avg_tps": clean(client_mean),
                "content_chars": str(len(first_content)) if first_content else "",
                "doctype": str(first_content.lower().count("<!doctype")) if first_content else "",
                "html_balance": clean(first_content.lower().count("<html") - first_content.lower().count("</html>")) if first_content else "",
                "has_form": str("<form" in first_content.lower()) if first_content else "",
                "has_script": str("<script" in first_content.lower()) if first_content else "",
                "has_popup": str("popup" in first_content.lower()) if first_content else "",
                "quality_signal": f"exact_hash={exact_text}; outputs_identical={str(outputs_identical).lower()}; L0-L3=not_graded",
                "metrics_text": f"server={clean(server_mean)} t/s; client={clean(client_mean)} t/s; TTFT={clean(src.get('server_prefill_ttft_mean_seconds'))}s",
                "setup_text": profile,
                "env_effective_json": clean(env),
                "request_settings_json": clean({
                    "context": src.get("context_requested"),
                    "max_tokens": src.get("requested_max_tokens"),
                    "temperature": 0,
                    "think": False,
                    "warmup": warmup,
                    "repeats": repeats,
                }),
                "runtime_platform": "windows_native_wddm",
                "cache_state": cache_state,
                "source_head": clean(src.get("head")),
                "executable_sha256": clean(src.get("executable_sha256")),
                "harness_sha256": clean(src.get("harness_sha256")),
                "ds4_cuda_sha256": clean(src.get("ds4_cuda_sha256")),
                "repeats": str(repeats),
                "replication_scope": replication_scope,
                "warmup": str(warmup).lower(),
                "server_decode_mean_tps": clean(server_mean),
                "server_decode_min_tps": clean(src.get("server_decode_min_tokens_per_second")),
                "server_decode_max_tps": clean(src.get("server_decode_max_tokens_per_second")),
                "client_completion_mean_tps": clean(client_mean),
                "client_completion_min_tps": clean(src.get("min_tokens_per_second")),
                "client_completion_max_tps": clean(src.get("max_tokens_per_second")),
                "server_prefill_ttft_s": clean(src.get("server_prefill_ttft_mean_seconds")),
                "outputs_identical": str(outputs_identical).lower(),
                "expected_hash_match": exact_text,
                "output_sha256": ";".join(output_hashes),
                "dynamic_arena_gib": clean(arena_gib),
                "dynamic_arena_window": clean(arena_window),
                "dynamic_arena_min_hits": clean(arena_min_hits),
                "dynamic_arena_grow_interval": clean(src.get("dynamic_arena_grow_interval_requested")),
                "dynamic_arena_carry": carry,
                "dynamic_arena_resident": clean(src.get("dynamic_arena_observer_resident")),
                "dynamic_arena_resident_gib": arena_resident_gib,
                "dynamic_arena_hit_rate": clean(src.get("dynamic_arena_hit_rate")),
                "dynamic_arena_final_hits": clean(src.get("dynamic_arena_final_hits")),
                "dynamic_arena_final_misses": clean(src.get("dynamic_arena_final_misses")),
                "dynamic_arena_fatal": clean(arena_fatal),
                "prefill_mass_wrap_result": clean(src.get("prefill_mass_wrap_result")),
                "prefill_mass_wrap_reason": clean(src.get("prefill_mass_wrap_reason")),
                "prefill_mass_wrap_candidate_entries": clean(src.get("prefill_mass_wrap_candidate_entries")),
                "prefill_mass_wrap_loads": clean(src.get("prefill_mass_wrap_loads")),
                "prefill_mass_wrap_workers": clean(src.get("prefill_mass_wrap_workers")),
                "prefill_mass_wrap_seconds": clean(src.get("prefill_mass_wrap_seconds")),
                "prefill_mass_wrap_snapshot_before": clean(src.get("prefill_mass_wrap_snapshot_before")),
                "prefill_mass_wrap_snapshot_after": clean(src.get("prefill_mass_wrap_snapshot_after")),
                "prefill_mass_wrap_resident_before": clean(src.get("prefill_mass_wrap_resident_before")),
                "prefill_mass_wrap_resident_after": clean(src.get("prefill_mass_wrap_resident_after")),
                "prefill_mass_decode_candidate_hits": clean(src.get("prefill_mass_decode_candidate_hits")),
                "reap_mass_wrap_publications": clean(src.get("reap_mass_wrap_publication_count")),
                "reap_mass_wrap_failures": clean(src.get("reap_mass_wrap_failure_count")),
                "reap_mass_wrap_entrants": clean(src.get("reap_mass_wrap_sum_entrants")),
                "reap_mass_wrap_victims": clean(src.get("reap_mass_wrap_sum_victims")),
                "reap_mass_wrap_seconds": clean(src.get("reap_mass_wrap_sum_seconds")),
                "reap_mass_wrap_capacity_entries": clean(src.get("reap_mass_wrap_capacity_entries")),
                "reap_mass_wrap_last_result": clean(src.get("reap_mass_wrap_last_result")),
                "reap_mass_wrap_last_reason": clean(src.get("reap_mass_wrap_last_reason")),
                "reap_mass_wrap_last_snapshot_after": clean(src.get("reap_mass_wrap_last_snapshot_after")),
                "q8_f16_cache_mib": clean(src.get("q8_f16_cache_mb_requested")),
                "q8_f16_reserve_mib": clean(src.get("q8_f16_cache_reserve_mb_requested")),
                "moe_io_qd": clean(src.get("moe_io_queue_depth_observed") or src.get("moe_io_queue_depth")),
                "resident_expert_cache": clean(src.get("expert_cache_requested")),
                "spex_stage": clean(src.get("spex_stage_observed") or src.get("spex_stage_requested")),
                "spex_cap": clean(src.get("spex_cap_observed") or src.get("spex_cap_requested")),
                "spex_recall": clean(src.get("spex_recall")),
                "standby_before_gib": gib(preflight_before.get("standby_bytes")),
                "wddm_shared_peak_gib": gib(runtime.get("gpu_process_shared_peak_bytes")),
                "wddm_dedicated_peak_gib": gib(runtime.get("gpu_process_dedicated_peak_bytes")),
                "process_read_gib": gib(runtime.get("win32_process_read_transfer_delta_bytes")),
                "gpu_util_median": clean(runtime.get("gpu_utilization_median_percent")),
                "vram_peak_mib": clean(runtime.get("vram_used_peak_mib")),
                "result_text": f"server={clean(server_mean)} t/s; client={clean(client_mean)} t/s; repeats={repeats}; exact={exact_text}; dirty={str(worktree_dirty).lower()}",
                "source_artifacts": rel(path),
            }
        )
        if "g7_g27_" in artifact_id:
            row["patches"] = (
                "ds4-win evidence commit "
                "c4bb45de31d122a5f1e7b7e11bfbf18ec242dffe; "
                f"base_commit={row['source_head']}"
            )
            row["result_text"] += (
                "; mechanism correct; 16-token run pays bootstrap; "
                "no long-run amortization verdict"
            )
        rows.append(row)
    if len(rows) != len(result_paths):
        raise RuntimeError(
            f"Windows G7 import lost artifacts: parsed {len(rows)} of {len(result_paths)}"
        )
    return rows + aggregate_windows_g7_campaigns(root, rows)


def parse_quality_full_decode_mass_weight_rows(repo: Path) -> list[dict[str, str]]:
    path = (
        repo
        / "runs/ds4/20260712_pod12_bake/"
        "quality_full_decode_mass_weight_results_20260715.csv"
    )
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for src in reader:
            suite_root = clean(src.get("suite/root"))
            suite_path = Path(suite_root) if suite_root else path.parent
            suite = rel(suite_path) if suite_root else rel(path.parent)
            suite_leaf = suite_path.name if suite_root else path.parent.name
            arm = clean(src.get("arm"))
            run = clean(src.get("run"))
            temperature = clean(src.get("temperature"))
            temperature_id = temperature.replace(".", "p")
            grade = clean(src.get("grade"))
            finish_reason = clean(src.get("finish_reason"))
            mask_sha = clean(src.get("mask_sha256"))
            prompt_sha = clean(src.get("prompt_sha256"))
            runner_sha = clean(src.get("runner_sha256"))
            binary_sha = clean(src.get("binary_sha256"))

            row = base_row()
            row.update(
                {
                    "row_id": (
                        f"QUALITY-FULL-DECODE-MASS-WEIGHT-{suite_leaf}-"
                        f"{arm}-run{run}-t{temperature_id}"
                    ),
                    "source_kind": "quality_full_decode_mass_weight_csv",
                    "evidence_level": "measured_robustness_temperature_condition",
                    "benchmark_usable": "linux_oracle_quality_speed_diagnostic",
                    "date": "20260715",
                    "suite": suite,
                    "run_id": f"{suite_leaf}/{arm}/run{run}/temp{temperature}",
                    "category": "linux_pod_oracle_quality_full_decode",
                    "experiment": "pod_full_decode_mass_weight_quality",
                    "variant": arm,
                    "profile": (
                        f"arm={arm}; run={run}; temperature={temperature}; "
                        "temperature is a robustness condition, not iid replication"
                    ),
                    "prompt_name": "software_build_dashboard_html",
                    "prompt_sha16": prompt_sha[:16],
                    "prompt_chars": clean(src.get("prompt_chars")),
                    "hardware": "Linux pod oracle",
                    "gpu": clean(src.get("GPU")),
                    "model": clean(src.get("model")),
                    "ctx": clean(src.get("ctx")),
                    "prefill_chunk": clean(src.get("prefill_chunk")),
                    "server_cache_experts": clean(src.get("cache_experts")),
                    "request_max_tokens": clean(src.get("request_max_tokens")),
                    "server_max_tokens": clean(src.get("server_max_tokens")),
                    "temperature": temperature,
                    "stream": clean(src.get("stream")),
                    "think": clean(src.get("think")),
                    "completion_tokens": clean(src.get("completion_tokens")),
                    "wall_s": clean(src.get("elapsed_s")),
                    "avg_tps": clean(src.get("tok/s")),
                    "client_completion_mean_tps": clean(src.get("tok/s")),
                    "content_chars": clean(src.get("content_chars")),
                    "l0l3": grade,
                    "client_stop_reason": (
                        finish_reason if finish_reason.startswith("client_stop") else ""
                    ),
                    "finish_s": clean(src.get("elapsed_s")),
                    "trace_rows": clean(src.get("stream_events")),
                    "quality_signal": (
                        f"{grade}; finish={finish_reason}; "
                        "temperature_condition=robustness_not_iid"
                    ),
                    "metrics_text": (
                        f"elapsed_s={clean(src.get('elapsed_s'))}; "
                        f"stream_events={clean(src.get('stream_events'))}; "
                        f"tok/s={clean(src.get('tok/s'))}"
                    ),
                    "setup_text": (
                        f"suite/root={suite_root}; prompt_sha256={prompt_sha}; "
                        f"runner_sha256={runner_sha}; binary_sha256={binary_sha}; "
                        f"mask_sha256={mask_sha}"
                    ),
                    "request_settings_json": clean(src),
                    "runtime_platform": "linux_pod_oracle",
                    "cache_state": "not_reported",
                    "executable_sha256": binary_sha,
                    "harness_sha256": runner_sha,
                    "repeats": "1",
                    "replication_scope": "temperature_robustness_condition_not_iid",
                    "warmup": "not_reported",
                    "result_text": (
                        f"{grade}; finish={finish_reason}; "
                        f"tok/s={clean(src.get('tok/s'))}; "
                        f"temperature={temperature}; robustness condition"
                    ),
                    "verdict": grade,
                    "source_artifacts": rel(path),
                }
            )
            if mask_sha:
                row["patches"] = f"mask_sha256={mask_sha}"
            rows.append(row)

    if len(rows) != 24:
        raise RuntimeError(
            f"Quality full-decode mass/weight CSV expected 24 rows, found {len(rows)}"
        )
    return rows


def aggregate_windows_g7_campaigns(
    root: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    campaigns = [
        {
            "name": "g19a_observed_residency",
            "pattern": re.compile(r"^g7_g19a_ab_arena12_[1-6]_(off|on)_n1$"),
            "arms": ("off", "on"),
            "cache_note": "independent new processes; no shared warmup",
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
        },
        {
            "name": "g19b_first_fetch_preload",
            "pattern": re.compile(r"^g7_g19b_ab128_arena12_[1-6]_(off|on)_n1$"),
            "arms": ("off", "on"),
            "cache_note": "independent new processes; no shared warmup",
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
        },
        {
            "name": "g20_grow8",
            "pattern": re.compile(r"^g7_g20_grow_ab_20260714_085148_[1-6]_(off|on)_n1$"),
            "arms": ("off", "on"),
            "cache_note": "independent new processes; warm unpurged standby state",
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
        },
        {
            "name": "g25_prefill_mass_bulk_wrap",
            "pattern": re.compile(r"^g7_g25_prefill_mass_(observe14_control|wrap14)(?:_safety)?_n[123]$"),
            "arms": ("observe14_control", "wrap14"),
            "cache_note": (
                "independent processes; 14 GiB prefill-mass A/B; 2 GiB safety run excluded"
            ),
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
            "source_artifact": "reports/G25_PREFILL_MASS_BULK_WRAP_RESULTS.md",
        },
        {
            "name": "g26_reap_mass_observer",
            "pattern": re.compile(
                r"^(g7_g26_reap_mass_observe_prefillfix_safety_n1|"
                r"g7_g26_reap_mass_ab_(?:on_n[23]|off_n[123]))$"
            ),
            "arms": ("on", "off"),
            "arm_by_run_id": {
                "g7_g26_reap_mass_observe_prefillfix_safety_n1": "on",
                "g7_g26_reap_mass_ab_on_n2": "on",
                "g7_g26_reap_mass_ab_on_n3": "on",
                "g7_g26_reap_mass_ab_off_n1": "off",
                "g7_g26_reap_mass_ab_off_n2": "off",
                "g7_g26_reap_mass_ab_off_n3": "off",
            },
            "cache_note": (
                "independent processes; REAP mass observe-only CPU readback A/B; "
                "prefill-fixed safety run is ON sample 1"
            ),
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
            "source_artifact": "reports/G26_REAP_MASS_OBSERVE_RESULTS.md",
        },
        {
            "name": "g26b_packed_router_trace",
            "pattern": re.compile(r"^g7_g26b_valid_counter_[1-3]_(on|off)$"),
            "arms": ("on", "off"),
            "cache_note": (
                "independent counterbalanced processes; packed router D2H trace A/B"
            ),
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
            "source_artifact": "reports/G26B_REAP_PACKED_TRACE_RESULTS.md",
        },
        {
            "name": "g27_reap_mass_wrap",
            "pattern": re.compile(r"^g7_g27_final_counter_[1-3]_(on|off)$"),
            "arms": ("on", "off"),
            "cache_note": (
                "independent counterbalanced processes; G27 REAP mass WRAP actuator "
                "versus packed observe-only; 16-token gate pays bootstrap and is not "
                "a long-run verdict"
            ),
            "cache_state": "independent_new_processes_uncontrolled",
            "warmup": "false",
            "source_artifact": "../20260714_windows_native_g27/reports/G27_REAP_MASS_WRAP_RESULTS.md",
        },
        {
            "name": "g22_arena_carry",
            "pattern": re.compile(r"^g7_g22_carry_ab_20260714_[1-6]_(drop|keep)_n1$"),
            "arms": ("drop", "keep"),
            "cache_note": (
                "independent processes; request 1 primes the arena; request 2 measured"
            ),
            "cache_state": "independent_processes_same_process_primed_request2",
            "warmup": "true",
            "aggregate_file": "g7_g22_carry_ab_20260714_aggregate.json",
            "aggregate_schema": "g7_dynamic_arena_carry_ab_v1",
        },
    ]
    aggregates: list[dict[str, str]] = []
    for spec in campaigns:
        campaign = str(spec["name"])
        pattern = spec["pattern"]
        arms = tuple(spec["arms"])
        cache_note = str(spec["cache_note"])
        matched: list[tuple[dict[str, str], str]] = []
        for row in rows:
            match = pattern.match(row["run_id"])
            if match:
                arm_by_run_id = spec.get("arm_by_run_id")
                if isinstance(arm_by_run_id, dict):
                    arm = arm_by_run_id.get(row["run_id"])
                else:
                    arm = match.group(1)
                if arm not in arms:
                    raise RuntimeError(
                        f"Campaign {campaign} could not classify arm for {row['run_id']}"
                    )
                matched.append((row, str(arm)))
        if len(matched) != 6:
            raise RuntimeError(f"Campaign {campaign} expected 6 runs, found {len(matched)}")

        campaign_artifact = ""
        source_artifact = spec.get("source_artifact")
        if source_artifact:
            source_artifact_path = root.parent / str(source_artifact)
            if not source_artifact_path.exists():
                raise RuntimeError(
                    f"Campaign {campaign} source artifact missing: {source_artifact_path}"
                )
            campaign_artifact = rel(source_artifact_path)
        aggregate_name = spec.get("aggregate_file")
        if aggregate_name:
            aggregate_path = root / str(aggregate_name)
            aggregate_src = read_json(aggregate_path)
            if not aggregate_src:
                raise RuntimeError(f"Campaign {campaign} aggregate missing: {aggregate_path}")
            if aggregate_src.get("schema") != spec.get("aggregate_schema"):
                raise RuntimeError(f"Campaign {campaign} aggregate schema mismatch")
            aggregate_runs = aggregate_src.get("runs")
            if not isinstance(aggregate_runs, list) or len(aggregate_runs) != 6:
                raise RuntimeError(f"Campaign {campaign} aggregate expected 6 runs")
            aggregate_ids = {
                f"g7_{clean(item.get('tag'))}"
                for item in aggregate_runs
                if isinstance(item, dict) and item.get("tag")
            }
            matched_ids = {row["run_id"] for row, _ in matched}
            if aggregate_ids != matched_ids:
                raise RuntimeError(f"Campaign {campaign} aggregate/result run mismatch")
            promotion = as_dict(aggregate_src.get("promotion_gate"))
            if promotion.get("all_hashes_exact") is not True:
                raise RuntimeError(f"Campaign {campaign} aggregate exactness gate failed")
            campaign_artifact = rel(aggregate_path)

        for arm in arms:
            arm_rows = [row for row, observed_arm in matched if observed_arm == arm]
            if len(arm_rows) != 3:
                raise RuntimeError(
                    f"Campaign {campaign}/{arm} expected 3 runs, found {len(arm_rows)}"
                )
            server_values = [
                value
                for value in (number(row["server_decode_mean_tps"]) for row in arm_rows)
                if value is not None
            ]
            client_values = [
                value
                for value in (number(row["client_completion_mean_tps"]) for row in arm_rows)
                if value is not None
            ]
            ttft_values = [
                value
                for value in (number(row["server_prefill_ttft_s"]) for row in arm_rows)
                if value is not None
            ]
            wall_values = [
                value
                for value in (number(row["wall_s"]) for row in arm_rows)
                if value is not None
            ]
            wrap_seconds_values = [
                value
                for value in (
                    number(row["prefill_mass_wrap_seconds"]) for row in arm_rows
                )
                if value is not None
            ]
            hashes = sorted({row["output_sha256"] for row in arm_rows if row["output_sha256"]})
            expected_statuses = {row["expected_hash_match"] for row in arm_rows}
            if expected_statuses == {"true"}:
                expected_status = "true"
            elif "false" in expected_statuses:
                expected_status = "false"
            else:
                expected_status = "not_requested"
            outputs_identical = len(hashes) == 1
            if aggregate_name:
                sample_key = f"{arm}_decode_tps_samples"
                aggregate_samples = [
                    float(value)
                    for value in aggregate_src.get(sample_key, [])
                    if number(value) is not None
                ]
                if sorted(round(value, 6) for value in aggregate_samples) != sorted(
                    round(value, 6) for value in server_values
                ):
                    raise RuntimeError(
                        f"Campaign {campaign}/{arm} aggregate sample mismatch"
                    )
            base = base_row()
            common_fields = [
                "date",
                "suite",
                "prompt_name",
                "prompt_sha16",
                "prompt_chars",
                "prompt_excerpt",
                "system_prompt",
                "hardware",
                "gpu",
                "model",
                "model_bytes",
                "ds4_base",
                "ds4_build",
                "patches",
                "reap_loop_commit",
                "runner_commit",
                "ctx",
                "prefill_chunk",
                "server_cache_experts",
                "request_max_tokens",
                "server_max_tokens",
                "temperature",
                "stream",
                "think",
                "hidden_spex",
                "completion_tokens",
                "content_chars",
                "doctype",
                "html_balance",
                "has_form",
                "has_script",
                "has_popup",
                "env_effective_json",
                "request_settings_json",
                "runtime_platform",
                "source_head",
                "executable_sha256",
                "harness_sha256",
                "ds4_cuda_sha256",
                "dynamic_arena_gib",
                "dynamic_arena_window",
                "dynamic_arena_min_hits",
                "dynamic_arena_grow_interval",
                "dynamic_arena_carry",
                "dynamic_arena_resident",
                "dynamic_arena_resident_gib",
                "dynamic_arena_hit_rate",
                "dynamic_arena_final_hits",
                "dynamic_arena_final_misses",
                "dynamic_arena_fatal",
                "prefill_mass_wrap_result",
                "prefill_mass_wrap_reason",
                "prefill_mass_wrap_candidate_entries",
                "prefill_mass_wrap_loads",
                "prefill_mass_wrap_workers",
                "prefill_mass_wrap_seconds",
                "prefill_mass_wrap_snapshot_before",
                "prefill_mass_wrap_snapshot_after",
                "prefill_mass_wrap_resident_before",
                "prefill_mass_wrap_resident_after",
                "prefill_mass_decode_candidate_hits",
                "reap_mass_wrap_publications",
                "reap_mass_wrap_failures",
                "reap_mass_wrap_entrants",
                "reap_mass_wrap_victims",
                "reap_mass_wrap_seconds",
                "reap_mass_wrap_capacity_entries",
                "reap_mass_wrap_last_result",
                "reap_mass_wrap_last_reason",
                "reap_mass_wrap_last_snapshot_after",
                "q8_f16_cache_mib",
                "q8_f16_reserve_mib",
                "moe_io_qd",
                "resident_expert_cache",
                "spex_stage",
                "spex_cap",
                "spex_recall",
            ]
            for field in common_fields:
                values = {row[field] for row in arm_rows}
                if len(values) == 1:
                    base[field] = values.pop()
            base.update(
                {
                    "row_id": f"WIN-G7-CAMPAIGN-{campaign}-{arm}",
                    "source_kind": "windows_g7_campaign_aggregate",
                    "evidence_level": "measured_independent_n3",
                    "benchmark_usable": "speed_only",
                    "run_id": f"campaign_{campaign}_{arm}_n3",
                    "category": "windows_native_campaign",
                    "experiment": campaign,
                    "variant": arm,
                    "profile": f"{campaign} arm={arm}; {cache_note}",
                    "repeats": "3",
                    "replication_scope": "independent_server_processes",
                    "warmup": str(spec["warmup"]),
                    "cache_state": str(spec["cache_state"]),
                    "avg_tps": decimal(mean(client_values)),
                    "wall_s": decimal(mean(wall_values)),
                    "prompt_s": decimal(mean(ttft_values)),
                    "server_decode_mean_tps": decimal(mean(server_values)),
                    "server_decode_min_tps": decimal(min(server_values) if server_values else None),
                    "server_decode_max_tps": decimal(max(server_values) if server_values else None),
                    "client_completion_mean_tps": decimal(mean(client_values)),
                    "client_completion_min_tps": decimal(min(client_values) if client_values else None),
                    "client_completion_max_tps": decimal(max(client_values) if client_values else None),
                    "server_prefill_ttft_s": decimal(mean(ttft_values)),
                    "outputs_identical": str(outputs_identical).lower(),
                    "expected_hash_match": expected_status,
                    "output_sha256": ";".join(hashes),
                    "quality_signal": (
                        f"expected_hash={expected_status}; outputs_identical={str(outputs_identical).lower()}; "
                        "independent_processes=3; L0-L3=not_graded"
                    ),
                    "metrics_text": (
                        f"server mean/median={decimal(mean(server_values))}/{decimal(median(server_values))} t/s; "
                        f"client mean={decimal(mean(client_values))} t/s"
                    ),
                    "setup_text": cache_note,
                    "result_text": (
                        f"independent n=3 arm={arm}; server mean={decimal(mean(server_values))} t/s; "
                        f"median={decimal(median(server_values))}; expected_hash={expected_status}; "
                        f"outputs_identical={str(outputs_identical).lower()}"
                    ),
                    "source_artifacts": ";".join(
                        [row["source_artifacts"] for row in arm_rows]
                        + ([campaign_artifact] if campaign_artifact else [])
                    ),
                }
            )
            if campaign == "g25_prefill_mass_bulk_wrap":
                wrap_seconds_mean = decimal(mean(wrap_seconds_values))
                base["prefill_mass_wrap_seconds"] = wrap_seconds_mean
                base["metrics_text"] += (
                    f"; prefill-mass WRAP mean={wrap_seconds_mean}s"
                )
                base["result_text"] += (
                    f"; prefill-mass WRAP mean={wrap_seconds_mean}s"
                )
            if campaign == "g27_reap_mass_wrap":
                reap_wrap_seconds_values = [
                    value
                    for value in (
                        number(row["reap_mass_wrap_seconds"]) for row in arm_rows
                    )
                    if value is not None
                ]
                reap_wrap_seconds_mean = decimal(mean(reap_wrap_seconds_values))
                base["reap_mass_wrap_seconds"] = reap_wrap_seconds_mean
                base["result_text"] += (
                    f"; REAP mass WRAP mean={reap_wrap_seconds_mean}s; "
                    "mechanism correct; 16-token run pays bootstrap; no long-run verdict"
                )
            aggregates.append(base)
    return aggregates


def parse_windows_g7_failure_rows(repo: Path) -> list[dict[str, str]]:
    root = repo / "runs/ds4/20260714_windows_native_g7/results"
    paths = sorted(root.glob("*_failure.json"))
    rows: list[dict[str, str]] = []
    for path in paths:
        src = read_json(path)
        if not src or not src.get("tag"):
            continue
        tag = clean(src.get("tag"))
        parameters = as_dict(src.get("parameters"))
        measured = as_dict(src.get("measured"))
        reasons = src.get("failure_reasons") if isinstance(src.get("failure_reasons"), list) else []
        reason_set = {clean(reason) for reason in reasons}
        expected_hash = clean(src.get("expected_content_sha256"))
        actual_hash = clean(src.get("actual_content_sha256"))
        if "output_hash_mismatch" in reason_set:
            expected_status = "false"
        elif expected_hash and actual_hash:
            expected_status = str(expected_hash.lower() == actual_hash.lower()).lower()
        elif expected_hash:
            expected_status = "unknown"
        else:
            expected_status = "not_requested"
        arena_hits = number(measured.get("dynamic_arena_hits"))
        arena_misses = number(measured.get("dynamic_arena_misses"))
        arena_hit_rate = ""
        if arena_hits is not None and arena_misses is not None and arena_hits + arena_misses > 0:
            arena_hit_rate = decimal(arena_hits / (arena_hits + arena_misses))
        verdict_target = "safety configuration"
        if parameters.get("q8_f16_cache_mb") is not None:
            verdict_target = (
                f"Q8-F16 {clean(parameters.get('q8_f16_cache_mb'))}/"
                f"{clean(parameters.get('q8_f16_cache_reserve_mb'))} configuration"
            )
        row = base_row()
        row.update(
            {
                "row_id": f"WIN-G7-FAILURE-{path.stem.removesuffix('_failure')}",
                "source_kind": "windows_g7_failure",
                "evidence_level": "measured_failed_safety_n1",
                "benchmark_usable": "rejected_safety_configuration",
                "date": WINDOWS_G7_SNAPSHOT_DATE,
                "suite": "20260714_windows_native_g7",
                "run_id": path.stem.removesuffix("_failure"),
                "category": "windows_native_failure",
                "experiment": tag,
                "variant": tag,
                "profile": clean(parameters),
                "prompt_name": windows_prompt_name(tag, ""),
                "prompt_sha16": clean(src.get("prompt_sha256"))[:16],
                "hardware": "native Windows / WDDM",
                "model": Path(clean(src.get("model"))).name if src.get("model") else "",
                "model_bytes": clean(src.get("model_bytes")),
                "ds4_base": "hawkli-1994/ds4-win",
                "ds4_build": clean(src.get("head")),
                "runner_commit": clean(src.get("head")),
                "ctx": clean(parameters.get("context")),
                "request_max_tokens": clean(parameters.get("max_tokens")),
                "server_max_tokens": clean(parameters.get("max_tokens")),
                "temperature": "0",
                "stream": "0",
                "think": "0",
                "prompt_s": clean(measured.get("server_prefill_ttft_seconds")),
                "quality_signal": (
                    f"failure_reasons={clean(reasons)}; expected_hash={expected_status}; "
                    "failed_safety_gate; L0-L3=not_graded"
                ),
                "metrics_text": clean(measured),
                "setup_text": clean(parameters),
                "runtime_platform": "windows_native_wddm",
                "cache_state": "new_process_uncontrolled",
                "source_head": clean(src.get("head")),
                "executable_sha256": clean(src.get("executable_sha256")),
                "repeats": "1",
                "replication_scope": "single_server_process",
                "warmup": "false",
                "server_decode_mean_tps": clean(measured.get("server_decode_tokens_per_second")),
                "server_prefill_ttft_s": clean(measured.get("server_prefill_ttft_seconds")),
                "outputs_identical": "not_applicable",
                "expected_hash_match": expected_status,
                "output_sha256": actual_hash,
                "dynamic_arena_gib": clean(parameters.get("dynamic_arena_gib")),
                "dynamic_arena_window": clean(parameters.get("dynamic_arena_observed_window")),
                "dynamic_arena_min_hits": clean(parameters.get("dynamic_arena_observed_min_hits")),
                "dynamic_arena_grow_interval": clean(parameters.get("dynamic_arena_grow_interval")),
                "dynamic_arena_resident": clean(measured.get("dynamic_arena_resident_experts")),
                "dynamic_arena_hit_rate": arena_hit_rate,
                "q8_f16_cache_mib": clean(parameters.get("q8_f16_cache_mb")),
                "q8_f16_reserve_mib": clean(parameters.get("q8_f16_cache_reserve_mb")),
                "wddm_shared_peak_gib": clean(measured.get("gpu_process_shared_peak_gib")),
                "wddm_dedicated_peak_gib": clean(measured.get("gpu_process_dedicated_peak_gib")),
                "process_read_gib": clean(measured.get("win32_process_read_transfer_delta_gib")),
                "gpu_util_median": clean(measured.get("gpu_utilization_median_percent")),
                "vram_peak_mib": clean(measured.get("nvidia_vram_used_peak_mib")),
                "result_text": (
                    f"status={clean(src.get('status'))}; reasons={clean(reasons)}; "
                    f"server={clean(measured.get('server_decode_tokens_per_second'))} t/s; "
                    f"expected={expected_hash}; actual={actual_hash or 'not_retained'}"
                ),
                "verdict": f"rejected {verdict_target}",
                "source_artifacts": rel(path),
            }
        )
        rows.append(row)
    if len(rows) != len(paths):
        raise RuntimeError(
            f"Windows G7 failure import lost artifacts: parsed {len(rows)} of {len(paths)}"
        )
    return rows


def parse_stage0_rows(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in sorted((repo / "runs/ds4_stage0_baseline").rglob("summary.csv")):
        meta = read_json(summary.parent / "meta.json")
        with summary.open(newline="", encoding="utf-8", errors="replace") as handle:
            for src in csv.DictReader(handle):
                row = base_row()
                row.update(
                    {
                        "row_id": f"STAGE0-{summary.parent.name}-r{src.get('run', '')}",
                        "source_kind": "stage0_summary",
                        "evidence_level": "measured",
                        "benchmark_usable": "legacy_only",
                        "date": meta.get("started", "")[:8],
                        "suite": summary.parent.parent.name,
                        "run_id": summary.parent.name,
                        "category": "ds4_baseline",
                        "experiment": "Stage0 baseline DS4 streaming/cache locality",
                        "variant": clean(src.get("mode")),
                        "prompt_name": "stage0_cache_locality",
                        "prompt_excerpt": clean(meta.get("prompt")),
                        "model": clean(meta.get("model")),
                        "ctx": clean(meta.get("ctx")),
                        "prefill_chunk": clean(meta.get("prefill_chunk")),
                        "server_cache_experts": clean(meta.get("cache_experts")),
                        "completion_tokens": clean(meta.get("tokens")),
                        "avg_tps": clean(src.get("generation_tps")),
                        "first50_tps": "",
                        "prefetch_ms": clean(src.get("copy_ms")),
                        "tier_miss_eviction": f"{src.get('cache_misses', '')}/",
                        "metrics_text": clean(src),
                        "setup_text": clean(meta),
                        "source_artifacts": rel(summary),
                    }
                )
                row["quality_signal"] = "speed/cache baseline only"
                rows.append(row)
    return rows


def parse_k91_rows(repo: Path) -> list[dict[str, str]]:
    meta_path = repo / "runs/reap/k91_coding_vram/meta.json"
    fit_path = repo / "runs/reap/k91_coding_vram/fit_results.json"
    meta = read_json(meta_path)
    fit = read_json(fit_path)
    rows: list[dict[str, str]] = []
    configs = fit.get("configs", {}) or meta.get("risultati", {})
    quality = meta.get("risultati", {}).get("quality_1run_greedy", {})
    for name, cfg in configs.items():
        if not isinstance(cfg, dict):
            continue
        row = base_row()
        row.update(
            {
                "row_id": f"K91-{name}",
                "source_kind": "legacy_reap_json",
                "evidence_level": "measured_legacy",
                "benchmark_usable": "legacy_only",
                "date": clean(meta.get("date_utc", "2026-07-06")).replace("-", ""),
                "suite": "runs/reap/k91_coding_vram",
                "run_id": name,
                "category": "legacy_reap_fit_quality",
                "experiment": clean(meta.get("test")),
                "variant": name,
                "prompt_name": "BST website coding prompt",
                "hardware": clean(meta.get("pod", {}).get("gpu")),
                "gpu": clean(meta.get("pod", {}).get("gpu")),
                "model": clean(meta.get("software", {}).get("model")),
                "model_bytes": clean(meta.get("software", {}).get("model_bytes")),
                "ds4_base": clean(meta.get("software", {}).get("ds4_base")),
                "patches": ",".join(meta.get("software", {}).get("patches", [])),
                "ctx": "2048",
                "prefill_chunk": "512",
                "server_cache_experts": "380",
                "temperature": "0",
                "completion_tokens": "90 fit / 2600 quality",
                "avg_tps": clean(cfg.get("gen_ts")),
                "metrics_text": clean(cfg),
                "setup_text": clean(fit.get("measure_config")),
                "result_text": clean(fit.get("verdict_fit")),
                "verdict": clean(quality.get(name.replace("_9pct", "")) or meta.get("risultati", {}).get("quality_verdict")),
                "source_artifacts": f"{rel(meta_path)}; {rel(fit_path)}",
            }
        )
        row["quality_signal"] = derive_quality(row)
        rows.append(row)
    return rows


def parse_reap_biasmask_rows(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cases = [
        (
            "REAP-BIASMASK-V1",
            repo / "runs/reap/2026-07-05_eval_biasmask/README.md",
            [
                ("full", "3.811", "1.000x", "5.344", "1.000x", "baseline"),
                ("reap_k50", "3.860", "1.013x", "", "", "PASS dom <=1.10x; near-lossless domain"),
                ("rand_k50", "5.200", "1.365x", "11.30", "2.115x", "selection matters; random worse"),
            ],
        ),
        (
            "REAP-BIASMASK-V2",
            repo / "runs/reap/2026-07-05_eval_biasmask_v2/README.md",
            [
                ("full", "3.852", "1.000x", "", "", "baseline H200 paired"),
                ("reap_k50", "3.891", "1.010x CI[0.996,1.025]", "1.403x CI[1.296,1.537]", "", "domain statistically lossless; gen tradeoff"),
                ("reap_k67", "4.143", "1.076x CI[1.046,1.110]", "1.892x", "", "operating point that fits 32GB"),
                ("rand_s0", "5.346", "1.388x", "", "", "selection control"),
            ],
        ),
    ]
    for suite, source, configs in cases:
        for variant, ppl_dom, vs_dom, ppl_gen, vs_gen, verdict in configs:
            row = base_row()
            row.update(
                {
                    "row_id": f"{suite}-{variant}",
                    "source_kind": "legacy_reap_readme",
                    "evidence_level": "measured_legacy",
                    "benchmark_usable": "quality_only",
                    "date": "20260705",
                    "suite": suite,
                    "run_id": variant,
                    "category": "reap_biasmask_ppl",
                    "experiment": "Bias-mask REAP DS4 perplexity",
                    "variant": variant,
                    "model": "DeepSeek-V4-Flash-IQ2XXS imatrix GGUF",
                    "metrics_text": f"ppl_dom={ppl_dom}; vs_dom={vs_dom}; ppl_gen={ppl_gen}; vs_gen={vs_gen}",
                    "verdict": verdict,
                    "source_artifacts": rel(source),
                }
            )
            row["quality_signal"] = derive_quality(row)
            rows.append(row)
    return rows


def parse_legacy_experiment_table(repo: Path) -> list[dict[str, str]]:
    path = repo / "docs/EXPERIMENTS_LEDGER.md"
    rows: list[dict[str, str]] = []
    for line in text(path).splitlines():
        cells = split_md_row(line)
        if not cells:
            continue
        key = cells[0].strip("`")
        if not re.match(r"^(?:[A-Z]\d+|HIST-|DS4-)", key):
            continue
        combined = " ".join(cells)
        relevant = bool(re.search(r"DS4|Flash|PACE|K91|K23|K96|cache|Dwarf|RunPod|SPEX|tier|CQ1|reap", combined, re.I))
        row = base_row()
        if key.startswith("HIST-") or key.startswith("DS4-"):
            row.update(
                {
                    "row_id": f"LEDGER-{key}",
                    "source_kind": "legacy_experiments_ledger",
                    "evidence_level": "legacy_doc",
                    "benchmark_usable": "legacy_only",
                    "date": "20260703-20260710",
                    "suite": "docs/EXPERIMENTS_LEDGER.md",
                    "run_id": key,
                    "category": "historical_claude_recovery",
                    "experiment": key,
                    "setup_text": clean(cells[1] if len(cells) > 1 else ""),
                    "result_text": clean(cells[2] if len(cells) > 2 else ""),
                    "verdict": clean(cells[3] if len(cells) > 3 else ""),
                    "source_artifacts": f"{clean(cells[1] if len(cells) > 1 else '')}; {rel(path)}",
                }
            )
            row["quality_signal"] = derive_quality(row)
            rows.append(row)
            continue
        row.update(
            {
                "row_id": f"LEDGER-{key}",
                "source_kind": "legacy_experiments_ledger",
                "evidence_level": "legacy_doc",
                "benchmark_usable": "legacy_only" if relevant else "context_only",
                "date": "20260703-20260710",
                "suite": "docs/EXPERIMENTS_LEDGER.md",
                "run_id": key,
                "category": clean(cells[1] if len(cells) > 1 else ""),
                "experiment": clean(cells[2] if len(cells) > 2 else ""),
                "setup_text": clean(cells[3] if len(cells) > 3 else ""),
                "result_text": clean(cells[4] if len(cells) > 4 else ""),
                "verdict": clean(cells[5] if len(cells) > 5 else ""),
                "source_artifacts": clean(cells[6] if len(cells) > 6 else rel(path)),
            }
        )
        row["quality_signal"] = derive_quality(row)
        rows.append(row)
    return rows


def parse_claim_rows(repo: Path) -> list[dict[str, str]]:
    path = repo / "docs/CLAIMS_CURRENT.md"
    rows: list[dict[str, str]] = []
    idx = 0
    for line in text(path).splitlines():
        cells = split_md_row(line)
        if len(cells) != 4 or cells[0] in ("Claim", ""):
            continue
        idx += 1
        row = base_row()
        row.update(
            {
                "row_id": f"CLAIM-{idx:03d}",
                "source_kind": "claims_current",
                "evidence_level": "current_claim",
                "benchmark_usable": "context_only",
                "date": "20260707",
                "suite": "docs/CLAIMS_CURRENT.md",
                "run_id": f"CLAIM-{idx:03d}",
                "category": clean(cells[1]),
                "experiment": clean(cells[0]),
                "result_text": clean(cells[2]),
                "source_artifacts": clean(cells[3]),
            }
        )
        row["quality_signal"] = derive_quality(row)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, str]], columns: list[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = clean(row.get(col, ""))
            value = value.replace("|", "/")
            if len(value) > 180:
                value = value[:177] + "..."
            cells.append(value)
        out.append("| " + " | ".join(cells) + " |")
    return out


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    runner_measured = [r for r in rows if r["source_kind"] == "runner_summary"]
    windows = [r for r in rows if r["source_kind"].startswith("windows_g7_")]
    windows_results = [r for r in windows if r["source_kind"] == "windows_g7_result"]
    windows_campaigns = [r for r in windows if r["source_kind"] == "windows_g7_campaign_aggregate"]
    windows_failures = [r for r in windows if r["source_kind"] == "windows_g7_failure"]
    legacy = [r for r in rows if r["source_kind"] != "runner_summary" and not r["source_kind"].startswith("windows_g7_")]
    best_rotation = [
        r
        for r in runner_measured
        if "requested4" in r["suite"] and "rotate32" in r["variant"]
    ]
    direct_pod = [
        r
        for r in runner_measured
        if r["suite"] == "20260710_pod_cache1024_html800"
    ]
    pace_advanced = [
        r
        for r in runner_measured
        if r["suite"] in ("20260710_pace_advanced_ab_html400", "20260710_pace_advanced_ab_html800")
    ]
    warmup_ab = [
        r
        for r in runner_measured
        if r["suite"] in (
            "20260710_w100_direct_k23_cache256_html2000",
            "20260710_w100_rotate32_k23_cache256_html2000",
            "20260710_w100_rotate32_k23_cache256_html2000_compact_prompt",
            "20260710_w50_rotate32_k23_cache256_html2000",
            "20260710_w50_rotate32_k23_cache256_html4000",
            "20260710_w50_rotate32_k23_cache256_html4000_ctx8192",
            "20260710_w100_rotate32_k23_cache256_html4000_ctx8192",
            "20260710_m1a_w50_w100_ctx8192_n3",
            "20260710_m1b_w50_stopguard_ctx8192_n3",
        )
    ]
    cache_sweep = [
        r
        for r in runner_measured
        if "local_cache_sweep" in r["suite"] and r["prompt_name"] in ("html", "code_mini")
    ]
    claude_recovery = [
        r
        for r in legacy
        if r["row_id"].startswith("LEDGER-HIST") or r["row_id"].startswith("CLAIM")
    ]

    windows_ranked = sorted(
        [r for r in windows if number(r.get("server_decode_mean_tps")) is not None],
        key=lambda r: number(r.get("server_decode_mean_tps")) or 0,
        reverse=True,
    )
    windows_all = sorted(windows, key=lambda r: (r["date"], r["run_id"]))
    windows_top = windows_ranked[:20]
    g22 = [r for r in windows if "g22_" in r["run_id"]]
    g25 = [r for r in windows if "g25_" in r["run_id"] or "g25_" in r["experiment"]]
    g26 = [
        r
        for r in windows
        if "g26_" in r["run_id"]
        or "g26b_" in r["run_id"]
        or "g26_" in r["experiment"]
        or "g26b_" in r["experiment"]
    ]
    g27 = [
        r
        for r in windows
        if "g27_" in r["run_id"] or "g27_" in r["experiment"]
    ]

    lines: list[str] = [
        "# DS4 / REAP Experiment Ledger - updated 2026-07-14",
        "",
        "This file is generated by `scripts/build_ds4_experiment_ledger.py`.",
        "Numbers are copied from artifacts when `source_kind=runner_summary` or `source_kind=windows_g7_result`; older Claude/research rows are preserved as legacy evidence and must not be mixed into benchmark plots unless re-run or explicitly marked comparable.",
        "",
        "## Output Files",
        "",
        f"- Master CSV: `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv`",
        f"- Rows total: {len(rows)}",
        f"- Runner-measured rows: {len(runner_measured)}",
        f"- Windows-native G7 result rows: {len(windows_results)}",
        f"- Windows-native independent campaign aggregates: {len(windows_campaigns)}",
        f"- Windows-native failed safety rows: {len(windows_failures)}",
        f"- Legacy / Claude / claim rows: {len(legacy)}",
        "",
        "## Current Readout",
        "",
        "- Treat repeat/ngram as diagnostics only. The requested HTML800 `K23 rotate32` row remains interesting, but its quality status must be read through L0-L3 grading or rendered artifacts, not through repeat absence.",
        "- Static/direct K23 is the speed baseline, not the quality answer: it is fast but repeatedly breaks HTML in multiple prompt/cache regimes; W100 direct K0->K23 at cache256 failed around token 183 despite a stable ~3.08 t/s tail.",
        "- W100+rotate32 at cache256 avoided the early loop through 2000 tokens and rendered a visible page. The run allocated most of the available budget to detailed CSS and reached body markup around token 1904; missing form/script/html close should be treated as token-budget-limited, not as degeneration.",
        "- W50+rotate32 with the same normal prompt, cache256, and 2000-token cap also avoided the early loop and reached body/card markup earlier, around token 1541, with slightly better average throughput than W100. It is still token-budget-limited: no form/script/html close within 2000 tokens.",
        "- The max_tokens=4000 A/B must use ctx8192. The W50 ctx4096 diagnostic hit total_tokens=4078 and looped in CSS before <body>, so it is context-confounded rather than a clean 4000-token quality result.",
        "- With ctx8192, W50+rotate32 completed a document at 2417 completion tokens and reached body/form/script/html close, but the produced page is not functional: malformed form close, popup commented out, and invalid JS. W100+rotate32 ctx8192 spent more budget in CSS, reached body/form much later, then looped on `//` inside script and ended by length without </html>.",
        "- `prompt_s` is prompt prefill/cache/order state, not generated-token warmup cost. The historical W100 ctx8192 `prompt_s=266.374s` must not be attributed solely to W=100 or VRAM filling; speed comparisons require paired cache-state/order.",
        "- The compact budget-aware prompt did not improve this A/B: it reached `<script>` earlier but entered a repeated `/* js */` placeholder loop, with first bad event around 961 and conclusive repetition around 977.",
        "- Breath variants that fire after visible n-gram damage are too late; useful post-return tokens were measured as zero in the requested A/B.",
        "- Cache1024 pod runs restore high throughput, but cache size alone did not restore quality on the cyberpunk HTML prompt. The old W50 session-learning result is real enough to keep as historical evidence, but freeze-point/prompt sensitivity is now explicit.",
        "- Tighten-time relearn and rotation plumbing are useful actuator milestones. Blind step-down and frequent periodic rotate are too expensive; next tests should be trigger/delta based.",
        "- Dynamic compression is not yet a speed win. Lossless cold RAM is too large, CQ1 works mechanically but synchronous selected-miss use is far too slow. The useful target is effective cap512-cap1024 behavior with background promotion/demotion.",
        "- Windows-native results must be read with `cache_state`, `replication_scope`, `repeats`, exact hash and L0-L3 status together. A high n=1 safety result is retained as mechanism evidence; it is not silently discarded and is not promoted to a sustained verdict.",
        "- G22 isolates same-process arena reuse: KEEP measured 4.32 server decode t/s and DROP 1.42 t/s with the same expected hash. This is a paired n=1 causal safety result, pending order-balanced independent n>=3 replication.",
        "- G25 prefill-mass bulk WRAP is decode-positive on the short Caesar prompt: 14 GiB WRAP independent n=3 averaged 4.37 server t/s versus 2.30 control with exact output, but the mean WRAP publication cost was 15.44 s and is not amortized by a 16-token request.",
        "- The 2 GiB G25 safety run is correctness/mechanism evidence only. No long-decode n=1 artifact is present in ds4-win commit `8f76b81`, imported in this G25 snapshot or used as a verdict.",
        "- G27 turns the packed G26b REAP mass signal into transactional pinned-RAM residency with an eight-slot swap ring. The mechanism is correct and exact, but the final counterbalanced 16-token n=3 gate is negative on speed: ON averaged 2.073 server t/s versus 2.513 OFF, because it pays about 2.36 s of WRAP bootstrap/rotation inside a very short decode. This is not a long-run amortization verdict.",
        "",
        "## High-Signal Runtime Rows",
        "",
    ]
    high = best_rotation + warmup_ab + direct_pod + pace_advanced[:4]
    lines.extend(
        md_table(
            high,
            [
                "suite",
                "variant",
                "server_cache_experts",
                "pace_warmup",
                "pace_keep",
                "pace_rotate",
                "pace_rotate_every",
                "completion_tokens",
                "avg_tps",
                "last_chunk_tps",
                "prefetch_gib",
                "l0l3",
                "client_stop_reason",
                "retry_attempts",
                "repeat_flag",
                "coherent_until_token_est",
                "quality_signal",
            ],
        )
    )
    lines.extend(["", "## Windows Native G7 - Highest Observed Server Decode", ""])
    lines.extend(
        md_table(
            windows_top,
            [
                "run_id",
                "date",
                "cache_state",
                "repeats",
                "replication_scope",
                "server_decode_mean_tps",
                "client_completion_mean_tps",
                "server_prefill_ttft_s",
                "dynamic_arena_gib",
                "dynamic_arena_window",
                "dynamic_arena_min_hits",
                "dynamic_arena_carry",
                "dynamic_arena_resident",
                "resident_expert_cache",
                "spex_stage",
                "expected_hash_match",
                "benchmark_usable",
            ],
        )
    )
    lines.extend(["", "## Windows Native G22 Causal Rows", ""])
    lines.extend(
        md_table(
            g22,
            [
                "run_id",
                "source_head",
                "executable_sha256",
                "cache_state",
                "repeats",
                "server_decode_mean_tps",
                "client_completion_mean_tps",
                "server_prefill_ttft_s",
                "dynamic_arena_carry",
                "dynamic_arena_resident",
                "dynamic_arena_hit_rate",
                "standby_before_gib",
                "process_read_gib",
                "expected_hash_match",
                "evidence_level",
            ],
        )
    )
    lines.extend(["", "## Windows Native G25 Prefill-Mass Bulk WRAP", ""])
    lines.extend(
        md_table(
            g25,
            [
                "run_id",
                "source_head",
                "request_max_tokens",
                "dynamic_arena_gib",
                "server_decode_mean_tps",
                "server_prefill_ttft_s",
                "prefill_mass_wrap_result",
                "prefill_mass_wrap_reason",
                "prefill_mass_wrap_candidate_entries",
                "prefill_mass_wrap_loads",
                "prefill_mass_wrap_workers",
                "prefill_mass_wrap_seconds",
                "prefill_mass_wrap_snapshot_after",
                "prefill_mass_wrap_resident_after",
                "dynamic_arena_final_hits",
                "dynamic_arena_final_misses",
                "ds4_cuda_sha256",
                "benchmark_usable",
            ],
        )
    )
    lines.extend(["", "## Windows Native G26/G26b REAP Evidence", ""])
    lines.extend(
        md_table(
            g26,
            [
                "run_id",
                "source_head",
                "server_decode_mean_tps",
                "server_prefill_ttft_s",
                "expected_hash_match",
                "repeats",
                "replication_scope",
                "cache_state",
                "ds4_cuda_sha256",
                "benchmark_usable",
                "result_text",
            ],
        )
    )
    lines.extend(["", "## Windows Native G27 REAP Mass WRAP", ""])
    lines.extend(
        md_table(
            g27,
            [
                "run_id",
                "source_head",
                "server_decode_mean_tps",
                "server_prefill_ttft_s",
                "expected_hash_match",
                "repeats",
                "replication_scope",
                "cache_state",
                "reap_mass_wrap_publications",
                "reap_mass_wrap_entrants",
                "reap_mass_wrap_victims",
                "reap_mass_wrap_seconds",
                "reap_mass_wrap_failures",
                "dynamic_arena_final_hits",
                "dynamic_arena_final_misses",
                "ds4_cuda_sha256",
                "benchmark_usable",
                "result_text",
            ],
        )
    )
    lines.extend(["", "## Windows Native Independent Campaign Aggregates", ""])
    lines.extend(
        md_table(
            windows_campaigns,
            [
                "run_id",
                "variant",
                "repeats",
                "replication_scope",
                "server_decode_mean_tps",
                "server_decode_min_tps",
                "server_decode_max_tps",
                "client_completion_mean_tps",
                "server_prefill_ttft_s",
                "dynamic_arena_resident",
                "prefill_mass_wrap_seconds",
                "expected_hash_match",
                "benchmark_usable",
                "result_text",
            ],
        )
    )
    lines.extend(["", "## Windows Native Failed Safety Gates", ""])
    lines.extend(
        md_table(
            windows_failures,
            [
                "run_id",
                "source_head",
                "server_decode_mean_tps",
                "server_prefill_ttft_s",
                "q8_f16_cache_mib",
                "q8_f16_reserve_mib",
                "process_read_gib",
                "expected_hash_match",
                "verdict",
                "result_text",
            ],
        )
    )
    lines.extend(["", "## Windows Native G7 - Complete Result Matrix", ""])
    lines.extend(
        md_table(
            windows_all,
            [
                "run_id",
                "date",
                "source_head",
                "prompt_name",
                "request_max_tokens",
                "cache_state",
                "repeats",
                "replication_scope",
                "server_decode_mean_tps",
                "client_completion_mean_tps",
                "server_prefill_ttft_s",
                "dynamic_arena_gib",
                "dynamic_arena_window",
                "dynamic_arena_min_hits",
                "dynamic_arena_grow_interval",
                "dynamic_arena_carry",
                "dynamic_arena_resident",
                "resident_expert_cache",
                "moe_io_qd",
                "spex_stage",
                "spex_cap",
                "q8_f16_cache_mib",
                "outputs_identical",
                "expected_hash_match",
                "benchmark_usable",
            ],
        )
    )
    lines.extend(["", "## Cache Pattern Rows", ""])
    lines.extend(
        md_table(
            cache_sweep,
            [
                "suite",
                "variant",
                "prompt_name",
                "server_cache_experts",
                "avg_tps",
                "last_chunk_tps",
                "tier_miss_eviction",
                "l0l3",
                "repeat_flag",
                "quality_signal",
            ],
        )
    )
    lines.extend(["", "## Legacy / Claude Evidence Index", ""])
    lines.extend(
        md_table(
            claude_recovery[:80],
            [
                "row_id",
                "source_kind",
                "benchmark_usable",
                "category",
                "experiment",
                "result_text",
                "source_artifacts",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Next Table Hygiene",
            "",
            "- Every manual UI test should get a card with prompt text/hash, launch env, ds4 build, reap-loop commit, cache, trace flags, max tokens, and observed metrics.",
            "- Benchmark rows should normally run with routing trace and verbose tier logs off; interesting candidates get a diagnostic twin with trace on for Scope replay.",
            "- Do not use `claims_current` or `legacy_experiments_ledger` rows as headline performance unless the row points to a measured artifact or has been replayed in `runs/ds4/**/summary.csv`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/ds4/20260710_experiment_ledger")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir.resolve()

    rows: list[dict[str, str]] = []
    rows.extend(parse_summary_rows(repo))
    rows.extend(parse_windows_g7_rows(repo))
    rows.extend(parse_quality_full_decode_mass_weight_rows(repo))
    rows.extend(parse_windows_g7_failure_rows(repo))
    rows.extend(parse_stage0_rows(repo))
    rows.extend(parse_k91_rows(repo))
    rows.extend(parse_reap_biasmask_rows(repo))
    rows.extend(parse_legacy_experiment_table(repo))
    rows.extend(parse_claim_rows(repo))

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for row in rows:
        row_id = row["row_id"]
        if row_id in seen_ids:
            duplicate_ids.add(row_id)
        seen_ids.add(row_id)
    if duplicate_ids:
        raise RuntimeError(f"Duplicate ledger row IDs: {', '.join(sorted(duplicate_ids)[:20])}")

    rows.sort(key=lambda r: (r["source_kind"], r["date"], r["suite"], r["run_id"], r["variant"]))
    csv_path = out_dir / "all_evidence_ledger.csv"
    md_path = repo / "docs/DS4_EXPERIMENT_LEDGER_20260710.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(csv_path)
    print(md_path)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
