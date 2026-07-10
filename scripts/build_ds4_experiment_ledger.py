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
    "coherent_until_token_est",
    "useful_tokens_after_return_est",
    "pace_events",
    "quality_signal",
    "metrics_text",
    "setup_text",
    "env_effective_json",
    "request_settings_json",
    "result_text",
    "verdict",
    "source_artifacts",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
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
    if row.get("repeat_flag") != "":
        bits.append(f"repeat={row['repeat_flag']}")
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
    measured = [r for r in rows if r["source_kind"] == "runner_summary"]
    legacy = [r for r in rows if r["source_kind"] != "runner_summary"]
    best_rotation = [
        r
        for r in measured
        if "requested4" in r["suite"] and "rotate32" in r["variant"]
    ]
    direct_pod = [
        r
        for r in measured
        if r["suite"] == "20260710_pod_cache1024_html800"
    ]
    pace_advanced = [
        r
        for r in measured
        if r["suite"] in ("20260710_pace_advanced_ab_html400", "20260710_pace_advanced_ab_html800")
    ]
    cache_sweep = [
        r
        for r in measured
        if "local_cache_sweep" in r["suite"] and r["prompt_name"] in ("html", "code_mini")
    ]
    claude_recovery = [
        r
        for r in legacy
        if r["row_id"].startswith("LEDGER-HIST") or r["row_id"].startswith("CLAIM")
    ]

    lines: list[str] = [
        "# DS4 / REAP Experiment Ledger - 2026-07-10",
        "",
        "This file is generated by `scripts/build_ds4_experiment_ledger.py`.",
        "Numbers are copied from artifacts when `source_kind=runner_summary`; older Claude/research rows are preserved as legacy evidence and must not be mixed into benchmark plots unless re-run or explicitly marked comparable.",
        "",
        "## Output Files",
        "",
        f"- Master CSV: `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv`",
        f"- Rows total: {len(rows)}",
        f"- Runner-measured rows: {len(measured)}",
        f"- Legacy / Claude / claim rows: {len(legacy)}",
        "",
        "## Current Readout",
        "",
        "- Best current 3060-local stability candidate in the requested HTML800 A/B is still `K23 rotate32`: it reached 800 streamed tokens without the repeat detector, but it is slower than static K23 and still needs render/functional grading.",
        "- Static/direct K23 is the speed baseline, not the quality answer: it is fast but repeatedly breaks HTML in multiple prompt/cache regimes; W100 direct K0->K23 at cache256 failed around token 183 despite a stable ~3.08 t/s tail.",
        "- Breath variants that fire after visible n-gram damage are too late; useful post-return tokens were measured as zero in the requested A/B.",
        "- Cache1024 pod runs restore high throughput, but cache size alone did not restore quality on the cyberpunk HTML prompt. The old W50 session-learning result is real enough to keep as historical evidence, but freeze-point/prompt sensitivity is now explicit.",
        "- Tighten-time relearn and rotation plumbing are useful actuator milestones. Blind step-down and frequent periodic rotate are too expensive; next tests should be trigger/delta based.",
        "- Dynamic compression is not yet a speed win. Lossless cold RAM is too large, CQ1 works mechanically but synchronous selected-miss use is far too slow. The useful target is effective cap512-cap1024 behavior with background promotion/demotion.",
        "",
        "## High-Signal Runtime Rows",
        "",
    ]
    high = best_rotation + direct_pod + pace_advanced[:4]
    lines.extend(
        md_table(
            high,
            [
                "suite",
                "variant",
                "server_cache_experts",
                "pace_keep",
                "pace_rotate",
                "pace_rotate_every",
                "completion_tokens",
                "avg_tps",
                "last_chunk_tps",
                "prefetch_gib",
                "repeat_flag",
                "coherent_until_token_est",
                "quality_signal",
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
    rows.extend(parse_stage0_rows(repo))
    rows.extend(parse_k91_rows(repo))
    rows.extend(parse_reap_biasmask_rows(repo))
    rows.extend(parse_legacy_experiment_table(repo))
    rows.extend(parse_claim_rows(repo))

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
