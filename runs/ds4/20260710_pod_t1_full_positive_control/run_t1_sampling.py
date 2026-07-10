#!/usr/bin/env python3
"""T1 sampling arm: FULL no-mask (DS4_PACE=0) with sampling instead of greedy.

Reuses run_ds4_exchange_matrix helpers (server launch, log parse, quality flags,
functional grading) so artifacts stay comparable with the greedy runs.

Params per coordinator mandate 2026-07-10: temperature 0.7, top_p 0.95, fixed seed.
ds4-server /v1/chat/completions accepts: max_tokens, temperature, top_p, top_k,
min_p, seed, stream (DwarfStar README "OpenAI-compatible endpoints" section).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_ds4_exchange_matrix as m  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="html", choices=list(m.PROMPTS))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--port", type=int, default=8014)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="/root/models/ds4-2bit.gguf")
    ap.add_argument("--ctx", type=int, default=3072)
    ap.add_argument("--server-max-tokens", type=int, default=2300)
    ap.add_argument("--cache-experts", type=int, default=1024)
    ap.add_argument("--prefill-chunk", type=int, default=128)
    args = ap.parse_args()

    variant = m.Variant("no_pace_sampled", {"DS4_PACE": "0"},
                        "T1 sampling arm: FULL routing, no PACE mask, sampled decode.")
    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    prompt = m.PROMPTS[args.prompt]
    rows = []

    m.stop_ds4()
    for run_idx in range(1, args.runs + 1):
        stem = f"{args.prompt}_no_pace_sampled_r{run_idx:02d}"
        run_dir = out_root / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        env = m.build_env(variant, run_dir)
        body = {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Rispondi in modo diretto, utile e senza ragionamento visibile."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stream": False,
            "think": False,
            "thinking": {"type": "disabled"},
        }
        manifest = {
            "runner_id": stem,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "profile": "POD_T1_SAMPLING_ARM",
            "variant": variant.name,
            "variant_rationale": variant.rationale,
            "prompt": {"name": args.prompt,
                       "sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                       "chars": len(prompt)},
            "sampling": {
                "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                "params_accepted_by_endpoint": ["max_tokens", "temperature", "top_p",
                                                 "top_k", "min_p", "seed", "stream"],
                "params_source": "DwarfStar ds4 README, OpenAI-compatible endpoints section",
                "note": "seed fixed for form; CUDA decode is not bit-reproducible anyway (POD_PLAYBOOK §6)",
            },
            "server": {"model": args.model, "ctx": args.ctx,
                       "server_max_tokens": args.server_max_tokens,
                       "cache_experts": args.cache_experts,
                       "prefill_chunk": args.prefill_chunk, "port": args.port,
                       "request_max_tokens": args.max_tokens},
            "env_effective": env,
        }
        (run_dir / "runner_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        ns = argparse.Namespace(model=args.model, ctx=args.ctx,
                                server_max_tokens=args.server_max_tokens,
                                cache_experts=args.cache_experts,
                                prefill_chunk=args.prefill_chunk)
        proc = m.start_server(variant, env, run_dir, args.port, ns)
        try:
            m.wait_models(args.port, timeout_s=180)
            (run_dir / "request_measured.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            wall_s, response = m.post_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", body, args.timeout)
            (run_dir / "response_measured.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
            content = m.response_content(response)
            (run_dir / "content_measured.txt").write_text(content, encoding="utf-8")
            log_metrics = m.parse_server_log(run_dir / "server.stderr.log")
            q = m.quality_flags(args.prompt, content)
            usage = response.get("usage") if isinstance(response, dict) else None
            l0l3 = m.grade_l0l3(args.prompt, content)
            row = {"stem": stem, "evidence_type": "measured",
                   "wall_s": round(wall_s, 3),
                   "prompt_tokens": (usage or {}).get("prompt_tokens"),
                   "completion_tokens": (usage or {}).get("completion_tokens"),
                   "avg_tps_pod_diagnostic": log_metrics.get("avg_tps"),
                   "l0l3": l0l3, **q}
            rows.append(row)
            print(f"[sampling] done {stem} wall={wall_s:.1f}s avg={log_metrics.get('avg_tps')} l0l3={l0l3}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            m.stop_ds4()
            time.sleep(1)

    (out_root / "summary_sampling.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_root / "summary_sampling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
