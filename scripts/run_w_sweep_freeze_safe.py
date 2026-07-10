"""T4 harness: W-sweep of two-phase session-learning with a SAFE freeze point.

Turn-key offline prep for test T4 (docs/NEXT_STEPS_PLAN_20260710.md, Fase 2).
The old cache1024 W-table was a lottery of the cut point (ledger J44): W values
that truncated the phase-2 re-prefill mid CSS declaration triggered a
document-restart, so "quality vs W" was really "did W land after a ``}``".
This harness removes that confound by freezing phase-1 at a structural boundary
(scripts/freeze_boundary.py) before re-prefilling phase-2, then sweeps
W in {30,50,70,90,110,130,150} x n=3 and grades each render L0-L3. If the W
scale becomes monotone under a safe freeze, the W-table is real; if it flattens,
the old table was the freeze-point lottery.

Reused verbatim from the pod replay
(runs/ds4/20260710_pod_cache1024_warmup_replay/README.md):
  * the two-phase env/flag recipe (DS4_SPEX_TRACE_ROUTING[_WEIGHTS] in phase 1,
    DS4_REAP_MASK_FILE in phase 2; --ssd-streaming[-cold] + cache-experts);
  * the gate-mass mask builder (now scripts/build_session_mask_canonical.py);
  * the ``ds4: prefill: X t/s, generation: Y t/s`` diag line;
  * ``deliverable = frozen_phase1 + phase2_continuation``;
  * the recovered compact coffee-shop prompt as the default.
New here: the freeze step between phase 1 and phase 2, n=3 with per-seed rows and
a median summary, and L0-L3 grading via scripts/functional_grade.py.

SAFETY: this only *builds* commands and (unless --dry-run) runs the provided
``--binary``. It touches no WSL/GPU config. It is NOT auto-run; on the 3060 run
it from inside WSL, on a pod run it directly. py_compile-clean; the command
builders and parsers are import-safe pure functions.

Example (pod)::

    python scripts/run_w_sweep_freeze_safe.py \
        --binary /root/ds4/ds4 --model /root/models/ds4-2bit.gguf \
        --cache 1024 --runs 3

Dry-run (no binary, prints the plan)::

    python scripts/run_w_sweep_freeze_safe.py --binary ds4 --model m.gguf --dry-run
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

DEFAULT_W = [30, 50, 70, 90, 110, 130, 150]
_TPS_RE = re.compile(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s")
_FENCE_OPEN_RE = re.compile(r"(?:^|\n)[ \t]*```[a-zA-Z0-9_-]*[ \t]*\r?\n")
_FENCE_CLOSE_RE = re.compile(r"\n```[ \t]*(?:\r?\n|$)")
_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_ALERT_RE = re.compile(r"alert\s*\(|confirm\s*\(|showModal", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(.{24,160})\1\1", re.S)

_HERE = pathlib.Path(__file__).resolve().parent
_REPLAY_DIR = _HERE.parent / "runs" / "ds4" / "20260710_pod_cache1024_warmup_replay"


def _load_sibling(name):
    """Import a sibling scripts/*.py module by path (no package install needed)."""
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


freeze_boundary = _load_sibling("freeze_boundary")
build_session_mask_canonical = _load_sibling("build_session_mask_canonical")
try:
    functional_grade = _load_sibling("functional_grade")
except Exception:  # pragma: no cover - grading is optional
    functional_grade = None


# ----------------------------- pure helpers -----------------------------------

def strip_markdown_fence(text):
    """Strip a markdown code fence (leading OR after prose) + its closing fence.

    The local live-tree binary (post-0018) wraps the phase-1 HTML in a
    ```` ```html ```` fence even with --nothink; the pod replay output had none.
    Pod arm S5 also observed PROSE before the fence, which a leading-only strip
    missed (freeze=none -> raw cut). Unstripped, the backticks read as an
    unterminated JS template literal in ``freeze_boundary._safe_boundaries``
    (zero safe boundaries -> raw-cut lottery, the exact J44 confound T4
    removes) and prose/fence would contaminate the phase-2 re-prefill (the
    historical recipe prefills pure HTML). Rule: drop everything up to and
    including the FIRST fence-open line; cut at the closing fence if one
    follows. No fence at all -> unchanged. Pure string function.
    """
    if not text:
        return text
    m = _FENCE_OPEN_RE.search(text)
    if not m:
        return text
    text = text[m.end():]
    m2 = _FENCE_CLOSE_RE.search(text)
    if m2:
        text = text[:m2.start() + 1]  # keep the newline that precedes the fence
    return text


def default_w_values():
    return list(DEFAULT_W)


def w_run_dir(outdir, w, run):
    return pathlib.Path(outdir) / f"W{w:03d}" / f"r{run:02d}"


def _base_cmd(binary, model, ctx, ntokens, cache, prompt_file, temp, seed,
              top_p=None):
    cmd = [
        str(binary), "-m", str(model),
        "--cuda", "--ssd-streaming", "--ssd-streaming-cold",
        "--ssd-streaming-cache-experts", str(cache),
        "-c", str(ctx), "--nothink", "--temp", str(temp),
        "-n", str(ntokens), "--prompt-file", str(prompt_file),
    ]
    if float(temp) > 0 and seed is not None:
        cmd += ["--seed", str(seed)]
    if float(temp) > 0 and top_p is not None:
        cmd += ["--top-p", str(top_p)]
    return cmd


def phase1_cmd(args, w, prompt_file, route_csv, seed):
    """Phase-1: observe wide W (+headroom) with routing-weight trace on.

    With --sample-p2-only, phase 1 stays GREEDY (temp 0, no seed) so the
    routing trace / session mask / freeze point are built in the same regime
    as the greedy arms; sampling then applies only to the masked phase 2
    (the sampling-under-mask probe isolates exactly that variable).
    """
    env = {
        "DS4_SPEX_TRACE_ROUTING": str(route_csv),
        "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "1",
    }
    p1_temp = 0.0 if getattr(args, "sample_p2_only", False) else args.temp
    cmd = _base_cmd(args.binary, args.model, args.ctx_p1, w + args.headroom,
                    args.cache, prompt_file, p1_temp, seed,
                    getattr(args, "top_p", None))
    return env, cmd


def phase2_cmd(args, w, p2prompt_file, mask_file, seed):
    """Phase-2: re-prefill [prompt + frozen phase-1] under the session mask."""
    env = {"DS4_REAP_MASK_FILE": str(mask_file)}
    ntokens = max(1, args.total - w)
    cmd = _base_cmd(args.binary, args.model, args.ctx_p2, ntokens,
                    args.cache, p2prompt_file, args.temp, seed,
                    getattr(args, "top_p", None))
    return env, cmd


def parse_tps(diag_text):
    """Return {'prefill': float|None, 'generation': float|None} from a diag.

    Uses the LAST matching ``ds4: prefill: X t/s, generation: Y t/s`` line.
    """
    last = None
    for m in _TPS_RE.finditer(diag_text or ""):
        last = m
    if not last:
        return {"prefill": None, "generation": None}
    return {"prefill": float(last.group(1)), "generation": float(last.group(2))}


def output_checks(text):
    """Cheap structural counts mirroring the replay summary line."""
    low = (text or "").lower()
    scripts = "\n".join(_SCRIPT_BLOCK_RE.findall(text or ""))
    return {
        "chars": len(text or ""),
        "doctype": len(re.findall(r"<!doctype html", low)),
        "html_close": low.count("</html>"),
        "form": low.count("<form"),
        "script": low.count("<script"),
        "alert_in_script": len(_ALERT_RE.findall(scripts)),
        "repeat": 1 if _REPEAT_RE.search(text or "") else 0,
    }


def grade_render(text):
    """(level, detail) via functional_grade.grade_frontpage, or (None, {})."""
    if functional_grade is None:
        return None, {}
    try:
        return functional_grade.grade_frontpage(text)
    except Exception as exc:  # pragma: no cover - defensive
        return None, {"grade_error": str(exc)}


def median_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def verdict_monotone(median_level_by_w):
    """T4 verdict input: is median L0-L3 non-decreasing across ascending W?

    Returns a dict with the sorted (w, level) pairs, whether the sequence is
    monotone non-decreasing (the W-table is real), and the spread (max-min).
    """
    pairs = sorted((w, lvl) for w, lvl in median_level_by_w.items() if lvl is not None)
    levels = [lvl for _, lvl in pairs]
    monotone = all(b >= a for a, b in zip(levels, levels[1:])) if len(levels) > 1 else None
    spread = (max(levels) - min(levels)) if levels else None
    return {"pairs": pairs, "monotone_non_decreasing": monotone, "level_spread": spread}


# ----------------------------- execution --------------------------------------

def _run(env, cmd, stdout_path, stderr_path, timeout):
    full_env = dict(os.environ)
    full_env.update(env)
    with open(stdout_path, "w", encoding="utf-8") as out, \
            open(stderr_path, "w", encoding="utf-8") as err:
        subprocess.run(cmd, env=full_env, stdout=out, stderr=err,
                       check=False, timeout=timeout)


def _read(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def run_one(args, w, run, prompt_text):
    """Execute one (W, run) two-phase cell; return a result row dict."""
    d = w_run_dir(args.outdir, w, run)
    d.mkdir(parents=True, exist_ok=True)
    seed = args.seed_base + run
    route_csv = d / "route.csv"
    tw = d / "tw.txt"
    p1diag = d / "p1.diag"
    frozen_path = d / "frozen.txt"
    mask_txt = d / "sess.txt"
    p2prompt = d / "p2prompt.txt"
    trest = d / "trest.txt"
    p2diag = d / "p2.diag"
    deliverable = d / "deliverable.html"

    # --- phase 1: wide observation with weighted routing trace ---
    env1, cmd1 = phase1_cmd(args, w, args.prompt_file, route_csv, seed)
    _run(env1, cmd1, tw, p1diag, args.timeout)

    # --- freeze at a safe boundary <= W tokens (fence-stripped: the local
    # live-tree binary emits a ```html fence that blinds the boundary scanner) ---
    gen_text = strip_markdown_fence(_read(tw))
    fp = freeze_boundary.find_safe_freeze_point(gen_text, w)
    frozen_path.write_text(fp.frozen_text, encoding="utf-8", newline="\n")

    # --- build the session mask (weighted by default; unit for the T5 arm) ---
    keep, seen, have_weights = build_session_mask_canonical.build(
        str(route_csv), args.keep_k, args.mask_mode, args.n_expert)
    with open(mask_txt, "w", newline="\n") as f:
        for layer, e in build_session_mask_canonical.pruned_pairs(keep, args.n_expert):
            f.write(f"{layer} {e}\n")

    # --- phase 2: re-prefill [prompt + frozen] under the mask ---
    p2prompt.write_text(prompt_text + fp.frozen_text, encoding="utf-8", newline="\n")
    env2, cmd2 = phase2_cmd(args, w, p2prompt, mask_txt, seed)
    _run(env2, cmd2, trest, p2diag, args.timeout)

    # --- assemble deliverable and grade ---
    deliverable_text = fp.frozen_text + _read(trest)
    deliverable.write_text(deliverable_text, encoding="utf-8", newline="\n")
    p1 = parse_tps(_read(p1diag))
    p2 = parse_tps(_read(p2diag))
    level, det = grade_render(deliverable_text)
    checks = output_checks(deliverable_text)

    return {
        "w": w, "run_index": run, "seed": seed,
        "freeze_boundary": fp.boundary, "freeze_tokens_est": fp.n_tokens,
        "freeze_within_target": int(fp.within_target),
        "p1_prefill_tps": p1["prefill"], "p1_gen_tps": p1["generation"],
        "p2_prefill_tps": p2["prefill"], "p2_gen_tps": p2["generation"],
        "l0l3": level,
        "restart": int(bool(det.get("restart"))),
        "button_wired": int(bool(det.get("button_wired"))),
        "form_wired": int(bool(det.get("form_wired"))),
        "tag_mismatch": det.get("tag_mismatch"),
        "chars": checks["chars"], "doctype": checks["doctype"],
        "html_close": checks["html_close"], "form": checks["form"],
        "script": checks["script"], "alert_in_script": checks["alert_in_script"],
        "repeat": checks["repeat"],
    }


CSV_FIELDS = [
    "w", "run_index", "seed", "freeze_boundary", "freeze_tokens_est",
    "freeze_within_target", "p1_prefill_tps", "p1_gen_tps", "p2_prefill_tps",
    "p2_gen_tps", "l0l3", "restart", "button_wired", "form_wired", "tag_mismatch",
    "chars", "doctype", "html_close", "form", "script", "alert_in_script", "repeat",
]


def write_summaries(outdir, rows):
    outdir = pathlib.Path(outdir)
    with open(outdir / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in CSV_FIELDS})

    by_w = {}
    for r in rows:
        by_w.setdefault(r["w"], []).append(r)
    median_rows = []
    median_level_by_w = {}
    for w in sorted(by_w):
        rs = by_w[w]
        levels = [r["l0l3"] for r in rs if r["l0l3"] is not None]
        lvl_med = statistics.median(levels) if levels else None
        median_level_by_w[w] = lvl_med
        median_rows.append({
            "w": w, "n": len(rs),
            "p2_gen_tps_median": median_or_none([r["p2_gen_tps"] for r in rs]),
            "p1_gen_tps_median": median_or_none([r["p1_gen_tps"] for r in rs]),
            "l0l3_median": lvl_med,
            "repeat_majority": 1 if sum(r["repeat"] for r in rs) * 2 >= len(rs) else 0,
            "restart_majority": 1 if sum(r["restart"] for r in rs) * 2 >= len(rs) else 0,
        })
    with open(outdir / "summary_median.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(median_rows[0].keys()) if median_rows
                            else ["w"])
        wr.writeheader()
        for r in median_rows:
            wr.writerow(r)

    verdict = verdict_monotone(median_level_by_w)
    (outdir / "VERDICT.txt").write_text(
        "T4 W-sweep freeze-safe verdict\n"
        f"median L0-L3 by W: {verdict['pairs']}\n"
        f"monotone non-decreasing: {verdict['monotone_non_decreasing']}\n"
        f"level spread (max-min): {verdict['level_spread']}\n"
        "Reading: monotone/rising => the W-table is real; flat/low spread => the "
        "old table was the freeze-point lottery (J44).\n",
        encoding="utf-8")
    return verdict


def write_manifest(args, prompt_text):
    manifest = {
        "harness": "run_w_sweep_freeze_safe.py",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "w_values": args.w_values, "runs": args.runs, "headroom": args.headroom,
        "total": args.total, "keep_k": args.keep_k, "mask_mode": args.mask_mode,
        "n_expert": args.n_expert, "temp": args.temp,
        "top_p": getattr(args, "top_p", None),
        "sample_p2_only": bool(getattr(args, "sample_p2_only", False)),
        "seed_base": args.seed_base,
        "cache_experts": args.cache, "ctx_p1": args.ctx_p1, "ctx_p2": args.ctx_p2,
        "port": args.port, "binary": str(args.binary), "model": str(args.model),
        "prompt_file": str(args.prompt_file), "prompt_chars": len(prompt_text),
        "recipe_source": "runs/ds4/20260710_pod_cache1024_warmup_replay/README.md",
        "note": ("port is recorded for manifest/server-mode parity; the executed "
                 "path is CLI-direct two-phase (as in the pod replay)."),
    }
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(args.outdir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="path to the ds4 runtime binary")
    ap.add_argument("--model", required=True, help="path to the ds4 gguf model")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default runs/ds4/<date>_w_sweep_freeze_safe)")
    ap.add_argument("--prompt-file", default=str(_REPLAY_DIR / "frontpage_prompt.txt"),
                    help="prompt file (default: the recovered compact coffee-shop prompt)")
    ap.add_argument("--w-values", default=",".join(str(w) for w in DEFAULT_W),
                    help="comma-separated W values")
    ap.add_argument("--runs", type=int, default=3, help="repetitions per W (n=3)")
    ap.add_argument("--headroom", type=int, default=16,
                    help="extra phase-1 tokens so the freeze can land <= W on a boundary")
    ap.add_argument("--total", type=int, default=1000, help="phase1+phase2 token budget")
    ap.add_argument("--keep-k", type=int, default=23, help="experts kept per layer")
    ap.add_argument("--mask-mode", choices=("weighted", "unit"), default="weighted",
                    help="ranking signal for the session mask (T5 arm: unit)")
    ap.add_argument("--n-expert", type=int, default=256)
    ap.add_argument("--cache", type=int, default=1024, help="--ssd-streaming-cache-experts")
    ap.add_argument("--ctx-p1", type=int, default=2048, help="phase-1 context length")
    ap.add_argument("--ctx-p2", type=int, default=3072, help="phase-2 context length")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None, dest="top_p",
                    help="nucleus sampling top-p (passed only when --temp > 0)")
    ap.add_argument("--sample-p2-only", action="store_true", dest="sample_p2_only",
                    help="keep phase 1 GREEDY (temp 0); apply --temp/--seed/--top-p "
                         "only to the masked phase 2 (sampling-under-mask probe)")
    ap.add_argument("--seed-base", type=int, default=0,
                    help="seed = seed_base + run_index (only passed when --temp > 0)")
    ap.add_argument("--port", type=int, default=None,
                    help="reserved for server-mode/manifest; CLI-direct is the executed path")
    ap.add_argument("--timeout", type=int, default=3600, help="per-phase subprocess timeout (s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and phase-1 commands; run nothing")
    args = ap.parse_args(argv)
    args.w_values = [int(x) for x in str(args.w_values).split(",") if x.strip()]
    if args.outdir is None:
        stamp = _dt.date.today().strftime("%Y%m%d")
        args.outdir = str(_HERE.parent / "runs" / "ds4" / f"{stamp}_w_sweep_freeze_safe")
    return args


def main(argv=None):
    args = parse_args(argv)
    prompt_text = _read(args.prompt_file)
    if not prompt_text:
        print(f"ERRORE: prompt vuoto/mancante: {args.prompt_file}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[dry-run] outdir = {args.outdir}")
        print(f"[dry-run] W = {args.w_values}  runs = {args.runs}  "
              f"mask-mode = {args.mask_mode}  cache = {args.cache}")
        for w in args.w_values:
            d = w_run_dir(args.outdir, w, 0)
            env1, cmd1 = phase1_cmd(args, w, args.prompt_file, d / "route.csv", args.seed_base)
            env_s = " ".join(f"{k}={v}" for k, v in env1.items())
            print(f"[dry-run] W={w} phase1: {env_s} {' '.join(cmd1)}")
        print("[dry-run] phase-2 commands depend on the phase-1 freeze output; "
              "not shown. No binary was executed.")
        return 0

    for tool in (args.binary, args.model):
        if os.sep in str(tool) and not pathlib.Path(tool).exists():
            print(f"ERRORE: non trovato: {tool}", file=sys.stderr)
            return 2

    write_manifest(args, prompt_text)
    rows = []
    for w in args.w_values:
        for run in range(args.runs):
            print(f"[run] W={w} run={run} ...", flush=True)
            rows.append(run_one(args, w, run, prompt_text))
    verdict = write_summaries(args.outdir, rows)
    print(f"[done] rows={len(rows)}  verdict={verdict}")
    print(f"[done] summaries in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
