"""Clock-breath fair trial (D6b, lever L6) — long-horizon offline reconstruction.

The user's original "respiro" recipe: a static session-learned keep-K mask, but
with a CLOCK-scheduled breath window every ``--breath-every`` generated tokens:
open fully to K0 (mask off) for ``--breath-len`` tokens so the router demand is
HONEST again (the demand-evaporation finding: under a frozen mask the gate stops
asking for pruned experts, so it cannot self-correct), RELEARN a fresh weighted
mask from that breath window, freeze on a safe boundary, and resume at keep-K.

This harness is the *offline reconstruction between phases* explicitly sanctioned
by the mandate (the in-engine PACE relearn is the descent actuator with a known
non-convergence caveat; the offline path is deterministic and fully measurable —
we build every mask and can diff them). It is a strict generalization of
``run_w_sweep_freeze_safe.py`` (two-phase): identical binary flags, identical
weighted mask builder, identical safe-freeze boundary — the ONLY added variable
is the periodic breath+relearn. With ``--static`` it degenerates to the plain
two-phase static-K23 arm (internal control on the same pod/binary).

Execution is CLI-direct, greedy (temp 0), trace only where a mask must be learnt
(phase-1 warmup + each breath window), manifest + per-run rows + median summary +
L0-L3 grading, exactly as the baselines (d3b4614 static-K23, pod3 FROZEN).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import json
import os
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load_sibling(name):
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the freeze-safe helpers verbatim so this arm is bit-comparable to baselines.
fs = _load_sibling("run_w_sweep_freeze_safe")
freeze_boundary = fs.freeze_boundary
mask_builder = fs.build_session_mask_canonical


def write_mask(keep, path, n_expert):
    with open(path, "w", newline="\n") as f:
        for layer, e in mask_builder.pruned_pairs(keep, n_expert):
            f.write(f"{layer} {e}\n")


def mask_churn(keep_old, keep_new):
    """Experts that entered/left the kept set, summed over layers (symmetric diff)."""
    if keep_old is None:
        return None, None
    layers = set(keep_old) | set(keep_new)
    changed = 0
    total_kept = 0
    for lyr in layers:
        a = set(keep_old.get(lyr, []))
        b = set(keep_new.get(lyr, []))
        changed += len(a ^ b)
        total_kept += len(b)
    frac = (changed / (2 * total_kept)) if total_kept else None  # /2: symmetric diff double-counts a swap
    return changed, frac


def _base_cmd(args, ctx, ntokens, prompt_file):
    return fs._base_cmd(args.binary, args.model, ctx, ntokens, args.cache,
                        prompt_file, args.temp, None)


def run_one(args, run, prompt_text):
    d = pathlib.Path(args.outdir) / f"r{run:02d}"
    d.mkdir(parents=True, exist_ok=True)
    seed = args.seed_base + run

    accumulated = ""
    tokens = 0
    cur_keep = None
    cur_mask_file = None
    masks = []          # keep dicts, in order
    churns = []         # experts changed per relearn
    tight_tps = []
    breath_tps = []
    phases = []         # per-phase log
    collapse_token = None

    # ---- phase 1: warmup observe (K0) with weighted routing trace ----
    route0 = d / "route_p1.csv"
    tw = d / "p1.out"; p1diag = d / "p1.diag"
    env1 = {"DS4_SPEX_TRACE_ROUTING": str(route0), "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "1"}
    cmd1 = _base_cmd(args, args.ctx_p1, args.warmup + args.headroom, args.prompt_file)
    fs._run(env1, cmd1, tw, p1diag, args.timeout)
    gen0 = fs.strip_markdown_fence(fs._read(tw))
    fp0 = freeze_boundary.find_safe_freeze_point(gen0, args.warmup)
    accumulated = fp0.frozen_text
    tokens = fp0.n_tokens
    keep0, _seen, _hw = mask_builder.build(str(route0), args.keep_k, args.mask_mode, args.n_expert)
    m0 = d / "mask_00.txt"; write_mask(keep0, m0, args.n_expert)
    cur_keep, cur_mask_file = keep0, m0
    masks.append(keep0)
    p1 = fs.parse_tps(fs._read(p1diag))
    phases.append({"kind": "warmup", "tokens_added": fp0.n_tokens,
                   "gen_tps": p1["generation"], "boundary": fp0.boundary})

    cycle = 0
    while tokens < args.total and cycle < args.max_cycles:
        # ---- TIGHT: generate <= breath_every at keep-K under current mask ----
        remaining = args.total - tokens
        ntok = min(args.breath_every, remaining)
        tp = d / f"tight{cycle}_prompt.txt"
        tp.write_text(prompt_text + accumulated, encoding="utf-8", newline="\n")
        tout = d / f"tight{cycle}.out"; tdiag = d / f"tight{cycle}.diag"
        env_t = {"DS4_REAP_MASK_FILE": str(cur_mask_file)}
        cmd_t = _base_cmd(args, args.ctx_p2, ntok, tp)
        fs._run(env_t, cmd_t, tout, tdiag, args.timeout)
        ttext = fs._read(tout)
        fpt = freeze_boundary.find_safe_freeze_point(ttext, ntok)
        prev_len = len(accumulated)
        accumulated += fpt.frozen_text
        tokens += fpt.n_tokens
        tt = fs.parse_tps(fs._read(tdiag))
        if tt["generation"]:
            tight_tps.append(tt["generation"])
        phases.append({"kind": "tight", "cycle": cycle, "keep": args.keep_k,
                       "tokens_added": fpt.n_tokens, "gen_tps": tt["generation"],
                       "boundary": fpt.boundary})
        if collapse_token is None and fs.output_checks(accumulated[max(0, prev_len - 200):])["repeat"]:
            collapse_token = tokens
        if tokens >= args.total or args.static:
            if args.static:
                # static control: no breath; keep generating tight until total
                if tokens >= args.total:
                    break
                cycle += 1
                continue
            break

        # ---- BREATH: generate breath_len at K0 (no mask) with weighted trace ----
        bp = d / f"breath{cycle}_prompt.txt"
        bp.write_text(prompt_text + accumulated, encoding="utf-8", newline="\n")
        bout = d / f"breath{cycle}.out"; bdiag = d / f"breath{cycle}.diag"
        route_b = d / f"route_breath{cycle}.csv"
        env_b = {"DS4_SPEX_TRACE_ROUTING": str(route_b), "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "1"}
        cmd_b = _base_cmd(args, args.ctx_p2, args.breath_len, bp)
        fs._run(env_b, cmd_b, bout, bdiag, args.timeout)
        btext = fs._read(bout)
        fpb = freeze_boundary.find_safe_freeze_point(btext, args.breath_len)
        accumulated += fpb.frozen_text
        tokens += fpb.n_tokens
        bt = fs.parse_tps(fs._read(bdiag))
        if bt["generation"]:
            breath_tps.append(bt["generation"])

        # ---- RELEARN weighted mask from the fresh breath window ----
        try:
            keep_new, _s, _h = mask_builder.build(str(route_b), args.keep_k,
                                                  args.mask_mode, args.n_expert)
        except SystemExit:
            keep_new = cur_keep  # empty breath trace -> keep old mask
        changed, frac = mask_churn(cur_keep, keep_new)
        if changed is not None:
            churns.append(changed)
        mfile = d / f"mask_{cycle + 1:02d}.txt"; write_mask(keep_new, mfile, args.n_expert)
        masks.append(keep_new)
        phases.append({"kind": "breath", "cycle": cycle, "keep": 0,
                       "tokens_added": fpb.n_tokens, "gen_tps": bt["generation"],
                       "boundary": fpb.boundary, "relearn_experts_changed": changed,
                       "relearn_churn_frac": frac})
        cur_keep, cur_mask_file = keep_new, mfile
        cycle += 1

    deliverable = d / "deliverable.html"
    deliverable.write_text(accumulated, encoding="utf-8", newline="\n")
    level, det = fs.grade_render(accumulated)
    checks = fs.output_checks(accumulated)

    row = {
        "run": run, "seed": seed, "l0l3": level,
        "total_tokens_est": tokens, "n_breaths": len(churns),
        "collapse_token": collapse_token,
        "tight_gen_tps_median": fs.median_or_none(tight_tps),
        "breath_gen_tps_median": fs.median_or_none(breath_tps),
        "speed_tax_frac": (
            (statistics.median(tight_tps) - statistics.median(breath_tps))
            / statistics.median(tight_tps)
            if tight_tps and breath_tps else None),
        "relearn_churn_experts_median": fs.median_or_none(churns) if churns else None,
        "restart": int(bool(det.get("restart"))),
        "button_wired": int(bool(det.get("button_wired"))),
        "form_wired": int(bool(det.get("form_wired"))),
        "chars": checks["chars"], "doctype": checks["doctype"],
        "html_close": checks["html_close"], "form": checks["form"],
        "script": checks["script"], "alert_in_script": checks["alert_in_script"],
        "repeat": checks["repeat"],
    }
    (d / "run.json").write_text(json.dumps({"row": row, "phases": phases}, indent=2),
                               encoding="utf-8")
    return row


CSV_FIELDS = ["run", "seed", "l0l3", "total_tokens_est", "n_breaths",
              "collapse_token", "tight_gen_tps_median", "breath_gen_tps_median",
              "speed_tax_frac", "relearn_churn_experts_median", "restart",
              "button_wired", "form_wired", "chars", "doctype", "html_close",
              "form", "script", "alert_in_script", "repeat"]


def write_summary(outdir, rows, args):
    outdir = pathlib.Path(outdir)
    with open(outdir / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in CSV_FIELDS})
    levels = [r["l0l3"] for r in rows if r["l0l3"] is not None]
    med = statistics.median(levels) if levels else None
    verdict = {
        "arm": args.tag, "static": args.static, "keep_k": args.keep_k,
        "total": args.total, "breath_every": args.breath_every,
        "breath_len": args.breath_len, "n": len(rows),
        "l0l3_per_run": [r["l0l3"] for r in rows],
        "l0l3_median": med,
        "collapse_token_per_run": [r["collapse_token"] for r in rows],
        "html_close_per_run": [r["html_close"] for r in rows],
        "tight_tps_median": fs.median_or_none([r["tight_gen_tps_median"] for r in rows]),
        "breath_tps_median": fs.median_or_none([r["breath_gen_tps_median"] for r in rows]),
        "speed_tax_frac_median": fs.median_or_none([r["speed_tax_frac"] for r in rows]),
        "relearn_churn_median": fs.median_or_none([r["relearn_churn_experts_median"] for r in rows]),
    }
    (outdir / "VERDICT.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="clock_breath")
    ap.add_argument("--keep-k", type=int, default=23)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--headroom", type=int, default=16)
    ap.add_argument("--breath-every", type=int, default=450)
    ap.add_argument("--breath-len", type=int, default=70)
    ap.add_argument("--total", type=int, default=1200)
    ap.add_argument("--ctx-p1", type=int, default=2048)
    ap.add_argument("--ctx-p2", type=int, default=4096)
    ap.add_argument("--cache", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--n-expert", type=int, default=256)
    ap.add_argument("--mask-mode", choices=("weighted", "unit"), default="weighted")
    ap.add_argument("--max-cycles", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--static", action="store_true",
                    help="control: no breath windows (plain two-phase static keep-K)")
    args = ap.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    prompt_text = fs._read(args.prompt_file)
    if not prompt_text:
        print(f"ERRORE: prompt vuoto/mancante: {args.prompt_file}", file=sys.stderr)
        return 2
    for tool in (args.binary, args.model):
        if os.sep in str(tool) and not pathlib.Path(tool).exists():
            print(f"ERRORE: non trovato: {tool}", file=sys.stderr)
            return 2
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    manifest = {k: (str(v) if isinstance(v, pathlib.Path) else v)
                for k, v in vars(args).items()}
    manifest["created"] = _dt.datetime.now().isoformat(timespec="seconds")
    manifest["prompt_chars"] = len(prompt_text)
    (pathlib.Path(args.outdir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    for run in range(args.runs):
        print(f"[clock_breath] {args.tag} run={run} ...", flush=True)
        rows.append(run_one(args, run, prompt_text))
        print(f"[clock_breath] {args.tag} run={run} -> {rows[-1]['l0l3']} "
              f"tokens={rows[-1]['total_tokens_est']} breaths={rows[-1]['n_breaths']} "
              f"collapse={rows[-1]['collapse_token']}", flush=True)
    verdict = write_summary(args.outdir, rows, args)
    print(f"[done] {args.tag} verdict={json.dumps(verdict)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
