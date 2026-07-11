#!/usr/bin/env python3
"""COMBO B — A/B: two-phase re-prefill (control) vs in-engine continuous PACE.

Question (retrospective 3f12b0e / T4): the phase-2 document-restart is an
attractor of the RE-PREFILL [prompt]+[partial HTML], not of the freeze cut.
Hypothesis: an in-engine CONTINUOUS stream (PACE warmup W50 -> learn -> apply
K23 on the fly -> keep decoding, NO re-prefill) removes the restart at the root.

ARM A (two-phase control): reuses run_w_sweep_freeze_safe.run_one verbatim
  (phase1 weighted routing trace over W50, safe freeze, build weighted K23 mask,
   phase2 re-prefill [prompt+frozen] under DS4_REAP_MASK_FILE, assemble+grade).
ARM B (in-engine continuous): one ds4 stream, DS4_PACE warmup=50, weighted
  warmup mask, KEEP=23 static (min=max=23, step=0), breath/rotate/relearn OFF,
  PREFILL_APPLY=0 (mask learned from the 50-decode-token warmup window, applied
  at tok50, same stream, NO re-prefill). Same base flags/prompt/cache/ctx/temp.

ABAB, n=3, greedy (temp 0). Metrics per run: L0-L3, doctype count (doc-restart),
</html> count, generation t/s, chars, repeat. Writes summary.csv + medians +
VERDICT.txt.

Usage:
  driver.py --prompt coffee --base /root/comboB/coffee --total 1200 --ctx 4096 --rounds 3
  driver.py --prompt cyber  --base /root/comboB/cyber  --total 4050 --ctx 8192 --rounds 3
  add --smoke for a fast 1-round short-budget validation.
"""
import argparse, csv, datetime, json, os, pathlib, re, statistics, subprocess, sys, time

SCRIPTS = "/root/reap-loop/scripts"
sys.path.insert(0, SCRIPTS)
import run_w_sweep_freeze_safe as tp  # noqa: E402

BIN = "/root/canon/ds4"
MODEL = "/root/models/ds4-2bit.gguf"
PROMPTS = {"coffee": "/root/prompts/frontpage_prompt.txt",
           "cyber": "/root/prompts/cyberpunk_prompt.txt"}


def build_argsA(args, prompt_file, outdir):
    return tp.parse_args([
        "--binary", BIN, "--model", MODEL, "--prompt-file", prompt_file,
        "--outdir", outdir, "--w-values", "50", "--runs", str(args.rounds),
        "--total", str(args.total), "--keep-k", "23", "--mask-mode", "weighted",
        "--n-expert", "256", "--cache", str(args.cache),
        "--ctx-p1", str(args.ctx), "--ctx-p2", str(args.ctx),
        "--temp", "0", "--headroom", "16", "--timeout", str(args.timeout),
    ])


def inengine_env(rundir):
    e = dict(os.environ)
    e.update({
        "DS4_PACE": "1",
        "DS4_PACE_WARMUP": "50",
        "DS4_PACE_KEEP": "23",
        "DS4_PACE_KEEP_MIN": "23",
        "DS4_PACE_KEEP_MAX": "23",
        "DS4_PACE_KEEP_STEP": "0",
        "DS4_PACE_BREATH_EVERY": "9999999",
        "DS4_PACE_BREATH_LEN": "1",
        "DS4_PACE_DRIFT": "99",
        "DS4_PACE_HYST": "9999999",
        "DS4_PACE_ROTATE": "0",
        "DS4_PACE_RELEARN": "0",
        "DS4_PACE_PREFILL_APPLY": "0",     # learn from the 50-tok warmup window, apply at tok50, no tok0 prefill mask
        "DS4_PACE_WEIGHTED_WARMUP": "1",   # weighted mass mask (match arm A weighted)
        "DS4_PACE_WEIGHTED_SELECTED": "1",
        "DS4_PACE_WRAP": "0",
        "DS4_PACE_DEBUG": "1",
        "DS4_PACE_LOG": os.path.join(rundir, "pace.jsonl"),
        # keep routing trace OFF (speed); this is the whole point of PACE
        "DS4_SPEX_TRACE_ROUTING": "",
        "DS4_SPEX_TRACE_ROUTING_WEIGHTS": "0",
    })
    return e


def run_in_engine(args, prompt_file, prompt_text, outdir, r):
    d = pathlib.Path(outdir) / f"r{r:02d}"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "gen.out"
    err = d / "gen.err"
    cmd = [BIN, "-m", MODEL, "--cuda", "--ssd-streaming", "--ssd-streaming-cold",
           "--ssd-streaming-cache-experts", str(args.cache),
           "-c", str(args.ctx), "--nothink", "--temp", "0",
           "-n", str(args.total), "--prompt-file", prompt_file]
    json.dump({"cmd": cmd, "arm": "in_engine", "round": r},
              open(d / "manifest.json", "w"), indent=2)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as o, open(err, "w", encoding="utf-8") as e2:
        subprocess.run(cmd, env=inengine_env(str(d)), stdout=o, stderr=e2,
                       check=False, timeout=args.timeout)
    wall = time.time() - t0
    raw = tp._read(out)
    deliverable = tp.strip_markdown_fence(raw)
    (d / "deliverable.html").write_text(deliverable, encoding="utf-8", newline="\n")
    diag = tp.parse_tps(tp._read(err))
    level, det = tp.grade_render(deliverable)
    checks = tp.output_checks(deliverable)
    # doctype count on RAW output too (honest doc-restart count regardless of fence)
    doctype_raw = len(re.findall(r"<!doctype html", raw.lower()))
    return {
        "arm": "in_engine", "round": r, "w": 50,
        "l0l3": level,
        "restart": int(bool(det.get("restart"))),
        "doctype": checks["doctype"], "doctype_raw": doctype_raw,
        "html_close": checks["html_close"],
        "button_wired": int(bool(det.get("button_wired"))),
        "form_wired": int(bool(det.get("form_wired"))),
        "gen_tps": diag["generation"], "prefill_tps": diag["prefill"],
        "chars": checks["chars"], "repeat": checks["repeat"],
        "wall_s": round(wall, 1),
    }


def row_from_A(a):
    doctype_raw = a["doctype"]  # arm A deliverable == frozen+phase2; count on it
    return {
        "arm": "two_phase", "round": a["run_index"], "w": a["w"],
        "l0l3": a["l0l3"], "restart": a["restart"],
        "doctype": a["doctype"], "doctype_raw": doctype_raw,
        "html_close": a["html_close"],
        "button_wired": a["button_wired"], "form_wired": a["form_wired"],
        "gen_tps": a["p2_gen_tps"], "prefill_tps": a["p2_prefill_tps"],
        "chars": a["chars"], "repeat": a["repeat"], "wall_s": None,
    }


FIELDS = ["arm", "round", "w", "l0l3", "restart", "doctype", "doctype_raw",
          "html_close", "button_wired", "form_wired", "gen_tps", "prefill_tps",
          "chars", "repeat", "wall_s"]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=list(PROMPTS), default="coffee")
    ap.add_argument("--base", required=True)
    ap.add_argument("--total", type=int, default=1200)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cache", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.total = 200
        args.rounds = 1
    prompt_file = PROMPTS[args.prompt]
    prompt_text = tp._read(prompt_file)
    base = pathlib.Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    armA_dir = str(base / "armA_twophase")
    armB_dir = str(base / "armB_inengine")

    argsA = build_argsA(args, prompt_file, armA_dir)
    tp.write_manifest(argsA, prompt_text)

    prog = base / "progress.log"
    def log(m):
        line = f"[{datetime.datetime.utcnow():%H:%M:%S}] {m}"
        print(line, flush=True)
        with open(prog, "a") as f:
            f.write(line + "\n")

    log(f"START comboB prompt={args.prompt} total={args.total} ctx={args.ctx} "
        f"cache={args.cache} rounds={args.rounds} smoke={args.smoke}")

    rows = []
    for r in range(args.rounds):
        log(f"round {r}: ARM A (two-phase) ...")
        a = tp.run_one(argsA, 50, r, prompt_text)
        rowA = row_from_A(a)
        rows.append(rowA)
        log(f"  A r{r}: L={rowA['l0l3']} doctype={rowA['doctype']} "
            f"</html>={rowA['html_close']} restart={rowA['restart']} "
            f"gen_tps={rowA['gen_tps']} chars={rowA['chars']}")
        log(f"round {r}: ARM B (in-engine) ...")
        rowB = run_in_engine(args, prompt_file, prompt_text, armB_dir, r)
        rows.append(rowB)
        log(f"  B r{r}: L={rowB['l0l3']} doctype={rowB['doctype']} "
            f"doctype_raw={rowB['doctype_raw']} </html>={rowB['html_close']} "
            f"restart={rowB['restart']} gen_tps={rowB['gen_tps']} "
            f"chars={rowB['chars']} wall={rowB['wall_s']}s")

    with open(base / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in FIELDS})

    # medians + verdict
    def arm_rows(name):
        return [r for r in rows if r["arm"] == name]
    summ = {}
    for name in ("two_phase", "in_engine"):
        rs = arm_rows(name)
        summ[name] = {
            "n": len(rs),
            "L_median": med([r["l0l3"] for r in rs]),
            "L_all": [r["l0l3"] for r in rs],
            "doctype_all": [r["doctype"] for r in rs],
            "docrestart_count": sum(1 for r in rs if (r["doctype"] or 0) >= 2),
            "htmlclose_all": [r["html_close"] for r in rs],
            "gen_tps_median": med([r["gen_tps"] for r in rs]),
            "chars_median": med([r["chars"] for r in rs]),
            "repeat_count": sum(r["repeat"] for r in rs),
        }
    verdict = {
        "prompt": args.prompt, "total": args.total, "rounds": args.rounds,
        "two_phase": summ["two_phase"], "in_engine": summ["in_engine"],
        "restart_eliminated": (summ["in_engine"]["docrestart_count"] == 0
                               and summ["two_phase"]["docrestart_count"] > 0),
    }
    json.dump(verdict, open(base / "verdict.json", "w"), indent=2)
    (base / "VERDICT.txt").write_text(
        "COMBO B — two-phase re-prefill vs in-engine continuous\n"
        f"prompt={args.prompt} total={args.total} rounds={args.rounds}\n\n"
        f"TWO-PHASE : L={summ['two_phase']['L_all']} median={summ['two_phase']['L_median']} "
        f"doctype={summ['two_phase']['doctype_all']} docrestart={summ['two_phase']['docrestart_count']}/"
        f"{summ['two_phase']['n']} </html>={summ['two_phase']['htmlclose_all']} "
        f"gen_tps_med={summ['two_phase']['gen_tps_median']}\n"
        f"IN-ENGINE : L={summ['in_engine']['L_all']} median={summ['in_engine']['L_median']} "
        f"doctype={summ['in_engine']['doctype_all']} docrestart={summ['in_engine']['docrestart_count']}/"
        f"{summ['in_engine']['n']} </html>={summ['in_engine']['htmlclose_all']} "
        f"gen_tps_med={summ['in_engine']['gen_tps_median']}\n\n"
        f"RESTART ELIMINATED (in-engine 0 vs two-phase >0): {verdict['restart_eliminated']}\n",
        encoding="utf-8")
    log("DONE")
    print("=== VERDICT ===")
    print((base / "VERDICT.txt").read_text())


if __name__ == "__main__":
    sys.exit(main())
