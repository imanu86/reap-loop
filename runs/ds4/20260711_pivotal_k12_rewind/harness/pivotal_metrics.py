"""Pivotal K12+rewind metrics: loop onset, retraction-aware text, rewind tax, good-tok/s.

Usage: python pivotal_metrics.py <arm_dir> [<arm_dir> ...]
Reads each <arm_dir>/W050/rNN/: trest.txt, deliverable.html, p2.diag,
pace_events.jsonl?, tokens.csv?, plus <arm_dir>/summary.csv (harness grades).
Emits <arm_dir>/pivotal_metrics.json and prints a per-run table.

Methods (declared):
- loop onset (char): smallest offset c in trest such that trest[c:] is (quasi-)
  periodic: for period p in 8..400, the tail must repeat the p-block >=3 times
  with exact match. useful_frac_char = c/len(trest); 1.0 if no lock found.
- retraction (arms with tokens.csv): replay rows sequentially; a pos <= last pos
  truncates the kept list back to that pos (client-side trim semantics of the
  0022 "rewind" event); final text = concat of kept pieces. Grade that too.
- good_tok/s = useful_frac * p2_gen_tps (rate basis; all runs share the 4000
  target budget; pod-only, ratios in-batch transfer).
"""
import csv
import io
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "pivotal_test"))
try:
    import functional_grade
except Exception:
    functional_grade = None

_TPS_RE = re.compile(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s")


def loop_onset_char(text, min_p=3, max_p=400, min_rep=3):
    """Return (onset_char, period) of the earliest tail repetition-lock, or (None, None)."""
    text = text.rstrip()  # a trailing newline/space breaks end-anchored periodicity
    n = len(text)
    if n < min_p * min_rep:
        return None, None
    best = None
    for p in range(min_p, min(max_p, n // min_rep) + 1):
        # how far back does exact p-periodicity extend from the end?
        i = n - 1 - p
        while i >= 0 and text[i] == text[i + p]:
            i -= 1
        periodic_len = (n - 1 - p) - i + p  # chars in the periodic tail
        if periodic_len >= p * min_rep:
            onset = n - periodic_len
            if best is None or onset < best[0]:
                best = (onset, p)
    return best if best else (None, None)


def parse_tps(path):
    try:
        txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    last = None
    for m in _TPS_RE.finditer(txt):
        last = m
    return float(last.group(2)) if last else None


def replay_tokens(tokens_csv):
    """Sequential replay with rewind truncation. Returns (final_pieces, events).

    events: list of dicts {row_index, from_pos, to_pos} for each backward jump.
    """
    kept = []  # list of (pos, piece)
    jumps = []
    with open(tokens_csv, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for idx, row in enumerate(rd):
            if len(row) < 3:
                continue
            try:
                pos = int(row[0])
            except ValueError:
                continue
            piece = row[2]
            if kept and pos <= kept[-1][0]:
                # rewind: drop kept rows with pos >= this pos
                from_pos = kept[-1][0]
                while kept and kept[-1][0] >= pos:
                    kept.pop()
                jumps.append({"row": idx, "from_pos": from_pos, "to_pos": pos})
            kept.append((pos, piece))
    return kept, jumps


def read_pace_events(path):
    evs = []
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return evs


def grade(text):
    if functional_grade is None:
        return None, {}
    try:
        return functional_grade.grade_frontpage(text)
    except Exception as e:
        return None, {"error": str(e)}


def analyze_run(rdir, frozen_prefix):
    rdir = pathlib.Path(rdir)
    trest = (rdir / "trest.txt").read_text(encoding="utf-8", errors="ignore") \
        if (rdir / "trest.txt").exists() else ""
    out = {"run": rdir.name}
    onset, period = loop_onset_char(trest)
    out["p2_chars"] = len(trest)
    out["loop_onset_char"] = onset
    out["loop_period"] = period
    out["useful_frac_char"] = (onset / len(trest)) if (onset is not None and trest) else 1.0
    out["p2_gen_tps"] = parse_tps(rdir / "p2.diag")
    out["good_tps"] = (out["useful_frac_char"] * out["p2_gen_tps"]) \
        if out["p2_gen_tps"] else None

    # rewind accounting
    evs = read_pace_events(rdir / "pace_events.jsonl")
    rw = [e for e in evs if e.get("ev") == "rewind"]
    out["rewind_arm_n"] = sum(1 for e in evs if e.get("ev") == "rewind_arm")
    out["rewind_n"] = len(rw)
    out["rewind_skip_n"] = sum(1 for e in evs if e.get("ev") == "rewind_skip")
    out["rewind_events"] = [{k: e.get(k) for k in ("from", "to", "reason", "tok", "regen")}
                            for e in rw]
    out["regen_total"] = rw[-1].get("regen") if rw else 0

    # retraction-aware reconstruction
    tok_csv = rdir / "tokens.csv"
    if tok_csv.exists():
        kept, jumps = replay_tokens(tok_csv)
        retracted = "".join(p for _, p in kept)
        out["tokens_final"] = len(kept)
        out["token_jumps"] = jumps
        onset2, period2 = loop_onset_char(retracted)
        out["retracted_chars"] = len(retracted)
        out["retracted_loop_onset_char"] = onset2
        out["retracted_useful_frac"] = (onset2 / len(retracted)) if (onset2 is not None and retracted) else 1.0
        lvl, det = grade(frozen_prefix + retracted)
        out["retracted_l0l3"] = lvl
        out["retracted_html_close"] = (frozen_prefix + retracted).lower().count("</html>")
        (rdir / "deliverable_retracted.html").write_text(
            frozen_prefix + retracted, encoding="utf-8", newline="\n")
    return out


def main():
    for arm in sys.argv[1:]:
        arm = pathlib.Path(arm)
        rows = []
        summary = {}
        s = arm / "summary.csv"
        if s.exists():
            with open(s, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    summary[f"r{int(r['run_index']):02d}"] = r
        for rdir in sorted(arm.glob("W050/r*")):
            frozen = (rdir / "frozen.txt").read_text(encoding="utf-8", errors="ignore") \
                if (rdir / "frozen.txt").exists() else ""
            row = analyze_run(rdir, frozen)
            srow = summary.get(rdir.name, {})
            row["l0l3"] = srow.get("l0l3")
            row["html_close"] = srow.get("html_close")
            row["chars"] = srow.get("chars")
            row["repeat"] = srow.get("repeat")
            rows.append(row)
        (arm / "pivotal_metrics.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n==== {arm.name} ====")
        for r in rows:
            print(f"  {r['run']}: L={r.get('l0l3')} close={r.get('html_close')} "
                  f"onset={r['loop_onset_char']}/{r['p2_chars']}ch "
                  f"useful={r['useful_frac_char']:.2f} tps={r.get('p2_gen_tps')} "
                  f"gtps={r.get('good_tps') and round(r['good_tps'],2)} "
                  f"rw(arm/fire/skip)={r['rewind_arm_n']}/{r['rewind_n']}/{r['rewind_skip_n']}"
                  + (f" retrL={r.get('retracted_l0l3')}" if 'retracted_l0l3' in r else ""))


if __name__ == "__main__":
    main()
