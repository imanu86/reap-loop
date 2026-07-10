#!/usr/bin/env python3
"""Retro-grade funzionale L0-L3 di tutti gli output ds4 archiviati.

Percorre runs/ds4/**/content_measured.txt e assegna un LIVELLO funzionale (L0-L3)
usando il grader scripts/functional_grade.py (portato dal branch reap/k91-coding-vram
del repo gemello moe-aggressive-commit). La rubrica del grader e' tenuta INVARIATA per
comparabilita' fra run.

Per i prompt HTML (frontpage cyberpunk single-file):
  - grade_frontpage → livello L0..L3 + dict `det` completo (tutte le sotto-colonne)
  - colonna extra `alert_in_script`: rileva alert(/popup SOLO dentro i blocchi <script>
    parsati (mai come substring del testo grezzo). Serve perche' il flag has_popup del
    runner e' BUGGATO: matcha l'eco del prompt ("...un popup JS che dice...") e non il
    codice generato.
Per i prompt code / code_mini (review di pseudocodice + patch plan, output prosa):
  - solo syntax-level: estrae l'eventuale blocco di codice e prova compile() Python →
    colonna `py_syntax_ok`; marcato "syntax-only, no unit tests". Nessun L-level assegnato
    (la rubrica frontpage non si applica). Vedi nota di bias nel REPORT.

Metadati agganciati per riga (se disponibili): suite/famiglia, variant, prompt, prompt_tokens,
completion_tokens, avg_tps, repeat_flag, has_popup(runner), doctype, html_balance,
s_init_count, content_chars — presi dal summary.csv della famiglia quando esiste, altrimenti
dal runner_manifest.json / matrix_config.json / nome-variante.

NOTA METODOLOGICA: questi grade sono RETROATTIVI (grader v-k91) su output n=1 greedy. Il
repeat/render precedente resta a ledger; il retro-grade e' una COLONNA NUOVA di evidenza,
NON sostituisce i replay.

Uso: python scripts/retro_grade_l0l3.py
Output: runs/ds4/20260710_retro_grade_l0l3/graded.csv (+ REPORT.md scritto a mano dopo).
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import functional_grade as fg  # noqa: E402

REPO = HERE.parent
RUNS = REPO / "runs" / "ds4"
OUTDIR = RUNS / "20260710_retro_grade_l0l3"


# ---------- classificazione famiglia (per la tabella del REPORT) ----------
def family_of(top):
    t = top
    if "exchange_matrix" in t:
        return "exchange_matrix"
    if t.startswith("20260709_breath_"):
        return "breath"
    if "descent" in t or "prebreath" in t:
        return "descent_prebreath"
    if "stepdown" in t or "pace_" in t:
        return "stepdown_pace"
    if "sota_candidates" in t:
        return "sota_candidates"
    if "cache_sweep" in t:
        return "cache_sweep"
    if "requested4" in t:
        return "requested4"
    if t.startswith("20260710_w100") or t.startswith("20260710_w50"):
        return "w_runs"
    if "pod_" in t:
        return "pod"
    if "trace_ab" in t:
        return "trace_ab"
    if "rotate_smoke" in t:
        return "rotate_smoke"
    if "k23_unit" in t or "k23_weighted" in t or "direct_k23_vs_stepdown" in t:
        return "k23_unit"
    return "other"


# ---------- tipo prompt ----------
def _normalize(label):
    """label grezza (es. 'html_compact_budget') → tipo canonico {html, code_mini, code}."""
    lo = (label or "").strip().lower()
    if lo.startswith("code_mini"):
        return "code_mini"
    if lo.startswith("code"):
        return "code"
    if lo.startswith("html"):
        return "html"
    return "unknown"


def prompt_type(leaf_name, summary_row, manifest):
    """priorita': summary.csv 'prompt' > manifest prompt.name > prefisso nome-variante.
    Ritorna (tipo_canonico, sorgente, label_grezza)."""
    if summary_row and summary_row.get("prompt"):
        raw = summary_row["prompt"].strip()
        return _normalize(raw), "summary", raw
    if manifest:
        p = manifest.get("prompt", {})
        if isinstance(p, dict) and p.get("name"):
            raw = p["name"].strip()
            return _normalize(raw), "manifest", raw
    ln = leaf_name.lower()
    if ln.startswith("code_mini"):
        return "code_mini", "leafname", "code_mini"
    if ln.startswith("code_"):
        return "code", "leafname", "code"
    if ln.startswith("html"):
        return "html", "leafname", "html"
    return "unknown", "leafname", leaf_name


# ---------- alert()/popup SOLO dentro <script> parsati ----------
def alert_in_script(text):
    html = fg.extract_html(text)
    c = fg.Collector()
    try:
        c.feed(html)
    except Exception:
        return False
    js = "\n".join(c.script_text)
    return bool(re.search(r"alert\s*\(|popup", js, re.IGNORECASE))


# ---------- syntax-level per prompt code (compile Python) ----------
def py_syntax_check(text):
    """Ritorna (py_syntax_ok:bool, has_code_fence:bool). Solo compile, nessun unit test."""
    scrub = fg.scrub(text)
    has_fence = bool(re.search(r"```(?:python|py)\b", scrub, re.IGNORECASE)
                     or re.search(r"```\s*\n\s*(?:def |import |from |class )", scrub))
    block = fg.extract_block(text, ["python", "py"])
    block = re.sub(r"```[a-zA-Z]*", "", block).strip()
    if not block:
        return False, has_fence
    try:
        compile(block, "<gen>", "exec")
        return True, has_fence
    except SyntaxError:
        return False, has_fence
    except Exception:
        return False, has_fence


DET_KEYS = ["has_body_struct", "nav", "hero", "form", "style", "script", "button",
            "button_wired", "form_wired", "js_errors", "tag_mismatch", "restart"]

SUMMARY_META = ["variant", "prompt", "prompt_tokens", "completion_tokens", "avg_tps",
                "repeat_flag", "has_popup", "doctype", "html_balance", "s_init_count",
                "content_chars"]


def load_summary(top_dir):
    p = top_dir / "summary.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            stem = (row.get("stem") or "").strip()
            if stem:
                out[stem] = row
    return out


def main():
    leaves = sorted(RUNS.glob("*/*/content_measured.txt"))
    summaries = {}  # top_dir_path -> {stem: row}
    rows = []
    for leaf in leaves:
        leaf_dir = leaf.parent
        top_dir = leaf_dir.parent
        top = top_dir.name
        stem = leaf_dir.name
        if top_dir not in summaries:
            summaries[top_dir] = load_summary(top_dir)
        srow = summaries[top_dir].get(stem)

        manifest = None
        mpath = leaf_dir / "runner_manifest.json"
        if mpath.exists():
            try:
                manifest = json.load(open(mpath, encoding="utf-8"))
            except Exception:
                manifest = None

        text = leaf.read_text(encoding="utf-8", errors="ignore")
        ptype, psrc, plabel = prompt_type(stem, srow, manifest)

        rec = {
            "path": str(leaf.relative_to(REPO)).replace("\\", "/"),
            "family": family_of(top),
            "top_dir": top,
            "runner_id": stem,
            "prompt_type": ptype,
            "prompt_label": plabel,
            "prompt_src": psrc,
            "content_len": len(text),
            "level": "",
            "alert_in_script": "",
            "py_syntax_ok": "",
            "has_code_fence": "",
            "note": "",
        }
        for k in DET_KEYS:
            rec[k] = ""

        if ptype == "html":
            lvl, det = fg.grade_frontpage(text)
            rec["level"] = lvl
            for k in DET_KEYS:
                rec[k] = det.get(k, "")
            rec["alert_in_script"] = int(alert_in_script(text))
            rec["note"] = "frontpage rubric (invariata v-k91)"
        elif ptype in ("code", "code_mini"):
            ok, fence = py_syntax_check(text)
            rec["py_syntax_ok"] = int(ok)
            rec["has_code_fence"] = int(fence)
            rec["note"] = "syntax-only, no unit tests"
        else:
            rec["note"] = "prompt-type unknown; not graded"

        # metadati dal summary della famiglia
        for m in SUMMARY_META:
            rec[m] = (srow.get(m) if srow else "") or ""
        # variant fallback dal manifest se il summary manca
        if not rec["variant"] and manifest:
            rec["variant"] = manifest.get("variant", "") or ""
        rec["has_summary"] = int(srow is not None)

        rows.append(rec)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cols = (["path", "family", "top_dir", "runner_id", "prompt_type", "prompt_label",
             "prompt_src", "level"] + DET_KEYS + ["alert_in_script", "py_syntax_ok",
             "has_code_fence", "content_len", "note"] + SUMMARY_META + ["has_summary"])
    out_csv = OUTDIR / "graded.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # ---------- riepiloghi rapidi su stdout (per costruire il REPORT a mano) ----------
    def dump(title, items):
        print(f"\n=== {title} ===")
        for k, v in items:
            print(f"  {k}: {v}")

    html = [r for r in rows if r["prompt_type"] == "html"]
    code = [r for r in rows if r["prompt_type"] in ("code", "code_mini")]
    from collections import Counter, defaultdict
    lvl_counts = Counter(r["level"] for r in html)
    dump("HTML level distribution", sorted(lvl_counts.items()))
    print(f"  (n_html={len(html)}, n_code={len(code)})")

    # per famiglia
    fam = defaultdict(lambda: Counter())
    for r in html:
        fam[r["family"]][r["level"]] += 1
    print("\n=== HTML L-dist per family ===")
    for f in sorted(fam):
        c = fam[f]
        print(f"  {f}: L0={c[0]} L1={c[1]} L2={c[2]} L3={c[3]} (n={sum(c.values())})")

    # repeat_flag vs L
    print("\n=== repeat_flag vs L-level (html con summary) ===")
    rep0_lowL = [r for r in html if str(r["repeat_flag"]) == "0" and r["level"] in (0, 1)]
    rep1 = [r for r in html if str(r["repeat_flag"]) == "1"]
    withrep = [r for r in html if str(r["repeat_flag"]) in ("0", "1")]
    print(f"  html with repeat_flag available: {len(withrep)}")
    print(f"  repeat_flag=1: {len(rep1)}")
    print(f"  repeat_flag=0 AND L0/L1 (proxy overrates): {len(rep0_lowL)}")
    for r in rep0_lowL:
        print(f"    - {r['top_dir']}/{r['runner_id']} L{r['level']}")

    # decision cases
    print("\n=== DECISION CASES ===")
    for r in rows:
        if r["prompt_type"] != "html":
            continue
        t, rid = r["top_dir"], r["runner_id"]
        hit = (("requested4" in t and "rotate32" in rid)
               or ("requested4" in t and rid.startswith("html_local_k23_cache"))
               or ("pod_e7w4_static" in t)
               or ("pod_qo6_rotate32" in t) or ("pod_id63_rotate16" in t)
               or (t.startswith("20260710_w100")) or (t.startswith("20260710_w50")))
        if hit:
            print(f"  L{r['level']} rep={r['repeat_flag']} alert_scr={r['alert_in_script']} "
                  f"| {t}/{rid}")


if __name__ == "__main__":
    main()
