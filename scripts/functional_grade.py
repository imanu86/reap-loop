"""Eval funzionale GRADUATO (L0-L3) su task autoconclusivi — mandato SPEX-main.

repeat-rate/ppl NON distinguono "sito fatto bene col bottone rotto" da "pagina che non
si apre". Questo grader assegna un LIVELLO funzionale automatico, non un pass/fail:

  FRONTPAGE (HTML single-file: hero + nav + BOTTONE che fa qualcosa + form + CSS/JS):
    L0 = non fa parse / non si apre (catastrofico)
    L1 = si apre ma feature rotta (bottone senza handler, form senza action, JS con errori)
    L2 = tutte le feature presenti, difetti minori (manca un elemento non-critico)
    L3 = pienamente funzionale e pulito
  PYTHON (implementa una funzione):
    L0 = syntax error · L1 = gira ma sbagliata · L2 = passa alcuni unit test · L3 = tutti
  JSON (estrai campi in uno schema):
    L0 = non parse · L1 = parse ma schema sbagliato · L2 = schema valido difetti minori · L3 = esatto

Uso: python functional_grade.py <task=frontpage|python|json> <gen_file> [--json]
Stampa "LIVELLO Lk" + dettaglio dei check. Con --json emette un dict per aggregazione.
Il check JS syntax usa `node --check` se disponibile, altrimenti euristica (bilanciamento +
pattern chiaramente rotti tipo `.metodo() {` inventato).
"""
import json
import re
import subprocess
import sys
from html.parser import HTMLParser


# ---------- pulizia diagnostiche ds4 (se sfuggite nello stdout) ----------
_DS4_DIAG = re.compile(
    r"ds4: ?(REAP mask (applied|reload)|CUDA loading|gpu prefill|prefill:|SSD streaming|"
    r"context buffers|using GPU|streaming initial)[^\n]*")


def scrub(text):
    return _DS4_DIAG.sub("", text)


# ---------- estrazione blocco codice dalla generazione ----------
def extract_block(text, langs):
    text = scrub(text)
    """prende il blocco ```lang ... ``` piu' lungo, altrimenti euristica per tag."""
    fences = re.findall(r"```(?:" + "|".join(langs) + r")?\s*\n(.*?)```", text,
                        re.DOTALL | re.IGNORECASE)
    if fences:
        return max(fences, key=len).strip()
    return text.strip()


def extract_html(text):
    # NB: non dipende dai fence ``` (il doc buono puo' stare FUORI dal fence chiuso quando il
    # modello riparte). Toglie diagnostiche + marcatori fence, poi sceglie il doc piu' completo.
    blk = scrub(text)
    blk = re.sub(r"```[a-zA-Z]*", "", blk)
    starts = [m.start() for m in re.finditer(r"(<!doctype html|<html)", blk, re.IGNORECASE)]
    if starts:
        bounds = starts + [len(blk)]
        cands = [blk[bounds[i]:bounds[i + 1]] for i in range(len(starts))]
        # piu' completo: ha </html>, poi ha <script>, poi il piu' lungo
        cands.sort(key=lambda c: ("</html>" in c.lower(), "<script" in c.lower(), len(c)))
        blk = cands[-1]
    else:
        m = re.search(r"(<head|<body|<div|<section|<main)", blk, re.IGNORECASE)
        if m:
            blk = blk[m.start():]
    last = max((blk.rfind(t) for t in ("</html>", "</body>", "</div>", "</section>")), default=-1)
    if last > 0:
        end = blk.find(">", last)
        if end > 0:
            blk = blk[:end + 1]
    return blk


# ---------- parser HTML tollerante ----------
class Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.stack = []
        self.mismatch = 0
        self.attrs_by_tag = []
        self.script_text = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs_by_tag.append((tag, dict(attrs)))
        if tag not in ("br", "img", "input", "meta", "link", "hr"):
            self.stack.append(tag)
        if tag == "script":
            self._in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        if self.stack:
            if self.stack[-1] == tag:
                self.stack.pop()
            elif tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    self.mismatch += 1
            else:
                self.mismatch += 1

    def handle_data(self, data):
        if self._in_script:
            self.script_text.append(data)


def js_syntax_errors(js):
    """0 = valido. Prova node --check; fallback euristico."""
    js = js.strip()
    if not js:
        return 0
    try:
        p = subprocess.run(["node", "--check"], input=js, text=True,
                           capture_output=True, timeout=15)
        return 0 if p.returncode == 0 else 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # euristica: bilanciamento parentesi + pattern rotti noti
    errs = 0
    for op, cl in [("(", ")"), ("{", "}"), ("[", "]")]:
        if js.count(op) != js.count(cl):
            errs += 1
    # metodo-inventato tipo ".clicked() {" o "onclick() {" fuori da una funzione dichiarata
    if re.search(r"\.\w+\s*\([^)]*\)\s*\{", js) and "function" not in js and "=>" not in js:
        errs += 1
    return errs


def grade_frontpage(text):
    html = extract_html(text)
    # restart del documento (piu' <!doctype) = difetto reale: cap a L2 anche se il doc buono c'e'
    restart = len(re.findall(r"<!doctype html", scrub(text), re.IGNORECASE)) > 1
    c = Collector()
    try:
        c.feed(html)
    except Exception:
        return 0, {"parse": "exception"}
    tags = set(c.tags)
    det = {}
    det["has_body_struct"] = bool(tags & {"body", "main", "div", "section"})
    det["nav"] = "nav" in tags or any("nav" in (a.get("class", "") + a.get("id", ""))
                                       for t, a in c.attrs_by_tag)
    det["hero"] = bool(tags & {"header", "h1"}) or any("hero" in (a.get("class", ""))
                                                       for t, a in c.attrs_by_tag)
    det["form"] = "form" in tags
    det["style"] = "style" in tags or any("style" in a for t, a in c.attrs_by_tag)
    det["script"] = "script" in tags
    # bottone
    buttons = [(t, a) for t, a in c.attrs_by_tag
               if t == "button" or (t == "input" and a.get("type") in ("button", "submit"))]
    det["button"] = bool(buttons)
    js = "\n".join(c.script_text)
    det["js_errors"] = js_syntax_errors(js)
    # bottone-ha-handler: onclick inline, oppure lo script referenzia id/classe del bottone
    # con un listener
    btn_wired = False
    for t, a in buttons:
        if a.get("onclick"):
            btn_wired = True
        bid = a.get("id")
        if bid and (f"getElementById('{bid}'" in js or f'getElementById("{bid}"' in js
                    or f"#{bid}" in js) and ("addeventlistener" in js.lower() or "onclick" in js.lower()):
            btn_wired = True
    if buttons and not btn_wired and "addeventlistener" in js.lower():
        btn_wired = True  # listener generico (querySelector('button') ecc.)
    det["button_wired"] = btn_wired
    # form-ha-action: attr action oppure submit-handler
    form_wired = False
    for t, a in c.attrs_by_tag:
        if t == "form" and (a.get("action") or a.get("onsubmit")):
            form_wired = True
    if det["form"] and ("addeventlistener('submit'" in js.lower().replace('"', "'")
                        or "onsubmit" in js.lower()):
        form_wired = True
    det["form_wired"] = form_wired
    det["tag_mismatch"] = c.mismatch

    # rubrica
    if not det["has_body_struct"] or len(c.tags) < 4:
        return 0, det
    critical_ok = det["button"] and det["button_wired"] and det["js_errors"] == 0
    form_ok = (det["form_wired"] if det["form"] else True)
    if not critical_ok or not form_ok:
        return 1, det
    required = ["nav", "hero", "form", "style", "script", "button"]
    present = sum(1 for r in required if det[r])
    det["restart"] = restart
    if present == len(required) and det["tag_mismatch"] == 0 and not restart:
        return 3, det
    return 2, det  # feature ci sono ma difetti minori (o restart del documento)


def _function_blocks(code, name=None):
    """tutti i blocchi def (def-line + corpo indentato) — gestisce ripetizioni dai due-fase."""
    code = re.sub(r"```[a-zA-Z]*", "", code)  # toglie marcatori fence ripetuti
    lines = code.splitlines()
    pat = re.compile(rf"^\s*def\s+{name}\b") if name else re.compile(r"^\s*def\s+\w+")
    starts = [i for i, ln in enumerate(lines) if pat.match(ln)]
    blocks = []
    for s in starts:
        indent = len(lines[s]) - len(lines[s].lstrip())
        out = [lines[s]]
        for ln in lines[s + 1:]:
            if ln.strip() == "" or (len(ln) - len(ln.lstrip())) > indent:
                out.append(ln)
            else:
                break
        blocks.append("\n".join(out))
    return blocks or [code]


def grade_python(text, unit_tests=None, func_name=None):
    # opera sul testo PIENO (fence-scrubbed): i due-fase ripetono la funzione, la completa
    # puo' stare in un fence successivo che extract_block non catturerebbe.
    raw = re.sub(r"```[a-zA-Z]*", "", scrub(text))
    det = {}

    def norm(x):
        return [norm(e) for e in x] if isinstance(x, (list, tuple)) else x

    def run_tests(ns):
        p = 0
        for args, expected in unit_tests or []:
            try:
                p += int(norm(ns[func_name](*args)) == norm(expected))
            except Exception:
                pass
        return p

    # prova ogni candidato-funzione (i due-fase ripetono: la completa puo' essere la 2a)
    best = None  # (livello, passed, det)
    for cand in [raw] + _function_blocks(raw, func_name):
        try:
            compile(cand, "<gen>", "exec")
        except SyntaxError:
            continue
        ns = {}
        try:
            exec(cand, ns)
        except Exception:
            best = best or (1, 0, {"syntax": "ok", "exec": "error"})
            continue
        if func_name and func_name not in ns:
            continue
        if not unit_tests:
            return 2, {"syntax": "ok"}
        passed = run_tests(ns)
        lvl = 3 if passed == len(unit_tests) else (2 if passed > 0 else 1)
        cd = {"syntax": "ok", "passed": f"{passed}/{len(unit_tests)}"}
        if best is None or (lvl, passed) > (best[0], best[1]):
            best = (lvl, passed, cd)
        if lvl == 3:
            break
    if best is None:
        return 0, {"syntax": "error"}
    lvl, _, det = best
    return lvl, det


def _first_json(blk):
    """primo oggetto/array JSON BILANCIATO (ignora ripetizioni/junk dopo la risposta)."""
    m = re.search(r"[\{\[]", blk)
    if not m:
        return blk
    start = m.start()
    open_c = blk[start]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(blk)):
        ch = blk[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return blk[start:i + 1]
    return blk[start:]


def grade_json(text, required_keys=None, expected=None):
    blk = _first_json(extract_block(text, ["json"]))
    det = {}
    try:
        obj = json.loads(blk)
        det["parse"] = "ok"
    except Exception as e:
        det["parse"] = f"error: {e}"
        return 0, det
    if not required_keys:
        return 2, det
    if not isinstance(obj, dict):
        return 1, det
    present = [k for k in required_keys if k in obj]
    det["keys"] = f"{len(present)}/{len(required_keys)}"
    if len(present) < len(required_keys):
        return (2 if present else 1), det
    # tutte le chiavi presenti: L3 solo se i valori sono esatti (se dati), altrimenti L2
    if expected:
        def eq(a, b):
            return str(a).strip().lower() == str(b).strip().lower()
        ok = all(eq(obj.get(k), v) for k, v in expected.items())
        det["values_exact"] = ok
        return (3 if ok else 2), det
    return 3, det


def main():
    task = sys.argv[1]
    gen = open(sys.argv[2], encoding="utf-8", errors="ignore").read()
    as_json = "--json" in sys.argv
    if task == "frontpage":
        lvl, det = grade_frontpage(gen)
    elif task == "python":
        lvl, det = grade_python(gen)
    elif task == "json":
        lvl, det = grade_json(gen)
    else:
        print("task sconosciuto"); sys.exit(2)
    if as_json:
        print(json.dumps({"task": task, "level": lvl, "detail": det}))
    else:
        print(f"LIVELLO L{lvl}")
        for k, v in det.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
