#!/usr/bin/env python3
"""POD3 S3 A/B analysis: grade + admit-event metrics per cell. Run locally after exfil."""
import sys, os, re, json, glob, importlib.util, statistics

REPO=r"C:/Users/imanu/source/repos/reap-loop"
def load(name):
    spec=importlib.util.spec_from_file_location(name, os.path.join(REPO,"scripts",name+".py"))
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
fg=load("functional_grade")

_TPS=re.compile(r"prefill:\s*([0-9.]+)\s*t/s,\s*generation:\s*([0-9.]+)\s*t/s")
_REPEAT=re.compile(r"(.{16,350})\1\1", re.S)  # broadened to catch long-period degeneration (e.g. 301-char tag-list loop)
WARMUP=50

def analyze_cell(d):
    name=os.path.basename(d)
    out=open(os.path.join(d,"out.txt"),encoding="utf-8",errors="ignore").read() if os.path.exists(os.path.join(d,"out.txt")) else ""
    diag=open(os.path.join(d,"diag.txt"),encoding="utf-8",errors="ignore").read() if os.path.exists(os.path.join(d,"diag.txt")) else ""
    # grade
    try: L,det=fg.grade_frontpage(out)
    except Exception as e: L,det=-1,{"err":str(e)}
    html=fg.extract_html(out)
    has_close="</html>" in html.lower()
    m=_REPEAT.search(fg.scrub(out)); loop=bool(m); loop_at=(out.find(m.group(1)) if m else -1)
    tps=_TPS.search(diag); pref=float(tps.group(1)) if tps else None; gen=float(tps.group(2)) if tps else None
    # admit events
    ev=[]
    pj=os.path.join(d,"pace.jsonl")
    if os.path.exists(pj):
        for ln in open(pj,encoding="utf-8",errors="ignore"):
            ln=ln.strip()
            if '"ev":"admit"' in ln:
                try: ev.append(json.loads(ln))
                except: pass
    admits=[e for e in ev if e.get("ev")=="admit"]
    toks=[e["tok"] for e in admits] if admits else []
    first=min(toks) if toks else None
    last=max(toks) if toks else None
    span=(last-WARMUP) if last and last>WARMUP else 0
    # per-100tok over post-warmup token span actually generated
    gen_toks=None
    mg=re.search(r"generation:\s*[0-9.]+\s*t/s.*?\(([0-9]+)\s*tok", diag)
    per100 = (len(admits)/(span/100.0)) if span>0 else 0.0
    # bounce: admitted (layer,expert) later evicted at same layer within 100 tok
    bounces=0
    admitted={}  # (layer,expert)->tok
    for e in admits:
        key=(e["layer"],e["expert"]); admitted[key]=e["tok"]
    for e in admits:
        vkey=(e["layer"],e["evicted"])
        if vkey in admitted and 0 < (e["tok"]-admitted[vkey]) <= 100:
            bounces+=1
    layers=len(set(e["layer"] for e in admits))
    # temporal buckets (per 200 tok)
    buckets={}
    for t in toks:
        b=(t//200)*200; buckets[b]=buckets.get(b,0)+1
    return dict(name=name, L=L, has_close=has_close, loop=loop, loop_at=loop_at,
        chars=len(out), pref=pref, gen=gen, n_admit=len(admits), first_admit=first,
        last_admit=last, per100=round(per100,1), bounces=bounces, layers=layers,
        buckets=buckets, det=det)

def main(root):
    cells=sorted(glob.glob(os.path.join(root,"*_r*")))
    rows=[analyze_cell(d) for d in cells]
    # print table
    print(f"{'cell':<20}{'L':>3}{'close':>6}{'loop':>5}{'chars':>7}{'gen_tps':>8}{'admit':>6}{'1st':>5}{'/100':>6}{'bnc':>4}{'lyr':>4}")
    for r in rows:
        print(f"{r['name']:<20}{r['L']:>3}{str(r['has_close']):>6}{str(r['loop']):>5}{r['chars']:>7}{str(r['gen']):>8}{r['n_admit']:>6}{str(r['first_admit']):>5}{r['per100']:>6}{r['bounces']:>4}{r['layers']:>4}")
    # per-arm medians
    for arm in ("A","B"):
        ar=[r for r in rows if f"_{arm}_r" in r['name']]
        if not ar: continue
        Ls=[r['L'] for r in ar]
        print(f"\nARM {arm}: n={len(ar)} L={Ls} L_median={statistics.median(Ls)} close={[r['has_close'] for r in ar]} loop={[r['loop'] for r in ar]} admit={[r['n_admit'] for r in ar]} per100={[r['per100'] for r in ar]} first={[r['first_admit'] for r in ar]} bounces={sum(r['bounces'] for r in ar)}")
    # dump json
    json.dump(rows, open(os.path.join(root,"analysis.json"),"w"), indent=1, default=str)
    print("\n[wrote analysis.json]")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else ".")
