#!/usr/bin/env python3
"""Emit warmup + measured chat-completion request JSONs for the COFFEE landing-page ARM.
Task prompt (temp=0, deterministic). Single-file HTML/CSS/JS artisan-coffee landing page.
"""
import json, sys, os

SYSTEM = ("Sei un web developer esperto. Rispondi con un UNICO file HTML completo e "
          "renderizzabile, con CSS e JavaScript inline nello stesso file. "
          "Nessuna spiegazione: emetti solo il codice, a partire da <!DOCTYPE html>.")
USER = ("Genera una landing page HTML/CSS/JS single-file per una caffetteria artigianale, "
        "completa e renderizzabile.")

def make(max_tokens):
    return {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "think": False,
        "thinking": {"type": "disabled"},
    }

if __name__ == "__main__":
    outdir = sys.argv[1]
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    meas = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
    os.makedirs(outdir, exist_ok=True)
    json.dump(make(warm), open(os.path.join(outdir, "request_warmup.json"), "w"), indent=2)
    json.dump(make(meas), open(os.path.join(outdir, "request_measured.json"), "w"), indent=2)
    print("wrote warmup(%d)+measured(%d) -> %s" % (warm, meas, outdir))
