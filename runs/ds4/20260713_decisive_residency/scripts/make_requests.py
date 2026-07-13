#!/usr/bin/env python3
"""Emit warmup + measured chat-completion request JSONs.

Wide-domain 'cyberpunk storico' prompt: deliberately stresses the expert
working set (history + speculative fiction + prose) so a residency win, if it
exists, must survive a broad token distribution rather than a narrow one.
temp=0 for determinism.
"""
import json, sys, os

SYSTEM = "Sei uno scrittore. Rispondi in prosa continua, senza ragionamento visibile, senza elenchi."
USER = (
    "Scrivi un racconto cyberpunk ambientato nella Roma imperiale del 100 d.C., "
    "come se la tecnologia moderna (reti neurali, impianti cibernetici, megacorporazioni, "
    "droni, realta' aumentata) fosse gia' esistita accanto a senatori, legionari, gladiatori "
    "e schiavi. Segui un ingegnere-liberto che ripara innesti neurali nella Suburra e viene "
    "coinvolto in una cospirazione che attraversa il Senato, i Pretoriani e una corporazione "
    "che vende memoria. Descrivi luoghi, tecnologia, politica e personaggi in dettaglio ricco. "
    "Scrivi un testo lungo e continuo."
)

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
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    meas = int(sys.argv[3]) if len(sys.argv) > 3 else 240
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "request_warmup.json"), "w") as f:
        json.dump(make(warm), f, indent=2)
    with open(os.path.join(outdir, "request_measured.json"), "w") as f:
        json.dump(make(meas), f, indent=2)
    print("wrote warmup(%d) + measured(%d) to %s" % (warm, meas, outdir))
