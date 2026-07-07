"""measure_modes_granite.py — confronto miss_mode su Granite-3B: hard-drop vs precision-cascade.

Granite-3.1-3B-A800M entra in 12GB a fp16 (no offload) -> modifica pesi in-place pulita. Stesso
modello, stesso router (selezione invariata): cambia solo la QUALITA' degli expert non residenti.

Protocollo: warmup FULL (baseline + traccia/working set) -> snapshot pesi -> per ogni (modo, K):
ripristina, applica il modo ai non residenti, rigenera, valida con gli unit-test. -> superficie
accuratezza-vs-VRAM con UNA curva per modo.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict

HF = "ibm-granite/granite-3.1-3b-a800m-instruct"
DATASET = "data/codegen_problems.jsonl"
K_FRACTIONS = [0.50, 0.25, 0.12, 0.06]
# (label, mode, nbits)
CELLS = [
    ("cascade-int8", "precision-cascade", 8),
    ("cascade-int4", "precision-cascade", 4),
    ("cascade-int2", "precision-cascade", 2),
    ("hard-drop", "hard-drop", None),
]


def resident_topk(trace_path, n):
    by_layer = defaultdict(Counter)
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_layer[int(r["layer"])].update(int(x) for x in r["topk_ids"])
    return {l: {e for e, _ in c.most_common(n)} for l, c in by_layer.items()}


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--warmup-only", action="store_true", help="solo baseline FULL + traccia, poi esci.")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.enforce.granite_cascade import snapshot_experts, restore_experts, apply_mode

    os.makedirs("runs", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n_exp = int(model.config.num_local_experts)
    print(f"[measure] {HF} | n_experts={n_exp} | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    validator = PythonUnitTestValidator(DATASET)
    items = validator.items()

    def generate(prompt):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=192, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---- warmup + baseline FULL + traccia ----
    trace_path = "runs/modes_granite_trace.jsonl"
    writer = TraceWriter(trace_path)
    logger = RouterLogger(model, RouterHookSpec.for_model(HF), writer)
    full_correct = 0
    with logger.capture(session_id="warmup", ctx_len=0):
        for it in items:
            full_correct += int(validator.verify(it, generate(it.prompt)))
    writer.close()
    full_acc = full_correct / len(items)
    print(f"\n[BASELINE] FULL accuracy = {full_acc:.2f} ({full_correct}/{len(items)})  <<< check\n")
    if args.warmup_only:
        print("[measure] --warmup-only: traccia salvata, esco.")
        return 0

    snap = snapshot_experts(model)
    rows = [{"mode": "FULL", "K_target": 1.0, "resident_frac": 1.0, "nbits": "", "accuracy": full_acc}]

    for kf in K_FRACTIONS:
        n = max(1, math.ceil(kf * n_exp))
        resident = resident_topk(trace_path, n)
        frac = statistics.mean(len(s) / n_exp for s in resident.values())
        for label, mode, nbits in CELLS:
            restore_experts(model, snap)
            apply_mode(model, resident, mode, nbits=nbits or 4)
            correct = sum(int(validator.verify(it, generate(it.prompt))) for it in items)
            acc = correct / len(items)
            rows.append({"mode": label, "K_target": kf, "resident_frac": round(frac, 4),
                         "nbits": nbits or "", "accuracy": acc})
            print(f"[K={kf:g} n={n}] {label:>13}: acc={acc:.2f} ({correct}/{len(items)}) "
                  f"drop_vs_FULL={acc-full_acc:+.2f}")
        restore_experts(model, snap)

    with open("runs/modes_granite.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== SUPERFICIE accuratezza-vs-VRAM (Granite-3B) — FULL={full_acc:.2f} ===")
    print(f"{'mode':>14} | {'K':>5} | {'cut':>5} | accuracy")
    print("-" * 44)
    for r in rows[1:]:
        cut = 1.0 - r["resident_frac"]
        print(f"{r['mode']:>14} | {r['K_target']:>5g} | {cut*100:>4.0f}% | {r['accuracy']:.2f}")
    print("[measure] CSV: runs/modes_granite.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
