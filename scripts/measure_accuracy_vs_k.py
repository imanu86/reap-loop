"""measure_accuracy_vs_k.py — PRIMO punto della superficie accuratezza-vs-VRAM (OLMoE, hard-drop).

Protocollo (docs/00_architecture.md §9, miss_mode=hard-drop, policy=AGGRESSIVE-COMMIT):
  1. WARMUP+BASELINE: genera le soluzioni sotto FULL (nessun enforcement) catturando la traccia di
     routing -> dà sia la baseline di accuratezza (K=100%) sia il working set della sessione.
  2. Per ogni K (frazione di expert residenti): resident_by_layer = top-ceil(K*64) per frequenza,
     per layer (dalla traccia di warmup). attach_hard_drop maschera il resto -> il modello instrada
     SOLO tra i residenti.
  3. Rigenera le soluzioni sotto enforcement e valida con gli unit-test nascosti.
  4. Registra (K, frazione residente media, accuratezza). Cut VRAM_expert = 1 - frazione residente.

NB: prima misura a contesto NATURALE (prompt corti, niente padding). La crescita di contesto
(Vincolo B) è il passo successivo. Warmup = stessi 10 problemi (setup "osserva la sessione, poi
committa") — circolarità accettabile per il primo punto, da separare poi.

Uso (offload, lento): python scripts/measure_accuracy_vs_k.py --max-new 192
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict

HF = "allenai/OLMoE-1B-7B-0924-Instruct"
N_EXPERTS = 64
K_FRACTIONS = [0.50, 0.25, 0.12, 0.06]   # K=1.0 (FULL) è la baseline, misurata a parte
DATASET = "data/codegen_problems.jsonl"


def resident_topk(trace_path: str, n: int) -> dict[int, set[int]]:
    """Per layer: i top-n expert per frequenza nella traccia di warmup."""
    by_layer: dict[int, Counter] = defaultdict(Counter)
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_layer[int(r["layer"])].update(int(x) for x in r["topk_ids"])
    return {layer: {e for e, _ in c.most_common(n)} for layer, c in by_layer.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-new", type=int, default=192)
    p.add_argument("--gpu-mem", default="9GiB")
    p.add_argument("--out", default="runs/accuracy_vs_k_olmoe.csv")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.enforce import attach_hard_drop

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(
        HF, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": "48GiB"})
    model.eval()
    print(f"[measure] VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB, model={HF}")

    validator = PythonUnitTestValidator(DATASET)
    items = validator.items()
    print(f"[measure] {len(items)} problemi di validazione")

    def generate(prompt: str) -> str:
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to("cuda")
        else:
            enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---- 1. WARMUP + BASELINE (FULL, nessun enforcement), cattura traccia ----
    trace_path = "runs/accuracy_vs_k_olmoe_trace.jsonl"
    writer = TraceWriter(trace_path)
    spec = RouterHookSpec.for_model(HF)
    logger = RouterLogger(model, spec, writer)
    rows = []
    full_correct = 0
    with logger.capture(session_id="measure", ctx_len=0):
        for it in items:
            ok = validator.verify(it, generate(it.prompt))
            full_correct += int(ok)
            print(f"[FULL] {it.item_id}: {'PASS' if ok else 'fail'}")
    writer.close()
    full_acc = full_correct / len(items)
    rows.append({"K_target": 1.0, "resident_frac": 1.0, "accuracy": full_acc,
                 "n_correct": full_correct, "miss_mode": "none(FULL)"})
    print(f"[measure] BASELINE FULL accuracy = {full_acc:.2f} ({full_correct}/{len(items)})")

    # ---- 2-4. Per ogni K: commit top-K, enforce hard-drop, rigenera, valida ----
    for kf in K_FRACTIONS:
        n = max(1, math.ceil(kf * N_EXPERTS))
        resident = resident_topk(trace_path, n)
        frac = statistics.mean(len(s) / N_EXPERTS for s in resident.values())
        handle = attach_hard_drop(model, resident)
        correct = 0
        for it in items:
            ok = validator.verify(it, generate(it.prompt))
            correct += int(ok)
            print(f"[K={kf:g} n={n}] {it.item_id}: {'PASS' if ok else 'fail'}")
        handle.remove()
        acc = correct / len(items)
        rows.append({"K_target": kf, "resident_frac": round(frac, 4), "accuracy": acc,
                     "n_correct": correct, "miss_mode": "hard-drop"})
        print(f"[measure] K={kf:g} (n={n}/64, frac={frac:.3f}) accuracy = {acc:.2f} "
              f"({correct}/{len(items)})  drop_vs_full={full_acc-acc:+.2f}")

    # ---- CSV + tabella ----
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[measure] === SUPERFICIE accuratezza-vs-VRAM (OLMoE, hard-drop, ctx naturale) ===")
    print(f"{'K':>6} | {'resid_frac':>10} | {'cut_VRAM':>8} | {'accuracy':>8} | drop_vs_FULL")
    print("-" * 60)
    for r in rows:
        cut = 1.0 - r["resident_frac"]
        print(f"{r['K_target']:>6g} | {r['resident_frac']:>10.3f} | {cut*100:>6.1f}% | "
              f"{r['accuracy']:>8.2f} | {r['accuracy']-full_acc:+.2f}")
    print(f"[measure] CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
