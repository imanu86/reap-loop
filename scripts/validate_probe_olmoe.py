"""validate_probe_olmoe.py — VALIDAZIONE DEL METODO: predire batte reagire?

Domanda falsificabile: il pre-staging PREDITTIVO della cache expert (dal probe sul prompt) riduce i
miss in generazione vs una cache REATTIVA (cold + LRU, = il meccanismo vero di ds4) abbastanza da
battere il COSTO del probe? A parita' di VRAM (capienza cache) e accuratezza (always-fetch lossless).

Policy confrontate, stesso modello (OLMoE) sullo stesso engine slice-cache:
  - FULL       : capienza=64 (tetto, 0 miss)
  - REACTIVE   : capienza=C, niente pin, LRU a freddo (= ds4 reattivo). La baseline DA BATTERE.
  - PREDICTIVE : capienza=C, pin del working-set predetto dal probe (prefill), poi LRU.
Misure: miss-count (indip. dalla precisione = la leva del probe), byte, tok/s, accuratezza, e il
COSTO del probe (miss del prefill di probe, contati a parte). Net = (react_miss - pred_miss) vs probe_miss.

Asse: capienza cache C (piccola = piu' spazio per il probe; grande = irrilevante). Sessione = i prompt.
Uso: python -u scripts/validate_probe_olmoe.py
"""

from __future__ import annotations

import csv
import os
import time

HF = "allenai/OLMoE-1B-7B-0924-Instruct"
N_EXPERTS = 64
N_PROMPTS = 6
MAX_NEW = 64
CAPS = [4, 8, 16]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.hierarchy.olmoe_cache import install_expert_cache
    from msc.hierarchy.calibrate import resident_from_trace

    tok = AutoTokenizer.from_pretrained(HF)
    print("[val] carico OLMoE su CPU (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers))
    validator = PythonUnitTestValidator("data/codegen_problems.jsonl")
    items = validator.items()[:N_PROMPTS]
    os.makedirs("runs", exist_ok=True)

    def enc_of(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True).to("cuda")

    def gen_session():
        """Genera tutte le soluzioni della sessione; ritorna (accuracy, tok, secondi)."""
        correct, tok_n, t0 = 0, 0, time.perf_counter()
        for it in items:
            enc = enc_of(it.prompt)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = out[0, enc["input_ids"].shape[1]:]
            tok_n += int(new.shape[0])
            correct += int(validator.verify(it, tok.decode(new, skip_special_tokens=True)))
        return correct / len(items), tok_n, time.perf_counter() - t0

    rows = []
    for cap in CAPS:
        pin = max(1, cap - 1)  # pin quasi tutta la cache col working-set predetto (1 slot LRU)
        empty = {i: set() for i in layers}

        # --- PROBE: prefill della sessione a freddo, cattura routing -> working-set predetto ---
        h = install_expert_cache(model, capacity_per_layer=cap, resident_by_layer=empty)
        trace_path = f"runs/val_probe_trace_c{cap}.jsonl"
        writer = TraceWriter(trace_path)
        logger = RouterLogger(model, RouterHookSpec.for_model(HF), writer)
        with logger.capture(session_id="probe", ctx_len=0):
            for it in items:
                with torch.no_grad():
                    model(**enc_of(it.prompt))  # prefill only
        writer.close()
        probe_miss = h.stats()["total"]["misses"]
        probe_fetched = h.stats()["fetched_gb"]
        h.remove()
        predicted = resident_from_trace(trace_path, pin)

        # --- REACTIVE: cold + LRU (baseline ds4) ---
        h = install_expert_cache(model, capacity_per_layer=cap, resident_by_layer=empty)
        acc_r, _, t_r = gen_session()
        st_r = h.stats(); h.remove()

        # --- PREDICTIVE: pin del working-set predetto, poi LRU ---
        h = install_expert_cache(model, capacity_per_layer=cap, resident_by_layer=predicted)
        acc_p, ntok, t_p = gen_session()
        st_p = h.stats(); h.remove()

        react_miss, pred_miss = st_r["total"]["misses"], st_p["total"]["misses"]
        saving = react_miss - pred_miss
        net = saving - probe_miss   # >0 => il probe ripaga il suo costo (in miss)
        rows.append({"cap": cap, "pin": pin,
                     "react_miss": react_miss, "pred_miss": pred_miss, "probe_miss": probe_miss,
                     "saving": saving, "net_vs_probe": net,
                     "react_tok_s": round(ntok / t_r, 1), "pred_tok_s": round(ntok / t_p, 1),
                     "acc_react": acc_r, "acc_pred": acc_p, "acc_match": acc_r == acc_p})
        print(f"[val] cap={cap} pin={pin}: react_miss={react_miss} pred_miss={pred_miss} "
              f"probe_miss={probe_miss} -> saving={saving} net_vs_probe={net} "
              f"(acc r={acc_r:.2f}/p={acc_p:.2f})")

    print("\n=== VALIDAZIONE: predire vs reagire (OLMoE, always-fetch lossless) ===")
    print(f"{'cap':>4} {'pin':>4} | {'react_miss':>10} {'pred_miss':>9} {'probe_miss':>10} | "
          f"{'saving':>7} {'net':>6} | acc r/p")
    print("-" * 78)
    for r in rows:
        print(f"{r['cap']:>4} {r['pin']:>4} | {r['react_miss']:>10} {r['pred_miss']:>9} "
              f"{r['probe_miss']:>10} | {r['saving']:>7} {r['net_vs_probe']:>6} | "
              f"{r['acc_react']:.2f}/{r['acc_pred']:.2f}")
    with open("runs/validate_probe_olmoe.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[val] CSV: runs/validate_probe_olmoe.csv")
    print("[val] VERDETTO: il probe e' validato se net_vs_probe > 0 in qualche regione (cache piccola).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
