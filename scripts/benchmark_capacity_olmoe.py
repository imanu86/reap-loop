"""benchmark_capacity_olmoe.py — il VALORE del probe: footprint-vs-latenza a iso-accuratezza.

OLMoE gira via slice-cache (expert su CPU, working-set in VRAM, always-fetch=lossless). A parita' di
capienza cache, confrontiamo il pinning del working-set CALIBRATO DAL PROBE vs un pinning CIECO:
l'accuratezza e' identica (lossless), ma il probe deve ABBATTERE il fetch-rate (= meno traffico
CPU->GPU = meno latenza). E' la dimostrazione del delta vs offload cieco.

Asse: CAPACITA'(footprint)-vs-LATENZA. Accuratezza ~costante per costruzione (always-fetch).
Uso: python -u scripts/benchmark_capacity_olmoe.py
"""

from __future__ import annotations

import time

HF = "allenai/OLMoE-1B-7B-0924-Instruct"
N_EXPERTS = 64
N_PROMPTS = 6
MAX_NEW = 64
CAPS = [8, 16, 32]   # capienza cache per layer; pinned = cap//2 (resto slot LRU)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.hierarchy.olmoe_cache import install_expert_cache
    from msc.hierarchy.calibrate import resident_from_trace, blind_resident
    from msc.hierarchy import metrics as M

    tok = AutoTokenizer.from_pretrained(HF)
    print("[bench] carico OLMoE su CPU (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16)  # CPU, no device_map
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    bytes_per_expert = None  # stimato da metrics

    validator = PythonUnitTestValidator("data/codegen_problems.jsonl")
    items = validator.items()[:N_PROMPTS]

    def gen_and_check(timed=True):
        """Genera le soluzioni, ritorna (accuracy, tok_per_s)."""
        correct, total_new, t0 = 0, 0, time.perf_counter()
        for it in items:
            enc = tok.apply_chat_template([{"role": "user", "content": it.prompt}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = out[0, enc["input_ids"].shape[1]:]
            total_new += int(new.shape[0])
            correct += int(validator.verify(it, tok.decode(new, skip_special_tokens=True)))
        dt = time.perf_counter() - t0
        return correct / len(items), (total_new / dt if timed else 0.0)

    # --- WARMUP: routing reale (lossless) per stimare il working-set della sessione ---
    print("[bench] warmup: cache piccola + cattura routing (prefill)...")
    h = install_expert_cache(model, capacity_per_layer=8,
                             resident_by_layer=blind_resident(list(range(n_layers)), 8, N_EXPERTS))
    trace_path = "runs/bench_olmoe_trace.jsonl"
    writer = TraceWriter(trace_path)
    logger = RouterLogger(model, RouterHookSpec.for_model(HF), writer)
    with logger.capture(session_id="warmup", ctx_len=0):
        for it in items:
            enc = tok.apply_chat_template([{"role": "user", "content": it.prompt}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to("cuda")
            with torch.no_grad():
                model(**enc)  # prefill only
    writer.close()
    h.remove()
    print(f"[bench] working-set catturato in {trace_path}")

    rows = []
    for cap in CAPS:
        p = max(1, cap // 2)  # quanti pinnare (resto LRU)
        plans = {
            "probe": resident_from_trace(trace_path, p),
            "blind": blind_resident(list(range(n_layers)), p, N_EXPERTS),
        }
        for mode, resident in plans.items():
            h = install_expert_cache(model, capacity_per_layer=cap, resident_by_layer=resident)
            acc, tps = gen_and_check()
            st = h.stats()
            h.remove()
            rep = M.capacity_report(st, model_total_gb=12.0,
                                    vram_expert_gb=M.estimate_vram_expert_gb(cap, n_layers, 12.58e6))
            rows.append({"cap": cap, "pinned": p, "mode": mode, "fetch_rate": st["fetch_rate"],
                         "fetched_gb": round(st["fetched_gb"], 1), "tok_s": round(tps, 1),
                         "accuracy": acc, "vram_expert_gb": round(rep["vram_expert_gb"], 2)})
            print(f"[bench] cap={cap} pin={p} {mode:>5}: fetch_rate={st['fetch_rate']:.3f} "
                  f"fetched={st['fetched_gb']:.1f}GB tok/s={tps:.1f} acc={acc:.2f}")

    print("\n=== CAPACITA-vs-LATENZA (OLMoE, always-fetch lossless) ===")
    print(f"{'cap':>4} {'mode':>6} | {'VRAM_exp':>8} | {'fetch_rate':>10} | {'fetched':>8} | {'tok/s':>6} | acc")
    print("-" * 66)
    for r in rows:
        print(f"{r['cap']:>4} {r['mode']:>6} | {r['vram_expert_gb']:>6.2f}GB | {r['fetch_rate']:>10.3f} | "
              f"{r['fetched_gb']:>6.1f}GB | {r['tok_s']:>6.1f} | {r['accuracy']:.2f}")

    import csv, os
    os.makedirs("runs", exist_ok=True)
    with open("runs/bench_capacity_olmoe.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[bench] CSV: runs/bench_capacity_olmoe.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
