"""diag_enforce_outputs.py — diagnostica: confronta gli output FULL vs hard-drop K=50% su 2 problemi.

Serve a capire la MODALITA di fallimento del crollo a 0/10: il modello sotto enforcement produce
codice degradato? prosa? garbage? O codice quasi-giusto che trippa un test? (no-spin, guardiamo).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

HF = "allenai/OLMoE-1B-7B-0924-Instruct"
N_EXPERTS = 64


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.enforce import attach_hard_drop

    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(
        HF, dtype=torch.bfloat16, device_map="auto", max_memory={0: "9GiB", "cpu": "48GiB"})
    model.eval()

    items = PythonUnitTestValidator("data/codegen_problems.jsonl").items()[:2]

    def gen(prompt):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    for it in items:
        print("\n" + "=" * 70)
        print(f"PROBLEMA {it.item_id}: {it.prompt[:90]}")
        print("-" * 70)
        print("[FULL]\n" + gen(it.prompt)[:500])

    resident = resident_topk("runs/accuracy_vs_k_olmoe_trace.jsonl", math.ceil(0.50 * N_EXPERTS))
    h = attach_hard_drop(model, resident)
    for it in items:
        print("\n" + "=" * 70)
        print(f"PROBLEMA {it.item_id} [K=50% hard-drop]")
        print("-" * 70)
        print(gen(it.prompt)[:500])
    h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
