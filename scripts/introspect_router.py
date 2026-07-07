"""introspect_router.py — scopre la struttura del router di un modello (transformers cambia spesso).

Stampa: tipo del modulo gate, suoi figli/parametri, e la STRUTTURA dell'output del forward del gate
(tupla? quale elemento e [seq, n_experts]?). Serve a fissare la ricetta di hook corretta.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF = "allenai/OLMoE-1B-7B-0924-Instruct"


def main() -> int:
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(
        HF, dtype=torch.bfloat16, device_map="auto", max_memory={0: "9GiB", "cpu": "48GiB"})
    model.eval()
    n_experts = int(getattr(model.config, "num_experts", -1))
    print(f"n_experts={n_experts}")

    mlp = model.model.layers[0].mlp
    gate = mlp.gate
    print(f"mlp type   = {type(mlp).__name__}")
    print(f"gate type  = {type(gate).__name__}")
    print(f"gate children  = {[(n, type(m).__name__) for n, m in gate.named_children()]}")
    print(f"gate params    = {[(n, tuple(p.shape)) for n, p in gate.named_parameters()]}")

    captured = {}

    def hook(mod, inp, out):
        captured["out"] = out

    h = gate.register_forward_hook(hook)
    enc = tok("def add(a, b):\n    return a + b\n", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**enc)
    h.remove()

    out = captured.get("out")
    print(f"gate forward output type = {type(out).__name__}")
    if isinstance(out, (tuple, list)):
        for i, e in enumerate(out):
            sh = tuple(e.shape) if hasattr(e, "shape") else None
            dt = getattr(e, "dtype", None)
            print(f"   [{i}] {type(e).__name__} shape={sh} dtype={dt}")
    else:
        print(f"   tensor shape={tuple(out.shape) if hasattr(out, 'shape') else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
