"""saturation_dsv2.py — quanto velocemente il working-set di expert SATURA con i token.

Risponde all'obiezione: "61/64 expert e' colpa di prompt lunghi/variegati". Misura, per un prompt,
quanti expert DISTINTI (media sui layer) sono stati usati dopo i primi K token. Se la curva e' piatta
(satura tardi) -> interazioni CORTE/narrow hanno margine di compressione. Se satura entro poche decine
di token -> anche le corte sono dense (load-balancing del router).

Solo prefill, ordine dei token preservato (ExpertsRoutingCapture scrive [n_token] righe per layer in
ordine). Niente predizione, niente generazione.

Uso:  python -u scripts/saturation_dsv2.py --source codegen --n 4
      python -u scripts/saturation_dsv2.py --prompts "Ciao|Riassumi: gatto|<prompt lungo...>"
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="D:/dsv2_memmap")
    p.add_argument("--source", default="codegen")
    p.add_argument("--prompts", default="", help="prompt separati da '|' (override del source)")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--cap", type=int, default=16)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer
    from validate_probe_dsv2 import load_nonexpert_on_gpu, open_memmaps, ExpertsRoutingCapture, HF
    from msc.hierarchy.olmoe_cache import install_expert_cache, MemmapBacking
    from msc.instrument.router_hooks import RouterHookSpec

    if args.prompts:
        prompts = [s for s in args.prompts.split("|") if s.strip()]
    else:
        from msc.validator.python_unit_tests import PythonUnitTestValidator
        prompts = [it.prompt for it in PythonUnitTestValidator("data/codegen_problems.jsonl").items()[:args.n]]

    tok = AutoTokenizer.from_pretrained(HF)
    model, cfg = load_nonexpert_on_gpu(HF, "cuda", torch.bfloat16)
    gu, dn = open_memmaps(args.memmap)
    h = install_expert_cache(model, args.cap, {i: set() for i in range(cfg.num_hidden_layers)},
                             backing_factory=lambda li, e: MemmapBacking(gu, dn, li, int(e.num_experts)))
    first_moe = RouterHookSpec.for_model(HF).first_moe_layer
    E = int(cfg.n_routed_experts)
    os.makedirs("runs", exist_ok=True)
    CHECK = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]

    print(f"\n=== SATURAZIONE working-set DS2-Lite (E={E}/layer) — distinti dopo K token ===")
    print(f"{'len':>4} | " + " ".join(f"{k:>4}" for k in CHECK) + "   | prompt")
    curves = []
    for prompt in prompts:
        tp = "runs/_sat.jsonl"
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        ntok = int(enc["input_ids"].shape[1])
        with ExpertsRoutingCapture(model, first_moe, tp):
            with torch.no_grad():
                model(**enc)
        # ricostruisci per layer la sequenza di topk per token (l'hook scrive in ordine token, per layer)
        per_layer_tokens = collections.defaultdict(list)  # layer -> [ [ids token0], [ids token1], ... ]
        with open(tp, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                per_layer_tokens[int(o["layer"])].append([int(e) for e in o["topk_ids"]])
        os.remove(tp)
        # per ogni K: distinti cumulativi medi sui layer
        row = []
        for k in CHECK:
            if k > ntok:
                row.append(None); continue
            per_layer_distinct = []
            for ly, toks in per_layer_tokens.items():
                seen = set()
                for t in toks[:k]:
                    seen.update(t)
                per_layer_distinct.append(len(seen))
            row.append(st.mean(per_layer_distinct) if per_layer_distinct else 0)
        curves.append(row)
        cells = " ".join((f"{v:>4.0f}" if v is not None else "   -") for v in row)
        print(f"{ntok:>4} | {cells}   | {prompt[:42].replace(chr(10),' ')}")

    # media sulle curve (solo dove tutti hanno valore)
    print("-" * 70)
    avg = []
    for j, k in enumerate(CHECK):
        vals = [c[j] for c in curves if c[j] is not None]
        avg.append(st.mean(vals) if vals else None)
    cells = " ".join((f"{v:>4.0f}" if v is not None else "   -") for v in avg)
    print(f"MEDIA| {cells}")
    print(f"\n[sat] LETTURA: se a 8-16 token sei gia' a ~{E//2}+/{E}, satura in fretta (anche le interazioni")
    print(f"[sat] corte sono dense). Se a 16 token sei ancora basso, le interazioni corte hanno margine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
