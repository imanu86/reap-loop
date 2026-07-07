"""probe_fidelity_granite.py — FIDELITY del routing sotto ultra-quant degli expert (Granite-3B, GPU).

DOMANDA: il routing di un modello con expert ULTRA-quantizzati predice il routing del modello PIENO?
Se sì, un probe economico (expert int2/int4, router fp16) puo' scoprire il working-set della sessione.

Setup: ibm-granite/granite-3.1-3b-a800m-instruct entra in 12GB a fp16 (device_map="cuda", no offload),
quindi modifichiamo i pesi degli expert IN-PLACE. Il ROUTER e' TENUTO a fp16 (quantizziamo SOLO gli
expert) -> e' la versione "smart" del probe.

Router per layer: model.model.layers[i].block_sparse_moe.router.layer (nn.Linear, out=40).
Il top-8 e' invariante a softmax => basta torch.topk(logits, 8) lungo l'ultima dim.

Confronto su PREFILL: stessi input_ids, UN forward (no generate). Per ogni (layer, token) registriamo
l'insieme top-8 di expert. Confronto diretto FULL vs ULTRA-QUANT (int4, int2).

Riusa msc.enforce.granite_cascade: snapshot_experts / restore_experts / apply_mode con
resident_by_layer = {i: set() for i in layer MoE} -> quantizza TUTTI gli expert, router intatto.
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from msc.enforce.granite_cascade import apply_mode, restore_experts, snapshot_experts

HF = "ibm-granite/granite-3.1-3b-a800m-instruct"
DATASET = "data/codegen_problems.jsonl"
TOPK = 8
NBITS_LIST = [4, 2]


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


class RouterTopkCapture:
    """Forward-hook su ogni router.layer (nn.Linear) dei layer MoE.

    Per ogni forward salva, per layer, gli indici top-8 [n_token, 8] (su CPU, long).
    Si assume UN solo forward per cattura (prefill, no generate) -> niente accumulo step.
    """

    def __init__(self, model, topk: int = TOPK) -> None:
        self._model = model
        self._topk = topk
        self._handles: list = []
        self.by_layer: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            # logits: [n_token, n_expert] (o [1, n_token, n_expert]); riduci a 2D
            x = logits.detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            top = torch.topk(x, self._topk, dim=-1).indices  # [n_token, 8]
            self.by_layer[layer_idx] = top.to("cpu")
        return hook

    def __enter__(self):
        self.by_layer = {}
        for i, layer in enumerate(self._model.model.layers):
            router_linear = layer.block_sparse_moe.router.layer
            self._handles.append(router_linear.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def capture_routing(model, tok, prompts: list[str]) -> dict[int, list[torch.Tensor]]:
    """Per ogni prompt fa UN forward (no generate) e raccoglie, per layer, il top-8 per token.

    Ritorna: {layer_idx: [tensor[n_token_p, 8] per ogni prompt p]}.
    """
    out: dict[int, list[torch.Tensor]] = {}
    for prompt in prompts:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        with RouterTopkCapture(model) as cap, torch.no_grad():
            model(**enc)
        for layer_idx, top in cap.by_layer.items():
            out.setdefault(layer_idx, []).append(top)
    return out


def top8_overlap(ref: dict, qnt: dict, topk: int = TOPK) -> float:
    """Media su (layer, prompt, token) di |top8_full ∩ top8_quant| / topk."""
    total = 0.0
    count = 0
    for layer_idx, ref_list in ref.items():
        qnt_list = qnt[layer_idx]
        for rp, qp in zip(ref_list, qnt_list):
            # rp, qp: [n_token, 8]
            assert rp.shape == qp.shape, (rp.shape, qp.shape)
            for t in range(rp.shape[0]):
                rset = set(rp[t].tolist())
                qset = set(qp[t].tolist())
                total += len(rset & qset) / topk
                count += 1
    return total / count if count else float("nan")


def workingset(routing: dict) -> dict[int, set[int]]:
    """WS_layer = unione dei top-8 su TUTTI i token di prefill di TUTTI i prompt."""
    ws: dict[int, set[int]] = {}
    for layer_idx, lst in routing.items():
        s: set[int] = set()
        for tens in lst:
            s.update(int(x) for x in tens.reshape(-1).tolist())
        ws[layer_idx] = s
    return ws


def workingset_jaccard(ref: dict, qnt: dict) -> tuple[float, dict[int, float]]:
    """Media sui layer di |WS_full ∩ WS_quant| / |WS_full ∪ WS_quant|, + per-layer."""
    ws_ref = workingset(ref)
    ws_qnt = workingset(qnt)
    per_layer = {}
    for layer_idx, s_ref in ws_ref.items():
        s_qnt = ws_qnt[layer_idx]
        union = s_ref | s_qnt
        per_layer[layer_idx] = (len(s_ref & s_qnt) / len(union)) if union else 1.0
    mean = sum(per_layer.values()) / len(per_layer) if per_layer else float("nan")
    return mean, per_layer


def main() -> int:
    os.makedirs("runs", exist_ok=True)
    prompts = load_prompts(DATASET)
    print(f"[probe] prompts: {len(prompts)} from {DATASET}")

    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float16, device_map="cuda")
    model.eval()
    n_exp = int(model.config.num_local_experts)
    n_moe_layers = len(model.model.layers)
    router_dtype = model.model.layers[0].block_sparse_moe.router.layer.weight.dtype
    print(f"[probe] {HF} | n_experts={n_exp} | moe_layers={n_moe_layers} | "
          f"router_dtype={router_dtype} | VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    # 1) ROUTING DI RIFERIMENTO (FULL, fp16)
    ref = capture_routing(model, tok, prompts)
    total_tokens = sum(t.shape[0] for t in next(iter(ref.values())))
    print(f"[probe] FULL routing catturato: {len(ref)} layer, {total_tokens} token totali/layer")

    # 2) snapshot pesi expert (per restore tra le celle)
    snap = snapshot_experts(model)
    resident_none = {i: set() for i in range(n_moe_layers)}  # quantizza TUTTI gli expert

    results = {}
    for nbits in NBITS_LIST:
        restore_experts(model, snap)
        touched = apply_mode(model, resident_none, "precision-cascade", nbits=nbits)
        # router intatto: verifichiamo che il dtype del router sia ancora fp16
        rd = model.model.layers[0].block_sparse_moe.router.layer.weight.dtype
        qnt = capture_routing(model, tok, prompts)
        ov = top8_overlap(ref, qnt)
        wj, per_layer = workingset_jaccard(ref, qnt)
        results[nbits] = {"top8_overlap": ov, "ws_jaccard": wj,
                          "touched": touched, "router_dtype": str(rd),
                          "ws_jaccard_per_layer": per_layer}
        print(f"[probe] int{nbits}: experts_quantizzati={touched} | router_dtype={rd} | "
              f"top8_overlap={ov:.4f} | workingset_jaccard={wj:.4f}")
    restore_experts(model, snap)

    # salva risultati
    out_path = "runs/probe_fidelity_granite.json"
    payload = {
        "model": HF, "n_experts": n_exp, "moe_layers": n_moe_layers,
        "topk": TOPK, "n_prompts": len(prompts), "total_tokens_per_layer": total_tokens,
        "router_kept_fp16": all(r["router_dtype"] == "torch.float16" for r in results.values()),
        "results": {str(k): {kk: vv for kk, vv in v.items() if kk != "ws_jaccard_per_layer"}
                    for k, v in results.items()},
        "ws_jaccard_per_layer": {str(k): v["ws_jaccard_per_layer"] for k, v in results.items()},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[probe] risultati -> {out_path}")

    # verdetto sintetico
    print("\n=== FIDELITY ROUTING sotto ultra-quant (Granite-3B, router fp16) ===")
    print(f"{'nbits':>6} | {'top8_overlap':>13} | {'ws_jaccard':>11}")
    print("-" * 40)
    for nbits in NBITS_LIST:
        r = results[nbits]
        print(f"int{nbits:>3} | {r['top8_overlap']:>13.4f} | {r['ws_jaccard']:>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
