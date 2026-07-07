"""validate_probe_dsv2.py — predire vs reagire su DeepSeek-V2-Lite (modello CONCENTRATO).

Stesso esperimento di validate_probe_qwen3, ma sul banco di prova GIUSTO per il metodo predittivo:
DeepSeek-V2-Lite ha shared expert + routing fine-grained (concentrato), a differenza di Qwen3-30B
(diffuso, no shared) dove predire NON ha battuto reagire. Domanda: su routing concentrato il
pre-staging PREDITTIVO (probe sul prompt) batte la cache REATTIVA (cold+LRU, = ds4)?

Differenze chiave vs Qwen3:
 - 64 routed expert (+ 2 shared, sempre attivi -> backbone su GPU, NON dal memmap);
 - layer 0 DENSE (first_k_dense_replace=1): l'engine lo salta da solo (install_expert_cache guarda
   hasattr(mlp,'experts')); RouterHookSpec.for_model lo salta (first_moe_layer=1);
 - rotary DeepseekV2RotaryEmbedding: stesso fix meta (re-istanzia sul device).

Prereq: scripts/build_dsv2_memmap.py ha creato D:/dsv2_memmap/{gate_up,down}.f16.mmap + meta.json.

Uso (dopo build_dsv2_memmap):  python -u scripts/validate_probe_dsv2.py --memmap D:/dsv2_memmap
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

HF = "deepseek-ai/DeepSeek-V2-Lite"
N_EXPERTS = 64
N_PROMPTS = 6
MAX_NEW = 48
CAPS = [8, 16, 32]   # capienza cache per-layer (su 64 routed)


def load_nonexpert_on_gpu(model_id, device, dtype):
    """Init su meta + carica SOLO i pesi non-expert su GPU (RAM-safe). Routed expert -> restano meta
    (serviti dal memmap). I SHARED expert NON contengono '.mlp.experts.' -> caricati su GPU (sempre attivi)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    snap = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "merges*", "vocab*"])
    nonexpert_sd = {}
    for shard in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if ".mlp.experts." in key:
                    continue  # routed expert: serviti dal memmap (shared_experts NON matcha -> caricati)
                nonexpert_sd[key] = f.get_tensor(key).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(nonexpert_sd, strict=False, assign=True)
    model.tie_weights()
    # Buffer non-persistenti del rotary (inv_freq) -> restano su meta dopo l'init-su-meta: li
    # materializziamo re-istanziando il rotary sul device reale (il ctor ricalcola inv_freq).
    rot = model.model.rotary_emb
    try:
        model.model.rotary_emb = type(rot)(config=config, device=torch.device(device))
    except TypeError:
        model.model.rotary_emb = type(rot)(config).to(device)
    model.eval()
    n_meta = sum(1 for _, p in model.named_parameters() if p.is_meta)
    print(f"[load] non-expert su GPU={len(nonexpert_sd)} | param meta rimasti (routed expert)={n_meta} "
          f"| unexpected={len(unexpected)}")
    return model, config


def open_memmaps(memmap_dir):
    import numpy as np
    meta = json.load(open(os.path.join(memmap_dir, "meta.json"), encoding="utf-8"))
    gu = np.memmap(os.path.join(memmap_dir, meta["gate_up"]["file"]), dtype=np.float16, mode="r",
                   shape=tuple(meta["gate_up"]["shape"]))
    dn = np.memmap(os.path.join(memmap_dir, meta["down"]["file"]), dtype=np.float16, mode="r",
                   shape=tuple(meta["down"]["shape"]))
    return gu, dn


class ExpertsRoutingCapture:
    """Cattura il routing dagli INPUT del modulo experts (top_k_index), scrivendo il formato che
    resident_from_trace si aspetta ({"layer","topk_ids"} per token).

    Serve per DeepSeek-V2: il gate NON passa per un forward hookabile (router_logits calcolati via
    nn.functional.linear(x, self.gate.weight)), quindi il forward-hook sul gate del RouterLogger non
    scatta. Hookare gli input di experts cattura gli indici REALI instradati (group-limited routing
    incluso) -> più corretto che ricostruire la top-k dai logit grezzi.
    """

    def __init__(self, model, first_moe_layer: int, trace_path: str) -> None:
        self._model = model
        self._first = int(first_moe_layer)
        self._path = trace_path
        self._fh = None
        self._handles: list = []

    def _make_hook(self, layer_idx: int):
        import json as _json

        def pre_hook(module, args):
            # args del forward di experts = (hidden_states, top_k_index, top_k_weights)
            top_k_index = args[1]
            rows = top_k_index.detach().to("cpu").tolist()  # [n_token, top_k]
            for row in rows:
                self._fh.write(_json.dumps({"layer": layer_idx,
                                            "topk_ids": [int(e) for e in row]}) + "\n")
        return pre_hook

    def __enter__(self) -> "ExpertsRoutingCapture":
        self._fh = open(self._path, "w", encoding="utf-8")
        for i, layer in enumerate(self._model.model.layers):
            if i < self._first:
                continue
            experts = getattr(layer.mlp, "experts", None)
            if experts is None:
                continue
            self._handles.append(experts.register_forward_pre_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="D:/dsv2_memmap", help="dir con gate_up/down .mmap + meta.json")
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS, help="n. prompt dal validator (default 6)")
    p.add_argument("--max-new", type=int, default=MAX_NEW, help="token generati per prompt (default 48)")
    p.add_argument("--caps", default=",".join(str(c) for c in CAPS),
                   help="capienze cache per-layer (csv, default 8,16,32)")
    args = p.parse_args()
    n_prompts = int(args.n_prompts)
    max_new = int(args.max_new)
    caps = [int(c) for c in str(args.caps).split(",") if c.strip()]

    import torch
    from transformers import AutoTokenizer

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter
    from msc.validator.python_unit_tests import PythonUnitTestValidator
    from msc.hierarchy.olmoe_cache import install_expert_cache, MemmapBacking
    from msc.hierarchy.calibrate import resident_from_trace

    if not os.path.exists(os.path.join(args.memmap, "meta.json")):
        print(f"[val] memmap mancante in {args.memmap}: esegui prima build_dsv2_memmap.py")
        return 2

    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(HF)
    print("[val] carico DeepSeek-V2-Lite (non-expert+shared su GPU, routed expert da memmap-SSD)...")
    model, cfg = load_nonexpert_on_gpu(HF, "cuda", dtype)
    n_layers = int(cfg.num_hidden_layers)
    layers = list(range(n_layers))
    gu, dn = open_memmaps(args.memmap)

    def backing_factory(layer_idx, experts):
        return MemmapBacking(gu, dn, layer_idx, int(experts.num_experts))

    validator = PythonUnitTestValidator("data/codegen_problems.jsonl")
    items = validator.items()[:n_prompts]

    def enc_of(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to("cuda")

    def gen_session():
        correct, ntok, t0 = 0, 0, time.perf_counter()
        for it in items:
            enc = enc_of(it.prompt)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            new = out[0, enc["input_ids"].shape[1]:]
            ntok += int(new.shape[0])
            correct += int(validator.verify(it, tok.decode(new, skip_special_tokens=True)))
        return correct / len(items), ntok, time.perf_counter() - t0

    os.makedirs("runs", exist_ok=True)
    rows = []
    for cap in caps:
        pin = max(1, cap - 1)
        empty = {i: set() for i in layers}

        # PROBE: prefill a freddo, cattura routing -> working-set predetto.
        # DS2: il gate non passa per un forward hookabile -> catturiamo gli indici REALI dagli
        # input del modulo experts (top_k_index), vedi ExpertsRoutingCapture.
        h = install_expert_cache(model, cap, empty, backing_factory=backing_factory)
        tp = f"runs/val_dsv2_probe_c{cap}.jsonl"
        with ExpertsRoutingCapture(model, RouterHookSpec.for_model(HF).first_moe_layer, tp):
            for it in items:
                with torch.no_grad():
                    model(**enc_of(it.prompt))
        probe_miss = h.stats()["total"]["misses"]; h.remove()
        predicted = resident_from_trace(tp, pin)

        # REACTIVE (cold+LRU = ds4)
        h = install_expert_cache(model, cap, empty, backing_factory=backing_factory)
        acc_r, _, t_r = gen_session(); react = h.stats()["total"]["misses"]; h.remove()

        # PREDICTIVE (pin working-set predetto)
        h = install_expert_cache(model, cap, predicted, backing_factory=backing_factory)
        acc_p, ntok, t_p = gen_session(); pred = h.stats()["total"]["misses"]; h.remove()

        saving = react - pred
        rows.append({"cap": cap, "pin": pin, "react_miss": react, "pred_miss": pred,
                     "probe_miss": probe_miss, "saving": saving, "net_vs_probe": saving - probe_miss,
                     "react_tok_s": round(ntok / t_r, 2), "pred_tok_s": round(ntok / t_p, 2),
                     "acc_react": acc_r, "acc_pred": acc_p})
        print(f"[val] cap={cap}/{N_EXPERTS} pin={pin}: react={react} pred={pred} probe={probe_miss} "
              f"saving={saving} net={saving-probe_miss} tok/s r={ntok/t_r:.1f}/p={ntok/t_p:.1f} "
              f"acc r={acc_r:.2f}/p={acc_p:.2f}", flush=True)

    print(f"\n=== DeepSeek-V2-Lite ({N_EXPERTS} routed expert, memmap-SSD): predire vs reagire ===")
    print(f"{'cap':>4} | {'react':>7} {'pred':>7} {'probe':>7} | {'saving':>7} {'net':>7} | tok/s r/p | acc r/p")
    for r in rows:
        print(f"{r['cap']:>4} | {r['react_miss']:>7} {r['pred_miss']:>7} {r['probe_miss']:>7} | "
              f"{r['saving']:>7} {r['net_vs_probe']:>7} | {r['react_tok_s']}/{r['pred_tok_s']} | "
              f"{r['acc_react']:.2f}/{r['acc_pred']:.2f}")
    with open("runs/validate_probe_dsv2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[val] CSV: runs/validate_probe_dsv2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
