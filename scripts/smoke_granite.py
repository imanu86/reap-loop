"""smoke_granite.py — smoke di LOCALITA multi-modello (Granite / OLMoE) su RTX 3060.

(nome storico; ora generico). Valida il "gpu seam" sul modello vero in 3 stadi:
  STADIO 0  introspezione: il modulo gate risolto da RouterHookSpec.for_model e un layer lineare
            con out_features == n_experts?
  STADIO 1  traccia: hook sul gate -> RouterLogger -> traccia jsonl su prompt di coding
  STADIO 2  analisi: working set per copertura theta + concentrazione (N_eff) per layer.

Modelli (registry sotto). OLMoE (~14GB fp16) NON entra in 12GB -> caricato in 8-bit (bitsandbytes),
tenendo PERO il router/gate in fp16 (llm_int8_skip_modules) per non perturbare il routing — la
quantizzazione int8 degli expert non altera la STRUTTURA del working set che vogliamo misurare.

Uso:
    python scripts/smoke_granite.py --model granite-3b --workload repetitive --n-prompts 8
    python scripts/smoke_granite.py --model olmoe      --workload repetitive --n-prompts 8
"""

from __future__ import annotations

import argparse
import os
import statistics

# key -> (hf_id, quant_default, int8_skip_modules per tenere il router in fp16)
# OLMoE non entra a fp16 in 12GB: default "offload" (bf16, expert in eccesso in RAM) -> routing a
# PIENA PRECISIONE = misura di localita fedele (l'8-bit perturberebbe il routing). "8bit" resta
# disponibile come alternativa piu veloce ma meno fedele.
MODEL_REGISTRY = {
    "granite-3b": ("ibm-granite/granite-3.1-3b-a800m-instruct", "none", ["router"]),
    "granite-1b": ("ibm-granite/granite-3.1-1b-a400m-instruct", "none", ["router"]),
    "olmoe":      ("allenai/OLMoE-1B-7B-0924-Instruct", "offload", ["mlp.gate"]),
}

# DUE workload per misurare l'effetto della LOCALITA (non della diversita):
#  - "diverse": problemi eterogenei (stringhe, matematica, ricorsione) = BASSA localita.
#  - "repetitive": stesso dominio (utility su stringhe) + stesso fraseggio = ALTA localita.
WORKLOADS = {
    "diverse": [
        "Write a Python function `is_palindrome(s: str) -> bool` that returns True if s is a palindrome.",
        "Write a Python function `factorial(n: int) -> int` computing n! iteratively.",
        "Write a Python function `fib(n: int) -> int` returning the n-th Fibonacci number.",
        "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words.",
        "Write a Python function `count_vowels(s: str) -> int` counting vowels in s.",
        "Write a Python function `gcd(a: int, b: int) -> int` using Euclid's algorithm.",
        "Write a Python function `is_prime(n: int) -> bool` testing primality.",
        "Write a Python function `flatten(xs: list) -> list` flattening one nesting level.",
    ],
    "repetitive": [
        "Complete the function `slugify(s: str) -> str`: lowercase, spaces to hyphens, drop non-alphanumerics.",
        "Complete the function `truncate(s: str, n: int) -> str`: shorten s to n chars adding an ellipsis if longer.",
        "Complete the function `count_words(s: str) -> int`: return the number of whitespace-separated words.",
        "Complete the function `strip_tags(s: str) -> str`: remove all <...> HTML tags from s.",
        "Complete the function `capitalize_words(s: str) -> str`: uppercase the first letter of each word.",
        "Complete the function `remove_vowels(s: str) -> str`: return s without any vowels.",
        "Complete the function `repeat_chars(s: str, n: int) -> str`: repeat each char of s n times.",
        "Complete the function `normalize_spaces(s: str) -> str`: collapse runs of whitespace to one space.",
    ],
}


def _resolve(obj, path: str):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke di localita multi-modello (hook + traccia + N_eff).")
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="granite-3b")
    p.add_argument("--quant", choices=["none", "8bit", "offload"], default=None,
                   help="none=fp16/bf16 tutto in VRAM | 8bit=bitsandbytes | offload=bf16 con expert "
                        "in eccesso in RAM. default: dal registry.")
    p.add_argument("--gpu-mem", default="9GiB", help="tetto VRAM per i pesi in modalita offload.")
    p.add_argument("--max-new", type=int, default=48)
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--theta", type=float, default=0.95)
    p.add_argument("--workload", choices=sorted(WORKLOADS), default="diverse")
    p.add_argument("--out", default=None, help="default: runs/smoke_<model>_<workload>.")
    args = p.parse_args()

    hf_id, quant_default, int8_skip = MODEL_REGISTRY[args.model]
    quant = args.quant or quant_default
    if args.out is None:
        args.out = f"runs/smoke_{args.model}_{args.workload}"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[smoke] torch/transformers mancanti: {exc}")
        return 2

    from msc.instrument.router_hooks import RouterHookSpec, RouterLogger
    from msc.instrument.trace import TraceWriter, TraceReader
    from msc.workingset.estimator import estimate_working_set

    os.makedirs(args.out, exist_ok=True)
    print(f"[smoke] model={hf_id} quant={quant} workload={args.workload}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    if quant == "8bit":
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=int8_skip)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=qcfg, device_map="auto", dtype=torch.float16)
        print(f"[smoke] 8-bit (router tenuto in fp16 via skip={int8_skip})")
    elif quant == "offload":
        # bf16 a piena precisione: accelerate riempie la GPU fino a --gpu-mem, il resto (expert)
        # va in RAM e viene spostato on-demand durante il forward. Routing = quello vero del modello.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        max_memory = {0: args.gpu_mem, "cpu": "48GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=dtype, device_map="auto", max_memory=max_memory)
        print(f"[smoke] offload bf16: pesi su GPU<={args.gpu_mem}, eccesso in RAM (fetch on-demand)")
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype, device_map="cuda")
    model.eval()
    if torch.cuda.is_available():
        print(f"[smoke] VRAM allocata: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    cfg = model.config
    n_experts = int(getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", -1)))
    topk = int(getattr(cfg, "num_experts_per_tok", 8))
    n_layers = int(getattr(cfg, "num_hidden_layers", len(model.model.layers)))
    print(f"[smoke] config: n_experts={n_experts} top_k={topk} n_layers={n_layers}")

    # ---------------- STADIO 0: introspezione via spec ----------------
    # Due forme di gate:
    #  - nn.Linear (Granite `.router.layer`, OLMoE vecchio): output = logit [n_token, n_experts].
    #  - modulo router (OLMoE 5.x `OlmoeTopKRouter`, ritorna tupla con logit a output[0]).
    # RouterLogger gestisce entrambi (tensore diretto, o output[0] della tupla). Qui verifichiamo.
    spec = RouterHookSpec.for_model(hf_id)
    gate_mod = _resolve(model.model.layers[spec.first_moe_layer], spec.gate_attr_path)
    out_features = getattr(gate_mod, "out_features", None)
    if out_features is not None:
        assert out_features == n_experts, (
            f"out_features {out_features} != n_experts {n_experts}: ricetta di hook da rivedere")
        print(f"[stadio0] gate '{spec.gate_attr_path}' = nn.Linear out={out_features} "
              f"-> logit [n_token, {n_experts}] OK")
    else:
        # Modulo router: dry-run per confermare che output[0] sia [*, n_experts].
        cap = {}
        hh = gate_mod.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o))
        with torch.no_grad():
            model(**tok("def f():\n    return 0\n", return_tensors="pt").to("cuda"))
        hh.remove()
        o = cap["o"]
        logits = o[0] if isinstance(o, (tuple, list)) else o
        assert logits.shape[-1] == n_experts, (
            f"router output last-dim {logits.shape[-1]} != n_experts {n_experts}")
        print(f"[stadio0] gate '{spec.gate_attr_path}' = {type(gate_mod).__name__} (router) "
              f"-> output[0] [n_token, {n_experts}] OK")
    print(f"[stadio0] topk={spec.topk}, norm_topk_prob={spec.norm_topk_prob}")

    # ---------------- STADIO 1: traccia ----------------
    trace_path = os.path.join(args.out, "trace.jsonl")
    writer = TraceWriter(trace_path)
    logger = RouterLogger(model, spec, writer)
    prompts = WORKLOADS[args.workload][: args.n_prompts]
    print(f"[stadio1] {len(prompts)} prompt")
    with logger.capture(session_id="smoke", ctx_len=0):
        for i, prompt in enumerate(prompts):
            if getattr(tok, "chat_template", None):
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
            else:
                enc = tok(prompt, return_tensors="pt").to("cuda")
            input_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            print(f"[stadio1] prompt {i + 1}/{len(prompts)} (input_len={input_len})")
    writer.close()
    print(f"[stadio1] traccia: {trace_path} ({os.path.getsize(trace_path)/1024:.0f} KiB)")

    # ---------------- STADIO 2: working set + concentrazione ----------------
    est = estimate_working_set(TraceReader(trace_path), theta=args.theta)
    ws = [len(v) for v in est.per_layer_working_set.values()]
    neff = [c.n_eff for c in est.per_layer_concentration.values()]
    print(f"\n[stadio2] === LOCALITA ({args.model}, theta={args.theta}) ===")
    print(f"[stadio2] committed_fraction media = {est.committed_fraction:.3f}")
    if ws:
        print(f"[stadio2] working set/layer: min={min(ws)} mediana={int(statistics.median(ws))} "
              f"max={max(ws)} (su {n_experts})")
    if neff:
        print(f"[stadio2] N_eff/layer: min={min(neff):.1f} mediana={statistics.median(neff):.1f} "
              f"max={max(neff):.1f} (1=concentrato, {n_experts}=diffuso)")
    print(f"[stadio2] -> analisi forma: python scripts/coverage_curve.py --trace {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
