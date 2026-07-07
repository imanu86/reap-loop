"""verify_enforce_olmoe.py — verifica l'enforcement hard-drop su OLMoE vero (GPU/offload).

Tre check:
  (a) resident = TUTTI i 64 expert  -> indici instradati IDENTICI al baseline (no-op);
  (b) resident = sottoinsieme {0..7} -> TUTTI gli indici instradati ⊆ resident,
      e model.generate(max_new_tokens=16) gira senza errori;
  (c) confronto numerico: # indici unici per layer prima/dopo + conferma contenimento.

Cattura gli indici instradati con forward-hook di SOLA LETTURA su ogni gate (logga output[2]).
Imposta PYTHONPATH=...\\src prima di lanciare (serve per 'import msc...').
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from msc.enforce import attach_hard_drop

HF = "allenai/OLMoE-1B-7B-0924-Instruct"
PROMPT = "Explain in one sentence why the sky is blue."


def _attach_readers(model) -> tuple[dict[int, torch.Tensor], list]:
    """Registra reader read-only su ogni gate; ritorna (dizionario_per_layer, handles)."""
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make_reader(idx):
            def reader(mod, inp, out):
                # out = (router_logits, router_scores, router_indices); ci serve [2].
                captured[idx] = out[2].detach().to("cpu")
            return reader
        handles.append(layer.mlp.gate.register_forward_hook(make_reader(i)))
    return captured, handles


def _remove(handles) -> None:
    for h in handles:
        h.remove()


def main() -> int:
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForCausalLM.from_pretrained(
        HF, dtype=torch.bfloat16, device_map="auto", max_memory={0: "9GiB", "cpu": "48GiB"})
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    n_experts = int(model.config.num_experts)
    top_k = int(model.config.num_experts_per_tok)
    print(f"[cfg] n_layers={n_layers} n_experts={n_experts} top_k={top_k} "
          f"norm_topk_prob={getattr(model.config, 'norm_topk_prob', None)}")

    enc = tok(PROMPT, return_tensors="pt").to(model.device if hasattr(model, "device") else "cuda")

    # -------- BASELINE: nessun enforcement, cattura indici instradati --------
    base_cap, base_h = _attach_readers(model)
    with torch.no_grad():
        model(**enc)
    _remove(base_h)
    baseline = {i: base_cap[i].clone() for i in range(n_layers)}

    # ====== CHECK (a): resident = TUTTI i 64 expert -> no-op (indici identici) ======
    # IMPORTANTE: i forward_hook su uno stesso modulo sparano in ordine di registrazione e
    # la return di un hook diventa l'output visto dal successivo. Per leggere cio' che il
    # blocco MoE ricevera' DAVVERO, il reader va registrato DOPO l'enforcement.
    all_experts = {i: set(range(n_experts)) for i in range(n_layers)}
    handle_a = attach_hard_drop(model, all_experts)
    noop_cap, noop_read_h = _attach_readers(model)
    with torch.no_grad():
        model(**enc)
    handle_a.remove()
    _remove(noop_read_h)

    all_noop = True
    for i in range(n_layers):
        same = torch.equal(baseline[i], noop_cap[i])
        all_noop = all_noop and same
    print(f"[a] no-op (resident=ALL64) -> indici identici al baseline su tutti i layer: {all_noop}")

    # ====== CHECK (b): resident = {0..7} -> contenimento + generate ======
    subset = set(range(top_k))  # {0,1,2,3,4,5,6,7}
    subset_by_layer = {i: set(subset) for i in range(n_layers)}
    handle_b = attach_hard_drop(model, subset_by_layer)
    sub_cap, sub_read_h = _attach_readers(model)  # reader DOPO l'enforcement (vedi nota in check a)
    with torch.no_grad():
        model(**enc)

    all_contained = True
    per_layer_ok = {}
    for i in range(n_layers):
        routed = set(sub_cap[i].reshape(-1).tolist())
        ok = routed.issubset(subset)
        per_layer_ok[i] = ok
        all_contained = all_contained and ok
    print(f"[b] contenimento (resident={sorted(subset)}) -> tutti gli indici sottoinsieme di resident "
          f"su tutti i layer: {all_contained}")
    if not all_contained:
        bad = {i: sorted(set(sub_cap[i].reshape(-1).tolist()) - subset) for i, ok in per_layer_ok.items() if not ok}
        print(f"    layer fuori-set: {bad}")

    _remove(sub_read_h)

    # generate con l'enforcement subset ancora attivo
    gen_ok = True
    text = ""
    try:
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001
        gen_ok = False
        text = f"<errore: {type(e).__name__}: {e}>"
    handle_b.remove()
    print(f"[b] generate(max_new_tokens=16) senza errori: {gen_ok}")
    print(f"    testo: {text!r}")

    # ====== CHECK (c): confronto numerico per layer ======
    print("[c] confronto numerico per layer (# expert unici instradati):")
    print(f"    {'layer':>5} | {'baseline_uniq':>13} | {'subset_uniq':>11} | {'in_set(0..7)':>13}")
    for i in range(n_layers):
        b_uniq = len(set(baseline[i].reshape(-1).tolist()))
        s_uniq = len(set(sub_cap[i].reshape(-1).tolist()))
        print(f"    {i:>5} | {b_uniq:>13} | {s_uniq:>11} | {str(per_layer_ok[i]):>13}")

    # globale
    base_all_uniq = sorted(set().union(*[set(baseline[i].reshape(-1).tolist()) for i in range(n_layers)]))
    sub_all_uniq = sorted(set().union(*[set(sub_cap[i].reshape(-1).tolist()) for i in range(n_layers)]))
    n_tokens = baseline[0].shape[0]
    print(f"[c] n_token nel prompt = {n_tokens}")
    print(f"[c] expert unici (globale) baseline = {len(base_all_uniq)} -> {base_all_uniq}")
    print(f"[c] expert unici (globale) subset   = {len(sub_all_uniq)} -> {sub_all_uniq}")

    print("\n=== RISULTATI BOOLEANI ===")
    print(f"(a) all_resident_noop   = {all_noop}")
    print(f"(b) subset_contained    = {all_contained}")
    print(f"(b) generation_ok       = {gen_ok}")
    return 0 if (all_noop and all_contained and gen_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
