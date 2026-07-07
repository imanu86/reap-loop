"""Step 2 — the Confidence Cascade controller and the adversarial arms.

Regime: SMALL NATIVE WINDOW. The model natively sees only the last `native_turns`
of the conversation; older facts have fallen out and must be RECOVERED. This is the
regime where the concept can pay off (Gate #1 showed Qwen2.5-7B barely decays inside
8k, so recovery only matters once the native window is small).

Cascade (B4): generate from the native window; if confidence < theta, escalate through
qualitatively different, cost-ASCENDING recovery mechanisms, early-exiting at the first
rung whose regenerated answer clears theta:
  rung-0  turn-level text re-boost  — cheapest: a single old turn picked by a trivial
          token-overlap scan of the recent buffer, re-injected. NO index, NO embed call.
          (This is the novelty under test; on MLA it is the honest stand-in for KV re-hydration.)
  rung-1  buffer retrieval          — top-k over the recent buffer (bounded). +1 embed call.
  rung-2  full retrieval            — top-k BM25 over the ENTIRE history (recall-reinject-like). +1 rag call.
  rung-3  compression               — out of MVP.

Arms sharing this machinery:
  B0_sw  native small window only (no recovery)         — isolates the small-window backbone
  B2_reactive  native -> if low conf, ONE full retrieval — binary FLARE/recall-reinject baseline
  B3_random    same rungs, RANDOM order, same theta      — equal-budget control: if B4 can't beat
               a random-order cascade, the win was the budget, not the cheap-first gate
  B4_cascade   rungs cost-ascending [0,1,2], early-exit  — THE method
  B5_cascade_no_rung0  rungs [1,2], early-exit           — ablation isolating rung-0's contribution

Every arm returns {result, cost, arm, meta}; meta carries rung_stop + per-rung trace.
Denominator discipline and cost vector are unchanged (cost of failures stays in the average).
"""
from __future__ import annotations
import re
import hashlib

from llm_client import chat
from cost import from_usage, CostVector
from confidence import score as conf_score
from scorer import is_abstention
from retrieval_adapter import MemoryRetriever

SYSTEM = (
    "You are a memory assistant. Answer the final question using ONLY information "
    "stated in the conversation or in the recovered notes. Reply with just the value, "
    "no explanation. If the information is not present, reply exactly: I don't know."
)

_STOP = set("the of a an is are for to and what value only answer with".split())


def _msgs(n_turns):
    return n_turns * 2  # one turn = user + assistant


def _windows(history, native_turns, buffer_turns):
    nm, bm = _msgs(native_turns), _msgs(buffer_turns)
    native = history[-nm:] if nm < len(history) else list(history)
    buffer = history[-bm:] if bm < len(history) else list(history)
    return native, buffer


def _question_entity(question):
    m = re.search(r"\bof\s+(.+?)\?", question)
    return m.group(1).strip() if m else ""


def _tok(s):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", (s or "").lower()) if t not in _STOP]


def _cheap_pick_turn(buffer, question):
    """rung-0: pick the SINGLE buffer message with max token-overlap vs the question.
    Trivial scan, no index — deliberately weaker than retrieval so it can fail on
    dense distractors and force escalation (that's what B5 ablation measures)."""
    qt = set(_tok(question))
    best, best_score = None, 0
    for m in buffer:
        if m.get("role") != "user":
            continue
        s = len(qt & set(_tok(m.get("content", ""))))
        if s > best_score:
            best, best_score = m, s
    return best["content"] if best else None


def _mock_meta(item):
    return {"answer": item["answer"], "item_id": item["item_id"], "distance": item["distance"]}


def _generate(messages, base_url, model, mock_meta):
    res = chat(messages, base_url=base_url, model=model, mock_meta=mock_meta)
    conf = conf_score(res)["primary"]
    c = from_usage(res.usage)
    c.wall_clock_s = res.wall_clock_s
    return res, conf, c


def _augmented(recovered_text, native, question):
    block = "Recovered earlier notes:\n" + recovered_text + "\n\n" if recovered_text else ""
    return [{"role": "system", "content": SYSTEM}] + native + \
           [{"role": "user", "content": block + question}]


def _native_only(native, question):
    return [{"role": "system", "content": SYSTEM}] + native + \
           [{"role": "user", "content": question}]


def _recover(rung, item, buffer, full, question, k):
    """Return (recovered_text, extra_cost) for a rung. extra_cost carries only the
    embed/rag call components; the LLM regeneration cost is added by the caller."""
    if rung == 0:
        return _cheap_pick_turn(buffer, question), CostVector()  # no embed/rag
    if rung == 1:
        hits = MemoryRetriever(buffer).search(question, k=k)
        txt = "\n".join(f"- {h['content']}" for h in hits)
        return txt, CostVector(n_embed_calls=1)                  # local buffer "embed"
    if rung == 2:
        hits = MemoryRetriever(full).search(question, k=k)
        txt = "\n".join(f"- {h['content']}" for h in hits)
        return txt, CostVector(n_rag_calls=1)                    # full retrieval
    raise ValueError(rung)


def _rung_order(item, order_mode):
    base = [0, 1, 2]
    if order_mode == "ascending":
        return base
    if order_mode == "no_rung0":
        return [1, 2]
    if order_mode == "reactive":
        return [2]
    if order_mode == "random":
        h = int(hashlib.md5(item["item_id"].encode()).hexdigest()[:8], 16)
        perm = base[:]
        # deterministic Fisher-Yates from the item hash
        for i in range(len(perm) - 1, 0, -1):
            h, j = divmod(h, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        return perm
    raise ValueError(order_mode)


def run_controller(item, base_url, model, theta=-1.0, order_mode="ascending",
                   native_turns=6, buffer_turns=45, k=3, **_):
    history = item["messages"]
    native, buffer = _windows(history, native_turns, buffer_turns)
    mm = _mock_meta(item)
    trace = []

    # A generation is "resolved" if it is NOT an abstention AND its logprob clears
    # theta. Abstention is the decisive low-confidence signal here: in the small-window
    # regime the model abstains at logprob ~0, so mean_logprob alone never triggers
    # (measured: abstained mean_logprob = -0.003). We escalate on abstention OR low conf.
    def _resolved(res_, conf_):
        return (not is_abstention(res_.text)) and conf_ >= theta

    # rung "native": generate from the small window alone
    res, conf, cost = _generate(_native_only(native, item["question"]), base_url, model, mm)
    abst = is_abstention(res.text)
    trace.append(("native", round(conf, 3), abst))
    # best-answer key: prefer a NON-abstaining answer, then higher confidence
    best = ((not abst, conf), res, "native")
    rung_stop = "native"

    if not _resolved(res, conf):
        early = False
        for rung in _rung_order(item, order_mode):
            recovered, extra = _recover(rung, item, buffer, history, item["question"], k)
            r2, c2, cc = _generate(_augmented(recovered, native, item["question"]),
                                   base_url, model, mm)
            cost = cost + cc + extra       # cost accrues for every rung actually run
            a2 = is_abstention(r2.text)
            trace.append((str(rung), round(c2, 3), a2))
            key = (not a2, c2)
            if key > best[0]:
                best = (key, r2, str(rung))
            if _resolved(r2, c2):          # early-exit at the first rung that resolves
                res, conf, rung_stop, early = r2, c2, str(rung), True
                break
        if not early:
            # nothing resolved -> return the BEST answer seen (non-abstain preferred),
            # not the last (more retrieval can regress: bigger haystack ranks plant lower)
            (_, res, rung_stop) = best
            conf = best[0][1]

    return {"result": res, "cost": cost, "arm": None,
            "meta": {"rung_stop": rung_stop, "theta": theta, "order_mode": order_mode,
                     "trace": trace, "native_turns": native_turns,
                     "buffer_turns": buffer_turns}}


# --- arm wrappers (registered in arms.py) ---------------------------------- #
def run_b0_sw(item, base_url, model, native_turns=6, buffer_turns=45, **_):
    history = item["messages"]
    native, _b = _windows(history, native_turns, buffer_turns)
    res, conf, cost = _generate(_native_only(native, item["question"]), base_url, model, _mock_meta(item))
    return {"result": res, "cost": cost, "arm": "B0_sw",
            "meta": {"rung_stop": "native", "native_turns": native_turns}}


def _cascade_arm(name, order_mode):
    def runner(item, base_url, model, **kw):
        out = run_controller(item, base_url, model, order_mode=order_mode, **kw)
        out["arm"] = name
        return out
    return runner


run_b2_reactive = _cascade_arm("B2_reactive", "reactive")
run_b3_random = _cascade_arm("B3_random", "random")
run_b4_cascade = _cascade_arm("B4_cascade", "ascending")
run_b5_cascade_no_rung0 = _cascade_arm("B5_cascade_no_rung0", "no_rung0")
