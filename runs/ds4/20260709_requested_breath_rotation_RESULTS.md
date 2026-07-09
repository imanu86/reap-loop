# Requested DS4 Breath/Rotation A/B - 2026-07-09

Prompt: HTML cyberpunk landing page. Local RTX 3060, same DS4 binary, `ctx=6144`,
streaming enabled, routing CSV off, hidden SPEX off. Each run uses one 64-token
warmup request before the measured request.

The requested tests were run twice: first with cache 128, then with cache 256.

## Results

| cache | test | wall s | avg t/s | breath | breath end | tokens after breath end | coherent until token est | useful tokens after return est | repeat | notes |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|
| 128 | breath K0 -> K23 | 407.314 | 2.28 | tok 170 -> K0 | tok 250 -> K23 | 550 | 244 | 0 | 1 | Coherent only until about tok 244; after return to K23, 0 useful tokens measured. |
| 128 | breath K96 -> K23 | 311.644 | 2.92 | tok 170 -> K96 | tok 250 -> K23 | 550 | 156 | 0 | 1 | Coherent only until about tok 156, before breath fired. |
| 128 | K23 static, no breath | 245.958 | 3.39 | - | - | - | 116 | - | 1 | Coherent only until about tok 116. |
| 128 | K23 rotate32, no breath | 274.447 | 3.03 | - | - | - | 800 | - | 0 | 23 rotations; no triple-repeat loop detected through 800 tokens. |
| 256 | breath K0 -> K23 | 399.099 | 2.36 | tok 194 -> K0 | tok 274 -> K23 | 526 | 198 | 0 | 1 | Coherent only until about tok 198; after return to K23, 0 useful tokens measured. |
| 256 | breath K96 -> K23 | 274.638 | 3.14 | tok 151 -> K96 | tok 231 -> K23 | 569 | 95 | 0 | 1 | Coherent only until about tok 95, before breath fired. |
| 256 | K23 static, no breath | 271.796 | 3.06 | - | - | - | 116 | - | 1 | Coherent only until about tok 116. |
| 256 | K23 rotate32, no breath | 318.003 | 2.61 | - | - | - | 800 | - | 0 | 23 rotations; no triple-repeat loop detected through 800 tokens, slower than cache128. |

`coherent until token est` is derived from the first detected triple repeated
span in the generated text and mapped back to the 800 streamed content events.
It is a reproducible token-level proxy for coherence, not a human quality grade. The raw
content, request/response JSON, stream events, env, and server logs are kept in:

- `runs/ds4/20260709_requested4_html800_cache128`
- `runs/ds4/20260709_requested4_html800_cache256`

## Readout

- The breath variants did not answer the user's quality question positively:
  after breath returns to K23, useful post-return tokens were zero in both K0 and
  K96 variants for both cache settings.
- K96 breath is faster than K0 breath, but in these runs it fires too late to
  rescue the HTML stream.
- Static K23 is fast but degenerates early.
- Raw-router K-constant rotation is the only tested actuator that carried the
  generation to 800 tokens without the repeat detector firing.
- Cache128 was faster than cache256 for rotation in this run (`3.03` vs
  `2.61 t/s`) and should remain the throughput candidate for this exact setup.

## Next Test: Top Expert Higher Precision

Hypothesis: for each layer, keep the highest-weight / most valuable expert in a
higher precision format, e.g. int4/Q4 instead of the routed expert's int2-ish
format, while the rest remain compressed.

Immediate check: `/root/models/ds4-staticQ4.gguf` does **not** provide Q4 routed
experts. It has routed experts in the same formats as `ds4-2bit.gguf`:
`ffn_gate_exps=iq2_xxs`, `ffn_up_exps=iq2_xxs`, `ffn_down_exps=q2_k`.
Only static tensors such as `ffn_gate_inp` differ (`q4_k` in staticQ4 versus
`f16` in ds4-2bit).

Therefore this is not a runtime flag test. The next useful test is to build a
sidecar or hybrid GGUF containing selected routed experts at Q4/int4, then run
the same HTML/code prompts against: all routed int2-ish baseline, top1-per-layer
Q4, and possibly topN-per-layer Q4.
