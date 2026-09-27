# Optional public final tool: implemented, model evaluation pending

**Status: implemented opt-in; offline mocked tests passed; NOT model-evaluated.** Parent authorized implementation after run127 completed: six calibration case-seed observations, zero completions, six redundant-action failures, twelve tool-call finishes and zero truncations. No hidden/partial results or heldout contents/results were used. This is a NEW protocol, not a correction to historical scores.

Implemented in `run_pilot.py`, default `--final-mode content` unchanged; enable explicitly with `--final-mode tool` only for native protocol. `test_final_tool.py` adds ten calibration-only tests; together with sampling/raw/policy suites, 27 tests pass. Model quality is not measured by these mocked roundtrips. The parent preregisters original-runtime greedy/seed0/v2/1024 on the same two calibration cases, changing ONLY final_mode against retained old content controls. No inference/network was executed by this implementation task.

Preflight verifies exact native JSON definitions inside the first Qwen system `<tools>` section, strictly before the first user role; unsupported layouts or injected user-side definitions fail closed. It also records `prompt_token_ids_sha256` over compact ASCII JSON of the existing `/tokenize` token-ID array; no additional request is made. Future input-token alignment can compare this fingerprint directly. Dataset, Simulator and validator semantics remain unchanged; the design below remains the normative safety contract.

## Purpose and non-goals

A later explicit terminal function could remove ambiguity between native action calls and prose/JSON final responses. It does not repair model quality by definition, reveal the next action, infer task completion for the model, weaken the simulator, or establish native-tool reliability. Choosing this transport changes the evaluation protocol and must receive a new manifest identity; results cannot be silently pooled with the existing content-final protocol.

Historical outputs calling an undefined `final` function remain failures under their original protocol. Do not rescore them after introducing a tool with that name. In particular, a correct numerical payload embedded in an undefined historical tool is not retrospectively a completed episode.

## Proposed opt-in contract

Append configuration `final_mode: str = "content"`; CLI `--final-mode content|tool`. Default **content** must retain old request/message bytes, tool inventory, parsing and scoring behavior. Merely adding configuration metadata to new manifests does not justify altering old model requests.

Initially require `protocol=native` when `final_mode=tool`, failing configuration validation before HTTP for incompatible text-json mode. Keep text-json/content behavior unchanged. This avoids inventing an undocumented hybrid fallback.

In tool mode append exactly one public native function:

```text
name: final
parameters: deepcopy(public["final_schema"])
description: Submit your final report using this schema. This ends the episode;
             it does not perform an action or establish that the task succeeded.
```

Arguments are the **final object itself**, not an extra `{kind,final}` envelope and not an action trace. The function name `final` is a reserved transport name, distinct from simulated action aliases such as `sim_observe`. Detect and reject an alias collision before constructing the request. Do not add `sim.final` to the dataset or simulator's operation inventory.

Only `model_input(episode)` supplies model-visible data. The tool declaration accesses `public["final_schema"]`, never `episode.expected`, transition count, remaining private steps, hidden FSM state, expected status, expected values, or oracle actions. A deep copy prevents accidental mutation of public schema data.

The schema constrains permitted keys/types only. Do not synthesize `const`, narrow enums, examples or defaults from expected answers. Existing public status guidance may remain public guidance, but must not be narrowed using the episode's correct result. Grammar-generated structural correctness is not evidence of semantic completion.

## Public instruction replacement, not contradictory append

In tool mode replace only the current generic transport instructions that demand final JSON content. Avoid retaining a higher-priority contradictory instruction to report via prose/JSON. Suggested generic wording:

> Choose exactly one native function call for this turn: a necessary authorized simulated action, or `final` to submit your final report. When the requested result or required safe stopping point is established by observations, call `final` with arguments matching its public schema. Do not call another action merely to transmit or reconfirm a report. Do not emit final JSON as message content and do not combine final with an action.

Retain existing trust-boundary, payment-stop, bounded recovery and unresolved-device guidance. Keep policy_version and sampling_profile orthogonal to final_mode; do not silently enable policy v2, change thinking, or alter sampling when switching final transport. No task-specific answer examples, concrete device IDs, semantic outcomes, or hidden next-action hints.

`tools` passed to generation and `/apply-template` must contain identical declarations, including `final`. Re-run existing prompt-token preflight with the added schema; keep the same total context/output budgets and fail on overflow. The reserved name must be rendered as an actual native function definition, not merely appear in instructional prose. Alias-name substring checks alone are insufficient for `final`, because that word also appears in instructions: strengthen any future omission test around the actual rendered definition/schema rather than reporting a false guarantee from a bare substring.

## Decode and dispatch: mandatory validation order

A future implementation should normalize a turn to `(kind, value, call_id, transport_name)` without executing anything. Suggested ordered checks:

1. Require one response choice and the existing acceptable finish conditions; reject truncation and malformed assistant messages as before.
2. Tool mode requires exactly one native `tool_call`, of type function with a nonempty string call ID. Keep the existing content rule: no simultaneous message content/final, using the same empty-content semantics as existing native actions. Multiple calls, mixed call+content, or simultaneous action+final are terminal errors. Separate `reasoning_content` remains private logging, not an additional action.
3. Validate the function name against the explicit request inventory. In content mode, unknown `final` must **still fail**. In tool mode, `final` has a special terminal mapping; other calls use the existing alias map.
4. Parse the JSON arguments with the existing bounded strict parser. Never eval or execute text.
5. **Check call-ID uniqueness before BOTH action and final dispatch.** The current runner only checks IDs on the action path after branching on `kind == final`; moving final into a tool path without relocating this check would permit a duplicate final call ID. Perform all shape/name/argument/ID validation before simulator mutation, record insertion or success decisions. Keep one seen-ID namespace across actions and terminal calls, even when function names differ.
6. If the normalized kind is action: invoke unchanged `Simulator.step(predicted_action)`, append only its mock response to the model history, and preserve existing bounded action/turn behavior.
7. If it is final: invoke unchanged `Simulator.finish(predicted_final_object)` directly, **not** `Simulator.step`, never an expected value. On success close/end the episode. On failure stop with the existing private final-validation error; no rubric-fed retry.

A terminal call does not count as a simulated action in `actions`; it consumes one response turn. Preserve the current maximum of twelve actions plus a terminal turn. Never infer when to force `tool_choice=final`, remove action tools dynamically using hidden state, fabricate a final call, or auto-finalize because a tool response looked successful. The model must choose final on its own from public evidence.

## Closure and result recording

After an accepted final there must be no further model request, simulated action or second final. Keep the existing closed-simulator guard, and test any future runner/session wrapper's terminal-state guard. If multiple calls arrive in one response, reject the entire turn before dispatching the first one, including a valid final followed by an action.

Maintain existing prediction compatibility:

```text
{id: evaluator_id, prediction: {actions: predicted_simulated_actions,
                                final: predicted_final_arguments}}
```

This is transport normalization only; the terminal call is retained separately in private transcript selection metadata with its native function name and call ID. Do not append the final function to `actions` or fabricate a mock tool response after successful termination. Preserve raw responses and provenance so content-final versus tool-final runs remain distinguishable.

New manifests must record final_mode, policy_version, sampling_profile, seed, runner/dataset hashes, selected calibration IDs and actual native tool definitions or their stable hash. Existing files/old manifests remain untouched. Full-completion scoring still requires `Simulator.finish`; structural validity enforced by server grammar must be reported separately, not counted as task success.

## Required offline tests before any authorized run

Tests should read **calibration only**, or use invented minimal public schemas; block real HTTP and heldout reads. Existing cross-split test suites are not needed for protocol tuning.

| Area | Required assertion |
|---|---|
| Default compatibility | content/default requests and messages remain byte-identical for v1/v2 and native/text-json; undefined `final` still rejected |
| Public declaration | one function `final`; parameters deep-equal copied public schema; mutations do not alter public input; no expected-values/constants/FSM dependency |
| Configuration | unknown final_mode and tool-final with text-json fail before HTTP; reserved-name collisions fail |
| Template/preflight | generation and apply-template receive same final schema; omission detection cannot pass just because prose says final; context bounds remain enforced |
| Valid terminal transport | after predicted allowed actions, predicted final arguments reach `Simulator.finish` unchanged; never reach step; output trace omits terminal from actions |
| Early final | schema-valid but premature final fails with missing evidence/actions; no oracle feedback retry |
| Wrong value | all keys/types valid but incorrect semantic result fails unchanged |
| Wrong structure | extra/missing keys, malformed JSON, nonfinite values and oversized arguments rejected |
| Wrong selection | action with wrong origin/scope/selector or unresolved-device set still fails unchanged, even if model could subsequently call final |
| Duplicate IDs | final reusing a previous action ID fails before finish; repeated action ID still fails; shared namespace across different function names |
| Mixed calls | two calls, action+final, final+action and call+content rejected before any simulator mutation |
| Terminal closure | zero requests/actions after accepted final; second final or post-final action rejected by session/simulator guard |
| No hidden automation | no automatic final call, no tool-choice forcing based on oracle/state; exact model-selected arguments are evaluated |
| Protocol separation | historical undefined-final traces remain failures under content mode; no retroactive rescore; final_mode recorded |

Passing these tests validates software contracts, not model behavior. A mock final call backed by a fixture oracle is a unit-test input only, never an inference success.

## Decision gate after run127

The parent confirmed run127 complete and authorized the implementation; model-evaluation approval and orchestration remain with the parent. Do not inspect partial runs or change active sources. The parent preregisters a small calibration-only comparison with fixed cases/seeds/runtime/sampling/policy/budgets and only final_mode changed. Freeze implementation and test evidence before starting. Keep original scores and new protocol scores separate, retain every failure, and treat any benefit as protocol-specific pending independent evaluation. No heldout tuning, pruning or expert selection follows from this design.
