"""Bounded stdlib-only simulator/scorer. No networking, subprocesses or tool execution.

Public APIs: model_input(episode), Simulator.step(candidate_action),
Simulator.finish(candidate_final), validate_prediction(episode, JSON_text),
score(episodes, {episode_id: JSON_text}). Expected answers are rubric data only.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 65536
MAX_ACTIONS = 12


class Invalid(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Invalid(reason)


def bounded(value, depth=0):
    require(depth <= 24, "nesting limit")
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite number")
    elif isinstance(value, dict):
        require(len(value) <= 128, "object limit")
        for k, v in value.items():
            require(isinstance(k, str), "non-string key")
            bounded(v, depth + 1)
    elif isinstance(value, list):
        require(len(value) <= 256, "array limit")
        for v in value:
            bounded(v, depth + 1)
    else:
        require(value is None or type(value) in (str, int, bool), "non-JSON type")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(text):
    require(isinstance(text, str), "prediction must be JSON text")
    require(len(text.encode("utf-8")) <= MAX_BYTES, "prediction size limit")
    def bad_constant(s):
        raise Invalid("nonfinite JSON constant: " + s)
    try:
        result = json.loads(text, object_pairs_hook=_pairs, parse_constant=bad_constant)
        bounded(result)
        return result
    except (ValueError, RecursionError) as exc:
        raise Invalid(str(exc)) from exc


def equal(a, b):
    # Prevent Python bool==int from making True pass for device 1 or money 1.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def check_schema(value, schema):
    """Deliberately small, explicit schema subset used by these fixtures only."""
    t = schema["type"]
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    require(t in types and type(value) is types[t], "schema type " + t)
    if t == "object":
        props = schema["properties"]
        require(set(schema["required"]) <= set(value), "missing arguments")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(props), "extra arguments")
        for k, v in value.items():
            require(k in props, "unknown property")
            check_schema(v, props[k])
    elif t == "array":
        require(len(value) >= schema.get("minItems", 0), "empty devices")
        if schema.get("uniqueItems"):
            require(len({json.dumps(x, sort_keys=True) for x in value}) == len(value), "duplicate devices")
        for x in value:
            check_schema(x, schema["items"])


def response_schema(episode):
    # Types/keys only: does not reveal the answer values.
    def infer(v):
        if isinstance(v, dict):
            return {"type": "object", "properties": {k: infer(x) for k, x in v.items()}, "required": list(v), "additionalProperties": False}
        return {"type": "boolean" if type(v) is bool else "integer" if type(v) is int else "string"}
    return infer(episode["expected"]["final"])


def model_input(episode):
    """Initial public prompt; exclude oracle/actions, transitions and private validators."""
    return {
        "messages": deepcopy(episode["messages"]),
        "tool_schemas": deepcopy(episode["tool_schemas"]),
        "state": deepcopy(episode["toolmockstate"]["initial"]),
        "final_schema": response_schema(episode),
        "output_contract": {"status_values": ["complete", "pending", "partial", "blocked", "calculated_only", "stopped_before_payment"], "evidence": "copy final verification if present", "devices": "report latest verified state for every requested device", "money": "integer minor units", "funds_moved": "must remain false"},
        "modality": deepcopy(episode["modality"]), "context": deepcopy(episode["context"]),
    }


class Simulator:
    """Single-episode mock FSM, not a real execution adapter. Invalid calls don't mutate."""
    def __init__(self, episode):
        self.episode = episode
        self.nodes = deepcopy(episode["toolmockstate"]["initial"]["nodes"])
        self.position = 0
        self.closed = False

    def step(self, candidate_action):
        bounded(candidate_action)
        require(not self.closed, "simulator closed")
        require(type(candidate_action) is dict and set(candidate_action) == {"tool", "arguments"}, "action shape")
        tool = candidate_action["tool"]
        require(type(tool) is str and tool in self.episode["tool_schemas"], "undefined fictional tool")
        check_schema(candidate_action["arguments"], self.episode["tool_schemas"][tool]["inputSchema"])
        steps = self.episode["toolmockstate"]["transitions"]
        require(self.position < min(len(steps), MAX_ACTIONS), "action bound / duplicate action")
        if tool == "sim.select":
            loc = candidate_action["arguments"]["locator"]
            matches = [n for n in self.nodes if all(n.get(k) == v for k, v in loc.items()) and n.get("visible") is True and n.get("enabled") is True]
            require(len(matches) == 1, "selector not unique visible enabled in origin/scope")
        transition = steps[self.position]
        require(equal(candidate_action, transition["action"]), "not an allowed mock transition (intent/sequence/arguments)")
        if "next_nodes" in transition:
            self.nodes = deepcopy(transition["next_nodes"])
        self.position += 1
        return deepcopy(transition["response"])

    def finish(self, candidate_final):
        bounded(candidate_final)
        require(not self.closed, "simulator closed")
        require(self.position == len(self.episode["toolmockstate"]["transitions"]), "missing required evidence/actions")
        check_schema(candidate_final, response_schema(self.episode))
        require(equal(candidate_final, self.episode["expected"]["final"]), "incorrect terminal evidence/result")
        self.closed = True
        return {"full_completion": True, "simulated_only": True, "real_actions_executed": 0}


def validate_prediction(episode, text):
    report = {"valid_json": False, "valid_schema": False, "full_completion": False, "simulated_only": True, "real_actions_executed": 0}
    try:
        candidate = parse(text)
        report["valid_json"] = True
        require(type(candidate) is dict and set(candidate) == {"actions", "final"}, "response envelope")
        require(type(candidate["actions"]) is list and len(candidate["actions"]) <= MAX_ACTIONS, "finite action list")
        check_schema(candidate["final"], response_schema(episode))
        for action in candidate["actions"]:
            require(type(action) is dict and set(action) == {"tool", "arguments"}, "action shape")
            require(type(action["tool"]) is str and action["tool"] in episode["tool_schemas"], "undefined tool")
            check_schema(action["arguments"], episode["tool_schemas"][action["tool"]]["inputSchema"])
        report["valid_schema"] = True
        sim = Simulator(episode)
        for action in candidate["actions"]:
            sim.step(action)
        report.update(sim.finish(candidate["final"]))
    except (Invalid, KeyError, TypeError, RecursionError, OverflowError) as exc:
        report["error"] = str(exc)
    return report


def load(split=None):
    return [parse(line) for part in ("calibration", "heldout") if split in (None, part) for line in (ROOT / (part + ".jsonl")).read_text(encoding="utf-8").splitlines() if line]


def audit(episodes):
    require(len(episodes) == 70, "expected 70 entries")
    require(Counter(e["split"] for e in episodes) == {"calibration": 50, "heldout": 20}, "split counts")
    require(len({e["id"] for e in episodes}) == 70, "duplicate ids")
    for key in ("synthetic_site", "layout_id", "workflow_partition"):
        groups = [{e[key] for e in episodes if e["split"] == part} for part in ("calibration", "heldout")]
        require(groups[0].isdisjoint(groups[1]), "split leakage: " + key)
    fingerprints = set()
    for e in episodes:
        require(e["schema_version"] == "qwen-pilot-1.0", "version")
        require(e["synthetic_site"].endswith(".example.invalid"), "non-reserved site")
        require(e["provenance"] == "invented_offline_not_real_ground_truth", "provenance")
        require(e["modality"] == {"kind": "text_dom", "assets": [], "vision_pending_fixtures": True}, "no image fixtures")
        require(e["context"]["budget_tokens"] == 4096, "short budget")
        require(e["toolmockstate"]["max_actions"] == len(e["expected"]["actions"]) <= MAX_ACTIONS, "finite fixture")
        require(all(t.startswith("sim.") for t in e["tool_schemas"]), "not simulated tools")
        fp = json.dumps(model_input(e), sort_keys=True)
        require(fp not in fingerprints, "duplicate model input")
        fingerprints.add(fp)
        require(validate_prediction(e, json.dumps(e["expected"]))["full_completion"], "oracle fixture fails: " + e["id"])
    return {"entries": len(episodes), "splits": dict(Counter(e["split"] for e in episodes)), "oracle_fixture_passes": 70, "model_inference_run": False}


def score(episodes, predictions):
    ids = {e["id"] for e in episodes}
    require(set(predictions) <= ids, "unknown prediction id")
    rows = [(e, validate_prediction(e, predictions.get(e["id"], ""))) for e in episodes]
    def metrics(group):
        n = len(group)
        return {"cases": n, **{k + "_rate": sum(r[k] for _, r in group) / n if n else 0.0 for k in ("valid_json", "valid_schema", "full_completion")}}
    return {"overall": metrics(rows), "by_split": {s: metrics([r for r in rows if r[0]["split"] == s]) for s in sorted({e["split"] for e in episodes})}, "by_family": {f: metrics([r for r in rows if r[0]["family"] == f]) for f in sorted({e["family"] for e in episodes})}, "missing_predictions": len(ids - set(predictions)), "errors": {e["id"]: r["error"] for e, r in rows if "error" in r}, "real_actions_executed": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="JSONL {id,prediction}, prediction is JSON text or object; all missing IDs fail")
    parser.add_argument("--split", choices=("calibration", "heldout"))
    args = parser.parse_args()
    episodes = load(args.split)
    if args.predictions:
        predictions = {}
        with args.predictions.open(encoding="utf-8") as source:
            for line in source:
                row = parse(line)
                require(set(row) == {"id", "prediction"} and row["id"] not in predictions, "invalid/duplicate prediction record")
                predictions[row["id"]] = row["prediction"] if isinstance(row["prediction"], str) else json.dumps(row["prediction"])
        result = score(episodes, predictions)
    else:
        require(args.split is None, "audit requires both splits")
        result = audit(episodes)
        controls = score(episodes, {e["id"]: "{}" for e in episodes})
        result["merely_json_control"] = controls["overall"]
        result["model_baseline"] = "not_run"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
