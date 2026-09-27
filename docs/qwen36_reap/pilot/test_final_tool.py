"""Calibration-only mocked terminal transport tests; never reads heldout or HTTP."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_pilot import Config, RunnerError, decode_turn, initial_messages, preflight, request_payload, run_episode, verify_qwen_tool_definitions
from validator import Invalid, Simulator, load, model_input
from test_run_pilot import FakeHTTP, scripted


# Frozen FORMAT from the public <tools> section in completed baseline calibration
# render: one JSON tool definition per line, system role, before user messages.
# No private prompt, reasoning, expected value or heldout content copied here.
def qwen_render(tools, body="Synthetic public context"):
    return '<|im_start|>system\n# Tools\n<tools>\n' + '\n'.join(json.dumps(t) for t in tools) + '\n</tools>\nUse final when done.<|im_end|>\n<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n<think>\n'


class QwenFakeHTTP(FakeHTTP):
    def request(self, path, payload=None):
        if path == "/apply-template":
            self.requests.append((path, deepcopy(payload)))
            return {"prompt": qwen_render(payload.get("tools", []))}
        return super().request(path, payload)


def terminal(value, call_id="terminal-call"):
    return {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": call_id, "function": {"name": "final", "arguments": json.dumps(value)}}]}


def with_terminal(e):
    messages = scripted(e)
    messages[-1] = terminal(e["expected"]["final"])
    return messages


class FinalToolTests(unittest.TestCase):
    def setUp(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name == "heldout.jsonl":
                raise AssertionError("Heldout reads forbidden")
            return original(path, *args, **kwargs)
        guard = patch.object(Path, "read_text", guarded)
        guard.start()
        self.addCleanup(guard.stop)
        http = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden"))
        http.start()
        self.addCleanup(http.stop)
        self.episodes = load("calibration")
        self.currency = next(e for e in self.episodes if e["family"] == "currency")
        self.cfg = Config(final_mode="tool", policy_version="v2")

    def test_schema_exact_public_copy_no_oracle_access(self):
        p = model_input(self.currency)
        before = deepcopy(p)
        request = request_payload(initial_messages(p, self.cfg), p, self.cfg)
        finals = [t for t in request["tools"] if t["function"]["name"] == "final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["function"]["parameters"], p["final_schema"])
        finals[0]["function"]["parameters"]["properties"].clear()
        self.assertEqual(p, before)
        text = json.dumps(initial_messages(p, self.cfg))
        self.assertNotIn('final JSON with NO tool call', text)
        self.assertNotIn('When finished, emit only JSON', text)
        self.assertNotIn(self.currency["id"], text)

    def test_unknown_final_still_rejected_by_default(self):
        self.assertEqual(Config().final_mode, "content")
        with self.assertRaises(RunnerError):
            decode_turn(terminal(self.currency["expected"]["final"]), model_input(self.currency), "native")
        legacy_content = scripted(self.currency)[-1]
        with self.assertRaises(RunnerError):
            decode_turn(legacy_content, model_input(self.currency), "native", "tool")

    def test_all_calibration_roundtrip_and_no_post_final_request(self):
        for e in self.episodes:
            with self.subTest(e=e["id"]):
                messages = with_terminal(e)
                messages.append(terminal(e["expected"]["final"], "must-not-request"))
                http = QwenFakeHTTP(messages)
                result = run_episode(e, http, self.cfg, 4096)
                self.assertTrue(result["full_completion"], result.get("private_error"))
                self.assertEqual(result["actions"], e["expected"]["actions"])
                self.assertEqual(result["final"], e["expected"]["final"])
                self.assertEqual(result["final_mode"], "tool")
                self.assertEqual(len(http.messages), 1)
                self.assertEqual(len(result["turns"]), len(e["expected"]["actions"]) + 1)

    def test_early_final_and_wrong_value_remain_semantic_failures(self):
        e = self.episodes[0]
        early = run_episode(e, QwenFakeHTTP([terminal(e["expected"]["final"])]), self.cfg, 4096)
        self.assertEqual(early["error_class"], "final_validation")
        self.assertIn("missing required", early["private_error"])
        wrong = deepcopy(self.currency["expected"]["final"])
        wrong["remaining_minor"] += 1
        result = run_episode(self.currency, QwenFakeHTTP([terminal(wrong)]), self.cfg, 4096)
        self.assertEqual(result["error_class"], "final_validation")
        self.assertIn("incorrect terminal", result["private_error"])

    def test_bad_schema_malformed_json_and_wrong_selector_fail(self):
        for bad in ({"extra": True}, {**self.currency["expected"]["final"], "extra": True}):
            result = run_episode(self.currency, QwenFakeHTTP([terminal(bad)]), self.cfg, 4096)
            self.assertEqual(result["error_class"], "final_validation")
        msg = terminal({})
        msg["tool_calls"][0]["function"]["arguments"] = 'NaN'
        self.assertEqual(run_episode(self.currency, QwenFakeHTTP([msg]), self.cfg, 4096)["error_class"], "response_format")
        e = self.episodes[0]
        messages = with_terminal(e)
        action = messages[0]["tool_calls"][0]["function"]
        args = json.loads(action["arguments"])
        args["locator"]["scope"] = "unobserved-scope"
        action["arguments"] = json.dumps(args)
        self.assertEqual(run_episode(e, QwenFakeHTTP(messages), self.cfg, 4096)["error_class"], "action_validation")

    def test_duplicate_final_id_checked_before_finish(self):
        e = self.episodes[0]
        messages = with_terminal(e)
        messages[-1]["tool_calls"][0]["id"] = messages[0]["tool_calls"][0]["id"]
        with patch("run_pilot.Simulator.finish", side_effect=AssertionError("Must reject duplicate before finish")):
            result = run_episode(e, QwenFakeHTTP(messages), self.cfg, 4096)
        self.assertEqual(result["error_class"], "response_format")
        self.assertEqual(result["private_error"], "Duplicate tool call id")
        self.assertIsNone(result["final"])

    def test_mixed_calls_content_rejected_before_any_dispatch(self):
        e = self.episodes[0]
        final = terminal(e["expected"]["final"])
        action = scripted(e)[0]
        for calls in ([final["tool_calls"][0], action["tool_calls"][0]], [action["tool_calls"][0], final["tool_calls"][0]]):
            bad = {"role": "assistant", "content": None, "tool_calls": calls}
            with patch("run_pilot.Simulator.step", side_effect=AssertionError("No step")), patch("run_pilot.Simulator.finish", side_effect=AssertionError("No finish")):
                result = run_episode(e, QwenFakeHTTP([bad]), self.cfg, 4096)
            self.assertEqual(result["error_class"], "response_format")
            self.assertEqual(result["actions"], [])
        final["content"] = '{"kind":"final"}'
        self.assertEqual(run_episode(e, QwenFakeHTTP([final]), self.cfg, 4096)["error_class"], "response_format")

    def test_closed_simulator_guards_unchanged(self):
        sim = Simulator(self.currency)
        sim.finish(self.currency["expected"]["final"])
        with self.assertRaises(Invalid):
            sim.finish(self.currency["expected"]["final"])
        with self.assertRaises(Invalid):
            sim.step({"tool": "sim.observe", "arguments": {"devices": [101]}})

    def test_preflight_checks_actual_system_definition_not_prose(self):
        p = model_input(self.currency)
        request = request_payload(initial_messages(p, self.cfg), p, self.cfg)
        tools = request["tools"]
        verify_qwen_tool_definitions(qwen_render(tools), tools)
        for malformed in (qwen_render(tools[:-1]), qwen_render(tools[:-1], '<tools>\n' + json.dumps(tools[-1]) + '\n</tools>'), qwen_render([{**t, "function": {**t["function"], "parameters": {"type": "object"}}} if t["function"]["name"] == "final" else t for t in tools])):
            with self.assertRaises(RunnerError):
                verify_qwen_tool_definitions(malformed, tools)
        result = preflight(QwenFakeHTTP(), request, 4096, 4096)
        expected_hash = hashlib.sha256(json.dumps([1] * 100, separators=(',', ':')).encode('ascii')).hexdigest()
        self.assertEqual(result["prompt_token_ids_sha256"], expected_hash)
        with self.assertRaises(RunnerError):
            preflight(QwenFakeHTTP(tokens=4096), request, 4096, 4096)

    def test_config_and_collision_rejected_without_http(self):
        for cfg in (Config(final_mode="unknown"), Config(protocol="text-json", final_mode="tool")):
            with self.assertRaises(RunnerError):
                cfg.validate()
        p = model_input(self.currency)
        p["tool_schemas"]["final"] = deepcopy(p["tool_schemas"]["sim.observe"])
        with self.assertRaises(RunnerError):
            request_payload([], p, self.cfg)


if __name__ == "__main__":
    unittest.main()
