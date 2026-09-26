"""All HTTP is fake. No inference/network or run directories are created."""
import copy
import json
import unittest
from unittest.mock import patch

from validator import load, model_input
from run_pilot import Config, HTTP, NoRedirect, RunnerError, decode_turn, initial_messages, loopback_url, preflight, request_payload, run_episode, server_context, select_episodes


class SmokeSelectionTests(unittest.TestCase):
    def test_calibration_limit_only(self):
        self.assertEqual(select_episodes('calibration', 2), load('calibration')[:2])
        self.assertEqual(len(select_episodes('calibration')), 50)
        self.assertEqual(len(select_episodes('heldout')), 20)
        for split, limit in [('heldout', 1), ('calibration', 0), ('calibration', 51), ('calibration', True)]:
            with self.assertRaises(RunnerError):
                select_episodes(split, limit)


class FakeHTTP:
    def __init__(self, messages=(), tokens=100, omit_tools=False):
        self.messages = list(messages)
        self.requests = []
        self.tokens = tokens
        self.omit_tools = omit_tools

    def request(self, path, payload=None):
        self.requests.append((path, copy.deepcopy(payload)))
        if path == "/props":
            return {"default_generation_settings": {"n_ctx": 4096}}
        if path == "/apply-template":
            data = copy.deepcopy(payload)
            if self.omit_tools:
                data.pop("tools", None)
            return {"prompt": json.dumps(data)}
        if path == "/tokenize":
            return {"tokens": [1] * self.tokens}
        if path == "/v1/chat/completions":
            return {"choices": [{"message": self.messages.pop(0), "finish_reason": "stop"}], "usage": {"prompt_tokens": self.tokens, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 2}}, "timings": {"prompt_n": self.tokens, "predicted_ms": 2}}
        raise AssertionError(path)


def scripted(episode, protocol="native"):
    """Test-only oracle-backed mock model; never an inference/model-quality result."""
    result = []
    for index, action in enumerate(episode["expected"]["actions"]):
        if protocol == "native":
            result.append({"role": "assistant", "content": None, "reasoning_content": "synthetic reasoning", "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {"name": action["tool"].replace(".", "_"), "arguments": json.dumps(action["arguments"])}}]})
        else:
            result.append({"role": "assistant", "content": json.dumps({"kind": "action", "action": action})})
    result.append({"role": "assistant", "content": json.dumps({"kind": "final", "final": episode["expected"]["final"]})})
    return result


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = load()

    def setUp(self):
        self.no_network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden in tests"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def test_url_gate_and_protected_ports(self):
        self.assertEqual(loopback_url("http://localhost:8116"), "http://127.0.0.1:8116")
        self.assertEqual(loopback_url("http://[::1]:8116"), "http://[::1]:8116")
        for url in ("http://localhost:8100", "http://127.0.0.1:8104", "http://[::1]:8104", "http://example.invalid:8116", "http://192.168.1.1:8116", "https://localhost:8116", "http://localhost:8116/v1", "http://user:pass@localhost:8116", "http://localhost:8116?x=1", "http://localhost.evil:8116", "http://2130706433:8116"):
            with self.subTest(url=url), self.assertRaises(RunnerError):
                loopback_url(url)
        with self.assertRaises(RunnerError):
            HTTP()
        with self.assertRaises(RunnerError):
            NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.invalid")

    def test_native_roundtrip_all_70_mocked(self):
        for e in self.episodes:
            with self.subTest(e=e["id"]):
                fake = FakeHTTP(scripted(e))
                result = run_episode(e, fake, Config(), 4096)
                self.assertTrue(result["full_completion"], result.get("private_error"))
                self.assertEqual(result["actions"], e["expected"]["actions"])
                self.assertEqual(result["final"], e["expected"]["final"])
                self.assertEqual(result["real_actions_executed"], 0)
                requests = [p for endpoint, p in fake.requests if endpoint == "/v1/chat/completions"]
                self.assertEqual(len(requests), len(e["expected"]["actions"]) + 1)
                self.assertTrue(all("tools" in p and p["parallel_tool_calls"] is False for p in requests))
                self.assertEqual(sum(m["role"] == "tool" for m in requests[-1]["messages"]), len(e["expected"]["actions"]))
                self.assertIsNotNone(result["turns"][0]["usage"])
                self.assertIsNotNone(result["turns"][0]["timings"])

    def test_text_fallback_distinct_and_no_native_tools(self):
        e = self.episodes[0]
        fake = FakeHTTP(scripted(e, "text-json"))
        result = run_episode(e, fake, Config(protocol="text-json"), 4096)
        self.assertTrue(result["full_completion"])
        self.assertEqual(result["protocol"], "text-json")
        for endpoint, payload in fake.requests:
            if endpoint == "/v1/chat/completions":
                self.assertNotIn("tools", payload)

    def test_public_api_no_ids_split_labels_and_opaque_sites(self):
        import re
        for e in self.episodes:
            public = model_input(e)
            self.assertNotIn("id", public)
            self.assertNotIn("split", public)
            self.assertNotIn(e["id"], json.dumps(public))
            self.assertIsNotNone(re.fullmatch(r"https://s-[0-9a-f]{16}\.example\.invalid", e["synthetic_site"]))
            self.assertNotIn(e["family"].replace("_", "-"), e["synthetic_site"])

    def test_no_oracle_future_response_or_ids_in_initial_requests(self):
        e = copy.deepcopy(self.episodes[0])
        e["expected"]["final"]["evidence"] = "PRIVATE_ORACLE_SENTINEL"
        e["toolmockstate"]["transitions"][0]["response"]["private_future"] = "FUTURE_SENTINEL"
        p = model_input(e)
        initial = json.dumps(request_payload(initial_messages(p, Config()), p, Config()))
        for forbidden in ("PRIVATE_ORACLE_SENTINEL", "FUTURE_SENTINEL", e["id"], '"expected"', '"transitions"', '"validators"'):
            self.assertNotIn(forbidden, initial)

    def test_predicted_wrong_action_stops_without_retry_or_rubric_leak(self):
        e = self.episodes[0]
        responses = scripted(e)
        args = json.loads(responses[0]["tool_calls"][0]["function"]["arguments"])
        args["locator"]["name"] = "wrong predicted control"
        responses[0]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
        fake = FakeHTTP(responses)
        result = run_episode(e, fake, Config(), 4096)
        self.assertFalse(result["full_completion"])
        self.assertEqual(result["error_class"], "action_validation")
        self.assertEqual(sum(p == "/v1/chat/completions" for p, _ in fake.requests), 1)
        self.assertEqual(result["actions"][0]["arguments"]["locator"]["name"], "wrong predicted control")
        self.assertNotIn(result["private_error"], json.dumps(fake.requests))

    def test_multi_calls_and_unknown_alias_rejected(self):
        e = self.episodes[0]
        p = model_input(e)
        message = scripted(e)[0]
        message["tool_calls"] *= 2
        with self.assertRaises(RunnerError):
            decode_turn(message, p, "native")
        message = scripted(e)[0]
        message["tool_calls"][0]["function"]["name"] = "real_payment"
        with self.assertRaises(RunnerError):
            decode_turn(message, p, "native")

    def test_early_final_fails_and_no_followup(self):
        e = self.episodes[0]
        fake = FakeHTTP([scripted(e)[-1]])
        result = run_episode(e, fake, Config(), 4096)
        self.assertEqual(result["error_class"], "final_validation")
        self.assertEqual(len(result["turns"]), 1)

    def test_context_fail_closed_before_generation(self):
        e = self.episodes[0]
        for ctx, tokens in ((4096, 3585), (1024, 513)):
            fake = FakeHTTP(scripted(e), tokens=tokens)
            result = run_episode(e, fake, Config(), ctx)
            self.assertEqual(result["error_class"], "preflight")
            self.assertIn("Context budget exceeded", result["private_error"])
            self.assertFalse(any(p == "/v1/chat/completions" for p, _ in fake.requests))
        fake = FakeHTTP(scripted(e), tokens=3584)
        self.assertTrue(run_episode(e, fake, Config(), 4096)["full_completion"])

    def test_native_template_omission_fails(self):
        e = self.episodes[0]
        result = run_episode(e, FakeHTTP(scripted(e), omit_tools=True), Config(), 4096)
        self.assertEqual(result["error_class"], "preflight")
        self.assertIn("omitted native tool schema", result["private_error"])

    def test_template_kwargs_and_greedy_alignment(self):
        e = self.episodes[0]
        with self.assertRaises(RunnerError):
            Config(thinking="off").validate()
        cfg = Config(thinking="off", template_supports_thinking=True)
        fake = FakeHTTP(scripted(e))
        self.assertTrue(run_episode(e, fake, cfg, 4096)["full_completion"])
        templates = [p for endpoint, p in fake.requests if endpoint == "/apply-template"]
        chats = [p for endpoint, p in fake.requests if endpoint == "/v1/chat/completions"]
        for template, chat in zip(templates, chats):
            self.assertEqual(template["chat_template_kwargs"], chat["chat_template_kwargs"])
            self.assertFalse(chat["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(chat["temperature"], 0)
            self.assertEqual(chat["top_k"], 1)
            self.assertEqual(template["tools"], chat["tools"])
            self.assertEqual(template["messages"], chat["messages"])

    def test_max_turns_no_implicit_completion(self):
        e = self.episodes[0]
        result = run_episode(e, FakeHTTP(scripted(e)), Config(max_turns=1), 4096)
        self.assertFalse(result["full_completion"])
        self.assertEqual(result["error_class"], "turn_limit")

    def test_props_contract_required(self):
        self.assertEqual(server_context(FakeHTTP())[0], 4096)
        class Missing:
            def request(self, *args):
                return {}
        with self.assertRaises(RunnerError):
            server_context(Missing())


if __name__ == "__main__":
    unittest.main()
