"""Pure payload tests: no corpus reads, network, inference or output files."""
from dataclasses import replace
import json
import unittest

from run_pilot import Config, request_payload


class DiagnosticRawTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Synthetic payload test only"}]
        self.public = {"tool_schemas": {"sim.observe": {"description": "Synthetic observation", "inputSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}}

    def test_default_false_and_normal_payload_frozen(self):
        self.assertFalse(Config().diagnostic_raw)
        base = {"model": "local-pilot", "messages": self.messages, "stream": False, "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 0, "max_tokens": 512, "cache_prompt": True}
        for policy in ("v1", "v2"):
            for protocol in ("native", "text-json"):
                expected = dict(base)
                if protocol == "native":
                    expected.update({"tools": [{"type": "function", "function": {"name": "sim_observe", "description": "Simulated tool sim.observe: Synthetic observation", "parameters": self.public["tool_schemas"]["sim.observe"]["inputSchema"]}}], "tool_choice": "auto", "parallel_tool_calls": False})
                actual = request_payload(self.messages, self.public, Config(protocol=protocol, policy_version=policy))
                self.assertEqual(json.dumps(actual), json.dumps(expected))
                self.assertNotIn("verbose", actual)
                self.assertNotIn("return_tokens", actual)

    def test_enabled_adds_exactly_two_flags_for_every_policy_protocol(self):
        for policy in ("v1", "v2"):
            for protocol in ("native", "text-json"):
                cfg = Config(protocol=protocol, policy_version=policy)
                plain = request_payload(self.messages, self.public, cfg)
                raw = request_payload(self.messages, self.public, replace(cfg, diagnostic_raw=True))
                self.assertEqual(set(raw) - set(plain), {"verbose", "return_tokens"})
                self.assertIs(raw.pop("verbose"), True)
                self.assertIs(raw.pop("return_tokens"), True)
                self.assertEqual(json.dumps(raw), json.dumps(plain))

    def test_config_serialization_records_opt_in(self):
        self.assertIs(json.loads(json.dumps(Config().__dict__))["diagnostic_raw"], False)
        self.assertIs(json.loads(json.dumps(Config(diagnostic_raw=True).__dict__))["diagnostic_raw"], True)


if __name__ == "__main__":
    unittest.main()
