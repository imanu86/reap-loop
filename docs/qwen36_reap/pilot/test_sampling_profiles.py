"""Sampling-only tests: no dataset reads, HTTP, model or filesystem outputs."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import patch

from run_pilot import Config, RunnerError, main, request_payload


class SamplingProfileTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "Synthetic sampling payload test"}]
        self.public = {"tool_schemas": {}}

    def test_defaults_preserve_legacy_greedy_payload_bytes(self):
        expected = {"model": "local-pilot", "messages": self.messages, "stream": False, "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 0, "max_tokens": 512, "cache_prompt": True}
        for policy in ("v1", "v2"):
            cfg = Config(protocol="text-json", policy_version=policy)
            self.assertEqual(cfg.sampling_profile, "greedy")
            self.assertEqual(cfg.seed, 0)
            self.assertEqual(json.dumps(request_payload(self.messages, self.public, cfg)), json.dumps(expected))

    def test_qwen_coding_exact_parameters_only(self):
        for policy in ("v1", "v2"):
            for protocol in ("native", "text-json"):
                base_cfg = Config(protocol=protocol, policy_version=policy, seed=17)
                before = request_payload(self.messages, self.public, base_cfg)
                after = request_payload(self.messages, self.public, replace(base_cfg, sampling_profile="qwen-coding"))
                expected = dict(before)
                expected.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0, "repeat_penalty": 1})
                self.assertEqual(after, expected)
                self.assertEqual(after["seed"], 17)
                self.assertNotIn("chat_template_kwargs", after)
                self.assertEqual(set(after) - set(before), {"presence_penalty", "repeat_penalty"})

    def test_seed_unsigned_32_bit_exact_type(self):
        for seed in (0, 1, 4294967294):
            cfg = Config(seed=seed)
            cfg.validate()
            self.assertEqual(request_payload(self.messages, self.public, cfg)["seed"], seed)
        for seed in (-1, 4294967295, 4294967296, True, False, 1.0, "1", None):
            with self.subTest(seed=seed), self.assertRaises(RunnerError):
                request_payload(self.messages, self.public, Config(seed=seed))

    def test_invalid_settings_rejected_before_http_or_dataset_load(self):
        with patch("run_pilot.HTTP", side_effect=AssertionError("No HTTP construction")), patch("run_pilot.load", side_effect=AssertionError("No dataset read")):
            for seed in ('-1', '4294967295', '4294967296'):
                with self.assertRaises(RunnerError):
                    main(['--split', 'calibration', '--seed', seed])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--split', 'calibration', '--sampling-profile', 'unknown'])
            with self.assertRaises(RunnerError):
                Config(sampling_profile="unknown").validate()

    def test_manifest_config_records_profile_seed(self):
        cfg = Config(sampling_profile="qwen-coding", seed=2)
        recorded = json.loads(json.dumps(cfg.__dict__))
        self.assertEqual(recorded["sampling_profile"], "qwen-coding")
        self.assertEqual(recorded["seed"], 2)


if __name__ == "__main__":
    unittest.main()
