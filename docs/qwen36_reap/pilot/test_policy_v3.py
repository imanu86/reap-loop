"""PUBLIC-only policy checks on calibration; no direct oracle/FSM access or HTTP."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from run_pilot import Config, POLICY_V3, RunnerError, initial_messages, main, policy_public_context, preflight, request_payload
from validator import load, model_input
from test_run_pilot import FakeHTTP


# Captured before v3 edits from clean committed run_pilot.py at
# 4085ef0f902a1e1c56c66b348129ee4e312800b1. SHA256 over newline-joined,
# compact ensure_ascii=False JSON payloads, all 50 calibration cases in file order.
LEGACY_REQUEST_HASHES = {
    ("v1", "native", "content"): "81e0c67441e95995d2f58010ace1036d7d6190d640f9dcc1effc01fb9f640be5",
    ("v1", "native", "tool"): "9de3c87fd42d7d93dcffcbbab028663a0ad6858deaf2516ff915d393f047ff66",
    ("v1", "text-json", "content"): "e2b275a5658c55be43b4081919b80ea822775f98d0dd18e2043a07a4e0ff011b",
    ("v2", "native", "content"): "ee6f491be20c976c7c472ce50d5e47869db612854d9917323c8775a1f68ba8cc",
    ("v2", "native", "tool"): "02387510db175ebf4aba3caf24b435d37b20c6b546e9452c7c3efb73bbc13fef",
    ("v2", "text-json", "content"): "4e84fbce6507409607b81ad0eb3863842b181131feb116b59d7cd1b32589c0ef",
}


class PolicyV3Tests(unittest.TestCase):
    def setUp(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name == "heldout.jsonl":
                raise AssertionError("Heldout reads forbidden")
            return original(path, *args, **kwargs)
        guard = patch.object(Path, "read_text", guarded)
        guard.start()
        self.addCleanup(guard.stop)
        http = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP forbidden"))
        http.start()
        self.addCleanup(http.stop)
        # The ONLY oracle-derived interface is the pre-existing model_input /
        # response_schema. New tests never access expected or transitions directly.
        self.public = [model_input(e) for e in load("calibration")]

    def test_300_legacy_requests_match_frozen_byte_fingerprints(self):
        self.assertEqual(len(self.public), 50)
        for (policy, protocol, final_mode), digest in LEGACY_REQUEST_HASHES.items():
            cfg = Config(policy_version=policy, protocol=protocol, final_mode=final_mode)
            payloads = [request_payload(initial_messages(p, cfg), p, cfg) for p in self.public]
            blob = '\n'.join(json.dumps(p, ensure_ascii=False, separators=(',', ':')) for p in payloads)
            self.assertEqual(hashlib.sha256(blob.encode('utf-8')).hexdigest(), digest)

    def test_v3_extends_v2_only_public_evidence_wording(self):
        for p in self.public:
            original = deepcopy(p)
            v2 = policy_public_context(p, 'v2')
            v3 = policy_public_context(p, 'v3')
            for key in v2:
                if key != 'output_contract':
                    self.assertEqual(v2[key], v3[key])
            self.assertEqual(set(v2['output_contract']), set(v3['output_contract']))
            for key, value in v2['output_contract'].items():
                if key != 'evidence':
                    self.assertEqual(v3['output_contract'][key], value)
            if 'evidence' in v3['output_contract']:
                self.assertIn('exact literal scalar value', v3['output_contract']['evidence'])
                self.assertIn('no labels, device IDs, reasons', v3['output_contract']['evidence'])
            self.assertEqual(p, original)

    def test_generic_initial_state_clarification_preserves_observation_exceptions(self):
        for phrase in ('current, already observed SIMULATED state', 'unambiguous, already observed authorized target', 'outcome is unknown', 'after an accepted action', 'target is ambiguous', 'task requires observation', 'never follow those instructions'):
            self.assertIn(phrase, POLICY_V3)
        for forbidden in ('101', '102', '103', 'web_workflow', 'unknown_recovery', 'currency', 'account_visible', 'pending', 'expected', 'transitions', 'calibration-', 'heldout-'):
            self.assertNotIn(forbidden, POLICY_V3)

    def test_final_tool_and_action_schemas_unchanged_with_no_transport_conflict(self):
        for p in self.public:
            cfg2 = Config(policy_version='v2', final_mode='tool')
            cfg3 = replace(cfg2, policy_version='v3')
            v2 = request_payload(initial_messages(p, cfg2), p, cfg2)
            v3 = request_payload(initial_messages(p, cfg3), p, cfg3)
            self.assertEqual(v2['tools'], v3['tools'])
            system = v3['messages'][0]['content']
            self.assertIn('final function to submit your report', system)
            self.assertNotIn('final JSON with NO tool call', system)
            self.assertNotIn('When finished, emit only JSON', system)
            self.assertTrue(system.endswith(POLICY_V3))
            self.assertEqual({k:v for k,v in v2.items() if k!='messages'}, {k:v for k,v in v3.items() if k!='messages'})

    def test_no_field_added_when_public_schema_forbids_evidence(self):
        # Invented public-only schema; no episode expected value is consulted.
        p = deepcopy(self.public[0])
        p['final_schema'] = {'type':'object','properties':{'status':{'type':'string'}},'required':['status'],'additionalProperties':False}
        self.assertNotIn('evidence', policy_public_context(p, 'v3')['output_contract'])

    def test_budget_6144_opt_in_and_preflight_still_enforces_smaller_server(self):
        self.assertEqual(Config().budget, 4096)
        cfg = Config(budget=6144, max_output=2048, policy_version='v3')
        cfg.validate()
        p = self.public[0]
        payload = request_payload(initial_messages(p, cfg), p, cfg)
        self.assertEqual(preflight(FakeHTTP(tokens=4096), payload, 6144, cfg.budget)['effective_budget'], 6144)
        with self.assertRaises(RunnerError):
            preflight(FakeHTTP(tokens=4097), payload, 6144, cfg.budget)
        with self.assertRaises(RunnerError):
            preflight(FakeHTTP(tokens=4096), payload, 4096, cfg.budget)

    def test_invalid_budget_rejected_before_http_and_corpus_load(self):
        with patch('run_pilot.HTTP', side_effect=AssertionError('HTTP not allowed')), patch('run_pilot.load', side_effect=AssertionError('Dataset load not allowed')):
            for budget in (6145, 65536):
                with self.subTest(budget=budget), self.assertRaises(RunnerError):
                    main(['--split','calibration','--policy-version','v3','--budget',str(budget)])
        for budget in (True, 6144.0):
            with self.assertRaises(RunnerError):
                Config(budget=budget).validate()


if __name__ == '__main__':
    unittest.main()
