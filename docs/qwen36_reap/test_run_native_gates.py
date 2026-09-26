"""Coordinator CPU tests: no model, network or process execution."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_native_gates as runner


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = Path(self.tmp.name) / 'trace.jsonl'
        self.metadata = {'prompt_token_ids': [50], 'decode_calls': [
            {'helper_call_ordinal_1based': 1, 'phase': 'prefill', 'position_start': 0, 'n_tokens': 1, 'seq_id': 0},
            {'helper_call_ordinal_1based': 2, 'phase': 'decode', 'position_start': 1, 'n_tokens': 1, 'seq_id': 0, 'token_id': 51}]}
        self.rows = [{'record_type': 'route', 'decode_call_id': i + 1, 'phase': 'unknown',
                      'token_position': i, 'token_id': 50 + i, 'seq_ids': [0], 'layer': 0} for i in range(2)]

    def check(self, rows):
        self.trace.write_text(json.dumps({'record_type': 'header'}) + '\n' +
                              ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        with patch.object(runner, 'load_metadata', return_value=(self.metadata, None)):
            return runner.check_trace_tokens('mock-prefix', self.trace)

    def test_single_token_prefill_not_mislabeled_decode(self):
        self.assertEqual(self.check(self.rows), {'prefill': 1, 'decode': 1})

    def test_input_mismatch_rejected(self):
        for field, value in [('token_id', 99), ('token_position', 99), ('phase', 'decode'),
                             ('seq_ids', [1]), ('decode_call_id', 99)]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError, msg=field):
                self.check(rows)

    def test_missing_duplicate_and_wrong_order_identity_rejected(self):
        for rows in [self.rows[:1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):
                self.check(rows)

    def test_exclusive_json_output(self):
        path = Path(self.tmp.name) / 'immutable.json'
        runner.save(path, {'original': True})
        with self.assertRaises(FileExistsError):
            runner.save(path, {'original': False})
        self.assertEqual(json.loads(path.read_text()), {'original': True})

    def test_experimental_environment_scrubbed(self):
        with patch.dict(os.environ, {'LLAMA_ARG_WARMUP': '1', 'QWEN_LOCAL_API_KEY': 'fake-only',
                                     'GGML_CUDA_Q4K_OCC12': '1', 'PATH': 'fake-path'}, clear=True):
            env, explicit = runner.environment()
        for forbidden in ('LLAMA_ARG_WARMUP', 'QWEN_LOCAL_API_KEY', 'GGML_CUDA_Q4K_OCC12'):
            self.assertNotIn(forbidden, env)
        self.assertEqual(env['PATH'], 'fake-path')
        self.assertEqual(env['LLAMA_MOE_CACHE_BATCH'], '1')
        self.assertEqual(env['LLAMA_MOE_ELASTIC'], '0')
        self.assertTrue(all(env[k] == v for k, v in explicit.items()))


if __name__ == '__main__':
    unittest.main()
