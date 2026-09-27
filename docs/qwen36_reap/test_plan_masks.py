import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plan_masks as planner


class MaskPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        (self.run / 'pilot').mkdir(parents=True)
        self.corpus = self.root / 'calibration.jsonl'
        self.ids = [f'calibration-synthetic-{i}' for i in range(50)]
        self.corpus.write_text(''.join(json.dumps({'id': i}) + '\n' for i in self.ids), encoding='utf-8')
        self.put('manifest.json', dict(scope='calibration_integration_not_performance',
                 eligible_quality_evaluation=True, skip_chat_parsing=False, model_sha256=planner.MODEL_SHA,
                 runtime_arm='candidate', runtime_sha256={'llama.dll': planner.CORE_SHA}, trace_enabled=True,
                 environment={'QWEN36_REAP_TRACE': str(self.run / 'routing.jsonl')}))
        self.put('shutdown.json', {'stopped': True, 'pid': 123})
        self.put('server.pid.json', {'pid': 123})
        self.put('pilot/manifest.json', dict(split='calibration', episode_ids=self.ids,
                 dataset_sha256=planner.digest(self.corpus), config=planner.FROZEN,
                 server_props={'default_generation_settings': {'n_ctx': 4096, 'params': {'speculative.types': 'none'}}}))
        self.put('pilot/summary.json', dict(cases=50, split='calibration', real_actions_executed=0, full_completion_rate=1.0))
        req = dict(temperature=0, top_k=1, top_p=1, min_p=0, seed=0, max_tokens=1024,
                   tool_choice='auto', parallel_tool_calls=False,
                   tools=[{'function': {'name': 'final', 'parameters': {'type': 'object'}}}])
        self.rows = [dict(id=i, split='calibration', protocol='native', policy_version='v2', final_mode='tool',
                    real_actions_executed=0, full_completion=True, error_class=None, final={'ok': True},
                    turns=[dict(request=req, response={'choices': [{'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}},
                                preflight={'prompt_tokens_preflight': 1}, selection={'kind': 'final', 'value': {'ok': True}})])
                     for i in self.ids]
        self.transcripts()
        header = dict(record_type='header', schema_version=1, model_sha256=planner.MODEL_SHA,
                      architecture='qwen35moe', layer_count=40, expert_count=256, top_k=8)
        trace = [header] + [dict(record_type='route', schema_version=1, batch_id=1, decode_call_id=1,
                 layer=layer, token_index=0, token_position=0, token_id=42, seq_ids=[0], phase='unknown',
                 ids=list(range(8)), weights=[0.125] * 8) for layer in range(40)]
        (self.run / 'routing.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in trace), encoding='utf-8')

    def put(self, name, obj):
        (self.run / name).write_text(json.dumps(obj), encoding='utf-8')

    def transcripts(self):
        (self.run / 'pilot/transcripts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows), encoding='utf-8')

    def test_one_aggregate_eight_unapproved_masks(self):
        with patch.object(planner, 'aggregate', wraps=planner.aggregate) as aggregate:
            r = planner.plan(self.run, self.corpus, self.root / 'out')
            aggregate.assert_called_once()
        self.assertEqual(len(r['candidates']), 8)
        self.assertFalse(r['quality_approval'])
        self.assertFalse(r['model_weights_written'])
        self.assertEqual(len(list((self.root / 'out').glob('*.json'))), 9)
        with self.assertRaisesRegex(ValueError, 'NEW'):
            planner.plan(self.run, self.corpus, self.root / 'out')

    def test_active_run_rejected_before_trace(self):
        self.put('shutdown.json', {'stopped': False, 'pid': 123})
        with patch.object(planner, 'aggregate') as aggregate:
            with self.assertRaisesRegex(ValueError, 'stopped'):
                planner.plan(self.run, self.corpus, self.root / 'out')
            aggregate.assert_not_called()
        self.assertFalse((self.root / 'out').exists())

    def test_readiness_floor_and_summary_recomputed(self):
        for row in self.rows[:11]:
            row['full_completion'] = False
            row['error_class'] = 'final_validation'
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Summary disagrees'):
            planner.validate_run(self.run, self.corpus)
        self.put('pilot/summary.json', dict(cases=50, split='calibration', real_actions_executed=0, full_completion_rate=0.78))
        with self.assertRaisesRegex(ValueError, '40/50'):
            planner.validate_run(self.run, self.corpus)

    def test_wrong_actual_sampling_and_infrastructure_error(self):
        self.rows[0]['turns'][0]['request']['seed'] = 1
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Actual request'):
            planner.validate_run(self.run, self.corpus)
        self.rows[0]['turns'][0]['request']['seed'] = 0
        self.rows[0]['error_class'] = 'transport'
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Infrastructure'):
            planner.validate_run(self.run, self.corpus)

    def test_missing_case_and_bypass_rejected(self):
        self.rows.pop()
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'transcript cases'):
            planner.validate_run(self.run, self.corpus)
        m = planner.load(self.run / 'manifest.json')
        m['skip_chat_parsing'] = True
        self.put('manifest.json', m)
        with self.assertRaisesRegex(ValueError, 'bypass'):
            planner.validate_run(self.run, self.corpus)


if __name__ == '__main__':
    unittest.main()
