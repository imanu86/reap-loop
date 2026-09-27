import copy
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
        self.episodes = [{'id': value, 'family': f'synthetic-family-{i // 5}'} for i, value in enumerate(self.ids)]
        self.corpus.write_text(''.join(json.dumps(e) + '\n' for e in self.episodes), encoding='utf-8')
        self.put('manifest.json', dict(scope='calibration_integration_not_performance',
                 eligible_quality_evaluation=True, skip_chat_parsing=False, model_sha256=planner.MODEL_SHA,
                 runtime_arm='candidate', runtime_sha256={'llama.dll': planner.CORE_SHA}, trace_enabled=True,
                 policy_version='v2', coordinator_sha256='c' * 64,
                 environment={'QWEN36_REAP_TRACE': str(self.run / 'routing.jsonl')}))
        self.put('shutdown.json', {'stopped': True, 'pid': 123})
        self.put('server.pid.json', {'pid': 123})
        self.put('pilot/manifest.json', dict(split='calibration', episode_ids=self.ids,
                 dataset_sha256=planner.digest(self.corpus), config=planner.FROZEN, runner_sha256='d' * 64,
                 server_props={'default_generation_settings': {'n_ctx': 4096, 'params': {'speculative.types': 'none'}}}))
        self.put('pilot/summary.json', dict(cases=50, split='calibration', real_actions_executed=0, full_completion_rate=1.0))
        req = dict(temperature=0, top_k=1, top_p=1, min_p=0, seed=0, max_tokens=1024,
                   tool_choice='auto', parallel_tool_calls=False,
                   tools=[{'function': {'name': 'final', 'parameters': {'type': 'object'}}}])
        self.rows = [dict(id=e['id'], family=e['family'], split='calibration', protocol='native', policy_version='v2', final_mode='tool',
                    real_actions_executed=0, full_completion=True, error_class=None, final={'ok': True},
                    turns=[dict(request=req, response={'choices': [{'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}},
                                preflight={'prompt_tokens_preflight': 1}, selection={'kind': 'final', 'value': {'ok': True}})])
                     for e in self.episodes]
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

    def set_version(self, version):
        manifest = planner.load(self.run / 'manifest.json')
        manifest['policy_version'] = version
        self.put('manifest.json', manifest)
        manifest = planner.load(self.run / 'pilot/manifest.json')
        manifest['config']['policy_version'] = version
        self.put('pilot/manifest.json', manifest)
        for row in self.rows:
            row['policy_version'] = version
        self.transcripts()

    def refresh_corpus(self):
        self.corpus.write_text(''.join(json.dumps(e) + '\n' for e in self.episodes), encoding='utf-8')
        manifest = planner.load(self.run / 'pilot/manifest.json')
        manifest['dataset_sha256'] = planner.digest(self.corpus)
        self.put('pilot/manifest.json', manifest)

    def set_failures(self, indices):
        for i in indices:
            self.rows[i]['full_completion'] = False
            self.rows[i]['error_class'] = 'final_validation'
        self.transcripts()
        self.put('pilot/summary.json', dict(cases=50, split='calibration', real_actions_executed=0,
                                          full_completion_rate=sum(r['full_completion'] for r in self.rows) / 50))

    def test_default_v2_legacy_and_explicit_v3_valid(self):
        v2 = planner.validate_run(self.run, self.corpus)
        self.assertEqual(v2['expected_config'], planner.FROZEN)
        self.assertEqual(v2['policy_version'], 'v2')
        self.assertTrue(v2['gpu_screen_ready'])
        self.set_version('v3')
        with self.assertRaisesRegex(ValueError, 'policy version mismatch'):
            planner.validate_run(self.run, self.corpus)  # no silent inference of v3
        v3 = planner.validate_run(self.run, self.corpus, policy_version='v3')
        self.assertEqual(v3['expected_config'], dict(planner.FROZEN, policy_version='v3'))
        self.assertNotEqual(v2['protocol_sha256'], v3['protocol_sha256'])
        self.assertTrue(v3['gpu_screen_ready'])
        out = self.root / 'v3-out'
        result = planner.plan(self.run, self.corpus, out, policy_version='v3')
        self.assertEqual(result['policy_version'], 'v3')
        self.assertTrue(result['gpu_screen_ready'])
        self.assertFalse(result['quality_approval'])
        for record in result['candidates']:
            mask = planner.load(out / record['file'])
            self.assertEqual(mask['policy_version'], 'v3')
            self.assertEqual(mask['protocol_sha256'], result['provenance']['protocol_sha256'])
            self.assertFalse(mask['quality_approval'])
            self.assertFalse(record['quality_approval'])
            self.assertTrue(mask['gpu_screen_ready'])
            self.assertEqual(result['mask_sha256'][record['file']], planner.digest(out / record['file']))

    def test_policy_version_must_match_manifest_and_every_record(self):
        with self.assertRaisesRegex(ValueError, 'v2 or v3'):
            planner.validate_run(self.run, self.corpus, policy_version='v1')
        manifest = planner.load(self.run / 'pilot/manifest.json')
        manifest['config']['policy_version'] = 'v3'
        self.put('pilot/manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
            planner.validate_run(self.run, self.corpus)
        manifest['config']['policy_version'] = 'v2'
        self.put('pilot/manifest.json', manifest)
        self.rows[-1]['policy_version'] = 'v3'
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Case protocol'):
            planner.validate_run(self.run, self.corpus)

    def test_family_counts_from_corpus_and_actual_completion_not_summary(self):
        self.set_failures(range(5))  # >=40 total, but first family has zero completions
        summary = planner.load(self.run / 'pilot/summary.json')
        summary.update(gpu_screen_ready=True, quality_approval=True,
                       family_counts={'synthetic-family-0': {'cases': 5, 'full_completions': 5}})
        self.put('pilot/summary.json', summary)
        proof = planner.validate_run(self.run, self.corpus)
        self.assertEqual(proof['successes'], 45)
        self.assertEqual(proof['family_counts']['synthetic-family-0'], {'cases': 5, 'full_completions': 0})
        self.assertEqual(proof['families_without_full_completion'], ['synthetic-family-0'])
        self.assertFalse(proof['gpu_screen_ready'])
        result = planner.plan(self.run, self.corpus, self.root / 'diagnostic-out')
        self.assertFalse(result['gpu_screen_ready'])
        self.assertFalse(result['quality_approval'])
        for entry in result['candidates']:
            self.assertFalse(entry['gpu_screen_ready'])
            self.assertFalse(entry['quality_approval'])
            self.assertFalse(planner.load(self.root / 'diagnostic-out' / entry['file'])['quality_approval'])

    def test_ready_exactly40_with_every_family_and_recomputes_false_summary_flag(self):
        self.set_failures(range(0, 50, 5))
        summary = planner.load(self.run / 'pilot/summary.json')
        summary['gpu_screen_ready'] = False
        self.put('pilot/summary.json', summary)
        proof = planner.validate_run(self.run, self.corpus)
        self.assertEqual(proof['successes'], 40)
        self.assertTrue(proof['gpu_screen_ready'])
        self.assertTrue(all(v == {'cases': 5, 'full_completions': 4} for v in proof['family_counts'].values()))

    def test_exactly10_families_and_transcript_agreement_required(self):
        original = copy.deepcopy(self.episodes)
        for bad in (None, '', ' ', 123):
            self.episodes = copy.deepcopy(original)
            self.episodes[0]['family'] = bad
            self.refresh_corpus()
            with self.assertRaisesRegex(ValueError, 'family'):
                planner.validate_run(self.run, self.corpus)
        self.episodes = copy.deepcopy(original)
        self.episodes[0]['family'] = 'eleventh-synthetic-family'
        self.refresh_corpus()
        with self.assertRaisesRegex(ValueError, 'exactly10'):
            planner.validate_run(self.run, self.corpus)
        self.episodes = copy.deepcopy(original)
        for episode in self.episodes:
            if episode['family'] == 'synthetic-family-0':
                episode['family'] = 'synthetic-family-1'
        self.refresh_corpus()
        with self.assertRaisesRegex(ValueError, 'exactly10'):
            planner.validate_run(self.run, self.corpus)
        self.episodes = original
        self.refresh_corpus()
        self.rows[0]['family'] = 'synthetic-family-1'
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Case family'):
            planner.validate_run(self.run, self.corpus)

    def test_full_source_config_and_protocol_hash_proof(self):
        manifest = planner.load(self.run / 'pilot/manifest.json')
        manifest['config']['extra_recorded_option'] = {'value': 'synthetic'}
        self.put('pilot/manifest.json', manifest)
        proof = planner.validate_run(self.run, self.corpus)
        self.assertEqual(proof['full_source_config'], manifest['config'])
        self.assertEqual(proof['full_source_config_sha256'], planner.object_digest(manifest['config']))
        binding = {'expected_config': proof['expected_config'], 'recorded_source_sha256': proof['recorded_source_sha256']}
        self.assertEqual(proof['protocol_sha256'], planner.object_digest(binding))
        self.assertEqual(proof['recorded_source_sha256'], {'pilot_runner_sha256': 'd' * 64, 'coordinator_sha256': 'c' * 64})
        manifest['config']['extra_recorded_option']['value'] = 'different'
        self.put('pilot/manifest.json', manifest)
        changed = planner.validate_run(self.run, self.corpus)
        self.assertNotEqual(proof['full_source_config_sha256'], changed['full_source_config_sha256'])
        manifest['runner_sha256'] = 'e' * 64
        self.put('pilot/manifest.json', manifest)
        self.assertNotEqual(changed['protocol_sha256'], planner.validate_run(self.run, self.corpus)['protocol_sha256'])
        del manifest['runner_sha256']
        self.put('pilot/manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'recorded pilot runner'):
            planner.validate_run(self.run, self.corpus)

    def test_partial_files_rejected_before_aggregate_or_output(self):
        transcript = self.run / 'pilot/transcripts.jsonl'
        transcript.write_bytes(transcript.read_bytes().rstrip(b'\n'))
        with patch.object(planner, 'aggregate') as aggregate:
            with self.assertRaisesRegex(ValueError, 'Uncommitted transcript'):
                planner.plan(self.run, self.corpus, self.root / 'partial-out')
            aggregate.assert_not_called()
        self.assertFalse((self.root / 'partial-out').exists())
        self.transcripts()
        summary = self.run / 'pilot/summary.json'
        summary.write_bytes(b'{"cases":')
        with self.assertRaises(ValueError):
            planner.validate_run(self.run, self.corpus)

    def test_v3_keeps_frozen_request_and_generation_lower_bound(self):
        self.set_version('v3')
        self.rows[0]['turns'][0]['response']['usage']['completion_tokens'] = 3
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'generation lower bound'):
            planner.plan(self.run, self.corpus, self.root / 'small-trace', policy_version='v3')
        self.assertFalse((self.root / 'small-trace').exists())
        self.rows[0]['turns'][0]['request']['max_tokens'] = 512
        self.transcripts()
        with self.assertRaisesRegex(ValueError, 'Actual request'):
            planner.validate_run(self.run, self.corpus, policy_version='v3')

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
