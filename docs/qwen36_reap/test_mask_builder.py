import copy
import hashlib
import json
import math
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mask_builder import aggregate, build, build_from_aggregate, selection_recipe, recipe_digest

SHA = 'a' * 64


def fixture():
    header = {'schema_version': 1, 'record_type': 'header', 'architecture': 'qwen35moe',
              'model_sha256': SHA, 'layer_count': 40, 'expert_count': 256, 'top_k': 8,
              'modeltag': 'Qwen3.6-35B-A3B', 'model_hash_source': 'launcher_asserted',
              'mask_enabled': False, 'routed_scale': 1.0, 'expected_weight_sum': 1.0,
              'phase_source': 'unknown; annotate externally from request metadata',
              'weights_source': 'cb_eval:ffn_moe_weights_effective'}
    rows = []
    for layer in range(40):
        for token in range(2):
            rows.append({'schema_version': 1, 'record_type': 'route', 'batch_id': 7, 'decode_call_id': 3,
                'layer': layer, 'phase': 'unknown', 'token_position': token,
                'token_index': token, 'token_id': 50 + token, 'seq_ids': [0],
                'ids': list(range(token * 8, token * 8 + 8)), 'weights': [0.125] * 8})
    return [header] + rows


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.jsonl'
        self.data = fixture()
        self.save()

    def save(self):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in self.data), encoding='utf-8')

    def make(self, **kwargs):
        return build([self.path], SHA, split='calibration', **kwargs)

    def test_dynamic_top8_and_full_denominator(self):
        totals, counts, _ = aggregate([self.path], SHA)
        self.assertEqual(len(totals[0]), 256)
        self.assertEqual(counts, [2] * 40)
        self.assertEqual(totals[0][7], 0.125)
        self.assertEqual(totals[0][15], 0.125)
        self.assertEqual(self.make()['layers']['0'], list(range(16)))

    def test_keep8_preserves_shape_and_reports_lost_mass(self):
        result = self.make(keep=8)
        self.assertEqual(result['layers']['39'], list(range(8)))
        self.assertEqual(result['diagnostics']['39']['retained_mass_fraction'], 0.5)

    def test_random_control_matches_every_layer_pool(self):
        ranked, random = self.make(), self.make(random_seed=42)
        self.assertEqual([len(v) for v in ranked['layers'].values()], [len(v) for v in random['layers'].values()])
        self.assertEqual(random, self.make(random_seed=42))
        self.assertNotEqual(ranked['layers'], random['layers'])

    def test_reject_heldout_and_keep_below_topk(self):
        with self.assertRaises(ValueError):
            build([self.path], SHA, split='heldout')
        for keep in [0, 7, 257, True]:
            with self.assertRaises(ValueError):
                self.make(keep=keep)

    def test_reject_top6_wrong_hash_and_mtp_geometry(self):
        for key, value in [('top_k', 6), ('layer_count', 41), ('model_sha256', 'b' * 64)]:
            self.data = fixture()
            self.data[0][key] = value
            self.save()
            with self.assertRaises(ValueError):
                self.make()

    def test_reject_duplicate_experts_and_nan(self):
        for change in ['duplicate', 'nan', 'negative', 'sum', 'out_of_range']:
            self.data = fixture()
            if change == 'duplicate': self.data[1]['ids'][1] = 0
            if change == 'nan': self.data[1]['weights'][0] = float('nan')
            if change == 'negative': self.data[1]['weights'][0] = -0.125
            if change == 'sum': self.data[1]['weights'][0] = 1
            if change == 'out_of_range': self.data[1]['ids'][0] = 256
            self.save()
            with self.assertRaises(ValueError, msg=change):
                self.make()

    def test_reject_missing_duplicate_and_misaligned_rows(self):
        for change in ['missing', 'duplicate', 'position']:
            self.data = fixture()
            if change == 'missing': self.data.pop()
            if change == 'duplicate': self.data.append(copy.deepcopy(self.data[-1]))
            if change == 'position': self.data[-1]['token_position'] = 99
            self.save()
            with self.assertRaises(ValueError, msg=change):
                self.make()

    def test_multiple_complete_batches_and_order_guard(self):
        extra = copy.deepcopy(self.data[1:])
        for r in extra: r['batch_id'] = 8
        self.data.extend(extra)
        self.save()
        self.assertEqual(self.make()['diagnostics']['0']['tokens'], 4)
        for r in extra: r['batch_id'] = 6
        self.save()
        with self.assertRaises(ValueError):
            self.make()

    def test_missing_or_malformed_actual_token_metadata(self):
        cases = {
            'token_id': [None, True, 1.5, '50', -1, 1 << 31, [], {}],
            'token_position': [None, False, 0.0, '0', -1, 1 << 31],
            'token_index': [None, True, 0.0, -1, 1 << 31],
            'decode_call_id': [None, True, 1.0, '3', -1, 0, 1 << 64],
            'batch_id': [None, True, 1.0, -1, 0, 1 << 64],
            'seq_ids': [None, [], [False], [-1], [1 << 31], [0, 0], ['0']],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.data = fixture()
                    self.data[1][key] = value
                    self.save()
                    with self.assertRaises(ValueError):
                        aggregate([self.path], SHA)
            self.data = fixture()
            self.data[1].pop(key)
            self.save()
            with self.assertRaises(ValueError, msg='missing ' + key):
                aggregate([self.path], SHA)

    def test_token_identity_and_call_mismatch_across_layers(self):
        for key, value in [('token_id', 99), ('seq_ids', [1]), ('phase', 'decode'),
                           ('decode_call_id', 4)]:
            with self.subTest(key=key):
                self.data = fixture()
                self.data[-1][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    aggregate([self.path], SHA)

    def test_decode_call_cannot_change_between_tokens_inside_one_batch(self):
        self.data[2]['decode_call_id'] = 4
        self.save()
        with self.assertRaisesRegex(ValueError, 'within routing batch'):
            aggregate([self.path], SHA)

    def test_one_call_can_have_many_ubatches_and_calls_can_have_gaps(self):
        for batch_id, call_id in [(8, 3), (12, 6), (13, 6)]:
            extra = copy.deepcopy(self.data[1:81])
            for row in extra:
                row['batch_id'], row['decode_call_id'] = batch_id, call_id
            self.data.extend(extra)
        self.save()
        _, counts, _ = aggregate([self.path], SHA)
        self.assertEqual(counts, [8] * 40)
        # Even with batch IDs increasing, decode call IDs may never regress.
        for row in self.data[-80:]:
            row['decode_call_id'] = 5
        self.save()
        with self.assertRaisesRegex(ValueError, 'Non-monotonic decode call'):
            aggregate([self.path], SHA)

    def test_dense_token_indices_not_just_equal_layer_coverage(self):
        for mapping in ({0: 1, 1: 2}, {0: 0, 1: 2}):
            self.data = fixture()
            for row in self.data[1:]:
                row['token_index'] = mapping[row['token_index']]
            self.save()
            with self.assertRaisesRegex(ValueError, 'dense from zero'):
                aggregate([self.path], SHA)

    def test_no_unjustified_vocabulary_or_complete_stream_claim(self):
        # Header has no vocabulary size; actual int32 bound is all the reader can check.
        # Entire missing batches and an omitted suffix across ALL layers are unprovable.
        self.data = [self.data[0]] + [r for r in self.data[1:] if r['token_index'] == 0]
        for row in self.data[1:]:
            row['token_id'] = (1 << 31) - 1
            row['batch_id'] = 900
            row['decode_call_id'] = 70
        self.save()
        _, counts, _ = aggregate([self.path], SHA)
        self.assertEqual(counts, [1] * 40)

    def test_counters_restart_between_independent_trace_files(self):
        second = Path(self.directory.name) / 'second.jsonl'
        other = fixture()
        for row in other[1:]:
            row['batch_id'], row['decode_call_id'] = 1, 1
        second.write_text(''.join(json.dumps(r) + '\n' for r in other), encoding='utf-8')
        _, counts, provenance = aggregate([self.path, second], SHA)
        self.assertEqual(counts, [4] * 40)
        self.assertEqual(len(provenance), 2)

    def test_reject_nonobjects_duplicate_keys_and_boolean_schema(self):
        for malformed in ([], None, {'schema_version': True, 'record_type': 'route'}):
            self.data = fixture()
            self.data[1] = malformed
            self.save()
            with self.assertRaises(ValueError):
                aggregate([self.path], SHA)
        self.data = fixture()
        self.data[0]['schema_version'] = True
        self.save()
        with self.assertRaises(ValueError):
            aggregate([self.path], SHA)
        self.data = fixture()
        self.save()
        raw = self.path.read_text(encoding='utf-8')
        self.path.write_text(raw.replace('"decode_call_id": 3', '"decode_call_id": 3, "decode_call_id": 3', 1), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate trace JSON key'):
            aggregate([self.path], SHA)

    def test_error_records_and_incomplete_batches_still_rejected(self):
        self.data.append({'schema_version': 1, 'record_type': 'error', 'batch_id': 8,
                          'decode_call_id': 4, 'message': 'synthetic capture failure'})
        self.save()
        with self.assertRaises(ValueError):
            aggregate([self.path], SHA)
        self.data = fixture()[:-2]
        self.save()
        with self.assertRaisesRegex(ValueError, 'missing layers'):
            aggregate([self.path], SHA)

    def legacy_fixture_expected(self, *, keep=None, coverage=0.95, random_seed=None):
        # Independent frozen expectation of pre-refactor output on this synthetic fixture.
        # Sixteen equal nonzero masses, remaining 240 experts zero; ties by expert ID.
        rng = random.Random(random_seed)
        n = keep
        if n is None:
            n = max(8, math.ceil(coverage * 16))
        source = {'file': str(self.path), 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(), 'rows': 80}
        expected = {'schema_version': 1, 'model_sha256': SHA, 'architecture': 'qwen35moe',
                    'expert_count': 256, 'top_k': 8, 'layer_count': 40, 'layers': {},
                    'method': 'mass_gate' if random_seed is None else 'random_matched_pool',
                    'calibration_split': 'calibration', 'source_traces': [source], 'diagnostics': {},
                    'random_seed': random_seed, 'requested_keep': keep, 'requested_coverage': coverage}
        for layer in range(40):
            selected = list(range(n)) if random_seed is None else rng.sample(range(256), n)
            observed = sum(e < 16 for e in selected)
            expected['layers'][str(layer)] = sorted(selected)
            expected['diagnostics'][str(layer)] = {'tokens': 2, 'kept': n,
                'observed_experts': 16, 'retained_mass_fraction': observed / 16,
                'zero_observed_mass_kept': n - observed}
        return expected

    def test_pure_builder_preserves_legacy_serialized_output(self):
        validated = aggregate([self.path], SHA)
        for options in ({}, {'keep': 8}, {'keep': 64}, {'keep': 128}, {'keep': 256},
                        {'coverage': 0.5}, {'coverage': 1.0}, {'random_seed': 42},
                        {'keep': 64, 'random_seed': 0}, {'keep': 128, 'random_seed': 42}):
            with self.subTest(options=options):
                pure = build_from_aggregate(validated, SHA, split='calibration', **options)
                self.assertEqual(pure, self.make(**options))
                self.assertEqual(json.dumps(pure, indent=2, allow_nan=False),
                                 json.dumps(self.legacy_fixture_expected(**options), indent=2, allow_nan=False))

    def test_one_aggregate_reused_without_io_mutation_or_global_rng_changes(self):
        validated = aggregate([self.path], SHA)
        before = copy.deepcopy(validated)
        rng_before = random.getstate()
        with patch('mask_builder.aggregate', side_effect=AssertionError('no reaggregate')):
            with patch.object(Path, 'open', side_effect=AssertionError('no file reads')):
                fixed64 = build_from_aggregate(validated, SHA, split='calibration', keep=64)
                fixed128 = build_from_aggregate(validated, SHA, split='calibration', keep=128)
                control = build_from_aggregate(validated, SHA, split='calibration', keep=64, random_seed=123)
                repeat = build_from_aggregate(validated, SHA, split='calibration', keep=64, random_seed=123)
        self.assertEqual(control, repeat)
        self.assertNotEqual(fixed64['layers'], control['layers'])
        self.assertEqual([len(v) for v in fixed64['layers'].values()], [len(v) for v in control['layers'].values()])
        self.assertEqual([len(v) for v in fixed128['layers'].values()], [128] * 40)
        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(validated, before)
        fixed64['source_traces'][0]['file'] = 'changed-only-in-result'
        self.assertEqual(validated, before)
        self.assertNotEqual(fixed64['source_traces'], fixed128['source_traces'])

    def test_pure_builder_rejects_invalid_inputs_and_heldout_without_io(self):
        validated = aggregate([self.path], SHA)
        corruptions = []
        for kind in ('geometry', 'expert_count', 'negative', 'nan', 'empty', 'counts', 'provenance'):
            value = copy.deepcopy(validated)
            if kind == 'geometry': value[0].pop()
            if kind == 'expert_count': value[0][0].pop()
            if kind == 'negative': value[0][0][0] = -1
            if kind == 'nan': value[0][0][0] = float('nan')
            if kind == 'empty': value[0][0] = [0.0] * 256
            if kind == 'counts': value[1][0] += 1
            if kind == 'provenance': value[2][0]['rows'] += 1
            corruptions.append(value)
        with patch.object(Path, 'open', side_effect=AssertionError('no file I/O')):
            for value in corruptions:
                with self.assertRaises(ValueError):
                    build_from_aggregate(value, SHA, split='calibration')
            with self.assertRaises(ValueError):
                build_from_aggregate(validated, SHA, split='heldout')
            with self.assertRaises(ValueError):
                build([self.path], SHA, split='heldout')

    def rare_high_mean_fixture(self):
        header = fixture()[0]
        self.data = [header]
        for layer in range(40):
            for token in range(10):
                ids = list(range(8)) if token < 9 else list(range(7)) + [8]
                weights = [0.125] * 8 if token < 9 else [0.1 / 7] * 7 + [0.9]
                self.data.append(dict(schema_version=1, record_type='route', batch_id=1,
                    decode_call_id=1, layer=layer, phase='unknown', token_position=token,
                    token_index=token, token_id=50 + token, seq_ids=[0], ids=ids, weights=weights))
        self.save()

    def test_rare_high_mean_changes_topk_without_relabeling_mass(self):
        self.rare_high_mean_fixture()
        data = aggregate([self.path], SHA, include_selected_counts=True)
        self.assertEqual(data[3][0][7:9], [9, 1])
        mass = build_from_aggregate(data, SHA, split='calibration', keep=8)
        mean = build_from_aggregate(data, SHA, split='calibration', keep=8, ranking_method='mean_selected_gate')
        self.assertEqual(mass['layers']['0'], list(range(8)))
        self.assertEqual(mean['layers']['0'], [0, 1, 2, 3, 4, 5, 7, 8])
        self.assertEqual(mean['method'], 'mean_selected_gate')
        self.assertFalse(mean['quality_approval'])
        self.assertFalse(mean['selection_recipe']['includes_expert_output_norm'])
        kept = mean['layers']['0']
        actual_mass = math.fsum(data[0][0][e] for e in kept)
        sum_of_means = math.fsum(data[0][0][e] / data[3][0][e] for e in kept)
        diag = mean['diagnostics']['0']
        self.assertAlmostEqual(diag['retained_gate_mass'], actual_mass)
        self.assertNotAlmostEqual(diag['retained_gate_mass'], sum_of_means)
        self.assertAlmostEqual(diag['retained_mass_fraction'], actual_mass / math.fsum(data[0][0]))
        self.assertEqual(diag['selected_count_total'], 80)
        self.assertEqual(mean, self.make(keep=8, ranking_method='mean_selected_gate'))

    def test_selected_count_includes_zero_weights_unobserved_score_and_ties(self):
        for row in self.data[1:]:
            row['weights'] = [1.0] + [0.0] * 7
        self.save()
        data = aggregate([self.path], SHA, include_selected_counts=True)
        self.assertEqual(data[3][0][:16], [1] * 16)
        self.assertEqual(data[3][0][16:], [0] * 240)
        result = build_from_aggregate(data, SHA, split='calibration', keep=8, ranking_method='mean_selected_gate')
        self.assertEqual(result['layers']['0'], [0, 1, 2, 3, 4, 5, 6, 8])
        large = build_from_aggregate(data, SHA, split='calibration', keep=32, ranking_method='mean_selected_gate')
        self.assertEqual(large['diagnostics']['0']['experts_with_selected_observations'], 16)
        self.assertEqual(large['diagnostics']['0']['observed_experts'], 2)  # legacy positive-mass diagnostic
        self.assertEqual(large['diagnostics']['0']['unobserved_experts_kept'], 16)

    def test_mean_requires_actual_counts_and_rejects_corrupted_counts(self):
        old = aggregate([self.path], SHA)
        self.assertEqual(len(old), 3)
        with self.assertRaisesRegex(ValueError, 'requires actual selected counts'):
            build_from_aggregate(old, SHA, split='calibration', ranking_method='mean_selected_gate')
        data = aggregate([self.path], SHA, include_selected_counts=True)
        for value in (True, -1, 1.0, 3, 0):
            corrupt = copy.deepcopy(data)
            corrupt[3][0][0] = value
            with self.assertRaises(ValueError):
                build_from_aggregate(corrupt, SHA, split='calibration', ranking_method='mean_selected_gate')
        bad_sum = copy.deepcopy(data)
        bad_sum[3][0][16] = 1
        with self.assertRaisesRegex(ValueError, 'times top8'):
            build_from_aggregate(bad_sum, SHA, split='calibration', ranking_method='mean_selected_gate')
        with patch.object(Path, 'open', side_effect=AssertionError('must validate recipe before I/O')):
            with self.assertRaisesRegex(ValueError, 'Unknown ranking_method'):
                self.make(ranking_method='full_REAP')

    def test_new_count_aggregate_preserves_default_serialization_and_random_ids(self):
        self.rare_high_mean_fixture()
        data = aggregate([self.path], SHA, include_selected_counts=True)
        original = copy.deepcopy(data)
        for keep in (128, 96, 64, 32):
            with self.subTest(keep=keep):
                old_mass = self.make(keep=keep)
                new_mass = build_from_aggregate(data, SHA, split='calibration', keep=keep)
                explicit_mass = build_from_aggregate(data, SHA, split='calibration', keep=keep, ranking_method='mass_gate')
                self.assertEqual(json.dumps(old_mass), json.dumps(new_mass))
                self.assertEqual(json.dumps(new_mass), json.dumps(explicit_mass))
                random_mass = build_from_aggregate(data, SHA, split='calibration', keep=keep, random_seed=20260713)
                random_mean = build_from_aggregate(data, SHA, split='calibration', keep=keep, random_seed=20260713,
                                                   ranking_method='mean_selected_gate')
                self.assertEqual(random_mean['layers'], random_mass['layers'])
                self.assertEqual(random_mean['method'], 'random_matched_pool')
                self.assertEqual(random_mean['ranking_method'], 'mean_selected_gate')
                self.assertEqual(random_mean['selection_recipe_sha256'], recipe_digest(selection_recipe('mean_selected_gate')))
                self.assertEqual(random_mean['source_traces'], random_mass['source_traces'])
        self.assertEqual(data, original)

    def test_mean_dynamic_coverage_still_uses_actual_gate_mass(self):
        self.rare_high_mean_fixture()
        result = self.make(coverage=0.95, ranking_method='mean_selected_gate')
        self.assertEqual(result['diagnostics']['0']['kept'], 9)
        self.assertGreaterEqual(result['diagnostics']['0']['retained_mass_fraction'], 0.95)

    def test_unseen_experts_explicit_when_fixed_pool_large(self):
        result = self.make(keep=32)
        self.assertEqual(result['diagnostics']['0']['zero_observed_mass_kept'], 16)
        self.assertEqual(result['diagnostics']['0']['observed_experts'], 16)


if __name__ == '__main__':
    unittest.main()
