import copy
import json
from pathlib import Path
import tempfile
import unittest

from mask_builder import aggregate, build

SHA = 'a' * 64


def fixture():
    header = {'schema_version': 1, 'record_type': 'header', 'architecture': 'qwen35moe',
              'model_sha256': SHA, 'layer_count': 40, 'expert_count': 256, 'top_k': 8}
    rows = []
    for layer in range(40):
        for token in range(2):
            rows.append({'schema_version': 1, 'record_type': 'route', 'batch_id': 7,
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

    def test_unseen_experts_explicit_when_fixed_pool_large(self):
        result = self.make(keep=32)
        self.assertEqual(result['diagnostics']['0']['zero_observed_mass_kept'], 16)
        self.assertEqual(result['diagnostics']['0']['observed_experts'], 16)


if __name__ == '__main__':
    unittest.main()
