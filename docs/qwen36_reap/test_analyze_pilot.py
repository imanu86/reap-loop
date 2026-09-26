import json
from pathlib import Path
import tempfile
import unittest

from analyze_pilot import analyze, complete_records


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'transcripts.jsonl'

    def test_snapshot_ignores_only_uncommitted_tail(self):
        self.path.write_bytes(b'{"ok":1}\n{"text":"\xe2\x82')
        self.assertEqual(list(complete_records(self.path, True)), [{'ok': 1}])
        with self.assertRaises(UnicodeDecodeError):
            list(complete_records(self.path))

    def test_complete_corruption_is_not_silently_skipped(self):
        self.path.write_bytes(b'{"bad":}\n')
        with self.assertRaises(json.JSONDecodeError):
            list(complete_records(self.path, True))

    def test_aggregates_actual_outcomes(self):
        rows = [dict(id=str(i), family='f', split='calibration', full_completion=i == 0,
                     error_class=None if i == 0 else 'turn_limit', private_error=None,
                     wall_seconds=2, turns=[]) for i in range(2)]
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        r = analyze(self.path)
        self.assertEqual(r['count'], 2)
        self.assertEqual(r['successes'], 1)
        self.assertEqual(r['families']['f'], {'cases': 2, 'successes': 1})
        self.assertEqual(r['total_wall_seconds'], 4)
        self.assertFalse(r['snapshot_not_final'])
        self.assertTrue(r['not_a_throughput_benchmark'])


if __name__ == '__main__':
    unittest.main()
