"""Coordinator preflight tests: stop BEFORE tasklist/process/file operations."""
import unittest
from unittest.mock import patch
import run_server_pilot as runner


class Validated(RuntimeError):
    pass


class ServerCoordinatorTests(unittest.TestCase):
    def test_selected_ids_use_runner_string_contract(self):
        argv = ['script', '--allow-inference', '--out', 'not-created', '--episode-ids',
                'calibration-semantic_selector-0,calibration-currency-0', '--policy-version', 'v2']
        with patch('sys.argv', argv), patch.object(runner.subprocess, 'run', side_effect=Validated('validated')) as process:
            with self.assertRaises(Validated):
                runner.main()
            process.assert_called_once()

    def test_heldout_ids_rejected_before_process_check(self):
        argv = ['script', '--allow-inference', '--out', 'not-created', '--episode-ids', 'heldout-semantic_selector-0']
        with patch('sys.argv', argv), patch.object(runner.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                runner.main()
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
