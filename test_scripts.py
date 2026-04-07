import unittest
from unittest.mock import patch, MagicMock
import sys
import runpy

# Mock pandas since it is missing in the environment
sys.modules['pandas'] = MagicMock()

class TestScripts(unittest.TestCase):
    @patch('subprocess.run')
    def test_seed_robustness(self, mock_run):
        runpy.run_path('scripts/seed_robustness.py')

        # Check if subprocess.run was called at least once
        self.assertTrue(mock_run.called)

        # Check that none of the calls used shell=True
        for call in mock_run.call_args_list:
            args, kwargs = call
            self.assertFalse(kwargs.get('shell', False), "shell=True was used in subprocess.run")
            self.assertIsInstance(args[0], list, "subprocess.run argument should be a list")

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('shutil.copy')
    def test_verify_ic_loss(self, mock_copy, mock_exists, mock_run):
        mock_exists.return_value = True

        runpy.run_path('scripts/verify_ic_loss.py')

        self.assertTrue(mock_run.called)
        for call in mock_run.call_args_list:
            args, kwargs = call
            self.assertFalse(kwargs.get('shell', False), "shell=True was used in subprocess.run")
            self.assertIsInstance(args[0], list, "subprocess.run argument should be a list")

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_model_comparison_ic(self, mock_exists, mock_run):
        mock_exists.return_value = True

        runpy.run_path('scripts/model_comparison_ic.py')

        self.assertTrue(mock_run.called)
        for call in mock_run.call_args_list:
            args, kwargs = call
            self.assertFalse(kwargs.get('shell', False), "shell=True was used in subprocess.run")
            self.assertIsInstance(args[0], list, "subprocess.run argument should be a list")

if __name__ == '__main__':
    unittest.main()
