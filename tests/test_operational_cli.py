"""The documented standalone script must bootstrap src without PYTHONPATH."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class OperationalCLITests(unittest.TestCase):
    def test_standalone_predictor_without_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        code = "import runpy; d=runpy.run_path('tools/operational_workflow.py'); assert len(d['prophet_forecast']([1.]*12,1))==1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
