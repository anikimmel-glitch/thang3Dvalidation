import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from diagnostics.native_bootstrap import focus_guard_failure


ROOT = Path(__file__).resolve().parents[1]


class FocusedGuardTests(unittest.TestCase):
    def test_subset_keeps_exact_original_candidate_data_and_seed(self):
        original = json.loads((ROOT / "diagnostics/native_plan.json").read_text())
        selected = focus_guard_failure(copy.deepcopy(original))
        subject = next(r for r in original["subjects"] if r["part_id"] == "f00122")
        self.assertEqual(len(selected["subjects"]), 1)
        current = selected["subjects"][0]
        for key in subject:
            if key != "suspected_next_candidate":
                self.assertEqual(current[key], subject[key])
        self.assertEqual(current["suspected_next_candidate"], "f00122_17")
        self.assertEqual(subject["suspected_next_candidate"], "f00122_19")
        self.assertTrue(selected["fatal_only_traceback"])

    def test_fatal_signal_has_dedicated_trace_and_preserved_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            code = (
                "import os,signal,resource\nfrom pathlib import Path\n"
                "from diagnostics.traceback_output import traceback_output\n"
                "resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
                "with traceback_output(Path(__import__('sys').argv[1]),True):\n"
                "    os.kill(os.getpid(),signal.SIGSEGV)\n")
            proc = subprocess.run([sys.executable, "-c", code, folder], cwd=ROOT,
                                  capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, -signal.SIGSEGV)
            trace = (Path(folder) / "fatal_traceback.log").read_text()
            self.assertIn("Fatal Python error", trace)
            self.assertIn("Segmentation fault", trace)
            self.assertNotIn("Fatal Python error", proc.stderr)
            policy = json.loads((Path(folder) / "traceback_policy.json").read_text())
            self.assertFalse(policy["periodic_traceback_enabled"])
            self.assertTrue(policy["process_timeout_enforced_by_parent"])

    def test_normal_execution_closes_fatal_handler_cleanly(self):
        from diagnostics.traceback_output import traceback_output
        import faulthandler
        with tempfile.TemporaryDirectory() as folder:
            with traceback_output(Path(folder), True):
                self.assertTrue(faulthandler.is_enabled())
            self.assertFalse(faulthandler.is_enabled())
            self.assertEqual((Path(folder) / "fatal_traceback.log").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
