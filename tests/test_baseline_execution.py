import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

from baselines.run import child


class BaselineExecutionTests(unittest.TestCase):
    def test_exit_signal_and_timeout_are_retained_without_changing_status(self):
        cases = [
            ("pass", 5, "ok", 0, None, False),
            ("raise SystemExit(7)", 5, "failed", 7, None, False),
            ("import os,signal,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
             "os.kill(os.getpid(),signal.SIGTERM)", 5, "failed", -15, "SIGTERM", False),
            ("import time; time.sleep(10)", .1, "timeout", -signal.SIGKILL, "SIGKILL", True)]
        for code, timeout, status, returncode, sig, timed_out in cases:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                log = root/"stage_2.log"
                self.assertEqual(child([sys.executable,"-c",code],root,log,os.environ.copy(),timeout),
                                 status)
                record = json.loads(log.with_suffix(".execution.json").read_text())
                self.assertEqual(record["returncode"],returncode)
                self.assertEqual(record["signal"],sig)
                self.assertEqual(record["timeout"],timed_out)
                self.assertEqual(record["timeout_seconds"],timeout)
