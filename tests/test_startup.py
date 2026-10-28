"""Regression for root-writable /tmp that is inaccessible to APT's sandbox user."""
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile
import unittest
from evaluation.startup import temp_access


@unittest.skipUnless(sys.platform == "linux" and os.geteuid() == 0,
                     "Requires the disposable Linux worker and uid switching")
class AptTemporaryAccess(unittest.TestCase):
    def test_root_writable_directory_must_also_allow_apt(self):
        user = pwd.getpwnam("_apt")
        with tempfile.TemporaryDirectory(prefix="apt-permission-regression-") as folder:
            p = Path(folder)
            p.chmod(0o755)
            probe = [sys.executable, "-c",
                     "import tempfile,sys; tempfile.TemporaryFile(dir=sys.argv[1]).close()",
                     str(p)]
            kwargs = dict(user=user.pw_uid, group=user.pw_gid, extra_groups=[],
                          env={"PATH": "/usr/bin:/bin"}, capture_output=True, timeout=10)
            self.assertNotEqual(subprocess.run(probe, **kwargs).returncode, 0)
            temp_access(p)
            self.assertEqual(subprocess.run(probe, **kwargs).returncode, 0)
