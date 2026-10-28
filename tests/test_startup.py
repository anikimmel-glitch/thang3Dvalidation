"""Regression for root-writable /tmp that is inaccessible to APT's sandbox user."""
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile
import unittest
from evaluation.startup import runtime_environment, temp_access


class RuntimeLibraryIsolation(unittest.TestCase):
    def test_older_host_library_does_not_precede_locked_nvjitlink(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder)
            lib = prefix / "lib/python3.11/site-packages/nvidia/nvjitlink/lib"
            lib.mkdir(parents=True)
            (lib / "libnvJitLink.so.12").touch()
            inherited = {"LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64:/usr/local/nvidia/lib64",
                         "OTHER_SETTING": "preserve"}
            env, info = runtime_environment(prefix, inherited)
            self.assertEqual(env["LD_LIBRARY_PATH"].split(":")[0], str(lib))
            self.assertIn("/usr/local/nvidia/lib64", env["LD_LIBRARY_PATH"].split(":"))
            self.assertEqual(env["OTHER_SETTING"], "preserve")
            self.assertEqual(inherited["LD_LIBRARY_PATH"],
                             "/usr/local/cuda-12.1/lib64:/usr/local/nvidia/lib64")

    def test_missing_locked_library_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "nvJitLink library missing"):
                runtime_environment(folder, {})


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
