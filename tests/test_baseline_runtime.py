import unittest
from unittest.mock import patch

from baselines.runtime import check_runtime


class BaselineRuntime(unittest.TestCase):
    def test_accepts_minimum_and_newer_glibc(self):
        for version in ("2.35", "2.35.1", "2.39", "2.100", "3.0"):
            self.assertEqual(check_runtime(("glibc", version))["version"], version)

    def test_rejects_ubuntu20_and_below_pymeshlab_minimum(self):
        for version in ("2.31", "2.32", "2.34"):
            with self.assertRaisesRegex(RuntimeError, "PyMeshLab 2025.7"):
                check_runtime(("glibc", version))

    def test_does_not_treat_unknown_or_other_libc_as_compatible(self):
        for value in (("", ""), ("musl", "2.40"), ("glibc", ""),
                      ("glibc", "2"), ("glibc", "not-known")):
            with self.assertRaisesRegex(RuntimeError, "glibc >= 2.35"):
                check_runtime(value)

    def test_detects_host_when_not_overridden(self):
        with patch("baselines.runtime.platform.libc_ver",
                   return_value=("glibc", "2.31")):
            with self.assertRaisesRegex(RuntimeError, "isolated Python"):
                check_runtime()
