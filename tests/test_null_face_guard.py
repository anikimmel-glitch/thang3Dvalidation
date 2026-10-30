import contextlib
import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from diagnostics import null_face_guard as guard


class NullFaceGuardTests(unittest.TestCase):
    def source(self):
        return (
            "def build(face_fixer):\n"
            "    try:\n"
            "        face_fixer.FixOrientation()\n"
            "        face_fixer.FixMissingSeam()\n"
            "        face_fixer.FixIntersectingWires()\n"
            "        return 'unchanged result'\n"
            "    except Exception:\n"
            "        return None\n"
        ).encode()

    def function(self, newline=b"\n"):
        raw = self.source().replace(b"\n", newline)
        with patch.object(guard, "EXPECTED_SHA256", hashlib.sha256(raw).hexdigest()):
            result = guard.guarded_source(raw)
        self.assertEqual(result.replace(guard.GUARD.encode().replace(b"\n", newline), b"", 1), raw)
        scope = {}
        exec(compile(result, "fixture", "exec"), scope)
        return scope["build"]

    def test_null_and_none_face_return_failure_before_unsafe_native_call(self):
        from types import SimpleNamespace
        for face in (None, SimpleNamespace(IsNull=lambda: True)):
            calls = []
            fixer = SimpleNamespace(
                FixOrientation=lambda: calls.append("orientation"), Face=lambda: face,
                FixMissingSeam=lambda: self.fail("Null face reached native seam call"),
                FixIntersectingWires=lambda: self.fail("Null face reached later native call"))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertIsNone(self.function()(fixer))
            self.assertEqual(calls, ["orientation"])
            self.assertIn(guard.MARKER, stream.getvalue())

    def test_nonnull_face_preserves_original_call_sequence_and_result(self):
        from types import SimpleNamespace
        calls = []
        fixer = SimpleNamespace(
            FixOrientation=lambda: calls.append("orientation"),
            Face=lambda: SimpleNamespace(IsNull=lambda: False),
            FixMissingSeam=lambda: calls.append("seam"),
            FixIntersectingWires=lambda: calls.append("intersections"))
        for newline in (b"\n", b"\r\n"):
            calls.clear()
            self.assertEqual(self.function(newline)(fixer), "unchanged result")
            self.assertEqual(calls, ["orientation", "seam", "intersections"])

    def test_unexpected_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected upstream"):
            guard.guarded_source(b"unrecognized source")


if __name__ == "__main__":
    unittest.main()
