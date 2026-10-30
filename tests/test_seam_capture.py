import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from diagnostics.seam_capture import SeamCapture, StateCaptureError


class SeamCaptureTests(unittest.TestCase):
    def fixture(self, folder):
        path = Path(folder) / "utils.py"
        path.write_text(
            "def create_trimmed_face_from_wire(face_fixer):\n"
            "    try:\n"
            "        face_fixer.FixMissingSeam()\n"
            "        return 17\n"
            "    except Exception:\n"
            "        return None\n")
        spec = importlib.util.spec_from_file_location("fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return path, module

    def test_operation_preserved_and_last_state_matches_last_call(self):
        with tempfile.TemporaryDirectory() as directory:
            source, module = self.fixture(directory)
            out = Path(directory) / "state"
            calls = []
            class Fixer:
                def FixMissingSeam(self):
                    calls.append("seam")
            def capture(frame, target):
                calls.append("capture")
                return {"wires": 2, "fixture": True}
            with SeamCapture(source, out, capture):
                self.assertEqual(module.create_trimmed_face_from_wire(Fixer()), 17)
                self.assertEqual(module.create_trimmed_face_from_wire(Fixer()), 17)
            self.assertEqual(calls, ["capture", "seam", "capture", "seam"])
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["stage"] for e in events], [
                "capturing_state", "before_native_operation", "native_operation_returned"] * 2)
            self.assertEqual(json.loads((out / "state_latest.json").read_text())["sequence"], 2)

    def test_capture_error_cannot_be_misclassified_as_cad_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source, module = self.fixture(directory)
            calls = []
            class Fixer:
                def FixMissingSeam(self):
                    calls.append("seam")
            def broken(*args):
                raise ValueError("observer export failed")
            out = Path(directory) / "state"
            with self.assertRaises(StateCaptureError):
                with SeamCapture(source, out, broken):
                    module.create_trimmed_face_from_wire(Fixer())
            self.assertEqual(calls, [])
            self.assertEqual(json.loads((out / "status.json").read_text())["stage"],
                             "state_capture_failed")

    def test_cad_exception_is_recorded_and_original_handler_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, module = self.fixture(directory)
            class Fixer:
                def FixMissingSeam(self):
                    raise RuntimeError("CAD exception")
            out = Path(directory) / "state"
            with SeamCapture(source, out, lambda *args: {"fixture": True}):
                self.assertIsNone(module.create_trimmed_face_from_wire(Fixer()))
            self.assertEqual(json.loads((out / "status.json").read_text())["stage"],
                             "native_operation_raised")

    def test_unexpected_source_is_rejected_before_installing_tracer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utils.py"
            path.write_text("def create_trimmed_face_from_wire(): pass\n")
            with self.assertRaises(ValueError):
                SeamCapture(path, Path(directory) / "state")


if __name__ == "__main__":
    unittest.main()
