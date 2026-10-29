import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from diagnostics.native_bootstrap import acquire, process


class NativeDiagnosticTests(unittest.TestCase):
    def test_worker_preserves_upstream_serial_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            repo = p / "repo"
            repo.mkdir()
            work = p / "input"
            candidate = work / "tmp" / "f00012_13" / "post.npz"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"fixture")
            (repo / "numpy.py").write_text("class random:\n    seed = staticmethod(lambda n: None)\n")
            (repo / "torch.py").write_text("def manual_seed(n): pass\n")
            (repo / "postprocess.py").write_text(
                "import json, sys\nfrom pathlib import Path\n"
                "def build_one(path, here, drop, cuda, optimize, iterations, seed):\n"
                "    assert (drop,cuda,optimize,iterations,seed)==(0,False,True,200,0)\n"
                "    return (Path(path).parent.name,False,None,0.01)\n"
                "def main():\n"
                "    assert sys.argv[-1]=='--serial'\n"
                "    root=Path(sys.argv[2]); paths=sorted(root.glob('tmp/*/post.npz'))\n"
                "    assert len(paths)==1\n"
                "    result=build_one(paths[0],str(Path(__file__).parent),0,False,True,200,seed=0)\n"
                "    (root/'fixture_result.json').write_text(json.dumps(result))\n")
            env = dict(os.environ, PYTHONPATH=str(repo))
            worker = Path(__file__).resolve().parents[1] / "diagnostics/native_worker.py"
            result = subprocess.run([sys.executable, worker, "--repository", repo,
                                     "--input", work, "--seed", "20271040"],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((work/"fixture_result.json").read_text())[0], "f00012_13")
            self.assertIn('"event": "candidate_started"', result.stdout)
            self.assertIn('"event": "candidate_finished"', result.stdout)

    def test_nonzero_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            result = process([sys.executable, "-c", "raise SystemExit(7)"], p, p/"log", {},
                             5, p/"execution.json")
            self.assertEqual(result["returncode"], 7)
            self.assertFalse(result["timeout"])
            self.assertIsNone(result["signal"])
            self.assertEqual(json.loads((p/"execution.json").read_text()), result)

    def test_timeout_records_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            result = process([sys.executable, "-c", "import time; time.sleep(10)"], p, p/"log", {},
                             .1, p/"execution.json")
            self.assertTrue(result["timeout"])
            self.assertEqual(result["signal"], "SIGKILL")
            self.assertEqual(result["returncode"], -9)

    def test_candidate_transfer_enforces_size(self):
        from datetime import datetime, timezone
        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        class Client:
            def head_object(self, **kwargs):
                return dict(ContentLength=3, LastModified=stamp, ETag='"observed"')
            def get_object(self, **kwargs):
                self.match = kwargs["IfMatch"]
                return dict(Body=io.BytesIO(b"too many bytes"))
        hit = dict(bucket="bucket", key="post.npz", size=3,
                   last_modified=stamp.isoformat(), s3_uri="s3://bucket/post.npz")
        with tempfile.TemporaryDirectory() as folder:
            client = Client()
            with self.assertRaisesRegex(RuntimeError, "exceeds observed size"):
                acquire(client, hit, Path(folder)/"post.npz")
            self.assertEqual(client.match, '"observed"')


if __name__ == "__main__":
    unittest.main()
