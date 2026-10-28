import json
from pathlib import Path
import tempfile
import unittest
from evaluation.retention import index,source_snapshot


class RetentionChecks(unittest.TestCase):
    def test_preserves_failure_metrics_and_prediction_files(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            (out/"predictions/m/p1").mkdir(parents=True)
            metric=out/"predictions/m/p1/metrics.json"
            raw='{"status":"invalid_geometry","pilot_id":"p1"}'
            metric.write_text(raw)
            (out/"predictions/m/p1/prediction.step").write_bytes(b"CAD geometry")
            (out/"unfinished.tmp").write_text("partial")
            index(out)
            self.assertEqual(metric.read_text(),raw)
            data=json.loads((out/"retention/artifact_index.json").read_text())
            self.assertEqual(data["file_count"],2)
            self.assertEqual(json.loads((out/"retention/metric_inventory.json").read_text())["statuses"],
                             {"invalid_geometry":1})
            index(out)
            self.assertEqual(json.loads((out/"retention/artifact_index.json").read_text())["file_count"],2)

    def test_snapshot_excludes_credentials_and_weight_blobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"source";out=Path(d)/"out";root.mkdir()
            (root/"code.py").write_text("x=1")
            (root/".env").write_text("SECRET=secret")
            (root/"weights.pt").write_bytes(b"not source")
            source_snapshot(root,out)
            files=json.loads((out/"retention/source_manifest.json").read_text())
            self.assertEqual([x["path"] for x in files],["code.py"])
