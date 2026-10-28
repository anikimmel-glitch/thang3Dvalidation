import copy
import json
from pathlib import Path
import tempfile
import unittest
from evaluation.manifest import validate_manifest
from evaluation.aggregate import aggregate


class FullManifest(unittest.TestCase):
    def manifest(self):
        return {"cohort_mode": "all_curated_bodies", "files": [
            dict(pilot_id="f1", file="one.step", sha256="a", source_group="g1", role="evaluation"),
            dict(pilot_id="f2", file="two.step", sha256="b", source_group="g1", role="evaluation")]}

    def test_full_cohort_keeps_siblings_together(self):
        m = self.manifest()
        self.assertEqual(len(validate_manifest(m)), 2)
        m["files"][1]["role"] = "development"
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_pilot_still_requires_one_body_per_group(self):
        m = self.manifest()
        del m["cohort_mode"]
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_full_cohort_rejects_duplicate_content_and_files(self):
        for key in ("sha256", "pilot_id", "file"):
            m = copy.deepcopy(self.manifest())
            m["files"][1][key] = m["files"][0][key]
            with self.assertRaises(ValueError):
                validate_manifest(m)

    def test_unknown_mode_rejected(self):
        m = self.manifest()
        m["cohort_mode"] = "typo"
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_source_macro_does_not_overweight_large_assemblies(self):
        rows = [dict(pilot_id=str(i), source_group="a" if i < 2 else "b",
                     role="evaluation", complexity_bin=0) for i in range(3)]
        scores = {mode: dict(surface_f1_at_1pct_side=1., squared_chamfer_sum=0.)
                  for mode in ("native_frame", "independent_bbox_normalization")}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for i in range(3):
                folder = out / "predictions" / "model" / str(i)
                folder.mkdir(parents=True)
                record = dict(status="evaluated" if i < 2 else "invalid_geometry",
                              valid_single_solid=i < 2, complexity_bin=0, scores=scores)
                (folder / "metrics.json").write_text(json.dumps(record))
            result = aggregate(dict(files=rows, bin_labels=["all"],
                                    cohort_mode="all_curated_bodies"), out)
            evaluation = result["model"]["evaluation"]
            self.assertAlmostEqual(evaluation["validity_rate"], 2 / 3)
            self.assertEqual(evaluation["source_macro"]["validity_rate"], .5)
            self.assertEqual(evaluation["source_macro"]["surface_f1_mean"]["native_frame"], .5)
            self.assertEqual(evaluation["source_macro"]["complete_sources"], 2)
