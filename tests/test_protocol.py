import copy
import json
from pathlib import Path
import unittest
import numpy as np
from evaluation.aggregate import summarize
from evaluation.geometry import fps, fscore, vetted_program
from evaluation.run import validate_manifest


class Protocol(unittest.TestCase):
    def test_manifest_groups_are_distinct(self):
        path=Path(__file__).resolve().parents[1]/"manifests/pilot.json"
        if not path.exists():
            self.skipTest("Private frozen manifest supplied separately to the job")
        m=json.loads(path.read_text())
        rows=validate_manifest(m)
        self.assertEqual(len(rows),500)
        self.assertEqual(sum(r["role"]=="evaluation" for r in rows),450)
        bad=copy.deepcopy(m)
        bad["files"][1]["source_group"]=bad["files"][0]["source_group"]
        with self.assertRaises(ValueError):
            validate_manifest(bad)

    def test_manifest_rejects_path_escape(self):
        m={"files":[dict(file="../escape.step",pilot_id="p1",source_group="g1",
                        sha256="a",role="evaluation")]}
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_missing_and_infrastructure_do_not_inflate_validity(self):
        rows=[{"pilot_id":str(i)} for i in range(5)]
        scores={k:{"surface_f1_at_1pct_side":.8,"squared_chamfer_sum":.001}
                for k in ("native_frame","independent_bbox_normalization")}
        records={"0":dict(status="evaluated",valid_single_solid=True,scores=scores),
                 "1":dict(status="execution_failed",infrastructure_failure=False),
                 "2":dict(status="render_failed",infrastructure_failure=True)}
        s=summarize(rows,records)
        self.assertEqual(s["validity_rate"],.5)
        self.assertEqual(s["missing"],2)
        self.assertFalse(s["complete_protocol_coverage"])
        self.assertEqual(s["scores"]["native_frame"]["surface_f1_mean"],.4)

    def test_correspondence_and_missing_edges(self):
        pts=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        self.assertEqual(fscore(pts,pts,.02)["f1"],1.)
        self.assertEqual(fscore(pts+10,pts,.02)["f1"],0.)
        self.assertEqual(fscore(np.empty((0,3)),pts,.02)["f1"],0.)
        np.testing.assert_array_equal(fps(pts,3),pts)

    def test_program_rejects_file_io_and_preserves_cad(self):
        vetted_program('import cadquery as cq\nr=cq.Workplane("XY").box(2,3,4)')
        for code in ('import os','open("secret")','x.__class__','cq.exporters.export(r,"x")'):
            with self.assertRaises(ValueError):
                vetted_program(code)


if __name__=="__main__":
    unittest.main()
