import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluation.geometry import cq, canonical, prepare, evaluate


@unittest.skipIf(cq is None,"CadQuery unavailable locally; required in remote bootstrap")
class Kernel(unittest.TestCase):
    def test_roundtrip_and_translation_error(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            shape=cq.Workplane("XY").box(4,6,8).faces(">Z").workplane().hole(1).val()
            shape=shape.translate((3,7,-10))
            norm,center,scale=canonical(shape)
            bb=norm.BoundingBox()
            self.assertAlmostEqual(max(bb.xlen,bb.ylen,bb.zlen),2,places=6)
            np.testing.assert_allclose((bb.center.x,bb.center.y,bb.center.z),0,atol=1e-6)
            cq.exporters.export(shape,str(p/"reference.step"))
            data=(p/"reference.step").read_bytes()
            row=dict(file="reference.step",bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                seed=1,pilot_id="fixture",role="development",complexity_bin=0,faces=len(shape.Faces()))
            prepare(row,p,p/"prepared")
            cq.exporters.export(norm.scale(100),str(p/"prediction.step"))
            evaluate(row,p/"prepared",p/"prediction.step",p/"scores.json",.01)
            result=json.loads((p/"scores.json").read_text())
            self.assertTrue(result["valid_single_solid"])
            self.assertGreater(result["scores"]["native_frame"]["surface_f1_at_1pct_side"],.9)
            cq.exporters.export(norm.translate((3,0,0)).scale(100),str(p/"prediction.step"))
            evaluate(row,p/"prepared",p/"prediction.step",p/"scores.json",.01)
            result=json.loads((p/"scores.json").read_text())
            self.assertLess(result["scores"]["native_frame"]["surface_f1_at_1pct_side"],.01)
            self.assertGreater(result["scores"]["independent_bbox_normalization"]["surface_f1_at_1pct_side"],.9)
