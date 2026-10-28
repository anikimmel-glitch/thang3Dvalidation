import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
import numpy as np
from studies.acquire import unpack
from studies.analysis import match
from studies.annotations import validated_contexts,shuffled_contexts,agreement,feature_report
from studies.ablations import make_input
from studies.geometry import mesh_structure

class Studies(unittest.TestCase):
    def test_matching_caliper_and_no_reuse(self):
        r=match(["a","b","c"],["x","y"],[[0,0,0],[.01,0,0],[10,0,0]],[[0,0,0],[10,0,0]])
        self.assertEqual(len(r["pairs"]),2)
        self.assertEqual(len({p["b"] for p in r["pairs"]}),2)
        self.assertEqual(len(r["unmatched_a"]),1)
        self.assertEqual(match(["a"],["x"],[[0,0,0]],[[10,0,0]])["pairs"],[])

    def test_bounded_zip_member_crc_and_identity(self):
        raw=b"ISO-10303-21;\nexample"
        name=b"a.step";compressed=zlib.compress(raw)[2:-4]
        header=struct.pack("<4s5H3I2H",b"PK\x03\x04",20,0,8,0,0,zlib.crc32(raw),len(compressed),len(raw),len(name),0)
        row=dict(member="a.step",compression=8,bytes=len(raw),compressed_bytes=len(compressed),crc32=zlib.crc32(raw))
        self.assertEqual(unpack(header+name+compressed,row),raw)
        with self.assertRaises(ValueError):unpack(header+name+compressed,{**row,"member":"wrong.step"})
        with self.assertRaises(ValueError):unpack(header+name+compressed,{**row,"crc32":0})
        with self.assertRaises(ValueError):unpack(header+name+compressed[:-1],row)

    def test_interventions_never_change_scoring_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/"source";src.mkdir()
            import trimesh
            trimesh.creation.box().export(src/"reference.stl")
            rng=np.random.default_rng(1);points=rng.normal(size=(8192,3)).astype("f4")
            np.savez_compressed(src/"reference.npz",points=points,normals=points,
                                pc256=points[:256],sharp_points=points[:20])
            for condition in ["clean","sparse_64","noise_005","partial_half","resample_1"]:
                dest=Path(temp)/condition
                make_input(src,dest,condition,123,"cad_recode_pc")
                with np.load(dest/"reference.npz") as d:
                    np.testing.assert_array_equal(d["points"],points)
                    self.assertEqual(d["pc256"].shape,(256,3))
                    if condition=="sparse_64":self.assertEqual(len(np.unique(d["pc256"],axis=0)),64)

    def test_unreviewed_context_fails_closed(self):
        with self.assertRaises(ValueError):validated_contexts({}, {}, {}, {})
        contexts={"a":"hole","b":"slot","c":"vent"}
        shuffled=shuffled_contexts(contexts)
        self.assertEqual(set(shuffled.values()),set(contexts.values()))
        self.assertTrue(all(shuffled[k]!=contexts[k] for k in contexts))
        with self.assertRaises(ValueError):shuffled_contexts(contexts,{"a":"same","b":"same","c":"other"})

    def test_annotation_metric_keeps_failed_and_unassessable(self):
        gold=dict(adjudication_complete=True,bodies=[dict(pilot_id="a",features=[dict(feature_id="hole"),dict(feature_id="slot")])])
        p=dict(adjudication_complete=True,predictions=[dict(pilot_id="a",blind_id="x",output_failed=True,feature_assessments=[])])
        r=feature_report(gold,p)["records"][0]
        self.assertEqual(r["feature_recall"],0)
        self.assertEqual(r["missing"],2)
        self.assertEqual(agreement({"x":True,"y":False},{"x":True,"y":False})["kappa"],1)

    def test_topology_not_reported_for_open_mesh(self):
        import trimesh
        box=trimesh.creation.box()
        self.assertEqual(mesh_structure(box)["boundary_genus_sum"],0)
        opened=trimesh.Trimesh(vertices=box.vertices,faces=box.faces[:-1])
        self.assertIsNone(mesh_structure(opened)["boundary_genus_sum"])

    def test_analysis_preserves_unscored_and_renders_empty_bins(self):
        from evaluation.common import save
        from studies.analysis import analyze
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rows=[]
            for i,dataset in enumerate(["Thangs","CADFS","Fusion","DeepCAD"]):
                row=dict(pilot_id=str(i),source_group="g"+str(i),dataset=dataset,
                         faces=12,complexity_bin=0,study_descriptors={})
                rows.append(row)
                # A reference-preparation exclusion must not become a model failure.
                metric=dict(pilot_id=str(i),complexity_bin=0,status="reference_preparation_failed",
                            infrastructure_failure=True)
                save(out/"benchmark/predictions/cadrille_pc"/str(i)/"metrics.json",metric)
            save(out/"matching_frozen_before_inference.json",dict(comparisons={d:dict(pairs=[]) for d in ["CADFS","Fusion","DeepCAD"]}))
            analyze(dict(files=rows,evaluation_ids=[r["pilot_id"] for r in rows],
                         bin_labels=["1-15","16-31","32-63","64-127","128+"]),out)
            result=json.loads((out/"analysis/comparisons.json").read_text())
            self.assertEqual(result["cadrille_pc"]["datasets"]["Thangs"]["scored"],0)
            self.assertIsNone(result["cadrille_pc"]["datasets"]["Thangs"]["validity_rate"])
            self.assertTrue((out/"analysis/cross_dataset_complexity.pdf").exists())

class CADStudyKernel(unittest.TestCase):
    def test_patch_diagnostics_identical_prediction(self):
        try:import cadquery as cq
        except ImportError:self.skipTest("Worker CadQuery environment required")
        from studies.geometry import reference,compare
        from evaluation.geometry import prepare
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"part.step"
            cq.exporters.export(cq.Workplane("XY").box(2,3,4).faces(">Z").workplane().hole(.7),str(source))
            row=dict(file="part.step",pilot_id="x",bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seed=4,study_descriptors={"nonplanar_fraction":.2})
            prepared=root/"prepared";prepare(row,root,prepared)
            reference(row,source,prepared,root/"diagnostics")
            compare(prepared,root/"diagnostics",prepared/"reference.stl",root/"score.json")
            r=json.loads((root/"score.json").read_text())
            self.assertGreater(r["patch_mean_recall"],.9)
            self.assertEqual(r["boundary_genus_absolute_error"],0)
