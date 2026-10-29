import tempfile
from pathlib import Path
import unittest

import numpy as np
import trimesh

from baselines.pointcloud_input import FIELDS, write_oriented_cloud
from baselines.runtime import select_methods


class PointCloudSerialization(unittest.TestCase):
    def test_binary_cloud_preserves_ascii_values_and_scalar_normal_fields(self):
        rng = np.random.default_rng(1729)
        xyz = rng.normal(size=(32768, 3))
        normals = rng.normal(size=xyz.shape)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as tmp:
            ascii_path, binary_path = Path(tmp) / "old.ply", Path(tmp) / "new.ply"
            with ascii_path.open("w") as stream:
                stream.write("ply\nformat ascii 1.0\nelement vertex 32768\n"
                             + "".join(f"property float {field}\n" for field in FIELDS)
                             + "end_header\n")
                np.savetxt(stream, np.column_stack((xyz, normals)), fmt="%.9g")
            prior = trimesh.load(ascii_path, process=False)
            raw_prior = prior.metadata["_ply_raw"]["vertex"]["data"]
            prior_values = np.column_stack([np.asarray(raw_prior[f]).reshape(-1) for f in FIELDS])
            receipt = write_oriented_cloud(binary_path, xyz, normals)
            cloud = trimesh.load(binary_path, process=False)
            raw = cloud.metadata["_ply_raw"]["vertex"]["data"]
            # Same operation as the released DualBrep loader: must be N-by-3.
            read_normals = np.stack([raw["nx"], raw["ny"], raw["nz"]], axis=1)
            self.assertEqual(read_normals.shape, (32768, 3))
            self.assertEqual(np.concatenate((cloud.vertices, read_normals), axis=1).shape, (32768, 6))
            np.testing.assert_array_equal(np.stack([raw[f] for f in FIELDS], axis=1), prior_values)
            self.assertEqual(receipt["points"], 32768)
            self.assertEqual(receipt["bytes"], binary_path.stat().st_size)

    def test_rejects_bad_normals_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.ply"
            for normals in (np.zeros((2, 3)), np.ones((2, 1, 3)),
                            np.full((2, 3), np.nan), np.ones((3, 3))):
                with self.assertRaises(ValueError):
                    write_oriented_cloud(path, np.ones((2, 3)), normals)
                self.assertFalse(path.exists())


class TargetedMethodSelection(unittest.TestCase):
    def test_default_keeps_both_and_explicit_retry_selects_only_repaired_method(self):
        available = {"cadfit_mesh": {}, "dualbrep_pc": {}}
        self.assertEqual(select_methods(None, available), list(available))
        self.assertEqual(select_methods(["dualbrep_pc"], available), ["dualbrep_pc"])
        self.assertEqual(len(available), 2)

    def test_rejects_empty_unknown_or_duplicate_methods(self):
        for requested in ([], ["unknown"], ["dualbrep_pc", "dualbrep_pc"]):
            with self.assertRaises(ValueError):
                select_methods(requested, {"cadfit_mesh": {}, "dualbrep_pc": {}})
