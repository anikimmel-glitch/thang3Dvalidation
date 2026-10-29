"""Scalar binary PLY fields compatible with DualBrep's stored-normal reader."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh


FIELDS = ("x", "y", "z", "nx", "ny", "nz")


def write_oriented_cloud(path, xyz, normals):
    xyz, normals = np.asarray(xyz), np.asarray(normals)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape or not len(xyz):
        raise ValueError("Expected matching nonempty N-by-3 points and normals")
    data = np.column_stack((xyz, normals))
    if not np.isfinite(data).all() or np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("Expected finite points and finite nonzero normals")
    # Preserve the former ASCII writer's nine-significant-digit scalar float
    # values. Binary scalar properties load as (N,), whereas ASCII properties
    # load as (N, 1) in the pinned upstream trimesh reader.
    stored = np.char.mod("%.9g", data).astype("<f4")
    if not np.isfinite(stored).all() or np.any(np.linalg.norm(stored[:, 3:], axis=1) == 0):
        raise ValueError("Point-cloud values are not representable as finite float32")
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(stored)}\n"
        + "".join(f"property float {field}\n" for field in FIELDS)
        + "end_header\n"
    ).encode("ascii")
    payload = header + stored.tobytes(order="C")
    Path(path).write_bytes(payload)
    cloud = trimesh.load(path, process=False)
    if not isinstance(cloud, trimesh.points.PointCloud):
        raise ValueError("Serialized input must remain a point cloud")
    raw = cloud.metadata["_ply_raw"]["vertex"]["data"]
    if any(np.asarray(raw[field]).shape != (len(stored),) for field in FIELDS):
        raise ValueError("PLY scalar properties have an incompatible dimension")
    decoded = np.stack([raw[field] for field in FIELDS], axis=1)
    np.testing.assert_array_equal(decoded, stored)
    return dict(
        encoding="binary_little_endian", scalar_type="float32",
        points=len(stored), properties=list(FIELDS), bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        prior_ascii_scalar_values_preserved=True,
    )


def check_upstream(repo, folder):
    """Exercise the pinned dataset reader before downloading model weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    xyz = np.array([[-1., -.4, -.2], [1., .3, .6], [.2, .9, -.8], [-.6, -.7, .4]])
    normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    receipt = write_oriented_cloud(folder / "canary.ply", xyz, normals)
    sys.path.insert(0, str(Path(repo).resolve()))
    from dataset import PointCloudDataset
    dataset = PointCloudDataset(dict(data_root=str(folder), name_list=None, ext=".ply",
                                    clip_value=.015, per_axis_norm=False, id_aug=3))
    sample = dataset[0]["sample_points_surfaces"]
    stored = np.char.mod("%.9g", np.column_stack((xyz, normals))).astype(np.float32)
    center = (stored[:, :3].max(axis=0) + stored[:, :3].min(axis=0)) / 2.
    scale = np.ptp(stored[:, :3], axis=0).max() / 1.8
    expected = np.column_stack(((stored[:, :3] - center) / scale, stored[:, 3:]))
    assert sample.shape == (4, 6)
    np.testing.assert_allclose(sample, expected, atol=2e-7, rtol=2e-7)
    receipt.update(upstream_loader_passed=True, sample_shape=list(sample.shape),
                   supplied_target_topology=False, inference_run=False)
    (folder / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    check_upstream(sys.argv[1], sys.argv[2])
