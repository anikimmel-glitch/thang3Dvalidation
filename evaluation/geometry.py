"""CAD preprocessing and metrics. Runs in bounded, isolated worker processes."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:
    import cadquery as cq
except ImportError:
    cq = None  # Metric/protocol unit tests do not require a CAD kernel.


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def canonical(shape):
    bb = shape.BoundingBox()
    center = np.array([bb.center.x, bb.center.y, bb.center.z])
    span = max(bb.xlen, bb.ylen, bb.zlen)
    if not math.isfinite(span) or span <= 1e-10:
        raise ValueError("Degenerate bounding box")
    return shape.translate(tuple(-center)).scale(2. / span), center, 2. / span


def mesh(shape):
    vertices, faces = shape.tessellate(0.0005, 0.05)
    result = trimesh.Trimesh(np.array([(v.x, v.y, v.z) for v in vertices]),
                            np.array(faces), process=True)
    if not len(result.faces) or not np.isfinite(result.vertices).all():
        raise ValueError("Empty or nonfinite tessellation")
    return result


def points(m, n, seed):
    xyz, face_ids = trimesh.sample.sample_surface(m, n, seed=seed)
    return xyz, m.face_normals[face_ids]


def fps(xyz, n=256):
    # Equivalent Euclidean farthest-point selection with initial index zero;
    # avoids installing PyTorch3D only for a CPU sampling operation.
    ids = np.empty(n, dtype=np.int64)
    distances = np.full(len(xyz), np.inf)
    current = 0
    for i in range(n):
        ids[i] = current
        distances = np.minimum(distances, np.sum((xyz - xyz[current]) ** 2, axis=1))
        current = int(np.argmax(distances))
    return xyz[ids]


def sharp_points(m, seed, n=8192):
    # Physical creases from adjacent triangles; smooth BRep seams are excluded.
    edges = m.face_adjacency_edges[m.face_adjacency_angles > np.deg2rad(20)]
    if not len(edges):
        return np.empty((0, 3))
    endpoints = m.vertices[edges]
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    if lengths.sum() <= 0:
        return np.empty((0, 3))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edges), n, p=lengths/lengths.sum())
    t = rng.random((n, 1))
    return endpoints[idx, 0] * (1-t) + endpoints[idx, 1] * t


def prepare(row, input_root, out):
    source = Path(input_root) / row["file"]
    if source.stat().st_size != row["bytes"]:
        raise ValueError("Reference byte-size mismatch")
    if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Reference SHA-256 mismatch")
    shape = cq.importers.importStep(str(source)).val()
    if not shape.isValid() or len(shape.Solids()) != 1:
        raise ValueError("Reference is not one valid solid")
    norm, center, scale = canonical(shape)
    m = mesh(norm)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m.export(out / "reference.stl")
    pc, normals = points(m, 32768, row["seed"])
    pre, _ = points(m, 8192, row["seed"])
    np.savez_compressed(out / "reference.npz", points=pc.astype("f4"),
        normals=normals.astype("f4"), pc256=fps(pre).astype("f4"),
        sharp_points=sharp_points(m, row["seed"]).astype("f4"))
    # PLY contains only points and normals, not face labels or ground-truth topology.
    cloud = trimesh.points.PointCloud(pc)
    ply = ["ply", "format ascii 1.0", f"element vertex {len(pc)}",
           "property float x", "property float y", "property float z",
           "property float nx", "property float ny", "property float nz", "end_header"]
    with (out / "reference.ply").open("w") as f:
        f.write("\n".join(ply) + "\n")
        np.savetxt(f, np.column_stack((pc, normals)), fmt="%.7g")
    before = len(norm.Faces())
    clean = norm.clean()
    volume_change = abs(clean.Volume() - norm.Volume()) / max(abs(norm.Volume()), 1e-15)
    clean_ok = bool(clean.isValid() and len(clean.Solids()) == 1 and volume_change < 1e-7)
    areas = collections.defaultdict(float)
    for face in norm.Faces():
        areas[face.geomType()] += face.Area()
    result = dict(pilot_id=row["pilot_id"], status="prepared",
        sha256_verified=True, reference_single_solid=True, reference_valid=True,
        bbox_center_original=center.tolist(), original_to_canonical_scale=scale,
        native_faces=before, unified_faces=len(clean.Faces()) if clean_ok else None,
        unification_valid_and_volume_preserved=clean_ok,
        unification_relative_volume_change=volume_change,
        surface_areas=dict(areas), triangle_count=len(m.faces),
        tessellation_watertight=bool(m.is_watertight),
        normalization="bbox midpoint at origin, longest side = 2",
        face_labels_given_to_models=False, seed=row["seed"])
    save(out / "preparation.json", result)
    return result


def vetted_program(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    tree = ast.parse(text)
    blocked = {"open", "exec", "eval", "compile", "getattr", "setattr", "delattr",
               "globals", "locals", "vars", "input", "breakpoint", "__import__"}
    allowed_imports = {"cadquery", "math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(x.name not in allowed_imports for x in node.names):
                raise ValueError("program_policy_rejection: import")
        if isinstance(node, ast.ImportFrom) and node.module not in allowed_imports:
            raise ValueError("program_policy_rejection: from import")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or
            node.attr in {"export", "exportStep", "exportStl", "exporters", "importers"}):
            raise ValueError("program_policy_rejection: attribute")
        if isinstance(node, ast.Name) and (node.id in blocked or node.id.startswith("__")):
            raise ValueError("program_policy_rejection: builtin")
    return compile(tree, "<predicted-cad-program>", "exec")


def execute(code_path, destination):
    # This policy reduces accidental side effects; it is NOT an OS sandbox.
    # Only run in a disposable evaluation container without user credentials.
    import builtins
    permitted = ("abs", "min", "max", "sum", "range", "len", "int", "float",
                 "round", "list", "tuple", "dict", "set", "zip", "enumerate",
                 "sorted", "reversed", "bool", "str")
    def restricted_import(name, *args, **kwargs):
        if name not in ("cadquery", "math"):
            raise ValueError("program_policy_rejection: import")
        return builtins.__import__(name, *args, **kwargs)
    safe_builtins = {k: getattr(builtins, k) for k in permitted}
    safe_builtins["__import__"] = restricted_import
    namespace = {"__builtins__": safe_builtins}
    exec(vetted_program(Path(code_path).read_text()), namespace)
    result = namespace["r"]
    objects = result.vals() if isinstance(result, cq.Workplane) else [result]
    if not objects or not all(isinstance(s, cq.Shape) for s in objects):
        raise ValueError("Result must contain CAD shapes")
    shape = objects[0] if len(objects) == 1 else cq.Compound.makeCompound(objects)
    valid = bool(shape.isValid() and len(shape.Solids()) == 1 and shape.Volume() > 0)
    save(Path(destination)/"execution.json", dict(status="executed",
        solid_count=len(shape.Solids()), valid=valid, faces=len(shape.Faces())))
    if not valid:
        return
    cq.exporters.export(shape, str(Path(destination) / "prediction.step"))
    m = mesh(shape.scale(.01))
    m.export(Path(destination) / "prediction.stl")


def fscore(pred, gt, threshold):
    if len(pred) == 0 or len(gt) == 0:
        return dict(precision=0., recall=0., f1=0.)
    precision = float(np.mean(cKDTree(gt).query(pred)[0] <= threshold))
    recall = float(np.mean(cKDTree(pred).query(gt)[0] <= threshold))
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def evaluate(row, reference_dir, prediction_path, out, native_scale):
    start = time.monotonic()
    shape = cq.importers.importStep(str(prediction_path)).val()
    valid = bool(shape.isValid() and len(shape.Solids()) == 1 and shape.Volume() > 0)
    gt = np.load(Path(reference_dir) / "reference.npz")
    result = dict(pilot_id=row["pilot_id"], reference_sha256=row["sha256"],
        role=row["role"], complexity_bin=row["complexity_bin"], reference_faces=row["faces"],
        status="evaluated", valid_single_solid=valid, predicted_faces=len(shape.Faces()),
        predicted_solids=len(shape.Solids()), sample_points=32768,
        threshold_units="canonical longest reference side = 2",
        invalid_outputs_receive_zero_f1=True, scores={})
    for mode in ["native_frame", "independent_bbox_normalization"]:
        s = shape.scale(native_scale) if mode == "native_frame" else canonical(shape)[0]
        m = mesh(s)
        pc, normals = points(m, 32768, row["seed"] + 100000)
        dg, ig = cKDTree(gt["points"]).query(pc)
        dp, ip = cKDTree(pc).query(gt["points"])
        cd = float(np.mean(dg**2) + np.mean(dp**2))
        normal = float((np.mean(np.abs(np.sum(normals * gt["normals"][ig], axis=1))) +
                        np.mean(np.abs(np.sum(gt["normals"] * normals[ip], axis=1)))) / 2)
        edges = sharp_points(m, row["seed"]+100000)
        sf = fscore(pc, gt["points"], 0.02)
        ef = fscore(edges, gt["sharp_points"], 0.02)
        result["scores"][mode] = dict(squared_chamfer_sum=cd if valid else None,
            absolute_normal_consistency=normal, surface_f1_at_1pct_side=sf["f1"] if valid else 0.,
            sharp_edge_f1_at_1pct_side=ef["f1"] if valid else 0.,
            surface_precision=sf["precision"], surface_recall=sf["recall"],
            edge_metric_applicable=bool(len(gt["sharp_points"])),
            reference_has_sharp_edges=bool(len(gt["sharp_points"])))
    result["evaluation_seconds"] = time.monotonic()-start
    save(out, result)


if __name__ == "__main__":
    mode = sys.argv[1]
    try:
        if mode == "prepare":
            prepare(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3], sys.argv[4])
        elif mode == "execute":
            execute(sys.argv[2], sys.argv[3])
        elif mode == "evaluate":
            evaluate(json.loads(Path(sys.argv[2]).read_text()), sys.argv[3],
                     sys.argv[4], sys.argv[5], float(sys.argv[6]))
    except Exception as error:
        print(type(error).__name__ + ": " + str(error)[:500], flush=True)
        raise
