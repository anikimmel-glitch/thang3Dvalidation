"""Run in CADFit's environment to export its saved construction program to STEP."""
from pathlib import Path
import argparse
import json
import sys

def export(source, dest, fuse=False, mesh=None):
    import cadquery as cq
    from refine_edges import load_solid_from_script
    obj=load_solid_from_script(str(source))
    if obj is None:
        raise RuntimeError("CADFit did not produce an executable final solid")
    shapes=obj.vals() if isinstance(obj,cq.Workplane) else [obj]
    shape=shapes[0] if len(shapes)==1 else cq.Compound.makeCompound(shapes)
    if fuse:
        cq.exporters.export(shape, str(dest.with_name("original_compound.step")))
        solids = shape.Solids()
        if not solids:
            raise RuntimeError("Generated program contains no solids")
        # Implements the same set union used by upstream _rebuild_clean_stl.
        # No largest-component selection, gap filling, or reference alignment.
        shape = solids[0].fuse(*solids[1:]).clean() if len(solids)>1 else solids[0]
        audit = dict(original_solids=len(solids), union_solids=len(shape.Solids()),
                     valid=shape.isValid(), operation="Boolean union of all output solids",
                     component_discarded=False, ground_truth_used=False)
        cq.exporters.export(shape, str(dest.with_name("union_candidate.step")))
        if mesh is not None:
            import trimesh
            from iou import _to_manifold, _OP_ADD, _OP_INTERSECT
            import manifold3d
            candidate = dest.with_name("union_candidate.stl")
            cq.exporters.export(shape, str(candidate), tolerance=0.001, angularTolerance=0.1)
            # Inputs already share CADFit's frame. Do not normalize each independently.
            a = trimesh.load(candidate, force="mesh")
            b = trimesh.load(mesh, force="mesh")
            ma, mb = _to_manifold(a), _to_manifold(b)
            if ma is None or mb is None:
                raise RuntimeError("Cannot compare final meshes without geometry repair")
            intersection = manifold3d.Manifold.batch_boolean([ma, mb], _OP_INTERSECT)
            union = manifold3d.Manifold.batch_boolean([ma, mb], _OP_ADD)
            audit["final_mesh_iou"] = float(abs(intersection.volume())/abs(union.volume()))
            audit["final_mesh_agreement_threshold"] = 0.995
            dest.with_suffix(".export_audit.json").write_text(json.dumps(audit,indent=2))
            if not audit["final_mesh_iou"] >= 0.995:
                raise RuntimeError("Union STEP differs from CADFit final mesh; export audit needs review")
        dest.with_suffix(".export_audit.json").write_text(json.dumps(audit,indent=2))
    # Disconnected results remain invalid outcomes under the common evaluator.
    cq.exporters.export(shape,str(dest))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("repository")
    p.add_argument("source",type=Path)
    p.add_argument("dest",type=Path)
    p.add_argument("--fuse",action="store_true")
    p.add_argument("--mesh",type=Path)
    a=p.parse_args()
    sys.path.insert(0,a.repository)
    export(a.source,a.dest,a.fuse,a.mesh)
