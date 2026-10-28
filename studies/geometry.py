"""Auxiliary topology and surface-patch diagnostics, not functional ground truth."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from evaluation.geometry import canonical, mesh, points
from evaluation.common import read,save

def mesh_structure(m):
    parts=m.split(only_watertight=False)
    # trimesh.split can repair small holes; never let that turn an open input
    # into a supposedly closed boundary for our topology measurement.
    closed=bool(m.is_watertight and m.is_winding_consistent and len(parts)
                and all(p.is_watertight and p.is_winding_consistent for p in parts))
    genus=sum((2-int(p.euler_number))/2 for p in parts) if closed else None
    return dict(watertight_oriented_components=closed,mesh_components=len(parts),
                mesh_euler=int(m.euler_number),boundary_genus_sum=genus,
                interpretation="Closed triangulated boundary topology; not a count of labeled functional holes.")

def reference(row,source,prepared,out):
    import cadquery as cq
    shape=cq.importers.importStep(str(source)).val()
    shape,_,_=canonical(shape)
    native=len(shape.Faces())
    try:
        clean=shape.clean()
        change=abs(clean.Volume()-shape.Volume())/max(abs(shape.Volume()),1e-15)
        clean_ok=bool(clean.isValid() and len(clean.Solids())==1 and change<1e-7)
    except Exception:
        clean_ok=False;change=None
    carrier_counts=defaultdict(int);areas=defaultdict(float)
    for f in shape.Faces():
        carrier_counts[f.geomType()]+=1;areas[f.geomType()]+=f.Area()
    # Face partition diagnostic; count/area weighting and unified version both retained.
    faces=(clean if clean_ok else shape).Faces()
    face_areas=np.array([max(0,f.Area()) for f in faces])
    total=float(face_areas.sum())
    fractions=face_areas/max(total,1e-15)
    chosen=np.sort(np.random.default_rng(row["seed"]).choice(len(faces),min(256,len(faces)),replace=False))
    patches=[];xyz=[];failures=[]
    for i in chosen:
        try:
            p,_=points(mesh(faces[i]),32,row["seed"]+int(i))
            xyz.append(p)
            patches.append(dict(face_index=int(i),area_fraction=float(fractions[i]),carrier=faces[i].geomType(),
                                small_patch=bool(fractions[i]<.005)))
        except Exception as exc:
            failures.append(dict(face_index=int(i),error=str(exc)))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"patches.npz",points=np.stack(xyz) if xyz else np.empty((0,32,3)))
    pc=np.load(prepared/"reference.npz")["pc256"]
    coverage=[float(np.mean(cKDTree(pc).query(p)[0]<=.02)) for p in xyz]
    m=trimesh.load(prepared/"reference.stl",force="mesh")
    result=dict(pilot_id=row["pilot_id"],native_faces=native,
        unified_faces=len(clean.Faces()) if clean_ok else None,unification_valid=clean_ok,
        unification_volume_change=change,surface_counts=dict(carrier_counts),surface_areas=dict(areas),
        carrier_area_entropy=float(-sum((a/sum(areas.values()))*np.log2(a/sum(areas.values())) for a in areas.values() if a>0)),
        nonplanar_fraction=row["study_descriptors"]["nonplanar_fraction"],
        small_face_fraction=float(np.mean(fractions<.005)),
        small_face_area_fraction=float(fractions[fractions<.005].sum()),
        patches=patches,patch_input_coverage=coverage,patch_failures=failures,
        mesh_structure=mesh_structure(m),patch_threshold=.02,
        caveat="Sampled geometric surface patches, at most 256 faces x 32 points. Small face area is a representation-sensitive proxy, not wall thickness or a functional feature label.")
    save(out/"reference_diagnostics.json",result)

def compare(prepared,diagnostics,prediction,out):
    gt=read(diagnostics/"reference_diagnostics.json")
    m=trimesh.load(prediction,force="mesh")
    pc,_=points(m,32768,20261028)
    patches=np.load(diagnostics/"patches.npz")["points"]
    recall=[float(np.mean(cKDTree(pc).query(p)[0]<=.02)) for p in patches]
    def avg(ids):
        return float(np.mean([recall[i] for i in ids])) if ids else None
    small=[i for i,p in enumerate(gt["patches"]) if p["small_patch"]]
    curved=[i for i,p in enumerate(gt["patches"]) if p["carrier"]!="PLANE"]
    pred=mesh_structure(m);ref=gt["mesh_structure"]
    save(out,dict(patch_recalls=recall,patch_mean_recall=avg(list(range(len(recall)))),
        small_patch_recall=avg(small),curved_carrier_patch_recall=avg(curved),
        reference_topology=ref,prediction_topology=pred,
        boundary_genus_absolute_error=abs(pred["boundary_genus_sum"]-ref["boundary_genus_sum"])
            if pred["boundary_genus_sum"] is not None and ref["boundary_genus_sum"] is not None else None,
        condition="Valid predictions only; invalid outcomes remain in main metrics. Geometric diagnostics do not certify function."))

if __name__=="__main__":
    if sys.argv[1]=="reference":
        reference(read(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),Path(sys.argv[5]))
    else:
        compare(*map(Path,sys.argv[2:]))
