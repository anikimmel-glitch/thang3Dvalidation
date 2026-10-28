"""Sensitivity controls; near-duplicate screening never silently edits the cohort."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from evaluation.common import read,save

def duplicate_screen(plan,out):
    rows=[r for r in plan["files"] if r["pilot_id"] in plan["evaluation_ids"] and
          (out/"benchmark/prepared"/r["pilot_id"]/"reference.npz").exists()]
    features=np.array([[np.log1p(r["faces"]),r["study_descriptors"]["nonplanar_fraction"],
                       r["study_descriptors"].get("sphericity") or 0] for r in rows])
    if len(rows)<2:
        save(out/"near_duplicate_screen.json",dict(candidates=[],screened=0));return
    std=np.std(features,axis=0);std=np.where(std>1e-12,std,1)
    tree=cKDTree(features/std)
    _,neighbors=tree.query(features/std,k=min(6,len(rows)))
    pairs=set()
    for i,near in enumerate(neighbors):
        for j in np.atleast_1d(near):
            if i!=j:pairs.add(tuple(sorted([i,int(j)])))
    candidates=[]
    for i,j in sorted(pairs):
        a,b=rows[i],rows[j]
        x=np.load(out/"benchmark/prepared"/a["pilot_id"]/"reference.npz")["points"]
        y=np.load(out/"benchmark/prepared"/b["pilot_id"]/"reference.npz")["points"]
        # First use fixed subsamples; only promising pairs incur a full comparison.
        small=sum(float(np.mean(cKDTree(t[::8]).query(s[::8])[0]**2)) for s,t in [(x,y),(y,x)])
        if small>.002:continue
        distances=[cKDTree(t).query(s)[0] for s,t in [(x,y),(y,x)]]
        recall=[float(np.mean(d<=.01)) for d in distances]
        chamfer=sum(float(np.mean(d*d)) for d in distances)
        candidates.append(dict(a=a["pilot_id"],b=b["pilot_id"],dataset_a=a["dataset"],dataset_b=b["dataset"],
            symmetric_squared_chamfer=chamfer,surface_coverage_at_005_side=recall,
            near_identical_candidate=bool(min(recall)>=.99 and chamfer<=.0001)))
    save(out/"near_duplicate_screen.json",dict(candidates=candidates,screened=len(rows),
        pair_candidates=len(pairs),automatic_exclusions=0,
        limitations="Six nearest descriptor neighbors; canonical orientation only; stochastic surface samples. Candidate screening is not proof of exact geometry equivalence or absence of rotations/remixes."))

def population_controls(plan,out):
    rows=plan.get("population_control_rows",[])
    if not rows:
        save(out/"analysis/population_controls.json",dict(status="not_available"));return
    def summary(rr):
        faces=[r["faces"] for r in rr]
        groups=defaultdict(list);creators=defaultdict(list)
        for r in rr:
            groups[r["source_group"]].append(r["faces"])
            creators["|".join(sorted(r.get("creators",[]))) or "unknown"].append(r["faces"])
        return dict(n=len(rr),sources=len(groups),face_mean=float(np.mean(faces)) if faces else None,
            face_median=float(np.median(faces)) if faces else None,
            source_macro_face_mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None,
            largest_source_fraction=max(map(len,groups.values()))/len(rr) if rr else None,
            top_ten_source_fraction=sum(sorted(map(len,groups.values()),reverse=True)[:10])/len(rr) if rr else None)
    by_group=defaultdict(int)
    for r in rows:by_group[r["source_group"]]+=1
    largest=set(sorted(by_group,key=lambda g:(-by_group[g],g))[:10])
    result=dict(curated=summary(rows),without_ten_largest_sources=summary([r for r in rows if r["source_group"] not in largest]),
        native_face_tail={str(f):summary(sorted(rows,key=lambda r:(r["faces"],r["sha256"]))[:len(rows)-int(np.ceil(f*len(rows)))])
                          for f in [0,.01,.05]},
        known_lettering_restored=summary(rows+plan.get("known_lettering_control_rows",[])),
        restored_lettering_bodies=len(plan.get("known_lettering_control_rows",[])),
        note="Descriptive full-cohort controls; model performance is separately measured on source-held-out bodies. Tail/source removals never alter the primary cohort.")
    save(out/"analysis/population_controls.json",result)

