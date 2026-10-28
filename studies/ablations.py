"""Paired observation interventions; scoring always uses the unchanged reference."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from evaluation.common import ROOT,read,save,evaluate_program,record_failure,worker
from evaluation.geometry import fps,points
from finetuning.protocol import paired_report

CONDITIONS={
    "cad_recode_pc":["clean","resample_1","resample_2","sparse_64","noise_005","partial_half"],
    "cadrille_pc":["clean","resample_1","resample_2","sparse_64","noise_005","partial_half"],
    "cadrille_rgb":["clean","low_resolution","single_view"],
    "cadreasoner_native":["clean","low_resolution"]}

def make_input(source,dest,condition,seed,model):
    dest.mkdir(parents=True,exist_ok=True)
    with np.load(source/"reference.npz") as z:data={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(seed+({"resample_1":1001,"resample_2":2002}.get(condition,3003)))
    original=data["pc256"].copy()
    if condition.startswith("resample"):
        import trimesh
        surface=trimesh.load(source/"reference.stl",force="mesh")
        # Draw new observations independently of the held-fixed scoring samples.
        pool,_=points(surface,8192,seed+({"resample_1":1001,"resample_2":2002}[condition]))
        data["pc256"]=fps(pool).astype("f4")
    elif condition=="sparse_64":
        data["pc256"]=np.tile(fps(original,64),(4,1)).astype("f4")
    elif condition=="noise_005":
        data["pc256"]=(original+rng.normal(0,.005,original.shape)).astype("f4")
    elif condition=="partial_half":
        visible=original[original[:,0]>=np.median(original[:,0])]
        data["pc256"]=visible[np.arange(256)%len(visible)].astype("f4")
    np.savez_compressed(dest/"reference.npz",**data)
    for name in ["cadrille","cadreasoner"]:
        p=source/(name+".png")
        if not p.exists():continue
        image=Image.open(p).convert("RGB")
        if condition=="low_resolution":
            image=image.resize((max(1,image.width//4),max(1,image.height//4)),Image.Resampling.BOX).resize(image.size,Image.Resampling.BILINEAR)
        elif condition=="single_view" and name=="cadrille":
            arr=np.asarray(image).copy();arr[arr.shape[0]//2:]=0;arr[:,arr.shape[1]//2:]=0
            image=Image.fromarray(arr)
        image.save(dest/(name+".png"))
    save(dest/"intervention.json",dict(condition=condition,seed=seed,model=model,
        point_tokens=256,unique_points=len(np.unique(data["pc256"],axis=0)),
        input_surface_coverage=float(np.mean(cKDTree(data["pc256"]).query(data["points"])[0]<=.02)),
        changed_reference_scoring=False,
        limitations="Sparse/partial inputs repeat visible points to retain the fixed 256-token adapter. Noise sigma .005 canonical units. Low-resolution images are downsampled 4x then resized; no camera recalibration or independent output alignment."))

def infer(plan,out,model,contexts=None):
    import torch
    from evaluation.adapters import Adapter
    adapter=Adapter(model)
    context_mode=contexts is not None
    ids=set(contexts["geometry_only"]) if context_mode else set(plan["ablation_ids"])
    rows=[r for r in plan["files"] if r["pilot_id"] in ids]
    conditions=list(contexts) if context_mode else CONDITIONS[model]
    root=out/("context" if context_mode else "ablations")/model
    for row in rows:
        ident=row["pilot_id"];source=out/"benchmark/prepared"/ident
        for condition in conditions:
            dest=root/condition/ident;dest.mkdir(parents=True,exist_ok=True)
            if (dest/"metrics.json").exists():continue
            if not (source/"preparation.json").exists():
                record_failure(row,dest/"metrics.json","reference_preparation_failed");continue
            kind={"cadrille_rgb":"cadrille","cadreasoner_native":"cadreasoner"}.get(model)
            if kind and not (source/(kind+".png")).exists():
                record_failure(row,dest/"metrics.json","render_failed");continue
            make_input(source,dest/"input","clean" if context_mode else condition,row["seed"],model)
            diag=out/"diagnostics"/ident
            if model.endswith("_pc") and (diag/"patches.npz").exists():
                cloud=np.load(dest/"input/reference.npz")["pc256"]
                patches=np.load(diag/"patches.npz")["points"]
                tree=cKDTree(cloud)
                save(dest/"patch_observation_coverage.json",dict(
                    per_patch_fraction_within_002=[float(np.mean(tree.query(p)[0]<=.02)) for p in patches],
                    diagnostic_reference=str(diag/"reference_diagnostics.json"),
                    interpretation="Geometric proximity of input samples to reference surface patches; not semantic visibility."))
            try:
                torch.manual_seed(row["seed"]);torch.cuda.reset_peak_memory_stats()
                start=time.monotonic()
                text=contexts[condition][ident] if context_mode else None
                code=adapter.generate(dest/"input",context=text)
                save(dest/"generation.json",dict(seconds=time.monotonic()-start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(),condition=condition,
                    context=text,seed=row["seed"],n_candidates=1,reference_candidate_selection=False))
                evaluate_program(row,code,dest,source)
                if (dest/"prediction.stl").exists() and (diag/"reference_diagnostics.json").exists():
                    worker(["compare",source,diag,dest/"prediction.stl",dest/"feature_scores.json"],
                           dest/"feature_scores.log",180,module="studies.geometry")
            except Exception as exc:
                record_failure(row,dest/"metrics.json","inference_runtime_error",str(exc))
                torch.cuda.empty_cache()
    baseline=conditions[0]
    reports={}
    for condition in conditions[1:]:
        a={p.parent.name:read(p) for p in (root/baseline).glob("*/metrics.json")}
        b={p.parent.name:read(p) for p in (root/condition).glob("*/metrics.json")}
        reports[condition]=dict(overall=paired_report(rows,a,b),
            by_face_bin={str(i):paired_report([r for r in rows if r["complexity_bin"]==i],a,b) for i in range(5)})
    save(root/"paired_condition_comparisons.json",reports)

def run(plan,out,context_file=None):
    import torch
    processes=[]
    names=["cadrille_rgb","cadreasoner_native"] if context_file else list(CONDITIONS)
    if torch.cuda.device_count()<len(names):raise RuntimeError("GPU slots unavailable")
    for i,name in enumerate(names):
        log=(out/f'{"context" if context_file else "ablation"}_{name}.log').open("w")
        command=[sys.executable,"-m","studies.ablations","--plan",str(out/"study_plan.json"),"--out",str(out),"--model",name]
        if context_file:command+=["--contexts",str(context_file)]
        processes.append((name,subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),HF_HUB_OFFLINE="1"),stdout=log,stderr=subprocess.STDOUT),log))
    codes={}
    for name,p,log in processes:codes[name]=p.wait();log.close()
    save(out/("context_exit_codes.json" if context_file else "ablation_exit_codes.json"),codes)
    if any(codes.values()):raise RuntimeError(f"Observation/context worker failures: {codes}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--model");p.add_argument("--contexts",type=Path)
    a=p.parse_args();infer(read(a.plan),a.out,a.model,read(a.contexts) if a.contexts else None)
