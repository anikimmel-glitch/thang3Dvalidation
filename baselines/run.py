"""Smoke-gated mesh/point-normal comparisons with retained per-subject failures."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import trimesh

from evaluation.common import ROOT,read,save,worker,record_failure
from evaluation.geometry import points
from evaluation.run import prepare_one
from evaluation.aggregate import summarize
from finetuning.split import validate
from baselines.setup import LOCK,setup,environment
from baselines.coordinates import inverse_bbox
from baselines.pointcloud_input import write_oriented_cloud
from baselines.runtime import select_methods
from baselines.rejected_candidates import all_candidates_rejected


def child(argv,cwd,dest,env,timeout):
    # Kill the full process group on timeout, including CADFit's worker pool.
    started=time.monotonic()
    timed_out=False
    with dest.open("w") as stream:
        p=subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdout=stream,
                           stderr=subprocess.STDOUT,start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()
    save(dest.with_suffix(".execution.json"),dict(
        command=list(map(str,argv)),returncode=p.returncode,
        signal=signal.Signals(-p.returncode).name if p.returncode<0 else None,
        timeout=timed_out,timeout_seconds=timeout,elapsed_seconds=time.monotonic()-started))
    return "timeout" if timed_out else "ok" if p.returncode==0 else "failed"


def one(row,name,runtime,input_root,out,rank,options=None):
    options = options or {}
    if options and name != "cadfit_mesh":
        raise ValueError("CADFit study options cannot change another baseline")
    target=out/"predictions"/name/row["pilot_id"]
    target.mkdir(parents=True,exist_ok=True)
    if (target/"metrics.json").exists():
        return read(target/"metrics.json")
    prepared=out/"prepared"/row["pilot_id"]
    if not (prepared/"preparation.json").exists():
        record_failure(row,target/"metrics.json","reference_preparation_failed")
        return read(target/"metrics.json")
    spec={**LOCK[name]}
    if options:
        spec["timeout_seconds"] = options.get("timeout_seconds", spec["timeout_seconds"])
    repo=Path(runtime["repository"])
    python=Path(runtime["python"])
    env=environment(runtime["environment_prefix"],spec["python"])
    # Untrusted shape-derived programs don't need cloud credentials.
    env={k:v for k,v in env.items() if k in ("PATH","LD_LIBRARY_PATH","CONDA_PREFIX","DISPLAY","LANG")}
    env.update(CUDA_VISIBLE_DEVICES=str(rank),PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    if name=="cadfit_mesh":
        env["CADFIT_INNER_WORKERS"]="8"
    work=target/"work"
    inputs=work/"input"
    inputs.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    try:
        mesh=trimesh.load(prepared/"reference.stl",force="mesh",process=False)
        if name=="cadfit_mesh":
            shutil.copy2(prepared/"reference.stl",inputs/f'{row["pilot_id"]}.stl')
            save(target/"coordinate_transform.json",inverse_bbox(mesh.vertices,2.))
            commands=[
                [python,repo/"run_pipeline.py",inputs,"--output-folder",work/"output",
                 "--fillet-chamfer","--keep-history"],
                [python,ROOT/"baselines/export_cadfit.py",repo,
                 work/"output"/row["pilot_id"]/"best_greedy_parallel_iterative.py",target/"raw.step"]]
            if options.get("fuse_export"):
                commands[1] += ["--fuse", "--mesh",
                    work/"output"/row["pilot_id"]/"best_greedy_parallel_iterative.stl"]
            if options.get("filter_model"):
                commands[0] += ["--filter-sketches", "--filter-model", options["filter_model"],
                                "--filter-norm", options["filter_norm"]]
        else:
            # Independent input samples, not the evaluation point sample itself.
            xyz,normals=points(mesh,32768,row["seed"]+200000)
            path=inputs/f'{row["pilot_id"]}.ply'
            save(target/"pointcloud_input.json",write_oriented_cloud(path,xyz,normals))
            # Mirror the actual serialized input, avoiding roundoff in transform inversion.
            cloud=trimesh.load(path,process=False)
            save(target/"coordinate_transform.json",inverse_bbox(cloud.vertices,1.8))
            commands=[
                [python,repo/"ae_reconstruct.py","config=config_pc.yaml",f"dataset.data_root={inputs}",
                 "dataset.name_list=null",f"runtime.output_dir={work/'recon'}","runtime.test_res=256"],
                [python,repo/"rebuild.py","--input",work/"recon","--out",work/"output"],
                [python,repo/"postprocess.py","--input",work/"output","--serial"]]
        for i,command in enumerate(commands):
            if name=="dualbrep_pc" and i==2:
                rejected=all_candidates_rejected(
                    work/"output",target/"stage_1.log",row["pilot_id"],
                    read(target/"stage_1.execution.json"))
                if rejected is not None:
                    save(target/"candidate_rejection.json",rejected)
                    record_failure(row,target/"metrics.json","invalid_geometry",
                                   "All 24 decoded candidates rejected: fewer than two usable faces")
                    return read(target/"metrics.json")
            remaining=spec["timeout_seconds"]-(time.monotonic()-started)
            command=[command[0],ROOT/"baselines/invoke.py",str(row["seed"]),*command[1:]]
            status=child(command,repo,target/f"stage_{i}.log",env,max(.1,remaining))
            if status!="ok":
                record_failure(row,target/"metrics.json",
                    "budget_exhausted" if status=="timeout" else "pipeline_runtime_error",
                    f"Stage {i}: {status}; see retained log")
                return read(target/"metrics.json")
        if name=="dualbrep_pc":
            prediction=work/"output"/f'{row["pilot_id"]}.step'
            if prediction.exists():
                shutil.copy2(prediction,target/"raw.step")
        if not (target/"raw.step").exists():
            record_failure(row,target/"metrics.json","invalid_geometry","Pipeline completed without a solid STEP")
            return read(target/"metrics.json")
        if not worker([target/"raw.step",target/"prediction.step",target/"coordinate_transform.json"],
                      target/"conversion.log",180,module="baselines.convert_step"):
            record_failure(row,target/"metrics.json","conversion_failed")
            return read(target/"metrics.json")
        if not read(target/"prediction.validity.json")["valid"]:
            record_failure(row,target/"metrics.json","invalid_geometry")
            return read(target/"metrics.json")
        if not worker(["evaluate",prepared/"row.json",prepared,target/"prediction.step",target/"metrics.json",1.],
                      target/"evaluation.log",300):
            record_failure(row,target/"metrics.json","evaluation_failed")
        return read(target/"metrics.json")
    except Exception as exc:
        record_failure(row,target/"metrics.json","pipeline_runtime_error",f"{type(exc).__name__}: {exc}")
        save(target/"failure.json",dict(error=str(exc),traceback=traceback.format_exc()))
        return read(target/"metrics.json")
    finally:
        save(target/"runtime.json",dict(seconds=time.monotonic()-started,method=name,
             input=spec["input"],procedure=spec["procedure"],budget_seconds=spec["timeout_seconds"],
             seed=row["seed"],bitwise_reproducible=False,
             source_text_or_topology_supplied=False,study_options=options))


def cohort(rows,name,runtime,args):
    def shard(rank):
        return [(r["pilot_id"],one(r,name,runtime,args.input,args.out,rank))
                for r in rows[rank::4]]
    records={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pairs in pool.map(shard,range(4)):
            records.update(pairs)
    result=summarize(rows,records)
    result["by_complexity_bin"]={str(b):summarize([r for r in rows if r["complexity_bin"]==b],
        {r["pilot_id"]:records[r["pilot_id"]] for r in rows if r["complexity_bin"]==b})
        for b in sorted({r["complexity_bin"] for r in rows})}
    # Fixed-budget results also count algorithm timeouts as zero. Keep the original
    # coverage/conditional benchmark metrics above, rather than hiding these failures.
    scored=[x for x in records.values() if x["status"] in
            ("evaluated","execution_failed","invalid_geometry","budget_exhausted")]
    result["fixed_budget"]=dict(scored=len(scored),timeout_count=sum(x["status"]=="budget_exhausted" for x in scored),
        native_surface_f1_mean=sum(x.get("scores",{}).get("native_frame",{}).get("surface_f1_at_1pct_side",0.)
                                   for x in scored)/len(scored) if scored else None)
    return result


def run(args):
    manifest=read(args.manifest)
    rows=validate(manifest)
    methods=select_methods(args.methods,LOCK)
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/"split_manifest.json",manifest)
    save(args.out/"baselines_lock.json",LOCK)
    # Ten train-only smoke subjects, two per reference face bin, fixed by ID.
    smoke=[]
    for b in range(5):
        candidates=sorted([r for r in rows if r["split"]=="train" and r["complexity_bin"]==b],
                          key=lambda r:r["pilot_id"])
        smoke.extend(candidates[:2])
    primary=set(manifest["primary_test_ids"])
    test=[r for r in rows if r["pilot_id"] in primary]
    save(args.out/"cohort.json",dict(smoke_ids=[r["pilot_id"] for r in smoke],test_ids=sorted(primary),
                                    split_sha256=manifest["split_sha256"],selected_methods=methods))
    # Setup independently; a failure in one new baseline does not suppress the other.
    results={}
    for name in methods:
        try:
            runtime=setup(name,args.out)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda r:prepare_one(r,args.input,args.out,[]),smoke))
            check=cohort(smoke,name,runtime,args)
            save(args.out/f"smoke_{name}.json",check)
            if check["valid"]<1 or any(s in check["statuses"] for s in
                ("pipeline_runtime_error","conversion_failed","evaluation_failed")):
                raise RuntimeError("Development smoke failed; test run not started")
            save(args.out/f"smoke_passed_{name}.json",dict(time=time.time()))
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda r:prepare_one(r,args.input,args.out,[]),test))
            result=cohort(test,name,runtime,args)
            save(args.out/f"summary_{name}.json",result)
            results[name]=dict(status="completed",expected=len(test),valid=result["valid"])
        except Exception as exc:
            results[name]=dict(status="failed",error=str(exc))
            save(args.out/f"failure_{name}.json",dict(error=str(exc),traceback=traceback.format_exc()))
        save(args.out/"status.json",results)
    if any(x["status"]=="failed" for x in results.values()):
        raise RuntimeError("One or more additional baselines failed; inspect status.json")
    save(args.out/"completed.json",results)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--method",choices=list(LOCK),action="append",dest="methods",
                   help="Run only named methods for a targeted infrastructure repair; default: both.")
    p.add_argument("--input",type=Path,default=Path("/opt/ml/input/data/training"))
    p.add_argument("--out",type=Path,default=Path("/opt/ml/checkpoints/irl-cad-extra-baselines"))
    run(p.parse_args())
