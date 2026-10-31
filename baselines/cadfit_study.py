"""Audited CADFit retry plus source-held-out MLP filter fine-tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from baselines.cadfit_protocol import plan
from baselines.run import one
from baselines.setup import setup, command, environment
from evaluation.common import ROOT, read, save
from evaluation.run import prepare_one
from evaluation.aggregate import summarize

def progress(out,phase,**kwargs):
    data=dict(phase=phase,time=time.time(),**kwargs)
    save(out/"progress.json",data)
    print(json.dumps(data),flush=True)

def run_cohort(rows, runtime, args, root, condition, options):
    root.mkdir(parents=True,exist_ok=True)
    prepared=root/"prepared"
    shared=args.out/"prepared"
    if not prepared.exists():prepared.symlink_to(shared,target_is_directory=True)
    records={}
    def shard(rank):
        completed=[]
        for row in rows[rank::8]:
            if shutil.disk_usage(args.out).free<20_000_000_000:
                raise RuntimeError("20GB remote disk reserve reached; retain artifacts and stop")
            rec=one(row,"cadfit_mesh",runtime,args.input,root,rank,options)
            completed.append((row["pilot_id"],rec))
            # A dedicated file per shard avoids races in the shared progress writer.
            save(root/f"worker_{rank}.json",dict(
                complete=len(completed),expected=len(rows[rank::8]),
                latest=row["pilot_id"],time=time.time()))
        return completed
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(shard,r) for r in range(8)]):
            records.update(f.result())
            progress(args.out,condition,recorded_shards=len(records),expected=len(rows))
    summary=summarize(rows,records)
    summary["by_complexity_bin"]={str(b):summarize(
        [r for r in rows if r["complexity_bin"]==b],
        {r["pilot_id"]:records[r["pilot_id"]] for r in rows if r["complexity_bin"]==b})
        for b in sorted({r["complexity_bin"] for r in rows})}
    eligible=[r for r in records.values() if r["status"] in
              ("evaluated","invalid_geometry","execution_failed","budget_exhausted")]
    summary["fixed_budget"]=dict(scored=len(eligible),
        mean_surface_f1=float(np.mean([r.get("scores",{}).get("native_frame",{}).get(
            "surface_f1_at_1pct_side",0.) for r in eligible])) if eligible else None,
        timeout_count=sum(r["status"]=="budget_exhausted" for r in eligible))
    save(root/"summary.json",summary)
    return records

def source_comparison(rows, before, after):
    allowed={"evaluated","invalid_geometry","execution_failed","budget_exhausted"}
    bysource={}
    changes=[]
    for row in rows:
        i=row["pilot_id"];a=before[i];b=after[i]
        if a["status"] not in allowed or b["status"] not in allowed:continue
        av=a.get("scores",{}).get("native_frame",{}).get("surface_f1_at_1pct_side",0.)
        bv=b.get("scores",{}).get("native_frame",{}).get("surface_f1_at_1pct_side",0.)
        bysource.setdefault(row["source_group"],[]).append(bv-av)
        changes.append(dict(pilot_id=i,source_group=row["source_group"],
                            pretrained=av,finetuned=bv,delta=bv-av))
    values=np.asarray([np.mean(x) for x in bysource.values()])
    if len(values):
        rng=np.random.default_rng(20261031)
        boots=[float(rng.choice(values,size=len(values),replace=True).mean())
               for _ in range(2000)]
        ci=np.percentile(boots,[2.5,97.5]).tolist()
    else:ci=None
    return dict(expected=len(rows),paired=len(changes),sources=len(values),
        source_mean_delta=float(values.mean()) if len(values) else None,
        paired_source_bootstrap_ci95=ci,records=changes,
        limitations="Single training seed; candidate-viability proxy supervision; "
                    "previously inspected pilot sources; matched 45-minute budget; "
                    "timeouts receive zero in this fixed-budget comparison.")

def main(a):
    manifest=read(a.manifest);protocol=plan(manifest)
    a.out.mkdir(parents=True,exist_ok=True)
    save(a.out/"plan.json",protocol);save(a.out/"split_manifest.json",manifest)
    lookup={r["pilot_id"]:r for r in manifest["files"]}
    cohorts={k:[lookup[i] for i in protocol[k+"_ids"]]
             for k in ("development","training","validation","test")}
    progress(a.out,"setup")
    runtime=setup("cadfit_mesh",a.out)
    from baselines.cadfit_patch import apply
    apply(runtime["repository"],a.out/"setup")
    env=environment(runtime["environment_prefix"],"3.10")
    # Module invocation finds this repository while upstream subprocesses retain their cwd.
    env["PYTHONPATH"]=str(ROOT)
    py=runtime["python"];repo=runtime["repository"]
    command([py,"-m","baselines.cadfit_export_canary","--repo",repo,
             "--out",a.out/"export_canary"],ROOT,a.out/"export_canary.log",env,300)
    progress(a.out,"prepare_references")
    wanted={r["pilot_id"]:r for rows in cohorts.values() for r in rows}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r:prepare_one(r,a.input,a.out,[]),wanted.values()))
    progress(a.out,"diagnostic")
    diagnostic=run_cohort(cohorts["development"],runtime,a,a.out/"diagnostic","diagnostic",
                          dict(fuse_export=True,timeout_seconds=7200))
    infra=[i for i,r in diagnostic.items() if r["status"] in
           ("pipeline_runtime_error","conversion_failed","evaluation_failed")]
    save(a.out/"diagnostic_gate.json",dict(infrastructure_cases=infra,
        valid=sum(bool(r.get("valid_single_solid")) for r in diagnostic.values()),
        heldout_used=False,score_based_gate=False))
    if infra:raise RuntimeError("Diagnostic export/runtime discrepancies need review: "+str(infra))
    # Poor algorithm scores do not suppress the authorized benchmark.
    baseline=run_cohort(cohorts["test"],runtime,a,a.out/"unfiltered_test","unfiltered_test",
                        dict(fuse_export=True,timeout_seconds=2700))
    progress(a.out,"filter_supervision")
    support=cohorts["training"]+cohorts["validation"]
    run_cohort(support,runtime,a,a.out/"supervision","filter_supervision",
               dict(fuse_export=True,timeout_seconds=7200))
    index=[]
    for r in support:
        root=a.out/"supervision/predictions/cadfit_mesh"/r["pilot_id"]
        index.append(dict(pilot_id=r["pilot_id"],split=r["split"],
            source_group=r["source_group"],source_urls=r.get("source_urls",[]),
            reference_sha256=r["sha256"],
            output=str(root/"work/output"/r["pilot_id"]),
            mesh=str(root/"work/input"/(r["pilot_id"]+".stl"))))
    save(a.out/"supervision_index.json",index)
    progress(a.out,"filter_setup")
    command([py,"-m","pip","install","--no-cache-dir","torch==2.5.1",
             "torchvision==0.20.1","pyvista==0.46.3","Pillow==11.3.0",
             "huggingface-hub==0.36.2","wandb==0.22.3"],repo,a.out/"filter_pip.log",env,3600)
    command([py,"-m","pip","check"],repo,a.out/"filter_pip_check.log",env,120)
    env=environment(runtime["environment_prefix"],"3.10")
    env["PYTHONPATH"]=str(ROOT)
    model_dir=a.out/"filter_weights";model_dir.mkdir(exist_ok=True)
    # Resolve one immutable HF commit for this job, then keep exact local copies.
    command([py,"-m","baselines.cadfit_weights","--repo",repo,"--out",model_dir],
            ROOT,a.out/"filter_weights.log",env,1800)
    model=model_dir/"model_mlp_med.pt";norm=model_dir/"normalization.npz"
    filter_out=a.out/"filter_training";filter_out.mkdir(exist_ok=True)
    progress(a.out,"filter_features")
    common=[py,"-m","baselines.cadfit_filter_train"]
    command(common+["extract","--repo",repo,"--model",model,"--norm",norm,
                    "--index",a.out/"supervision_index.json","--out",filter_out/"features"],
            ROOT,a.out/"filter_features.log",env,21600)
    progress(a.out,"filter_training")
    command(common+["train","--repo",repo,"--model",model,"--norm",norm,"--out",filter_out],
            ROOT,a.out/"filter_training.log",env,7200)
    selection=read(filter_out/"selection_frozen.json")
    learned=filter_out/f'filter_epoch_{selection["best_learned_epoch"]}.pt'
    # Freeze endpoints before generating any filtered test result.
    save(a.out/"test_conditions_frozen.json",dict(
        pretrained_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        learned_sha256=hashlib.sha256(learned.read_bytes()).hexdigest(),
        selection=selection,plan_sha256=protocol["plan_sha256"]))
    outcomes={}
    for condition,weight in [("pretrained_filter",model),("finetuned_filter",learned)]:
        options=dict(fuse_export=True,timeout_seconds=2700,filter_model=str(weight),
                     filter_norm=str(norm))
        outcomes[condition]=run_cohort(cohorts["test"],runtime,a,
            a.out/(condition+"_test"),condition+"_test",options)
    save(a.out/"comparison.json",source_comparison(cohorts["test"],
        outcomes["pretrained_filter"],outcomes["finetuned_filter"]))
    save(a.out/"completed.json",dict(time=time.time(),plan_sha256=protocol["plan_sha256"],
        baseline_cases=len(baseline),fine_tuning="optional sketch-filter MLP only",
        test_cases=len(cohorts["test"]),conditions=protocol["conditions"]))
    progress(a.out,"completed")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--input",type=Path,default=Path("/opt/ml/input/data/training"))
    p.add_argument("--out",type=Path,default=Path("/opt/ml/checkpoints/irl-cad-cadfit-study"))
    a=p.parse_args()
    try:main(a)
    except Exception as e:
        save(a.out/"failure.json",dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
