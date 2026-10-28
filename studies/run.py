"""Queued multi-study benchmark; human-dependent claims are never auto-completed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from evaluation.common import ROOT,read,save,worker
from evaluation.run import MODELS,prepare_one,run as benchmark
from finetuning.split import digest,validate
from studies.acquire import acquire
from studies.analysis import freeze_matches,analyze,training_analysis
from studies.annotations import packet,blinded_outputs,validated_contexts,shuffled_contexts

def check(plan):
    if digest({k:v for k,v in plan.items() if k!="plan_sha256"})!=plan["plan_sha256"]:
        raise ValueError("Study plan digest changed")
    from evaluation.manifest import validate_manifest
    validate_manifest(plan)
    protected=set(plan["protected_pilot_test_groups"])
    for row in plan["files"]:
        if row["dataset"]=="Thangs" and row["split"]!="test" and row["source_group"] in protected:
            raise ValueError("Protected test family entered optimization")
    return plan

def diagnostics(plan,out,inputs):
    rows=[r for r in plan["files"] if r["pilot_id"] in plan["evaluation_ids"]]
    def one(row):
        ident=row["pilot_id"];dest=out/"diagnostics"/ident;dest.mkdir(parents=True,exist_ok=True)
        prepared=out/"benchmark/prepared"/ident
        if (dest/"reference_diagnostics.json").exists():return
        if not (prepared/"preparation.json").exists():
            save(dest/"failure.json",dict(kind="reference_preparation_failed"));return
        save(dest/"row.json",row)
        if not worker(["reference",dest/"row.json",inputs/row["file"],prepared,dest],
                      dest/"diagnostics.log",300,module="studies.geometry"):
            save(dest/"failure.json",dict(kind="diagnostic_worker_failed",log=str(dest/"diagnostics.log")))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,rows))

def feature_scores(plan,out):
    ids=set(plan["evaluation_ids"]);work=[]
    for model in (out/"benchmark/predictions").glob("*"):
        for p in model.glob("*/metrics.json"):
            ident=p.parent.name
            if ident not in ids:continue
            diag=out/"diagnostics"/ident
            dest=out/"feature_scores"/model.name/(ident+".json");dest.parent.mkdir(parents=True,exist_ok=True)
            if not read(p).get("valid_single_solid") or not (diag/"reference_diagnostics.json").exists():continue
            work.append((out/"benchmark/prepared"/ident,diag,p.parent/"prediction.stl",dest))
    def one(item):
        prepared,diag,pred,dest=item
        if dest.exists():return
        if not worker(["compare",prepared,diag,pred,dest],dest.with_suffix(".log"),180,module="studies.geometry"):
            save(dest.with_suffix(".failure.json"),dict(status="feature_diagnostic_failed"))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,work))

def benchmark_suite(plan,out,acquisition_python=None):
    inputs=acquire(plan["files"],out,acquisition_python)
    # Annotation reference labels are requested before predictions/metrics are exposed.
    packet(plan,out/"annotation")
    args=argparse.Namespace(manifest=out/"study_plan.json",input=inputs,out=out/"benchmark",
                            models=list(MODELS),limit=0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r:prepare_one(r,inputs,args.out,["cadrille","cadreasoner"]),plan["files"]))
    diagnostics(plan,out,inputs)
    freeze_matches(plan,out)
    from studies.controls import duplicate_screen,population_controls
    duplicate_screen(plan,out)
    population_controls(plan,out)
    benchmark(args,plan)
    feature_scores(plan,out)
    analyze(plan,out)
    blinded_outputs(plan,out)
    from studies.ablations import run
    run(plan,out)
    save(out/"automated_studies_completed.json",dict(time=time.time(),
        completed=["cross_dataset","surface_patch_topology","input_ablations","representation_controls"],
        pending=["independent_human_annotation","validated_context_intervention"],
        separate_job="adaptation recipes and acceptance bias",
        note="Inspect actual coverage and failure receipts; this does not claim full scientific completion."))

def recipe_manifest(plan,dataset,n):
    group_order=plan["train_group_order"] if dataset=="Thangs" else plan["reference_train_group_order"]
    train_groups=set(group_order[:n])
    eval_ids=set(plan["evaluation_ids"])
    rows=[]
    for r in plan["files"]:
        include=(r["dataset"]==dataset and r["split"]=="train" and r["source_group"] in train_groups)
        include|=(r["dataset"]=="Thangs" and (r["split"]=="validation" or r["pilot_id"] in eval_ids))
        if include:rows.append(r)
    protected={r["source_group"] for r in rows if r["split"]=="test"}
    m=dict(schema_version=1,files=rows,seed=plan["seed"],bin_labels=plan["bin_labels"],
           protected_pilot_test_groups=sorted(protected),
           original_protected_pilot_test_groups=plan["protected_pilot_test_groups"],
           primary_test_ids=[r["pilot_id"] for r in rows if r["split"]=="test"],
           original_split_sha256=plan["original_split_sha256"],study_plan_sha256=plan["plan_sha256"],
           recipe=dict(dataset=dataset,attempted_train_sources=n,one_body_per_source=True,
              validation="Same 20 Thangs validation source groups; never protected test families",
              test="Same 250 frozen Thangs held-out bodies across recipes",
              compute="Same four candidates per attempted body; 82 equal-source-sampled examples per epoch, 3 epochs and 33 optimizer steps in every recipe. Actual candidate generation, token counts, accepted labels and timings retained."),
           dataset_pointer="Each row carries an exact acquisition source")
    for r in rows:
        if r["split"]!="test" and r["source_group"] in plan["protected_pilot_test_groups"]:
            raise ValueError("Known original test family entered recipe optimization")
    m["split_sha256"]=digest(m);validate(m)
    return m

def adaptation_suite(plan,out,acquisition_python=None):
    from studies.analysis import interval
    recipes=[("Thangs",20),("Thangs",41),("Thangs",82),("CADFS",82)]
    manifests=[recipe_manifest(plan,d,n) for d,n in recipes]
    needed={r["pilot_id"]:r for m in manifests for r in m["files"]}
    inputs=acquire(list(needed.values()),out,acquisition_python)
    statuses={}
    for (dataset,n),manifest in zip(recipes,manifests):
        name=f"{dataset.lower()}_{n}_sources";dest=out/"adaptation"/name;dest.mkdir(parents=True,exist_ok=True)
        path=dest/"split_manifest.json";save(path,manifest)
        with (dest/"orchestrator.log").open("w") as log:
            code=subprocess.run([sys.executable,"-m","finetuning.train","--manifest",str(path),
                "--input",str(inputs),"--out",str(dest),"--epoch-draws","82"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        training_analysis(manifest,dest,dest/"study_analysis.json")
        exits=read(dest/"track_exit_codes.json") if (dest/"track_exit_codes.json").exists() else {}
        failures=[model for model,c in exits.items() if c]
        gate_only=bool(exits and failures and all(
            (dest/f"track_{model}"/model/"insufficient_pseudo_labels.json").exists() for model in failures))
        statuses[name]=dict(returncode=code,status="complete" if not code else
                           "scientific_pseudo_label_gate_not_met" if gate_only else "infrastructure_or_code_failure")
        save(out/"adaptation_status.json",statuses)
        if code and not gate_only:
            raise RuntimeError(f"Adaptation recipe failed: {name}; diagnose before retry")
    # Every point carries realized selection/coverage; failed gates remain missing.
    summaries={name:read(out/"adaptation"/name/"study_analysis.json") for name in statuses}
    save(out/"learning_curve_data.json",dict(recipes=statuses,results=summaries,
        caveat="One seed; 82 training draws/epoch and 33 optimizer steps fixed. Token lengths and candidate generation still change wall time. Small recipes can legitimately fail the original minimum-label gate."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for model in MODELS:
        ns=[20,41,82];ys=[];accept=[]
        for n in ns:
            r=summaries[f"thangs_{n}_sources"].get(model,{})
            metric=r.get("overall",{}).get("metrics",{}).get("native_frame",{})
            ys.append(metric.get("delta",float("nan")))
            accept.append(len(r.get("accepted_ids",[]))/n)
        axes[0].plot(ns,ys,marker="o",label=model)
        axes[1].plot(ns,accept,marker="o",label=model)
    axes[0].set(xlabel="Attempted independent training sources",ylabel="Paired test surface-F1 improvement")
    axes[1].set(xlabel="Attempted independent training sources",ylabel="Fraction accepted for self-training")
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/"adaptation_learning_curves.pdf");plt.close(fig)
    save(out/"automated_studies_completed.json",dict(recipes=statuses,
        note="Gate failures are scientific limitations, not authorization to relax thresholds or train on test."))

def context_suite(plan,out,reviews,acquisition_python=None):
    if reviews is None:raise ValueError("Verified human review bundle required")
    bundle=read(reviews)
    contexts=validated_contexts(plan,bundle["reviewer_1"],bundle["reviewer_2"],bundle["adjudication"])
    shuffled=shuffled_contexts(contexts,{r["pilot_id"]:r["source_group"] for r in plan["files"]})
    condition_file=out/"verified_context_conditions.json"
    save(condition_file,dict(geometry_only={k:None for k in contexts},verified_context=contexts,shuffled_context=shuffled))
    rows=[r for r in plan["files"] if r["pilot_id"] in contexts]
    inputs=acquire(rows,out,acquisition_python)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r:prepare_one(r,inputs,out/"benchmark",["cadrille","cadreasoner"]),rows))
    from huggingface_hub import snapshot_download
    for entry in read(ROOT/"models.lock.json").values():
        if isinstance(entry,dict) and "model_id" in entry:
            snapshot_download(entry["model_id"],revision=entry["revision"],allow_patterns=["*.json","*.txt","*.model","*.safetensors","*.tiktoken","*.jinja"])
    from studies.ablations import run
    run(plan,out,condition_file)
    save(out/"automated_studies_completed.json",dict(contexts=len(contexts),review_bundle_sha256=digest(bundle)))

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--mode",choices=["benchmark","adaptation","context"],default="benchmark")
    p.add_argument("--reviews",type=Path)
    p.add_argument("--acquisition-python",type=Path,
                   help="Image Python with boto3; supplied by bootstrap, not a Sagey setting")
    p.add_argument("--out",type=Path,default=Path("/opt/ml/checkpoints/irl-cad-studies"))
    a=p.parse_args();plan=check(read(a.manifest));a.out.mkdir(parents=True,exist_ok=True)
    save(a.out/"study_plan.json",plan)
    try:
        if a.mode=="benchmark":benchmark_suite(plan,a.out,a.acquisition_python)
        elif a.mode=="adaptation":adaptation_suite(plan,a.out,a.acquisition_python)
        else:context_suite(plan,a.out,a.reviews,a.acquisition_python)
    except Exception as exc:
        save(a.out/"study_failure.json",dict(mode=a.mode,error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        raise

if __name__=="__main__":main()
