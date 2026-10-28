"""Outcome-blind matching and source-aware comparisons with reusable raw rows."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from evaluation.common import read,save
from evaluation.aggregate import summarize
from finetuning.protocol import scored,f1,paired_report

def covariates(d):
    if not d.get("unification_valid"):
        return None
    return [np.log1p(d["unified_faces"]),d["nonplanar_fraction"],np.log1p(100*d["small_face_fraction"])]

def match(a,b,xa,xb):
    """One-to-one maximum-cardinality caliper matching, then minimum distance."""
    if not len(a) or not len(b):return dict(pairs=[],unmatched_a=a,unmatched_b=b)
    x=np.array(xa);y=np.array(xb)
    sd=np.std(np.vstack([x,y]),axis=0);sd=np.where(sd>1e-12,sd,1.)
    delta=(x[:,None,:]-y[None,:,:])/sd
    allowed=np.all(np.abs(delta)<=.5,axis=2)
    costs=np.where(allowed,np.sum(delta**2,axis=2),1e9)
    costs=np.hstack([costs,np.full((len(a),len(a)),len(a)+len(b)+1.)])
    ia,ib=linear_sum_assignment(costs)
    pairs=[dict(a=a[i],b=b[j],distance=float(costs[i,j])) for i,j in zip(ia,ib) if j<len(b) and allowed[i,j]]
    ai={p["a"] for p in pairs};bi={p["b"] for p in pairs}
    idxa=[a.index(p["a"]) for p in pairs];idxb=[b.index(p["b"]) for p in pairs]
    return dict(pairs=pairs,unmatched_a=[i for i in a if i not in ai],unmatched_b=[i for i in b if i not in bi],
        standardization_sd=sd.tolist(),standardized_mean_difference_before=((x.mean(0)-y.mean(0))/sd).tolist(),
        standardized_mean_difference_after=((x[idxa].mean(0)-y[idxb].mean(0))/sd).tolist() if pairs else None,
        caliper_per_covariate=.5,covariates=["log1p unified faces","nonplanar area fraction","log1p(100 * small-face count fraction)"])

def freeze_matches(plan,out):
    rows=[r for r in plan["files"] if r["pilot_id"] in plan["evaluation_ids"]]
    available={}
    for r in rows:
        p=out/"diagnostics"/r["pilot_id"]/"reference_diagnostics.json"
        if p.exists() and covariates(read(p)) is not None:available[r["pilot_id"]]=covariates(read(p))
    th=[r["pilot_id"] for r in rows if r["dataset"]=="Thangs" and r["pilot_id"] in available]
    result={}
    for dataset in ("CADFS","Fusion","DeepCAD"):
        ids=[r["pilot_id"] for r in rows if r["dataset"]==dataset and r["pilot_id"] in available]
        result[dataset]=match(th,ids,[available[i] for i in th],[available[i] for i in ids])
    save(out/"matching_frozen_before_inference.json",dict(plan_sha256=plan["plan_sha256"],comparisons=result,
         diagnostic_missing_or_unification_failed=[r["pilot_id"] for r in rows if r["pilot_id"] not in available],
         outcome_data_used=False))
    return result

def interval(values):
    if not values:return None
    v=np.array(values,dtype=float);rng=np.random.default_rng(20261028)
    boot=[np.mean(rng.choice(v,len(v),replace=True)) for _ in range(2000)]
    return dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(v))

def analyze(plan,out):
    rows=[r for r in plan["files"] if r["pilot_id"] in plan["evaluation_ids"]]
    matching=read(out/"matching_frozen_before_inference.json")["comparisons"]
    report={};flat=[]
    models=sorted((out/"benchmark/predictions").glob("*"))
    plot={}
    for folder in models:
        records={p.parent.name:read(p) for p in folder.glob("*/metrics.json")}
        datasets={}
        for dataset in ("Thangs","CADFS","Fusion","DeepCAD"):
            rr=[r for r in rows if r["dataset"]==dataset];ids={r["pilot_id"] for r in rr}
            datasets[dataset]=summarize(rr,{k:v for k,v in records.items() if k in ids})
            datasets[dataset]["bins"]={}
            for b in range(5):
                br=[r for r in rr if r["complexity_bin"]==b]
                ids={r["pilot_id"] for r in br}
                datasets[dataset]["bins"][str(b)]=summarize(br,{k:v for k,v in records.items() if k in ids})
                datasets[dataset]["bins"][str(b)]["f1_interval"]=interval(
                    [f1(records[r["pilot_id"]]) for r in br if scored(records.get(r["pilot_id"],{}))])
            success=[r for r in rr if scored(records.get(r["pilot_id"],{}))]
            datasets[dataset]["f1_interval"]=interval([f1(records[r["pilot_id"]]) for r in success])
            for fraction in [0,.01,.05]:
                ordered=sorted(success,key=lambda r:(r["faces"],r["pilot_id"]))
                k=int(np.ceil(len(ordered)*fraction));kept=ordered[:-k] if k else ordered
                datasets[dataset].setdefault("outlier_sensitivity",[]).append(dict(
                    excluded_fraction=fraction,excluded_ids=[r["pilot_id"] for r in ordered[-k:]] if k else [],
                    n=len(kept),f1_mean=float(np.mean([f1(records[r["pilot_id"]]) for r in kept])) if kept else None,
                    f1_median=float(np.median([f1(records[r["pilot_id"]]) for r in kept])) if kept else None,
                    face_mean=float(np.mean([r["faces"] for r in kept])) if kept else None,
                    face_median=float(np.median([r["faces"] for r in kept])) if kept else None))
        contrasts={}
        for dataset,m in matching.items():
            pairs=[p for p in m["pairs"] if scored(records.get(p["a"],{})) and scored(records.get(p["b"],{}))]
            contrasts[dataset]=dict(expected_matched=len(m["pairs"]),paired_scored=len(pairs),
                thangs_minus_reference=interval([f1(records[p["a"]])-f1(records[p["b"]]) for p in pairs]),
                excluded_for_protocol_coverage=[p for p in m["pairs"] if p not in pairs])
        report[folder.name]=dict(datasets=datasets,matched=contrasts)
        screen=read(out/"near_duplicate_screen.json") if (out/"near_duplicate_screen.json").exists() else {}
        flagged=set()
        for pair in screen.get("candidates",[]):
            if pair.get("near_identical_candidate"):
                # Sensitivity omits both members; primary results keep both.
                flagged.update([pair["a"],pair["b"]])
        report[folder.name]["near_duplicate_sensitivity"]={}
        for dataset in datasets:
            rr=[r for r in rows if r["dataset"]==dataset and r["pilot_id"] not in flagged]
            ids={r["pilot_id"] for r in rr}
            report[folder.name]["near_duplicate_sensitivity"][dataset]=summarize(rr,{k:v for k,v in records.items() if k in ids})
        plot[folder.name]={d:[datasets[d]["bins"][str(b)]["scores"]["native_frame"]["surface_f1_mean"]
                               for b in range(5)] for d in datasets}
        for r in rows:
            ident=r["pilot_id"];dpath=out/"diagnostics"/ident/"reference_diagnostics.json"
            d=read(dpath) if dpath.exists() else {}
            feature=out/"feature_scores"/folder.name/(ident+".json")
            flat.append(dict(model=folder.name,pilot_id=ident,dataset=r["dataset"],source_group=r["source_group"],
                faces=r["faces"],descriptors=r["study_descriptors"],diagnostics=d,
                feature_scores=read(feature) if feature.exists() else None,metrics=records.get(ident)))
    save(out/"analysis/comparisons.json",report);save(out/"analysis/plot_data.json",plot)
    (out/"analysis/rows.jsonl").write_text("".join(__import__("json").dumps(x)+"\n" for x in flat))
    # Explicit diagnostics, no body-semantic claims from surface carrier labels.
    relationships={}
    for model in report:
        rr=[r for r in flat if r["model"]==model and scored(r["metrics"] or {})]
        relationships[model]={}
        for key in ["native_faces","unified_faces","nonplanar_fraction","small_face_fraction"]:
            keep=[r for r in rr if r["diagnostics"].get(key) is not None]
            corr=spearmanr([r["diagnostics"][key] for r in keep],[f1(r["metrics"]) for r in keep]) if len(keep)>2 else None
            relationships[model][key]=dict(n=len(keep),spearman=float(corr.statistic) if corr is not None and np.isfinite(corr.statistic) else None)
    save(out/"analysis/exploratory_correlations.json",relationships)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for ax,(model,lines) in zip(axes.flat,plot.items()):
        for d,values in lines.items():
            centers=np.array([np.nan if v is None else v for v in values])
            intervals=[report[model]["datasets"][d]["bins"][str(b)]["f1_interval"] for b in range(5)]
            lower=[x["ci95"][0] if x else np.nan for x in intervals]
            upper=[x["ci95"][1] if x else np.nan for x in intervals]
            ax.errorbar(range(5),centers,yerr=np.maximum(0,np.array([centers-lower,upper-centers])),marker="o",label=d,capsize=2)
        ax.set_title(model);ax.set_xticks(range(5),plan["bin_labels"]);ax.set_ylim(0,1)
        ax.set_xlabel("Native face-count bin");ax.set_ylabel("Failure-aware surface F1")
    axes.flat[0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/"analysis/cross_dataset_complexity.pdf");fig.savefig(out/"analysis/cross_dataset_complexity.png",dpi=180);plt.close(fig)
    # Function/feature plots preserve actual rows and conditioning.
    fig,ax=plt.subplots(figsize=(6,4))
    for model in report:
        rr=[r for r in flat if r["model"]==model and r["feature_scores"] and
            r["feature_scores"]["small_patch_recall"] is not None and scored(r["metrics"] or {})]
        ax.scatter([f1(r["metrics"]) for r in rr],[r["feature_scores"]["small_patch_recall"] for r in rr],
                   s=8,alpha=.35,label=model)
    ax.set(xlabel="Global surface F1",ylabel="Recall on sampled small surface patches",xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"analysis/small_patch_preservation.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for offset,model in enumerate(report):
        for i,dataset in enumerate(("CADFS","Fusion","DeepCAD")):
            r=report[model]["matched"][dataset]["thangs_minus_reference"]
            if r:
                y=i+offset*.15
                ax.errorbar(r["mean"],y,xerr=[[max(0,r["mean"]-r["ci95"][0])],[max(0,r["ci95"][1]-r["mean"])]],
                            fmt="o",label=model if i==0 else None,capsize=3)
    ax.axvline(0,color="gray",linewidth=.7);ax.set_yticks(np.arange(3)+.225,["CADFS","Fusion","DeepCAD"])
    ax.set_xlabel("Matched mean F1 difference: Thangs − reference (95% pair-bootstrap CI)")
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/"analysis/matched_dataset_gap.pdf");plt.close(fig)

def training_analysis(manifest,root,out):
    rows=manifest["files"];result={}
    for p in sorted(root.glob("track_*/*/pseudo_labels.json")):
        d=read(p);accepted={x["row"]["pilot_id"] for x in d["accepted"]}
        train=[r for r in rows if r["split"]=="train"]
        records={}
        for b in range(5):
            eligible=[r for r in train if r["complexity_bin"]==b]
            records[str(b)]=dict(expected=len(eligible),accepted=sum(r["pilot_id"] in accepted for r in eligible))
        model=p.parent.name
        test=[r for r in rows if r["pilot_id"] in manifest["primary_test_ids"]]
        baseline={f.parent.name:read(f) for f in p.parent.glob("test_pretrained/*/metrics.json")}
        tuned={f.parent.name:read(f) for f in p.parent.glob("test_selected/*/metrics.json")}
        feature_groups={
            "high_nonplanar_area":[r for r in test if r.get("study_descriptors",{}).get("nonplanar_fraction",0)>=.5],
            "low_nonplanar_area":[r for r in test if r.get("study_descriptors",{}).get("nonplanar_fraction",0)<.5],
            "spline_carrier_present":[r for r in test if any("spline" in k.lower() and v for k,v in r.get("study_descriptors",{}).get("surface_counts",{}).items())]}
        acceptance_curvature={}
        for label,high in [("high_nonplanar_area",True),("low_nonplanar_area",False)]:
            rr=[r for r in train if (r.get("study_descriptors",{}).get("nonplanar_fraction",0)>=.5)==high]
            acceptance_curvature[label]=dict(expected=len(rr),accepted=sum(r["pilot_id"] in accepted for r in rr))
        result[model]=dict(acceptance_by_complexity=records,acceptance_by_curvature=acceptance_curvature,
            expected_training=len(train),completed_training=d["completed_training_bodies"],
            accepted_ids=sorted(accepted),by_complexity={str(b):paired_report([r for r in test if r["complexity_bin"]==b],baseline,tuned) for b in range(5)},
            by_geometric_descriptor={label:paired_report(rr,baseline,tuned) for label,rr in feature_groups.items()},
            overall=paired_report(test,baseline,tuned),
            limitation="One optimization seed; 33 optimizer updates fixed in study recipes, accepted labels and token lengths vary. Inspect saved protocol for other runs.")
    save(out,result)
