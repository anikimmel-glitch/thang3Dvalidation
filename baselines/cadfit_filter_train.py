"""Run inside CADFit's environment: cache real-loop features and fine-tune its MLP."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False))

def feature_rows(module, mesh_path, sketches, labels, dino, device, pv):
    import numpy as np
    mesh=pv.read(str(mesh_path))
    module._normalize_mesh(mesh,np)
    for si,y in sorted(labels.items()):
        entry=sketches[si]
        loop=np.asarray(entry["loops_3d"][0],dtype=np.float32)
        plane=(entry.get("origin",[0,0,0]),entry.get("normal",[0,0,1]),
               entry.get("xdir",[1,0,0]))
        # Keep the released four-view feature contract; render failure is missing data.
        iso=module._render_one(pv,np,mesh,loop,plane,"iso")
        face=module._render_one(pv,np,mesh,loop,plane,"face")
        images=[module._preprocess(im) for im in (*iso,*face)]
        emb=module._embed_images(images,dino,device,bs=4).reshape(-1)
        feat=np.concatenate([emb,module._geom_feats(entry,0,np)])
        if feat.shape!=(4113,) or not np.isfinite(feat).all():
            raise RuntimeError("Filter feature contract mismatch")
        yield si,y,feat

def extract(a):
    import numpy as np
    import filter_loops as fl
    from baselines.cadfit_protocol import proxy_labels
    dino,_,mu,sd,_,device=fl.load_filter(a.model,a.norm)
    pv=fl._setup_pyvista()
    manifest=json.loads(a.index.read_text())
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    records=[];failures=[]
    for row in manifest:
        dest=out/(row["pilot_id"]+".npz")
        evidence=out/(row["pilot_id"]+".json")
        if dest.exists() and evidence.exists():
            records.append(json.loads(evidence.read_text()));continue
        try:
            root=Path(row["output"])
            # Only initial input-space sketches, never residual geometry or test subjects.
            pre=root/"precomputed_operations.json"
            if not pre.exists():
                raise ValueError("Missing initial candidate table")
            operations=json.loads(pre.read_text())["operations"]
            audit=[json.loads(p.read_text()) for p in sorted(
                (pre.parent/"candidate_measurements").glob("*.json"))]
            xs=[];ys=[];identities=[]
            for kind,name in [("extrude","loops_split.json"),("revolve","loops_split_revolve.json")]:
                files=list(root.glob(name))
                if not files:
                    # Initial files are one level below the top output in some release paths.
                    files=[p for p in root.glob("*/"+name) if "residual" not in str(p)]
                if len(files)!=1:
                    raise ValueError("Missing/ambiguous initial loop table: "+name)
                sketches=json.loads(files[0].read_text())
                labels=proxy_labels(operations,audit,kind,sketches)
                effective_mesh=root/"decimated_input.stl"
                if not effective_mesh.exists():
                    effective_mesh=Path(row["mesh"])
                for si,y,x in feature_rows(fl,effective_mesh,sketches,labels,dino,device,pv):
                    xs.append(x);ys.append(y);identities.append([kind,si,0])
            if not xs:raise ValueError("No unambiguous measured supervision")
            x=(np.stack(xs)-mu)/sd
            if not np.isfinite(x).all():raise ValueError("Nonfinite standardized features")
            np.savez_compressed(dest,x=x.astype(np.float32),y=np.asarray(ys,dtype=np.int64))
            record={**row,"cache":str(dest),"queries":len(ys),"positives":sum(ys),
                    "identities":identities,"cache_sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
            write(evidence,record);records.append(record)
        except Exception as e:
            failures.append(dict(pilot_id=row["pilot_id"],error=repr(e)))
        write(out/"progress.json",dict(expected=len(manifest),cached=len(records),failures=failures))
    write(out/"index.json",records)
    write(out/"completed.json",dict(records=len(records),failures=failures))

def metrics(y,p,threshold=.425):
    import numpy as np
    from sklearn.metrics import average_precision_score,accuracy_score,precision_recall_fscore_support
    pr,re,f1,_=precision_recall_fscore_support(y,p>=threshold,average="binary",zero_division=0)
    return dict(ap=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
                accuracy=float(accuracy_score(y,p>=threshold)),precision=float(pr),
                recall=float(re),f1=float(f1),threshold=threshold)

def train(a):
    import numpy as np
    import torch
    import filter_loops as fl
    import os
    random.seed(20261031);np.random.seed(20261031);torch.manual_seed(20261031)
    ckpt=torch.load(a.model,map_location="cpu",weights_only=False)
    threshold=float(ckpt.get("default_threshold",.425))
    model=fl._build_mlp(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    device="cuda" if torch.cuda.is_available() else "cpu";model.to(device)
    rows=json.loads((a.out/"features/index.json").read_text())
    arrays={}
    for split,minimum in [("train",5),("validation",3)]:
        selected=[r for r in rows if r["split"]==split]
        if len({r["source_group"] for r in selected})<minimum:
            raise RuntimeError("Insufficient independently sourced filter supervision: "+split)
        xs=[];ys=[];weights=[];groups=[]
        for r in selected:
            z=np.load(r["cache"]);n=len(z["y"])
            xs.append(z["x"]);ys.append(z["y"]);weights.extend([1/n]*n)
            groups.extend([r["source_group"]]*n)
        x=np.concatenate(xs);y=np.concatenate(ys)
        if min(int((y==0).sum()),int((y==1).sum()))<20:
            raise RuntimeError("Insufficient measured positive/negative examples: "+split)
        arrays[split]=(x,y,np.asarray(weights),np.asarray(groups))
    wb=None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wb=wandb.init(project="thangs-cadfit-filter",entity="tri",
                      name=os.environ.get("TRAINING_JOB_NAME","cadfit-filter"),
                      dir=str(a.out),config=dict(epochs=10,lr=1e-5,
                        backbone_frozen=True,label="measured_candidate_viability",
                        train_sources=len(np.unique(arrays["train"][3])),
                        validation_sources=len(np.unique(arrays["validation"][3]))))
        write(a.out/"wandb_identity.json",dict(url=wb.url,id=wb.id))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    history=[];best=None
    for epoch in range(11):
        if epoch:
            model.train()
            x,y,w,_=arrays["train"]
            # Equal source weighting, retaining each source's natural class mix.
            order=np.random.permutation(len(y));loss_sum=0.
            for idx in np.array_split(order,max(1,(len(order)+255)//256)):
                xb=torch.tensor(x[idx],device=device)
                yb=torch.tensor(y[idx],dtype=torch.float32,device=device)
                wb=torch.tensor(w[idx]/w.mean(),dtype=torch.float32,device=device)
                optimizer.zero_grad()
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(
                    model(xb).squeeze(-1),yb,reduction="none")*wb).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();loss_sum+=float(loss)
        model.eval()
        x,y,_,groups=arrays["validation"];pred=[]
        with torch.no_grad():
            for start in range(0,len(y),1024):
                pred.extend(torch.sigmoid(model(torch.tensor(x[start:start+1024],
                    device=device)).squeeze(-1)).cpu().tolist())
        pred=np.asarray(pred);group_ap=[]
        for g in np.unique(groups):
            m=metrics(y[groups==g],pred[groups==g],threshold)
            if m["ap"] is not None:group_ap.append(m["ap"])
        if not group_ap:raise RuntimeError("No validation source with both classes")
        result=dict(epoch=epoch,source_macro_ap=float(np.mean(group_ap)),**metrics(y,pred,threshold))
        history.append(result)
        if wb:wb.log({"validation/"+k:v for k,v in result.items()},step=epoch)
        np.savez_compressed(a.out/f"validation_epoch_{epoch}.npz",y=y,p=pred,groups=groups)
        payload={**ckpt,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}
        torch.save(payload,a.out/f"filter_epoch_{epoch}.pt")
        torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate()),
                   a.out/f"training_state_epoch_{epoch}.pt")
        # Keep a genuine learned endpoint, even if epoch-zero remains the deployment choice.
        if epoch and (best is None or result["source_macro_ap"]>best["source_macro_ap"]):
            best=result
        write(a.out/"training_progress.json",dict(history=history,best_learned=best))
    selected=0 if history[0]["source_macro_ap"]>=best["source_macro_ap"] else best["epoch"]
    write(a.out/"selection_frozen.json",dict(best_learned_epoch=best["epoch"],
        deployment_epoch=selected,selection_metric="source_macro_validation_proxy_ap",
        history=history,test_used=False,backbone_frozen=True,normalization_frozen=True,
        label_limitation="Candidate viability proxy; not ground-truth construction sketches"))
    if wb:wb.finish()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["extract","train"])
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--model",required=True);p.add_argument("--norm",required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--index",type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.repo))
    extract(a) if a.mode=="extract" else train(a)
