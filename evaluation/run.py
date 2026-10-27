"""Source-frozen pretrained pilot with per-subject receipts and resumable output."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluation.common import ROOT, read, save, worker, record_failure, evaluate_program
from evaluation.aggregate import aggregate

MODELS = ("cad_recode_pc", "cadrille_pc", "cadrille_rgb", "cadreasoner_native")


def validate_manifest(m):
    rows = m["files"]
    for key in ("pilot_id", "source_group", "sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Repeated {key}")
    for r in rows:
        if Path(r["file"]).name != r["file"] or r["role"] not in ("development","evaluation"):
            raise ValueError("Invalid manifest path or role")
    return rows


def prepare_one(row, input_root, out, image_kinds):
    target = out/"prepared"/row["pilot_id"]
    target.mkdir(parents=True, exist_ok=True)
    save(target/"row.json", row)
    if not (target/"preparation.json").exists():
        if not worker(["prepare", target/"row.json", input_root, target],
                      target/"prepare.log", 300):
            save(target/"preparation_failure.json", dict(status="reference_preparation_failed",
                detail=(target/"prepare.log").read_text()[-500:]))
            return
    for kind in image_kinds:
        if not (target/f"{kind}.png").exists():
            if not worker([kind,target/"reference.stl",target/f"{kind}.png"],
                          target/f"{kind}_render.log", 90, module="evaluation.render"):
                save(target/f"{kind}_render_failure.json",dict(status="render_failed",
                    detail=(target/f"{kind}_render.log").read_text()[-500:]))


def infer(args, rows):
    import torch
    from evaluation.adapters import Adapter
    # The parent populates the checkpoint cache once before forking GPU workers.
    torch.manual_seed(20261027)
    adapter = Adapter(args.model)
    save(args.out/f"loaded_{args.model}_{args.shard}.json",dict(
        model=args.model, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        adapter="greedy, one candidate, SDPA, original view conventions",
        model_lock=read(ROOT/"models.lock.json"), started_at=time.time()))
    for row in rows[args.shard::args.shards]:
        dest=args.out/"predictions"/args.model/row["pilot_id"]
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/"metrics.json").exists():
            continue
        prepared=args.out/"prepared"/row["pilot_id"]
        kind = {"cadrille_rgb":"cadrille","cadreasoner_native":"cadreasoner"}.get(args.model)
        if not (prepared/"preparation.json").exists():
            record_failure(row,dest/"metrics.json","reference_preparation_failed")
            continue
        if kind and not (prepared/f"{kind}.png").exists():
            record_failure(row,dest/"metrics.json","render_failed")
            continue
        start=time.monotonic()
        try:
            code=adapter.generate(prepared)
        except Exception as exc:
            record_failure(row,dest/"metrics.json","inference_runtime_error",
                           f"{type(exc).__name__}: {str(exc)[:400]}")
            torch.cuda.empty_cache()
            continue
        save(dest/"generation.json",dict(inference_seconds=time.monotonic()-start,
            prompt_contains_source_text=False, ground_truth_candidate_selection=False,
            max_new_tokens=1500 if args.model=="cadreasoner_native" else 768,
            n_candidates=1, refinement_steps=0))
        evaluate_program(row,code,dest,prepared)
        print(json.dumps(dict(model=args.model,subject=row["pilot_id"],
              status=read(dest/"metrics.json")["status"])),flush=True)


def run(args, manifest):
    import torch
    from huggingface_hub import snapshot_download
    rows=validate_manifest(manifest)
    if args.limit:
        # Smoke uses development subjects only, balanced across the five bins.
        dev=[r for r in rows if r["role"]=="development"]
        ordered=sorted(dev,key=lambda r:(sum(x["complexity_bin"]==r["complexity_bin"]
                      for x in dev[:dev.index(r)]),r["complexity_bin"]))
        rows=ordered[:args.limit]
        manifest={**manifest,"files":rows,"run_is_smoke":True}
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/"run_manifest.json",manifest)
    save(args.out/"runtime.json",dict(python=sys.version,torch=torch.__version__,
        cuda=torch.version.cuda,device_count=torch.cuda.device_count(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob("*.py") if ".git" not in p.parts}))
    gpu_count=torch.cuda.device_count()
    if not gpu_count:
        raise RuntimeError("GPU required for pretrained inference")
    kinds = [kind for model,kind in (("cadrille_rgb","cadrille"),
             ("cadreasoner_native","cadreasoner")) if model in args.models]
    with ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as pool:
        list(pool.map(lambda r:prepare_one(r,args.input,args.out,kinds), rows))
    good=sum((args.out/"prepared"/r["pilot_id"]/"preparation.json").exists() for r in rows)
    save(args.out/"preparation_summary.json",dict(expected=len(rows),prepared=good))
    if not good:
        raise RuntimeError("No prepared references; refusing meaningless inference")
    lock=read(ROOT/"models.lock.json")
    for name in args.models:
        key="cad_recode" if name.startswith("cad_recode") else (
            "cadreasoner" if name.startswith("cadreasoner") else "cadrille")
        tokenizer_key={"cad_recode":"Qwen_Qwen2-1.5B","cadrille":"Qwen_Qwen2-VL-2B-Instruct",
                       "cadreasoner":"cadreasoner"}[key]
        for k in set([key,tokenizer_key]):
            item=lock[k]
            snapshot_download(item["model_id"],revision=item["revision"],
                allow_patterns=["*.json","*.txt","*.model","*.safetensors","*.tiktoken"])
        shards=min(gpu_count,len(rows))
        processes=[]
        for rank in range(shards):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),HF_HUB_OFFLINE="1")
            command=[sys.executable,"-m","evaluation.run","--manifest",
                str(args.out/"run_manifest.json"),"--out",str(args.out),
                "--model",name,"--shard",str(rank),"--shards",str(shards)]
            processes.append(subprocess.Popen(command,cwd=ROOT,env=env))
        codes=[p.wait() for p in processes]
        aggregate(manifest,args.out)
        if any(codes):
            raise RuntimeError(f"{name} infrastructure worker failed: {codes}")
    aggregate(manifest,args.out)
    save(args.out/"completed.json",dict(expected_shapes=len(rows),models=args.models,
         note="Inspect summary coverage: completed orchestration is not 100% model validity."))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=ROOT/"manifests/pilot.json")
    p.add_argument("--input",type=Path,default=Path("/opt/ml/input/data/training"))
    p.add_argument("--out",type=Path,default=Path("/opt/ml/checkpoints/irl-cad-pilot"))
    p.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    a=p.parse_args()
    m=read(a.manifest)
    if a.model:
        infer(a,validate_manifest(m))
    else:
        run(a,m)
