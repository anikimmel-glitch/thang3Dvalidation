"""Create isolated upstream runtimes on Sagey; never change the scoring runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from evaluation.common import ROOT, read, save

LOCK = read(ROOT/"baselines/lock.json")


def command(argv, cwd, logfile, env=None, timeout=3600):
    with Path(logfile).open("w") as stream:
        result = subprocess.run(list(map(str,argv)),cwd=cwd,env=env,stdout=stream,
                                stderr=subprocess.STDOUT,timeout=timeout)
    print(json.dumps(dict(stage=Path(logfile).stem,returncode=result.returncode,
                         tail=Path(logfile).read_text(errors="replace")[-2000:])),flush=True)
    result.check_returncode()


def environment(prefix, python_version):
    prefix = Path(prefix)
    # Avoid loading the scoring environment's CUDA wheel libraries into a
    # different torch version. Keep system driver paths after this environment.
    own = prefix/f"lib/python{python_version}/site-packages/nvidia"
    paths = sorted(str(p) for p in own.glob("*/lib"))
    paths = [str(prefix/"lib")] + paths
    paths += [p for p in os.environ.get("LD_LIBRARY_PATH","").split(":")
              if p and "site-packages/nvidia" not in p]
    return dict(os.environ, LD_LIBRARY_PATH=":".join(dict.fromkeys(paths)),
                PATH=f"{prefix}/bin:"+os.environ["PATH"], CONDA_PREFIX=str(prefix))


def setup(name, out):
    spec = LOCK[name]
    dest = Path("/tmp/irl-extra-baselines")/name
    repo = dest/"source"
    prefix = dest/"env"
    dest.mkdir(parents=True,exist_ok=True)
    logs = out/"setup"/name
    logs.mkdir(parents=True,exist_ok=True)
    if not repo.exists():
        command(["git","init",repo],dest,logs/"git_init.log")
        command(["git","fetch","--depth","1",spec["repository"],spec["commit"]],
                repo,logs/"git_fetch.log")
        command(["git","checkout","--detach","FETCH_HEAD"],repo,logs/"git_checkout.log")
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if revision != spec["commit"]:
        raise RuntimeError("Upstream revision mismatch")
    requirement = repo/"requirements.txt"
    req_sha = hashlib.sha256(requirement.read_bytes()).hexdigest()
    if spec.get("requirements_sha256") and spec["requirements_sha256"] != req_sha:
        raise RuntimeError("Upstream requirements mismatch")
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("Conda required")
    if not (prefix/"bin/python").exists():
        packages = [f'python={spec["python"]}',"pip"]
        if name == "dualbrep_pc":
            packages.append("pythonocc-core=8.0.1.1")
        command([conda,"create","-y","-p",prefix,"--override-channels","-c","conda-forge",*packages],
                repo,logs/"conda.log",timeout=2400)
    python = prefix/"bin/python"
    env = environment(prefix,spec["python"])
    command([python,"-m","pip","install","--no-cache-dir","-r",requirement],
            repo,logs/"pip_install.log",env,3600)
    command([python,"-m","pip","check"],repo,logs/"pip_check.log",env,120)
    env = environment(prefix,spec["python"])
    command([python,"-m","pip","freeze"],repo,logs/"environment.txt",env,120)
    if name == "dualbrep_pc":
        # Only the two reconstruction weights are extracted; no author test set.
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(spec["model_id"],"checkpoints.zip",revision=spec["model_revision"])
        weights = {}
        with zipfile.ZipFile(archive) as z:
            for filename in spec["checkpoint_files"]:
                matches=[i for i in z.infolist() if Path(i.filename).name==filename]
                if len(matches)!=1:
                    raise RuntimeError(f"Ambiguous/missing checkpoint {filename}")
                path=repo/"checkpoints"/filename
                path.parent.mkdir(exist_ok=True)
                h=hashlib.sha256()
                with z.open(matches[0]) as src,path.open("wb") as target:
                    while data:=src.read(1024*1024):
                        target.write(data);h.update(data)
                weights[filename]=dict(sha256=h.hexdigest(),bytes=path.stat().st_size)
        save(logs/"checkpoint_receipt.json",weights)
        command([python,"-c","import torch,torch_cluster,OCC; "
                 "from model import DualVAE_PC; from rebuild_model import Parametrizer; "
                 "x=torch.ones(8,device='cuda'); assert (x+x).sum().item()==16; "
                 "print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"],
                repo,logs/"imports_canary.log",env,180)
    else:
        command([python,"-c","import cadquery as cq; import pymeshlab; import manifold3d; "
                 "from run_pipeline import process_single_stl; "
                 "s=cq.Workplane('XY').box(2,3,4).faces('>Z').workplane().hole(1).val(); "
                 "assert s.isValid() and len(s.Solids())==1; print('CADFit imports and CAD passed')"],
                repo,logs/"imports_canary.log",env,180)
    receipt=dict(repository=str(repo),python=str(python),environment_prefix=str(prefix),
                 revision=revision,requirements_sha256=req_sha,spec=spec)
    save(out/f"runtime_{name}.json",receipt)
    return receipt
