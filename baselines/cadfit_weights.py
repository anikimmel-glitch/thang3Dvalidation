"""Resolve and retain the optional upstream filter and its DINO source."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from evaluation.common import save

def main(repo,out):
    from huggingface_hub import HfApi,hf_hub_download
    import torch
    out.mkdir(parents=True,exist_ok=True)
    rev=HfApi().model_info("ghadinehme/CADFit").sha
    records=[]
    for name in ("model_mlp_med.pt","normalization.npz"):
        source=Path(hf_hub_download("ghadinehme/CADFit",name,revision=rev))
        target=out/name;shutil.copy2(source,target)
        records.append(dict(name=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            bytes=target.stat().st_size))
    sys.path.insert(0,str(repo))
    import filter_loops as fl
    dino,mlp,mu,sd,threshold,device=fl.load_filter(
        str(out/"model_mlp_med.pt"),str(out/"normalization.npz"))
    # Check real checkpoint forward/backward before committing to feature extraction.
    import copy
    head=copy.deepcopy(mlp)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-5)
    original=[p.detach().clone() for p in head.parameters()]
    x=torch.randn(8,len(mu),device=device);y=torch.arange(8,device=device)%2
    loss=torch.nn.functional.binary_cross_entropy_with_logits(
        head(x).squeeze(-1),y.float())
    loss.backward();optimizer.step()
    assert torch.isfinite(loss) and any(not torch.equal(a,b) for a,b in zip(original,head.parameters()))
    save(out/"training_canary.json",dict(real_checkpoint_forward_backward_passed=True,
        feature_dimension=len(mu),original_checkpoint_modified=False))
    torch.save(dino.state_dict(),out/"dino_state.pt")
    # Existing torch.hub cache pins the resolved source for all subsequent conditions.
    hub=Path(torch.hub.get_dir())
    sources=sorted(hub.glob("facebookresearch_dinov2_*"))
    if len(sources)!=1:raise RuntimeError("Ambiguous DINO source cache")
    files=[]
    for f in sorted(sources[0].rglob("*.py")):
        files.append(dict(path=str(f.relative_to(sources[0])),
                          sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    shutil.make_archive(str(out/"dino_source"),"gztar",sources[0])
    save(out/"receipt.json",dict(hf_revision=rev,weights=records,dino_source=files,
        threshold=threshold,feature_dimension=len(mu),device=device))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();main(a.repo,a.out)
