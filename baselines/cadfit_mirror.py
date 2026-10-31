"""Mirror reusable CADFit study features, metrics and checkpoints to datasets."""
import hashlib
import json
from pathlib import Path
import sys
import boto3

def main(root):
    plan=root/"plan.json"
    if not plan.exists():return
    identity=json.loads(plan.read_text())["plan_sha256"]
    bucket="tri-hcai-datasets"
    prefix=f"ani-kimmel-ctr/thang3Dvalidation/cadfit-filter-study/{identity}/"
    client=boto3.client("s3");receipts=[]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():continue
        r=p.relative_to(root)
        if "prepared" in r.parts or "work" in r.parts or "setup" in r.parts:continue
        if not (p.suffix in (".json",".jsonl") or
                r.parts[0] in ("filter_training","filter_weights")):continue
        if "dataset_mirror" in p.name:continue
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        key=prefix+str(r)
        client.upload_file(str(p),bucket,key,ExtraArgs={"Metadata":{"sha256":digest}})
        remote=client.head_object(Bucket=bucket,Key=key)
        if remote["ContentLength"]!=p.stat().st_size or remote.get("Metadata",{}).get("sha256")!=digest:
            raise RuntimeError("Dataset mirror verification failed: "+str(r))
        receipts.append(dict(path=str(r),size=p.stat().st_size,sha256=digest))
    (root/"dataset_mirror_receipt.json").write_text(json.dumps(dict(
        prefix="s3://"+bucket+"/"+prefix,objects=receipts,
        verification="Uploaded length and stored SHA256 metadata; not independent remote-byte readback"),indent=2))

if __name__=="__main__":main(Path(sys.argv[1]))
