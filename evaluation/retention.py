"""Index durable outputs and preserve reproducibility data without changing scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import threading
import time
from evaluation.common import save

TEXT_SUFFIXES={".json",".jsonl",".log",".txt",".py",".md",".yaml",".yml",".toml",".sh"}


def source_snapshot(root, out):
    """Keep source/configuration, never credentials, caches, or model weight blobs."""
    target=Path(out)/"retention"
    target.mkdir(parents=True,exist_ok=True)
    files=[]
    with tarfile.open(target/"source_snapshot.tar.gz","w:gz") as archive:
        for p in sorted(Path(root).rglob("*")):
            relative=p.relative_to(root)
            if (not p.is_file() or p.is_symlink() or
                    any(part.startswith(".") or part in ("__pycache__","scratch","checkpoints")
                        for part in relative.parts) or
                    p.suffix not in TEXT_SUFFIXES or p.stat().st_size>4*1024*1024):
                continue
            archive.add(p,arcname=str(relative),recursive=False)
            files.append(dict(path=str(relative),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              bytes=p.stat().st_size))
    save(target/"source_manifest.json",files)


def index(out):
    root=Path(out)
    target=root/"retention"
    target.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    metrics=[]
    statuses=Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink() or target in p.parents or p.suffix==".tmp":
            continue
        try:
            before=p.stat()
            entry=dict(path=str(p.relative_to(root)),bytes=before.st_size,mtime_ns=before.st_mtime_ns)
            # Small text files carry analysis/protocol data. Large geometry and
            # checkpoints are retained in full; avoid repeated large-file hashing.
            if p.suffix in TEXT_SUFFIXES and before.st_size<=4*1024*1024:
                data=p.read_bytes()
                after=p.stat()
                if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
                    continue
                entry["sha256"]=hashlib.sha256(data).hexdigest()
                if p.name=="metrics.json":
                    record=json.loads(data)
                    metrics.append(dict(artifact=entry["path"],record=record))
                    statuses[record.get("status","unknown")]+=1
            artifacts.append(entry)
        except (OSError,ValueError):
            continue  # an actively changing file is indexed on the next pass
    save(target/"artifact_index.json",dict(updated_at=time.time(),artifacts=artifacts,
        file_count=len(artifacts),bytes=sum(x["bytes"] for x in artifacts),
        note="Local checkpoint-directory inventory, not S3 upload verification. "
             "Metrics span smoke/train/validation/test; never pool these records without split/model grouping."))
    tmp=target/"metrics_records.tmp"
    with tmp.open("w") as f:
        for record in metrics:
            f.write(json.dumps(record,allow_nan=False)+"\n")
    tmp.replace(target/"metrics_records.jsonl")
    save(target/"metric_inventory.json",dict(updated_at=time.time(),records=len(metrics),
                                             statuses=dict(statuses),scores_modified=False))


class Retention:
    def __init__(self,out,interval=600):
        self.out=Path(out)
        self.interval=interval
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)

    def collect(self):
        try:
            index(self.out)
        except Exception as exc:
            save(self.out/"retention/index_error.json",dict(error=str(exc),time=time.time()))

    def loop(self):
        while not self.stop_event.wait(self.interval):
            self.collect()

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.collect()
