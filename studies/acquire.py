"""Worker-only bounded acquisition of exact presampled STEP members."""
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zlib
from evaluation.common import ROOT, read, save


def sdk_runtime():
    """Check the image's SDK offline, without requesting or exposing credentials."""
    import boto3
    import botocore
    if "s3" not in boto3.Session().get_available_services():
        raise RuntimeError("Acquisition SDK does not provide the S3 service")
    return dict(python=sys.executable, boto3=boto3.__version__,
                botocore=botocore.__version__, s3_service_available=True)

def verified(path,row):
    return (path.exists() and path.stat().st_size==row["bytes"] and
            hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"])

def unpack(block, row):
    if len(block)<30:
        raise ValueError("Truncated zip header")
    header=struct.unpack("<4s5H3I2H",block[:30])
    if header[0]!=b"PK\x03\x04" or header[2]&1 or header[3]!=row["compression"]:
        raise ValueError("Wrong/encrypted member")
    n,e=header[-2:]
    name=block[30:30+n].decode("utf-8" if header[2]&2048 else "cp437")
    if name!=row["member"]:
        raise ValueError("Member name mismatch")
    compressed=block[30+n+e:30+n+e+row["compressed_bytes"]]
    if len(compressed)!=row["compressed_bytes"]:
        raise ValueError("Truncated member")
    if row["compression"]==8:
        d=zlib.decompressobj(-15);raw=d.decompress(compressed,row["bytes"]+1)
        if not d.eof or d.unused_data or d.unconsumed_tail:
            raise ValueError("Invalid deflate stream")
    elif row["compression"]==0:
        raw=compressed
    else:
        raise ValueError("Unsupported compression")
    if len(raw)!=row["bytes"] or zlib.crc32(raw)!=row["crc32"]:
        raise ValueError("Member length/CRC mismatch")
    return raw

def acquire(rows,out,acquisition_python=None):
    # Keep SDK dependencies outside the locked scoring/CAD environment. The
    # bootstrap image already uses this interpreter to fetch the study manifest.
    out=Path(out)
    inputs=out/"inputs";inputs.mkdir(parents=True,exist_ok=True)
    if acquisition_python is not None:
        request=out/"acquisition_rows.json"
        save(request,rows)
        with (out/"acquisition_process.log").open("a") as log:
            subprocess.run([str(acquisition_python),"-m","studies.acquire",
                            "--rows",str(request.resolve()),"--out",str(out.resolve())],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not all(verified(inputs/r["file"],r) for r in rows):
            raise RuntimeError("Acquisition process returned without all frozen geometry hashes")
        return inputs
    import boto3
    ledger=read(out/"acquisition_ledger.json") if (out/"acquisition_ledger.json").exists() else dict(
        reserved_bytes=0,cap_bytes=2500000000,scope="Worker raw geometry transfers; no local laptop downloads",receipts=[])
    s3=boto3.client("s3")
    for row in rows:
        target=inputs/row["file"]
        if verified(target,row):
            continue
        f=row["fetch"]
        if f["kind"]=="zip_range":
            member=f["member"]
            start=member["header_offset"]
            # Full local filename+extra fields are each at most 65535 bytes.
            end=min(f["archive_bytes"]-1,start+30+131070+member["compressed_bytes"]-1)
            reserve=end-start+1
        elif f["kind"]=="s3":
            reserve=row["bytes"]
        else:
            raise ValueError("Only materialized S3 or pinned zip sources allowed")
        if ledger["reserved_bytes"]+reserve>ledger["cap_bytes"]:
            raise RuntimeError("Raw geometry transfer cap reached")
        ledger["reserved_bytes"]+=reserve
        save(out/"acquisition_ledger.json",ledger)
        if f["kind"]=="s3":
            uri=urlparse(f["uri"])
            meta=s3.head_object(Bucket=uri.netloc,Key=uri.path.lstrip("/"))
            if meta["ContentLength"]!=row["bytes"]:
                raise ValueError("S3 size differs from frozen manifest")
            response=s3.get_object(Bucket=uri.netloc,Key=uri.path.lstrip("/"),IfMatch=meta["ETag"])
            with response["Body"] as stream:
                raw=stream.read(row["bytes"]+1)
        else:
            headers={"Range":f"bytes={start}-{end}"}
            if f.get("etag"):headers["If-Match"]=f["etag"]
            with urlopen(Request(f["url"],headers=headers),timeout=90) as response:
                if response.status!=206 or response.headers.get("Content-Range")!=f'bytes {start}-{end}/{f["archive_bytes"]}':
                    raise ValueError("Server did not honor exact bounded range")
                block=response.read(reserve+1)
            if len(block)!=reserve:
                raise ValueError("Range length mismatch")
            raw=unpack(block,member)
        if len(raw)!=row["bytes"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]:
            raise ValueError("Geometry differs from frozen hash")
        tmp=target.with_suffix(".tmp");tmp.write_bytes(raw);tmp.replace(target)
        ledger["receipts"].append(dict(pilot_id=row["pilot_id"],sha256=row["sha256"],bytes=len(raw),source=f))
        save(out/"acquisition_ledger.json",ledger)
    return inputs


def main():
    import argparse
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument("--check-runtime",action="store_true")
    parser.add_argument("--rows",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    runtime=sdk_runtime()
    if args.check_runtime:
        print(json.dumps(runtime),flush=True)
        return
    if args.rows is None or args.out is None:
        parser.error("--rows and --out are required for acquisition")
    save(args.out/"acquisition_sdk.json",runtime)
    acquire(read(args.rows),args.out)


if __name__=="__main__":
    main()
