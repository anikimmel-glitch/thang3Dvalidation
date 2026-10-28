"""Worker-only bounded acquisition of exact presampled STEP members."""
import hashlib
from pathlib import Path
import struct
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zlib
from evaluation.common import save

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

def acquire(rows,out):
    import boto3
    from evaluation.common import read
    inputs=out/"inputs";inputs.mkdir(parents=True,exist_ok=True)
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
