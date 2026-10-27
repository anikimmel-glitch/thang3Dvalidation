"""Missing/infrastructure errors stay visible; never enter a success denominator."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(rows, records):
    if len({r["pilot_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pilot IDs")
    wanted = {r["pilot_id"] for r in rows}
    if not set(records) <= wanted:
        raise ValueError("Unexpected subjects in metrics")
    status = Counter(r["status"] for r in records.values())
    scored = [r for r in records.values() if r["status"] == "evaluated" or
              (r["status"] == "execution_failed" and not r.get("infrastructure_failure"))]
    valid = [r for r in scored if r.get("valid_single_solid")]
    output = dict(expected=len(rows), recorded=len(records), missing=len(rows)-len(records),
        statuses=dict(status), scored=len(scored), valid=len(valid),
        validity_rate=len(valid)/len(scored) if scored else None,
        complete_protocol_coverage=len(scored)==len(rows), scores={})
    for mode in ("native_frame", "independent_bbox_normalization"):
        sf = [r.get("scores",{}).get(mode,{}).get("surface_f1_at_1pct_side",0.)
              if r.get("valid_single_solid") else 0. for r in scored]
        cd = [r["scores"][mode]["squared_chamfer_sum"] for r in valid]
        output["scores"][mode] = dict(
            surface_f1_mean=statistics.mean(sf) if sf else None,
            surface_f1_median=statistics.median(sf) if sf else None,
            conditional_valid_chamfer_mean=statistics.mean(cd) if cd else None,
            conditional_valid_chamfer_median=statistics.median(cd) if cd else None)
    return output


def aggregate(manifest, out):
    results = {}
    for model in sorted((out/"predictions").glob("*")):
        records = {p.parent.name:json.loads(p.read_text()) for p in model.glob("*/metrics.json")}
        group = {}
        for role in ("development", "evaluation"):
            rows = [r for r in manifest["files"] if r["role"] == role]
            ids = {r["pilot_id"] for r in rows}
            group[role] = summarize(rows, {k:v for k,v in records.items() if k in ids})
            group[role]["by_complexity_bin"] = {
                label:summarize([r for r in rows if r["complexity_bin"]==i],
                    {k:v for k,v in records.items() if k in ids and v["complexity_bin"]==i})
                for i,label in enumerate(manifest["bin_labels"])}
        results[model.name] = group
    (out/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=False))
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=Path("manifests/pilot.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(aggregate(json.loads(a.manifest.read_text()),a.out),indent=2))
