"""Frozen CADFit retry and optional-filter fine-tuning protocol."""
from collections import defaultdict
from finetuning.split import digest, validate

def source_balanced(rows, split, limit):
    groups=defaultdict(list)
    for r in rows:
        if r["split"]==split:
            groups[r["source_group"]].append(r)
    # One relatively complex body per source; the split and selection ignore outcomes.
    representatives=[sorted(g,key=lambda r:(-r["faces"],r["pilot_id"]))[0]
                     for g in groups.values()]
    return sorted(representatives,key=lambda r:(-r["faces"],r["source_group"]))[:limit]

def plan(manifest):
    rows=validate(manifest)
    smoke=[]
    for b in range(5):
        smoke += sorted([r for r in rows if r["split"]=="train" and
                         r["complexity_bin"]==b],key=lambda r:r["pilot_id"])[:2]
    train=source_balanced(rows,"train",64)
    validation=source_balanced(rows,"validation",21)
    test=[r for r in rows if r["pilot_id"] in set(manifest["primary_test_ids"])]
    p=dict(split_sha256=manifest["split_sha256"],
        development_ids=[r["pilot_id"] for r in smoke],
        training_ids=[r["pilot_id"] for r in train],
        validation_ids=[r["pilot_id"] for r in validation],
        test_ids=[r["pilot_id"] for r in test],
        diagnostic_seconds=7200, reconstruction_seconds=2700, workers=8,
        conditions=["unfiltered","pretrained_filter","finetuned_filter"],
        filter_training=dict(epochs=10,learning_rate=1e-5,weight_decay=0.01,
            batch_size=256,seed=20261031,dino_frozen=True,normalization_frozen=True,
            selection="source-macro validation proxy AP; best learned epoch",
            label="candidate viability proxy, not CAD design history or final program membership",
            positive="at least one measured candidate IoU >= 0.01",
            negative="every precomputed candidate measured, all IoU <= 0.0005",
            unknown="unmeasured, failed, timed-out, ambiguous multi-loop, or intermediate-IoU",
            minimum_train_sources=5,minimum_validation_sources=3,
            minimum_examples_per_class=20),
        audit="All solids unioned; final mesh agreement >= .995; no component discarded",
        comparison="pretrained-filter vs fine-tuned-filter paired on frozen 450 test sources",
        training_is="MLP weight fine-tuning; core CADFit optimizer unchanged",
        disk_reserve_bytes=20_000_000_000,
        original_results="Original 45-minute raw-compound results retained separately",
        version=1)
    p["plan_sha256"]=digest(p)
    return p

def proxy_labels(operations, audited, kind, sketches):
    """Only fully measured poor candidates can supply a negative label."""
    from collections import defaultdict
    expected=defaultdict(set)
    measured=defaultdict(dict)
    for op in operations:
        if op["type"]==kind:
            expected[int(op["sketch_idx"])].add(digest(op))
    for item in audited:
        op=item["operation"]
        if op["type"]==kind:
            measured[int(op["sketch_idx"])][digest(op)]=float(item["iou"])
    result={}
    for i,entry in enumerate(sketches):
        if len(entry.get("loops_3d",[]))!=1 or not expected[i]:
            continue
        observed=measured[i]
        vals=[observed[k] for k in expected[i] if k in observed]
        if vals and max(vals)>=0.01:
            result[i]=1
        elif set(observed)>=expected[i] and vals and max(vals)<=0.0005:
            result[i]=0
    return result
