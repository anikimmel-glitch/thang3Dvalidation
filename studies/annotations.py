"""Blinded review templates and fail-closed gates for creator-grounded text."""
import hashlib
import random
import shutil
from pathlib import Path
from evaluation.common import save

FEATURE_TYPES=("through_hole","slot","thin_wall","curved_interface","repeated_structure","other")

def packet(plan,out):
    out.mkdir(parents=True,exist_ok=True)
    for reviewer in ("reviewer_1","reviewer_2"):
        save(out/(reviewer+".json"),dict(reviewer_id=None,review_complete=False,
            instructions="Review reference geometry and source evidence, without model scores. Mark absent/ambiguous claims explicitly. Never transfer assembly-level statements or physical dimensions to a normalized body without evidence.",
            bodies=[dict(pilot_id=r["pilot_id"],sha256=r["sha256"],features=[],
                feature_schema=dict(feature_id="unique ID",type=list(FEATURE_TYPES),
                    reference_region="native face IDs or annotated canonical coordinates",
                    functional_claim=None,source_url=None,source_excerpt=None,body_grounded=None),
                verified_context=None,body_grounded=False,source_urls=r["source_urls"],
                reference_render="../benchmark/prepared/"+r["pilot_id"]+"/cadrille.png",
                reference_mesh="../benchmark/prepared/"+r["pilot_id"]+"/reference.stl",
                reviewed=False,uncertainty=None) for r in plan["annotation_candidates"]]))
    save(out/"source_evidence.json",plan["annotation_candidates"])
    save(out/"adjudication_template.json",dict(adjudicator_id=None,adjudication_complete=False,
        bodies=[],required="Two distinct completed human reviews, disagreement resolution, body-grounded evidence, no absolute dimensions or identifying creator/product/model IDs in experimental context."))
    save(out/"study_status.json",dict(status="awaiting_independent_human_review",
        human_validated_bodies=0,planned_bodies=len(plan["annotation_candidates"]),
        context_conditions=["geometry_only","verified_context","shuffled_context"],
        human_feature_metrics=["feature existence precision/recall","relative dimension error where measurable","reviewer agreement before adjudication"],
        no_fabricated_annotations=True))

def validated_contexts(plan,a,b,gold):
    if not a.get("review_complete") or not b.get("review_complete") or not gold.get("adjudication_complete"):
        raise ValueError("Human reviews/adjudication incomplete")
    if not a.get("reviewer_id") or not b.get("reviewer_id") or a["reviewer_id"]==b["reviewer_id"] or not gold.get("adjudicator_id"):
        raise ValueError("Two distinct human reviewers and adjudicator required")
    known={r["pilot_id"]:r for r in plan["annotation_candidates"]}
    am={r["pilot_id"]:r for r in a["bodies"]};bm={r["pilot_id"]:r for r in b["bodies"]}
    result={}
    for row in gold["bodies"]:
        ident=row["pilot_id"]
        if ident not in known or not am.get(ident,{}).get("reviewed") or not bm.get(ident,{}).get("reviewed"):
            raise ValueError("Unreviewed or unexpected subject")
        if row["sha256"]!=known[ident]["sha256"]:
            raise ValueError("Annotation attached to wrong geometry")
        if not row.get("body_grounded") or not row.get("source_url") or not row.get("source_excerpt"):
            continue
        if row["source_url"] not in known[ident]["source_urls"]:
            raise ValueError("Evidence URL is not linked to this source")
        if not row.get("identifiers_removed") or not row.get("absolute_dimensions_removed"):
            raise ValueError("Context must be de-identified and compatible with normalization")
        text=row.get("verified_context","").strip()
        if not text or len(text)>1200:
            raise ValueError("Missing or oversized verified context")
        result[ident]=text
    if len(result)<2:
        raise ValueError("At least two validated contexts needed for shuffle control")
    return result

def shuffled_contexts(contexts,groups=None):
    ids=sorted(contexts)
    random.Random(20261028).shuffle(ids)
    # Dataset annotation pool has one subject per source; refuse unsafe extensions.
    if groups and len({groups[i] for i in ids})!=len(ids):
        raise ValueError("Shuffle needs distinct source groups")
    return {ident:contexts[ids[(i+1)%len(ids)]] for i,ident in enumerate(ids)}

def blinded_outputs(plan,out):
    """Model identity mapping stays separate from the review packet."""
    ids={r["pilot_id"] for r in plan["annotation_candidates"]}
    mapping=[];review=[]
    for model in sorted((out/"benchmark/predictions").glob("*")):
        for pred in sorted(model.glob("*/metrics.json")):
            if pred.parent.name not in ids:continue
            token=hashlib.sha256(f"{model.name}/{pred.parent.name}/20261028".encode()).hexdigest()[:16]
            mapping.append(dict(blind_id=token,model=model.name,pilot_id=pred.parent.name,path=str(pred.parent)))
            geometry=out/"annotation/blinded_geometry"/(token+".stl")
            geometry.parent.mkdir(parents=True,exist_ok=True)
            if (pred.parent/"prediction.stl").exists():
                shutil.copyfile(pred.parent/"prediction.stl",geometry)
            review.append(dict(blind_id=token,pilot_id=pred.parent.name,
                geometry=str(geometry.relative_to(out/"annotation")) if geometry.exists() else None,
                feature_assessments=[],hallucinated_feature_count=None,reviewed=False,
                schema="Per adjudicated feature: present/absent/unassessable; relative dimension error if measurable. Failed output stays a failure."))
    random.Random(20261028).shuffle(review)
    save(out/"annotation/admin_model_mapping.json",mapping)
    for reviewer in ("reviewer_1","reviewer_2"):
        save(out/f"annotation/{reviewer}_prediction_assessments.json",dict(reviewer_id=None,complete=False,predictions=review))

def agreement(a,b):
    """Cohen kappa on overlapping feature existence judgments; missingness explicit."""
    from collections import Counter
    common=sorted(set(a)&set(b))
    valid=[k for k in common if a[k] in (True,False) and b[k] in (True,False)]
    if not valid:return dict(paired_judgments=0,kappa=None)
    observed=sum(a[k]==b[k] for k in valid)/len(valid)
    pa=sum(a[k] for k in valid)/len(valid);pb=sum(b[k] for k in valid)/len(valid)
    expected=pa*pb+(1-pa)*(1-pb)
    return dict(paired_judgments=len(valid),unassessable_or_missing=len(set(a)|set(b))-len(valid),
                agreement=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None)

def feature_report(gold,predictions):
    """Human-adjudicated feature detection and relative dimensions, no inferred labels."""
    import numpy as np
    if not gold.get("adjudication_complete") or not predictions.get("adjudication_complete"):
        raise ValueError("Completed human adjudication required")
    truth={(b["pilot_id"],f["feature_id"]):f for b in gold["bodies"] for f in b.get("features",[])}
    scores=[]
    for p in predictions["predictions"]:
        ident=p["pilot_id"];expected={key:f for key,f in truth.items() if key[0]==ident}
        observed={f["feature_id"]:f for f in p["feature_assessments"]}
        retained=0;missing=0;unknown=0;errors=[]
        for (_,fid),f in expected.items():
            prediction=observed.get(fid,{})
            state=prediction.get("present")
            if state is True:retained+=1
            elif state is False or p.get("output_failed"):missing+=1
            else:unknown+=1
            if prediction.get("relative_dimension_error") is not None:
                errors.append(abs(float(prediction["relative_dimension_error"])))
        hallucinations=p.get("hallucinated_feature_count")
        scores.append(dict(pilot_id=ident,blind_id=p["blind_id"],reference_features=len(expected),
            preserved=retained,missing=missing,unassessable=unknown,
            feature_recall=retained/(retained+missing) if retained+missing else None,
            feature_precision=retained/(retained+hallucinations) if hallucinations is not None and retained+hallucinations else None,
            relative_dimension_errors=errors))
    return dict(records=scores,note="Adjudicated human geometric feature judgments. No proof of load capacity, fit to unavailable mating parts, or mechanical function.")

if __name__=="__main__":
    import argparse
    from evaluation.common import read
    p=argparse.ArgumentParser();p.add_argument("--gold",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();save(a.out,feature_report(read(a.gold),read(a.predictions)))
