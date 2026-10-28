"""Freeze source families, including every sibling of the original pilot test."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from evaluation.common import read, save

SEED = 20261028


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(manifest):
    rows = manifest["files"]
    if manifest["split_sha256"] != digest({k: v for k, v in manifest.items()
                                         if k != "split_sha256"}):
        raise ValueError("Frozen split digest mismatch")
    for key in ("pilot_id", "file", "sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    maps = defaultdict(dict)
    for row in rows:
        role = row["split"]
        if role not in ("train", "validation", "test") or Path(row["file"]).name != row["file"]:
            raise ValueError("Invalid split or unsafe file path")
        identities = [("group", row["source_group"]), ("bytes", row["sha256"])]
        identities += [("numeric_geometry", row["numeric_geometry_sha256"])] \
            if row.get("numeric_geometry_sha256") else []
        identities += [("listing", str(x)) for x in row.get("source_model_ids", [])]
        identities += [("url", x) for x in row.get("source_urls", [])]
        for key, value in identities:
            previous = maps[key].setdefault(value, role)
            if previous != role:
                raise ValueError(f"{key} crosses split boundaries")
    protected = set(manifest["protected_pilot_test_groups"])
    actual = {r["source_group"] for r in rows if r["split"] == "test"}
    if actual != protected:
        raise ValueError("Original evaluation groups must all remain test")
    primary = set(manifest["primary_test_ids"])
    if not primary <= {r["pilot_id"] for r in rows if r["split"] == "test"}:
        raise ValueError("Primary test includes optimization data")
    if set(maps["group"].values()) != {"train", "validation", "test"}:
        raise ValueError("All three splits required")
    return rows


def build(full, pilot):
    protected = {r["source_group"] for r in pilot["files"] if r["role"] == "evaluation"}
    primary = [r["pilot_id"] for r in pilot["files"] if r["role"] == "evaluation"]
    families = defaultdict(list)
    for row in full["files"]:
        if row["source_group"] not in protected:
            families[row["source_group"]].append(row)
    groups = sorted(families)
    n_val = max(1, round(len(groups) * .2))
    bins = Counter(r["complexity_bin"] for g in groups for r in families[g])
    total = sum(bins.values())
    # Balance only reference metadata, never any model outputs. Fixed number of
    # seeded proposals; the criterion is frozen along with the chosen membership.
    rng = random.Random(SEED)
    candidates = []
    for _ in range(4096):
        candidate = sorted(rng.sample(groups, n_val))
        counts = Counter(r["complexity_bin"] for g in candidate for r in families[g])
        score = abs(sum(counts.values()) / total - .2)
        score += sum(abs(counts[b] / bins[b] - .2) for b in bins) / len(bins)
        candidates.append((score, candidate))
    val = set(min(candidates)[1])
    rows = [{**r, "split": "test" if r["source_group"] in protected else
             "validation" if r["source_group"] in val else "train"} for r in full["files"]]
    result = dict(schema_version=1, seed=SEED,
        method="450 original pilot evaluation source groups and all siblings protected; "
               "20% of remaining groups for validation using 4096 seeded metadata-only "
               "proposals minimizing body and face-bin fraction deviation from 0.2.",
        full_manifest_digest=digest(full), pilot_manifest_digest=digest(pilot),
        protected_pilot_test_groups=sorted(protected), primary_test_ids=primary,
        limitation="Pilot test results were previously inspected. This is an optimization "
                   "and checkpoint-selection holdout, not a prospectively unseen research test. "
                   "Known source/representation links are protected; creator/remix/unknown "
                   "near-duplicate separation and upstream pretraining contamination are not guaranteed.",
        counts={s: dict(bodies=sum(r["split"] == s for r in rows),
                       groups=len({r["source_group"] for r in rows if r["split"] == s}),
                       face_bins=dict(Counter(str(r["complexity_bin"]) for r in rows if r["split"] == s)))
                for s in ("train", "validation", "test")},
        dataset_pointer=full["dataset_pointer"], bin_labels=full["bin_labels"], files=rows)
    result["split_sha256"] = digest(result)
    validate(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(read(a.full), read(a.pilot))
    if a.out.exists() and read(a.out) != result:
        raise RuntimeError("Refusing to overwrite a different frozen split")
    save(a.out, result)
    print(json.dumps(dict(split_sha256=result["split_sha256"], counts=result["counts"]), indent=2))
