"""Validate frozen pilot or all-body cohorts without splitting source families."""
from pathlib import Path


def validate_manifest(manifest):
    rows = manifest["files"]
    mode = manifest.get("cohort_mode", "source_distinct_pilot")
    if mode not in ("source_distinct_pilot", "all_curated_bodies"):
        raise ValueError(f"Unknown cohort mode: {mode}")
    unique_keys = ["pilot_id", "sha256", "file"]
    if mode == "source_distinct_pilot":
        unique_keys.append("source_group")
    for key in unique_keys:
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Repeated {key}")
    group_roles = {}
    for row in rows:
        if (Path(row["file"]).name != row["file"]
                or row["role"] not in ("development", "evaluation")):
            raise ValueError("Invalid manifest path or role")
        group = row["source_group"]
        if group in group_roles and group_roles[group] != row["role"]:
            raise ValueError("Source group crosses development/evaluation roles")
        group_roles[group] = row["role"]
    return rows
