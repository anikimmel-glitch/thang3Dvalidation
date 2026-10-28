"""Fixed training gates, selection rules, and paired held-out analysis."""
from collections import defaultdict, Counter
import random
import statistics

CONFIG = dict(
    method="geometry-verified self-training with decoder query/value LoRA",
    seed=20261028, candidates_per_training_body=4,
    native_surface_f1_min=.8, native_precision_min=.8, native_recall_min=.8,
    native_sharp_f1_min=.5, native_normal_min=.85,
    min_accepted_bodies=20, min_accepted_source_groups=8,
    rank=8, alpha=16, learning_rate=1e-4, epochs=3, accumulation=8,
    max_grad_norm=1., warmup_fraction=.1,
    checkpoint_selection="highest validation equal-source native surface F1; "
                         "baseline is eligible; ties prefer the earlier checkpoint",
    test_policy="evaluate the original 450 pilot test subjects after selection freezes; "
                "all 6306 siblings remain held out",
)


def acceptable(record):
    if record.get("status") != "evaluated" or not record.get("valid_single_solid"):
        return False
    s = record["scores"]["native_frame"]
    return (s["surface_f1_at_1pct_side"] >= CONFIG["native_surface_f1_min"]
            and s["surface_precision"] >= CONFIG["native_precision_min"]
            and s["surface_recall"] >= CONFIG["native_recall_min"]
            and s["absolute_normal_consistency"] >= CONFIG["native_normal_min"]
            and (not s["edge_metric_applicable"] or
                 s["sharp_edge_f1_at_1pct_side"] >= CONFIG["native_sharp_f1_min"]))


def scored(record):
    return record.get("status") in ("evaluated", "execution_failed", "invalid_geometry")


def f1(record, mode="native_frame"):
    return record["scores"][mode]["surface_f1_at_1pct_side"] \
        if record.get("valid_single_solid") else 0.


def selection_score(rows, records):
    # A trial that loses evaluation coverage cannot win by dropping hard cases.
    if set(records) != {r["pilot_id"] for r in rows} or not all(scored(v) for v in records.values()):
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(f1(records[row["pilot_id"]]))
    return statistics.mean(statistics.mean(x) for x in groups.values()) if groups else None


def paired_report(rows, baseline, tuned):
    paired = [r for r in rows if scored(baseline.get(r["pilot_id"], {}))
              and scored(tuned.get(r["pilot_id"], {}))]
    result = dict(expected=len(rows), paired_scored=len(paired),
        baseline_statuses=dict(Counter(baseline.get(r["pilot_id"], {}).get("status", "missing") for r in rows)),
        tuned_statuses=dict(Counter(tuned.get(r["pilot_id"], {}).get("status", "missing") for r in rows)),
        note="Paired subjects only, invalid geometry/execution receive zero F1. "
             "Report coverage alongside scores. One training seed; intervals reflect "
             "source sampling, not training-seed variability.", metrics={})
    def interval(deltas):
        families = defaultdict(list)
        for row, value in zip(paired, deltas):
            families[row["source_group"]].append(value)
        values = [statistics.mean(x) for x in families.values()]
        rng = random.Random(CONFIG["seed"])
        boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        return dict(source_macro_delta=statistics.mean(values),
                    source_bootstrap_95ci=[boot[49], boot[1949]])
    if paired:
        for mode in ("native_frame", "independent_bbox_normalization"):
            a = [f1(baseline[r["pilot_id"]], mode) for r in paired]
            b = [f1(tuned[r["pilot_id"]], mode) for r in paired]
            result["metrics"][mode] = dict(baseline_mean=statistics.mean(a),
                tuned_mean=statistics.mean(b), delta=statistics.mean(y-x for x,y in zip(a,b)),
                **interval([y-x for x,y in zip(a,b)]))
        a = [float(bool(baseline[r["pilot_id"]].get("valid_single_solid"))) for r in paired]
        b = [float(bool(tuned[r["pilot_id"]].get("valid_single_solid"))) for r in paired]
        result["metrics"]["validity"] = dict(baseline=statistics.mean(a), tuned=statistics.mean(b),
                                             **interval([y-x for x,y in zip(a,b)]))
    return result
