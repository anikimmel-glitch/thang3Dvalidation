"""Reproducible, exploratory pilot analysis; does not alter frozen predictions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["cad_recode_pc", "cadrille_pc", "cadrille_rgb", "cadreasoner_native"]
NAMES = ["CAD-Recode / points", "Cadrille / points", "Cadrille / images", "CADReasoner / native"]
COLORS = ["#2463a1", "#008477", "#c46b1e", "#8959a6"]
B = 2000


def read(p):
    return json.loads(Path(p).read_text())


def ci(a):
    return np.quantile(a, [.025, .975], axis=0).tolist()


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    manifest, summary = read(a.manifest), read(a.summary)
    rows = [r for r in manifest["files"] if r["role"] == "evaluation"]
    geom = {r["pilot_id"]: r for r in read(a.geometry)}
    records = {m: {} for m in MODELS}
    prep = {}
    failures = []
    for p in a.artifacts.rglob("metrics.json"):
        model, subject = p.parent.parent.name, p.parent.name
        if model in records:
            if subject in records[model]:
                raise ValueError(f"Duplicate artifact: {model}/{subject}")
            records[model][subject] = read(p)
    for p in a.artifacts.rglob("preparation.json"):
        prep[p.parent.name] = read(p)
    for p in a.artifacts.rglob("preparation_failure.json"):
        failures.append(dict(pilot_id=p.parent.name, **read(p)))
    assert all(len(records[m]) == 500 for m in MODELS), {m: len(v) for m, v in records.items()}

    def scorable(r):
        return r["status"] == "evaluated" or (
            r["status"] in ("invalid_geometry", "execution_failed") and not r.get("infrastructure_failure"))
    included = [{r["pilot_id"] for r in rows if scorable(records[m][r["pilot_id"]])} for m in MODELS]
    assert all(s == included[0] for s in included), "Models have different coverage; paired analysis must change."
    rows = [r for r in rows if r["pilot_id"] in included[0]]
    assert len({r["source_group"] for r in rows}) == len(rows)
    n = len(rows)
    bins = np.array([r["complexity_bin"] for r in rows])
    faces = np.array([r["faces"] for r in rows])
    valid = np.array([[bool(records[m][r["pilot_id"]].get("valid_single_solid")) for m in MODELS] for r in rows])
    score = np.array([[records[m][r["pilot_id"]].get("scores", {}).get("native_frame", {}).get(
        "surface_f1_at_1pct_side", 0.) if valid[i, j] else 0.
        for j, m in enumerate(MODELS)] for i, r in enumerate(rows)])
    norm = np.array([[records[m][r["pilot_id"]].get("scores", {}).get("independent_bbox_normalization", {}).get(
        "surface_f1_at_1pct_side", 0.) if valid[i, j] else 0.
        for j, m in enumerate(MODELS)] for i, r in enumerate(rows)])
    for j, m in enumerate(MODELS):
        assert np.isclose(score[:, j].mean(), summary[m]["evaluation"]["scores"]["native_frame"]["surface_f1_mean"])
        assert int(valid[:, j].sum()) == summary[m]["evaluation"]["valid"]
    rng = np.random.default_rng(20261028)
    parts = [np.flatnonzero(bins == k) for k in range(5)]
    # One body per source group: resample groups within frozen complexity strata.
    boots = np.concatenate([rng.choice(ids, size=(B, len(ids))) for ids in parts], axis=1)
    boot_f1, boot_valid = score[boots].mean(axis=1), valid[boots].mean(axis=1)
    overall, by_bin = {}, {}
    for j, m in enumerate(MODELS):
        rr = [records[m][r["pilot_id"]] for i, r in enumerate(rows) if valid[i, j]]
        native = [x["scores"]["native_frame"] for x in rr]
        edge = [s["sharp_edge_f1_at_1pct_side"] for s in native if s["edge_metric_applicable"]]
        overall[m] = dict(n=n, valid=int(valid[:, j].sum()), validity=valid[:, j].mean(),
            validity_ci=ci(boot_valid[:, j]), f1=score[:, j].mean(), f1_ci=ci(boot_f1[:, j]),
            normalized_f1=norm[:, j].mean(), valid_only_f1=score[valid[:, j], j].mean(),
            chamfer_valid_mean=np.mean([x["squared_chamfer_sum"] for x in native]),
            chamfer_valid_median=np.median([x["squared_chamfer_sum"] for x in native]),
            normal_valid_mean=np.mean([x["absolute_normal_consistency"] for x in native]),
            crease_valid_mean=np.mean(edge), crease_valid_n=len(edge),
            face_f1_spearman=float(spearmanr(faces, score[:, j]).statistic),
            median_predicted_to_reference_faces=float(np.median([
                records[m][r["pilot_id"]]["predicted_faces"] / r["faces"] for i, r in enumerate(rows) if valid[i, j]])))
        ranked = sorted(rr, key=lambda r: r["scores"]["native_frame"]["squared_chamfer_sum"], reverse=True)
        cd = np.array([r["scores"]["native_frame"]["squared_chamfer_sum"] for r in ranked])
        overall[m]["chamfer_sensitivity"] = dict(largest_subject=ranked[0]["pilot_id"],
            largest=float(cd[0]), largest_share_of_sum=float(cd[0]/cd.sum()),
            mean_without_largest=float(cd[1:].mean()), median_without_largest=float(np.median(cd[1:])),
            note="Diagnostic sensitivity only; no result removed from the reported aggregates.")
    bin_boots = []
    for k, ids in enumerate(parts):
        ix = rng.choice(ids, size=(B, len(ids)))
        bin_boots.append(score[ix].mean(axis=1))
        by_bin[manifest["bin_labels"][k]] = dict(n=len(ids),
            expected=sum(r["role"] == "evaluation" and r["complexity_bin"] == k for r in manifest["files"]),
            f1=score[ids].mean(axis=0).tolist(), f1_ci=ci(score[ix].mean(axis=1)),
            validity=valid[ids].mean(axis=0).tolist(), validity_ci=ci(valid[ix].mean(axis=1)),
            valid_only_f1=[float(score[ids, j][valid[ids, j]].mean()) for j in range(4)])
    contrasts = {}
    for j, m in enumerate(MODELS):
        delta = bin_boots[4][:, j] - bin_boots[0][:, j]
        contrasts[m] = dict(high_minus_low_f1=float(score[parts[4], j].mean()-score[parts[0], j].mean()),
                            ci=ci(delta))
    paired = {f"{MODELS[j]}_minus_{MODELS[i]}": dict(
        difference=float((score[:, j]-score[:, i]).mean()), ci=ci(boot_f1[:, j]-boot_f1[:, i]))
        for i, j in [(0, 1), (2, 1)]}
    feature_names = ["Log2 native faces (1 SD)", "B-spline surface present", "Toroidal surface present",
                     "Conical surface present", "Spherical surface present", "Nonplanar area (1 SD)",
                     "Log(1 + extra wires) (1 SD)"]
    matrix, subjects = [], []
    for i, row in enumerate(rows):
        g = geom[row["pilot_id"]]
        assert row["sha256"] == g["sha256"]
        surf = g["surface_counts"]
        area = sum(g["surface_areas"].values())
        nonplanar = 1 - g["planar_area"] / area
        matrix.append([np.log2(row["faces"]), float(surf.get("bspline", 0) > 0),
                       float(surf.get("torus", 0) > 0), float(surf.get("cone", 0) > 0),
                       float(surf.get("sphere", 0) > 0), nonplanar,
                       np.log1p(g.get("extra_face_wires", 0))])
        subjects.append(dict(pilot_id=row["pilot_id"], source_group=row["source_group"],
            faces=row["faces"], unified_faces=prep.get(row["pilot_id"], {}).get("unified_faces"),
            source_titles=row.get("source_titles"), source_urls=row["source_urls"],
            surface_counts=surf, nonplanar_area=nonplanar, extra_wires=g.get("extra_face_wires", 0),
            f1=dict(zip(MODELS, score[i].tolist())), valid=dict(zip(MODELS, valid[i].tolist()))))
    x = np.array(matrix)
    scales = {}
    for j in [0, 5, 6]:
        scales[feature_names[j]] = dict(mean=float(x[:, j].mean()), sd=float(x[:, j].std()))
        x[:, j] = (x[:, j] - x[:, j].mean()) / x[:, j].std()
    x = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(x, score, rcond=None)[0]
    beta = np.array([np.linalg.lstsq(x[idx], score[idx], rcond=None)[0] for idx in boots])
    se = beta[:, 1:].std(axis=0, ddof=1)
    max_deviation = np.max(np.abs((beta[:, 1:] - coef[None, 1:]) / se), axis=(1, 2))
    simultaneous_critical = float(np.quantile(max_deviation, .95))
    simultaneous = np.array([coef[1:] - simultaneous_critical * se,
                             coef[1:] + simultaneous_critical * se])
    regression = dict(features=feature_names, scaling=scales,
        coefficients=coef[1:].tolist(), coefficient_ci=ci(beta[:, 1:]),
        simultaneous_coefficient_ci=simultaneous.tolist(),
        simultaneous_critical=simultaneous_critical,
        binary_feature_counts={feature_names[j]: int(x[:, j+1].sum()) for j in [1, 2, 3, 4]},
        design_condition_number=float(np.linalg.cond(x)),
        interpretation="Exploratory joint OLS associations, not causal effects. The plot uses approximate simultaneous 95% bootstrap max-standardized-deviation intervals over all 28 coefficients; pointwise percentile intervals are also retained.")
    sensitivity = {}
    unified = np.array([prep[r["pilot_id"]].get("unified_faces") or np.nan for r in rows])
    ok = np.isfinite(unified)
    for j, model in enumerate(MODELS):
        sensitivity[model] = dict(
            unified_n=int(ok.sum()),
            native_face_f1_spearman=float(spearmanr(faces[ok], score[ok, j]).statistic),
            unified_face_f1_spearman=float(spearmanr(unified[ok], score[ok, j]).statistic))
    # Source-title groups are weak labels for listings, never validated body classes.
    patterns = {"Mount / bracket / adapter": r"\b(mount|bracket|adapter)\b",
                "Gear / turbine / impeller": r"\b(gear|turbine|impeller)\b",
                "Case / enclosure / housing": r"\b(case|enclosure|housing)\b",
                "Figurine / animal / character": r"\b(figurine|animal|dragon|dog|cat|pokemon|batman|ironman)\b"}
    classes = {}
    for label, pattern in patterns.items():
        mask = np.array([bool(re.search(pattern, " ".join(t or "" for t in row.get("source_titles", [])), re.I))
                         for row in rows])
        if mask.sum() >= 10:
            ids = np.flatnonzero(mask)
            ix = rng.choice(ids, size=(B, len(ids)))
            classes[label] = dict(n=int(mask.sum()), median_faces=float(np.median(faces[mask])),
                f1=score[mask].mean(axis=0).tolist(), f1_ci=ci(score[ix].mean(axis=1)),
                label_scope="Regex over creator listing titles; mixed assembly bodies; exploratory, overlapping, not validated body semantics.")
    result = dict(method=dict(bootstrap_replicates=B, seed=20261028, evaluation_expected=450,
        evaluated_common=n, resampling="Source groups within frozen face strata; one body per source in pilot.",
        intervals="95% percentile intervals, conditional on prepared evaluation subjects and purposive pilot; exploratory, no multiplicity adjustment.",
        models=MODELS), overall=overall, by_complexity=by_bin, high_minus_low=contrasts,
        paired_contrasts=paired, feature_regression=regression, unification_sensitivity=sensitivity,
        source_title_groups=classes, preparation_failures=failures,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.manifest, a.geometry, a.summary]})
    save(a.out/"pilot_analysis.json", result)
    save(a.out/"evaluation_subjects.json", subjects)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
        "axes.spines.right":False, "pdf.fonttype":42, "ps.fonttype":42})
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.65))
    for j, name in enumerate(NAMES):
        for ax, key, interval in zip(axs, ["f1", "validity"], ["f1_ci", "validity_ci"]):
            values=np.array([v[key][j] for v in by_bin.values()])
            bounds=np.array([[v[interval][0][j], v[interval][1][j]] for v in by_bin.values()])
            xx=np.arange(5)+(j-1.5)*.035
            ax.errorbar(xx,values,yerr=[values-bounds[:,0],bounds[:,1]-values],
                color=COLORS[j], marker=["o","s","^","D"][j], ms=3.5, lw=1.3, capsize=2, label=name)
            ax.set_xticks(range(5),[f"{label}\n(n={v['n']})" for label,v in by_bin.items()],fontsize=8)
            ax.set_ylim(0,1);ax.grid(axis="y",alpha=.18);ax.set_xlabel("Native BRep faces")
    axs[0].set_ylabel("Mean surface F1")
    axs[1].set_ylabel("Valid single-solid fraction")
    axs[0].set_title("(a) Geometric accuracy");axs[1].set_title("(b) Executability and validity")
    handles, labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.02),ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.14,1,1])
    for ext in ("pdf","png"):fig.savefig(a.out/f"pilot_complexity.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,3.1))
    bounds=np.array(regression["simultaneous_coefficient_ci"])
    for j,name in enumerate(NAMES):
        values=coef[1:,j]; y=np.arange(7)+(j-1.5)*.16
        ax.errorbar(values,y,xerr=[values-bounds[0,:,j],bounds[1,:,j]-values],
                    fmt=["o","s","^","D"][j],color=COLORS[j],ms=3,capthick=.6,capsize=1.4,lw=.8,label=name)
    ax.axvline(0,color=".4",lw=.8);ax.set_yticks(range(7),feature_names,fontsize=8)
    ax.invert_yaxis();ax.set_xlabel("Adjusted F1 association (simultaneous 95% bootstrap interval)")
    ax.grid(axis="x",alpha=.18);ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(.57,-.48),fontsize=8,frameon=False)
    fig.tight_layout()
    for ext in ("pdf","png"):fig.savefig(a.out/f"pilot_feature_associations.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    for j,name in enumerate(NAMES):
        values=np.array([v["f1"][j] for v in classes.values()])
        bounds=np.array([[v["f1_ci"][0][j],v["f1_ci"][1][j]] for v in classes.values()])
        xx=np.arange(len(classes))+(j-1.5)*.14
        ax.errorbar(xx,values,yerr=[values-bounds[:,0],bounds[:,1]-values],fmt=["o","s","^","D"][j],
                    color=COLORS[j],ms=4,capsize=2,label=name)
    ax.set_xticks(range(len(classes)),[label.replace(" / "," /\n",1)+f"\nn={v['n']}, median faces={v['median_faces']:g}"
                                      for label,v in classes.items()],fontsize=8)
    ax.set_ylim(0,1);ax.set_ylabel("Mean surface F1 (failures = 0)")
    ax.grid(axis="y",alpha=.18)
    ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.5,1.32),fontsize=8,frameon=False)
    fig.tight_layout()
    for ext in ("pdf","png"):fig.savefig(a.out/f"pilot_source_title_groups.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(overall=overall, high_minus_low=contrasts,paired_contrasts=paired,
                         regression=regression,unification=sensitivity,classes=classes),indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("artifacts","manifest","geometry","summary","out"):
        p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
