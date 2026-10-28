# Preliminary pilot analysis

`evaluation/analyze_pilot.py` reads the frozen pilot manifest, saved per-subject
metrics and preparation receipts, the existing geometric audit, and the saved
aggregate summary. It verifies the common evaluation denominator, source-group
independence within the pilot, geometry hashes, and aggregate agreement before
producing tables' source data and PDF/PNG figures.

```sh
python -m evaluation.analyze_pilot \
  --artifacts /path/to/irl-cad-pilot \
  --manifest /path/to/pilot_manifest.json \
  --geometry /path/to/pilot_geometry_metrics.json \
  --summary /path/to/summary.json \
  --out /path/to/analysis
```

The raw inputs remain in authorized project storage rather than this repository.
The script uses NumPy, SciPy, and Matplotlib. It requires the complete four-track,
500-subject pilot and intentionally refuses differing per-track coverage.
It is not the all-body analysis: full-cohort uncertainty must resample entire
source groups, which may contain multiple correlated bodies.

Statistical conventions:

- Use the 435 prepared evaluation subjects; report the 15 excluded evaluation
  references separately. The 50 development subjects do not enter results.
- Resample source groups within the five frozen face strata, using 2,000
  replicates and seed 20261028. The pilot contains one body per source group.
- Surface F1 includes zero for invalid geometry and model execution failures.
  Chamfer and normals condition on each model's valid predictions; crease F1
  additionally requires applicable reference creases.
- Overall F1 contrasts are paired. Intervals describe this purposive pilot,
  not an unbiased population sample or a controlled cross-dataset experiment.
- Feature regressions are exploratory joint linear associations. The plot
  uses approximate simultaneous 95% intervals over 28 coefficients, based on
  maximum standardized bootstrap deviations. Binary surface carriers do not
  identify CAD operations.
- Source-title keyword groups are overlapping, unvalidated listing labels.
  They must not be presented as human-validated body classes.
- Diagnostic outlier removal never changes primary result membership.

The original pilot was run with `v0.2.4-pilot`. `v0.3.0-full` adds full-cohort
source-role validation and source-weighted summaries without changing the
geometry, input, model, or decoding protocol. Both releases retain reference
preparation failures; same-domain cleanup errors are pipeline exclusions, not
proof that a source shape was invalid.
