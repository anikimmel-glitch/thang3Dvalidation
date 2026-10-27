# Thang3D validation

Evaluation code for a source-separated pilot of real-world CAD reconstruction.
This repository contains code, pinned model metadata, and protocol documentation.
Geometry, private corpus manifests, and model weights are supplied separately.

## Status

The initial implementation includes CAD-Recode v1.5 and Cadrille RL point-cloud
adapters, Cadrille shaded-view inference, and CADReasoner native-view first-pass
inference. **No reconstruction scores are claimed by the repository itself.**
Run receipts and measured metrics establish experimental completion.

The frozen pilot has 500 source/representation-hash groups: 50 development and
450 evaluation subjects. Its purposive face-complexity sampling is a stress test,
not a population estimate. Independent human caption validation remains pending.

## Run

Install the versions in `requirements.txt` in a disposable Linux GPU environment.
Image tracks require Xvfb/OpenGL. Sagey uses `bootstrap.py` to install these
dependencies remotely and run the tests before inference.

```bash
python -m unittest discover -s tests -v
python -m evaluation.run --manifest /path/to/pilot.json \
  --input /path/to/step-folder --out /path/to/results \
  --models cad_recode_pc cadrille_pc
python -m evaluation.aggregate --manifest /path/to/results/run_manifest.json \
  --out /path/to/results
```

Omit `--models` to run all four tracks. `--limit 10` selects ten development
subjects across complexity bins for a smoke run. It never substitutes these for
the 450 evaluation subjects. Existing per-subject metric files are reused; use a
new output directory when changing source, checkpoints, or protocol.

`--smoke-first` runs ten development subjects first, then proceeds to all 500
only when every track has complete smoke coverage and at least one valid
prediction. The smoke results are retained and reused. Infrastructure failures
or a failed gate stop the job with diagnostic records.

## Sagey

Use `sagey.example.toml`, replacing the dataset and manifest URIs with your
authorized storage. `sagey submit` performs preflight; `sagey submit --go` submits.
The example sets only a cluster, code source, input, entrypoint, runtime bound,
and required project tags. It does not override instance type, node count,
account, region, or GPU environment.

The public repository can be cloned without a GitHub token. Submit an immutable
release tag and retain the resolved source commit. The private manifest is
downloaded by `bootstrap.py` from S3 on the worker. Results are written to
`/opt/ml/checkpoints/irl-cad-pilot` for Sagey retention. Model weights and STEP
inputs are downloaded/staged only on the remote worker.

## Comparisons and limitations

See [PROTOCOL.md](PROTOCOL.md) for metrics, exclusions, failure accounting, and
input differences. CADReasoner uses coordinate-colored views and a single greedy
first-pass adaptation; it is not a reproduction of its full iterative/oracle
evaluation. Cadrille and CAD-Recode use identical 256-point inputs.

Generated CadQuery is checked by a conservative AST policy and executed in a
subprocess with a timeout and a reduced environment. **This is not a security
sandbox.** Use disposable evaluation infrastructure without user credentials.
The policy can reject some valid programs; those rejections are recorded
separately from model geometry failures.

## Source attribution

Model repositories, checkpoint revisions and model-card terms are recorded in
`models.lock.json`. The authors' source licenses are retained under `vendor/`;
see [vendor/NOTICE.md](vendor/NOTICE.md). Weight licenses can differ from code
licenses. This repository grants no rights to redistribute source CAD models.
Creator/model links and license-review records belong in the private provenance
manifest until a dataset release has been reviewed.
