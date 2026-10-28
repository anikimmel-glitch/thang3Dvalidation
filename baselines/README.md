# Additional reconstruction baselines

Two complementary recent methods are selected from their official releases:

| Method | Reference | Input | Reconstruction |
| --- | --- | --- | --- |
| DualBrep | [SIGGRAPH 2026](https://arxiv.org/abs/2606.31579), [code](https://github.com/AutodeskAILab/DualBrep) | 32,768 oriented surface points | Point-cloud autoencoder → dual distance fields → segmentation → parametrized/trimmed B-rep |
| CADFit | [2026 paper](https://arxiv.org/abs/2605.01171), [code](https://github.com/ghadinehme/CADFit) | Complete triangulated mesh | IoU-guided sketch/extrusion/revolution and Boolean fitting, with fillet/chamfer refinement |

These extend the representation comparison beyond neural CadQuery prediction.
They **do not have the same information or inference budget** as the 256-point
and rendered-image tracks. Report separate input/budget columns, never an
unqualified modality-matched ranking. Neither receives source text or B-rep face,
edge, topology, or construction-history labels. CADFit optimizes against its
input mesh as part of its released algorithm; this is not fitting to a hidden
evaluation target unavailable at inference.

Code and DualBrep weights are revision-pinned in `lock.json`. Separate upstream
Python/OCC environments preserve the common locked scoring environment.
The host container must provide glibc >= 2.35 (Ubuntu 22.04 or a compatible
newer Linux image). CADFit's pinned PyMeshLab 2025.7 Python 3.10 wheel targets
manylinux_2_35; the DualBrep torch_cluster wheel also needs newer glibc than
Ubuntu 20.04 provides. The extra-baseline bootstrap checks this before installing
dependencies and records the accepted ABI. This changes the host runtime only;
the upstream Python versions, requirements, inputs, and scoring remain fixed.
Resolved dependency versions and checkpoint SHA-256 values are recorded on the
worker. Upstream transitive dependencies are not yet an independently validated
fully hashed lock. Environment checks and ten **training-only** smoke subjects
must pass before attempting the protected 450 pilot evaluation subjects.

DualBrep uses the point-cloud autoencoder, not the STEP/Voronoi autoencoder that
would expose target topology. Input points/normals use an independent sample
from the scoring points. Keep the released 256³ field resolution, 24 rotation
candidates and 200 latent optimization steps. Candidate promotion follows the
upstream sealing/gap rule, never benchmark Chamfer/F1. Its permissive sealing
tolerances and kernel version differ from the code models; common reimport and
single-solid validation are applied, and geometry metrics remain necessary.
The public release does not provide training scripts.

CADFit uses its default one residual iteration with fillet/chamfer refinement
enabled. Mesh preprocessing/decimation/alpha wrapping are part of the upstream
pipeline. The optional learned sketch filter is disabled. This baseline performs
test-time optimization and has no dataset fine-tuning stage.

Both receive a fixed 2,700-second reconstruction budget per subject, with
four concurrent subjects per method on the node. Methods run sequentially.
Retain all statuses and timings; budget exhaustion is reported separately and
also receives zero F1 in the explicit fixed-budget summary. Infrastructure
failures remain visible and are not declared invalid CAD.

Invert normalization using input-derived bounding boxes before computing
native-frame metrics; never align using the prediction's fitted scale/pose.
Also report the independently normalized metric, as in the original protocol.
All per-subject native/normalized Chamfer, surface/crease F1, normal consistency,
validity and prediction complexity records are saved.

Run `bootstrap.py --task extra-baselines --manifest-s3 s3://.../split_manifest.json`.
Artifacts live in `/opt/ml/checkpoints/irl-cad-extra-baselines`. No checkpoint
weights or new dataset geometry are downloaded to the laptop.
