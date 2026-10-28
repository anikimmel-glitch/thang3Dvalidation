# Held-out adaptation experiment

This is **geometry-verified self-training**, not supervised learning of the
original CAD construction history. The dataset supplies final STEP solids;
the original CadQuery programs are unavailable.

The frozen split contains 987 training bodies (82 source groups), 244 validation
bodies (20 groups), and 6,306 test bodies (450 groups). Every original pilot
evaluation group and every known sibling remains test. The original 450 pilot
evaluation subjects form the primary paired comparison. Their prior results
were already inspected, so this is an optimization/selection holdout, not a
prospectively unseen research test.

Build `split_manifest.json` using `python -m finetuning.split --full FULL
--pilot PILOT --out OUT`. The manifest is private experiment data supplied to
Sagey separately. Its canonical digest is
`1c0828e0bfc0b1268c255d7076f4578b8d99ad95265bfe3df21797572ca9a438`.
The validator rejects cross-split source groups, known listing URLs/IDs,
byte hashes, and numeric-geometry hashes. This cannot prove all unknown remixes,
near-duplicates, creator overlap, or upstream pretraining contamination absent.

## Fixed procedure

Each of the four existing input tracks is adapted independently, starting from
the original pinned model. Training references alone receive four candidates:
one greedy and three sampled at temperature 0.7/top-p 0.9. Candidate execution
uses the existing bounded CAD worker and program policy. Select the highest
native-frame surface F1 among candidates passing all fixed gates:

- Valid positive-volume single solid.
- Native surface F1, precision, and recall each at least 0.8.
- Absolute normal consistency at least 0.85.
- Native sharp-edge F1 at least 0.5 where reference edges exist.
- Program plus end token fits the same generation budget used at test time.

Keep acceptance/coverage receipts, including rejected bodies and complexity
bins. These checks are approximate fidelity checks, not proof of design intent
or of the original modeling sequence. If fewer than 20 bodies or 8 source
groups pass, stop that track without silently relaxing the gates.

Train FP32 LoRA weights on decoder query/value projections (rank 8, alpha 16);
the pretrained model and geometric encoders remain frozen. Three epochs,
learning rate 1e-4, accumulation 8, gradient norm cap 1, one fixed seed.
Sample source groups equally and then bodies within source groups to avoid
domination by large assemblies. Mask the entire input prompt from the loss.
No model titles, descriptions, filenames, test programs, or test metrics enter
training.

Select from epochs 0–3 using validation equal-source native surface F1. Epoch 0
is the unchanged model. A checkpoint with missing/infrastructure-failed
validation predictions is ineligible; invalid CAD/execution errors score zero.
Ties keep the earlier epoch. Freeze the selected adapter and its digest before
preparing or evaluating test subjects.

Final evaluation uses the same input generation, greedy decoding, token budgets,
CAD checks, and native/independently normalized scores as the raw benchmark.
Run both pretrained and selected models on the same 450 protected subjects.
Save per-subject metrics and a paired comparison with source-bootstrap intervals
and explicit coverage. One training seed does not measure optimization variance.
The remaining protected sibling bodies are retained for later expanded testing.

## Execution

Use `bootstrap.py --task finetuning --manifest-s3 s3://.../split_manifest.json`.
The proven locked runtime and CUDA/render checks remain in force; no additional
Python package is required. Four independent GPU tracks run on the Sagey node.
Each must pass a real model forward/backward/update/reset canary before candidate
generation. No checkpoints or dataset geometry are downloaded to the laptop.

Artifacts under `/opt/ml/checkpoints/irl-cad-finetuning` include the split,
configuration, environment, candidate programs/geometry/metrics, acceptance
coverage, training losses, adapter checkpoints, validation histories, frozen
selection, paired test predictions, and comparison JSON. Training reruns require
a new output folder; implicit resume after partially completed training is
rejected.

Sources: [CAD-Recode](https://github.com/filaPro/cad-recode),
[Cadrille](https://github.com/col14m/cadrille),
[CADReasoner](https://github.com/kulibinai/CADReasoner).
Model code and weight licenses remain those recorded in `models.lock.json`
and `vendor/NOTICE.md`; adaptation does not change dataset licensing.
