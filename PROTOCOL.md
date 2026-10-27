# Reconstruction pilot protocol

## Cohort and leakage

Freeze all selection decisions before evaluating the 450 held-out subjects.
The 50 development subjects diagnose adapters, rendering, and scoring.
One subject per connected listing/exact-representation-hash group reduces
assembly sibling leakage. It does not establish creator independence, eliminate
all geometric near duplicates, or prove non-overlap with model training data.
Exclude held-out source siblings from any future fine-tuning.

Face bins are 1–15, 16–31, 32–63, 64–127, and 128+; the 500-subject pilot
contains 87, 87, 100, 100, and 126 subjects respectively. Report development
and evaluation separately, including every exclusion and failed preparation.
Pure prisms, cylinders, straight cylindrical rings/pipes, complete spheres,
and confirmed lettering-dominated families were excluded during curation.
Broader functional complexity review remains provisional.

## Input tracks

| Track | Model | Input | Output |
|---|---|---|---|
| Matched PC | CAD-Recode v1.5 | 256 FPS surface points | CadQuery / STEP |
| Matched PC | Cadrille RL | The same 256 points | CadQuery / STEP |
| Shaded views | Cadrille RL | Four 128 px isometric views with borders | CadQuery / STEP |
| Native coordinate views | CADReasoner | Six coordinate-colored orthographic and two shaded views | CadQuery / STEP |

Use one greedy candidate. Never choose a prediction using GT Chamfer distance.
CADReasoner is a first-pass adaptation; iterative self-refinement is an
additional experiment, not an implemented claim. No titles, descriptions,
filenames, face labels, or GT construction programs enter model prompts.
SDPA replaces flash-attention; record this implementation difference.

DualBrep is a useful future direct-BRep/point-normal contrast. Its released
Python 3.12/pythonocc environment and reconstruction/postprocessing pipeline
are not part of the initial runner. CADFS requires Onshape credentials for
STEP export. StepCAD was not executable from the inspected public release.
Do not rank unavailable/incomparable baselines as if they were evaluated.

## Geometry

Verify each input's SHA-256 and byte size. Require one valid reference solid.
For model inputs, center the bounding box and map its longest side to 2.
This canonical benchmark transform does not modify the archived unit-sphere
STEP files. Record the center/scale for every reference.

Tessellate at 0.0005 linear and 0.05 angular deflection in canonical units.
Sample 32,768 surface points for scoring, with an independent prediction seed.
Sample 8,192 surface points then deterministic FPS for 256-point model inputs.
CadQuery model outputs conventionally use ±100 units: divide by 100 for the
native-frame score. Also report independent bbox normalization, explicitly
separating shape similarity from placement/scale correctness.

## Metrics and failure accounting

- Valid single-solid BRep rate.
- Symmetric squared Chamfer **sum**, mean and median conditional on validity.
- Surface F1 at distance 0.02 (1% of canonical reference side).
- Absolute normal consistency.
- Physical sharp-edge F1 from tessellation dihedrals above 20 degrees. This
  measures creases, not CAD topological edge-entity correspondence.
- Native/unified face counts, area by analytic surface type, and triangle count.

Invalid outputs and program execution failures receive zero surface F1.
Missing subjects, preparation/render errors, inference infrastructure failures,
scoring errors, and execution-policy rejections are reported separately.
Report evaluated coverage alongside every aggregate; no incomplete run is a
complete benchmark. Sharp-edge scores are not applicable when the reference
has no detected sharp edges. Native tolerances, watertightness, and BRep validity
are distinct properties and must not be conflated.

Means alone can be dominated by highly fragmented geometry. Report face-bin
results, medians, tails, and sensitivity to predeclared outlier exclusions.
Unifying same-domain faces preserves geometry only when validity and volume
checks pass; it is diagnostic, not ground-truth simplification for the model.

## Planned paper-strengthening experiments

Paired bootstrap confidence intervals by source group; reconstruction performance
versus geometric complexity; controlled point-count/noise and image-resolution
ablations; all-outcome failure taxonomy; blinded human review separating functional
geometry from engraving; and a cross-dataset test under this same evaluator.
The current runner emits per-subject data needed for these analyses. It does not
claim those experiments or human annotations are already complete.
