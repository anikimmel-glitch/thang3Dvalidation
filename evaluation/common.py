import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False))
    tmp.replace(p)


def worker(arguments, log, timeout=300, module="evaluation.geometry"):
    # Generated programs run without inherited cloud credentials or tokens.
    env = {k: v for k, v in os.environ.items() if k in
           ("PATH", "LD_LIBRARY_PATH", "CONDA_PREFIX", "DISPLAY", "LANG")}
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    with open(log, "w") as f:
        try:
            r = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=ROOT)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            f.write("\nworker_timeout\n")
            return False


def record_failure(row, destination, kind, detail=""):
    model_failure = kind in ("execution_failed", "invalid_geometry")
    save(destination, dict(pilot_id=row["pilot_id"], role=row["role"],
        complexity_bin=row["complexity_bin"], reference_faces=row["faces"],
        status=kind, detail=detail[:500],
        infrastructure_failure=not model_failure,
        valid_single_solid=False if model_failure else None))


def evaluate_program(row, code, destination, prepared):
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"prediction.py").write_text(code)
    if not worker(["execute", destination/"prediction.py", destination],
                  destination/"execution.log", 120):
        detail = (destination/"execution.log").read_text()[-500:]
        kind = "policy_rejected" if "program_policy_rejection" in detail else "execution_failed"
        record_failure(row, destination/"metrics.json", kind, detail)
        return
    if not worker(["evaluate", prepared/"row.json", prepared,
                   destination/"prediction.step", destination/"metrics.json", .01],
                  destination/"evaluation.log", 300):
        record_failure(row, destination/"metrics.json", "evaluation_failed",
                       (destination/"evaluation.log").read_text()[-500:])


def progress(out, model, rows):
    files = sorted((out/"predictions"/model).glob("*/metrics.json"))
    values = [read(p) for p in files]
    save(out/f"progress_{model}.json", dict(model=model, expected=len(rows),
        recorded=len(values), evaluated=sum(r["status"] == "evaluated" for r in values),
        valid=sum(r.get("valid_single_solid", False) for r in values),
        updated_at=time.time()))
    print(json.dumps(read(out/f"progress_{model}.json")), flush=True)
