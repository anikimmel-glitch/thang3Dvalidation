"""Sagey entrypoint with retained setup diagnostics and runtime checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse
from evaluation.startup import runtime_environment, temp_access, wait_display
from evaluation.retention import Retention, source_snapshot

root = Path(__file__).resolve().parent


def main():
    arguments = sys.argv[1:]
    task = "evaluation.run"
    if "--task" in arguments:
        index = arguments.index("--task")
        selected = arguments[index+1]
        if selected not in ("finetuning", "extra-baselines"):
            raise ValueError("Unknown bootstrap task")
        task = {"finetuning": "finetuning.train", "extra-baselines": "baselines.run"}[selected]
        del arguments[index:index+2]
    out = Path("/opt/ml/checkpoints") / {
        "finetuning.train": "irl-cad-finetuning",
        "baselines.run": "irl-cad-extra-baselines",
        "evaluation.run": "irl-cad-pilot"}[task]
    out.mkdir(parents=True, exist_ok=True)
    source_snapshot(root, out)
    retention = Retention(out)
    retention.start()
    stage = "initializing"
    xvfb = None
    display_log = None

    def run_stage(name, command, timeout=None, env=None):
        nonlocal stage
        stage = name
        print(f"[pilot-bootstrap] {name}", flush=True)
        with (out / f"bootstrap_{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        # Keep complete files in checkpoints and the useful tail in CloudWatch.
        print((out / f"bootstrap_{name}.log").read_text(errors="replace")[-6000:],
              flush=True)
        result.check_returncode()

    try:
        stage = "temporary_directory"
        temp_access("/tmp")
        # Test the real APT user and install system libraries before large wheels.
        run_stage("apt_update", ["apt-get", "update"], 300)
        run_stage("apt_install", ["apt-get", "install", "-y", "xvfb", "libgl1",
                  "libglu1-mesa", "libxrender1", "libosmesa6"], 600)

        stage = "manifest"
        if "--manifest-s3" in arguments:
            import boto3
            i = arguments.index("--manifest-s3")
            uri = urlparse(arguments[i + 1])
            if uri.scheme != "s3" or not uri.netloc or not uri.path:
                raise ValueError("Expected a manifest S3 URI")
            manifest = Path("/tmp/irl-cad-run-manifest.json")
            boto3.client("s3").download_file(uri.netloc, uri.path.lstrip("/"), str(manifest))
            arguments[i:i + 2] = ["--manifest", str(manifest)]

        conda = shutil.which("conda")
        if not conda:
            raise RuntimeError("Sagey image needs conda")
        env = Path("/tmp/irl-cad-eval-env")
        python = env / "bin/python"
        if not python.exists():
            run_stage("conda", [conda, "create", "-y", "-p", str(env),
                      "--override-channels", "-c", "conda-forge", "python=3.11", "pip"], 1200)
        run_stage("pip_install", [str(python), "-m", "pip", "install", "--no-cache-dir",
                  "--only-binary=:all:", "--require-hashes", "-r",
                  str(root / "requirements-linux.lock.txt")], 2400)
        run_stage("pip_check", [str(python), "-m", "pip", "check"], 60)
        run_env, cuda_paths = runtime_environment(env, os.environ)
        run_env["PYTHONUNBUFFERED"] = "1"
        (out / "cuda_library_paths.json").write_text(json.dumps(cuda_paths, indent=2))
        print(json.dumps(dict(stage="cuda_library_paths", **cuda_paths)), flush=True)
        run_stage("cad_kernel", [str(python), "-c",
                  "import cadquery as cq; import OCP; "
                  "s=cq.Workplane('XY').box(2,3,4).faces('>Z').workplane().hole(1).val(); "
                  "assert s.isValid() and len(s.Solids())==1; "
                  "print('CAD kernel import and solid construction passed',flush=True)"],
                  120, run_env)

        stage = "display"
        display = ":97"
        display_log = (out / "xvfb.log").open("w")
        xvfb = subprocess.Popen(["Xvfb", display, "-screen", "0", "1024x1024x24",
                                 "-nolisten", "tcp"], stdout=display_log,
                                stderr=subprocess.STDOUT)
        wait_display(xvfb, display)
        run_env["DISPLAY"] = display
        with (out / "environment.txt").open("w") as stream:
            subprocess.run([str(python), "-m", "pip", "freeze"],
                           stdout=stream, check=True, timeout=60)
        run_stage("tests", [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
                  600, run_env)
        # Exercise both real rendering paths before model checkpoint downloads.
        run_stage("render_canary", [str(python), "-m", "evaluation.runtime_canary",
                                   str(out / "runtime_canary")], 300, run_env)
        (out / "bootstrap_passed.json").write_text(json.dumps(
            dict(time=time.time(), checks=["apt", "dependencies", "CAD", "tests", "renderers"]),
            indent=2))
        stage = task
        print(f"[pilot-bootstrap] setup passed; starting {task}", flush=True)
        subprocess.run([str(python), "-m", task, *arguments],
                       cwd=root, env=run_env, check=True)
    except Exception as error:
        report = dict(stage=stage, error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc(), time=time.time(),
                      tmp_mode=oct(Path("/tmp").stat().st_mode & 0o7777),
                      tmp_free_bytes=shutil.disk_usage("/tmp").free)
        (out / "bootstrap_failure.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        raise
    finally:
        retention.finish()
        if xvfb is not None:
            xvfb.terminate()
            try:
                xvfb.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb.kill()
                xvfb.wait()
        if display_log is not None:
            display_log.close()


if __name__ == "__main__":
    main()
