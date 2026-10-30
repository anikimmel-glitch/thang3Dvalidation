"""Replay saved train-only DualBrep candidates remotely, preserving exit signals."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.common import read, save
from evaluation.retention import Retention, source_snapshot
from evaluation.startup import temp_access
from baselines.setup import environment

OUT = Path("/opt/ml/checkpoints/dualbrep-native-diagnostic")


def process(command, cwd, log, env, timeout, receipt):
    """Keep the process-group exit and elapsed time even after timeout or a signal."""
    started = time.monotonic()
    with Path(log).open("w") as stream:
        child = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    result = dict(command=list(map(str, command)), returncode=child.returncode,
                  signal=signal.Signals(-child.returncode).name if child.returncode < 0 else None,
                  timeout=timed_out, elapsed_seconds=time.monotonic()-started,
                  timeout_seconds=timeout, log=str(log))
    save(receipt, result)
    print(json.dumps(dict(event="process_finished", **result)), flush=True)
    return result


def acquire(client, hit, path):
    """Download only observed candidate keys, checking identity and a strict byte cap."""
    metadata = client.head_object(Bucket=hit["bucket"], Key=hit["key"])
    assert metadata["ContentLength"] == hit["size"]
    expected_time = datetime.fromisoformat(hit["last_modified"].replace("Z", "+00:00"))
    assert metadata["LastModified"] == expected_time
    response = client.get_object(Bucket=hit["bucket"], Key=hit["key"], IfMatch=metadata["ETag"])
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    try:
        with path.open("wb") as target:
            while chunk := response["Body"].read(1024 * 1024):
                total += len(chunk)
                if total > hit["size"]:
                    raise RuntimeError("Candidate exceeds observed size")
                h.update(chunk)
                target.write(chunk)
    finally:
        response["Body"].close()
    assert total == hit["size"]
    return dict(s3_uri=hit["s3_uri"], bytes=total, sha256=h.hexdigest(),
                etag=metadata["ETag"], last_modified=hit["last_modified"])


def main(capture_seam_state=False, null_face_guard=False):
    assert not (capture_seam_state and null_face_guard)
    started = time.monotonic()
    # Leave time for final receipts before the one-hour diagnostic job limit.
    deadline = started + 3450
    OUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(ROOT, OUT)
    retention = Retention(OUT, interval=120)
    retention.start()
    plan = read(ROOT / "diagnostics/native_plan.json")
    if null_face_guard:
        plan["null_face_guard"] = True
    if capture_seam_state:
        for subject in plan["subjects"]:
            subject["candidates"] = [h for h in subject["candidates"]
                                     if Path(h["relative_key"]).parts[1]
                                     == subject["suspected_next_candidate"]]
            assert len(subject["candidates"]) == 1
        plan["capture_seam_state"] = True
        plan["original_plan_sha256"] = hashlib.sha256(
            (ROOT / "diagnostics/native_plan.json").read_bytes()).hexdigest()
    save(OUT / "plan.json", plan)
    stage = "initializing"
    try:
        assert plan["development_only"] and not plan["benchmark_scores_written"]
        assert {r["part_id"] for r in plan["subjects"]} == {"f00012", "f00122"}
        assert all(r["split"] == "train" for r in plan["subjects"])
        assert sum(h["size"] for r in plan["subjects"] for h in r["candidates"]) < 30_000_000
        stage = "temporary_directory"
        save(OUT / "status.json", dict(stage=stage, time=time.time(), benchmark=False))
        # The image's root-writable /tmp can still reject the _apt sandbox user.
        temp_access("/tmp")
        save(OUT / "temporary_directory.json", dict(
            path="/tmp", sandbox_user="_apt", access_probe_passed=True, time=time.time()))
        repo = Path("/tmp/dualbrep-native-diagnostic/source")
        prefix = Path("/tmp/dualbrep-native-diagnostic/env")
        repo.parent.mkdir(exist_ok=True)
        # Check cross-job S3/KMS access before spending time installing dependencies.
        stage = "acquiring_saved_candidates"
        save(OUT / "status.json", dict(stage=stage, time=time.time(), benchmark=False))
        import boto3
        client = boto3.client("s3")
        transfer_receipts = []
        for subject in plan["subjects"]:
            for hit in subject["candidates"]:
                relative = Path(hit["relative_key"])
                assert not relative.is_absolute() and ".." not in relative.parts
                assert relative.parts[0] == "tmp" and relative.name == "post.npz"
                path = OUT / "inputs" / subject["part_id"] / relative
                transfer_receipts.append(acquire(client, hit, path))
                save(OUT / "input_receipts.json", transfer_receipts)

        def setup_step(name, command, timeout):
            nonlocal stage
            stage = name
            save(OUT / "status.json", dict(stage=stage, time=time.time(), benchmark=False))
            remaining = min(timeout, deadline-time.monotonic()-120)
            if remaining <= 0:
                raise TimeoutError("Diagnostic setup time budget exhausted")
            result = process(command, repo.parent, OUT / (name + ".log"), os.environ.copy(),
                             remaining, OUT / (name + ".execution.json"))
            if result["returncode"] != 0:
                raise RuntimeError(f"{name}: {result}")

        setup_step("apt_update", ["apt-get", "update"], 300)
        setup_step("apt_install", ["apt-get", "install", "-y", "libgl1", "libglu1-mesa", "libxrender1"], 600)
        setup_step("git_init", ["git", "init", repo], 60)
        setup_step("git_fetch", ["git", "-C", repo, "fetch", "--depth", "1",
                                plan["upstream_repository"], plan["upstream_commit"]], 180)
        setup_step("git_checkout", ["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"], 60)
        actual = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        assert actual == plan["upstream_commit"]
        if null_face_guard:
            from diagnostics.null_face_guard import apply
            stage = "apply_null_face_guard"
            save(OUT / "status.json", dict(stage=stage, time=time.time(), benchmark=False))
            save(OUT / "null_face_guard_patch.json", apply(repo))
            shutil.copyfile(repo / "brep_post/utils.py", OUT / "patched_upstream_utils.py")
        requirement = repo / "requirements.txt"
        assert hashlib.sha256(requirement.read_bytes()).hexdigest() == plan["requirements_sha256"]
        conda = shutil.which("conda")
        assert conda, "Conda is required for the same pythonocc build"
        setup_step("conda", [conda, "create", "-y", "-p", prefix, "--override-channels", "-c",
                            "conda-forge", "python=3.12", "pip", "pythonocc-core=8.0.1.1"], 1200)
        python = prefix / "bin/python"
        setup_step("pip", [python, "-m", "pip", "install", "--no-cache-dir", "-r", requirement], 2400)
        setup_step("pip_check", [python, "-m", "pip", "check"], 60)
        setup_step("freeze", [python, "-m", "pip", "freeze"], 60)
        # No model inference or weight acquisition is needed for the saved face/edge grids.
        env = environment(prefix, "3.12")
        env = {k: v for k, v in env.items() if k in
               ("PATH", "LD_LIBRARY_PATH", "CONDA_PREFIX", "DISPLAY", "LANG")}
        env.update(PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        # CPU CAD work: identical upstream --serial settings, without changing CUDA visibility.
        results = []
        modes = (("seam_state",) if capture_seam_state else
                 ("guarded_candidate", "guarded_serial_sequence") if null_face_guard else
                 ("suspected_candidate", "original_serial_sequence"))
        for mode in modes:
            if deadline-time.monotonic() < 90:
                break
            stage = mode
            save(OUT / "status.json", dict(stage=stage, time=time.time(), benchmark=False))

            def replay(subject):
                part = subject["part_id"]
                work = OUT / mode / part
                shutil.copytree(OUT / "inputs" / part, work)
                command = [python, ROOT / "diagnostics/native_worker.py",
                           "--repository", repo, "--input", work, "--seed", str(subject["seed"])]
                isolated = mode in ("suspected_candidate", "seam_state", "guarded_candidate")
                if isolated:
                    command += ["--candidate", subject["suspected_next_candidate"]]
                if mode == "seam_state":
                    command.append("--capture-seam-state")
                timeout = min(600 if isolated else 1800,
                              deadline-time.monotonic()-30)
                if timeout <= 0:
                    raise TimeoutError("Replay deadline exhausted")
                execution = process(command, repo, work / "postprocess.log", env, timeout,
                                    work / "execution.json")
                extra = {}
                if null_face_guard:
                    from diagnostics.null_face_guard import MARKER
                    events, malformed, hits = [], 0, 0
                    with (work / "postprocess.log").open(errors="replace") as stream:
                        for line in stream:
                            hits += MARKER in line
                            token = '{"event": "candidate_'
                            if token in line:
                                try:
                                    event, _ = json.JSONDecoder().raw_decode(line[line.index(token):])
                                    events.append(event)
                                except ValueError:
                                    malformed += 1
                    extra["null_face_guard_hits"] = hits
                    save(work / "candidate_coverage.json", dict(
                        events=events, malformed_event_lines=malformed,
                        null_face_guard_hits=hits, execution=execution,
                        scope="Parsed upstream candidate-return events; not independently "
                              "validated geometry or benchmark scores. Full log retained."))
                return dict(part_id=part, mode=mode, **execution, **extra)

            with ThreadPoolExecutor(max_workers=2) as pool:
                results.extend(pool.map(replay, plan["subjects"]))
            save(OUT / "replay_results.json", results)
        save(OUT / "completed.json", dict(time=time.time(), results=results,
             models_inferred=0, benchmark_scores_written=False, heldout_used=False,
             capture_seam_state=capture_seam_state,
             null_face_guard=null_face_guard,
             interpretation="Diagnostic replay only. Candidate isolation and shorter diagnostic time budgets "
                            "do not replace original benchmark outcomes. A successful isolated candidate "
                            "does not disprove a failure involving accumulated process state."))
        save(OUT / "status.json", dict(stage="completed", time=time.time(), benchmark=False))
    except Exception as exc:
        save(OUT / "failure.json", dict(stage=stage, error=repr(exc),
             traceback=traceback.format_exc(), time=time.time(), benchmark_scores_written=False))
        raise
    finally:
        retention.finish()


if __name__ == "__main__":
    main()
