"""Recognize a fully documented model rejection without concealing missing pipeline output."""
import hashlib
import re


def all_candidates_rejected(root, log, subject, execution):
    if (execution["returncode"] != 0 or execution["signal"] is not None
            or execution["timeout"]):
        return None
    if any(root.rglob("post.npz")):
        return None
    raw = log.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    headers = [line for line in lines if re.match(r"\d+ shape\(s\) x \d+ rotation\(s\)", line)]
    if len(headers) != 1 or not headers[0].startswith(
            "1 shape(s) x 24 rotation(s), latent optimisation 200 steps, weights "):
        return None
    rejected = []
    for line in lines:
        match = re.fullmatch(r"\[([^ ]+) rot(\d{2})\] <2 faces, skipping", line)
        if match:
            rejected.append((match[1], int(match[2])))
    if rejected != [(subject, i) for i in range(24)]:
        return None
    return dict(
        status="all_decoded_candidates_rejected", subject=subject,
        rejected_rotations=list(range(24)), rejected_count=24, expected_count=24,
        reason="Every decoded rotation has fewer than two usable faces.",
        preceding_stage_execution=execution, stage_log=str(log),
        stage_log_sha256=hashlib.sha256(raw).hexdigest(), post_npz_count=0,
        source_geometry_invalidity_proven=False,
        interpretation="Documented model-output rejection; no CAD candidate exists to postprocess.")
