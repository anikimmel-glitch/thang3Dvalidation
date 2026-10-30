"""Observe the unchanged seam call and retain the last face/wires before a crash."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class StateCaptureError(BaseException):
    """An observer failure must escape upstream's CAD exception handlers."""


def capture_occ_state(frame, directory):
    """Read/export state only. Do not heal, validate, or otherwise modify topology."""
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Compound

    local = frame.f_locals
    face = local["face_fixer"].Face()
    surface = local["geom_face"]
    face_is_null = face is None or bool(face.IsNull())
    result = local["face_fixer"].Result()
    result_is_null = result is None or bool(result.IsNull())
    status_path = directory / "status.json"
    inspection = dict(
        capture_sequence=json.loads(status_path.read_text())["sequence"]
        if status_path.exists() else None,
        surface_type=surface.DynamicType().Name(),
        u_periodic=bool(surface.IsUPeriodic()), v_periodic=bool(surface.IsVPeriodic()),
        face_is_null=face_is_null,
        face_wrapper_is_none=face is None,
        face_tolerance=None if face_is_null else float(BRep_Tool.Tolerance(face)),
        result_is_null=result_is_null,
        result_wrapper_is_none=result is None,
        result_shape_type=None if result_is_null else int(result.ShapeType()),
        connected_tolerance=float(local["connected_tolerance"]),
        face_edges=len(local["face_edges"]), original_wires=len(local["wire_list"]),
        fixed_wires=len(local["fixed_wire_list"]),
        inspection_scope="Read-only attributes; no healing or validity test performed.")
    # Retain inspection before any native export, which can itself fail.
    atomic_json(directory / "capture_inspection.json", inspection)
    if face_is_null:
        # BRep_Builder.Add on a null face is unsafe. Let the original seam call
        # run unchanged after recording the null state, without exporting it.
        return dict(**inspection, state_brep=None, bytes=0, sha256=None,
                    export_skipped_reason="null_fixer_face",
                    prior_brep_is_current_state=False)
    # Compound child 0 is the fixer's face; remaining children are its fixed input wires.
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face)
    for wire in local["fixed_wire_list"]:
        builder.Add(compound, wire)
    path = directory / "state_latest.brep"
    temporary = directory / "state_latest.brep.tmp"
    if not breptools.Write(compound, str(temporary)):
        raise RuntimeError("OpenCascade state export returned false")
    digest = hashlib.sha256()
    with temporary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    temporary.replace(path)
    return dict(
        **inspection,
        state_brep=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest(),
        compound_layout="child 0: fixer face; following children: fixed input wires",
        prior_brep_is_current_state=True)


class SeamCapture:
    """Trace one verified source line; never replace or suppress a CAD operation."""

    def __init__(self, source, directory, capture=capture_occ_state):
        self.source = Path(source).resolve()
        self.source_names = {str(Path(source).absolute()), str(self.source)}
        lines = self.source.read_text().splitlines()
        matches = [i + 1 for i, line in enumerate(lines)
                   if line.strip() == "face_fixer.FixMissingSeam()"]
        if len(matches) != 1:
            raise ValueError("Expected exactly one explicit seam-repair operation")
        self.line = matches[0]
        self.source_sha256 = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capture = capture
        self.count = 0
        self.pending = {}
        self.previous_trace = None

    def event(self, **values):
        record = dict(time=time.time(), **values)
        atomic_json(self.directory / "status.json", record)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def trace(self, frame, event, arg):
        if (frame.f_code.co_filename not in self.source_names
                or frame.f_code.co_name != "create_trimmed_face_from_wire"):
            return None
        identity = id(frame)
        if identity in self.pending and event in ("line", "return", "exception"):
            sequence = self.pending.pop(identity)
            self.event(stage="native_operation_raised" if event == "exception"
                       else "native_operation_returned", sequence=sequence)
        if event == "line" and frame.f_lineno == self.line:
            self.count += 1
            record = dict(sequence=self.count, source=str(self.source), line=self.line,
                          source_sha256=self.source_sha256,
                          function=frame.f_code.co_name,
                          operation="face_fixer.FixMissingSeam()",
                          benchmark=False, cad_operations_changed=False)
            self.event(stage="capturing_state", **record)
            (self.directory / "capture_inspection.json").unlink(missing_ok=True)
            try:
                state = self.capture(frame, self.directory)
            except Exception as exc:
                self.event(stage="state_capture_failed", error=repr(exc), **record)
                # Distinguish observer failures from the original kernel crash.
                raise StateCaptureError("CAD observer failed before native seam operation") from exc
            atomic_json(self.directory / "state_latest.json", dict(**record, **state))
            self.event(stage="before_native_operation", **record)
            self.pending[identity] = self.count
        return self.trace

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        if self.previous_trace is not None:
            raise RuntimeError("Refuse to replace an existing Python tracer")
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
