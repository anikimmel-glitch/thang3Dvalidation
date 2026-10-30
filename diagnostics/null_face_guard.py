"""Apply one pinned upstream safety guard; no tolerance or topology changes."""
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "4857bf4c858e125e1d20ce5cc7b623c6c08c66f4fedd601d20a1733e07ed1804"
MARKER = "[dualbrep-null-face-guard] rejected null internal face before seam repair"
GUARD = (
    "        _manual_fix_face = face_fixer.Face()\n"
    "        if _manual_fix_face is None or _manual_fix_face.IsNull():\n"
    f"            print({MARKER!r}, flush=True)\n"
    "            return None\n"
)
ANCHOR = "        face_fixer.FixOrientation()\n        face_fixer.FixMissingSeam()"


def guarded_source(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Refuse to patch unexpected upstream geometry code")
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    anchor = ANCHOR.replace("\n", newline)
    if text.count(anchor) != 1:
        raise ValueError("Expected exactly one explicit orientation/seam sequence")
    replacement = anchor.replace("        face_fixer.FixMissingSeam()",
                                 GUARD.replace("\n", newline) + "        face_fixer.FixMissingSeam()")
    patched = text.replace(anchor, replacement).encode("utf-8")
    assert patched.replace(GUARD.replace("\n", newline).encode(), b"", 1) == raw
    compile(patched, "brep_post/utils.py", "exec")
    return patched


def apply(repository):
    path = Path(repository) / "brep_post/utils.py"
    before = path.read_bytes()
    after = guarded_source(before)
    temporary = path.with_suffix(".py.tmp")
    temporary.write_bytes(after)
    temporary.replace(path)
    return dict(
        file="brep_post/utils.py", before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(after).hexdigest(), added_code=GUARD,
        policy="Return None via the existing face-construction failure path if the "
        "internal face is null immediately before explicit seam repair.",
        valid_face_operations_unchanged=True, thresholds_unchanged=True,
        result_shape_not_substituted=True, marker=MARKER)
