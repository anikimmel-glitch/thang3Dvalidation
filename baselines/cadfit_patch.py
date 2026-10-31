"""Record uncensored candidate IoUs without changing candidate selection."""
import hashlib
from pathlib import Path
from evaluation.common import save

def apply(repo, out):
    path=Path(repo)/"greedy_search.py"
    old=path.read_text()
    anchor="        result['iou'] = iou\n"
    if old.count(anchor)!=1:
        raise RuntimeError("Pinned CADFit candidate instrumentation anchor changed")
    extra="""        # Retain measured candidate evidence before upstream IoU censoring.
        import hashlib as _audit_hash
        _audit_dir = os.path.join(output_folder, 'candidate_measurements')
        os.makedirs(_audit_dir, exist_ok=True)
        _audit_key = _audit_hash.sha256(json.dumps(operation, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        with open(os.path.join(_audit_dir, _audit_key + '.json'), 'w') as _af:
            json.dump({'operation': operation, 'iou': float(iou)}, _af, allow_nan=False)
"""
    new=old.replace(anchor,anchor+extra)
    path.write_text(new)
    save(Path(out)/"candidate_instrumentation.json",dict(
        before_sha256=hashlib.sha256(old.encode()).hexdigest(),
        after_sha256=hashlib.sha256(new.encode()).hexdigest(),
        algorithm_changed=False, records_below_upstream_cutoff=True))
    path=Path(repo)/"run_pipeline.py"
    old=path.read_text()
    anchor='            print(f"⚠️  Sketch filtering failed ({type(e).__name__}: {e}); "\n                  "continuing with unfiltered loops")'
    if old.count(anchor)!=1:
        raise RuntimeError("Pinned filter failure guard changed")
    path.write_text(old.replace(anchor,'            raise RuntimeError("Sketch filter failed; no unfiltered fallback") from e'))
    save(Path(out)/"filter_failure_guard.json",dict(
        silent_unfiltered_fallback_disabled=True,
        before_sha256=hashlib.sha256(old.encode()).hexdigest(),
        after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
