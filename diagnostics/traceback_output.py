"""Keep fatal diagnostics readable without changing CAD processing or its time limit."""
from contextlib import contextmanager
import faulthandler
import json


@contextmanager
def traceback_output(work, fatal_only=False):
    stream = None
    try:
        if fatal_only:
            stream = (work / "fatal_traceback.log").open("w")
            faulthandler.enable(file=stream, all_threads=True)
            policy = dict(fatal_output="fatal_traceback.log",
                          periodic_traceback_enabled=False,
                          process_timeout_enforced_by_parent=True,
                          geometry_operations_changed=False)
        else:
            faulthandler.enable(all_threads=True)
            faulthandler.dump_traceback_later(120, repeat=True)
            policy = dict(fatal_output="stderr", periodic_traceback_enabled=True,
                          periodic_seconds=120, process_timeout_enforced_by_parent=True,
                          geometry_operations_changed=False)
        (work / "traceback_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        if stream is not None:
            stream.close()
