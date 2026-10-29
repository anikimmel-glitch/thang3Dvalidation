"""Instrument the pinned upstream postprocessor without changing its CAD operations."""
import argparse
import faulthandler
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--candidate", help="One suspected next candidate; omit for original serial sequence.")
    args = p.parse_args()
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(120, repeat=True)
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repository))
    spec = importlib.util.spec_from_file_location("diagnostic_postprocess", args.repository / "postprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.build_one
    ordinal = 0

    def instrumented(*call_args, **kwargs):
        nonlocal ordinal
        ordinal += 1
        started = time.monotonic()
        event = dict(ordinal=ordinal, candidate=str(call_args[0]), time=time.time(),
                     maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps(dict(event="candidate_started", **event)), flush=True)
        result = original(*call_args, **kwargs)
        print(json.dumps(dict(event="candidate_finished", **event, result=result,
                              seconds=time.monotonic()-started,
                              ending_maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
              flush=True)
        return result

    module.build_one = instrumented
    if args.candidate:
        candidate = args.input / "tmp" / args.candidate / "post.npz"
        assert candidate.is_file()
        # Match the upstream serial defaults, including its per-candidate random seed 0.
        result = module.build_one(candidate, str(args.repository), 0, False, True, 200, seed=0)
        (args.input / "single_candidate_result.json").write_text(json.dumps(result))
    else:
        sys.argv = [str(args.repository / "postprocess.py"), "--input", str(args.input), "--serial"]
        module.main()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
