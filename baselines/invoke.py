"""Seed an upstream script without editing its released inference implementation."""
import faulthandler
import random
import resource
import runpy
from pathlib import Path
import sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
faulthandler.enable(all_threads=True)
import numpy as np

seed=int(sys.argv[1])
script=Path(sys.argv[2])
random.seed(seed)
np.random.seed(seed)
try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(seed)
sys.path.insert(0,str(script.parent))
sys.argv=[str(script),*sys.argv[3:]]
runpy.run_path(str(script),run_name="__main__")
