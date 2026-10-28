"""Exercise both actual renderer paths on known CAD before loading model weights."""
from pathlib import Path
import os
import sys
from evaluation.common import save, worker


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    def cuda_mappings():
        maps = Path("/proc/self/maps").read_text()
        return sorted({line.split()[-1] for line in maps.splitlines()
                       if any(name in line for name in
                              ("libnvJitLink", "libcusparse", "libcudart", "libcuda.so"))})
    try:
        import torch
    finally:
        save(out / "cuda_loader.json", dict(
            library_search_path=os.environ.get("LD_LIBRARY_PATH", ""),
            loaded_cuda_libraries=cuda_mappings()))
    pinned = Path(sys.prefix) / "lib/python3.11/site-packages/nvidia/nvjitlink/lib"
    nvjitlink = [Path(p) for p in cuda_mappings() if "libnvJitLink" in p]
    if not nvjitlink or any(not p.resolve().is_relative_to(pinned.resolve()) for p in nvjitlink):
        raise RuntimeError("PyTorch loaded nvJitLink outside the locked environment")
    print("CUDA library origin verified against the locked environment", flush=True)
    import cadquery as cq
    from evaluation.adapters import Adapter
    from vendor.cad_recode.model import CADRecode
    from vendor.cadrille.cadrille import Cadrille
    from evaluation.geometry import canonical, mesh
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on the GPU worker")
    for device in range(torch.cuda.device_count()):
        x = torch.ones((16, 16), device=f"cuda:{device}")
        assert (x @ x).sum().item() == 4096
    print("Model imports and CUDA canary passed", flush=True)
    solid = cq.Workplane("XY").box(4, 6, 8).faces(">Z").workplane().hole(1).val()
    shape = canonical(solid)[0]
    mesh(shape).export(out / "canary.stl")
    for kind in ("cadrille", "cadreasoner"):
        if not worker([kind, out / "canary.stl", out / f"{kind}.png"],
                      out / f"{kind}.log", 90, module="evaluation.render"):
            raise RuntimeError(f"{kind} renderer failed: "
                               f"{(out / f'{kind}.log').read_text()[-2000:]}")
        print(f"{kind} renderer canary passed", flush=True)
    save(out / "passed.json", dict(renderers=["cadrille", "cadreasoner"],
         cuda_devices_checked=torch.cuda.device_count(), model_imports_passed=True,
         subject="synthetic bored box; excluded from benchmark scores"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
