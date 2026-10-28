"""Run in CADFit's environment to export its saved construction program to STEP."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0,sys.argv[1])
    import cadquery as cq
    from refine_edges import load_solid_from_script
    source,dest=map(Path,sys.argv[2:4])
    obj=load_solid_from_script(str(source))
    if obj is None:
        raise RuntimeError("CADFit did not produce an executable final solid")
    shapes=obj.vals() if isinstance(obj,cq.Workplane) else [obj]
    shape=shapes[0] if len(shapes)==1 else cq.Compound.makeCompound(shapes)
    # Export the raw result even when invalid; common scoring records invalidity.
    cq.exporters.export(shape,str(dest))
