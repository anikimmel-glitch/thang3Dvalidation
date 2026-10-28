"""Input-derived coordinate inversion and validation in the common scoring kernel."""
from pathlib import Path
import sys
import numpy as np
import cadquery as cq
from evaluation.common import read,save

if __name__=="__main__":
    source,dest,transform=map(Path,sys.argv[1:4])
    spec=read(transform)
    shape=cq.importers.importStep(str(source)).val()
    shape=shape.scale(spec["scale"]).translate(tuple(spec["translation"]))
    valid=bool(shape.isValid() and len(shape.Solids())==1 and shape.Volume()>0)
    save(dest.with_suffix(".validity.json"),dict(valid=valid,solids=len(shape.Solids()),
                                                input_derived_transform=spec))
    if valid:
        cq.exporters.export(shape,str(dest))
