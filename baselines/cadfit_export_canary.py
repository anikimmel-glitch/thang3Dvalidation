"""Real-kernel export checks required before any CADFit benchmark work."""
import argparse
from pathlib import Path
import sys
from evaluation.common import save

def main(repo,out):
    import cadquery as cq
    sys.path.insert(0,str(repo))
    from baselines.export_cadfit import export
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,offset,expected in [("overlap",1.,1),("disconnected",5.,2)]:
        source=out/(name+".py");dest=out/(name+".step")
        source.write_text("import cadquery as cq\n"
            "a=cq.Workplane('XY').box(2,2,2)\n"
            f"b=cq.Workplane('XY').box(2,2,2).translate(({offset},0,0))\n"
            "result=a.add(b)\n")
        # Known analytic set union, independently constructed for the canary.
        a=cq.Workplane("XY").box(2,2,2).val()
        b=cq.Workplane("XY").box(2,2,2).translate((offset,0,0)).val()
        mesh=out/(name+".stl")
        cq.exporters.export(a.fuse(b),str(mesh))
        export(source,dest,True,mesh)
        result=cq.importers.importStep(str(dest)).val()
        assert len(result.Solids())==expected
        expected_volume=12. if name=="overlap" else 16.
        assert abs(result.Volume()-expected_volume)<1e-5
        records.append(dict(case=name,solids=expected,volume=result.Volume()))
    save(out/"passed.json",dict(cases=records,components_discarded=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();main(a.repo,a.out)
