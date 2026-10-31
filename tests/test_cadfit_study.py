import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from baselines.cadfit_protocol import proxy_labels,source_balanced
from baselines.cadfit_patch import apply
from finetuning.split import digest

class ProxySupervision(unittest.TestCase):
    def setUp(self):
        self.loops=[{"loops_3d":[[[0,0,0],[1,0,0],[0,0,0]]]}]
        self.ops=[dict(type="extrude",sketch_idx=0,height=h) for h in (1,2)]
    def test_timeout_is_not_negative(self):
        measured=[dict(operation=self.ops[0],iou=0.)]
        self.assertEqual(proxy_labels(self.ops,measured,"extrude",self.loops),{})
    def test_fully_measured_negative(self):
        measured=[dict(operation=op,iou=.0001) for op in self.ops]
        self.assertEqual(proxy_labels(self.ops,measured,"extrude",self.loops),{0:0})
    def test_positive_with_unknown_sibling(self):
        self.assertEqual(proxy_labels(self.ops,[dict(operation=self.ops[0],iou=.2)],
                                      "extrude",self.loops),{0:1})
    def test_intermediate_not_negative(self):
        measured=[dict(operation=op,iou=.004) for op in self.ops]
        self.assertEqual(proxy_labels(self.ops,measured,"extrude",self.loops),{})
    def test_revolve_ids_do_not_collide(self):
        wrong=[dict(operation=dict(type="revolve",sketch_idx=0),iou=.9)]
        self.assertEqual(proxy_labels(self.ops,wrong,"extrude",self.loops),{})
    def test_multiloop_is_ambiguous(self):
        self.loops[0]["loops_3d"]*=2
        self.assertEqual(proxy_labels(self.ops,[dict(operation=self.ops[0],iou=.2)],
                                      "extrude",self.loops),{})
    def test_source_selection_keeps_holdout_out(self):
        rows=[dict(pilot_id=str(i),source_group=str(i//2),faces=i,
                   split="test" if i>=6 else "train") for i in range(8)]
        out=source_balanced(rows,"train",64)
        self.assertEqual([r["pilot_id"] for r in out],["5","3","1"])
        self.assertEqual(len({r["source_group"] for r in out}),len(out))
    def test_instrumentation_rejects_changed_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"greedy_search.py").write_text("unrecognized")
            with self.assertRaises(RuntimeError):apply(p,p/"out")
    def test_no_test_features_loaded_by_training(self):
        # The extractor index is built solely from support rows; the held-out
        # condition names cannot be silently folded into the supervised cache.
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/"baselines/cadfit_study.py").read_text())
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=="support" for t in n.targets)]
        self.assertEqual(len(assignments),1)
        self.assertEqual(ast.unparse(assignments[0].value),
                         "cohorts['training'] + cohorts['validation']")

if __name__=="__main__":unittest.main()
