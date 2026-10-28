import copy
import tempfile
from pathlib import Path
import unittest

from finetuning.split import build, digest, validate
from finetuning.protocol import acceptable, selection_score, paired_report


def row(i, group=None):
    return dict(pilot_id=f"p{i}", file=f"{i}.step", sha256=f"hash{i}",
        source_group=group or f"g{i}", numeric_geometry_sha256=f"geometry{i}",
        source_model_ids=[group or f"g{i}"], role="development" if i<20 else "evaluation",
        complexity_bin=i%5)


def metric(value=.9):
    s = dict(surface_f1_at_1pct_side=value, surface_precision=value, surface_recall=value,
        absolute_normal_consistency=.95, edge_metric_applicable=True, sharp_edge_f1_at_1pct_side=.8)
    return dict(status="evaluated", valid_single_solid=True,
                scores=dict(native_frame=s, independent_bbox_normalization=s))


class SplitProtocol(unittest.TestCase):
    def fixture(self):
        pilot = dict(files=[row(i) for i in range(25)])
        full = dict(files=pilot["files"]+[row(30,"g20"),row(31,"g21")],
                    dataset_pointer={},bin_labels=list(range(5)))
        return full,pilot

    def test_frozen_test_preserves_siblings_and_determinism(self):
        full,pilot=self.fixture()
        m=build(full,pilot)
        self.assertEqual(m,build(full,pilot))
        self.assertEqual(m["counts"]["test"]["bodies"],7)
        self.assertEqual({r["split"] for r in m["files"] if r["source_group"]=="g20"},{"test"})
        m["files"][0]["file"]="other.step"
        with self.assertRaisesRegex(ValueError,"digest"): validate(m)

    def test_rejects_cross_split_listing_and_geometry_links(self):
        full,pilot=self.fixture()
        m=build(full,pilot)
        train=next(r for r in m["files"] if r["split"]=="train")
        test=next(r for r in m["files"] if r["split"]=="test")
        for key in ("source_model_ids","numeric_geometry_sha256"):
            bad=copy.deepcopy(m)
            next(r for r in bad["files"] if r["pilot_id"]==train["pilot_id"])[key]=test[key]
            bad["split_sha256"]=digest({k:v for k,v in bad.items() if k!="split_sha256"})
            with self.assertRaisesRegex(ValueError,"crosses"): validate(bad)

    def test_geometry_gate_rejects_scale_and_missing_detail(self):
        self.assertTrue(acceptable(metric()))
        bad=metric(.4)
        bad["scores"]["independent_bbox_normalization"]=metric()["scores"]["native_frame"]
        self.assertFalse(acceptable(bad))
        bad=metric()
        bad["scores"]["native_frame"]["sharp_edge_f1_at_1pct_side"]=.1
        self.assertFalse(acceptable(bad))
        self.assertFalse(acceptable(dict(status="execution_failed")))

    def test_checkpoint_selection_penalizes_invalid_and_disallows_lost_coverage(self):
        rows=[row(1),row(2),row(3,"g1")]
        scores={"p1":metric(1),"p2":metric(.3),"p3":dict(status="invalid_geometry")}
        self.assertAlmostEqual(selection_score(rows,scores),.4)
        scores["p3"]={"status":"inference_runtime_error"}
        self.assertIsNone(selection_score(rows,scores))

    def test_paired_comparison_preserves_failures_and_coverage(self):
        rows=[row(1),row(2),row(3)]
        a={"p1":metric(.4),"p2":dict(status="invalid_geometry"),"p3":dict(status="render_failed")}
        b={"p1":metric(.7),"p2":metric(.2),"p3":metric(1)}
        r=paired_report(rows,a,b)
        self.assertEqual(r["paired_scored"],2)
        self.assertAlmostEqual(r["metrics"]["native_frame"]["delta"],.25)


class AdapterTraining(unittest.TestCase):
    def test_lora_zero_equivalence_updates_and_checkpoint_restore(self):
        import torch
        from torch import nn
        from finetuning.lora import install,state,load
        model=nn.Module()
        model.model=nn.Module()
        model.model.layers=nn.ModuleList([nn.Module()])
        model.model.layers[0].q_proj=nn.Linear(5,3)
        x=torch.randn(2,5)
        baseline=model.model.layers[0].q_proj(x).detach().clone()
        install(model,rank=2,alpha=4)
        original=state(model)
        projection=model.model.layers[0].q_proj
        torch.testing.assert_close(projection(x),baseline)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        projection(x).square().sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(projection(x),baseline))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"adapter.pt"
            torch.save(original,path)
            load(model,torch.load(path,weights_only=True))
        torch.testing.assert_close(projection(x),baseline)
        self.assertFalse(projection.base.weight.requires_grad)

    def test_teacher_forcing_masks_prompt_and_rejects_truncation(self):
        import torch
        from finetuning.train import training_inputs
        class Tokenizer:
            unk_token_id=-1
            def __call__(self,text,**kwargs): return {"input_ids":[7]*len(text)}
            def convert_tokens_to_ids(self,x): return 8
        class Adapter:
            name="cad_recode_pc"
            processor=Tokenizer()
            def inputs(self,prepared):
                return dict(input_ids=torch.tensor([[1,2,3]]),
                            attention_mask=torch.tensor([[-1,-1,1]]))
        values=training_inputs(Adapter(),None,"code")
        self.assertEqual(values["labels"].tolist(),[[-100,-100,-100,7,7,7,7,8]])
        self.assertEqual(values["attention_mask"].tolist(),[[-1,-1,1,1,1,1,1,1]])
        with self.assertRaisesRegex(ValueError,"budget"): training_inputs(Adapter(),None,"x"*768)
