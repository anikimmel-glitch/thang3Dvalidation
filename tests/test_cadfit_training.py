"""Exercise learned checkpoint retention and telemetry across real optimizer steps."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class FilterTraining(unittest.TestCase):
    def test_training_logs_every_epoch_and_retains_learned_weights(self):
        import numpy as np
        import torch
        # This is required on Sagey; unsupported local Torch/NumPy ABIs fail openly.
        torch.from_numpy(np.zeros(1, dtype=np.float32))
        from baselines.cadfit_filter_train import train
        torch.set_num_threads(1)
        head = lambda arch: torch.nn.Sequential(torch.nn.Linear(arch["D_in"], 1))
        events = []
        run = SimpleNamespace(url="offline:test", id="test",
            log=lambda values, step: events.append((step, values)),
            finish=lambda: events.append("finished"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out/"features").mkdir()
            model = out/"pretrained.pt"
            original = head({"D_in": 6}).state_dict()
            torch.save(dict(arch={"D_in": 6}, state_dict=original), model)
            rows = []
            rng = np.random.default_rng(42)
            for split, count in (("train", 5), ("validation", 3)):
                for i in range(count):
                    dest = out/"features"/f"{split}_{i}.npz"
                    y = np.tile([0, 1], 20)
                    x = rng.normal(size=(len(y), 6)).astype(np.float32)
                    x[:, 0] += y * 2
                    np.savez(dest, x=x, y=y)
                    rows.append(dict(split=split, source_group=f"{split}_{i}",
                                     cache=str(dest)))
            (out/"features/index.json").write_text(json.dumps(rows))
            with patch.dict(sys.modules, {
                    "filter_loops": SimpleNamespace(_build_mlp=head),
                    "wandb": SimpleNamespace(init=lambda **kwargs: run)}), \
                    patch.dict(os.environ, WANDB_API_KEY="test-placeholder"):
                train(SimpleNamespace(model=str(model), out=out))
            self.assertEqual([e[0] for e in events[:-1] if "validation/epoch" in e[1]],
                             list(range(11)))
            self.assertEqual([e[0] for e in events[:-1] if "train/loss" in e[1]],
                             list(range(1, 11)))
            self.assertEqual(events[-1], "finished")
            selection = json.loads((out/"selection_frozen.json").read_text())
            self.assertFalse(selection["test_used"])
            self.assertIn(selection["best_learned_epoch"], range(1, 11))
            learned = torch.load(out/f'filter_epoch_{selection["best_learned_epoch"]}.pt',
                                 weights_only=False)["state_dict"]
            self.assertTrue(any(not torch.equal(original[k], learned[k]) for k in original))
            for epoch in range(11):
                self.assertTrue((out/f"validation_epoch_{epoch}.npz").exists())
                self.assertTrue((out/f"training_state_epoch_{epoch}.pt").exists())


if __name__ == "__main__":
    unittest.main()
