"""Minimal LoRA for decoder query/value projections; no new runtime dependencies."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features,
                                  device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank,
                                  device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        # Keep optimizer state in FP32, compute the low-rank path in the activation dtype.
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + update * self.scale


def install(model, rank=8, alpha=16):
    model.requires_grad_(False)
    names = []
    for name, module in list(model.named_modules()):
        if (name.startswith("model.layers.") and name.endswith((".q_proj", ".v_proj"))
                and isinstance(module, nn.Linear)):
            parent_name, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child, LoRALinear(module, rank, alpha))
            names.append(name)
    if not names:
        raise RuntimeError("No decoder query/value projections found")
    return names


def state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load(model, values):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(values) != expected:
        raise ValueError("Adapter checkpoint parameter mismatch")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in values:
                p.copy_(values[n])
