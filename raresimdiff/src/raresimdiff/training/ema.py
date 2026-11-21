from __future__ import annotations
import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def effective_decay(self, step: int) -> float:
        return min(self.decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        d = self.effective_decay(step)
        ep = [p for p in self.model.parameters()]
        mp = [p for p in model.parameters()]
        torch._foreach_mul_(ep, d)
        torch._foreach_add_(ep, mp, alpha=1.0 - d)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
