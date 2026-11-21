"""Empirical categorical/structural prior (task, slot mask, asset ids, environment features), per event type.

The diffusion model generates the continuous variables; categorical layout is conditioning. For generation
without a user-supplied layout we bootstrap layouts from the TRAIN split (documented limitation: layouts are not novel).
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch

from . import schema as S


class CategoricalPrior:
    def __init__(self, task, event, slot_mask, asset_ids, env):
        self.task = np.asarray(task, np.int64)
        self.event = np.asarray(event, np.int64)
        self.slot_mask = np.asarray(slot_mask, np.float32)
        self.asset_ids = np.asarray(asset_ids, np.int64)
        self.env = np.asarray(env, np.float32)

    @classmethod
    def from_split(cls, sd) -> "CategoricalPrior":
        return cls(sd.task, sd.event, sd.slot_mask, sd.asset_ids, sd.env)

    def supported_events(self):
        return sorted(set(int(e) for e in self.event))

    def sample(self, event: int, n: int, rng: np.random.Generator, task: Optional[int] = None) -> Dict[str, np.ndarray]:
        sel = np.where(self.event == event)[0]
        if task is not None:
            sel = sel[self.task[sel] == task]
        if len(sel) == 0:
            raise ValueError(f"event {S.EVENTS[event]!r} has no support in the training prior")
        pick = rng.choice(sel, size=n, replace=True)
        return {"task": self.task[pick], "event": np.full(n, event, np.int64), "robot": np.zeros(n, np.int64),
                "slot_mask": self.slot_mask[pick], "asset_ids": self.asset_ids[pick], "env": self.env[pick]}

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {"prior.task": torch.from_numpy(self.task), "prior.event": torch.from_numpy(self.event),
                "prior.slot_mask": torch.from_numpy(self.slot_mask), "prior.asset_ids": torch.from_numpy(self.asset_ids),
                "prior.env": torch.from_numpy(self.env)}

    @classmethod
    def from_tensors(cls, t: Dict[str, torch.Tensor]) -> "CategoricalPrior":
        return cls(t["prior.task"].numpy(), t["prior.event"].numpy(), t["prior.slot_mask"].numpy(),
                   t["prior.asset_ids"].numpy(), t["prior.env"].numpy())
