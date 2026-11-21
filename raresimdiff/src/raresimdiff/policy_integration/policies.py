"""Minimal policies used for evaluation plumbing. None of these is an expert demonstrator or a humanoid policy."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from ..simulation.base import ACT_DIM
from ..simulation.controllers import ScriptedBaseline


class Policy(Protocol):
    name: str
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class ScriptedBaselinePolicy:
    """Hand-written baseline (state machine / reactive stop). Used to MEASURE difficulty."""
    name = "scripted_baseline"
    is_expert = False

    def __init__(self):
        self.c = ScriptedBaseline()

    def reset(self):
        self.c.reset()

    def act(self, obs):
        return self.c.act(obs)


class NullPolicy:
    name = "null"
    is_expert = False

    def reset(self): ...

    def act(self, obs):
        return np.zeros(ACT_DIM)


class RandomPolicy:
    name = "random"
    is_expert = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self): ...

    def act(self, obs):
        return self.rng.uniform(-1, 1, ACT_DIM)


POLICIES = {"scripted_baseline": ScriptedBaselinePolicy, "null": NullPolicy, "random": RandomPolicy}
