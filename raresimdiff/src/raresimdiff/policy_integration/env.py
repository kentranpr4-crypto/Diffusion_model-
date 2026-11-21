"""Gymnasium-style environment over generated / procedural scenarios (no hard gymnasium dependency)."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from ..scenarios import schema as S
from ..simulation.base import ACT_DIM, OBS_DIM
from ..simulation.mujoco_backend import MuJoCoBackend

try:  # optional
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False


class ScenarioEnv:
    """reset(options={'scenario': ScenarioSpec}) or sample from `scenario_fn(rng)`.
    obs: 28-d vector, action: 6-d in [-1,1] (see simulation/controllers.py for the layout)."""
    metadata = {"render_modes": []}

    def __init__(self, scenario_fn: Optional[Callable[[np.random.Generator], S.ScenarioSpec]] = None):
        self.backend = MuJoCoBackend()
        self.scenario_fn = scenario_fn
        self.rng = np.random.default_rng(0)
        self.observation_size, self.action_size = OBS_DIM, ACT_DIM
        if _HAS_GYM:
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float64)
            self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float64)
        self.spec: Optional[S.ScenarioSpec] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        spec = (options or {}).get("scenario") or (self.scenario_fn(self.rng) if self.scenario_fn else None)
        if spec is None:
            raise ValueError("provide options={'scenario': ScenarioSpec} or a scenario_fn")
        self.spec = spec
        obs = self.backend.reset(spec)
        return obs, {"scenario_id": spec.scenario_id, "event_type": S.EVENTS[spec.event], "severity": spec.severity,
                     "is_synthetic": True}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.backend.step(action)
