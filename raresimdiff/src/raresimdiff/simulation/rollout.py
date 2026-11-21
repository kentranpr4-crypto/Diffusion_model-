from __future__ import annotations
from typing import Optional
import numpy as np
from ..scenarios.schema import ScenarioSpec
from .base import ACT_DIM, EpisodeResult
from .controllers import Controller
from .mujoco_backend import MuJoCoBackend


def run_episode(backend: MuJoCoBackend, spec: ScenarioSpec, controller: Controller, record_transitions: bool = False,
                max_steps: Optional[int] = None) -> EpisodeResult:
    obs = backend.reset(spec)
    controller.reset()
    traj_every = max(1, int(round(1.0 / backend.cfg.traj_hz / backend.cfg.control_dt)))
    states, acts, rews = [backend.get_state()["flat"]], [np.zeros(ACT_DIM)], [0.0]
    tr = {k: [] for k in ("observation", "action", "reward", "next_observation", "terminated", "truncated")}
    terminated = truncated = False
    n = 0
    while not (terminated or truncated):
        a = controller.act(obs)
        nobs, r, terminated, truncated, info = backend.step(a)
        n += 1
        if record_transitions:
            for k, v in zip(tr, (obs, a, r, nobs, terminated, truncated)):
                tr[k].append(v)
        if n % traj_every == 0 or terminated or truncated:
            states.append(backend.get_state()["flat"])
            acts.append(a)
            rews.append(r)
        obs = nobs
        if max_steps is not None and n >= max_steps and not (terminated or truncated):
            truncated = True
            backend.metrics["termination_reason"] = "max_steps"
    m = backend.get_metrics()
    trans = {k: np.asarray(v) for k, v in tr.items()} if record_transitions else None
    return EpisodeResult(scenario_id=spec.scenario_id, success=bool(m["success"]), terminated=terminated,
                         truncated=truncated, termination_reason=m["termination_reason"], steps=n,
                         sim_time=backend.t, metrics=m, traj_states=np.asarray(states, dtype=np.float32),
                         traj_actions=np.asarray(acts, dtype=np.float32), traj_rewards=np.asarray(rews, dtype=np.float32),
                         transitions=trans)
