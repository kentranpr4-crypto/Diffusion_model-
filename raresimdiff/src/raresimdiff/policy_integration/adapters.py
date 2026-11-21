"""Integration points for downstream learners. The generator is NOT a policy; it only supplies scenarios.

Pipeline: Frozen RareSimDiff -> Generated Scenario -> MuJoCo rollout -> (obs, action, reward) -> policy training.

Implemented here: dataset export (exporters.py), a Gymnasium-style env (env.py), evaluation (metrics.py).
Learner adapters below are interface contracts only (no training algorithm is bundled in the MVP).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class LearnerAdapter(ABC):
    """Contract for BC/ACT, Diffusion Policy, PPO, SAC and future world-model planners."""
    name: str = "abstract"
    needs_expert_demos: bool = False
    online: bool = False

    @abstractmethod
    def fit(self, dataset_npz: str, **kw) -> Any: ...

    @abstractmethod
    def act(self, obs) -> Any: ...


def _todo(name: str, needs_demos: bool, online: bool):
    class _A(LearnerAdapter):
        def fit(self, dataset_npz, **kw):
            raise NotImplementedError(f"{name} training is not bundled; consume the exported transitions or ScenarioEnv")

        def act(self, obs):
            raise NotImplementedError

    _A.name, _A.needs_expert_demos, _A.online = name, needs_demos, online
    _A.__name__ = name.replace("-", "_")
    return _A


BehaviorCloningACT = _todo("bc_act", True, False)            # needs genuine expert demos (not provided)
DiffusionPolicyAdapter = _todo("diffusion_policy", True, False)  # needs genuine expert demos (not provided)
PPOAdapter = _todo("ppo", False, True)                       # uses ScenarioEnv online
SACAdapter = _todo("sac", False, True)                       # uses ScenarioEnv online
WorldModelPlannerAdapter = _todo("world_model_planner", False, False)  # future work

REGISTRY: Dict[str, type] = {c.name: c for c in (BehaviorCloningACT, DiffusionPolicyAdapter, PPOAdapter, SACAdapter,
                                                  WorldModelPlannerAdapter)}
