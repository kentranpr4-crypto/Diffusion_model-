"""Abstract simulator interface (MuJoCo is the MVP backend; Isaac Sim / Isaac Lab can implement the same API)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..scenarios.schema import ScenarioSpec

OBS_DIM = 28
ACT_DIM = 6


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    scenario_id: str
    success: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    steps: int
    sim_time: float
    metrics: Dict[str, Any]
    traj_states: np.ndarray  # (T, S) at traj_hz
    traj_actions: np.ndarray  # (T, ACT_DIM)
    traj_rewards: np.ndarray
    transitions: Optional[Dict[str, np.ndarray]] = None


class SimulatorBackend(ABC):
    """Contract every backend must satisfy. The simulator is the authoritative source of physical state."""

    @abstractmethod
    def reset(self, scenario: ScenarioSpec) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: ...

    @abstractmethod
    def get_observation(self) -> np.ndarray: ...

    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]: ...

    @abstractmethod
    def get_contacts(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]: ...

    @abstractmethod
    def validate_scenario(self, scenario: ScenarioSpec) -> ValidationResult: ...

    @abstractmethod
    def export_episode(self) -> Dict[str, Any]: ...
