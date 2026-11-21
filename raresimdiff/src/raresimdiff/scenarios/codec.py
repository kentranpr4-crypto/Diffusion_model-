"""Spec <-> tensor codec with explicit normalisation and inverse transforms.

Physical values --(bounds)--> u in [-1,1] --(train-split mean/std)--> model space x.
Inverse transforms are provided; denormalisation can optionally skip clamping so that the amount
of constraint repair applied to a generated sample can be measured.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from . import schema as S
from .severity import compute_severity


# --- spec -> arrays ---------------------------------------------------------------------------
def pack_spec(spec: S.ScenarioSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phys[L,D] in physical units, slot_mask[L], asset_ids[K_OBJ])."""
    phys = S.CH_NOM.copy()
    mask = np.zeros(S.L)
    mask[S.SLOT_GLOBAL] = 1.0
    for v in S.GLOBAL_VARS:
        phys[S.SLOT_GLOBAL, S.GLOBAL_IDX[v.name]] = spec.glob[v.name]
    asset_ids = np.zeros(S.K_OBJ, dtype=np.int64)
    for o in spec.objects:
        s = S.OBJ_SLOTS[o.slot]
        mask[s] = 1.0
        asset_ids[o.slot] = o.asset_id
        R = np.asarray(o.rot).reshape(3, 3)
        r6 = S.matrix_to_rot6(R)
        row = {"x": o.x, "y": o.y, "dz": o.dz, "scale": o.scale, "mass": o.mass, "friction": o.friction}
        for k, n in enumerate(("r0x", "r0y", "r0z", "r1x", "r1y", "r1z")):
            row[n] = r6[k]
        for v in S.OBJ_VARS:
            phys[s, S.OBJ_IDX[v.name]] = row[v.name]
    for ob in spec.obstacles:
        s = S.OBS_SLOTS[ob.slot]
        mask[s] = 1.0
        row = {"x0": ob.x0, "y0": ob.y0, "vx": ob.vx, "vy": ob.vy, "half_size": ob.half_size, "delay": ob.delay}
        for v in S.OBS_VARS:
            phys[s, S.OBS_IDX[v.name]] = row[v.name]
    return phys, mask, asset_ids


def env_vector(env: Dict[str, float]) -> np.ndarray:
    return np.array([env[v.name] for v in S.ENV_VARS], dtype=np.float64)


def unpack_to_spec(phys: np.ndarray, mask: np.ndarray, asset_ids: np.ndarray, task: int, event: int,
                   robot_profile: int, env: Dict[str, float], scenario_id: str, parent_id: str, seed: int,
                   provenance: Optional[dict] = None, recompute_severity: bool = True,
                   severity: float = 0.0) -> S.ScenarioSpec:
    glob = {v.name: float(phys[S.SLOT_GLOBAL, S.GLOBAL_IDX[v.name]]) for v in S.GLOBAL_VARS}
    objects: List[S.ObjectSpec] = []
    for k, s in enumerate(S.OBJ_SLOTS):
        if mask[s] < 0.5 or int(asset_ids[k]) == 0:
            continue
        r = {v.name: float(phys[s, S.OBJ_IDX[v.name]]) for v in S.OBJ_VARS}
        R = S.rot6_to_matrix(np.array([r["r0x"], r["r0y"], r["r0z"], r["r1x"], r["r1y"], r["r1z"]]))
        objects.append(S.ObjectSpec(slot=k, asset_id=int(asset_ids[k]), x=r["x"], y=r["y"], dz=r["dz"],
                                    rot=[float(a) for a in R.reshape(-1)], scale=r["scale"], mass=r["mass"],
                                    friction=r["friction"]))
    obstacles: List[S.ObstacleSpec] = []
    for k, s in enumerate(S.OBS_SLOTS):
        if mask[s] < 0.5:
            continue
        r = {v.name: float(phys[s, S.OBS_IDX[v.name]]) for v in S.OBS_VARS}
        obstacles.append(S.ObstacleSpec(slot=k, **r))
    if recompute_severity:
        severity = compute_severity(event, glob, [o.__dict__ for o in obstacles])
    return S.ScenarioSpec(scenario_id=scenario_id, parent_id=parent_id, seed=int(seed), task=int(task),
                          event=int(event), robot_profile=int(robot_profile), severity=float(severity),
                          env={k: float(v) for k, v in env.items()}, glob=glob, objects=objects,
                          obstacles=obstacles, provenance=provenance or {})


# --- normalisation -----------------------------------------------------------------------------
def to_unit(phys: np.ndarray) -> np.ndarray:
    """Physical -> [-1,1] via declared bounds (pad channels -> 0)."""
    u = 2.0 * (phys - S.CH_LO) / (S.CH_HI - S.CH_LO) - 1.0
    return u * S.CH_MASK


def from_unit(u: np.ndarray, clamp: bool = True) -> np.ndarray:
    if clamp:
        u = np.clip(u, -1.0, 1.0)
    phys = S.CH_LO + (u + 1.0) * 0.5 * (S.CH_HI - S.CH_LO)
    return np.where(S.CH_MASK > 0, phys, 0.0)


class Normalizer:
    """Standardisation statistics (computed from the TRAIN split only) on top of bound normalisation."""

    MIN_STD = 0.05

    def __init__(self, mean: np.ndarray, std: np.ndarray, source: Optional[dict] = None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), self.MIN_STD)
        self.source = source or {}

    @classmethod
    def fit(cls, phys: np.ndarray, slot_mask: np.ndarray, source: Optional[dict] = None) -> "Normalizer":
        """phys: (N,L,D) physical; slot_mask: (N,L)."""
        u = np.stack([to_unit(p) for p in phys])
        w = slot_mask[:, :, None] * S.CH_MASK[None]
        cnt = w.sum(0)
        mean = (u * w).sum(0) / np.maximum(cnt, 1)
        var = (((u - mean) ** 2) * w).sum(0) / np.maximum(cnt, 1)
        std = np.sqrt(var)
        poor = cnt < 10
        mean = np.where(poor, 0.0, mean)
        std = np.where(poor, 0.5, std)
        mean = mean * S.CH_MASK
        std = np.where(S.CH_MASK > 0, std, 1.0)
        return cls(mean, std, source)

    def normalize(self, phys: np.ndarray) -> np.ndarray:
        u = np.stack([to_unit(p) for p in phys]) if phys.ndim == 3 else to_unit(phys)
        return ((u - self.mean) / self.std * S.CH_MASK).astype(np.float32)

    def denormalize(self, x: np.ndarray, clamp: bool = False) -> np.ndarray:
        u = x * self.std + self.mean
        if x.ndim == 3:
            return np.stack([from_unit(a, clamp=clamp) for a in u])
        return from_unit(u, clamp=clamp)

    def x0_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-channel limits of the model-space variable implied by physical bounds."""
        lo = (-1.0 - self.mean) / self.std
        hi = (1.0 - self.mean) / self.std
        return lo.astype(np.float32), hi.astype(np.float32)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "source": self.source}

    @classmethod
    def from_json(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32), d.get("source"))
