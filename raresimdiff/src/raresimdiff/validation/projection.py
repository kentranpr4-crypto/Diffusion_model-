"""Constraint projection: repair a decoded (possibly out-of-bounds) sample into a valid ScenarioSpec candidate.

Operates on physical-unit arrays phys[L,D]. Returns the repaired array and the repair magnitude measured in the
bound-normalised space (mean absolute change over active channels).
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from ..scenarios import schema as S
from ..scenarios.codec import to_unit
from .geometry import MAX_TILT, TABLE_X, TABLE_HALF_Y


def _limit_tilt(R: np.ndarray, max_tilt: float) -> np.ndarray:
    z = R[:, 2]
    tilt = np.arccos(np.clip(z[2], -1.0, 1.0))
    if tilt <= max_tilt:
        return R
    axis = np.cross([0.0, 0.0, 1.0], z)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R
    axis /= n
    ang = max_tilt  # rotate world z by max_tilt about axis
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rz = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    znew = Rz @ np.array([0.0, 0.0, 1.0])
    x = R[:, 0] - np.dot(R[:, 0], znew) * znew
    nx = np.linalg.norm(x)
    x = x / nx if nx > 1e-8 else np.cross([0.0, 1.0, 0.0], znew)
    y = np.cross(znew, x)
    return np.stack([x, y, znew], axis=1)


def _separate(objs_xy, radii, lo_hi, iters: int = 60, margin: float = 0.01):
    xy = objs_xy.copy()
    n = len(xy)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = xy[j] - xy[i]
                dist = np.linalg.norm(d)
                need = radii[i] + radii[j] + margin
                if dist < need:
                    u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    push = 0.5 * (need - dist) + 1e-4
                    wi, wj = (0.0, 1.0) if i == 0 else (0.5, 0.5)  # keep the target (slot 0) in place
                    xy[i] -= u * push * 2 * wi
                    xy[j] += u * push * 2 * wj
                    moved = True
        for i in range(n):
            xy[i] = np.clip(xy[i], lo_hi[i][0], lo_hi[i][1])
        if not moved:
            break
    return xy


def project(phys_raw: np.ndarray, mask: np.ndarray, asset_ids: np.ndarray, task: int, event: int,
            env: dict) -> Tuple[np.ndarray, float, dict]:
    """phys_raw: (L,D) unclamped physical values decoded from the model."""
    u_raw = to_unit(phys_raw)
    p = phys_raw.copy()
    flags = {}
    # 1. nominal for padded channels / absent slots / inactive groups
    for s in range(S.L):
        if mask[s] < 0.5:
            p[s] = S.CH_NOM[s]
    active = S.active_global_channels(event)
    for v in S.GLOBAL_VARS:
        if v.name not in active:
            p[S.SLOT_GLOBAL, S.GLOBAL_IDX[v.name]] = v.nominal
    # 2. clamp to bounds
    lo, hi = np.where(S.CH_MASK > 0, S.CH_LO, -1.0), np.where(S.CH_MASK > 0, S.CH_HI, 1.0)
    p = np.clip(p, lo, hi)
    for s in range(S.L):  # absent slots keep their nominal placeholder (outside bounds by construction)
        if mask[s] < 0.5:
            p[s] = S.CH_NOM[s]
    # 3. direction (sin, cos) unit norm
    si, ci = S.GLOBAL_IDX["disturbance_dir_sin"], S.GLOBAL_IDX["disturbance_dir_cos"]
    if "disturbance_dir_sin" in active:
        v = np.array([p[0, si], p[0, ci]])
        n = np.linalg.norm(v)
        p[0, si], p[0, ci] = (v / n) if n > 1e-6 else (0.0, 1.0)
    # 4. rotations: Gram-Schmidt -> SO(3) -> tilt limit -> back to 6D
    for k, s in enumerate(S.OBJ_SLOTS):
        if mask[s] < 0.5 or asset_ids[k] == 0:
            continue
        r6 = np.array([p[s, S.OBJ_IDX[n]] for n in ("r0x", "r0y", "r0z", "r1x", "r1y", "r1z")])
        Rm = _limit_tilt(S.rot6_to_matrix(r6), MAX_TILT)
        r6n = S.matrix_to_rot6(Rm)
        for n, val in zip(("r0x", "r0y", "r0z", "r1x", "r1y", "r1z"), r6n):
            p[s, S.OBJ_IDX[n]] = val
    # 5. support clipping, target envelope, overlap separation
    idx = [(k, s) for k, s in enumerate(S.OBJ_SLOTS) if mask[s] > 0.5 and asset_ids[k] != 0]
    if idx:
        scales = [p[s, S.OBJ_IDX["scale"]] for _, s in idx]
        radii = [S.asset_footprint_radius(int(asset_ids[k]), sc) for (k, _), sc in zip(idx, scales)]
        xy = np.array([[p[s, S.OBJ_IDX["x"]], p[s, S.OBJ_IDX["y"]]] for _, s in idx])
        lims = []
        goal = env.get("goal_distance", 0.0)
        for (k, s), r in zip(idx, radii):
            if task == 0:
                if k == 0:
                    lims.append(((0.55, -0.35), (0.85, 0.35)))
                else:
                    lims.append(((TABLE_X[0] + r, -(TABLE_HALF_Y - r)), (TABLE_X[1] - r, TABLE_HALF_Y - r)))
            else:
                lims.append(((0.5, -2.0), (goal + 1.0, 2.0)))
        lo_hi = [(np.array(a), np.array(b)) for a, b in lims]
        xy = _separate(xy, radii, lo_hi)
        for (k, s), pos in zip(idx, xy):
            p[s, S.OBJ_IDX["x"]], p[s, S.OBJ_IDX["y"]] = pos
        if task == 0 and idx[0][0] == 0:
            # grasp feasibility: shrink target scale if the closing width exceeds the finger opening
            s0 = idx[0][1]
            R0 = S.rot6_to_matrix(np.array([p[s0, S.OBJ_IDX[n]] for n in ("r0x", "r0y", "r0z", "r1x", "r1y", "r1z")]))
            hw = S.asset_grasp_halfwidth(int(asset_ids[0]), p[s0, S.OBJ_IDX["scale"]], R0)
            if hw > 0.058:
                p[s0, S.OBJ_IDX["scale"]] = max(S.OBJ_VARS[S.OBJ_IDX["scale"]].lo, p[s0, S.OBJ_IDX["scale"]] * 0.058 / hw)
    # 6. event timing
    if "event_onset" in active:
        tmax = 4.5 if task == 0 else 10.0
        on, du = p[0, S.GLOBAL_IDX["event_onset"]], p[0, S.GLOBAL_IDX["event_duration"]]
        if on + du > tmax:
            p[0, S.GLOBAL_IDX["event_duration"]] = max(0.1, tmax - on)
    # repair magnitude over active channels
    u_new = to_unit(p)
    w = mask[:, None] * S.CH_MASK
    act = np.ones_like(w)
    for v in S.GLOBAL_VARS:
        if v.name not in active:
            act[S.SLOT_GLOBAL, S.GLOBAL_IDX[v.name]] = 0.0
    w = w * act
    mag = float((np.abs(u_new - u_raw) * w).sum() / max(w.sum(), 1.0))
    return p, mag, flags
