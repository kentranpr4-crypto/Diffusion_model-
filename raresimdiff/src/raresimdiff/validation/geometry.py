"""Geometry / schema validation of a ScenarioSpec (no physics)."""
from __future__ import annotations
from typing import List
import numpy as np
from ..scenarios import schema as S
from . import reasons as R

T_MAX = {0: 4.5, 1: 10.0}
MAX_TILT = 0.25
TABLE_X = (0.5, 1.2)
TABLE_HALF_Y = 0.55
OVERLAP_MARGIN = 0.005


def check_bounds(spec: S.ScenarioSpec, tol: float = 1e-6) -> List[str]:
    bad = []
    for v in S.GLOBAL_VARS:
        x = spec.glob[v.name]
        if not np.isfinite(x):
            return [R.NON_FINITE]
        if x < v.lo - tol or x > v.hi + tol:
            bad.append(R.OUT_OF_BOUNDS)
    for o in spec.objects:
        vals = {"x": o.x, "y": o.y, "dz": o.dz, "scale": o.scale, "mass": o.mass, "friction": o.friction}
        for v in S.OBJ_VARS:
            if v.name in vals:
                x = vals[v.name]
                if not np.isfinite(x):
                    return [R.NON_FINITE]
                if x < v.lo - tol or x > v.hi + tol:
                    bad.append(R.OUT_OF_BOUNDS)
    for ob in spec.obstacles:
        for v in S.OBS_VARS:
            x = getattr(ob, v.name)
            if not np.isfinite(x):
                return [R.NON_FINITE]
            if x < v.lo - tol or x > v.hi + tol:
                bad.append(R.OUT_OF_BOUNDS)
    return sorted(set(bad))


def validate_geometry(spec: S.ScenarioSpec) -> List[str]:
    out: List[str] = []
    out += check_bounds(spec)
    if R.NON_FINITE in out:
        return out
    ev = S.EVENTS[spec.event]
    if spec.task not in S.EVENT_TASKS[ev]:
        out.append(R.EVENT_TASK_MISMATCH)
    # masks / structure
    slots = [o.slot for o in spec.objects]
    if len(set(slots)) != len(slots) or any(not (0 <= s < S.K_OBJ) for s in slots):
        out.append(R.MASK_INCONSISTENT)
    if any(o.asset_id not in S.ASSET_GEOM for o in spec.objects):
        out.append(R.MASK_INCONSISTENT)
    if (ev == "moving_obstacle") != (len(spec.obstacles) > 0) or len(spec.obstacles) > S.M_OBS:
        out.append(R.MASK_INCONSISTENT)
    # rotations
    for o in spec.objects:
        Rm = np.asarray(o.rot).reshape(3, 3)
        if not S.is_rotation(Rm, tol=1e-5):
            out.append(R.ROT_INVALID)
        elif S.tilt_angle(Rm) > MAX_TILT + 1e-3:
            out.append(R.TILT_EXCEEDED)
    # inactive parameter groups must be nominal
    active = S.active_global_channels(spec.event)
    for v in S.GLOBAL_VARS:
        if v.name not in active and abs(spec.glob[v.name] - v.nominal) > 1e-6 * max(1.0, abs(v.hi - v.lo)):
            out.append(R.INACTIVE_NOT_NOMINAL)
            break
    # event timing
    if ev != "nominal" and ev != "perception_degradation":
        if spec.glob["event_onset"] + spec.glob["event_duration"] > T_MAX[spec.task] + 1e-6:
            out.append(R.EVENT_TIMING)
    # task-specific geometry
    objs = spec.objects
    if spec.task == 0:
        tgt = next((o for o in objs if o.slot == 0), None)
        if tgt is None:
            out.append(R.NO_TARGET)
        else:
            if tgt.asset_id not in S.TARGET_ASSETS:
                out.append(R.TARGET_NOT_GRASPABLE)
            Rm = np.asarray(tgt.rot).reshape(3, 3)
            if S.asset_grasp_halfwidth(tgt.asset_id, tgt.scale, Rm) > 0.058:
                out.append(R.TARGET_NOT_GRASPABLE)
            if not (0.55 <= tgt.x <= 0.85 and abs(tgt.y) <= 0.35):
                out.append(R.TARGET_UNREACHABLE)
        for o in objs:
            r = S.asset_footprint_radius(o.asset_id, o.scale)
            if not (TABLE_X[0] + r <= o.x <= TABLE_X[1] - r and abs(o.y) <= TABLE_HALF_Y - r):
                out.append(R.OUT_OF_SUPPORT)
                break
    else:
        goal = spec.env["goal_distance"]
        for o in objs:
            if not (0.5 <= o.x <= goal + 1.0 and abs(o.y) <= 2.0):
                out.append(R.OUT_OF_SUPPORT)
                break
        if not (1.0 <= goal <= 5.0):
            out.append(R.OUT_OF_BOUNDS)
    # object-object overlaps (conservative circles)
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ra, rb = S.asset_footprint_radius(a.asset_id, a.scale), S.asset_footprint_radius(b.asset_id, b.scale)
            if np.hypot(a.x - b.x, a.y - b.y) < ra + rb + OVERLAP_MARGIN:
                out.append(R.OBJ_OVERLAP)
    # obstacles vs robot start (robot box half 0.2 at origin) and objects
    for ob in spec.obstacles:
        if abs(ob.x0) - ob.half_size < 0.3 and abs(ob.y0) - ob.half_size < 0.3:
            out.append(R.OBSTACLE_OVERLAP)
        for o in objs:
            if abs(ob.x0 - o.x) < ob.half_size + 0.1 and abs(ob.y0 - o.y) < ob.half_size + 0.1:
                out.append(R.OBSTACLE_OVERLAP)
    return sorted(set(out))
