"""Deterministic severity function: severity = f(event, physical parameters), in [0, 1].

Severity is *defined* by this function so that conditioning accuracy can be re-computed from any
generated scenario (no learned or hand-labelled severity).
"""
from __future__ import annotations
import numpy as np
from .schema import EVENTS


def _c(x):
    return float(np.clip(x, 0.0, 1.0))


def slip_severity(g: dict) -> float:
    drop = (1.0 - g["friction_multiplier"]) / 0.95
    force = g["disturbance_force"] / 12.0
    return _c(0.5 * (drop + force))


def perception_severity(g: dict) -> float:
    terms = [
        g["sensor_noise_std"] / 0.04,
        g["occlusion_dropout_prob"] / 0.7,
        (3.0 - g["visibility_range"]) / 2.2,
        0.5 * (abs(g["calib_yaw_err"]) / 5.0 + np.hypot(g["calib_tx"], g["calib_ty"]) / 0.0707),
    ]
    return _c(np.mean([_c(t) for t in terms]))


def obstacle_severity(g: dict, obstacles: list) -> float:
    if not obstacles:
        return 0.0
    vals = []
    for o in obstacles:
        speed = np.hypot(o["vx"], o["vy"])
        speed_n = _c((speed - 0.4) / 1.4)
        size_n = _c((o["half_size"] - 0.12) / 0.28)
        # urgency: how soon the obstacle crosses the corridor relative to nominal robot arrival
        t_cross = max(abs(o["y0"]) - 0.3, 0.0) / max(abs(o["vy"]), 1e-3)
        t_obs = g["event_onset"] + o["delay"] + t_cross
        t_robot = o["x0"] / 0.8
        urgency = _c(1.0 - abs(t_obs - t_robot) / 2.0)
        vals.append(np.mean([speed_n, size_n, urgency]))
    return _c(max(vals))


def compute_severity(event: int, g: dict, obstacles: list) -> float:
    name = EVENTS[event]
    if name == "nominal":
        return 0.0
    if name == "object_slip":
        return slip_severity(g)
    if name == "perception_degradation":
        return perception_severity(g)
    if name == "slip_and_perception":
        return _c(0.5 * (slip_severity(g) + perception_severity(g)))
    if name == "moving_obstacle":
        return obstacle_severity(g, obstacles)
    raise ValueError(name)
