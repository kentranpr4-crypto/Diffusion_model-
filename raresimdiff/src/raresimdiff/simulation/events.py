"""Event injection, deliberately separate from robot control.

Physically interpretable mechanisms (applied at control-step granularity, 20 ms):
  * friction_multiplier: contact friction of the manipulated object scaled from `onset` to the end of the
    episode (e.g. oily / wet surface).
  * disturbance pulse: external force (N) applied to the target object centre during [onset, onset+duration]
    in the object-frame-independent world x-z plane, direction angle atan2(dir_sin, dir_cos).
  * moving obstacle: obstacle servo velocity set to (vx, vy) during [onset+delay, onset+delay+duration]
    and zero otherwise; the obstacle is a heavy body driven through its own actuators (no teleporting).
Perception parameters are handled by perception.py.
"""
from __future__ import annotations
import numpy as np
from ..scenarios import schema as S


class EventInjector:
    def __init__(self, spec: S.ScenarioSpec, ids: dict):
        self.spec = spec
        self.g = spec.glob
        self.ids = ids
        self.name = S.EVENTS[spec.event]
        self.slip = self.name in ("object_slip", "slip_and_perception") and spec.task == 0
        self.obst = self.name == "moving_obstacle"
        self.onset = float(self.g["event_onset"])
        self.dur = float(self.g["event_duration"])
        self.triggered = False
        self._friction_applied = False
        self.force_applied_impulse = 0.0

    def active(self) -> bool:
        return self.slip or self.obst

    def apply(self, model, data, t: float, dt: float):
        if self.slip:
            gid = self.ids["target_geom"]
            if t >= self.onset and not self._friction_applied and gid >= 0:
                model.geom_friction[gid, 0] = max(0.05, model.geom_friction[gid, 0] * self.g["friction_multiplier"])
                self._friction_applied = True
                self.triggered = True
            bid = self.ids["target_body"]
            if bid >= 0:
                if self.onset <= t < self.onset + self.dur:
                    th = np.arctan2(self.g["disturbance_dir_sin"], self.g["disturbance_dir_cos"])
                    F = self.g["disturbance_force"]
                    data.xfrc_applied[bid, :3] = [F * np.cos(th), 0.0, F * np.sin(th)]
                    self.force_applied_impulse += F * dt
                else:
                    data.xfrc_applied[bid, :3] = 0.0
        if self.obst:
            for ob, aid in zip(self.spec.obstacles, self.ids["obstacle_act"]):
                t0 = self.onset + ob.delay
                moving = t0 <= t < t0 + self.dur
                data.ctrl[aid[0]] = ob.vx if moving else 0.0
                data.ctrl[aid[1]] = ob.vy if moving else 0.0
                if t >= t0:
                    self.triggered = True
