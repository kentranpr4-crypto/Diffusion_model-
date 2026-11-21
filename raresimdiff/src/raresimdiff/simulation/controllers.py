"""Scripted baseline controllers. These are *baselines for difficulty measurement*, not expert demonstrators.

They read only the observation vector (no ground-truth access), so perception degradation affects them.
Observation layout (OBS_DIM=28): q[7] qd[7] t | tgt_valid tgt_xy[2] | obst0(valid,xy) obst1(valid,xy) | goal | task onehot
"""
from __future__ import annotations
import numpy as np
from .base import ACT_DIM

REACH_MAX, LIFT_MAX, GRIP_MAX = 0.55, 0.35, 0.07


class Controller:
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NullController(Controller):
    def act(self, obs):
        return np.zeros(ACT_DIM)


class ScriptedBaseline(Controller):
    """Task dispatcher: grasp_lift state machine / reactive navigation."""

    def __init__(self, hand_rest_x: float = 0.30, lift_height: float = 0.18):
        self.hand_rest_x = hand_rest_x
        self.lift_height = lift_height
        self.reset()

    def reset(self):
        self.est = None
        self.n_valid = 0
        self.phase = 0
        self.phase_t0 = 0.0
        self.reach_cmd = 0.0
        self.ob_est = [None, None]
        self.ob_vel = [np.zeros(2), np.zeros(2)]
        self.ob_t = [None, None]
        self.blocked_t = None
        self.detour_y = None

    def act(self, obs):
        return self._nav(obs) if obs[27] > 0.5 else self._grasp(obs)

    # ---------------------------------------------------------------- grasp
    def _grasp(self, o):
        t = float(o[14])
        by, reach_q = float(o[1]), float(o[3])
        if o[15] > 0.5:
            m = o[16:18]
            self.est = m.copy() if self.est is None else 0.7 * self.est + 0.3 * m
            self.n_valid += 1
        a = np.zeros(ACT_DIM)
        est = self.est
        if est is None and t > 1.0:
            est = np.array([0.70, 0.0])  # blind default grasp pose when nothing was ever observed
        # phase transitions
        if self.phase == 0 and t >= 0.5 and est is not None:
            self._goto(1, t)
        elif self.phase == 1:
            err = (est[1] - by) if est is not None else 0.0
            if abs(err) < 0.006 or t - self.phase_t0 > 1.2:
                self._goto(2, t)
        elif self.phase == 2:
            tgt = np.clip((est[0] - self.hand_rest_x), 0.0, REACH_MAX)
            if abs(self.reach_cmd - tgt) < 1e-3 and t - self.phase_t0 > 0.4 or t - self.phase_t0 > 1.4:
                self._goto(3, t)
        elif self.phase == 3 and t - self.phase_t0 > 0.5:
            self._goto(4, t)
        # actions
        if self.phase >= 1 and est is not None:
            a[1] = np.clip(4.0 * (est[1] - by), -0.5, 0.5)
        if self.phase >= 2 and est is not None:
            tgt = np.clip(est[0] - self.hand_rest_x, 0.0, REACH_MAX)
            self.reach_cmd += np.clip(tgt - self.reach_cmd, -0.01, 0.01)
        a[3] = self.reach_cmd / REACH_MAX
        grip = 0.0
        if self.phase >= 3:
            grip = 1.0
        a[5] = grip
        if self.phase >= 4:
            frac = np.clip((t - self.phase_t0) / 0.8, 0.0, 1.0)
            a[4] = frac * self.lift_height / LIFT_MAX
        return a

    def _goto(self, ph, t):
        self.phase, self.phase_t0 = ph, t

    # ------------------------------------------------------------------ nav
    def _nav(self, o):
        t = float(o[14])
        bx, by, yaw = float(o[0]), float(o[1]), float(o[2])
        goal = float(o[24])
        for i in range(2):
            if o[18 + 3 * i] > 0.5:
                p = o[19 + 3 * i:21 + 3 * i].copy()
                if self.ob_est[i] is not None and self.ob_t[i] is not None and t > self.ob_t[i]:
                    v = (p - self.ob_est[i]) / max(t - self.ob_t[i], 1e-3)
                    self.ob_vel[i] = 0.6 * self.ob_vel[i] + 0.4 * v
                self.ob_est[i], self.ob_t[i] = p, t
        vmax = 0.8
        dx, dy = goal - bx, 0.0 - by
        dist = np.hypot(dx, dy)
        v_des = vmax * np.array([dx, dy]) / max(dist, 1e-6) * min(1.0, dist / 0.5)
        scale = 1.0
        blocker = None
        for i in range(2):
            if self.ob_est[i] is None or self.ob_t[i] is None or t - self.ob_t[i] > 0.6:
                continue  # stale / never seen
            ox, oy = self.ob_est[i]
            for look in (0.0, 0.4, 0.8):
                px, py = ox + self.ob_vel[i][0] * look, oy + self.ob_vel[i][1] * look
                ahead = px - bx
                if -0.1 < ahead < 1.9 and abs(py - by) < 0.62:
                    sc = float(np.clip((ahead - 0.85) / 1.0, 0.0, 1.0))
                    if sc < scale:
                        scale, blocker = sc, (ox, oy, i)
        if scale < 0.05 and blocker is not None:
            self.blocked_t = getattr(self, "blocked_t", t) if getattr(self, "blocked_t", None) is not None else t
            speed = float(np.linalg.norm(self.ob_vel[blocker[2]]))
            if t - self.blocked_t > 0.8 and speed < 0.15:  # obstacle parked in the corridor: sidestep
                side = 1.0 if by - blocker[1] >= 0 else -1.0
                self.detour_y = blocker[1] + side * 1.05
        else:
            self.blocked_t = None
        detour = getattr(self, "detour_y", None)
        if detour is not None:
            if blocker is None or abs(blocker[1] - by) > 0.7:
                if blocker is None:
                    self.detour_y = detour = None
            if detour is not None:
                v_des = np.array([0.0, float(np.clip(1.5 * (detour - by), -0.5, 0.5))])
                scale = 1.0
        v = v_des * scale
        a = np.zeros(ACT_DIM)
        a[0], a[1] = v[0], v[1]
        a[2] = np.clip(-2.0 * yaw, -1, 1)
        return a
