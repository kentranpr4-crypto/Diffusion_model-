"""Perception degradation model (applied to observations only; it never changes physics).

Cosmetic/perceptual parameters: sensor noise, occlusion dropout, visibility range, camera extrinsic
(calibration) error. Dynamics-affecting parameters (friction, disturbance, obstacle motion) live in events.py.
"""
from __future__ import annotations
import numpy as np


class PerceptionModel:
    def __init__(self, glob: dict, seed: int):
        self.noise = float(glob["sensor_noise_std"])
        self.dropout = float(glob["occlusion_dropout_prob"])
        self.vis = float(glob["visibility_range"])
        yaw = np.deg2rad(float(glob["calib_yaw_err"]))
        c, s = np.cos(yaw), np.sin(yaw)
        self.R_err = np.array([[c, -s], [s, c]])
        self.t_err = np.array([float(glob["calib_tx"]), float(glob["calib_ty"])])
        self.rng = np.random.default_rng(seed)

    def observe(self, base_xy: np.ndarray, base_yaw: float, points_xy: np.ndarray):
        """points_xy: (n,2) true world xy. Returns (valid (n,), est_xy (n,2)). RNG draws are branch-independent."""
        n = len(points_xy)
        u = self.rng.random(n)
        eps = self.rng.normal(0.0, 1.0, size=(n, 2)) * self.noise
        if n == 0:
            return np.zeros(0, bool), np.zeros((0, 2))
        d = points_xy - base_xy[None]
        dist = np.linalg.norm(d, axis=1)
        valid = (dist <= self.vis) & (u >= self.dropout)
        c, s = np.cos(base_yaw), np.sin(base_yaw)
        Rb = np.array([[c, -s], [s, c]])
        p_cam = d @ Rb  # world -> base frame (row-vector form of Rb^T d)
        p_err = p_cam @ self.R_err.T + self.t_err[None]
        est = base_xy[None] + p_err @ Rb.T + eps
        return valid, est
