"""Scenario schema: variables, units, bounds, categorical vocabularies, ScenarioSpec.

Token layout of one scenario (L = 1 + K_OBJ + M_OBS tokens, each padded to D channels):
  token 0                : GLOBAL   (event schedule, disturbance, sensor degradation)
  tokens 1..K_OBJ        : OBJECT slots (pose, rotation (6D), scale, mass, friction)
  tokens 1+K_OBJ..L-1    : OBSTACLE slots (dynamic obstacle trajectory parameters)

Categorical variables (task, event, robot profile, asset ids, slot masks) are NOT diffused:
they are conditioning inputs. Continuous variables carry explicit units and physical bounds.
Frame convention: right-handed, +x forward, +z up, origin = robot base start pose (robot-relative).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np

K_OBJ = 6
M_OBS = 2
L = 1 + K_OBJ + M_OBS
D = 12

TASKS = ("grasp_lift", "navigate")
EVENTS = ("nominal", "object_slip", "moving_obstacle", "perception_degradation", "slip_and_perception")
ASSETS = ("none", "block", "can", "ball", "crate", "bottle")  # id 0 = empty slot
ROBOT_PROFILES = ("proxy_mobile_manipulator_v0",)
TARGET_ASSETS = (1, 2, 3, 5)  # graspable assets (crate is clutter only)

TABLE_TOP_Z = 0.5


@dataclass(frozen=True)
class Var:
    name: str
    lo: float
    hi: float
    unit: str
    nominal: float


GLOBAL_VARS = (
    Var("event_onset", 0.3, 4.0, "s", 1.0),
    Var("event_duration", 0.1, 1.5, "s", 0.1),
    Var("disturbance_force", 0.0, 12.0, "N", 0.0),
    Var("disturbance_dir_sin", -1.0, 1.0, "1", 0.0),
    Var("disturbance_dir_cos", -1.0, 1.0, "1", 1.0),
    Var("friction_multiplier", 0.05, 1.0, "1", 1.0),
    Var("sensor_noise_std", 0.0, 0.04, "m", 0.0),
    Var("occlusion_dropout_prob", 0.0, 0.7, "1", 0.0),
    Var("visibility_range", 0.8, 3.0, "m", 3.0),
    Var("calib_yaw_err", -5.0, 5.0, "deg", 0.0),
    Var("calib_tx", -0.05, 0.05, "m", 0.0),
    Var("calib_ty", -0.05, 0.05, "m", 0.0),
)
OBJ_VARS = (
    Var("x", 0.3, 5.0, "m", 0.0),
    Var("y", -1.7, 1.7, "m", 0.0),
    Var("dz", 0.0, 0.03, "m", 0.0),
    Var("r0x", -1.0, 1.0, "1", 1.0), Var("r0y", -1.0, 1.0, "1", 0.0), Var("r0z", -1.0, 1.0, "1", 0.0),
    Var("r1x", -1.0, 1.0, "1", 0.0), Var("r1y", -1.0, 1.0, "1", 1.0), Var("r1z", -1.0, 1.0, "1", 0.0),
    Var("scale", 0.7, 1.4, "1", 1.0),
    Var("mass", 0.05, 1.2, "kg", 0.2),
    Var("friction", 0.05, 1.0, "1", 0.5),
)
OBS_VARS = (
    Var("x0", 0.8, 4.5, "m", 2.0),
    Var("y0", -2.2, 2.2, "m", 1.5),
    Var("vx", -0.6, 0.6, "m/s", 0.0),
    Var("vy", -1.8, 1.8, "m/s", 0.0),
    Var("half_size", 0.12, 0.4, "m", 0.2),
    Var("delay", 0.0, 0.6, "s", 0.0),
)
ENV_VARS = (
    Var("goal_distance", 0.0, 5.0, "m", 0.0),
    Var("surface_friction", 0.2, 0.9, "1", 0.5),
    Var("clutter_level", 0.0, 1.0, "1", 0.0),
)
E = len(ENV_VARS)
assert len(GLOBAL_VARS) == D and len(OBJ_VARS) == D and len(OBS_VARS) <= D

GLOBAL_IDX = {v.name: i for i, v in enumerate(GLOBAL_VARS)}
OBJ_IDX = {v.name: i for i, v in enumerate(OBJ_VARS)}
OBS_IDX = {v.name: i for i, v in enumerate(OBS_VARS)}

SLOT_GLOBAL = 0
OBJ_SLOTS = tuple(range(1, 1 + K_OBJ))
OBS_SLOTS = tuple(range(1 + K_OBJ, L))


def slot_vars(slot: int):
    if slot == SLOT_GLOBAL:
        return GLOBAL_VARS
    if slot in OBJ_SLOTS:
        return OBJ_VARS
    return OBS_VARS


def build_channel_tables():
    lo = np.full((L, D), -1.0)
    hi = np.full((L, D), 1.0)
    nom = np.zeros((L, D))
    chmask = np.zeros((L, D))
    for s in range(L):
        for j, v in enumerate(slot_vars(s)):
            lo[s, j], hi[s, j], nom[s, j], chmask[s, j] = v.lo, v.hi, v.nominal, 1.0
    return lo, hi, nom, chmask


CH_LO, CH_HI, CH_NOM, CH_MASK = build_channel_tables()

# --- event groups: which parameter groups are active for each event type -----------------------
GROUP_CHANNELS = {
    "slip": ("event_onset", "event_duration", "disturbance_force", "disturbance_dir_sin",
             "disturbance_dir_cos", "friction_multiplier"),
    "perception": ("sensor_noise_std", "occlusion_dropout_prob", "visibility_range",
                   "calib_yaw_err", "calib_tx", "calib_ty"),
    "obstacle": ("event_onset", "event_duration"),
}
EVENT_GROUPS = {
    "nominal": (),
    "object_slip": ("slip",),
    "moving_obstacle": ("obstacle",),
    "perception_degradation": ("perception",),
    "slip_and_perception": ("slip", "perception"),
}
EVENT_TASKS = {  # which tasks each event is defined for
    "nominal": (0, 1), "object_slip": (0,), "moving_obstacle": (1,),
    "perception_degradation": (0,), "slip_and_perception": (0,),
}


def active_global_channels(event: int) -> set:
    names = set()
    for g in EVENT_GROUPS[EVENTS[event]]:
        names.update(GROUP_CHANNELS[g])
    return names


# --- spec dataclasses ---------------------------------------------------------------------------
@dataclass
class ObjectSpec:
    slot: int  # 0..K_OBJ-1 (slot 0 is the manipulation target for task 'grasp_lift')
    asset_id: int
    x: float
    y: float
    dz: float
    rot: List[float]  # row-major 3x3 rotation matrix
    scale: float
    mass: float
    friction: float


@dataclass
class ObstacleSpec:
    slot: int  # 0..M_OBS-1
    x0: float
    y0: float
    vx: float
    vy: float
    half_size: float
    delay: float


@dataclass
class ScenarioSpec:
    scenario_id: str
    parent_id: str
    seed: int
    task: int
    event: int
    robot_profile: int
    severity: float
    env: Dict[str, float]
    glob: Dict[str, float]
    objects: List[ObjectSpec] = field(default_factory=list)
    obstacles: List[ObstacleSpec] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ScenarioSpec":
        d = dict(d)
        d["objects"] = [ObjectSpec(**o) for o in d.get("objects", [])]
        d["obstacles"] = [ObstacleSpec(**o) for o in d.get("obstacles", [])]
        return ScenarioSpec(**d)

    @property
    def task_name(self) -> str:
        return TASKS[self.task]

    @property
    def event_name(self) -> str:
        return EVENTS[self.event]


def nominal_global() -> Dict[str, float]:
    return {v.name: v.nominal for v in GLOBAL_VARS}


# --- rotations ---------------------------------------------------------------------------------
def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot6_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: continuous 6D representation -> valid SO(3) matrix. r6 = [col0(3), col1(3)]."""
    a1, a2 = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    n1 = np.linalg.norm(a1)
    b1 = a1 / n1 if n1 > 1e-8 else np.array([1.0, 0, 0])
    a2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2)
    if n2 < 1e-8:  # degenerate: pick any vector orthogonal to b1
        helper = np.array([0.0, 0, 1.0]) if abs(b1[2]) < 0.9 else np.array([0.0, 1.0, 0])
        a2 = helper - np.dot(b1, helper) * b1
        n2 = np.linalg.norm(a2)
    b2 = a2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_rot6(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]])


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    R = np.asarray(R).reshape(3, 3)
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


def tilt_angle(R: np.ndarray) -> float:
    """Angle between the object's z axis and world z."""
    return float(np.arccos(np.clip(np.asarray(R).reshape(3, 3)[2, 2], -1.0, 1.0)))


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R).reshape(3, 3)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# --- asset geometry ----------------------------------------------------------------------------
# (shape, base size tuple). Heights are half-heights along z.
ASSET_GEOM = {
    1: ("box", (0.035, 0.020, 0.040)),
    2: ("cylinder", (0.030, 0.050)),
    3: ("sphere", (0.035,)),
    4: ("box", (0.060, 0.060, 0.060)),
    5: ("cylinder", (0.025, 0.080)),
}


def asset_half_height(asset_id: int, scale: float) -> float:
    shape, size = ASSET_GEOM[asset_id]
    return float(size[-1] * scale) if shape != "sphere" else float(size[0] * scale)


def asset_footprint_radius(asset_id: int, scale: float) -> float:
    """Conservative xy circle radius enclosing the asset."""
    shape, size = ASSET_GEOM[asset_id]
    if shape == "box":
        return float(np.hypot(size[0], size[1]) * scale)
    return float(size[0] * scale)


def asset_grasp_halfwidth(asset_id: int, scale: float, R: Optional[np.ndarray] = None) -> float:
    """Half extent of the asset along world y (finger closing axis) given its orientation."""
    shape, size = ASSET_GEOM[asset_id]
    if shape == "box":
        half = np.array(size) * scale
        Rm = np.eye(3) if R is None else np.asarray(R).reshape(3, 3)
        return float(np.abs(Rm[1, :]) @ half)
    return float(size[0] * scale)
