"""MJCF generation from a ScenarioSpec (physics is authoritative: no kinematic teleporting)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..scenarios import schema as S

# collision bit-masks: env=1, object=2, hand/fingers=4, base=8, obstacle=16
C_ENV, C_OBJ, C_HAND, C_BASE, C_OBS = 1, 2, 4, 8, 16


@dataclass
class SimConfig:
    timestep: float = 0.004
    control_dt: float = 0.02
    settle_time: float = 0.5
    t_max_grasp: float = 4.5
    t_max_nav: float = 10.0
    traj_hz: float = 10.0
    penetration_tol: float = 0.004
    grip_kp: float = 400.0
    grip_force: float = 25.0
    hand_rest_x: float = 0.30
    finger_open: float = 0.075
    lift_height: float = 0.18


def rest_height(asset_id: int, scale: float, R: np.ndarray) -> float:
    """Height of the object centre above its support when resting (lowest point touches support)."""
    shape, size = S.ASSET_GEOM[asset_id]
    R = np.asarray(R).reshape(3, 3)
    if shape == "box":
        half = np.array(size) * scale
        return float(np.abs(R[2, :]) @ half)
    if shape == "cylinder":
        r, hh = size[0] * scale, size[1] * scale
        cz = abs(R[2, 2])
        return float(hh * cz + r * np.sqrt(max(0.0, 1.0 - cz * cz)))
    return float(size[0] * scale)


def _geom_xml(o: S.ObjectSpec, name: str, mu: float) -> str:
    shape, size = S.ASSET_GEOM[o.asset_id]
    sz = " ".join(f"{v * o.scale:.5f}" for v in size)
    return (f'<geom name="{name}" type="{shape}" size="{sz}" mass="{o.mass:.5f}" friction="{mu:.5f} 0.005 0.0001" '
            f'condim="4" contype="{C_OBJ}" conaffinity="{C_ENV | C_OBJ | C_HAND | C_BASE | C_OBS}" '
            f'rgba="0.8 0.4 0.2 1"/>')


def build_mjcf(spec: S.ScenarioSpec, cfg: SimConfig) -> str:
    task_grasp = spec.task == 0
    support_z = S.TABLE_TOP_Z if task_grasp else 0.0
    surf = spec.env["surface_friction"]
    parts = []
    parts.append(f'<mujoco model="raresimdiff"><compiler angle="radian" autolimits="true"/>'
                 f'<option timestep="{cfg.timestep}" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>'
                 f'<worldbody>')
    floor_mu = 0.8 if task_grasp else surf
    parts.append(f'<geom name="floor" type="plane" size="12 12 0.1" friction="{floor_mu:.4f} 0.005 0.0001" '
                 f'contype="{C_ENV}" conaffinity="0"/>')
    if task_grasp:
        parts.append(f'<geom name="table" type="box" pos="0.85 0 0.25" size="0.35 0.55 0.25" '
                     f'friction="{surf:.4f} 0.005 0.0001" contype="{C_ENV}" conaffinity="0" rgba="0.6 0.5 0.4 1"/>')
    # --- robot (proxy mobile manipulator; NOT a humanoid) ---
    hand_z_rel = 0.56 - 0.25
    parts.append(f'''
<body name="base" pos="0 0 0.25">
  <joint name="base_x" type="slide" axis="1 0 0" range="-1 8" damping="30"/>
  <joint name="base_y" type="slide" axis="0 1 0" range="-3 3" damping="30"/>
  <joint name="base_yaw" type="hinge" axis="0 0 1" range="-3.3 3.3" damping="10"/>
  <geom name="base_geom" type="box" size="0.2 0.2 0.15" mass="10" contype="{C_BASE}" conaffinity="0" rgba="0.2 0.3 0.8 1"/>
  <body name="reach_link" pos="{cfg.hand_rest_x} 0 {hand_z_rel}" gravcomp="1">
    <joint name="reach" type="slide" axis="1 0 0" range="0 0.55" damping="30"/>
    <geom name="reach_geom" type="box" size="0.01 0.01 0.01" mass="0.3" contype="0" conaffinity="0"/>
    <body name="lift_link" gravcomp="1">
      <joint name="lift" type="slide" axis="0 0 1" range="0 0.35" damping="30"/>
      <geom name="palm" type="box" pos="-0.075 0 0" size="0.02 0.09 0.045" mass="0.5" contype="{C_HAND}" conaffinity="0" rgba="0.3 0.3 0.3 1"/>
      <body name="finger_l_body" pos="0 {cfg.finger_open} 0" gravcomp="1">
        <joint name="finger_l" type="slide" axis="0 -1 0" range="0 0.07" damping="2"/>
        <geom name="finger_l_pad" type="box" size="0.03 0.006 0.045" mass="0.1" friction="0.05 0.005 0.0001" condim="4" contype="{C_HAND}" conaffinity="0"/>
      </body>
      <body name="finger_r_body" pos="0 {-cfg.finger_open} 0" gravcomp="1">
        <joint name="finger_r" type="slide" axis="0 1 0" range="0 0.07" damping="2"/>
        <geom name="finger_r_pad" type="box" size="0.03 0.006 0.045" mass="0.1" friction="0.05 0.005 0.0001" condim="4" contype="{C_HAND}" conaffinity="0"/>
      </body>
    </body>
  </body>
</body>''')
    # --- objects ---
    for o in spec.objects:
        R = np.asarray(o.rot).reshape(3, 3)
        q = S.matrix_to_quat(R)
        z = support_z + rest_height(o.asset_id, o.scale, R) + o.dz + 0.002
        parts.append(f'<body name="obj{o.slot}" pos="{o.x:.5f} {o.y:.5f} {z:.5f}" '
                     f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"><freejoint name="obj{o.slot}_free"/>'
                     f'{_geom_xml(o, f"obj{o.slot}_geom", o.friction)}</body>')
    # --- dynamic obstacles ---
    for ob in spec.obstacles:
        parts.append(f'''<body name="obs{ob.slot}" pos="{ob.x0:.5f} {ob.y0:.5f} 0.4">
  <joint name="obs{ob.slot}_x" type="slide" axis="1 0 0" damping="1"/>
  <joint name="obs{ob.slot}_y" type="slide" axis="0 1 0" damping="1"/>
  <geom name="obs{ob.slot}_geom" type="box" size="{ob.half_size:.4f} {ob.half_size:.4f} 0.4" mass="40" contype="{C_OBS}" conaffinity="{C_BASE | C_HAND}" rgba="0.9 0.1 0.1 1"/>
</body>''')
    parts.append('</worldbody><actuator>')
    parts.append('<velocity name="a_bx" joint="base_x" kv="400" ctrlrange="-1.5 1.5"/>')
    parts.append('<velocity name="a_by" joint="base_y" kv="400" ctrlrange="-1.5 1.5"/>')
    parts.append('<velocity name="a_byaw" joint="base_yaw" kv="100" ctrlrange="-2 2"/>')
    parts.append('<position name="a_reach" joint="reach" kp="1500" kv="80" ctrlrange="0 0.55"/>')
    parts.append('<position name="a_lift" joint="lift" kp="1500" kv="80" ctrlrange="0 0.35"/>')
    fk, ff = cfg.grip_kp, cfg.grip_force
    parts.append(f'<position name="a_fl" joint="finger_l" kp="{fk}" kv="10" ctrlrange="0 0.07" forcerange="-{ff} {ff}"/>')
    parts.append(f'<position name="a_fr" joint="finger_r" kp="{fk}" kv="10" ctrlrange="0 0.07" forcerange="-{ff} {ff}"/>')
    for ob in spec.obstacles:
        parts.append(f'<velocity name="obs{ob.slot}_avx" joint="obs{ob.slot}_x" kv="2000" ctrlrange="-3 3"/>')
        parts.append(f'<velocity name="obs{ob.slot}_avy" joint="obs{ob.slot}_y" kv="2000" ctrlrange="-3 3"/>')
    parts.append('</actuator></mujoco>')
    return "".join(parts)
