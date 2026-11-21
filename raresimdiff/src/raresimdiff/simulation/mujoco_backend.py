"""MuJoCo backend implementing SimulatorBackend."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import mujoco

from ..scenarios import schema as S
from .base import ACT_DIM, OBS_DIM, SimulatorBackend, ValidationResult
from .events import EventInjector
from .mjcf import SimConfig, build_mjcf, C_ENV
from .perception import PerceptionModel

ROBOT_JOINTS = ("base_x", "base_y", "base_yaw", "reach", "lift", "finger_l", "finger_r")
CAT = {"floor": 0, "table": 0, "base": 1, "palm": 2, "finger": 2, "target": 3, "clutter": 4, "obstacle": 5}
mujoco.set_mju_user_warning(lambda msg: None)  # warnings are detected via data.warning counters instead
MAX_SPEED = 300.0  # |qvel| above this (m/s or rad/s) is treated as numerical instability
STATE_DIM = 7 + 7 + S.K_OBJ * 7 + S.M_OBS * 4
ACT_SCALE = np.array([1.0, 1.0, 1.0, 0.55, 0.35, 0.07])  # [vx, vy, wz, reach, lift, grip]


class MuJoCoBackend(SimulatorBackend):
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.model = None
        self.data = None
        self.spec: S.ScenarioSpec | None = None
        self.n_sub = int(round(self.cfg.control_dt / self.cfg.timestep))

    # ------------------------------------------------------------------ reset
    def reset(self, scenario: S.ScenarioSpec) -> np.ndarray:
        self.spec = scenario
        xml = build_mjcf(scenario, self.cfg)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {n: int(m.joint(n).qposadr[0]) for n in ROBOT_JOINTS}
        self.vadr = {n: int(m.joint(n).dofadr[0]) for n in ROBOT_JOINTS}
        self.obj_body = {o.slot: int(m.body(f"obj{o.slot}").id) for o in scenario.objects}
        self.obj_qadr = {o.slot: int(m.joint(f"obj{o.slot}_free").qposadr[0]) for o in scenario.objects}
        self.obs_qadr = {ob.slot: (int(m.joint(f"obs{ob.slot}_x").qposadr[0]), int(m.joint(f"obs{ob.slot}_y").qposadr[0]))
                         for ob in scenario.obstacles}
        self.obs_body = {ob.slot: int(m.body(f"obs{ob.slot}").id) for ob in scenario.obstacles}
        self.obs_vadr = {ob.slot: (int(m.joint(f"obs{ob.slot}_x").dofadr[0]), int(m.joint(f"obs{ob.slot}_y").dofadr[0]))
                         for ob in scenario.obstacles}
        self.act_id = {n: int(m.actuator(n).id) for n in ("a_bx", "a_by", "a_byaw", "a_reach", "a_lift", "a_fl", "a_fr")}
        target = next((o for o in scenario.objects if o.slot == 0), None)
        ids = {
            "target_body": self.obj_body[0] if (target is not None and scenario.task == 0) else -1,
            "target_geom": int(m.geom("obj0_geom").id) if (target is not None and scenario.task == 0) else -1,
            "obstacle_act": [(int(m.actuator(f"obs{ob.slot}_avx").id), int(m.actuator(f"obs{ob.slot}_avy").id))
                             for ob in scenario.obstacles],
        }
        self._build_geom_categories()
        self.events = EventInjector(scenario, ids)
        self.perception = PerceptionModel(scenario.glob, seed=scenario.seed + 1)
        self.t = 0.0
        self.step_count = 0
        self._last_action = np.zeros(ACT_DIM)
        self.metrics: Dict[str, Any] = self._fresh_metrics()
        mujoco.mj_forward(m, self.data)
        # settle: robot ctrl at rest pose, objects fall onto support
        self.data.ctrl[:] = 0.0
        mujoco.mj_step(m, self.data, nstep=int(round(self.cfg.settle_time / self.cfg.timestep)))
        self.settle_stats = self._settle_check()
        self.t = 0.0
        self.data.time = 0.0
        self._prev_goal_dist = self._goal_dist()
        self.t_max = self.cfg.t_max_grasp if scenario.task == 0 else self.cfg.t_max_nav
        self.metrics["initial_penetration"] = self.settle_stats["max_penetration"]
        self._obs_cache = self._compute_observation()
        return self._obs_cache.copy()

    def _bad_qacc(self) -> bool:
        """MuJoCo silently resets state on unstable simulation; detect it through its warning counter."""
        return bool(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0)

    def _fresh_metrics(self) -> Dict[str, Any]:
        return {"max_penetration": 0.0, "robot_obstacle_contact_steps": 0, "robot_clutter_contact_steps": 0,
                "finger_target_contact_steps": 0, "max_object_height": 0.0, "event_triggered": False,
                "object_dropped": False, "success": False, "termination_reason": "running",
                "nan_detected": False, "max_speed": 0.0}

    def _build_geom_categories(self):
        m = self.model
        cat = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            n = m.geom(g).name
            if n in ("floor", "table"):
                cat[g] = CAT["floor"]
            elif n == "base_geom":
                cat[g] = CAT["base"]
            elif n in ("palm", "finger_l_pad", "finger_r_pad"):
                cat[g] = CAT["palm"]
            elif n.startswith("obj"):
                slot = int(n[3:].split("_")[0])
                cat[g] = CAT["target"] if (slot == 0 and self.spec.task == 0) else CAT["clutter"]
            elif n.startswith("obs"):
                cat[g] = CAT["obstacle"]
        self.geom_cat = cat

    # --------------------------------------------------------------- helpers
    def _hand_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("lift_link").id].copy()

    def _target_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj_body[0]].copy() if 0 in self.obj_body else np.zeros(3)

    def _goal_dist(self) -> float:
        if self.spec.task != 1:
            return 0.0
        bx, by = self.data.qpos[self.qadr["base_x"]], self.data.qpos[self.qadr["base_y"]]
        return float(np.hypot(self.spec.env["goal_distance"] - bx, 0.0 - by))

    def _settle_check(self) -> Dict[str, Any]:
        d, m = self.data, self.model
        finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))) and not self._bad_qacc()
        con = self.get_contacts()
        vel = 0.0
        for slot in self.obj_body:
            adr = int(m.joint(f"obj{slot}_free").dofadr[0])
            vel = max(vel, float(np.linalg.norm(d.qvel[adr:adr + 6])) if finite else np.inf)
        support = S.TABLE_TOP_Z if self.spec.task == 0 else 0.0
        on_support = True
        for o in self.spec.objects:
            z = d.xpos[self.obj_body[o.slot]][2]
            on_support &= bool(z > support - 0.005)
        return {"finite": finite, "max_penetration": con["max_penetration"], "max_obj_speed": vel,
                "on_support": on_support, "robot_contacts": con["n_robot_contacts_initial"]}

    # -------------------------------------------------------------- contacts
    def get_contacts(self) -> Dict[str, Any]:
        d = self.data
        n = d.ncon
        out = {"n_contacts": int(n), "max_penetration": 0.0, "robot_obstacle": 0, "robot_clutter": 0,
               "finger_target": 0, "n_robot_contacts_initial": 0, "pairs": []}
        if n == 0:
            return out
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        c1, c2 = self.geom_cat[g1], self.geom_cat[g2]
        dist = d.contact.dist[:n]
        out["max_penetration"] = float(max(0.0, -dist.min()))
        robot = lambda c: (c == CAT["base"]) | (c == CAT["palm"])  # noqa: E731
        ro = (robot(c1) & (c2 == CAT["obstacle"])) | (robot(c2) & (c1 == CAT["obstacle"]))
        rc = (robot(c1) & (c2 == CAT["clutter"])) | (robot(c2) & (c1 == CAT["clutter"]))
        ft = ((c1 == CAT["palm"]) & (c2 == CAT["target"])) | ((c2 == CAT["palm"]) & (c1 == CAT["target"]))
        rt = (robot(c1) & (c2 == CAT["target"])) | (robot(c2) & (c1 == CAT["target"]))
        out["robot_obstacle"] = int(ro.sum())
        out["robot_clutter"] = int(rc.sum())
        out["finger_target"] = int(ft.sum())
        out["n_robot_contacts_initial"] = int((ro | rc | rt).sum())
        return out

    # ------------------------------------------------------------ observation
    def _compute_observation(self) -> np.ndarray:
        d = self.data
        q = [d.qpos[self.qadr[n]] for n in ROBOT_JOINTS]
        v = [d.qvel[self.vadr[n]] for n in ROBOT_JOINTS]
        base_xy = np.array(q[:2])
        yaw = q[2]
        obs = np.zeros(OBS_DIM)
        obs[0:7], obs[7:14], obs[14] = q, v, self.t
        if self.spec.task == 0 and 0 in self.obj_body:
            valid, est = self.perception.observe(base_xy, yaw, self._target_pos()[None, :2])
            obs[15] = float(valid[0])
            obs[16:18] = est[0] if valid[0] else 0.0
        else:
            pts = np.array([d.xpos[b][:2] for b in self.obs_body.values()]).reshape(-1, 2)
            valid, est = self.perception.observe(base_xy, yaw, pts)
            for i in range(min(len(pts), S.M_OBS)):
                o = 18 + 3 * i
                obs[o] = float(valid[i])
                obs[o + 1:o + 3] = est[i] if valid[i] else 0.0
        obs[24] = self.spec.env["goal_distance"] if self.spec.task == 1 else 0.0
        obs[25] = 0.0
        obs[26 + self.spec.task] = 1.0
        return obs

    def get_observation(self) -> np.ndarray:
        return self._obs_cache.copy()

    def get_state(self) -> Dict[str, np.ndarray]:
        d = self.data
        flat = np.zeros(STATE_DIM)
        flat[0:7] = [d.qpos[self.qadr[n]] for n in ROBOT_JOINTS]
        flat[7:14] = [d.qvel[self.vadr[n]] for n in ROBOT_JOINTS]
        for slot, b in self.obj_body.items():
            o = 14 + 7 * slot
            flat[o:o + 3] = d.xpos[b]
            flat[o + 3:o + 7] = d.xquat[b]
        base = 14 + 7 * S.K_OBJ
        for slot, (qa, qb) in self.obs_qadr.items():
            va, vb = self.obs_vadr[slot]
            flat[base + 4 * slot:base + 4 * slot + 4] = [d.xpos[self.obs_body[slot]][0], d.xpos[self.obs_body[slot]][1], d.qvel[va], d.qvel[vb]]
        return {"flat": flat, "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "time": np.array([self.t])}

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._last_action = a
        d, m = self.data, self.model
        ctrl = a * ACT_SCALE
        d.ctrl[self.act_id["a_bx"]] = ctrl[0]
        d.ctrl[self.act_id["a_by"]] = ctrl[1]
        d.ctrl[self.act_id["a_byaw"]] = ctrl[2]
        d.ctrl[self.act_id["a_reach"]] = np.clip(ctrl[3], 0, 0.55)
        d.ctrl[self.act_id["a_lift"]] = np.clip(ctrl[4], 0, 0.35)
        g = np.clip(ctrl[5], 0, 0.07)
        d.ctrl[self.act_id["a_fl"]] = g
        d.ctrl[self.act_id["a_fr"]] = g
        self.events.apply(m, d, self.t, self.cfg.control_dt)
        mujoco.mj_step(m, d, nstep=self.n_sub)
        self.t += self.cfg.control_dt
        self.step_count += 1
        con = self.get_contacts()
        mt = self.metrics
        mt["max_penetration"] = max(mt["max_penetration"], con["max_penetration"])
        mt["robot_obstacle_contact_steps"] += int(con["robot_obstacle"] > 0)
        mt["robot_clutter_contact_steps"] += int(con["robot_clutter"] > 0)
        mt["finger_target_contact_steps"] += int(con["finger_target"] > 0)
        mt["event_triggered"] = bool(self.events.triggered)
        finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))) and not self._bad_qacc()
        mt["nan_detected"] = mt["nan_detected"] or not finite
        mt["max_speed"] = max(mt["max_speed"], float(np.max(np.abs(d.qvel))) if finite else np.inf)
        terminated, truncated, reward = False, False, 0.0
        reason = "running"
        if not finite:
            terminated, reason = True, "numerical_instability"
        elif self.spec.task == 0:
            tp, hp = self._target_pos(), self._hand_pos()
            mt["max_object_height"] = max(mt["max_object_height"], float(tp[2] - S.TABLE_TOP_Z))
            dxy = float(np.hypot(*(tp[:2] - hp[:2])))
            reward = -0.5 * dxy + 2.0 * max(0.0, float(tp[2] - S.TABLE_TOP_Z))
            if tp[2] < S.TABLE_TOP_Z - 0.03 or dxy > 0.6:
                terminated, reason = True, "object_dropped"
                mt["object_dropped"] = True
                reward -= 5.0
            elif self.t >= self.t_max - 1e-9:
                ok = bool(tp[2] > S.TABLE_TOP_Z + 0.07 and dxy < 0.06 and abs(tp[2] - hp[2]) < 0.09)
                terminated, reason = True, ("success" if ok else "grasp_failed")
                mt["success"] = ok
                reward += 10.0 if ok else -2.0
        else:
            dist = self._goal_dist()
            reward = 2.0 * (self._prev_goal_dist - dist) - 0.01
            self._prev_goal_dist = dist
            if con["robot_obstacle"] > 0:
                terminated, reason = True, "collision_obstacle"
                reward -= 5.0
            elif dist < 0.2:
                terminated, reason = True, "success"
                mt["success"] = True
                reward += 10.0
            elif self.t >= self.t_max - 1e-9:
                truncated, reason = True, "timeout"
        if terminated or truncated:
            mt["termination_reason"] = reason
        self._obs_cache = self._compute_observation()
        info = {"termination_reason": reason, "t": self.t, "contacts": con, "event_triggered": mt["event_triggered"]}
        return self._obs_cache.copy(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- metrics etc
    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)

    def validate_scenario(self, scenario: S.ScenarioSpec, dry_run: bool = True) -> ValidationResult:
        """Simulation-level validation: settle, penetration, stability, and dry-run of the event schedule
        with the robot idle."""
        reasons, details = [], {}
        try:
            self.reset(scenario)
        except Exception as e:  # compile errors => invalid scenario
            return ValidationResult(False, ["MJCF_COMPILE_ERROR"], {"error": str(e)[:200]})
        st = self.settle_stats
        details.update(st)
        tol = self.cfg.penetration_tol
        if not st["finite"]:
            reasons.append("NAN_STATE")
        if st["max_penetration"] > tol:
            reasons.append("INITIAL_PENETRATION")
        if st["robot_contacts"] > 0:
            reasons.append("INITIAL_ROBOT_CONTACT")
        if st["max_obj_speed"] > 0.05:
            reasons.append("UNSTABLE_SETTLE")
        if not st["on_support"]:
            reasons.append("OFF_SUPPORT")
        if dry_run and not reasons:
            horizon = min(self.t_max, scenario.glob["event_onset"] + scenario.glob["event_duration"] + 0.6)
            idle = np.zeros(ACT_DIM)
            moved = 0.0
            p0 = {s: self.data.xpos[b].copy() for s, b in self.obj_body.items()}
            ob0 = {s: np.array([self.data.qpos[a], self.data.qpos[b]]) for s, (a, b) in self.obs_qadr.items()}
            while self.t < horizon:
                self.step(idle)
                if self.metrics["nan_detected"]:
                    break
            if self.metrics["nan_detected"]:
                reasons.append("EVENT_DRYRUN_UNSTABLE")
            elif self.metrics["max_speed"] > MAX_SPEED:
                reasons.append("EVENT_DRYRUN_UNSTABLE")
            if S.EVENTS[scenario.event] == "moving_obstacle":
                for s, (a, b) in self.obs_qadr.items():
                    moved = max(moved, float(np.linalg.norm(np.array([self.data.qpos[a], self.data.qpos[b]]) - ob0[s])))
                details["obstacle_displacement"] = moved
                if moved < 0.1:
                    reasons.append("EVENT_INFEASIBLE")
            details["dryrun_max_speed"] = self.metrics["max_speed"]
        return ValidationResult(not reasons, reasons, details)

    def export_episode(self) -> Dict[str, Any]:
        return {"scenario": self.spec.to_dict(), "metrics": self.get_metrics(), "settle": self.settle_stats,
                "final_state": self.get_state()["flat"].tolist(), "sim_time": self.t, "steps": self.step_count}
