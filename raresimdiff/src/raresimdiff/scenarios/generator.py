"""Deterministic procedural scenario generator (no learned model involved).

A *parent scene* fixes layout (assets, poses, masses, frictions, environment). *Variants* of a parent
add one event (slip / moving obstacle / perception degradation / combination) at a requested severity.
Splits are made at parent level to prevent leakage.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from . import schema as S
from .severity import compute_severity
from ..utils.seeding import rng_from

CLUTTER_ASSETS = (1, 2, 3, 4, 5)
FLOOR_ASSETS = (1, 2, 4, 5)

DEFAULT_EVENT_WEIGHTS = {
    0: {"nominal": 0.10, "object_slip": 0.35, "perception_degradation": 0.30, "slip_and_perception": 0.25},
    1: {"nominal": 0.15, "moving_obstacle": 0.85},
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _random_rot(rng, max_tilt: float = 0.10) -> List[float]:
    R = S.rot_z(rng.uniform(-np.pi, np.pi)) @ S.rot_x(rng.uniform(-max_tilt, max_tilt)) @ S.rot_y(
        rng.uniform(-max_tilt, max_tilt))
    return [float(a) for a in R.reshape(-1)]


def make_parent(seed: int, base_seed: int = 0, ood: bool = False, task: Optional[int] = None,
                max_objects: int = S.K_OBJ) -> dict:
    """Create a parent scene. Returns a JSON-friendly dict."""
    rng = rng_from("parent", base_seed, seed)
    if task is None:
        task = 0 if rng.random() < 0.65 else 1
    parent_id = f"p{base_seed:03d}-{seed:08d}"
    env = {"goal_distance": 0.0, "surface_friction": float(rng.uniform(0.3, 0.7)), "clutter_level": 0.0}
    objects: List[S.ObjectSpec] = []

    def sample_obj_phys(ood_flag: bool, target: bool):
        if ood_flag and target:
            mass, fr = rng.uniform(0.85, 1.15), rng.uniform(0.05, 0.20)
        else:
            mass = float(np.clip(rng.lognormal(np.log(0.25), 0.55), 0.06, 0.75))
            fr = rng.uniform(0.3, 0.95)
            if mass > 0.5:
                fr = max(fr, 0.45)
        return float(mass), float(fr)

    if task == 0:
        n_clutter = int(rng.integers(4, 6)) if ood else int(rng.integers(0, 4))
        n_clutter = min(n_clutter, max_objects - 1)
        aid = int(rng.choice(S.TARGET_ASSETS))
        scale = float(rng.uniform(0.8, 1.25))
        mass, fr = sample_obj_phys(ood, True)
        objects.append(S.ObjectSpec(0, aid, float(rng.uniform(0.58, 0.82)), float(rng.uniform(-0.2, 0.2)),
                                    float(rng.uniform(0.0, 0.008)), _random_rot(rng, 0.06), scale, mass, fr))
        for k in range(1, 1 + n_clutter):
            for _ in range(60):
                a = int(rng.choice(CLUTTER_ASSETS))
                sc = float(rng.uniform(0.75, 1.25))
                r = S.asset_footprint_radius(a, sc)
                x = float(rng.uniform(0.5 + r + 0.01, 1.2 - r - 0.01))
                y = float(rng.uniform(-(0.55 - r - 0.01), 0.55 - r - 0.01))
                ok = all(np.hypot(x - o.x, y - o.y) >= r + S.asset_footprint_radius(o.asset_id, o.scale) + 0.02
                         for o in objects)
                # keep clutter out of the finger swept corridor in front of the target
                tx, ty = objects[0].x, objects[0].y
                swept = (x < tx + 0.04) and abs(y - ty) < (0.09 + r)
                if ok and not swept:
                    m, f = sample_obj_phys(False, False)
                    objects.append(S.ObjectSpec(k, a, x, y, float(rng.uniform(0, 0.008)), _random_rot(rng, 0.06),
                                                sc, m, f))
                    break
        env["clutter_level"] = (len(objects) - 1) / (S.K_OBJ - 1)
    else:
        goal = float(rng.uniform(2.5, 4.5))
        env["goal_distance"] = goal
        env["surface_friction"] = float(rng.uniform(0.4, 0.8))
        n = int(rng.integers(4, 6)) if ood else int(rng.integers(0, 4))
        for k in range(n):
            for _ in range(60):
                a = int(rng.choice(FLOOR_ASSETS))
                sc = float(rng.uniform(0.8, 1.3))
                x = float(rng.uniform(0.9, goal - 0.5))
                y = float(rng.choice([-1, 1]) * rng.uniform(0.55, 1.5))
                r = S.asset_footprint_radius(a, sc)
                if all(np.hypot(x - o.x, y - o.y) >= r + S.asset_footprint_radius(o.asset_id, o.scale) + 0.03
                       for o in objects):
                    m = float(rng.uniform(0.3, 1.2))
                    objects.append(S.ObjectSpec(k, a, x, y, float(rng.uniform(0, 0.01)), _random_rot(rng, 0.05),
                                                sc, m, float(rng.uniform(0.3, 0.9))))
                    break
        env["clutter_level"] = len(objects) / S.K_OBJ
    return {"parent_id": parent_id, "task": task, "ood": bool(ood), "env": env,
            "objects": [o.__dict__ for o in objects], "seed": int(seed), "base_seed": int(base_seed)}


# --- event parameter sampling ----------------------------------------------------------------
def _slip_params(rng, s: float) -> Dict[str, float]:
    th = rng.uniform(-np.pi, np.pi)
    jf = _clip01(s + rng.normal(0, 0.10))
    jm = _clip01(s + rng.normal(0, 0.08))
    return {
        "event_onset": float(rng.uniform(1.5, 3.2)), "event_duration": float(rng.uniform(0.2, 1.0)),
        "disturbance_force": 12.0 * jf, "disturbance_dir_sin": float(np.sin(th)),
        "disturbance_dir_cos": float(np.cos(th)), "friction_multiplier": 1.0 - 0.95 * jm,
    }


def _perception_params(rng, s: float) -> Dict[str, float]:
    j = lambda: _clip01(s + rng.normal(0, 0.12))  # noqa: E731
    return {
        "sensor_noise_std": 0.04 * j(), "occlusion_dropout_prob": 0.7 * j(),
        "visibility_range": 3.0 - 2.2 * j(),
        "calib_yaw_err": float(rng.choice([-1, 1]) * 5.0 * j()),
        "calib_tx": float(rng.choice([-1, 1]) * 0.05 * j()), "calib_ty": float(rng.choice([-1, 1]) * 0.05 * j()),
    }


def _obstacles(rng, s: float, goal: float, objects=()):
    """Crossing obstacles. Speed is raised if needed so the obstacle fully crosses (and clears) the corridor
    within the maximum event duration (physically interpretable pedestrian/cart crossing)."""
    n = 1 if rng.random() < 0.7 else 2
    obs: List[S.ObstacleSpec] = []
    onset_ref = None
    dur = 0.1
    for k in range(n):
        speed = 0.4 + 1.4 * _clip01(s + rng.normal(0, 0.10))
        hs = 0.12 + 0.28 * _clip01(s + rng.normal(0, 0.15))
        for _try in range(30):
            x0 = float(rng.uniform(1.2, max(1.3, goal - 0.6)))
            y0 = float(rng.choice([-1, 1]) * rng.uniform(0.9, 1.5))
            if all(abs(x0 - o.x) >= hs + 0.15 or abs(y0 - o.y) >= hs + 0.15 for o in objects) and \
               all(abs(x0 - q.x0) >= hs + q.half_size + 0.1 or abs(y0 - q.y0) >= hs + q.half_size + 0.1 for q in obs):
                break
        ang = rng.uniform(-0.25, 0.25)
        clear_dist = abs(y0) + hs + 0.55
        speed = min(1.78, max(speed, clear_dist / 1.4))
        vy = float(-np.sign(y0) * speed * np.cos(ang))
        vx = float(np.clip(speed * np.sin(ang), -0.6, 0.6))
        vy = float(np.clip(vy, -1.8, 1.8))
        t_cross = max(abs(y0) - 0.3, 0.05) / max(abs(vy), 1e-3)
        t_robot = x0 / 0.8
        mismatch = (1.0 - _clip01(s + rng.normal(0, 0.12))) * 2.2
        onset = float(np.clip(t_robot - t_cross + rng.choice([-1, 1]) * mismatch + rng.uniform(-0.2, 0.2), 0.3, 4.0))
        if onset_ref is None:
            onset_ref = onset
        dur = max(dur, float(np.clip(clear_dist / max(abs(vy), 1e-3) + 0.05, 0.1, 1.5)))
        obs.append(S.ObstacleSpec(k, x0, y0, vx, vy, float(hs), float(rng.uniform(0, 0.6)) if k else 0.0))
    return obs, float(onset_ref), float(min(dur, 1.5))


def make_variant(parent: dict, event: int, severity: float, seed: int, variant_idx: int = 0) -> S.ScenarioSpec:
    rng = rng_from("variant", parent["parent_id"], event, variant_idx, seed)
    g = S.nominal_global()
    obstacles: List[S.ObstacleSpec] = []
    name = S.EVENTS[event]
    if name in ("object_slip", "slip_and_perception"):
        g.update(_slip_params(rng, severity))
    if name in ("perception_degradation", "slip_and_perception"):
        g.update(_perception_params(rng, severity))
    if name == "moving_obstacle":
        objects_ = [S.ObjectSpec(**o) for o in parent["objects"]]
        obstacles, onset, dur = _obstacles(rng, severity, parent["env"]["goal_distance"], objects_)
        g["event_onset"], g["event_duration"] = onset, dur
    objects = [S.ObjectSpec(**o) for o in parent["objects"]]
    sev = compute_severity(event, g, [o.__dict__ for o in obstacles])
    sid = f"{parent['parent_id']}-e{event}-v{variant_idx}"
    return S.ScenarioSpec(
        scenario_id=sid, parent_id=parent["parent_id"], seed=int(rng.integers(0, 2**31 - 1)),
        task=parent["task"], event=event, robot_profile=0, severity=sev, env=dict(parent["env"]), glob=g,
        objects=objects, obstacles=obstacles,
        provenance={"generator": "procedural_v1", "parent_seed": parent["seed"], "target_severity": float(severity),
                    "ood_parent": parent["ood"], "base_seed": parent["base_seed"]})


def sample_event(rng, task: int, weights: Optional[dict] = None) -> int:
    w = (weights or DEFAULT_EVENT_WEIGHTS)[task]
    names = list(w)
    p = np.array([w[n] for n in names], float)
    return S.EVENTS.index(names[int(rng.choice(len(names), p=p / p.sum()))])


def variants_for_parent_index(i: int, variants_per_parent: int, base_seed: int, ood_fraction: float = 0.08,
                              weights: Optional[dict] = None):
    """All variants of parent #i. Deterministic in (i, variants_per_parent, base_seed)."""
    rng = rng_from("plan", base_seed, i)
    ood = bool(rng.random() < ood_fraction)
    parent = make_parent(i, base_seed=base_seed, ood=ood)
    out = []
    used = set()
    for v in range(variants_per_parent):
        for _ in range(20):
            ev = sample_event(rng, parent["task"], weights)
            if ev != 0 or 0 not in used:
                break
        else:
            ev = [e for e in range(1, len(S.EVENTS)) if parent["task"] in S.EVENT_TASKS[S.EVENTS[e]]][0]
        used.add(ev)
        sev = float(rng.uniform(0.0, 1.0))
        if ood and ev != 0:
            sev = float(rng.uniform(0.7, 1.0))
        out.append(make_variant(parent, ev, sev, seed=base_seed, variant_idx=v))
    return out


def generate_parents_and_variants(n_parents: int, variants_per_parent: int, base_seed: int,
                                  ood_fraction: float = 0.08, weights: Optional[dict] = None):
    for i in range(n_parents):
        yield from variants_for_parent_index(i, variants_per_parent, base_seed, ood_fraction, weights)


def generate_conditioned(event: int, severity: float, n: int, base_seed: int = 0, task: Optional[int] = None):
    """Procedural baseline sampler for a given (event, severity): used for comparison with the learned model."""
    out = []
    ptask = S.EVENT_TASKS[S.EVENTS[event]]
    for i in range(n):
        rng = rng_from("cond", base_seed, event, round(severity, 3), i)
        t = ptask[int(rng.integers(0, len(ptask)))] if task is None else task
        parent = make_parent(int(rng.integers(0, 2**31 - 1)), base_seed=base_seed + 1000, ood=False, task=t)
        spec = make_variant(parent, event, severity, seed=int(rng.integers(0, 2**31 - 1)), variant_idx=i)
        spec.scenario_id = f"cond-{event}-{int(severity * 1000):04d}-{i:05d}"
        out.append(spec)
    return out
