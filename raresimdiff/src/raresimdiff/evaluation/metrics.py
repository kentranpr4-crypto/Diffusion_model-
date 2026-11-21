"""Scenario-set quality metrics: validity, conditioning accuracy, diversity, rare-event coverage, novelty, difficulty."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from ..scenarios import schema as S
from ..scenarios.codec import pack_spec, to_unit
from ..scenarios.severity import compute_severity

SIM_INSTABILITY_CODES = {"NAN_STATE", "EVENT_DRYRUN_UNSTABLE", "UNSTABLE_SETTLE", "EPISODE_UNSTABLE"}


def event_is_active(spec: S.ScenarioSpec) -> bool:
    """Does the scenario actually exhibit the requested event type (not just carry the label)?"""
    g, name = spec.glob, S.EVENTS[spec.event]
    slip = g["friction_multiplier"] < 0.95 or g["disturbance_force"] > 0.5
    perc = (g["sensor_noise_std"] > 0.002 or g["occlusion_dropout_prob"] > 0.03 or g["visibility_range"] < 2.8 or
            abs(g["calib_yaw_err"]) > 0.25 or abs(g["calib_tx"]) > 0.0025 or abs(g["calib_ty"]) > 0.0025)
    if name == "nominal":
        return not (slip or perc) and not spec.obstacles
    if name == "object_slip":
        return slip
    if name == "perception_degradation":
        return perc
    if name == "slip_and_perception":
        return slip and perc
    if name == "moving_obstacle":
        return len(spec.obstacles) > 0 and any(np.hypot(o.vx, o.vy) > 0.3 for o in spec.obstacles)
    return False


def key_features(spec: S.ScenarioSpec) -> np.ndarray:
    """4 normalised (0..1) features used to form a coverage grid per event type."""
    g, name = spec.glob, S.EVENTS[spec.event]
    tgt = next((o for o in spec.objects if o.slot == 0), None)
    mass = (tgt.mass - 0.05) / 1.15 if tgt else 0.0
    fr = (tgt.friction - 0.05) / 0.95 if tgt else 0.0
    if name in ("object_slip", "slip_and_perception"):
        return np.array([(1 - g["friction_multiplier"]) / 0.95, g["disturbance_force"] / 12.0, mass, 1 - fr])
    if name == "perception_degradation":
        return np.array([g["sensor_noise_std"] / 0.04, g["occlusion_dropout_prob"] / 0.7, (3 - g["visibility_range"]) / 2.2,
                         np.hypot(g["calib_tx"], g["calib_ty"]) / 0.0707])
    if name == "moving_obstacle" and spec.obstacles:
        o = spec.obstacles[0]
        return np.array([(np.hypot(o.vx, o.vy) - 0.0) / 1.9, (o.half_size - 0.12) / 0.28, (g["event_onset"] - 0.3) / 3.7,
                         (abs(o.y0) - 0.9) / 1.3])
    return np.zeros(4)


def cell_ids(feats: np.ndarray, bins: int = 4) -> np.ndarray:
    q = np.clip((feats * bins).astype(int), 0, bins - 1)
    return (q * (bins ** np.arange(q.shape[1]))[None]).sum(1)


def unit_vectors(specs: Sequence[S.ScenarioSpec]) -> np.ndarray:
    """Flattened bound-normalised vectors restricted to channels active for the requested event."""
    vs = []
    for sp in specs:
        phys, mask, _ = pack_spec(sp)
        u = to_unit(phys) * mask[:, None]
        active = S.active_global_channels(sp.event)
        for v in S.GLOBAL_VARS:
            if v.name not in active:
                u[0, S.GLOBAL_IDX[v.name]] = 0.0
        vs.append(u.reshape(-1))
    return np.asarray(vs, np.float32)


def nn_distance(a: np.ndarray, ref: np.ndarray, exclude_self: bool = False, batch: int = 512) -> np.ndarray:
    """Mean-absolute-difference (per channel) nearest neighbour distance from each row of a to ref."""
    ta, tr = torch.from_numpy(a), torch.from_numpy(ref)
    out = []
    for i in range(0, len(ta), batch):
        d = torch.cdist(ta[i:i + batch], tr, p=1) / ta.shape[1]
        if exclude_self:
            idx = torch.arange(i, min(i + batch, len(ta)))
            d[torch.arange(len(idx)), idx] = float("inf")
        out.append(d.min(1).values)
    return torch.cat(out).numpy()


def pairwise_diversity(u: np.ndarray, max_n: int = 600, seed: int = 0) -> float:
    if len(u) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(u) > max_n:
        u = u[rng.choice(len(u), max_n, replace=False)]
    t = torch.from_numpy(u)
    d = torch.cdist(t, t, p=1) / t.shape[1]
    n = len(t)
    return float(d.sum() / (n * (n - 1)))


def set_metrics(items: List[tuple], requested: np.ndarray, train_cells: Optional[set], train_u: Optional[np.ndarray],
                rare_cell_ids: Optional[set]) -> Dict:
    """items: list of (spec, info) from sim_eval. requested: requested severity per item."""
    n = len(items)
    acc = [(s, i, r) for (s, i), r in zip(items, requested) if i["accepted"]]
    reasons: Dict[str, int] = {}
    for _, i in items:
        for r in i["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    out: Dict = {"n": n, "n_accepted": len(acc), "valid_ratio": len(acc) / max(n, 1), "reject_reasons": reasons,
                 "sim_instability_rate": sum(1 for _, i in items if SIM_INSTABILITY_CODES & set(i["reasons"])) / max(n, 1),
                 "repair_mean": float(np.mean([i["repair"] for _, i in items])) if n else 0.0}
    if not acc:
        return out
    specs = [s for s, _, _ in acc]
    sev_err = [abs(compute_severity(s.event, s.glob, [o.__dict__ for o in s.obstacles]) - r) for s, _, r in acc]
    out["condition_event_accuracy"] = float(np.mean([event_is_active(s) for s in specs]))
    out["condition_severity_mae"] = float(np.mean(sev_err))
    out["condition_severity_within_0.1"] = float(np.mean(np.array(sev_err) <= 0.1))
    u = unit_vectors(specs)
    out["diversity_mean_l1"] = pairwise_diversity(u)
    cells = cell_ids(np.stack([key_features(s) for s in specs]))
    out["coverage_cells_occupied"] = int(len(set(cells.tolist())))
    out["coverage_cells_fraction"] = out["coverage_cells_occupied"] / 4 ** 4
    if rare_cell_ids is not None:
        out["rare_cell_hits"] = int(len(set(cells.tolist()) & rare_cell_ids))
        out["rare_cell_coverage"] = out["rare_cell_hits"] / max(len(rare_cell_ids), 1)
    if train_u is not None and len(train_u):
        d = nn_distance(u, train_u)
        out["novelty_nn_median"] = float(np.median(d))
        out["novelty_nn_p10"] = float(np.percentile(d, 10))
        out["memorization_rate"] = float(np.mean(d < 0.01))
    ro = [i for _, i, _ in acc if i["success"] is not None]
    if ro:
        out["baseline_failure_rate"] = float(1 - np.mean([i["success"] for i in ro]))
        out["baseline_obstacle_collision_rate"] = float(np.mean([i["robot_obstacle_steps"] > 0 for i in ro]))
        out["event_triggered_rate"] = float(np.mean([i["event_triggered"] for i in ro]))
    return out
