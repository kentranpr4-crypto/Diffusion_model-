"""Policy evaluation on scenario sets."""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np

from ..scenarios import schema as S
from ..simulation.mujoco_backend import MuJoCoBackend
from ..simulation.rollout import run_episode

SEV_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001))


def evaluate_policy(policy, specs: Sequence[S.ScenarioSpec], nominal_reference: Optional[float] = None) -> Dict:
    be = MuJoCoBackend()
    rows = []
    for sp in specs:
        ep = run_episode(be, sp, policy)
        m = ep.metrics
        rows.append({"event": S.EVENTS[sp.event], "sev": sp.severity, "success": bool(ep.success),
                     "triggered": bool(m["event_triggered"]),
                     "violation": bool(m["robot_obstacle_contact_steps"] > 0 or m["robot_clutter_contact_steps"] > 0)})
    n = len(rows)
    out: Dict = {"policy": policy.name, "n": n,
                 "task_completion_rate": float(np.mean([r["success"] for r in rows])) if n else None,
                 "contact_or_collision_violation_rate": float(np.mean([r["violation"] for r in rows])) if n else None}
    out["failure_rate_by_event_family"] = {
        e: float(1 - np.mean([r["success"] for r in rows if r["event"] == e]))
        for e in sorted(set(r["event"] for r in rows))}
    trig = [r for r in rows if r["triggered"] and r["event"] != "nominal"]
    out["recovery_success_rate"] = float(np.mean([r["success"] for r in trig])) if trig else None
    nom = [r for r in rows if r["event"] == "nominal"]
    out["nominal_success_rate"] = float(np.mean([r["success"] for r in nom])) if nom else None
    out["nominal_regression"] = (None if (nominal_reference is None or out["nominal_success_rate"] is None)
                                 else float(nominal_reference - out["nominal_success_rate"]))
    out["success_by_severity_bin"] = {
        f"{lo:.2f}-{min(hi, 1.0):.2f}": (float(np.mean([r["success"] for r in rows if lo <= r["sev"] < hi]))
                                         if any(lo <= r["sev"] < hi for r in rows) else None)
        for lo, hi in SEV_BINS}
    return out
