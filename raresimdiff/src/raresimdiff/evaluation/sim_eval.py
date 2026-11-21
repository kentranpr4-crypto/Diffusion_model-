"""Run validation + baseline-policy rollouts over many scenarios (optionally in parallel)."""
from __future__ import annotations
import multiprocessing as mp
from typing import List

from ..simulation.controllers import ScriptedBaseline
from ..simulation.mujoco_backend import MuJoCoBackend
from ..simulation.rollout import run_episode
from ..validation.pipeline import Candidate, validate_candidates


def _work(args):
    cands, rollout, repair_threshold = args
    backend = MuJoCoBackend()
    res = validate_candidates(cands, backend, simulate=True, repair_threshold=repair_threshold, dedupe=False)
    ctl = ScriptedBaseline()
    out = []
    for r in res:
        d = {"accepted": r.accepted, "reasons": r.reasons, "repair": r.repair, "stage": r.stage, "success": None,
             "termination": None, "robot_obstacle_steps": 0, "robot_clutter_steps": 0, "event_triggered": False}
        if r.accepted and rollout:
            ep = run_episode(backend, r.spec, ctl)
            m = ep.metrics
            d.update({"success": bool(ep.success), "termination": ep.termination_reason,
                      "robot_obstacle_steps": m["robot_obstacle_contact_steps"],
                      "robot_clutter_steps": m["robot_clutter_contact_steps"], "event_triggered": bool(m["event_triggered"]),
                      "nan": bool(m["nan_detected"])})
        out.append((r.spec, d))
    return out


def simulate_candidates(cands: List[Candidate], workers: int = 1, rollout: bool = True,
                        repair_threshold: float = 0.35, dedupe: bool = True):
    """Returns list of (spec, info dict) in the input order. Duplicate detection is done up-front (global)."""
    from ..validation.pipeline import scenario_fingerprint
    dup = [False] * len(cands)
    if dedupe:
        seen = set()
        for i, c in enumerate(cands):
            fp = scenario_fingerprint(c.spec)
            dup[i] = fp in seen
            seen.add(fp)
    chunk = max(1, len(cands) // max(1, workers * 4))
    parts = [cands[i:i + chunk] for i in range(0, len(cands), chunk)]
    jobs = [(p, rollout, repair_threshold) for p in parts]
    if workers <= 1:
        outs = [_work(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            outs = pool.map(_work, jobs)
    flat = [x for o in outs for x in o]
    final = []
    for (spec, info), d in zip(flat, dup):
        if d:
            info = dict(info, accepted=False, reasons=info["reasons"] + ["DUPLICATE"], stage="duplicate")
        final.append((spec, info))
    return final
