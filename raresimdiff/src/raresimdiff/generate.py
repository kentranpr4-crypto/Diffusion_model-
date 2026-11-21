"""Conditional scenario generation from a frozen checkpoint.

    python -m raresimdiff.generate --checkpoint runs/official/best.safetensors --event-type object_slip \
        --severity 0.8 --num-scenarios 1000 --validate-physics --output generated/slip_cases
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from .diffusion.gaussian import GaussianDiffusion
from .evaluation.sim_eval import simulate_candidates
from .scenarios import schema as S
from .training.export import load_generator
from .utils.io import write_json
from .validation.pipeline import Candidate, decode_candidates


def build_condition(prior, event: int, severity: float, n: int, rng: np.random.Generator, task: Optional[int] = None):
    c = prior.sample(event, n, rng, task)
    c["severity"] = np.full(n, severity, np.float32)
    return c


def support_warnings(support: Dict, event_name: str, severity: float) -> List[str]:
    w = []
    sup = support.get(event_name)
    if sup is None:
        w.append(f"OUT_OF_DISTRIBUTION: event '{event_name}' never appeared in training; samples are NOT reliable")
    elif not (sup["severity_min"] - 0.02 <= severity <= sup["severity_max"] + 0.02):
        w.append(f"OUT_OF_DISTRIBUTION: severity {severity:.2f} outside the training range "
                 f"[{sup['severity_min']:.2f}, {sup['severity_max']:.2f}] for '{event_name}'; reliability not claimed")
    return w


@torch.no_grad()
def sample_candidates(model, diff_cfg, normalizer, prior, event: int, severity: float, n: int, seed: int, device: str = "cpu",
                      ddim_steps: Optional[int] = None, cfg_scale: Optional[float] = None, task: Optional[int] = None,
                      prefix: str = "gen", batch: int = 1024) -> List[Candidate]:
    dev = torch.device(device)
    diffusion = GaussianDiffusion(diff_cfg.num_timesteps, diff_cfg.schedule, diff_cfg.prediction, 0.0, diff_cfg.clip_x0).to(dev)
    lo, hi = normalizer.x0_limits()
    lo, hi = torch.from_numpy(lo).to(dev), torch.from_numpy(hi).to(dev)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cands: List[Candidate] = []
    for b0 in range(0, n, batch):
        m = min(batch, n - b0)
        cond_np = build_condition(prior, event, severity, m, rng, task)
        cond = {k: torch.from_numpy(v).to(dev) for k, v in cond_np.items()}
        x = diffusion.ddim_sample(model, cond, lo, hi, steps=ddim_steps or diff_cfg.ddim_steps, eta=diff_cfg.ddim_eta,
                                  cfg_scale=cfg_scale if cfg_scale is not None else diff_cfg.cfg_scale, generator=gen)
        cands += decode_candidates(x.cpu().numpy(), cond_np, normalizer, prefix=f"{prefix}-{b0:06d}", seed=seed)
    for i, c in enumerate(cands):  # globally unique ids
        c.spec.scenario_id = f"{prefix}-{i:06d}"
    return cands


def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--event-type", required=True, choices=S.EVENTS)
    ap.add_argument("--severity", type=float, nargs="+", default=[0.5], help="one or more severities in [0,1]")
    ap.add_argument("--num-scenarios", type=int, default=100, help="per severity")
    ap.add_argument("--task", choices=S.TASKS, default=None)
    ap.add_argument("--validate-physics", action="store_true")
    ap.add_argument("--export-episodes", action="store_true", help="roll out the baseline policy on accepted scenarios")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    model, dcfg, norm, prior, support, meta = load_generator(a.checkpoint, a.device)
    event = S.EVENTS.index(a.event_type)
    task = S.TASKS.index(a.task) if a.task else None
    warns: List[str] = []
    all_cands: List[Candidate] = []
    for k, sev in enumerate(a.severity):
        if not 0.0 <= sev <= 1.0:
            raise SystemExit("severity must be in [0, 1]")
        for w in support_warnings(support, a.event_type, sev):
            warns.append(w)
            warnings.warn(w)
        all_cands += sample_candidates(model, dcfg, norm, prior, event, sev, a.num_scenarios, a.seed + 1000 * k, a.device,
                                       a.ddim_steps, a.cfg_scale, task, prefix=f"gen-{a.event_type}-s{int(sev * 100):03d}")
    summary: Dict = {"checkpoint": str(a.checkpoint), "checkpoint_run_id": meta.get("run_id"), "event_type": a.event_type,
                     "severities": a.severity, "requested": len(all_cands), "warnings": warns,
                     "reliability_note": "random-weight or untrained checkpoints carry no evidence of learned capability"}
    if a.validate_physics:
        res = simulate_candidates(all_cands, workers=a.workers, rollout=a.export_episodes)
        acc, rej = [], []
        reasons: Dict[str, int] = {}
        for (spec, info), c in zip(res, all_cands):
            rec = {"scenario": spec.to_dict(), "requested_severity": c.requested_severity, "repair": c.repair,
                   "validation": {"accepted": info["accepted"], "reasons": info["reasons"], "stage": info["stage"]}}
            if info["success"] is not None:
                rec["baseline_rollout"] = {k: info[k] for k in ("success", "termination", "robot_obstacle_steps",
                                                                "robot_clutter_steps", "event_triggered")}
            (acc if info["accepted"] else rej).append(rec)
            for r in info["reasons"]:
                reasons[r] = reasons.get(r, 0) + 1
        summary.update({"accepted": len(acc), "rejected": len(rej), "reject_reasons": reasons})
    else:
        acc = [{"scenario": c.spec.to_dict(), "requested_severity": c.requested_severity, "repair": c.repair,
                "validation": {"accepted": None, "reasons": [], "stage": "not_validated"}} for c in all_cands]
        rej = []
        summary.update({"accepted": None, "rejected": None, "note": "run with --validate-physics to validate"})
    if a.validate_physics and a.export_episodes and acc:
        from .policy_integration.exporters import export_transitions
        from .policy_integration.policies import ScriptedBaselinePolicy
        specs = [S.ScenarioSpec.from_dict(r["scenario"]) for r in acc]
        summary["episodes"] = export_transitions(specs, ScriptedBaselinePolicy(), out / "episodes.npz")
    with open(out / "scenarios_accepted.jsonl", "w") as f:
        for r in acc:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(out / "scenarios_rejected.jsonl", "w") as f:
        for r in rej:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in summary if k not in ("reliability_note",)}, indent=2))


if __name__ == "__main__":
    main()
