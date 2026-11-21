"""Decode -> project -> geometry validate -> simulation validate -> quality scoring inputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..scenarios import schema as S
from ..scenarios.codec import Normalizer, unpack_to_spec, to_unit
from ..simulation.mujoco_backend import MuJoCoBackend
from ..utils.hashing import stable_json_hash
from ..utils.seeding import derive_seed
from . import reasons as R
from .geometry import validate_geometry
from .projection import project


@dataclass
class Candidate:
    spec: S.ScenarioSpec
    repair: float
    requested_severity: float
    raw_out_of_bounds_frac: float = 0.0


@dataclass
class CandidateResult:
    spec: S.ScenarioSpec
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    repair: float = 0.0
    requested_severity: float = 0.0
    stage: str = "accepted"
    details: Dict = field(default_factory=dict)


def decode_candidates(x_std: np.ndarray, cond: Dict[str, np.ndarray], normalizer: Normalizer,
                      prefix: str = "gen", seed: int = 0) -> List[Candidate]:
    """x_std (N,L,D) model-space samples + condition arrays -> projected ScenarioSpec candidates."""
    raw_phys = normalizer.denormalize(x_std, clamp=False)
    out: List[Candidate] = []
    for i in range(len(x_std)):
        env = {v.name: float(cond["env"][i][j]) for j, v in enumerate(S.ENV_VARS)}
        u_raw = to_unit(raw_phys[i])
        w = cond["slot_mask"][i][:, None] * S.CH_MASK
        oob = float(((np.abs(u_raw) > 1.0 + 1e-6) * w).sum() / max(w.sum(), 1.0))
        p, mag, _ = project(raw_phys[i], cond["slot_mask"][i], cond["asset_ids"][i], int(cond["task"][i]),
                            int(cond["event"][i]), env)
        sid = f"{prefix}-{i:06d}"
        spec = unpack_to_spec(p, cond["slot_mask"][i], cond["asset_ids"][i], int(cond["task"][i]), int(cond["event"][i]),
                              int(cond["robot"][i]), env, sid, "generated", derive_seed(prefix, seed, i) % (2**31 - 1),
                              provenance={"generator": "RareScenarioDiT", "requested_severity": float(cond["severity"][i])})
        out.append(Candidate(spec, mag, float(cond["severity"][i]), oob))
    return out


def scenario_fingerprint(spec: S.ScenarioSpec, ndigits: int = 3) -> str:
    d = spec.to_dict()
    for k in ("scenario_id", "parent_id", "seed", "provenance"):
        d.pop(k, None)

    def rnd(o):
        if isinstance(o, float):
            return round(o, ndigits)
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o
    return stable_json_hash(rnd(d))


def validate_candidates(cands: List[Candidate], backend: Optional[MuJoCoBackend] = None, simulate: bool = True,
                        repair_threshold: float = 0.35, dedupe: bool = True) -> List[CandidateResult]:
    backend = backend or (MuJoCoBackend() if simulate else None)
    seen = set()
    out: List[CandidateResult] = []
    for c in cands:
        reasons = list(validate_geometry(c.spec))
        stage = "geometry" if reasons else "accepted"
        if not reasons and c.repair > repair_threshold:
            reasons.append(R.REPAIR_TOO_LARGE)
            stage = "repair"
        if not reasons and dedupe:
            fp = scenario_fingerprint(c.spec)
            if fp in seen:
                reasons.append(R.DUPLICATE)
                stage = "duplicate"
            seen.add(fp)
        details: Dict = {}
        if not reasons and simulate:
            vr = backend.validate_scenario(c.spec, dry_run=True)
            details = vr.details
            if not vr.ok:
                reasons += vr.reasons
                stage = "simulation"
        out.append(CandidateResult(c.spec, not reasons, reasons, c.repair, c.requested_severity, stage, details))
    return out
