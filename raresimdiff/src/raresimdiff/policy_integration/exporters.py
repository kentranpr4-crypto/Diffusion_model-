"""Export rollouts as RL/IL-ready transitions. Actions are always those actually executed by a real policy/controller
(never fabricated). `is_expert_demo` is False unless the caller explicitly attests the policy is an expert source."""
from __future__ import annotations
import gzip, json
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np

from ..scenarios import schema as S
from ..simulation.mujoco_backend import MuJoCoBackend
from ..simulation.rollout import run_episode
from ..utils.hashing import sha256_file
from ..utils.io import write_json
from .policies import POLICIES

FIELDS = ("observation", "action", "reward", "next_observation", "terminated", "truncated", "scenario_id", "event_type",
          "severity", "is_synthetic")


def load_scenarios_jsonl(path: str | Path) -> List[S.ScenarioSpec]:
    op = gzip.open if str(path).endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if "scenario" in d:  # generate.py format
                if d.get("validation", {}).get("accepted") is False:
                    continue
                d = d["scenario"]
            d = {k: v for k, v in d.items() if k in S.ScenarioSpec.__dataclass_fields__}
            out.append(S.ScenarioSpec.from_dict(d))
    return out


def export_transitions(specs: Iterable[S.ScenarioSpec], policy, out_path: str | Path, attest_expert: bool = False) -> Dict:
    backend = MuJoCoBackend()
    cols: Dict[str, list] = {k: [] for k in FIELDS}
    extra = {"episode_index": [], "action_source": [], "is_expert_demo": [], "source_generator": []}
    n_ep, n_succ = 0, 0
    for k, spec in enumerate(specs):
        ep = run_episode(backend, spec, policy, record_transitions=True)
        tr = ep.transitions
        T = len(tr["reward"])
        for key in ("observation", "action", "reward", "next_observation", "terminated", "truncated"):
            cols[key].append(tr[key])
        cols["scenario_id"] += [spec.scenario_id] * T
        cols["event_type"] += [S.EVENTS[spec.event]] * T
        cols["severity"] += [spec.severity] * T
        cols["is_synthetic"] += [True] * T  # all simulator-derived data is synthetic (no real-world data)
        extra["episode_index"] += [k] * T
        extra["action_source"] += [policy.name] * T
        extra["is_expert_demo"] += [bool(attest_expert)] * T
        extra["source_generator"] += [str(spec.provenance.get("generator", "unknown"))] * T
        n_ep += 1
        n_succ += int(ep.success)
    arrs = {k: np.concatenate(v) if k in ("observation", "action", "reward", "next_observation", "terminated", "truncated")
            and v else np.asarray(v) for k, v in cols.items()}
    arrs = {k: (v.astype(np.float32) if k in ("observation", "action", "reward", "next_observation") else v) for k, v in arrs.items()}
    arrs["scenario_id"] = np.asarray(cols["scenario_id"], dtype="U64")
    arrs["event_type"] = np.asarray(cols["event_type"], dtype="U32")
    arrs["severity"] = np.asarray(cols["severity"], np.float32)
    arrs["is_synthetic"] = np.asarray(cols["is_synthetic"], bool)
    for k, v in extra.items():
        arrs[k] = np.asarray(v, dtype=bool if k == "is_expert_demo" else ("int64" if k == "episode_index" else "U48"))
    arrs["terminated"] = arrs["terminated"].astype(bool)
    arrs["truncated"] = arrs["truncated"].astype(bool)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrs)
    meta = {"path": str(out_path), "n_episodes": n_ep, "n_transitions": int(len(arrs["reward"])),
            "policy": policy.name, "is_expert_demo": bool(attest_expert), "episode_success_rate": n_succ / max(n_ep, 1),
            "sha256": sha256_file(out_path)}
    write_json(str(out_path) + ".meta.json", meta)
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--policy", default="scripted_baseline", choices=list(POLICIES))
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-scenarios", type=int, default=None)
    a = ap.parse_args(argv)
    specs = load_scenarios_jsonl(a.scenarios)[: a.max_scenarios]
    print(json.dumps(export_transitions(specs, POLICIES[a.policy](), a.output), indent=2))


if __name__ == "__main__":
    main()
