"""Post-training evaluation: learned generator vs procedural generator at matched conditions."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List
import numpy as np

from ..data.dataset import load_split
from ..generate import sample_candidates, support_warnings
from ..scenarios import schema as S
from ..scenarios.generator import generate_conditioned
from ..training.export import load_generator
from ..utils.io import write_json
from ..validation.pipeline import Candidate
from . import metrics as M
from .sim_eval import simulate_candidates

EVENT_NAMES = list(S.EVENTS)


def _train_reference(data_dir: Path):
    """Train-split reference sets per event: unit vectors and occupied coverage cells."""
    tr = load_split(data_dir, "train")
    from ..scenarios.codec import unpack_to_spec
    refs: Dict[int, dict] = {}
    for e in sorted(set(int(x) for x in tr.event)):
        idx = np.where(tr.event == e)[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, min(len(idx), 4000), replace=False)
        specs = [unpack_to_spec(tr.phys[i].astype(np.float64), tr.slot_mask[i], tr.asset_ids[i], int(tr.task[i]), e,
                                int(tr.robot[i]), {v.name: float(tr.env[i][j]) for j, v in enumerate(S.ENV_VARS)}, "t", "t", 0)
                 for i in idx]
        feats = np.stack([M.key_features(s) for s in specs])
        cells = M.cell_ids(feats)
        counts = np.bincount(cells, minlength=4 ** 4)
        refs[e] = {"u": M.unit_vectors(specs), "counts": counts,
                   "rare_cells": set(np.where(counts <= 2)[0].tolist())}  # cells that are sparse in training data
    return refs


def run_evaluation(ckpt: str | Path, data_dir: str | Path, out_dir: str | Path, eval_cfg: Dict, device: str = "cpu",
                   workers: int = 1, seed: int = 4242) -> Dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, dcfg, norm, prior, support, meta = load_generator(ckpt, device)
    refs = _train_reference(Path(data_dir))
    n = int(eval_cfg["n_per_condition"])
    rows: List[Dict] = []
    simulate = bool(eval_cfg.get("simulate", True))
    for ev_name in eval_cfg["events"]:
        e = S.EVENTS.index(ev_name)
        if e not in refs:
            continue
        for sev in eval_cfg["severities"]:
            warns = support_warnings(support, ev_name, sev)
            learned = sample_candidates(model, dcfg, norm, prior, e, float(sev), n, seed + e * 100 + int(sev * 100), device,
                                        eval_cfg.get("ddim_steps"), eval_cfg.get("cfg_scale"), prefix=f"ev-{ev_name}-{int(sev * 100)}")
            proc = [Candidate(s, 0.0, float(sev)) for s in generate_conditioned(e, float(sev), n, base_seed=seed)]
            row = {"event": ev_name, "severity": sev, "ood_warnings": warns}
            for name, cands in (("learned", learned), ("procedural", proc)):
                items = simulate_candidates(cands, workers=workers, rollout=simulate, dedupe=True)
                req = np.array([c.requested_severity for c in cands])
                row[name] = M.set_metrics(items, req, None, refs[e]["u"], refs[e]["rare_cells"])
            rows.append(row)
    # held-out reference novelty: val-set samples vs train (natural level of NN distance for real draws)
    report = {"checkpoint": str(ckpt), "run_id": meta.get("run_id"), "train_step": meta.get("step"),
              "n_per_condition": n, "rows": rows, "eval_seconds": time.time() - t0,
              "note": "Compares the learned generator with the procedural generator at matched (event, severity). "
                      "Validity/diversity/novelty do not by themselves imply usefulness for robot learning; "
                      "see docs/EXPERIMENT_PROTOCOL.md for the downstream protocol."}
    report["summary"] = summarize(rows)
    write_json(out_dir / "evaluation_report.json", report)
    (out_dir / "evaluation_report.md").write_text(render_markdown(report))
    return report


def summarize(rows: List[Dict]) -> Dict:
    def avg(which, key):
        v = [r[which][key] for r in rows if key in r[which]]
        return float(np.mean(v)) if v else None
    keys = ["valid_ratio", "repair_mean", "condition_event_accuracy", "condition_severity_mae", "diversity_mean_l1",
            "coverage_cells_fraction", "rare_cell_coverage", "novelty_nn_median", "memorization_rate",
            "sim_instability_rate", "baseline_failure_rate"]
    return {w: {k: avg(w, k) for k in keys} for w in ("learned", "procedural")}


def render_markdown(rep: Dict) -> str:
    L = [f"# RareSimDiff evaluation\n\nrun: `{rep.get('run_id')}`  step: {rep.get('train_step')}  n/condition: {rep['n_per_condition']}\n",
         "## Summary (mean over conditions)\n", "| metric | learned | procedural |", "|---|---|---|"]
    s = rep["summary"]
    for k in s["learned"]:
        a, b = s["learned"][k], s["procedural"][k]
        f = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
        L.append(f"| {k} | {f(a)} | {f(b)} |")
    L += ["\n## Per condition\n", "| event | sev | gen | valid | ev.acc | sev MAE | diversity | rare-cell cov | novelty | baseline fail |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        for w in ("learned", "procedural"):
            m = r[w]
            g = lambda k: f"{m[k]:.3f}" if k in m else "n/a"  # noqa: E731
            L.append(f"| {r['event']} | {r['severity']} | {w} | {g('valid_ratio')} | {g('condition_event_accuracy')} | "
                     f"{g('condition_severity_mae')} | {g('diversity_mean_l1')} | {g('rare_cell_coverage')} | "
                     f"{g('novelty_nn_median')} | {g('baseline_failure_rate')} |")
        if r["ood_warnings"]:
            L.append(f"|  |  | warnings | {'; '.join(r['ood_warnings'])} |||||||")
    L.append("\n" + rep["note"])
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    from ..utils.config import load_config
    ap = argparse.ArgumentParser(description="Evaluate a frozen generator vs the procedural generator (no training).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-per-condition", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config_dir)
    ev = dict(cfg.evaluation)
    if a.n_per_condition:
        ev["n_per_condition"] = a.n_per_condition
    rep = run_evaluation(a.checkpoint, a.data_dir, a.out, ev, a.device, a.workers)
    print(json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
