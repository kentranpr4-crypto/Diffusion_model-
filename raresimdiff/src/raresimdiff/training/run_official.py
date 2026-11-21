"""Orchestrates THE official training run: verify -> create immutable run dir -> train -> evaluate -> export -> report.

    python -m raresimdiff.training.run_official --data-dir data/raresimdiff_v1 [--preflight-only] [--resume]

Never starts automatically; a completed run is never overwritten; resume continues the SAME run.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch

from ..data.integrity import verify_dataset
from ..utils.config import load_config
from ..utils.hashing import stable_json_hash
from ..utils.io import read_json, write_json
from .gpu_preflight import run_preflight
from .run_dir import (RunExistsError, code_revision, create_run_dir, dependency_versions, gpu_info, official_run_id,
                      open_for_resume, set_state)


def build_identity(cfg, data_dir: Path) -> dict:
    ver = verify_dataset(data_dir)
    if not ver["ok"]:
        raise SystemExit("dataset integrity FAILED: " + "; ".join(ver["failures"]))
    th = cfg.training_hash()
    fp = ver["fingerprint"]
    rid = official_run_id(cfg.run.name, th, fp)
    ident = {"run_id": rid, "training_hash": th, "dataset_fingerprint": fp}
    ident["identity_hash"] = stable_json_hash(ident)
    return ident


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="continue the interrupted SAME run (never a new experiment)")
    ap.add_argument("--preflight-only", action="store_true", help="run all checks, take NO optimizer steps")
    ap.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only (changes the run identity)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--tensorboard", action="store_true", help="optional local TensorBoard logging (no online service)")
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=None, help="smoke tests only: override evaluation.n_per_condition")
    a = ap.parse_args(argv)

    cfg_dir, data_dir = Path(a.config_dir), Path(a.data_dir)
    cfg = load_config(cfg_dir)
    if a.max_steps is not None:
        cfg.train.max_steps = a.max_steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(1, a.max_steps // 10))
    if a.eval_n is not None:
        cfg.evaluation = dict(cfg.evaluation, n_per_condition=a.eval_n)
    pre = run_preflight(str(cfg_dir), str(data_dir), allow_cpu=a.allow_cpu, probe=True)
    for k, c in pre["checks"].items():
        print(f"[{'PASS' if c['ok'] else ('WARN' if c['warning_only'] else 'FAIL')}] {k}: {c['detail']}", flush=True)
    if not pre["ok"]:
        print("PREFLIGHT FAILED: no optimizer steps were taken.", file=sys.stderr)
        sys.exit(2)
    ident = build_identity(cfg, data_dir)
    print(f"run_id={ident['run_id']} training_hash={ident['training_hash'][:12]} dataset={ident['dataset_fingerprint'][:12]}")
    if a.preflight_only:
        print("PREFLIGHT OK (--preflight-only): exiting without optimization.")
        return
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.allow_cpu:
        raise SystemExit("refusing to run the official job on CPU (use --allow-cpu for smoke tests)")
    root = Path(a.runs_root)
    manifest = {**ident, "config": cfg.to_dict(), "gpu": gpu_info(), "deps": dependency_versions(),
                "code_revision": code_revision(cfg_dir.resolve().parent), "dataset_manifest": read_json(data_dir / "dataset_manifest.json"),
                "preflight": pre, "created_at": time.time(), "device": device}
    try:
        run_dir = open_for_resume(root, ident["run_id"], ident["identity_hash"]) if a.resume else create_run_dir(root, ident["run_id"], manifest)
    except RunExistsError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
    from .trainer import NonFiniteTrainingError, Trainer
    from ..utils.seeding import seed_everything
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    tr = Trainer(cfg, data_dir, run_dir, device, ident, tensorboard=a.tensorboard)
    if a.resume:
        last = run_dir / "checkpoints" / "last.pt"
        if not last.exists():
            raise SystemExit("no checkpoints/last.pt to resume from")
        tr.load_checkpoint(last)
        print(f"resumed at step {tr.step}")
    set_state(run_dir, "running", step=tr.step)
    try:
        res = tr.train()
    except NonFiniteTrainingError as e:
        print(f"TRAINING STOPPED (non-finite): {e}. Resumable from checkpoints/last.pt; NOT silently retrained.", file=sys.stderr)
        sys.exit(4)
    if res["status"] == "interrupted":
        print("INTERRUPTED: progress saved. Continue the SAME run with the same command plus --resume.")
        sys.exit(130)
    report = {"run_id": ident["run_id"], "training": res, "identity": ident}
    best = run_dir / "best.safetensors"
    if not a.no_eval:
        from ..evaluation.evaluate import run_evaluation
        try:
            ev = run_evaluation(best, data_dir, run_dir / "evaluation", cfg.evaluation,
                                device="cuda" if device == "cuda" else "cpu", workers=a.eval_workers)
            report["evaluation_summary"] = ev["summary"]
        except Exception as e:  # evaluation failure must not erase a finished training run
            report["evaluation_error"] = repr(e)
    import shutil
    for f in ("best.safetensors", "last.safetensors"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, run_dir / "export" / f)
    write_json(run_dir / "REPORT.json", report)
    write_json(run_dir / "COMPLETED", {"completed_at": time.time(), "report": "REPORT.json"})
    set_state(run_dir, "completed", step=res["step"], best_val=res["best_val"], best_step=res["best_step"])
    print(json.dumps(report["training"], indent=2))
    print("DONE: run complete. Export:", run_dir / "export" / "best.safetensors")


if __name__ == "__main__":
    main()
