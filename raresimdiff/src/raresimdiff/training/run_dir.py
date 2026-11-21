"""Immutable official run directory management. A completed run can never be overwritten automatically."""
from __future__ import annotations
import os, platform, sys, time
from pathlib import Path
from typing import Dict

from ..utils.hashing import sha256_bytes, stable_json_hash
from ..utils.io import read_json, write_json


class RunExistsError(RuntimeError):
    pass


def official_run_id(name: str, training_hash: str, dataset_fingerprint: str) -> str:
    return f"{name}_{sha256_bytes((training_hash + dataset_fingerprint).encode())[:10]}"


def gpu_info() -> Dict:
    info = {"cuda_available": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({"name": p.name, "total_memory_gb": round(p.total_memory / 2**30, 2),
                         "compute_capability": f"{p.major}.{p.minor}", "cuda_runtime": torch.version.cuda,
                         "bf16_supported": bool(torch.cuda.is_bf16_supported())})
    except Exception as e:  # pragma: no cover
        info["error"] = str(e)
    return info


def dependency_versions() -> Dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "numpy", "mujoco", "safetensors", "yaml"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:
            out[m] = None
    return out


def code_revision(root: Path) -> str:
    """Hash of all source files (works without git inside the transferred bundle)."""
    h = []
    for p in sorted((root / "src").rglob("*.py")):
        h.append((str(p.relative_to(root)), sha256_bytes(p.read_bytes())))
    for p in sorted((root / "configs").glob("*.yaml")):
        h.append((str(p.relative_to(root)), sha256_bytes(p.read_bytes())))
    return stable_json_hash(h)


def create_run_dir(root: str | Path, run_id: str, manifest: Dict) -> Path:
    run = Path(root) / run_id
    try:
        os.makedirs(run.parent, exist_ok=True)
        os.mkdir(run)  # exclusive: fails if it already exists
    except FileExistsError:
        state = read_state(run)
        raise RunExistsError(f"run directory {run} already exists (status={state.get('status')}). "
                             f"Completed runs are never overwritten. To continue an INTERRUPTED run use --resume.")
    for sub in ("checkpoints", "logs", "evaluation", "export"):
        (run / sub).mkdir()
    write_json(run / "RUN_MANIFEST.json", manifest)
    write_json(run / "RUN_STATE.json", {"status": "created", "created_at": time.time()})
    return run


def read_state(run: Path) -> Dict:
    p = Path(run) / "RUN_STATE.json"
    return read_json(p) if p.exists() else {}


def set_state(run: Path, status: str, **kw) -> None:
    st = read_state(run)
    st.update({"status": status, "updated_at": time.time(), **kw})
    write_json(Path(run) / "RUN_STATE.json", st)


def is_completed(run: Path) -> bool:
    return (Path(run) / "COMPLETED").exists() or read_state(run).get("status") == "completed"


def open_for_resume(root: str | Path, run_id: str, expected_manifest_hash: str) -> Path:
    run = Path(root) / run_id
    if not run.exists():
        raise FileNotFoundError(f"nothing to resume: {run} does not exist")
    if is_completed(run):
        raise RunExistsError(f"{run} is COMPLETED; refusing to resume/overwrite. Start requires explicit new approval.")
    m = read_json(run / "RUN_MANIFEST.json")
    if m.get("identity_hash") != expected_manifest_hash:
        raise RunExistsError("config/dataset identity differs from the run being resumed; this would be a NEW experiment")
    return run
