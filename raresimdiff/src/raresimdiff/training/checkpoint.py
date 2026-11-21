from __future__ import annotations
import os, random, tempfile
from pathlib import Path
from typing import Dict
import numpy as np
import torch


def capture_rng(gen: torch.Generator) -> Dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "diffusion_generator": gen.get_state()}


def restore_rng(state: Dict, gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    gen.set_state(state["diffusion_generator"].cpu())


def atomic_torch_save(obj, path: str | Path) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
