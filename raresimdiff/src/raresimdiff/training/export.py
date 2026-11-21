"""Safetensors export / load of an inference-ready, self-contained generator (no optimizer state)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from ..scenarios import schema as S
from ..scenarios.codec import Normalizer
from ..scenarios.prior import CategoricalPrior
from ..utils.config import DiffusionConfig, ModelConfig


def train_support(train_sd) -> Dict:
    """Per-event severity range seen in training (used to warn about out-of-distribution conditions)."""
    out = {}
    for e in sorted(set(int(x) for x in train_sd.event)):
        s = train_sd.severity[train_sd.event == e]
        out[S.EVENTS[e]] = {"n": int(len(s)), "severity_min": float(s.min()), "severity_max": float(s.max()),
                            "tasks": sorted(set(int(t) for t in train_sd.task[train_sd.event == e]))}
    return out


def export_generator(path: str | Path, ema_state: Dict[str, torch.Tensor], model_cfg: ModelConfig,
                     diff_cfg: DiffusionConfig, normalizer: Normalizer, prior: CategoricalPrior, support: Dict,
                     extra_meta: Dict) -> None:
    from dataclasses import asdict
    tensors = {f"model.{k}": v.detach().cpu().contiguous().float() for k, v in ema_state.items()}
    tensors.update({k: v.contiguous() for k, v in prior.to_tensors().items()})
    meta = {"format": "raresimdiff-generator-v1", "model_config": json.dumps(asdict(model_cfg)),
            "diffusion_config": json.dumps(asdict(diff_cfg)), "normalizer": json.dumps(normalizer.to_json()),
            "train_support": json.dumps(support), "schema": json.dumps({"L": S.L, "D": S.D, "K": S.K_OBJ, "M": S.M_OBS}),
            **{k: str(v) for k, v in extra_meta.items()}}
    save_file(tensors, str(path), metadata=meta)


def load_generator(path: str | Path, device: str = "cpu"):
    """Returns (model, diff_cfg, normalizer, prior, support, meta). Loads weights only (no optimizer state)."""
    from ..models.dit import RareScenarioDiT
    path = str(path)
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    if meta.get("format") != "raresimdiff-generator-v1":
        raise ValueError(f"{path} is not a raresimdiff generator export")
    sch = json.loads(meta["schema"])
    if (sch["L"], sch["D"], sch["K"], sch["M"]) != (S.L, S.D, S.K_OBJ, S.M_OBS):
        raise ValueError("schema mismatch between checkpoint and installed raresimdiff")
    t = load_file(path)
    model = RareScenarioDiT(ModelConfig(**json.loads(meta["model_config"])))
    model.load_state_dict({k[len("model."):]: v for k, v in t.items() if k.startswith("model.")})
    model.to(device).eval()
    prior = CategoricalPrior.from_tensors({k: v for k, v in t.items() if k.startswith("prior.")})
    return (model, DiffusionConfig(**json.loads(meta["diffusion_config"])), Normalizer.from_json(json.loads(meta["normalizer"])),
            prior, json.loads(meta["train_support"]), meta)
