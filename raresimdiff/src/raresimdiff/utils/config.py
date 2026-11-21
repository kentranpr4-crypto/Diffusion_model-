"""Typed configuration loading + hashing."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from .hashing import stable_json_hash


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    schedule: str = "cosine"
    prediction: str = "epsilon"
    sampler: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    cfg_scale: float = 1.5
    p_uncond: float = 0.1
    clip_x0: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 512
    grad_accum_steps: int = 2
    max_steps: int = 120000
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    adam_betas: List[float] = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    amp_dtype: str = "bfloat16"
    log_every: int = 100
    eval_every: int = 2000
    ckpt_every: int = 2000
    keep_last_checkpoints: int = 3
    max_consecutive_nonfinite: int = 20
    val_max_samples: int = 8192

    def __post_init__(self):
        if self.adam_betas is None:
            self.adam_betas = [0.9, 0.99]


@dataclass
class RunConfig:
    name: str = "official"
    seed: int = 1234
    deterministic: bool = True


def _build(cls, d: Optional[dict]):
    d = d or {}
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


@dataclass
class AllConfig:
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    run: RunConfig
    gpu: Dict[str, Any]
    evaluation: Dict[str, Any]
    dataset: Dict[str, Any]
    simulation: Dict[str, Any]
    validation: Dict[str, Any]

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "diffusion": asdict(self.diffusion), "train": asdict(self.train),
                "run": asdict(self.run), "gpu": self.gpu, "evaluation": self.evaluation, "dataset": self.dataset,
                "simulation": self.simulation, "validation": self.validation}

    def training_hash(self) -> str:
        """Hash of everything that defines the official training run (not dataset-build options)."""
        d = self.to_dict()
        return stable_json_hash({k: d[k] for k in ("model", "diffusion", "train", "run", "gpu", "evaluation")})


def _load_yaml(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path, gpu_file: str = "gpu_4090.yaml") -> AllConfig:
    cd = Path(config_dir)
    m = _load_yaml(cd / "model.yaml")
    ds = _load_yaml(cd / "dataset.yaml")
    g = _load_yaml(cd / gpu_file)
    return AllConfig(
        model=_build(ModelConfig, m.get("model")), diffusion=_build(DiffusionConfig, m.get("diffusion")),
        train=_build(TrainConfig, g.get("training")), run=_build(RunConfig, g.get("run")),
        gpu=g.get("gpu", {}), evaluation=g.get("evaluation", {}), dataset=ds.get("dataset", {}),
        simulation=ds.get("simulation", {}), validation=ds.get("validation", {}))
