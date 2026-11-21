from __future__ import annotations
import math
import torch


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021) cosine schedule; returns alpha_bar[0..T-1] (float64)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, dim=0)


def linear_alphas_cumprod(T: int, b0: float = 1e-4, b1: float = 0.02) -> torch.Tensor:
    return torch.cumprod(1 - torch.linspace(b0, b1, T, dtype=torch.float64), dim=0)


def make_alphas_cumprod(kind: str, T: int) -> torch.Tensor:
    if kind == "cosine":
        return cosine_alphas_cumprod(T)
    if kind == "linear":
        return linear_alphas_cumprod(T)
    raise ValueError(kind)
