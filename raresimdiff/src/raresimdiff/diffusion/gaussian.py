"""Gaussian diffusion: forward process, masked epsilon-prediction loss, DDIM sampler with classifier-free guidance."""
from __future__ import annotations
from typing import Dict, Optional
import torch

from ..scenarios import schema as S
from .schedule import make_alphas_cumprod


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine", prediction: str = "epsilon",
                 p_uncond: float = 0.1, clip_x0: bool = True):
        if prediction != "epsilon":
            raise NotImplementedError("only epsilon prediction is implemented")
        self.T = num_timesteps
        self.p_uncond = p_uncond
        self.clip_x0 = clip_x0
        ac = make_alphas_cumprod(schedule, num_timesteps)
        self.alphas_cumprod = ac.float()
        self.sqrt_ac = ac.sqrt().float()
        self.sqrt_1m_ac = (1 - ac).sqrt().float()
        self.ch_mask = torch.tensor(S.CH_MASK, dtype=torch.float32)

    def to(self, device):
        for n in ("alphas_cumprod", "sqrt_ac", "sqrt_1m_ac", "ch_mask"):
            setattr(self, n, getattr(self, n).to(device))
        return self

    def element_weights(self, slot_mask: torch.Tensor) -> torch.Tensor:
        return slot_mask[:, :, None].float() * self.ch_mask[None]

    def q_sample(self, x0, t, noise):
        a = self.sqrt_ac[t][:, None, None]
        b = self.sqrt_1m_ac[t][:, None, None]
        return a * x0 + b * noise

    def training_loss(self, model, x0: torch.Tensor, cond: Dict[str, torch.Tensor], generator: Optional[torch.Generator] = None,
                      t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None,
                      drop: Optional[torch.Tensor] = None):
        B = x0.shape[0]
        dev = x0.device
        w = self.element_weights(cond["slot_mask"])
        if t is None:
            t = torch.randint(0, self.T, (B,), device=dev, generator=generator)
        if noise is None:
            noise = torch.randn(x0.shape, device=dev, generator=generator)
        if drop is None:
            drop = torch.rand(B, device=dev, generator=generator) < self.p_uncond
        xt = self.q_sample(x0, t, noise) * w
        eps = model(xt, t, cond, drop)
        se = ((eps.float() - noise) ** 2) * w
        loss = se.sum() / w.sum().clamp(min=1.0)
        return loss

    @torch.no_grad()
    def ddim_sample(self, model, cond: Dict[str, torch.Tensor], x0_lo: torch.Tensor, x0_hi: torch.Tensor, steps: int = 50,
                    eta: float = 0.0, cfg_scale: float = 1.0, generator: Optional[torch.Generator] = None,
                    amp_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        B = cond["slot_mask"].shape[0]
        dev = cond["slot_mask"].device
        w = self.element_weights(cond["slot_mask"])
        x = torch.randn((B, S.L, S.D), device=dev, generator=generator) * w
        ts = torch.linspace(self.T - 1, 0, steps, device=dev).round().long()
        ts = torch.unique_consecutive(ts)
        use_cfg = cfg_scale != 1.0
        no_drop = torch.zeros(B, dtype=torch.bool, device=dev)
        all_drop = torch.ones(B, dtype=torch.bool, device=dev)
        for i, t in enumerate(ts):
            tb = t.expand(B)
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                eps_c = model(x, tb, cond, no_drop).float()
                if use_cfg:
                    eps_u = model(x, tb, cond, all_drop).float()
                    eps = eps_u + cfg_scale * (eps_c - eps_u)
                else:
                    eps = eps_c
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            if self.clip_x0:
                x0 = torch.maximum(torch.minimum(x0, x0_hi[None]), x0_lo[None])
                eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt() if eta > 0 else torch.tensor(0.0, device=dev)
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            noise = torch.randn(x.shape, device=dev, generator=generator) if eta > 0 else 0.0
            x = (a_prev.sqrt() * x0 + dir_xt + sigma * noise) * w
        return x
