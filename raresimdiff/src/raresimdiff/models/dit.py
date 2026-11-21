"""RareScenarioDiT: compact conditional structured 1D Diffusion Transformer over scenario tokens."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scenarios import schema as S
from ..utils.config import ModelConfig
from .condition import ConditionEncoder


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_mask):
        B, L, C = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        return self.proj(o.transpose(1, 2).reshape(B, L, C))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, key_mask):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c)[:, None].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + sc1) + sh1, key_mask)
        x = x + g2 * self.mlp(self.n2(x) * (1 + sc2) + sh2)
        return x


class RareScenarioDiT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.hidden_dim
        self.cfg = cfg
        self.in_global = nn.Linear(S.D, H)
        self.in_obj = nn.Linear(S.D, H)
        self.in_obs = nn.Linear(len(S.OBS_VARS), H)
        self.slot_emb = nn.Parameter(torch.randn(S.L, H) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.cond = ConditionEncoder(H)
        self.blocks = nn.ModuleList([DiTBlock(H, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.num_layers)])
        self.norm_f = nn.LayerNorm(H, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(H, 2 * H))
        nn.init.zeros_(self.ada_f[1].weight)
        nn.init.zeros_(self.ada_f[1].bias)
        self.out_global = nn.Linear(H, S.D)
        self.out_obj = nn.Linear(H, S.D)
        self.out_obs = nn.Linear(H, len(S.OBS_VARS))
        for m in (self.out_global, self.out_obj, self.out_obs):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        self.register_buffer("ch_mask", torch.tensor(S.CH_MASK, dtype=torch.float32), persistent=False)

    def forward(self, x, t, cond, drop=None):
        """x: (B,L,D) noisy scenario tokens; t: (B,) int timesteps; cond: dict; drop: (B,) bool."""
        B = x.shape[0]
        if drop is None:
            drop = torch.zeros(B, dtype=torch.bool, device=x.device)
        K = S.K_OBJ
        c_cond, asset_tok = self.cond(cond, drop)
        c = c_cond + self.time_mlp(timestep_embedding(t, self.cfg.hidden_dim))
        h = torch.cat([self.in_global(x[:, :1]), self.in_obj(x[:, 1:1 + K]),
                       self.in_obs(x[:, 1 + K:, :len(S.OBS_VARS)])], dim=1)
        h = h + self.slot_emb[None]
        h = h + F.pad(asset_tok, (0, 0, 1, S.M_OBS))  # asset embedding on object slots only
        key_mask = cond["slot_mask"] > 0.5
        for blk in self.blocks:
            h = blk(h, c, key_mask)
        sh, sc = self.ada_f(c)[:, None].chunk(2, dim=-1)
        h = self.norm_f(h) * (1 + sc) + sh
        og = self.out_global(h[:, :1])
        oo = self.out_obj(h[:, 1:1 + K])
        ob = F.pad(self.out_obs(h[:, 1 + K:]), (0, S.D - len(S.OBS_VARS)))
        return torch.cat([og, oo, ob], dim=1) * self.ch_mask[None]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
