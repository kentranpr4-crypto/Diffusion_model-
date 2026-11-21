from __future__ import annotations
import math
import torch
import torch.nn as nn

from ..scenarios import schema as S

ENV_LO = torch.tensor([v.lo for v in S.ENV_VARS], dtype=torch.float32)
ENV_HI = torch.tensor([v.hi for v in S.ENV_VARS], dtype=torch.float32)


class ConditionEncoder(nn.Module):
    """Embeds categorical + continuous conditions. `drop` replaces task/event/robot/severity/env by learned null
    embeddings (classifier-free guidance); structural inputs (slot mask, asset ids) are never dropped."""

    def __init__(self, hidden: int, n_freq: int = 8):
        super().__init__()
        self.task = nn.Embedding(len(S.TASKS) + 1, hidden)
        self.event = nn.Embedding(len(S.EVENTS) + 1, hidden)
        self.robot = nn.Embedding(len(S.ROBOT_PROFILES) + 1, hidden)
        self.asset = nn.Embedding(len(S.ASSETS), hidden)
        self.n_freq = n_freq
        self.sev = nn.Sequential(nn.Linear(2 * n_freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.env = nn.Sequential(nn.Linear(S.E, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.null_sev = nn.Parameter(torch.zeros(hidden))
        self.null_env = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("env_lo", ENV_LO.clone(), persistent=False)
        self.register_buffer("env_hi", ENV_HI.clone(), persistent=False)

    def forward(self, cond, drop: torch.Tensor):
        task = torch.where(drop, torch.full_like(cond["task"], len(S.TASKS)), cond["task"]).long()
        event = torch.where(drop, torch.full_like(cond["event"], len(S.EVENTS)), cond["event"]).long()
        robot = torch.where(drop, torch.full_like(cond["robot"], len(S.ROBOT_PROFILES)), cond["robot"]).long()
        s = cond["severity"].float()
        freqs = (2.0 ** torch.arange(self.n_freq, device=s.device)) * math.pi
        ang = s[:, None] * freqs[None]
        sev = self.sev(torch.cat([ang.sin(), ang.cos()], -1))
        env_u = 2 * (cond["env"].float() - self.env_lo) / (self.env_hi - self.env_lo) - 1
        env = self.env(env_u)
        d = drop[:, None].float()
        sev = sev * (1 - d) + self.null_sev[None] * d
        env = env * (1 - d) + self.null_env[None] * d
        c = self.task(task) + self.event(event) + self.robot(robot) + sev + env
        asset_tok = self.asset(cond["asset_ids"].long())  # (B,K,H)
        return c, asset_tok
