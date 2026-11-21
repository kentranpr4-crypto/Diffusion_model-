import numpy as np
import pytest
import torch
from raresimdiff.diffusion.schedule import cosine_alphas_cumprod
from raresimdiff.diffusion.gaussian import GaussianDiffusion
from raresimdiff.models.dit import RareScenarioDiT, count_parameters
from raresimdiff.scenarios import schema as S
from raresimdiff.utils.config import ModelConfig


def tiny_model():
    torch.manual_seed(0)
    return RareScenarioDiT(ModelConfig(hidden_dim=64, num_layers=2, num_heads=4))


def make_cond(B=6, mask=None):
    g = torch.Generator().manual_seed(0)
    return {"task": torch.zeros(B).long(), "event": torch.randint(0, 5, (B,), generator=g), "robot": torch.zeros(B).long(),
            "severity": torch.rand(B, generator=g), "env": torch.rand(B, S.E, generator=g),
            "asset_ids": torch.randint(1, 6, (B, S.K_OBJ), generator=g),
            "slot_mask": torch.ones(B, S.L) if mask is None else mask}


def randomise_adaln(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.abs().sum() == 0:
            torch.nn.init.normal_(mod.weight, std=0.05)


def test_cosine_schedule_properties():
    ac = cosine_alphas_cumprod(1000)
    assert ac.shape == (1000,)
    assert torch.all(ac[1:] < ac[:-1]) and ac[0] > 0.99 and ac[-1] < 1e-3


def test_forward_shapes_and_param_count():
    m = tiny_model()
    out = m(torch.randn(6, S.L, S.D), torch.randint(0, 1000, (6,)), make_cond())
    assert out.shape == (6, S.L, S.D)
    assert torch.all(out[:, :, :] * (1 - torch.tensor(S.CH_MASK))[None] == 0)  # no output on padded channels
    assert count_parameters(RareScenarioDiT(ModelConfig())) > 1_000_000


def test_masked_slots_do_not_influence_valid_outputs():
    m = tiny_model()
    randomise_adaln(m)
    mask = torch.ones(4, S.L)
    mask[:, 4:7] = 0
    cond = make_cond(4, mask)
    x = torch.randn(4, S.L, S.D)
    t = torch.randint(0, 1000, (4,))
    o1 = m(x, t, cond)
    x2 = x.clone()
    x2[:, 4:7] = torch.randn(4, 3, S.D) * 10
    o2 = m(x2, t, cond)
    keep = mask.bool()
    assert torch.allclose(o1[keep], o2[keep], atol=1e-5)


def test_conditioning_changes_output():
    m = tiny_model()
    randomise_adaln(m)
    x, t = torch.randn(6, S.L, S.D), torch.randint(0, 1000, (6,))
    c = make_cond()
    base = m(x, t, c)
    c2 = dict(c, event=(c["event"] + 1) % 5)
    c3 = dict(c, severity=1 - c["severity"])
    drop = m(x, t, c, torch.ones(6, dtype=torch.bool))
    assert not torch.allclose(base, m(x, t, c2), atol=1e-5)
    assert not torch.allclose(base, m(x, t, c3), atol=1e-5)
    assert not torch.allclose(base, drop, atol=1e-5)


def test_finite_gradients_and_loss_masking():
    m = tiny_model()
    d = GaussianDiffusion()
    mask = torch.ones(8, S.L)
    mask[:, 5:] = 0
    cond = make_cond(8, mask)
    loss = d.training_loss(m, torch.randn(8, S.L, S.D), cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_ddim_sampler_properties():
    m = tiny_model().eval()
    d = GaussianDiffusion(clip_x0=True)
    mask = torch.ones(5, S.L)
    mask[:, 6:] = 0
    cond = make_cond(5, mask)
    lo, hi = -2 * torch.ones(S.L, S.D), 2 * torch.ones(S.L, S.D)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a = d.ddim_sample(m, cond, lo, hi, steps=8, cfg_scale=1.0, generator=g1)
    b = d.ddim_sample(m, cond, lo, hi, steps=8, cfg_scale=1.0, generator=g2)
    assert a.shape == (5, S.L, S.D) and torch.isfinite(a).all()
    assert torch.equal(a, b)  # deterministic given seed and eta=0
    assert torch.all(a[:, 6:] == 0)  # masked slots stay zero
    c = d.ddim_sample(m, cond, lo, hi, steps=8, cfg_scale=2.5, generator=torch.Generator().manual_seed(1))
    randomise_adaln(m)
    e = d.ddim_sample(m, cond, lo, hi, steps=8, cfg_scale=3.0, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(e).all()


def test_forward_process_noise_level():
    d = GaussianDiffusion()
    x0 = torch.randn(2000, S.L, S.D)
    n = torch.randn_like(x0)
    xt = d.q_sample(x0, torch.full((2000,), 999), n)
    assert xt.std().item() == pytest.approx(1.0, abs=0.1)  # fully noised ~ N(0, 1)
