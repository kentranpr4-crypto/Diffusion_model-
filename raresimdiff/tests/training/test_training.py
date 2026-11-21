import copy, json
from pathlib import Path
import numpy as np
import pytest
import torch
from raresimdiff.data.dataset import InMemoryDataset, load_normalizer, load_split
from raresimdiff.diffusion.gaussian import GaussianDiffusion
from raresimdiff.models.dit import RareScenarioDiT
from raresimdiff.training import run_dir as RD
from raresimdiff.training.export import load_generator
from raresimdiff.training.trainer import NonFiniteTrainingError, Trainer

IDENT = {"identity_hash": "t", "run_id": "t", "dataset_fingerprint": "x", "training_hash": "y"}


def make_trainer(cfg, ds, rd):
    rd.mkdir(parents=True, exist_ok=True)
    return Trainer(copy.deepcopy(cfg), ds, rd, "cpu", IDENT)


def test_small_batch_overfit_on_cpu(tiny_dataset, small_cfg):
    torch.manual_seed(0)
    ds = InMemoryDataset(load_split(tiny_dataset, "train"), load_normalizer(tiny_dataset), torch.device("cpu"), max_samples=16)
    cfg = small_cfg
    model = RareScenarioDiT(cfg.model)
    diff = GaussianDiffusion(p_uncond=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    x0, cond = ds.batch(torch.arange(16))
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, 1000, (16,), generator=g)
    noise = torch.randn(x0.shape, generator=g)
    drop = torch.zeros(16, dtype=torch.bool)
    first = None
    for i in range(400):
        loss = diff.training_loss(model, x0, cond, t=t, noise=noise, drop=drop)
        first = loss.item() if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.35 * first, (first, loss.item())


def test_loss_decreases_with_random_timesteps(tiny_dataset, small_cfg, tmp_path):
    cfg = copy.deepcopy(small_cfg)
    cfg.train.max_steps, cfg.train.lr, cfg.train.eval_every = 60, 1e-3, 60
    tr = make_trainer(cfg, tiny_dataset, tmp_path / "r")
    v0 = tr.evaluate_val()
    tr.train()
    assert tr.best_val < v0


def test_resume_is_exact_continuation(tiny_dataset, small_cfg, tmp_path):
    cfg = copy.deepcopy(small_cfg)
    cfg.train.max_steps = 24
    a = make_trainer(cfg, tiny_dataset, tmp_path / "a")
    a.train()
    b1 = make_trainer(cfg, tiny_dataset, tmp_path / "b")
    b1.train(max_steps=10)
    b2 = make_trainer(cfg, tiny_dataset, tmp_path / "b")
    b2.load_checkpoint(tmp_path / "b" / "checkpoints" / "last.pt")
    assert b2.step == 10
    b2.train()
    pa = torch.cat([p.flatten() for p in a.model.parameters()])
    pb = torch.cat([p.flatten() for p in b2.model.parameters()])
    ea = torch.cat([p.flatten() for p in a.ema.model.parameters()])
    eb = torch.cat([p.flatten() for p in b2.ema.model.parameters()])
    assert torch.equal(pa, pb) and torch.equal(ea, eb)
    assert a.step == b2.step == 24


def test_checkpoint_contents_and_best_selection(tiny_dataset, small_cfg, tmp_path):
    cfg = copy.deepcopy(small_cfg)
    cfg.train.max_steps = 20
    tr = make_trainer(cfg, tiny_dataset, tmp_path / "r")
    res = tr.train()
    ck = torch.load(tmp_path / "r" / "checkpoints" / "last.pt", weights_only=False)
    for k in ("model", "ema", "opt", "sched", "step", "rng", "identity", "best_val"):
        assert k in ck
    assert ck["rng"]["diffusion_generator"] is not None and "python" in ck["rng"] and "numpy" in ck["rng"]
    assert (tmp_path / "r" / "best.safetensors").exists() and (tmp_path / "r" / "checkpoints" / "best.pt").exists()
    assert res["best_step"] in (10, 20)
    assert min(h["val_loss_ema"] for h in tr.val_history) == pytest.approx(tr.best_val)
    log = [json.loads(l) for l in open(tmp_path / "r" / "logs" / "train.jsonl")]
    assert any("loss" in r for r in log) and any("val_loss_ema" in r for r in log)
    assert (tmp_path / "r" / "logs" / "train.csv").exists()


def test_nan_guard_skips_then_aborts(tiny_dataset, small_cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(small_cfg)
    cfg.train.max_steps, cfg.train.max_consecutive_nonfinite = 50, 3
    tr = make_trainer(cfg, tiny_dataset, tmp_path / "r")
    orig = tr.diffusion.training_loss
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        l = orig(*a, **k)
        return l * float("nan") if calls["n"] > 4 else l  # first 2 optimizer steps fine, then NaN forever
    monkeypatch.setattr(tr.diffusion, "training_loss", bad)
    before = [p.detach().clone() for p in tr.model.parameters()]
    with pytest.raises(NonFiniteTrainingError):
        tr.train()
    assert tr.skipped >= 3
    assert all(torch.isfinite(p).all() for p in tr.model.parameters())  # parameters never corrupted
    assert (tmp_path / "r" / "checkpoints" / "last.pt").exists()  # progress preserved


def test_run_dir_immutability_and_resume_rules(tmp_path):
    man = {"identity_hash": "H", "x": 1}
    run = RD.create_run_dir(tmp_path, "official_abc", man)
    with pytest.raises(RD.RunExistsError):
        RD.create_run_dir(tmp_path, "official_abc", man)  # never overwrite
    RD.set_state(run, "interrupted", step=5)
    assert RD.open_for_resume(tmp_path, "official_abc", "H") == run
    with pytest.raises(RD.RunExistsError):
        RD.open_for_resume(tmp_path, "official_abc", "OTHER")  # different identity = new experiment
    RD.set_state(run, "completed")
    (run / "COMPLETED").write_text("{}")
    with pytest.raises(RD.RunExistsError):
        RD.open_for_resume(tmp_path, "official_abc", "H")  # completed runs cannot be resumed/overwritten


def test_export_is_inference_ready_without_optimizer_state(tiny_dataset, small_cfg, tmp_path):
    cfg = copy.deepcopy(small_cfg)
    cfg.train.max_steps = 10
    tr = make_trainer(cfg, tiny_dataset, tmp_path / "r")
    tr.train()
    model, dcfg, norm, prior, support, meta = load_generator(tmp_path / "r" / "best.safetensors")
    assert not any(k.startswith("opt") for k in meta)
    assert meta["format"] == "raresimdiff-generator-v1" and "object_slip" in support or len(support) > 0
    x = torch.randn(2, 9, 12)
    cond = {"task": torch.zeros(2).long(), "event": torch.ones(2).long(), "robot": torch.zeros(2).long(),
            "severity": torch.rand(2), "env": torch.rand(2, 3), "asset_ids": torch.ones(2, 6).long(), "slot_mask": torch.ones(2, 9)}
    assert torch.isfinite(model(x, torch.tensor([5, 500]), cond)).all()
