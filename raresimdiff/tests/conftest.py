import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def cfg_dir():
    return str(ROOT / "configs")


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, cfg_dir):
    """Small real dataset built with the real pipeline (CPU)."""
    from raresimdiff.data.build_dataset import build_dataset
    out = tmp_path_factory.mktemp("ds") / "tiny"
    build_dataset(cfg_dir, str(out), n_parents=60, variants=4, workers=1, base_seed=777, quiet=True)
    return out


@pytest.fixture(scope="session")
def small_cfg(cfg_dir):
    from raresimdiff.utils.config import load_config
    cfg = load_config(cfg_dir)
    cfg.model.hidden_dim, cfg.model.num_layers, cfg.model.num_heads = 64, 2, 4
    t = cfg.train
    t.batch_size, t.grad_accum_steps, t.warmup_steps, t.log_every, t.eval_every, t.ckpt_every = 32, 2, 3, 5, 10, 10
    return cfg
