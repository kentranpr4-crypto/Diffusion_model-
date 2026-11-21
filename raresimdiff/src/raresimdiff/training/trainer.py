"""Single-run trainer: AdamW + warmup/cosine LR, AMP (bf16), grad accumulation, EMA, clipping, NaN guards,
atomic periodic checkpoints, best-checkpoint selection, deterministic resume, JSONL/CSV logging."""
from __future__ import annotations
import csv, json, math, signal, time
from pathlib import Path
from typing import Dict, Optional
import torch

from ..data.dataset import InMemoryDataset, load_normalizer, load_split
from ..diffusion.gaussian import GaussianDiffusion
from ..models.dit import RareScenarioDiT, count_parameters
from ..scenarios.prior import CategoricalPrior
from ..utils.config import AllConfig
from .checkpoint import atomic_torch_save, capture_rng, restore_rng
from .ema import EMA
from .export import export_generator, train_support
from .run_dir import set_state


class NonFiniteTrainingError(RuntimeError):
    pass


class Trainer:
    def __init__(self, cfg: AllConfig, data_dir: str | Path, run_dir: str | Path, device: str = "cpu",
                 identity: Optional[Dict] = None, tensorboard: bool = False):
        self.cfg, self.tc = cfg, cfg.train
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.identity = identity or {}
        self.data_dir = Path(data_dir)
        self.normalizer = load_normalizer(self.data_dir)
        self.train_split = load_split(self.data_dir, "train")
        self.train_ds = InMemoryDataset(self.train_split, self.normalizer, self.device)
        self.val_ds = InMemoryDataset(load_split(self.data_dir, "val"), self.normalizer, self.device,
                                      max_samples=self.tc.val_max_samples)
        torch.manual_seed(cfg.run.seed)
        self.model = RareScenarioDiT(cfg.model).to(self.device)
        self.ema = EMA(self.model, self.tc.ema_decay)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.tc.lr, betas=tuple(self.tc.adam_betas),
                                     weight_decay=self.tc.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)
        self.diffusion = GaussianDiffusion(cfg.diffusion.num_timesteps, cfg.diffusion.schedule, cfg.diffusion.prediction,
                                           cfg.diffusion.p_uncond, cfg.diffusion.clip_x0).to(self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.run.seed + 1)
        self.step = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.skipped = 0
        self.consecutive_nonfinite = 0
        self.val_history = []
        self.amp_dtype = None
        if self.device.type == "cuda" and self.tc.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        self.N = len(self.train_ds)
        self.mpe = max(1, self.N // self.tc.batch_size)  # micro-batches per epoch
        self._perm_epoch, self._perm = -1, None
        self._stop = False
        self.n_params = count_parameters(self.model)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "logs" / "train.csv"
        self.jsonl_path = self.run_dir / "logs" / "train.jsonl"
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.run_dir / "logs" / "tb"))
            except Exception:  # pragma: no cover
                self.tb = None

    # ------------------------------------------------------------------ schedule / data
    def _lr_lambda(self, step: int) -> float:
        tc = self.tc
        if step < tc.warmup_steps:
            return (step + 1) / max(1, tc.warmup_steps)
        prog = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
        return tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    def _micro_indices(self, global_micro: int) -> torch.Tensor:
        epoch, i = divmod(global_micro, self.mpe)
        if epoch != self._perm_epoch:
            g = torch.Generator().manual_seed(self.cfg.run.seed * 1_000_003 + epoch)
            self._perm = torch.randperm(self.N, generator=g)
            self._perm_epoch = epoch
        B = self.tc.batch_size
        return self._perm[i * B:(i + 1) * B].to(self.device)

    # ---------------------------------------------------------------- checkpointing
    def state(self) -> Dict:
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "step": self.step, "best_val": self.best_val, "best_step": self.best_step,
                "skipped": self.skipped, "val_history": self.val_history, "rng": capture_rng(self.gen),
                "identity": self.identity, "n_params": self.n_params}

    def save_checkpoint(self, tag: str, keep_last: Optional[int] = None) -> Path:
        ck = self.run_dir / "checkpoints"
        ck.mkdir(exist_ok=True)
        path = ck / f"{tag}.pt"
        atomic_torch_save(self.state(), path)
        if keep_last:
            steps = sorted(ck.glob("step_*.pt"))
            for old in steps[:-keep_last]:
                old.unlink()
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        if st["identity"].get("identity_hash") != self.identity.get("identity_hash"):
            raise RuntimeError("checkpoint identity (config/dataset) differs from this run: refusing to resume")
        self.model.load_state_dict(st["model"])
        self.ema.load_state_dict(st["ema"])
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.step, self.best_val, self.best_step = st["step"], st["best_val"], st["best_step"]
        self.skipped, self.val_history = st["skipped"], st["val_history"]
        restore_rng(st["rng"], self.gen)

    # ---------------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate_val(self) -> float:
        self.ema.model.eval()
        g = torch.Generator(device=self.device).manual_seed(self.cfg.run.seed + 777)
        B = self.tc.batch_size
        tot, cnt = 0.0, 0.0
        for i in range(0, len(self.val_ds), B):
            idx = torch.arange(i, min(i + B, len(self.val_ds)), device=self.device)
            x0, cond = self.val_ds.batch(idx)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                loss = self.diffusion.training_loss(self.ema.model, x0, cond, generator=g,
                                                    drop=torch.zeros(len(idx), dtype=torch.bool, device=self.device))
            w = self.diffusion.element_weights(cond["slot_mask"]).sum().item()
            tot += loss.item() * w
            cnt += w
        return tot / max(cnt, 1.0)

    # --------------------------------------------------------------------- logging
    def _log(self, rec: Dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(rec)
        if self.tb is not None:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k != "step":
                    self.tb.add_scalar(k, v, rec["step"])

    def _gpu_mem(self) -> Dict:
        if self.device.type != "cuda":
            return {}
        return {"gpu_mem_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
                "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

    def _export(self, name: str) -> Path:
        path = self.run_dir / name
        export_generator(path, self.ema.state_dict(), self.cfg.model, self.cfg.diffusion, self.normalizer,
                         CategoricalPrior.from_split(self.train_split), train_support(self.train_split),
                         {"run_id": self.identity.get("run_id", ""), "step": self.step,
                          "dataset_fingerprint": self.identity.get("dataset_fingerprint", ""),
                          "training_hash": self.identity.get("training_hash", ""), "n_params": self.n_params})
        return path

    # -------------------------------------------------------------------- train
    def request_stop(self, *_):
        self._stop = True

    def train(self, max_steps: Optional[int] = None) -> Dict:
        tc = self.tc
        total = tc.max_steps if max_steps is None else max_steps
        old = {s: signal.signal(s, self.request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
        t_log, loss_acc, n_acc, gn_acc = time.time(), 0.0, 0, 0.0
        status = "completed"
        try:
            self.model.train()
            while self.step < total:
                self.opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                ok = True
                for k in range(tc.grad_accum_steps):
                    idx = self._micro_indices(self.step * tc.grad_accum_steps + k)
                    x0, cond = self.train_ds.batch(idx)
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                        loss = self.diffusion.training_loss(self.model, x0, cond, generator=self.gen)
                    if not torch.isfinite(loss):
                        ok = False
                        break
                    (loss / tc.grad_accum_steps).backward()
                    step_loss += loss.item() / tc.grad_accum_steps
                gnorm = float("nan")
                if ok:
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                    gnorm = float(gn)
                    ok = bool(torch.isfinite(gn))
                if ok:
                    self.opt.step()
                    self.sched.step()
                    self.ema.update(self.model, self.step)
                    self.consecutive_nonfinite = 0
                    loss_acc += step_loss
                    gn_acc += gnorm
                    n_acc += 1
                else:
                    self.opt.zero_grad(set_to_none=True)
                    self.sched.step()  # keep the LR schedule aligned with the step counter
                    self.skipped += 1
                    self.consecutive_nonfinite += 1
                    if self.consecutive_nonfinite > tc.max_consecutive_nonfinite:
                        self.step += 1
                        self.save_checkpoint("last")
                        raise NonFiniteTrainingError(
                            f"{self.consecutive_nonfinite} consecutive non-finite steps at step {self.step}")
                self.step += 1
                if self.step % tc.log_every == 0 or self.step == total:
                    dt = time.time() - t_log
                    rec = {"step": self.step, "loss": loss_acc / max(n_acc, 1), "grad_norm": gn_acc / max(n_acc, 1),
                           "lr": self.sched.get_last_lr()[0], "ema_decay": self.ema.effective_decay(self.step),
                           "skipped_steps": self.skipped, "samples_per_sec": n_acc * tc.batch_size * tc.grad_accum_steps / max(dt, 1e-9),
                           "wall_time": time.time(), **self._gpu_mem()}
                    self._log(rec)
                    t_log, loss_acc, n_acc, gn_acc = time.time(), 0.0, 0, 0.0
                if self.step % tc.eval_every == 0 or self.step == total:
                    v = self.evaluate_val()
                    self.val_history.append({"step": self.step, "val_loss_ema": v})
                    self._log({"step": self.step, "val_loss_ema": v})
                    if v < self.best_val:
                        self.best_val, self.best_step = v, self.step
                        self.save_checkpoint("best")
                        self._export("best.safetensors")
                    if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                        raise NonFiniteTrainingError("non-finite parameters detected at validation")
                if self.step % tc.ckpt_every == 0:
                    self.save_checkpoint(f"step_{self.step:08d}", keep_last=tc.keep_last_checkpoints)
                    self.save_checkpoint("last")
                    set_state(self.run_dir, "running", step=self.step, best_val=self.best_val, best_step=self.best_step)
                if self._stop:
                    status = "interrupted"
                    break
        except NonFiniteTrainingError as e:
            status = "failed_nonfinite"
            set_state(self.run_dir, status, step=self.step, error=str(e))
            raise
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            if status == "interrupted":
                self.save_checkpoint("last")
                set_state(self.run_dir, "interrupted", step=self.step, best_val=self.best_val, best_step=self.best_step)
        if status == "completed":
            self.save_checkpoint("last")
            self._export("last.safetensors")
            if self.best_step < 0:
                self._export("best.safetensors")
        return {"status": status, "step": self.step, "best_val": self.best_val, "best_step": self.best_step,
                "skipped_steps": self.skipped, "n_params": self.n_params}
