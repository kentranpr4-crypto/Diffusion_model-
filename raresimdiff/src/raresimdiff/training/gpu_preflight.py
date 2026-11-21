"""GPU preflight: validates hardware, software, data and config BEFORE any optimizer step.
The memory probe runs forward+backward at the configured batch size WITHOUT calling optimizer.step()."""
from __future__ import annotations
import argparse, json, shutil, sys, time
from typing import Dict
import torch

from ..data.integrity import verify_dataset
from ..utils.config import load_config
from ..utils.io import write_json
from .run_dir import gpu_info, dependency_versions


def run_preflight(config_dir: str, data_dir: str, allow_cpu: bool = False, probe: bool = True, out: str | None = None) -> Dict:
    cfg = load_config(config_dir)
    checks: Dict[str, Dict] = {}

    def rec(name, ok, detail="", warn=False):
        checks[name] = {"ok": bool(ok), "detail": detail, "warning_only": warn}

    rec("python_version", sys.version_info >= (3, 10), sys.version.split()[0])
    for m in ("torch", "numpy", "yaml", "safetensors", "mujoco"):
        try:
            mod = __import__(m)
            rec(f"import_{m}", True, getattr(mod, "__version__", "?"))
        except Exception as e:
            rec(f"import_{m}", False, str(e))
    gi = gpu_info()
    cuda = gi["cuda_available"]
    rec("cuda_available", cuda or allow_cpu, "CUDA device present" if cuda else
        ("NO CUDA (CPU allowed by --allow-cpu; this is not a GPU run)" if allow_cpu else "no CUDA device: official run requires a GPU"))
    if cuda:
        need = float(cfg.gpu.get("min_vram_gb", 16))
        rec("vram", gi["total_memory_gb"] >= need, f"{gi['total_memory_gb']} GB (need >= {need})")
        exp = str(cfg.gpu.get("expected_gpu_name_contains", ""))
        rec("gpu_model", exp.lower() in gi["name"].lower(), f"{gi['name']} (expected '{exp}')", warn=True)
        rec("bf16", gi["bf16_supported"], "bf16 supported" if gi["bf16_supported"] else "bf16 NOT supported: float32 will be used",
            warn=True)
    free_gb = shutil.disk_usage(".").free / 2**30
    rec("disk_space", free_gb >= float(cfg.gpu.get("min_free_disk_gb", 5)), f"{free_gb:.1f} GB free")
    try:
        ver = verify_dataset(data_dir)
        rec("dataset_integrity", ver["ok"], "; ".join(ver["failures"]) or f"fingerprint {ver['fingerprint'][:12]}")
    except Exception as e:
        rec("dataset_integrity", False, repr(e))
    tc = cfg.train
    rec("config_valid", tc.batch_size > 0 and tc.grad_accum_steps > 0 and tc.max_steps > tc.warmup_steps and
        cfg.model.hidden_dim % cfg.model.num_heads == 0, "ok")
    mem = {}
    if probe and checks["import_torch"]["ok"] and checks.get("dataset_integrity", {}).get("ok"):
        try:
            from ..data.dataset import InMemoryDataset, load_normalizer, load_split
            from ..diffusion.gaussian import GaussianDiffusion
            from ..models.dit import RareScenarioDiT, count_parameters
            dev = torch.device("cuda" if cuda else "cpu")
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ds = InMemoryDataset(load_split(data_dir, "train"), load_normalizer(data_dir), dev, max_samples=tc.batch_size)
            model = RareScenarioDiT(cfg.model).to(dev)
            diff = GaussianDiffusion(cfg.diffusion.num_timesteps, cfg.diffusion.schedule, cfg.diffusion.prediction,
                                     cfg.diffusion.p_uncond, cfg.diffusion.clip_x0).to(dev)
            amp = torch.bfloat16 if (dev.type == "cuda" and gi.get("bf16_supported") and tc.amp_dtype == "bfloat16") else None
            idx = torch.arange(min(tc.batch_size, len(ds)), device=dev)
            x0, cond = ds.batch(idx)
            t0 = time.time()
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                    loss = diff.training_loss(model, x0, cond)
                loss.backward()  # NOTE: no optimizer.step() in preflight
            if dev.type == "cuda":
                torch.cuda.synchronize()
            gfin = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            mem = {"n_params": count_parameters(model), "probe_batch": int(len(idx)), "probe_seconds_per_fwdbwd": (time.time() - t0) / 3,
                   "probe_device": str(dev), "probe_amp": str(amp), "finite_gradients": gfin}
            if dev.type == "cuda":
                mem["measured_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            rec("memory_probe", gfin and bool(torch.isfinite(loss)), json.dumps(mem))
            vram_ok = (not cuda) or mem.get("measured_peak_alloc_gb", 0) < 0.9 * gi["total_memory_gb"]
            rec("memory_fits", vram_ok, f"peak {mem.get('measured_peak_alloc_gb', 'n/a')} GB of {gi.get('total_memory_gb', 'n/a')} GB")
        except Exception as e:
            rec("memory_probe", False, repr(e))
    ok = all(c["ok"] or c["warning_only"] for c in checks.values())
    rep = {"ok": ok, "gpu": gi, "deps": dependency_versions(), "checks": checks, "probe": mem, "optimizer_steps_taken": 0,
           "time": time.time()}
    if out:
        write_json(out, rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--allow-cpu", action="store_true", help="local testing only; the official run needs CUDA")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = run_preflight(a.config_dir, a.data_dir, a.allow_cpu, not a.no_probe, a.out)
    for k, c in rep["checks"].items():
        print(f"[{'PASS' if c['ok'] else ('WARN' if c['warning_only'] else 'FAIL')}] {k}: {c['detail']}")
    print("GPU PREFLIGHT:", "PASS" if rep["ok"] else "FAIL")
    sys.exit(0 if rep["ok"] else 2)


if __name__ == "__main__":
    main()
