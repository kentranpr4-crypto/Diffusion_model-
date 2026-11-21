"""GO / NO-GO gates G0..G6. Each gate records what was actually executed; nothing is marked PASS unless it ran.

    python -m raresimdiff.gates --stage local  --data-dir data/raresimdiff_v1      # G0..G5
    python -m raresimdiff.gates --stage bundle --data-dir data/raresimdiff_v1 --bundle dist/raresimdiff_gpu_bundle.tar.zst
"""
from __future__ import annotations
import argparse, re, subprocess, sys, time
from pathlib import Path
from typing import Dict, List

from .utils.hashing import sha256_file
from .utils.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
PRIOR_DOCS = ["docs/ai-native/tasks/2026-11-05-diffusion-survey.md", "docs/ai-native/tasks/2026-11-05-humanoid-world-model-design.md",
              "docs/ai-native/tasks/2026-11-05-humanoid-scenario-app.md"]
MACHINE_PATH_PATTERNS = [r"/home/[A-Za-z0-9_.-]+/", r"/Users/[A-Za-z0-9_.-]+/", r"/tmp/claude", r"C:\\Users\\", r"/root/\.ccr"]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".txt", ".lock", ".json", ".cfg"}


def _pytest(paths: List[str], extra: List[str] | None = None) -> Dict:
    t0 = time.time()
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"] + paths + (extra or [])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    full = p.stdout + p.stderr
    tail = full.strip().splitlines()[-1] if full.strip() else ""
    m = re.search(r"(\d+) passed", full)
    f = re.search(r"(\d+) failed", full)
    e = re.search(r"(\d+) error", full)
    return {"cmd": " ".join(["pytest"] + paths), "returncode": p.returncode, "passed": int(m.group(1)) if m else 0,
            "failed": int(f.group(1)) if f else 0, "errors": int(e.group(1)) if e else 0, "summary": tail,
            "seconds": round(time.time() - t0, 1)}


def gate_g0() -> Dict:
    d: Dict = {"name": "G0 repository audit", "checks": {}, "notes": []}
    d["checks"]["prior_docs_present"] = {p: (REPO / p).exists() for p in PRIOR_DOCS}
    existing = [str(p.relative_to(REPO)) for ext in ("*.xml", "*.mjcf", "*.urdf") for p in REPO.rglob(ext)
                if ".venv" not in p.parts and "raresimdiff" not in p.parts and ".git" not in p.parts]
    d["notes"].append("reusable simulator/robot assets found in repo (outside raresimdiff): " + (", ".join(existing) or "NONE"))
    d["humanoid_model_found"] = bool(existing)
    git = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True)
    changed = [l[3:] for l in git.stdout.splitlines() if l[:2].strip() in ("M", "D", "R", "MM", "AM")] if git.returncode == 0 else None
    outside = None if changed is None else [c for c in changed if not c.startswith("raresimdiff/")]
    d["checks"]["prior_tracked_files_unmodified"] = (outside == []) if outside is not None else "git unavailable"
    d["modified_outside_raresimdiff"] = outside
    missing = []
    for m in ("torch", "numpy", "yaml", "mujoco", "safetensors", "pytest"):
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    try:
        import zstandard  # noqa: F401  (availability check only)
    except Exception:
        missing.append("zstandard (needed only for packaging)")
    d["missing_dependencies"] = missing
    d["ok"] = all(v is True for v in d["checks"]["prior_docs_present"].values()) and d["checks"]["prior_tracked_files_unmodified"] is True and not missing
    return d


def gate_g1() -> Dict:
    r = _pytest(["tests/unit/test_schema_codec.py", "tests/regression/test_regression.py"])
    return {"name": "G1 scenario schema", "pytest": r, "ok": r["returncode"] == 0 and r["failed"] == 0}


def gate_g2() -> Dict:
    r = _pytest(["tests/physics"])
    return {"name": "G2 simulation", "pytest": r, "ok": r["returncode"] == 0 and r["failed"] == 0}


def gate_g3(data_dir: str) -> Dict:
    from .data.integrity import verify_dataset
    rep = verify_dataset(data_dir)
    r = _pytest(["tests/integration/test_pipeline.py", "-k", "dataset or integrity"])
    man = read_json(Path(data_dir) / "dataset_manifest.json")
    return {"name": "G3 data integrity", "verify_dataset": rep, "pytest": r, "counts": man["counts"],
            "reject_reasons": man["reject_reasons"], "throughput": man["throughput"],
            "ok": rep["ok"] and r["returncode"] == 0}


def gate_g4() -> Dict:
    r1 = _pytest(["tests/unit/test_diffusion.py"])
    r2 = _pytest(["tests/training"])
    return {"name": "G4 diffusion model", "pytest": [r1, r2], "ok": r1["returncode"] == 0 and r2["returncode"] == 0}


def gate_g5() -> Dict:
    r = _pytest(["tests/integration"])
    return {"name": "G5 end-to-end pipeline", "pytest": r, "ok": r["returncode"] == 0 and r["failed"] == 0,
            "note": "random-weight samples are structural smoke evidence only, never evidence of learned capability"}


def gate_g6(bundle: str, data_dir: str, config_dir: str = "configs") -> Dict:
    from .packaging import DATA_REQUIRED, list_bundle
    from .utils.config import load_config
    d: Dict = {"name": "G6 GPU package readiness", "checks": {}}
    bundle = Path(bundle)
    sha_file = bundle.with_suffix("").with_suffix(".sha256")
    sha_file = bundle.parent / "raresimdiff_gpu_bundle.sha256"
    c = d["checks"]
    c["bundle_exists"] = bundle.exists()
    if not bundle.exists():
        d["ok"] = False
        return d
    expected = sha_file.read_text().split()[0] if sha_file.exists() else None
    c["bundle_checksum_valid"] = expected == sha256_file(bundle)
    files = list_bundle(bundle)
    prefix = "raresimdiff/"
    need = ["run_gpu_once.sh", "pyproject.toml", "requirements.lock", "README.md", "MANIFEST.sha256", "docs/GPU_DEPLOY.md",
            "docs/ARCHITECTURE.md", "docs/DATASET_SPEC.md", "docs/EXPERIMENT_PROTOCOL.md", "docs/RESULTS_TEMPLATE.md",
            "configs/model.yaml", "configs/dataset.yaml", "configs/gpu_4090.yaml", "configs/scenario_families.yaml",
            "scripts/local_preflight.sh", "scripts/build_dataset.sh", "scripts/package_for_gpu.sh", "scripts/gpu_preflight.sh",
            "scripts/train_one_shot.sh", "scripts/evaluate_model.sh", "scripts/generate_scenarios.sh",
            "scripts/export_policy_dataset.sh", "scripts/setup_env.sh",
            "src/raresimdiff/training/run_official.py", "src/raresimdiff/training/gpu_preflight.py",
            "src/raresimdiff/generate.py"] + [f"data/{Path(data_dir).name}/{x}" for x in DATA_REQUIRED]
    missing = [n for n in need if prefix + n not in files]
    c["required_files_present"] = not missing
    d["missing_files"] = missing
    # inner manifest consistency
    inner = files.get(prefix + "MANIFEST.sha256")
    bad = []
    if inner:
        for line in inner["data"].decode().splitlines():
            h, name = line.split("  ", 1)
            if name not in files or files[name]["sha256"] != h:
                bad.append(name)
    c["inner_manifest_matches"] = bool(inner) and not bad
    d["manifest_mismatches"] = bad[:10]
    # absolute / machine-specific paths in text files
    offenders = []
    for name, f in files.items():
        if Path(name).suffix in TEXT_SUFFIXES and f["data"] is not None:
            txt = f["data"].decode("utf-8", "ignore")
            if name.endswith("tests/regression/test_regression.py") or name.endswith("gates.py"):
                continue
            for pat in MACHINE_PATH_PATTERNS:
                if re.search(pat, txt):
                    offenders.append((name, pat))
    c["no_machine_specific_paths"] = not offenders
    d["path_offenders"] = offenders[:10]
    lock = files.get(prefix + "requirements.lock")
    c["dependencies_versioned"] = bool(lock) and all("==" in l for l in lock["data"].decode().splitlines() if l.strip() and not l.startswith("#"))
    run = files.get(prefix + "run_gpu_once.sh")
    txt = run["data"].decode() if run else ""
    c["launcher_checks_cuda_and_vram"] = "gpu_preflight" in txt and "sha256sum" in txt
    c["launcher_executable"] = bool(run) and bool(run["mode"] & 0o111)
    cfg = load_config(config_dir)
    c["training_config_frozen"] = (prefix + "configs/gpu_4090.yaml") in files and \
        files[prefix + "configs/gpu_4090.yaml"]["sha256"] == sha256_file(Path(config_dir) / "gpu_4090.yaml")
    d["training_hash"] = cfg.training_hash()
    deploy = files.get(prefix + "docs/GPU_DEPLOY.md")
    c["memory_estimate_documented"] = bool(deploy) and "ESTIMATED" in deploy["data"].decode()
    c["no_runs_or_caches"] = not any("/runs/" in n or "__pycache__" in n or ".pytest_cache" in n for n in files)
    d["ok"] = all(c.values())
    d["n_files"] = len(files)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["local", "bundle"], default="local")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bundle", default="dist/raresimdiff_gpu_bundle.tar.zst")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    gates = []
    if a.stage == "local":
        for fn in (gate_g0, gate_g1, gate_g2, lambda: gate_g3(a.data_dir), gate_g4, gate_g5):
            g = fn()
            gates.append(g)
            print(f"[{'GO' if g['ok'] else 'NO-GO'}] {g['name']}", flush=True)
    else:
        g = gate_g6(a.bundle, a.data_dir)
        gates.append(g)
        for k, v in g["checks"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"[{'GO' if g['ok'] else 'NO-GO'}] {g['name']}")
    rep = {"stage": a.stage, "gates": gates, "all_go": all(g["ok"] for g in gates), "seconds": round(time.time() - t0, 1),
           "time": time.time()}
    out = a.out or f"dist/gates_{a.stage}.json"
    write_json(out, rep)
    sys.exit(0 if rep["all_go"] else 1)


if __name__ == "__main__":
    main()
