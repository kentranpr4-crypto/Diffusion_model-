"""Build the reproducible GPU transfer bundle (tar.zst) + checksum. Local machine only."""
from __future__ import annotations
import argparse, io, json, os, tarfile
from pathlib import Path
from typing import List

from .utils.hashing import sha256_file, sha256_bytes
from .utils.io import write_json

SRC_DIRS = ("src", "configs", "scripts", "docs", "tests")
TOP_FILES = ("run_gpu_once.sh", "pyproject.toml", "README.md", "requirements.lock", "requirements.txt")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv", "runs", "dist", "generated", ".mypy_cache", "egg-info"}
DATA_REQUIRED = ("tensors_train.npz", "tensors_val.npz", "tensors_test.npz", "tensors_ood_test.npz",
                 "preprocessing_stats.json", "dataset_manifest.json")


def _skip(p: Path) -> bool:
    return any(part in SKIP_PARTS or part.endswith(".egg-info") or part.endswith(".pyc") for part in p.parts)


def collect_files(root: Path, data_dir: Path, include_trajectories: bool, include_specs: bool) -> List[tuple]:
    """Returns [(archive_path, source_path)] sorted."""
    out = []
    for d in SRC_DIRS:
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and not _skip(p.relative_to(root)):
                out.append((f"raresimdiff/{p.relative_to(root).as_posix()}", p))
    for f in TOP_FILES:
        if (root / f).exists():
            out.append((f"raresimdiff/{f}", root / f))
    dd = "data/" + data_dir.name
    for f in DATA_REQUIRED:
        out.append((f"raresimdiff/{dd}/{f}", data_dir / f))
    for p in sorted((data_dir / "manifests").glob("*.json")):
        out.append((f"raresimdiff/{dd}/manifests/{p.name}", p))
    if include_specs:
        for p in sorted(data_dir.glob("scenarios_*.jsonl.gz")):
            out.append((f"raresimdiff/{dd}/{p.name}", p))
        out.append((f"raresimdiff/{dd}/rejects.jsonl.gz", data_dir / "rejects.jsonl.gz"))
    if include_trajectories:
        for p in sorted((data_dir / "trajectories").glob("*.npz")):
            out.append((f"raresimdiff/{dd}/trajectories/{p.name}", p))
    return sorted(out)


def write_run_approval(root: Path, data_dir: Path) -> Path:
    """docs/RUN_APPROVAL.md: the exact dataset/config/budget the user is asked to approve (generated, bundled)."""
    from .utils.config import load_config
    cfg = load_config(root / "configs")
    man = json.loads((data_dir / "dataset_manifest.json").read_text())
    tc = cfg.train
    n_train = man["counts"]["train"]
    eff = tc.batch_size * tc.grad_accum_steps
    txt = f"""# Run approval sheet (generated; approve BEFORE running `bash run_gpu_once.sh`)

| item | value |
|---|---|
| dataset | `{man['dataset_name']}` fingerprint `{man['fingerprint']}` |
| scenarios (train / val / test / ood_test) | {man['counts']['train']} / {man['counts']['val']} / {man['counts']['test']} / {man['counts']['ood_test']} |
| training-config hash | `{cfg.training_hash()}` |
| model | RareScenarioDiT hidden {cfg.model.hidden_dim}, {cfg.model.num_layers} layers, {cfg.model.num_heads} heads |
| optimiser | AdamW lr {tc.lr}, wd {tc.weight_decay}, warmup {tc.warmup_steps}, cosine to {tc.min_lr_ratio} |
| batch | {tc.batch_size} x {tc.grad_accum_steps} accumulation = {eff} effective |
| budget | {tc.max_steps} optimizer steps (~{tc.max_steps * eff / max(n_train, 1):.0f} epochs over train) |
| precision | {tc.amp_dtype} if supported, else float32 |
| checkpoints | every {tc.ckpt_every} steps (keep last {tc.keep_last_checkpoints}) + best-by-validation-EMA |
| evaluation after training | {cfg.evaluation['n_per_condition']} scenarios x {len(cfg.evaluation['events'])} events x {len(cfg.evaluation['severities'])} severities, learned vs procedural |
| number of training runs | ONE (a second independent run needs new approval) |

Wall-clock time and GPU memory on the 4090 are NOT known yet; the preflight probe reports the measured memory (see GPU_DEPLOY.md).
"""
    p = root / "docs" / "RUN_APPROVAL.md"
    p.write_text(txt)
    return p


def build_bundle(root: str | Path, data_dir: str | Path, out_dir: str | Path, include_trajectories: bool = False,
                 include_specs: bool = False) -> dict:
    import zstandard as zstd
    root, data_dir, out_dir = Path(root).resolve(), Path(data_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_approval(root, data_dir)
    files = collect_files(root, data_dir, include_trajectories, include_specs)
    manifest_lines = [f"{sha256_file(src)}  {arc}" for arc, src in files if not arc.endswith("MANIFEST.sha256")]
    manifest = ("\n".join(manifest_lines) + "\n").encode()
    bundle = out_dir / "raresimdiff_gpu_bundle.tar.zst"
    tmp = bundle.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        with zstd.ZstdCompressor(level=12, threads=-1).stream_writer(fh) as zw:
            with tarfile.open(fileobj=zw, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                def add_bytes(name, data, mode=0o644):
                    ti = tarfile.TarInfo(name)
                    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, mode, 0, 0, "", ""
                    tf.addfile(ti, io.BytesIO(data))
                for arc, src in files:
                    mode = 0o755 if src.suffix == ".sh" else 0o644
                    add_bytes(arc, src.read_bytes(), mode)
                add_bytes("raresimdiff/MANIFEST.sha256", manifest)
    os.replace(tmp, bundle)
    sha = sha256_file(bundle)
    (out_dir / "raresimdiff_gpu_bundle.sha256").write_text(f"{sha}  {bundle.name}\n")
    info = {"bundle": str(bundle), "sha256": sha, "size_bytes": bundle.stat().st_size, "n_files": len(files) + 1,
            "include_trajectories": include_trajectories, "include_specs": include_specs}
    write_json(out_dir / "bundle_info.json", info)
    return info


def list_bundle(bundle: str | Path) -> dict:
    """Read the bundle back: {archive_path: (sha256, mode, bytes)} (used by gate G6)."""
    import zstandard as zstd
    res = {}
    with open(bundle, "rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as zr:
            with tarfile.open(fileobj=zr, mode="r|") as tf:
                for ti in tf:
                    if ti.isfile():
                        data = tf.extractfile(ti).read()
                        res[ti.name] = {"sha256": sha256_bytes(data), "mode": ti.mode, "data": data if ti.size < 2_000_000 else None,
                                        "size": ti.size}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--include-trajectories", action="store_true")
    ap.add_argument("--include-specs", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build_bundle(a.root, a.data_dir, a.out, a.include_trajectories, a.include_specs), indent=2))


if __name__ == "__main__":
    main()
