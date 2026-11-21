from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from .hashing import _default


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: str | Path, obj, indent: int = 2) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=indent, sort_keys=True, default=_default) + "\n").encode())


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_npz_deterministic(path, **arrays) -> None:
    """np.savez_compressed with fixed zip timestamps so identical content gives identical bytes (stable hashes)."""
    import io, zipfile
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())


def open_gzip_text_deterministic(path):
    """gzip text writer with mtime=0 (no timestamp in the header)."""
    import gzip, io
    return io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=open(path, "wb"), mtime=0), encoding="utf-8")
