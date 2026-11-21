#!/usr/bin/env bash
# Probe the machine and prepare a Python environment. Sourced by run_gpu_once.sh; sets $PYTHON.
# Never touches the NVIDIA driver. Prefers an existing working PyTorch+CUDA; otherwise creates ./.venv.
set -uo pipefail
RSD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RSD_ROOT}"
BASE_PY="${PYTHON:-python3}"

echo "[env] python: $(command -v "${BASE_PY}") ($(${BASE_PY} --version 2>&1))"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[env] nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  echo "[env] driver supports CUDA <= ${DRV_CUDA:-unknown}"
else
  echo "[env] WARNING: nvidia-smi not found"
  DRV_CUDA=""
fi

if [ "${RSD_SKIP_ENV_SETUP:-0}" = "1" ]; then   # testing / air-gapped use: trust the caller's interpreter
  export PYTHON="${BASE_PY}"
  echo "[env] RSD_SKIP_ENV_SETUP=1: using PYTHON=${PYTHON} without installing anything"
  "${PYTHON}" -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())"
  return 0 2>/dev/null || exit 0
fi

probe() {  # $1 = python; success iff torch imports, CUDA available, and the other deps import
  "$1" - <<'PY' >/dev/null 2>&1
import torch, numpy, yaml, safetensors, mujoco
assert torch.cuda.is_available()
PY
}

if probe "${BASE_PY}"; then
  echo "[env] existing environment already provides torch+CUDA and all dependencies: using it"
  export PYTHON="${BASE_PY}"
else
  echo "[env] creating ./.venv (existing environment is not sufficient)"
  "${BASE_PY}" -m venv .venv || { echo "[env] ERROR: cannot create venv (python3-venv missing?)"; return 1 2>/dev/null || exit 1; }
  VPY="${RSD_ROOT}/.venv/bin/python"
  "${VPY}" -m pip install --quiet --upgrade pip
  # 1) pinned lock (versions recorded on the preparation machine)
  "${VPY}" -m pip install --quiet -r requirements.lock || echo "[env] WARNING: pinned install failed; trying unpinned"
  if ! probe "${VPY}"; then
    "${VPY}" -m pip install --quiet -r requirements.txt || true
  fi
  # 2) if CUDA still unavailable, choose a PyTorch wheel index from the driver's maximum CUDA version (best effort)
  if ! probe "${VPY}" && [ -n "${DRV_CUDA}" ]; then
    MAJ="${DRV_CUDA%%.*}"; MIN="${DRV_CUDA##*.}"
    for tag in cu128 cu126 cu124 cu121 cu118; do
      t="${tag#cu}"; tmaj="${t:0:2}"; tmin="${t:2}"
      if [ "${tmaj}" -lt "${MAJ}" ] || { [ "${tmaj}" -eq "${MAJ}" ] && [ "${tmin}" -le "${MIN}" ]; }; then
        echo "[env] trying PyTorch wheels for ${tag}"
        "${VPY}" -m pip install --quiet --upgrade --force-reinstall torch --index-url "https://download.pytorch.org/whl/${tag}" && probe "${VPY}" && break
      fi
    done
  fi
  export PYTHON="${VPY}"
fi
echo "[env] using PYTHON=${PYTHON}"
"${PYTHON}" -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())"
