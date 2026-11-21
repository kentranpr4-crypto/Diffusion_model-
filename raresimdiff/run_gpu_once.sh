#!/usr/bin/env bash
# ONE command for the single official GPU run (RTX 4090 24 GB):   bash run_gpu_once.sh
#   --preflight-only   run every check, take NO optimizer steps
#   --resume           continue the SAME interrupted run (never a new experiment)
# Steps: (1) bundle integrity  (2) environment  (3) GPU preflight  (4) dataset+manifest+config validation
#        (5) immutable run dir  (6) training with checkpoints  (7) evaluation  (8) export  (9) machine-readable report
# No optimizer step happens before every preflight check passes. A completed run is never overwritten.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export RSD_DATA_DIR="${RSD_DATA_DIR:-data/raresimdiff_v1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"   # deterministic cuBLAS

echo "== [1/4] package integrity =="
if [ -f MANIFEST.sha256 ]; then
  # MANIFEST.sha256 holds paths prefixed with "raresimdiff/"; verify relative to the parent view
  bad=0
  while read -r h f; do
    rel="${f#raresimdiff/}"
    [ "${rel}" = "MANIFEST.sha256" ] && continue
    if [ ! -f "${rel}" ]; then echo "MISSING ${rel}"; bad=1; continue; fi
    got="$(sha256sum "${rel}" | cut -d' ' -f1)"
    [ "${got}" = "${h}" ] || { echo "CHECKSUM MISMATCH ${rel}"; bad=1; }
  done < MANIFEST.sha256
  [ "${bad}" = 0 ] || { echo "ABORT: package integrity check failed (nothing was trained)"; exit 1; }
  echo "package integrity OK"
else
  echo "ABORT: MANIFEST.sha256 missing"; exit 1
fi

echo "== [2/4] environment =="
# shellcheck disable=SC1091
source scripts/setup_env.sh || { echo "ABORT: could not prepare a Python environment"; exit 1; }
export PYTHON PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "== [3/4] GPU / dataset / config preflight (no optimizer steps) =="
mkdir -p preflight
"${PYTHON}" -m raresimdiff.training.gpu_preflight --config-dir configs --data-dir "${RSD_DATA_DIR}" --out preflight/gpu_preflight.json \
  || { echo "ABORT: GPU preflight failed (see preflight/gpu_preflight.json). No optimizer step was taken."; exit 2; }

echo "== [4/4] official run =="
exec "${PYTHON}" -m raresimdiff.training.run_official --config-dir configs --data-dir "${RSD_DATA_DIR}" --runs-root runs "$@"
