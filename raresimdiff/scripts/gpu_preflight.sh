#!/usr/bin/env bash
# GPU preflight: checks hardware/software/data/config and probes memory with forward+backward ONLY (NO optimizer step).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mkdir -p preflight
"${PYTHON}" -m raresimdiff.training.gpu_preflight --config-dir configs --data-dir "${RSD_DATA_DIR}" --out preflight/gpu_preflight.json "$@"
