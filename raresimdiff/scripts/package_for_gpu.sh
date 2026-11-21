#!/usr/bin/env bash
# Build dist/raresimdiff_gpu_bundle.tar.zst (+ .sha256) and run gate G6 on it. Local machine only.
# Usage: scripts/package_for_gpu.sh [--include-trajectories] [--include-specs]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"${PYTHON}" -m raresimdiff.packaging --root . --data-dir "${RSD_DATA_DIR}" --out dist "$@"
"${PYTHON}" -m raresimdiff.gates --stage bundle --data-dir "${RSD_DATA_DIR}" --bundle dist/raresimdiff_gpu_bundle.tar.zst --out dist/gates_bundle.json
cat dist/raresimdiff_gpu_bundle.sha256
