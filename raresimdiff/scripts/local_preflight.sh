#!/usr/bin/env bash
# Local (CPU) gates G0..G5. Runs tests and dataset verification. Spends NO GPU compute.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mkdir -p dist
"${PYTHON}" -m raresimdiff.gates --stage local --data-dir "${RSD_DATA_DIR}" --out dist/gates_local.json
