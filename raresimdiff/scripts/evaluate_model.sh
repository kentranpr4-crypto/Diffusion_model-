#!/usr/bin/env bash
# Evaluate a frozen generator against the procedural generator (no training).
# Usage: scripts/evaluate_model.sh CHECKPOINT.safetensors OUT_DIR [--device cuda] [--workers N] [--n-per-condition N]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CK="${1:?checkpoint .safetensors}"; OUT="${2:?output dir}"; shift 2
"${PYTHON}" -m raresimdiff.evaluation.evaluate --checkpoint "${CK}" --data-dir "${RSD_DATA_DIR}" --out "${OUT}" "$@"
