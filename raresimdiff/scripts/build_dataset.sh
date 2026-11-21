#!/usr/bin/env bash
# Build the simulation dataset (CPU). Refuses to overwrite a non-empty output directory.
# Usage: scripts/build_dataset.sh [--out DIR] [--n-parents N] [--variants V] [--workers W] [--base-seed S]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"${PYTHON}" -m raresimdiff.data.build_dataset --config-dir configs "$@"
