#!/usr/bin/env bash
# THE official training job (inner launcher). Normally invoked through ../run_gpu_once.sh.
#   scripts/train_one_shot.sh                 # start the official run
#   scripts/train_one_shot.sh --preflight-only
#   scripts/train_one_shot.sh --resume        # continue the SAME interrupted run (never a new experiment)
# A completed run is never overwritten (exit 3). Non-finite training stops with exit 4 and is NOT retried silently.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"${PYTHON}" -m raresimdiff.training.run_official --config-dir configs --data-dir "${RSD_DATA_DIR}" --runs-root runs "$@"
