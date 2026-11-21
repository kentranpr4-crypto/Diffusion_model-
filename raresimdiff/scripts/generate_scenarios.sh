#!/usr/bin/env bash
# Conditional generation from a frozen checkpoint. All arguments are passed to raresimdiff.generate.
# Example: scripts/generate_scenarios.sh --checkpoint runs/<RUN_ID>/best.safetensors --event-type object_slip \
#            --severity 0.8 --num-scenarios 1000 --validate-physics --output generated/slip_cases
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"${PYTHON}" -m raresimdiff.generate "$@"
