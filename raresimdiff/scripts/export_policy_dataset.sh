#!/usr/bin/env bash
# Roll out a policy on scenarios and export (observation, action, reward, next_observation, terminated, truncated,
# scenario_id, event_type, severity, is_synthetic) transitions.
# Example: scripts/export_policy_dataset.sh --scenarios generated/slip_cases/scenarios_accepted.jsonl --output policy_data/slip.npz
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"${PYTHON}" -m raresimdiff.policy_integration.exporters "$@"
