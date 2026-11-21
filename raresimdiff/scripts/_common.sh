# shellcheck shell=bash
# Common helpers. Resolves the project root; never uses absolute machine-specific paths.
RSD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${RSD_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RSD_DATA_DIR="${RSD_DATA_DIR:-data/raresimdiff_v1}"
PYTHON="${PYTHON:-python3}"
cd "${RSD_ROOT}"
