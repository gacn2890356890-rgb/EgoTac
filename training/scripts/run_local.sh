#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
TASK_TRAINING_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$TASK_TRAINING_ROOT/scripts/run_bounded.py" "$@"
