#!/usr/bin/env bash
# Full training. Usage: bash scripts/train_full.sh baseline|diffusion [num_gpus]
# Rerunning the same command resumes from the output directory's latest.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${1:?baseline or diffusion}; GPUS=${2:-8}
ROOT=${EGOTAC_ROOT:-/data/capstone/EgoTac}
export EGOFOUND3R_ROOT=${EGOFOUND3R_ROOT:-$ROOT/upstream} PYTHONPATH=$PWD
PY=${EGOTAC_PYTHON:-$ROOT/venv/bin/python}
exec $PY -m torch.distributed.run --nproc_per_node="$GPUS" -m egotac.train --config configs/full.toml \
  --set train.variant=\"$VARIANT\" --set train.output_dir=\"$ROOT/outputs/full_$VARIANT\"
