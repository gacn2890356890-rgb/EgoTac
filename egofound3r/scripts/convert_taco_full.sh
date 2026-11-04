#!/usr/bin/env bash
# All of TACO without the allocentric videos (about 40 GB of the 975 GB release).
# Rerunning resumes. Usage: bash scripts/convert_taco_full.sh <mano_dir> [workers] [gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
MANO=${1:?MANO directory}; WORKERS=${2:-6}; GPU=${3:-0}
ROOT=${EGOTAC_ROOT:-/data/capstone/EgoTac}
export EGOFOUND3R_ROOT=${EGOFOUND3R_ROOT:-$ROOT/upstream} PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
PY=${EGOTAC_PYTHON:-$ROOT/venv/bin/python}
$PY -m egotac.data.convert_taco --root "$ROOT/data/taco" --output "$ROOT/data/clips_full" --fetch
ARGS="--root $ROOT/data/taco --output $ROOT/data/clips_full --mano-dir $MANO --device cuda"
for ((i = 0; i < WORKERS; i++)); do $PY -m egotac.data.convert_taco $ARGS --shard "$i/$WORKERS" & done
wait
$PY -m egotac.data.convert_taco $ARGS --manifest-only
