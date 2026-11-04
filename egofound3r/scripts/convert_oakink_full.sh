#!/usr/bin/env bash
# Stream all of OakInk-v2 through the converter, keeping no source frames afterwards.
# Rerunning skips finished sequences. Usage: bash scripts/convert_oakink_full.sh <mano_dir> [workers] [gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
MANO=${1:?MANO directory}; WORKERS=${2:-6}; GPU=${3:-0}
ROOT=${EGOTAC_ROOT:-/data/capstone/EgoTac}
export EGOFOUND3R_ROOT=${EGOFOUND3R_ROOT:-$ROOT/upstream} PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
PY=${EGOTAC_PYTHON:-$ROOT/venv/bin/python}
ARGS="--root $ROOT/data/oakink2 --output $ROOT/data/clips_full --mano-dir $MANO --device cuda --from-hf"
for ((i = 0; i < WORKERS; i++)); do $PY -m egotac.data.convert_oakink2 $ARGS --shard "$i/$WORKERS" & done
wait
$PY -m egotac.data.convert_oakink2 $ARGS --manifest-only
