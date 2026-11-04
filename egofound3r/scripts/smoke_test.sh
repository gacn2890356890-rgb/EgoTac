#!/usr/bin/env bash
# Original versus diffusion on the same data and steps, ending in a comparison and an I/O check.
# Usage: bash scripts/smoke_test.sh <gpu_baseline> <gpu_diffusion> [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_A=${1:-0}; GPU_B=${2:-1}; STEPS=${3:-600}
ROOT=${EGOTAC_ROOT:-/data/capstone/EgoTac}
PY=${EGOTAC_PYTHON:-$ROOT/venv/bin/python}
export EGOFOUND3R_ROOT=${EGOFOUND3R_ROOT:-$ROOT/upstream} PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID
OUT=$ROOT/outputs/smoke
CONFIG=$ROOT/configs/default.toml; CKPT=$ROOT/checkpoints/EgoFound3R.pt; CLIPS=$ROOT/data/clips_smoke
EVAL_ARGS="--upstream-config $CONFIG --clips $CLIPS --split test --image-size 256 --image-mode center_crop --batch-size 2"
mkdir -p "$OUT"

echo "[1/5] unit tests"
CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests

echo "[2/5] official weights, no training"
CUDA_VISIBLE_DEVICES=$GPU_A $PY -m egotac.evaluate $EVAL_ARGS --checkpoint "$CKPT" --output "$OUT/eval_official.json" > "$OUT/eval_official.log" 2>&1

echo "[3/5] training both variants, $STEPS steps each"
common="--config configs/smoke.toml --set train.steps=$STEPS --set train.eval_every=0 --set train.save_every=$STEPS"
CUDA_VISIBLE_DEVICES=$GPU_A $PY -m egotac.train $common --set train.variant=\"baseline\" --set train.output_dir=\"$OUT/baseline\" > "$OUT/train_baseline.log" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=$GPU_B $PY -m egotac.train $common --set train.variant=\"diffusion\" --set train.output_dir=\"$OUT/diffusion\" > "$OUT/train_diffusion.log" 2>&1 &
PID_B=$!
wait $PID_A; wait $PID_B

echo "[4/5] held-out evaluation"
CUDA_VISIBLE_DEVICES=$GPU_A $PY -m egotac.evaluate $EVAL_ARGS --checkpoint "$OUT/baseline/final.pt" --output "$OUT/eval_baseline.json" > "$OUT/eval_baseline.log" 2>&1
CUDA_VISIBLE_DEVICES=$GPU_B $PY -m egotac.evaluate $EVAL_ARGS --variant diffusion --checkpoint "$OUT/diffusion/final.pt" --output "$OUT/eval_diffusion.json" > "$OUT/eval_diffusion.log" 2>&1
$PY scripts/compare_runs.py "official=$OUT/eval_official.json" "baseline=$OUT/eval_baseline.json" "diffusion=$OUT/eval_diffusion.json" --output "$OUT/compare.md"

echo "[5/5] input/output contract on a real video"
if [ -n "${IO_VIDEO:-}" ]; then
  ( cd "$EGOFOUND3R_ROOT"
    COMMON="--config $CONFIG --checkpoint $CKPT --video-path $IO_VIDEO --max-frames 120 --no-overlay-frame-images --device cuda"
    CUDA_VISIBLE_DEVICES=$GPU_A $PY infer_marker_video.py $COMMON --output-dir "$OUT/io_official" > "$OUT/io_official.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU_A $PY -m egotac.infer_video --tactile-checkpoint "$OUT/diffusion/final.pt" $COMMON --output-dir "$OUT/io_diffusion" > "$OUT/io_diffusion.log" 2>&1 )
  $PY scripts/compare_io.py --official "$OUT/io_official" --diffusion "$OUT/io_diffusion" --same-weights --output "$OUT/io_contract.json"
else
  echo "skipped: set IO_VIDEO=/path/to/video.mp4 to run it"
fi
echo "done: $OUT/compare.md"
