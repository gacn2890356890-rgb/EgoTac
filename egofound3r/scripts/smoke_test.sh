#!/usr/bin/env bash
# Very small end-to-end check of both versions: short training, then evaluation and a comparison.
# It only shows the code runs. Usage: bash scripts/smoke_test.sh [gpu_baseline] [gpu_world] [steps]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GPU_A=${1:-0}; GPU_B=${2:-1}; STEPS=${3:-30}
OUT=$EGOTAC_ROOT/outputs/smoke
CONFIG=$EGOTAC_ROOT/configs/default.toml; CLIPS=$EGOTAC_ROOT/data/clips_smoke
EVAL_ARGS="--upstream-config $CONFIG --clips $CLIPS --split test --image-size 256 --image-mode center_crop --batch-size 2 --max-batches 8"
TRAIN_ARGS=(--config configs/smoke.toml --set "train.steps=$STEPS" --set train.eval_every=0 --set "train.save_every=$STEPS" --set train.warmup_steps=5)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"

echo "[1/3] unit tests"
CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q tests

echo "[2/3] training both versions, $STEPS steps each"
EGOFOUND3R_ROOT=$(tree_for_variant baseline) CUDA_VISIBLE_DEVICES=$GPU_A "$PY" -m egotac.train "${TRAIN_ARGS[@]}" --set 'train.variant="baseline"' --set "train.output_dir=\"$OUT/baseline\"" > "$OUT/train_baseline.log" 2>&1 &
PID_A=$!
EGOFOUND3R_ROOT=$(tree_for_variant world) CUDA_VISIBLE_DEVICES=$GPU_B "$PY" -m egotac.train "${TRAIN_ARGS[@]}" --set 'train.variant="world"' --set "train.output_dir=\"$OUT/world\"" > "$OUT/train_world.log" 2>&1 &
PID_B=$!
wait $PID_A; wait $PID_B

echo "[3/3] evaluation"
EGOFOUND3R_ROOT=$(tree_for_variant baseline) CUDA_VISIBLE_DEVICES=$GPU_A "$PY" -m egotac.evaluate $EVAL_ARGS --checkpoint "$OUT/baseline/final.pt" --output "$OUT/eval_baseline.json" > "$OUT/eval_baseline.log" 2>&1
EGOFOUND3R_ROOT=$(tree_for_variant world) CUDA_VISIBLE_DEVICES=$GPU_B "$PY" -m egotac.evaluate $EVAL_ARGS --variant world --checkpoint "$OUT/world/final.pt" --output "$OUT/eval_world.json" > "$OUT/eval_world.log" 2>&1
"$PY" scripts/compare_runs.py "baseline=$OUT/eval_baseline.json" "world=$OUT/eval_world.json" --output "$OUT/compare.md"
echo "done: $OUT/compare.md"
