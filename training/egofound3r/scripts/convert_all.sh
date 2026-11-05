#!/usr/bin/env bash
# Convert every dataset at the paper's sequence counts, then write the manifest. Reads the data already on
# the cluster from the DATA_* paths in cluster.env, downloads nothing, and never writes to those folders.
# Needs GPU access for label computation. Rerunning resumes. Usage: bash scripts/convert_all.sh [workers]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
WORKERS=${1:-6}
CLIPS=$EGOTAC_ROOT/data/clips_full
[ -f "$MANO_DIR/MANO_LEFT.pkl" ] && [ -f "$MANO_DIR/MANO_RIGHT.pkl" ] || { echo "error: MANO files not found in $MANO_DIR" >&2; exit 1; }

COMMON=(--output "$CLIPS" --mano-dir "$MANO_DIR" --device cuda)
convert_if_present() {
  local dataset=$1 module=$2 root=$3
  shift 3
  if [ ! -d "$root" ]; then
    echo "skipping $dataset: $root not found"
    return 0
  fi
  local pids=()
  for ((i = 0; i < WORKERS; i++)); do
    "$PY" -m "egotac.data.$module" --root "$root" "${COMMON[@]}" --shard "$i/$WORKERS" "$@" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}
convert_if_present oakink2 convert_oakink2 "$DATA_OAKINK2"
convert_if_present taco convert_taco "$DATA_TACO"
convert_if_present hoi4d convert_hoi4d "$DATA_HOI4D"
convert_if_present h2o convert_h2o "$DATA_H2O" --object-dir "$DATA_H2O_OBJECTS"
convert_if_present hot3d_aria convert_hot3d "$DATA_HOT3D"
convert_if_present arctic convert_arctic "$DATA_ARCTIC"

"$PY" -m egotac.data.convert_taco --root "$DATA_TACO" --output "$CLIPS" --manifest-only
