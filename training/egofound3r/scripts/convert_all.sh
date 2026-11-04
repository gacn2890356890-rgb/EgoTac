#!/usr/bin/env bash
# Convert every dataset at the paper's sequence counts, then write the manifest. OakInk-v2 and TACO
# download themselves; the others convert if their folder exists under $EGOTAC_ROOT/data (see README).
# Needs internet and GPU access for label computation. Rerunning resumes. Usage: bash scripts/convert_all.sh [workers]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
WORKERS=${1:-6}
CLIPS=$EGOTAC_ROOT/data/clips_full
[ -f "$MANO_DIR/MANO_LEFT.pkl" ] && [ -f "$MANO_DIR/MANO_RIGHT.pkl" ] || { echo "error: MANO files not found in $MANO_DIR" >&2; exit 1; }

COMMON=(--output "$CLIPS" --mano-dir "$MANO_DIR" --device cuda)
for ((i = 0; i < WORKERS; i++)); do
  "$PY" -m egotac.data.convert_oakink2 --root "$EGOTAC_ROOT/data/oakink2" "${COMMON[@]}" --from-hf --shard "$i/$WORKERS" &
done
wait

"$PY" -m egotac.data.convert_taco --root "$EGOTAC_ROOT/data/taco" --output "$CLIPS" --fetch
for ((i = 0; i < WORKERS; i++)); do
  "$PY" -m egotac.data.convert_taco --root "$EGOTAC_ROOT/data/taco" "${COMMON[@]}" --shard "$i/$WORKERS" &
done
wait

convert_if_present() {
  local dataset=$1 module=$2 root=$EGOTAC_ROOT/data/$1
  shift 2
  if [ ! -d "$root" ]; then
    echo "skipping $dataset: $root not found"
    return
  fi
  for ((i = 0; i < WORKERS; i++)); do
    "$PY" -m "egotac.data.$module" --root "$root" "${COMMON[@]}" --shard "$i/$WORKERS" "$@" &
  done
  wait
}
convert_if_present hoi4d convert_hoi4d
convert_if_present h2o convert_h2o --object-dir "$EGOTAC_ROOT/data/h2o/object"
convert_if_present hot3d_aria convert_hot3d
convert_if_present arctic convert_arctic

"$PY" -m egotac.data.convert_taco --root "$EGOTAC_ROOT/data/taco" --output "$CLIPS" --manifest-only
