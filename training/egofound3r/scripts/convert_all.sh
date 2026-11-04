#!/usr/bin/env bash
# Convert OakInk-v2 and TACO at the paper's sequence counts, then write the manifest. Needs internet
# and GPU access for label computation. Rerunning resumes. Usage: bash scripts/convert_all.sh [workers]
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

"$PY" -m egotac.data.convert_taco --root "$EGOTAC_ROOT/data/taco" --output "$CLIPS" --manifest-only
