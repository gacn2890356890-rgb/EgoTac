#!/usr/bin/env bash
# Full training on one node. Rerunning resumes from latest.pt.
# Usage: bash scripts/train_full.sh baseline|world [num_gpus] [--set section.key=value ...]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
VARIANT=${1:?baseline or world}; GPUS=${2:-8}; shift; shift || true
export EGOFOUND3R_ROOT=$(tree_for_variant "$VARIANT")
"$PY" scripts/preflight.py --config configs/full.toml "$@"
exec "$PY" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29500}" -m egotac.train --config configs/full.toml \
  --set "train.variant=\"$VARIANT\"" --set "train.output_dir=\"$EGOTAC_ROOT/outputs/full_$VARIANT\"" "$@"
