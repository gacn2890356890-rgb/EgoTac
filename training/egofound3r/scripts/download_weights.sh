#!/usr/bin/env bash
# Fetch the frozen VGGT-Omega backbone if it is not already there. Rerunning does nothing once it is.
# The file is gated: request access at the URL below, then export HF_TOKEN from that account.
# With the file already on the cluster, set VGGT_CHECKPOINT in cluster.env and nothing is downloaded.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
URL=https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt
TARGET=$EGOTAC_ROOT/checkpoints/vggt_omega_1b_512.pt
if [ -f "$TARGET" ]; then
  exit 0
fi
mkdir -p "$(dirname "$TARGET")"
if [ -n "${VGGT_CHECKPOINT:-}" ]; then
  [ -f "$VGGT_CHECKPOINT" ] || { echo "error: VGGT_CHECKPOINT=$VGGT_CHECKPOINT not found" >&2; exit 1; }
  ln -s "$VGGT_CHECKPOINT" "$TARGET"
  exit 0
fi
echo "downloading VGGT-Omega (4.6 GB) from $URL"
if ! "$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt', local_dir='$EGOTAC_ROOT/checkpoints')"; then
  echo "error: could not download VGGT-Omega. Request access at $URL and export HF_TOKEN, or set VGGT_CHECKPOINT" >&2
  exit 1
fi
