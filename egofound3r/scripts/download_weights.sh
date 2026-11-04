#!/usr/bin/env bash
# Download the model weights. EgoFound3R.pt is gated: accept its terms on Hugging Face, then export
# HF_TOKEN or run `.venv/bin/hf auth login`. MANO must be downloaded by hand from mano.is.tue.mpg.de.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
"$PY" - <<PY
from huggingface_hub import hf_hub_download
for repo, name in (("Frank-F2021/EgoFound3R", "EgoFound3R.pt"), ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")):
    print(hf_hub_download(repo_id=repo, filename=name, local_dir="$EGOTAC_ROOT/checkpoints"))
PY
