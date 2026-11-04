#!/usr/bin/env bash
# Fetch the official code at the tested commit and download what is public.
# EgoFound3R.pt is gated: accept its terms on Hugging Face first, then `huggingface-cli login`.
set -euo pipefail
cd "$(dirname "$0")/.."
UPSTREAM_COMMIT=264f4df9fd42536aae4dba982d18272cc3334077
if [ ! -d third_party/EgoFound3R ]; then
  git clone https://github.com/MINT-SJTU/EgoFound3R.git third_party/EgoFound3R
fi
git -C third_party/EgoFound3R checkout -q "$UPSTREAM_COMMIT"
pip install -r requirements.txt
echo "Next: put EgoFound3R.pt and vggt_omega_1b_512.pt somewhere, set vggt_checkpoint_path in a copy of"
echo "third_party/EgoFound3R/configs/default.toml, and point configs/*.toml at both."
