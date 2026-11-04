#!/usr/bin/env bash
# One-click install for Python 3.10: environment, pinned packages, generated config.
# Rerunning is safe. Works even when the system Python has no pip, ensurepip, or virtualenv.
# Usage: bash scripts/setup.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BASE=${EGOTAC_BASE_PYTHON:-python3}

version=$("$BASE" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$version" != "3.10" ]; then
  echo "error: $BASE is Python $version; this environment is pinned for 3.10. Set EGOTAC_BASE_PYTHON." >&2
  exit 1
fi

if [ ! -x "$EGOTAC_VENV/bin/python" ]; then
  if ! "$BASE" -m venv "$EGOTAC_VENV" 2>/dev/null; then
    # Debian and Ubuntu ship Python without ensurepip unless python3-venv is installed.
    rm -rf "$EGOTAC_VENV"
    "$BASE" -m venv --without-pip "$EGOTAC_VENV"
  fi
fi
PY=$EGOTAC_VENV/bin/python
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing from the environment, bootstrapping it"
  bootstrap=$(mktemp -d)
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$bootstrap/get-pip.py"
  "$PY" "$bootstrap/get-pip.py" --quiet "pip==25.0.1"
  rm -rf "$bootstrap"
fi

# Wheels carry their own CUDA, so only the driver has to be new enough.
if [ -z "${TORCH_CUDA:-}" ]; then
  TORCH_CUDA=cu124
  if command -v nvidia-smi >/dev/null 2>&1; then
    driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    if [ -n "$driver_cuda" ] && "$PY" -c "import sys; sys.exit(0 if tuple(map(int, '$driver_cuda'.split('.'))) < (12, 4) else 1)"; then
      TORCH_CUDA=cu118
    fi
  fi
fi
echo "installing packages with PyTorch build $TORCH_CUDA"
# The official wheels bundle CUDA; the PyPI ones would pull several more gigabytes of NVIDIA packages.
torch_version=$(sed -n 's/^torch==//p' requirements.txt)
vision_version=$(sed -n 's/^torchvision==//p' requirements.txt)
"$PY" -m pip install --quiet --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
  --extra-index-url https://pypi.org/simple "torch==$torch_version+$TORCH_CUDA" "torchvision==$vision_version+$TORCH_CUDA"
"$PY" -m pip install --quiet -r requirements.txt
"$PY" -m pip check

mkdir -p "$EGOTAC_ROOT"/{checkpoints,configs,data,outputs}
sed "s#^vggt_checkpoint_path = .*#vggt_checkpoint_path = \"$EGOTAC_ROOT/checkpoints/vggt_omega_1b_512.pt\"#" \
  "$REPO_DIR/EgoFound3R_original/configs/default.toml" > "$EGOTAC_ROOT/configs/default.toml"

EGOFOUND3R_ROOT="$REPO_DIR/EgoFound3R_world" "$PY" scripts/check_env.py
bash scripts/verify_original.sh
echo
echo "Environment ready: $EGOTAC_VENV"
echo "Next: bash scripts/download_weights.sh, then see README.md"
